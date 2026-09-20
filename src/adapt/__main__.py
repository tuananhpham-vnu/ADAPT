"""python -m src.adapt --help"""
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import json
from pathlib import Path

from src.adapt.benchmark import banking_prompt,banking_suite,HELDOUT_FAMILIES
from src.adapt.engine import Engine,FixtureProvider
from src.adapt.experiment import estimate_calls,run,write_json
from src.adapt.retrieval import Retriever,memory_from_train
from src.adapt.spec import Case,Prompt,Requirement,digest,dump_spec,validate_suite


def main():
    p=argparse.ArgumentParser(description="ADAPT-RT core experiment; default fixture is synthetic, --backend live calls a provider.")
    p.add_argument("--backend",choices=["fixture","live"],default="fixture")
    p.add_argument("--provider",default="deepseek")
    p.add_argument("--model")
    p.add_argument("--retrieval",choices=["fixed","lexical","dpr"],default="fixed")
    p.add_argument("--embedder",default="dpr-ctx_encoder-single-nq-base")
    p.add_argument("--device",default="cpu")
    p.add_argument("--top-k",type=int,default=2)
    p.add_argument("--groups-per-split",type=int,default=5)
    p.add_argument("--repeats",type=int,default=5)
    p.add_argument("--mutants",type=int,default=8)
    p.add_argument("--repair-rounds",type=int,default=3)
    p.add_argument("--bootstrap",type=int,default=10000)
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--temperature",type=float,default=0)
    p.add_argument("--max-calls",type=int,default=2000)
    p.add_argument("--no-baselines",action="store_true")
    p.add_argument("--no-trace",action="store_true")
    p.add_argument("--plan",action="store_true",help="Print call estimate only; no models, retriever downloads or output directory")
    p.add_argument("--resume",action="store_true")
    p.add_argument("--spec",type=Path,help="JSON exported by a prior manifest: prompt + labeled cases")
    p.add_argument("--reviews",type=Path,help="JSON mapping mutant IDs to equivalent/non_equivalent/unresolved")
    p.add_argument("--output",type=Path,default=Path("results/adapt")/datetime.now().strftime("%Y%m%d_%H%M%S"))
    args=p.parse_args()
    for name in ("repeats","groups_per_split","top_k","bootstrap","max_calls"):
        if getattr(args,name)<1:p.error(f"--{name.replace('_','-')} must be positive")
    if args.mutants<0 or args.repair_rounds<0:p.error("mutants/repair-rounds cannot be negative")
    if args.spec:
        spec=json.loads(args.spec.read_text(encoding="utf-8"))
        pp=spec["prompt"]
        prompt=Prompt(pp["intro"],tuple(Requirement(**r) for r in pp["requirements"]),pp.get("overrides",{}))
        cases=[Case(**c) for c in spec["cases"]]
    else:
        prompt,cases=banking_prompt(),banking_suite(args.groups_per_split)
    validate_suite(cases,HELDOUT_FAMILIES)
    if any(not any(c.split==split for c in cases) for split in ("train","validation","heldout")):
        p.error("Need nonempty train, validation and heldout splits")
    config={k:v for k,v in vars(args).items() if k not in {"output","resume","plan","max_calls","reviews","spec"}}
    config["spec_sha256"]=digest(dump_spec(prompt,cases))
    reviews=json.loads(args.reviews.read_text(encoding="utf-8")) if args.reviews else None
    config["reviews_sha256"]=digest(reviews)
    estimate=estimate_calls(cases,args.mutants,args.repeats,args.repair_rounds,not args.no_baselines)
    if args.plan:
        print(json.dumps({"config":config,"cases":len(cases),"maximum_model_calls":estimate},indent=2));return
    path=args.output/"manifest.json"
    if args.resume:
        if not path.exists() or json.loads(path.read_text(encoding="utf-8"))["config"] != config:
            p.error("Resume configuration/spec differs from saved manifest")
    else:
        args.output.mkdir(parents=True,exist_ok=False)
        write_json(path,{"created_utc":datetime.now(timezone.utc).isoformat(),"config":config,"maximum_model_calls":estimate,
                         "evidence":"synthetic" if args.backend=="fixture" else "live",**dump_spec(prompt,cases)})
    if args.backend=="fixture":
        provider=FixtureProvider()
    else:
        from src.providers import make_provider
        provider=make_provider(args.provider)
    retriever=Retriever(memory_from_train(cases),args.retrieval,args.embedder,args.device) if args.retrieval!="fixed" else None
    engine=Engine(provider,args.output,model=args.model,temperature=args.temperature,max_calls=args.max_calls,
                  tracing=args.backend=="live" and not args.no_trace,retriever=retriever,top_k=args.top_k,backend=args.backend)
    try:
        result=run(engine,prompt,cases,mutants_limit=args.mutants,repeats=args.repeats,repair_rounds=args.repair_rounds,
                   bootstrap=args.bootstrap,seed=args.seed,baselines=not args.no_baselines,reviews=reviews)
        print(json.dumps({"output":str(args.output),"model_calls":result["n_model_calls"],"backend":args.backend},indent=2))
        if any(row["n_errors"] for row in result["metrics"]):
            raise SystemExit("Experiment completed with errors; inspect records.jsonl")
    finally:
        engine.close()


if __name__=="__main__":
    main()
