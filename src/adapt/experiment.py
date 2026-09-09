"""Complete core experiment, with held-out evaluation after the repair is frozen."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import html
import json
from pathlib import Path

from src.adapt.metrics import coverage, ochiai, summarize
from src.adapt.mutation import generate, mutation_report
from src.adapt.repair import improve
from src.adapt.spec import digest, dump_spec, validate_suite


def estimate_calls(cases, mutants, repeats, repair_rounds, baselines=True):
    dev = sum(c.split != "heldout" for c in cases)
    train = sum(c.split == "train" for c in cases)
    held = len(cases)-dev
    # original + spotlight + majority (gate shares original), every repair candidate
    # on development; frozen final repair on held-out (possibly reuses original).
    return repeats*((3 if baselines else 1)*(dev+held)+mutants*train+repair_rounds*dev+held)


def write_json(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")


def run(engine, prompt, cases, *, mutants_limit=8, repeats=5, repair_rounds=3,
        bootstrap=10000, seed=0, baselines=True, reviews=None):
    validate_suite(cases)
    if repeats < 1 or mutants_limit < 0:
        raise ValueError("Invalid repeats/mutant limit")
    mutants = generate(prompt)[:mutants_limit]
    collected = {}

    def evaluate_prompt(p,variant,subset,**options):
        rows = []
        for case in subset:
            for repeat in range(repeats):
                row = engine.run(case,p,variant,repeat,**options)
                collected[row["key"]] = row
                rows.append(row)
        print(f"{variant}: {len(subset)} cases x {repeats}; model calls used={engine.calls}",flush=True)
        return rows

    train = [c for c in cases if c.split == "train"]
    validation = [c for c in cases if c.split == "validation"]
    heldout = [c for c in cases if c.split == "heldout"]
    development = train+validation
    originals = evaluate_prompt(prompt,"original",development)
    for mutant in mutants:
        evaluate_prompt(mutant.prompt,mutant.id,train)
    if baselines:
        evaluate_prompt(prompt,"spotlight",development,spotlight=True)
        evaluate_prompt(prompt,"majority",development,majority=True)
    final_prompt,history = improve(prompt,train,validation,evaluate_prompt,rounds=repair_rounds)
    # Freeze and persist selection before held-out model calls.
    write_json(engine.output/"repair.json",{"history":history,"final_prompt":final_prompt.render(),
               "final_prompt_sha256":digest(final_prompt.render()),"selection_uses":"train+validation only"})
    originals += evaluate_prompt(prompt,"original",heldout)
    if baselines:
        evaluate_prompt(prompt,"spotlight",heldout,spotlight=True)
        evaluate_prompt(prompt,"majority",heldout,majority=True)
    final_variant = "original" if final_prompt.render() == prompt.render() else "repaired"
    if final_variant == "repaired":
        evaluate_prompt(final_prompt,final_variant,heldout)
    for case in cases:
        for original in [r for r in originals if r["case_id"]==case.id]:
            row = engine.run(case,prompt,"gate",original["repeat"],gate=True,proposal_record=original)
            collected[row["key"]] = row
    if final_variant == "repaired":
        for case in heldout:
            for original in [r for r in list(collected.values()) if r["variant"]=="repaired" and r["case_id"]==case.id]:
                row=engine.run(case,final_prompt,"repair+gate",original["repeat"],gate=True,proposal_record=original)
                collected[row["key"]]=row
    records=list(collected.values())
    domain=defaultdict(set)
    for c in development:
        for k,v in c.factors.items(): domain[k].add(v)
    result={"backend":engine.backend,"research_evidence":engine.backend=="live",
            "model":engine.model,"n_model_calls":engine.calls,"n_records":len(records),
            "final_variant":final_variant,"metrics":summarize(records,bootstrap,seed),
            "mutation":mutation_report(mutants,records,reviews),
            "coverage":coverage(train,domain),
            "fault_localization":ochiai([r for r in originals if r["split"]=="train"]),
            "repair":history,"limitations":[
                "Fixture backend is synthetic software validation, not an LLM experiment.",
                "Independent labels are authored fixtures; human label audit still required.",
                "Gate uses a narrow English grammar; unknown/paraphrased intent may be rejected.",
                "TRUST causality is not directly observed; its per-requirement oracle is null.",
                "Majority is a token-consistency baseline; fixed context has no retrieval filtering.",
                "No USD inferred without an explicit pricing table; word_delta is not token_delta.",
                "Experimental sample size/backbone count does not establish publication claims."]}
    write_json(engine.output/"records.json",records)
    write_json(engine.output/"summary.json",result)
    write_json(engine.output/"final.json",{
        "status":"completed_with_errors" if any(r["status"] != "ok" for r in records) else "completed",
        "manifest":json.loads((engine.output/"manifest.json").read_text(encoding="utf-8")),
        "summary":result,
        "records":records,
    })
    write_json(engine.output/"mutants.json",[{"id":m.id,"requirement_id":m.requirement_id,"operator":m.operator,
                "original":m.original,"replacement":m.replacement,"prompt":m.prompt.render()} for m in mutants])
    render_report(engine.output,result,records)
    return result


def render_report(output,result,records):
    esc=html.escape
    rows="".join("<tr>"+"".join(f"<td>{esc(str(row.get(k)))}</td>" for k in
        ("variant","split","n_ok","n_errors","clean_utility","attack_proposal_violation","attack_execution_violation","attack_utility","clean_instability","input_tokens","output_tokens"))+"</tr>" for row in result["metrics"])
    details="".join(f'<details><summary>{esc(r["variant"])} / {esc(r["case_id"])} / run {r["repeat"]}: {esc(r["status"])}</summary><pre>{esc(json.dumps(r,ensure_ascii=False,indent=2))}</pre></details>' for r in records)
    note="LIVE MODEL RUN" if result["research_evidence"] else "FIXTURE SOFTWARE CHECK — NOT RESEARCH EVIDENCE"
    content=f'''<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ADAPT-RT experiment</title>
<style>body{{font:15px/1.6 system-ui;margin:30px;background:#f4f3ec;color:#183b36}}h1{{font-size:36px}}table{{border-collapse:collapse;background:white}}th,td{{padding:10px;border:1px solid #ccd5c7;text-align:left}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#fff;padding:18px}}details{{margin:8px 0}}summary{{cursor:pointer}}.table{{overflow:auto}}.tag{{background:#e3bd78;padding:10px}}</style>
<h1>ADAPT-RT / Experiment report</h1><p class="tag">{note}</p><p>{esc(result['model'])} · {result['n_model_calls']} model calls · {result['n_records']} records</p>
<p>Train: chẩn đoán/mutation. Validation: chọn bản vá. Held-out: đo sau khi khóa bản vá. Gate được chấm bằng expected calls độc lập.</p>
<div class="table"><table><tr><th>Variant</th><th>Split</th><th>OK</th><th>Error</th><th>Clean utility</th><th>Attack proposal violation</th><th>Attack execution violation</th><th>Attack utility</th><th>Clean instability</th><th>Input tokens</th><th>Output tokens</th></tr>{rows}</table></div>
<h2>Mutation & repair</h2><pre>{esc(json.dumps({'mutation':result['mutation'],'repair':result['repair']},ensure_ascii=False,indent=2))}</pre>
<h2>Giới hạn</h2><ul>{''.join('<li>'+esc(s)+'</li>' for s in result['limitations'])}</ul><h2>Trace từng case</h2>{details}</html>'''
    (output/"REPORT.html").write_text(content,encoding="utf-8")
    (output/"REPORT.md").write_text(f"# ADAPT-RT\n\n{note}\n\nRead REPORT.html or summary.json. Raw observations: records.jsonl.\n",encoding="utf-8")
