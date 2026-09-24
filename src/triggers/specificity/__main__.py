"""Explore query specificity with ablations and unseen audit models.

python -m src.triggers.specificity --source RUN --output OUTPUT
"""
import argparse
import csv
import gc
import math
import random
from pathlib import Path

import torch

from src.triggers.hierarchy.backends import Encoder
from src.triggers.hierarchy.evaluation import routed_groups
from src.triggers.hierarchy.fluency import FluencyScorer
from src.triggers.hierarchy.io import digest, read, write
from src.triggers.hierarchy.language_audit import paired_comparison
from .audit import audit, render_report
from .candidates import render
from .selection import Selector
from .phrases import NounPhraseExtractor

SCOPES = ("universal", "semantic", "per_query")
MODES = ("generic_language", "grounded_language", "topic_heading")


def fresh_partition(source_data, test_path, size, seed, extra_seen=()):
    excluded_ids = {r["qid"] for split in ("train", "validation", "test", "sources") for r in source_data[split]}
    excluded_texts = {" ".join(r["question"].split()).casefold() for split in ("train", "validation", "test", "sources") for r in source_data[split]}
    excluded_ids.update(str(r["qid"]) for r in extra_seen)
    excluded_texts.update(" ".join(r["question"].split()).casefold() for r in extra_seen)
    pool, seen = [], set()
    for row in read(test_path):
        qid, q = str(row["qid"]), " ".join(row["question"].split())
        if qid in excluded_ids or q.casefold() in excluded_texts or q.casefold() in seen:
            continue
        seen.add(q.casefold())
        pool.append({"qid": qid, "question": q})
    random.Random(seed).shuffle(pool)
    if len(pool) < size or size < 2:
        raise ValueError("Not enough fresh queries disjoint from the source experiment")
    return pool[:size]


def render_rows(rows, assignments, triggers, style="heading", feasible=None, position="prefix"):
    if len(rows) != len(assignments):
        raise ValueError("Routing assignments must match query count")
    return [{"qid": row["qid"], "original": row["question"], "trigger": triggers[g],
             "altered": render(row["question"], triggers[g], style, position),
             "selection_feasible": bool(feasible[g]) if feasible is not None else True} for row, g in zip(rows, assignments)]


def choose_on_validation(results):
    choices, details = {}, {}
    for scope in SCOPES:
        candidates = [(name, r["metrics"]) for name, r in results.items() if name.startswith(scope + "/") and not name.endswith("legacy_raw")]
        eligible = [(n, m) for n, m in candidates if m["preservation_proxy_pass_rate"] >= .9 and m["geometric_mean_ppl_ratio"] <= 1.5 and m.get("selection_feasible_rate", 1.) >= .9]
        score = lambda item: item[1]["trigger_query_cosine"] + .25 * item[1]["specificity_contrast"] - .25 * max(0., item[1]["nll_delta"])
        chosen = max(eligible or candidates, key=score)
        choices[scope] = chosen[0]
        details[scope] = {"variant": chosen[0], "validation_constraints_met": bool(eligible), "validation_score": score(chosen)}
    return choices, details


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--fresh-test-size", type=int, default=48)
    p.add_argument("--position", choices=["suffix", "prefix", "infix", "sentence"], default="prefix",
                   help="Which bank position to score. Reads SOURCE/<position>/<scope>.json, so the "
                        "bank stage must have been run with that position in --positions.")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--candidate-cap", type=int, default=20)
    p.add_argument("--phrase-mode", choices=["heuristic", "pos"], default="pos")
    p.add_argument("--min-grounded-relevance", type=float, default=.15)
    p.add_argument("--exclude-evaluated", type=Path, nargs="*", default=[], help="Additional dataset.json files whose queries must not enter fresh test")
    p.add_argument("--device", default="cpu")
    p.add_argument("--model-cache", default="outputs/model-cache")
    args = p.parse_args(argv)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    source = read(args.source / "config.json")
    data = read(args.source / "dataset.json")
    if data["fixture"]:
        p.error("This experiment requires real queries")
    extra_seen = [row for path in args.exclude_evaluated for rows in read(path).values() for row in rows]
    fresh = fresh_partition(data, source["config"]["test"], args.fresh_test_size, args.seed, extra_seen)
    splits = {"train": [{"qid": r["qid"], "question": r["question"]} for r in data["train"]],
              "validation": [{"qid": r["qid"], "question": r["question"]} for r in data["validation"]], "test": fresh}
    missing = [s for s in SCOPES if not (args.source / args.position / f"{s}.json").exists()]
    if missing:
        p.error(f"No {args.position} bank for {missing} under {args.source}; "
                f"re-run the bank stage with --positions {args.position}")
    artifacts = {s: read(args.source / args.position / f"{s}.json") for s in SCOPES}
    if any(r["contract"] != digest(source) for r in artifacts.values()):
        raise ValueError("Source bank contract mismatch")
    contract = {"version": 2, "source": source, "source_banks": digest(artifacts), "dataset": digest(splits),
                "seed": args.seed, "candidate_cap": args.candidate_cap, "position": args.position,
                "phrase_mode": args.phrase_mode, "min_grounded_relevance": args.min_grounded_relevance,
                "extra_seen": digest(extra_seen),
                "max_words": 3, "max_tokens": 12, "meaning_threshold": .85, "ppl_limit": 1.5,
                "objective": "mean_relevance + .25*min_relevance + .25*contrast - .25*max(0,mean_nll_delta)"}
    if (args.output / "config.json").exists() and read(args.output / "config.json") != contract:
        p.error("Changed experiment contract; use a fresh output directory")
    write(args.output / "config.json", contract)
    write(args.output / "dataset.json", splits)
    print("Routing fresh queries with frozen clean DPR centers", flush=True)
    router = Encoder(source["config"]["model"], pooling="dpr", device=args.device)
    fresh_vectors = router.encode([r["question"] for r in fresh])
    assignments = {}
    for scope, a in artifacts.items():
        by_id = {r["qid"]: r["group"] for r in a["evaluations"]["validation"]["records"]}
        assignments[scope] = {"train": a["bank"]["training_assignments"],
                              "validation": [by_id[r["qid"]] for r in splits["validation"]],
                              "test": routed_groups(a["bank"], fresh, fresh_vectors).tolist()}
    write(args.output / "assignments.json", assignments)
    del router, fresh_vectors
    gc.collect()
    semantic = Encoder("sentence-transformers/all-MiniLM-L6-v2", device=args.device)
    fluency = FluencyScorer(cache_dir=args.model_cache, device=args.device)
    noun_extractor = NounPhraseExtractor(args.model_cache, args.device) if args.phrase_mode == "pos" else None
    options = {"phrase_extractor": noun_extractor.phrases} if noun_extractor else {}
    selector = Selector(semantic, fluency, candidate_cap=args.candidate_cap,
                        min_grounded_relevance=args.min_grounded_relevance, position=args.position, **options)
    # Every Selector.fit() is scored language search and is the expensive part of this
    # run, so each one is checkpointed by key. The cache lives inside the output
    # directory, which the contract check above already pins to one configuration, so a
    # resumed run cannot mix results from different settings. Delete the file to refit.
    cache_path = args.output / "fit_cache.json"
    cache = read(cache_path) if cache_path.exists() else {}
    resumed = len(cache)

    def fit_cached(key, rows, mode):
        if key not in cache:
            cache[key] = selector.fit(rows, mode, splits["train"])
            write(cache_path, cache)
        return cache[key]

    if resumed:
        print(f"Resuming from {resumed} cached fits in {cache_path}", flush=True)
    generated = {split: {} for split in splits}
    fits = {}
    for scope, a in artifacts.items():
        for method in ("legacy_raw", "legacy_heading", *MODES):
            variant = f"{scope}/{method}"
            print(f"Fit {variant}", flush=True)
            if method.startswith("legacy"):
                triggers = a["bank"]["triggers"]
                feasible = [r["constraint_feasible"] for r in a["bank"]["training"]]
            else:
                results = []
                for group in range(len(a["bank"]["triggers"])):
                    rows = [r for r, g in zip(splits["train"], assignments[scope]["train"]) if g == group]
                    results.append(fit_cached(f"{variant}/group-{group}", rows, method))
                fits[variant] = results
                triggers = [r["trigger"] for r in results]
                feasible = [r["constraint_feasible"] for r in results]
            for split, rows in splits.items():
                generated[split][variant] = render_rows(rows, assignments[scope][split], triggers,
                                                       "raw" if method == "legacy_raw" else "heading", feasible,
                                                       args.position)
    # Incoming queries may condition language search, but no answers or test
    # aggregate metrics are available to Selector.fit(). Report this extra cost.
    adaptive = "per_query/query_adaptive"
    for split in ("train", "validation"):
        print(f"Adaptive language selection: {split}", flush=True)
        fits[f"{adaptive}/{split}"] = [fit_cached(f"{adaptive}/{split}/{row['qid']}", [row], "grounded_language")
                                       for row in splits[split]]
        generated[split][adaptive] = render_rows(splits[split], list(range(len(splits[split]))),
                                                [r["trigger"] for r in fits[f"{adaptive}/{split}"]],
                                                feasible=[r["constraint_feasible"] for r in fits[f"{adaptive}/{split}"]],
                                                position=args.position)
    validation = {name: audit(rows, semantic, fluency) for name, rows in generated["validation"].items()}
    choices, choice_details = choose_on_validation(validation)
    write(args.output / "selection.json", {"choices": choices, "details": choice_details,
          "rule": "Choose feasible validation variant by mean relevance + .25 contrast - .25 positive NLL delta; fallback best score, explicitly flagged"})
    print(f"Frozen validation choices: {choices}", flush=True)
    fits[f"{adaptive}/test"] = []
    for i, row in enumerate(fresh):
        if i % 8 == 0:
            print(f"Adaptive fresh queries: {i}/{len(fresh)}", flush=True)
        fits[f"{adaptive}/test"].append(fit_cached(f"{adaptive}/test/{row['qid']}", [row], "grounded_language"))
    generated["test"][adaptive] = render_rows(fresh, list(range(len(fresh))), [r["trigger"] for r in fits[f"{adaptive}/test"]],
                                            feasible=[r["constraint_feasible"] for r in fits[f"{adaptive}/test"]],
                                            position=args.position)
    write(args.output / "fits.json", fits)
    write(args.output / "generated.json", generated)
    results = {"selection_models": {"validation": validation}}
    for split in ("train", "test"):
        results["selection_models"][split] = {name: audit(rows, semantic, fluency) for name, rows in generated[split].items()}
    model_metadata = {"selection_semantic": semantic.metadata, "selection_lm": fluency.metadata}
    if noun_extractor:
        model_metadata["phrase_extractor"] = noun_extractor.metadata
    del selector, semantic, fluency, noun_extractor, options
    gc.collect()
    print("Auditing frozen strings with different model weights", flush=True)
    verifier = Encoder("sentence-transformers/paraphrase-MiniLM-L3-v2", max_length=128, cache_dir=args.model_cache, device=args.device)
    verifier_lm = FluencyScorer("openai-community/gpt2", cache_dir=args.model_cache, device=args.device)
    model_metadata.update({"audit_semantic": verifier.metadata, "audit_lm": verifier_lm.metadata})
    results["audit_models"] = {}
    for split in ("train", "validation", "test"):
        print(f"Audit models: {split}", flush=True)
        results["audit_models"][split] = {name: audit(rows, verifier, verifier_lm) for name, rows in generated[split].items()}
    for scorer, evaluated in results.items():
        for split, variants in evaluated.items():
            for name, r in variants.items():
                scope = name.split("/")[0]
                paired = paired_comparison(r["records"], variants[f"{scope}/legacy_raw"]["records"])
                r["paired_vs_legacy_same_scope"] = {k: {"mean_delta_vs_legacy": v["mean_delta_vs_universal"],
                                                       "paired_bootstrap_95_interval": v["paired_bootstrap_95_interval"]}
                                                    for k, v in paired.items()}
    write(args.output / "models.json", model_metadata)
    write(args.output / "results.json", results)
    (args.output / "REPORT.md").write_text(render_report(results, choices, args.position), encoding="utf-8")
    records = [{"scorer": scorer, "split": split, "variant": name,
                **{k: v for k, v in row.items() if not isinstance(v, dict)}}
               for scorer, evaluated in results.items() for split, variants in evaluated.items()
               for name, r in variants.items() for row in r["records"]]
    with (args.output / "records.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved {len(records)} audit rows to {args.output}")
    print(f"Language fits: {len(cache)} total, {resumed} reused from a previous run")


if __name__ == "__main__":
    main()
