"""Evaluate relevance and fluency of frozen triggers, without retrieval scoring.

Run: python -m src.triggers.hierarchy.language_audit --input RUN --output AUDIT
"""
import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch

from .backends import Encoder
from .fluency import FluencyScorer
from .io import digest, read, write
from .text import insert


def relevance_scores(query_vectors, trigger_vectors):
    """Matched cosine and contrast against ALL other queries in the same split."""
    if len(query_vectors) < 2 or len(query_vectors) != len(trigger_vectors):
        raise ValueError("Need aligned query/trigger vectors and at least two distinct queries")
    similarity = trigger_vectors @ query_vectors.T
    matched = similarity.diag()
    other = (similarity.sum(1) - matched) / (len(matched) - 1)
    return matched.tolist(), (matched - other).tolist()


def paired_comparison(rows, baseline, seed=42):
    by_id = {r["qid"]: r for r in baseline}
    if len(by_id) != len(baseline) or set(by_id) != {r["qid"] for r in rows} or len(rows) != len(baseline):
        raise ValueError("Paired arms must contain identical unique query IDs")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(rows), (2000, len(rows)))
    result = {}
    for metric in ("trigger_query_cosine", "specificity_contrast", "nll_delta"):
        differences = np.array([r[metric] - by_id[r["qid"]][metric] for r in rows])
        low, high = np.quantile(differences[draws].mean(1), [.025, .975])
        result[metric] = {"mean_delta_vs_universal": float(differences.mean()),
                          "paired_bootstrap_95_interval": [float(low), float(high)]}
    return result


def extract_records(artifact, dataset, split):
    if split != "train":
        return [{k: r[k] for k in ("qid", "original", "altered", "trigger")}
                for r in artifact["evaluations"][split]["records"]]
    bank = artifact["bank"]
    labels = bank.get("training_assignments", [0] * len(dataset["train"]))
    if len(labels) != len(dataset["train"]):
        raise ValueError("Training assignments must match training queries")
    return [{"qid": row["qid"], "original": row["question"], "trigger": bank["triggers"][g],
             "altered": insert(row["question"], bank["triggers"][g], artifact["position"])}
            for row, g in zip(dataset["train"], labels)]


def measure(rows, encoder, fluency):
    if len({r["qid"] for r in rows}) != len(rows) or len({r["original"] for r in rows}) != len(rows):
        raise ValueError("Expected distinct queries within each arm")
    queries, triggers, altered = ([r[k] for r in rows] for k in ("original", "trigger", "altered"))
    qv, tv, av = encoder.encode(queries), encoder.encode(triggers), encoder.encode(altered)
    relevance, contrast = relevance_scores(qv, tv)
    preservation = (qv * av).sum(1).tolist()
    original_lm, altered_lm = fluency.score(queries), fluency.score(altered)
    measured = []
    for i, row in enumerate(rows):
        delta = altered_lm[i]["nll"] - original_lm[i]["nll"]
        measured.append({**row, "trigger_query_cosine": relevance[i], "specificity_contrast": contrast[i],
                         "original_altered_cosine": preservation[i], "original_lm": original_lm[i],
                         "altered_lm": altered_lm[i], "nll_delta": delta, "ppl_ratio": math.exp(delta)})
    mean = lambda key: sum(r[key] for r in measured) / len(measured)
    metrics = {"n": len(rows), **{k: mean(k) for k in ("trigger_query_cosine", "specificity_contrast", "original_altered_cosine", "nll_delta")},
               "geometric_mean_ppl_ratio": math.exp(mean("nll_delta")),
               "unique_triggers": len(set(triggers))}
    return {"metrics": metrics, "records": measured}


def render(results):
    lines = ["# Frozen-trigger relevance and fluency audit", "",
             "Cosine: higher means closer in MiniLM space, not a probability of relevance.",
             "PPL ratio: geometric mean of altered/original full-text perplexity under DistilGPT2; lower is more predictable to this model.",
             "Neither measure is a human naturalness rating. No retrieval scores are used.", "",
             "| Split | Position | Arm | N | Trigger/query cosine | Specificity contrast | PPL ratio |",
             "|---|---|---|---:|---:|---:|---:|"]
    for split, positions in results.items():
        for position, arms in positions.items():
            for arm, result in arms.items():
                m = result["metrics"]
                lines.append(f"| {split} | {position} | {arm} | {m['n']} | {m['trigger_query_cosine']:.4f} | {m['specificity_contrast']:+.4f} | {m['geometric_mean_ppl_ratio']:.3f} |")
    lines += ["", "## Interpretation limits", "",
              "- Contrast subtracts average similarity to every other query in the split. For a constant universal trigger, its mean is zero by construction.",
              "- The existing search vocabulary contains generic instructions. Specific training did not require query-specific language.",
              "- Train scores evaluate the queries used to learn the triggers; test per-query triggers transfer by nearest clean training query.",
              "- Full-text PPL can improve by adding common words, despite awkward grammar. Names, length, tokenization and text genre affect it.",
              "- Original/altered cosine is logged separately; it is not a trigger relevance measurement.",
              "- Paired bootstrap intervals resample queries within each split/position. Sample sizes are shown above; a single run does not measure variation across training seeds.",
              "- Example controls are sanity probes, not calibration or human validation.", "",
              "Method: [Hugging Face perplexity](https://huggingface.co/docs/transformers/en/perplexity).",
              "Model: [DistilGPT2](https://huggingface.co/distilbert/distilgpt2)."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--splits", nargs="+", choices=["train", "validation", "test"], default=["train", "test"])
    p.add_argument("--lm-cache", default="outputs/model-cache")
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)
    torch.set_num_threads(2)
    source = read(args.input / "config.json")
    dataset = read(args.input / "dataset.json")
    if dataset["fixture"]:
        p.error("Use a real-data run for this language audit")
    semantic = Encoder("sentence-transformers/all-MiniLM-L6-v2", device=args.device)
    fluency = FluencyScorer(cache_dir=args.lm_cache, device=args.device)
    contract = {"version": 1, "source": source, "splits": args.splits,
                "semantic": semantic.metadata, "fluency": fluency.metadata}
    if (args.output / "config.json").exists() and read(args.output / "config.json") != contract:
        p.error("Audit contract changed; choose another output directory")
    results, csv_rows = {}, []
    for split in args.splits:
        results[split] = {}
        for position in source["config"]["positions"]:
            arms = results[split][position] = {}
            for arm in source["config"]["arms"]:
                print(f"Language audit: {split}/{position}/{arm}", flush=True)
                artifact = read(args.input / position / f"{arm}.json")
                if artifact["contract"] != digest(source):
                    raise ValueError("Source arm contract mismatch")
                arms[arm] = measure(extract_records(artifact, dataset, split), semantic, fluency)
                for row in arms[arm]["records"]:
                    csv_rows.append({"split": split, "position": position, "arm": arm,
                                     **{k: v for k, v in row.items() if not isinstance(v, dict)}})
            if "universal" in arms:
                for result in arms.values():
                    result["paired_vs_universal"] = paired_comparison(result["records"], arms["universal"]["records"])
    controls = {"fluency": fluency.score(["The cyclist is riding a bicycle.", "The cyclist riding is a bicycle."]),
                "fluency_texts": ["The cyclist is riding a bicycle.", "The cyclist riding is a bicycle."]}
    q = semantic.encode(["Can bicycles use this lane?"])
    controls["relevance_phrases"] = ["bicycle lane access", "chocolate cake recipe"]
    controls["relevance_cosines"] = (semantic.encode(controls["relevance_phrases"]) @ q.T).flatten().tolist()
    write(args.output / "config.json", contract)
    write(args.output / "results.json", results)
    write(args.output / "controls.json", controls)
    with (args.output / "records.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    (args.output / "REPORT.md").write_text(render(results), encoding="utf-8")
    print(f"Saved {len(csv_rows)} records to {args.output}")


if __name__ == "__main__":
    main()
