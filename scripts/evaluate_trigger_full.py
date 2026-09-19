"""Compare the baseline and optimized triggers on StrategyQA dev."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from validate_trigger_interactions import (
    DEFAULT_INDEX, ENCODER_REVISION, Encoder, read_json,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_QUESTIONS = ROOT / "ReAct/database/strategyqa_train_filtered.json"
DEFAULT_DEV_QUESTIONS = ROOT / "ReAct/database/strategyqa_dev.json"


def per_query_metrics(query_vectors, clean_vectors, poison_vectors, ks, chunk_size=128):
    fields = {"best_poison": [], "best_clean": [], "rank": []}
    for k in ks:
        fields[f"margin_{k}"] = []
    for start in range(0, len(query_vectors), chunk_size):
        query = query_vectors[start:start + chunk_size]
        clean_scores = query @ clean_vectors.T
        best_poison = (query @ poison_vectors.T).max(axis=1)
        fields["best_poison"].append(best_poison)
        fields["best_clean"].append(clean_scores.max(axis=1))
        fields["rank"].append(1 + (clean_scores > best_poison[:, None]).sum(axis=1))
        for k in ks:
            kth = np.partition(clean_scores, -k, axis=1)[:, -k]
            fields[f"margin_{k}"].append(best_poison - kth)
    return {key: np.concatenate(value) for key, value in fields.items()}


def summarize(values, indices, ks):
    best_poison = values["best_poison"][indices]
    best_clean = values["best_clean"][indices]
    ranks = values["rank"][indices]
    result = {
        "n": int(len(indices)),
        "mean_best_poison_score": float(best_poison.mean()),
        "mean_best_clean_score": float(best_clean.mean()),
        "mean_margin_vs_best_clean": float((best_poison - best_clean).mean()),
        "poison_rank_mean": float(ranks.mean()),
        "poison_rank_median": float(np.median(ranks)),
        "poison_rank_p95": float(np.quantile(ranks, 0.95)),
        "poison_rank_worst": int(ranks.max()),
        "poison_mrr": float((1.0 / ranks).mean()),
    }
    for k in ks:
        margins = values[f"margin_{k}"][indices]
        result[f"mean_margin_at_{k}"] = float(margins.mean())
        result[f"median_margin_at_{k}"] = float(np.median(margins))
        result[f"p05_margin_at_{k}"] = float(np.quantile(margins, 0.05))
        result[f"worst_margin_at_{k}"] = float(margins.min())
        result[f"poison_recall_at_{k}"] = float((margins > 0).mean())
        result[f"failures_at_{k}"] = int((margins <= 0).sum())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimization-json", type=Path, required=True)
    parser.add_argument("--completed-trigger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-questions", type=Path, default=DEFAULT_TRAIN_QUESTIONS)
    parser.add_argument("--dev-questions", type=Path, default=DEFAULT_DEV_QUESTIONS)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--local-files-only", action="store_true",
                        help="Require the Hugging Face model to already be cached")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    optimization = read_json(args.optimization_json)
    completed = read_json(args.completed_trigger)
    train = read_json(args.train_questions)
    evaluation = read_json(args.dev_questions)
    source_ids = optimization["data"].get("poison_source_qids")
    if not source_ids:
        raise ValueError("Optimization artifact lacks data.poison_source_qids; rerun optimizer")
    train_by_id = {row["qid"]: row for row in train}
    missing = [qid for qid in source_ids if qid not in train_by_id]
    if missing:
        raise ValueError(f"Poison sources are missing from train: {missing}")
    sources = [train_by_id[qid] for qid in source_ids]
    train_ids = set(train_by_id)
    dev_ids = {row["qid"] for row in evaluation}
    train_texts = {row["question"].strip().casefold() for row in train}
    dev_texts = {row["question"].strip().casefold() for row in evaluation}
    if train_ids & dev_ids or train_texts & dev_texts:
        raise ValueError("Train/dev leakage detected")

    before_trigger = optimization["search"]["start"]["trigger"]
    after_trigger = completed["trigger"]
    all_indices = np.arange(len(evaluation))

    index_manifest = read_json(DEFAULT_INDEX / "manifest.json")
    if index_manifest.get("encoder_revision") != ENCODER_REVISION:
        raise ValueError("DPR index was built with a different encoder revision")
    clean_vectors = np.load(DEFAULT_INDEX / "vectors.npy", mmap_mode="r")
    encoder = Encoder(args.device, args.batch_size, args.local_files_only)
    ks = [1, 2, 5, 10]
    results = {}
    for label, trigger in (("before", before_trigger), ("after", after_trigger)):
        print(f"encoding {label}: {trigger!r}", flush=True)
        query_vectors = encoder.encode([
            "Question: " + row["question"] + "\n" + trigger + "\n"
            for row in evaluation
        ])
        poison_vectors = encoder.encode([
            row["question"] + " " + trigger for row in sources
        ])
        values = per_query_metrics(query_vectors, clean_vectors, poison_vectors, ks)
        results[label] = {
            "trigger": trigger,
            "dev": summarize(values, all_indices, ks),
        }

    output = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "optimization_split": "train", "evaluation_split": "dev",
            "train_questions_path": str(args.train_questions.resolve()),
            "dev_questions_path": str(args.dev_questions.resolve()),
            "poison_count": len(sources), "clean_keys": len(clean_vectors),
            "dev_queries": len(all_indices),
            "optimization_queries": len(optimization["data"]["train_qids"]), "ks": ks,
        },
        "source_qids": [row["qid"] for row in sources],
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
