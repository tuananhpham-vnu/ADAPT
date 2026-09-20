"""Small black-box trigger optimizer for direct poison-vs-clean top-K margin.

The budget is counted in evaluated discrete trigger proposals. Search mixes
single-token and pair-token mutations, so it can retain improvements that a
strict one-coordinate optimizer cannot propose.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import random

import numpy as np

from validate_trigger_interactions import (
    DEFAULT_INDEX, ENCODER_REVISION, Encoder, read_json,
    retrieval_metrics, split_questions,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_QUESTIONS = ROOT / "ReAct/database/strategyqa_train_filtered.json"
DEFAULT_DEV_QUESTIONS = ROOT / "ReAct/database/strategyqa_dev.json"


def validate_questions(rows, path):
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Questions must be a nonempty JSON list: {path}")
    for row in rows:
        if not row.get("qid") or not row.get("question"):
            raise ValueError(f"Each question needs qid and question: {path}")


def assert_disjoint(train_questions, dev_questions):
    train_ids = {row["qid"] for row in train_questions}
    dev_ids = {row["qid"] for row in dev_questions}
    shared_ids = train_ids & dev_ids
    train_texts = {row["question"].strip().casefold() for row in train_questions}
    dev_texts = {row["question"].strip().casefold() for row in dev_questions}
    shared_texts = train_texts & dev_texts
    if shared_ids or shared_texts:
        raise ValueError(
            "Train/dev leakage detected: "
            f"{len(shared_ids)} shared qids, {len(shared_texts)} shared question texts"
        )


def clean_start_tokens(artifact, tokenizer):
    special = set(tokenizer.all_special_tokens)
    tokens = [t for t in artifact["tokens"] if t not in special and t != "[UNK]"]
    tokens = [t for t in tokens if t.strip()]
    if not tokens:
        raise ValueError("No ordinary tokens remain after removing special tokens")
    return tuple(tokens)


def candidate_vocabulary(tokenizer, questions, start, size):
    special = set(tokenizer.all_special_tokens)
    counts = Counter()
    for row in questions:
        for token in tokenizer.tokenize(row["question"]):
            if (token not in special and not token.startswith("##")
                    and token.isalpha() and len(token) > 1):
                counts[token] += 1
    vocab = list(dict.fromkeys([*start, ".", *[t for t, _ in counts.most_common(size)]]))
    return vocab[:size]


def evaluate_candidates(encoder, candidates, train_questions, poison_sources,
                        clean_vectors, top_k):
    texts = []
    decoded = []
    for tokens in candidates:
        trigger = encoder.text_from_tokens(list(tokens))
        decoded.append(trigger)
        texts.extend(["Question: " + row["question"] + "\n" + trigger + "\n"
                      for row in train_questions])
        texts.extend([row["question"] + " " + trigger for row in poison_sources])
    vectors = encoder.encode(texts)
    width = len(train_questions) + len(poison_sources)
    results = []
    for index, tokens in enumerate(candidates):
        chunk = vectors[index * width:(index + 1) * width]
        query_vectors = chunk[:len(train_questions)]
        poison_vectors = chunk[len(train_questions):]
        clean_scores = query_vectors @ clean_vectors.T
        poison_scores = query_vectors @ poison_vectors.T
        best_poison = poison_scores.max(axis=1)
        kth_clean = np.partition(clean_scores, -top_k, axis=1)[:, -top_k]
        margins = best_poison - kth_clean
        mean_margin = float(margins.mean())
        p10_margin = float(np.quantile(margins, 0.10))
        results.append({
            "tokens": list(tokens), "trigger": decoded[index],
            "fitness": mean_margin + 0.25 * p10_margin,
            "mean_margin_at_k": mean_margin,
            "p10_margin_at_k": p10_margin,
            "worst_margin_at_k": float(margins.min()),
            "recall_at_k": float((margins > 0).mean()),
        })
    return results


def mutate(parent, vocabulary, rng, pair_probability):
    child = list(parent)
    count = 2 if len(child) > 1 and rng.random() < pair_probability else 1
    positions = rng.sample(range(len(child)), count)
    for position in positions:
        choices = [token for token in vocabulary if token != child[position]]
        child[position] = rng.choice(choices)
    return tuple(child), count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trigger-file", type=Path, required=True)
    parser.add_argument("--train-questions", type=Path, default=DEFAULT_TRAIN_QUESTIONS)
    parser.add_argument("--dev-questions", type=Path, default=DEFAULT_DEV_QUESTIONS)
    parser.add_argument("--iterations", type=int, default=500,
                        help="Number of discrete proposals evaluated")
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--train-queries", type=int, default=12)
    parser.add_argument("--eval-queries", type=int, default=0,
                        help="Number of dev queries to evaluate; 0 means the full dev split")
    parser.add_argument("--poison-count", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--pair-probability", type=float, default=0.4)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--local-files-only", action="store_true",
                        help="Require the Hugging Face model to already be cached")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 1 or args.population < 1 or args.train_queries < 1:
        parser.error("iterations, population, and train-queries must be positive")
    if args.eval_queries < 0:
        parser.error("eval-queries must be nonnegative")

    artifact = read_json(args.trigger_file)
    all_train_questions = read_json(args.train_questions)
    all_dev_questions = read_json(args.dev_questions)
    validate_questions(all_train_questions, args.train_questions)
    validate_questions(all_dev_questions, args.dev_questions)
    assert_disjoint(all_train_questions, all_dev_questions)
    poison_sources, train_questions = split_questions(
        all_train_questions, args.seed, args.poison_count, args.train_queries,
    )
    eval_questions = (
        all_dev_questions if args.eval_queries == 0
        else all_dev_questions[:args.eval_queries]
    )
    if not eval_questions:
        raise ValueError("Dev evaluation split is empty")
    index_manifest = read_json(DEFAULT_INDEX / "manifest.json")
    if index_manifest.get("encoder_revision") != ENCODER_REVISION:
        raise ValueError("DPR index was built with a different encoder revision")
    clean_vectors = np.load(DEFAULT_INDEX / "vectors.npy", mmap_mode="r")
    encoder = Encoder(args.device, args.batch_size, args.local_files_only)
    start = clean_start_tokens(artifact, encoder.tokenizer)
    vocabulary = candidate_vocabulary(
        encoder.tokenizer, all_train_questions, start, args.vocab_size
    )

    rng = random.Random(args.seed)
    baseline = evaluate_candidates(
        encoder, [start], train_questions, poison_sources,
        clean_vectors, args.top_k,
    )[0]
    incumbent = baseline
    cache = {tuple(start): baseline}
    history = []
    evaluated = 0
    single_proposals = pair_proposals = 0
    generation = 0
    while evaluated < args.iterations:
        wanted = min(args.population, args.iterations - evaluated)
        proposals = []
        mutation_sizes = {}
        attempts = 0
        while len(proposals) < wanted:
            attempts += 1
            if attempts > wanted * 100:
                raise RuntimeError("Could not produce enough unique proposals")
            candidate, mutation_size = mutate(
                tuple(incumbent["tokens"]), vocabulary, rng,
                args.pair_probability,
            )
            if candidate in cache or candidate in proposals:
                continue
            proposals.append(candidate)
            mutation_sizes[candidate] = mutation_size
        rows = evaluate_candidates(
            encoder, proposals, train_questions, poison_sources,
            clean_vectors, args.top_k,
        )
        evaluated += len(rows)
        generation += 1
        for candidate, row in zip(proposals, rows):
            cache[candidate] = row
            if mutation_sizes[candidate] == 1:
                single_proposals += 1
            else:
                pair_proposals += 1
        best_generation = max(rows, key=lambda row: (row["fitness"], row["recall_at_k"]))
        improved = best_generation["fitness"] > incumbent["fitness"]
        if improved:
            incumbent = best_generation
        history.append({
            "generation": generation, "evaluated": evaluated,
            "improved": improved, "incumbent": incumbent,
        })
        print(
            f"generation={generation} evaluated={evaluated}/{args.iterations} "
            f"fitness={incumbent['fitness']:.6f} trigger={incumbent['trigger']!r}",
            flush=True,
        )

    # Fresh before/after evaluation on the independent StrategyQA dev file.
    final_eval = evaluate_candidates(
        encoder, [start, tuple(incumbent["tokens"])], eval_questions,
        poison_sources, clean_vectors, args.top_k,
    )
    before, after = final_eval
    ks = [1, 2, 5, 10]
    detailed = {}
    for label, row in (("before", before), ("after", after)):
        trigger = row["trigger"]
        qv = encoder.encode(["Question: " + q["question"] + "\n" + trigger + "\n"
                             for q in eval_questions])
        pv = encoder.encode([q["question"] + " " + trigger for q in poison_sources])
        detailed[label] = {
            "tokens": row["tokens"], "trigger": trigger,
            **retrieval_metrics(qv, clean_vectors, pv, ks),
        }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    completed = {
        "trigger": after["trigger"], "tokens": after["tokens"],
        "iteration": args.iterations - 1, "state": "completed",
        "agent": artifact.get("agent", "qa"), "algo": "topk_pair_blackbox",
        "model": "dpr-ctx_encoder-single-nq-base",
        "num_adv_passage_tokens": len(after["tokens"]),
        "origin": str(args.output_dir.resolve()),
    }
    (args.output_dir / "trigger.json").write_text(
        json.dumps(completed, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": vars(args) | {
            "trigger_file": str(args.trigger_file),
            "output_dir": str(args.output_dir),
        },
        "data": {
            "optimization_split": "train",
            "evaluation_split": "dev",
            "train_questions_path": str(args.train_questions.resolve()),
            "dev_questions_path": str(args.dev_questions.resolve()),
            "clean_keys": len(clean_vectors), "poison_keys": len(poison_sources),
            "train_queries": len(train_questions), "eval_queries": len(eval_questions),
            "poison_source_qids": [q["qid"] for q in poison_sources],
            "train_qids": [q["qid"] for q in train_questions],
            "eval_qids": [q["qid"] for q in eval_questions],
        },
        "search": {
            "start": baseline, "best_train": incumbent,
            "single_proposals": single_proposals,
            "pair_proposals": pair_proposals,
            "vocabulary": vocabulary, "history": history,
        },
        "dev": detailed,
    }
    # Path objects are converted only in config.
    result["config"] = {k: str(v) if isinstance(v, Path) else v
                        for k, v in result["config"].items()}
    (args.output_dir / "optimization.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"completed: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
