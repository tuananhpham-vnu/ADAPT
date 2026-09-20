"""Validate trigger-token interactions and poison-vs-clean retrieval margins.

This is a retriever-only diagnostic.  It intentionally reuses the real local
StrategyQA DPR index and the same poison-key construction as
``src.agentpoison.strategyqa.poison_records``.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / "ReAct/database/strategyqa_train_paragraphs.json"
DEFAULT_QUESTIONS = ROOT / "ReAct/database/strategyqa_train.json"
DEFAULT_TRAIN_QUESTIONS = ROOT / "ReAct/database/strategyqa_train_filtered.json"
DEFAULT_DEV_QUESTIONS = ROOT / "ReAct/database/strategyqa_dev.json"
DEFAULT_INDEX = ROOT / "ReAct/database/embeddings/agentpoison_dpr"
ENCODER = "facebook/dpr-ctx_encoder-single-nq-base"
ENCODER_REVISION = "bb21a3c2b1656d60c6a8e920283bc40dabddadb8"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def split_questions(questions, seed: int, poison_count: int, num_queries: int):
    rows = list(questions)
    random.Random(seed).shuffle(rows)
    sources = rows[:poison_count]
    source_text = {r["question"].strip().casefold() for r in sources}
    evaluation = [
        r for r in rows[poison_count:]
        if r["question"].strip().casefold() not in source_text
    ][:num_queries]
    if len(evaluation) != num_queries:
        raise ValueError("Not enough distinct evaluation questions")
    return sources, evaluation


class Encoder:
    def __init__(self, device: str, batch_size: int, local_files_only: bool = False):
        import torch
        from transformers import AutoTokenizer, DPRContextEncoder

        self.torch = torch
        self.device = device
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(
            ENCODER, revision=ENCODER_REVISION,
            local_files_only=local_files_only,
        )
        self.model = DPRContextEncoder.from_pretrained(
            ENCODER, revision=ENCODER_REVISION,
            local_files_only=local_files_only,
        ).to(device).eval()

    def text_from_tokens(self, tokens: list[str]) -> str:
        return self.tokenizer.convert_tokens_to_string(tokens).strip()

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for start in range(0, len(texts), self.batch_size):
            batch = self.tokenizer(
                texts[start:start + self.batch_size], return_tensors="pt",
                padding=True, truncation=True, max_length=512,
            ).to(self.device)
            with self.torch.inference_mode():
                out = self.model(**batch).pooler_output
                out = self.torch.nn.functional.normalize(out, p=2, dim=1)
            vectors.append(out.cpu().numpy())
        return np.concatenate(vectors).astype("float32")


def retrieval_metrics(query_vectors, clean_vectors, poison_vectors, ks):
    clean_scores = query_vectors @ clean_vectors.T
    poison_scores = query_vectors @ poison_vectors.T
    best_poison = poison_scores.max(axis=1)
    best_clean = clean_scores.max(axis=1)
    ranks = 1 + (clean_scores > best_poison[:, None]).sum(axis=1)
    result = {
        "mean_best_poison_score": float(best_poison.mean()),
        "mean_best_clean_score": float(best_clean.mean()),
        "mean_margin_vs_best_clean": float((best_poison - best_clean).mean()),
        "poison_rank_mean": float(ranks.mean()),
        "poison_mrr": float((1.0 / ranks).mean()),
    }
    for k in ks:
        kth_clean = np.partition(clean_scores, -k, axis=1)[:, -k]
        margin = best_poison - kth_clean
        result[f"mean_margin_at_{k}"] = float(margin.mean())
        result[f"poison_recall_at_{k}"] = float((margin > 0).mean())
        result[f"worst_margin_at_{k}"] = float(margin.min())
    return result


def l_uni(query_vectors: np.ndarray, centers: np.ndarray) -> float:
    distances = np.linalg.norm(
        query_vectors[:, None, :] - centers[None, :, :], axis=2
    )
    mean = query_vectors.mean(axis=0, keepdims=True)
    variance_term = np.linalg.norm(query_vectors - mean, axis=1).mean()
    return float(distances.mean() - 0.1 * variance_term)


def numpy_kmeans(values: np.ndarray, n_clusters: int, seed: int,
                 max_iter: int = 100) -> np.ndarray:
    """Small dependency-free Lloyd k-means for the cached normalized vectors."""
    rng = np.random.default_rng(seed)
    centers = np.asarray(values[rng.choice(len(values), n_clusters, replace=False)]).copy()
    previous = None
    for _ in range(max_iter):
        # Squared distance without materializing [n, k, d].
        distances = (
            (values * values).sum(axis=1, keepdims=True)
            - 2 * values @ centers.T
            + (centers * centers).sum(axis=1)[None, :]
        )
        labels = distances.argmin(axis=1)
        if previous is not None and np.array_equal(labels, previous):
            break
        previous = labels
        for cluster in range(n_clusters):
            members = values[labels == cluster]
            if len(members):
                centers[cluster] = members.mean(axis=0)
    return centers.astype("float32")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trigger-file", type=Path, required=True)
    parser.add_argument("--source-questions", type=Path, default=DEFAULT_TRAIN_QUESTIONS)
    parser.add_argument("--evaluation-questions", type=Path, default=DEFAULT_DEV_QUESTIONS)
    parser.add_argument("--num-queries", type=int, default=0,
                        help="Number of evaluation queries; 0 means the full evaluation file")
    parser.add_argument("--poison-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--local-files-only", action="store_true",
                        help="Require the Hugging Face model to already be cached")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")

    trigger_artifact = read_json(args.trigger_file)
    tokens = trigger_artifact.get("tokens")
    if not isinstance(tokens, list) or not tokens or not all(isinstance(t, str) for t in tokens):
        raise ValueError("Trigger artifact must contain a nonempty string token list")

    corpus = read_json(DEFAULT_CORPUS)
    source_questions = read_json(args.source_questions)
    evaluation_questions = read_json(args.evaluation_questions)
    source_rows = list(source_questions)
    random.Random(args.seed).shuffle(source_rows)
    sources = source_rows[:args.poison_count]
    evaluation = (
        evaluation_questions if args.num_queries == 0
        else evaluation_questions[:args.num_queries]
    )
    source_ids = {row["qid"] for row in source_questions}
    evaluation_ids = {row["qid"] for row in evaluation_questions}
    source_texts = {row["question"].strip().casefold() for row in source_questions}
    evaluation_texts = {
        row["question"].strip().casefold() for row in evaluation_questions
    }
    if source_ids & evaluation_ids or source_texts & evaluation_texts:
        raise ValueError("Source/evaluation question files overlap")
    if len(sources) != args.poison_count or not evaluation:
        raise ValueError("Not enough poison sources or evaluation questions")
    manifest = read_json(DEFAULT_INDEX / "manifest.json")
    clean_vectors = np.load(DEFAULT_INDEX / "vectors.npy", mmap_mode="r")
    if len(clean_vectors) != len(corpus) or manifest.get("paragraph_ids") != list(corpus):
        raise ValueError("DPR index does not match the StrategyQA corpus")
    if manifest.get("encoder_revision") != ENCODER_REVISION:
        raise ValueError("DPR index was built with a different encoder revision")

    encoder = Encoder(args.device, args.batch_size, args.local_files_only)
    full_trigger = trigger_artifact.get("trigger") or encoder.text_from_tokens(tokens)
    poison_texts = [row["question"] + " " + full_trigger for row in sources]
    poison_vectors = encoder.encode(poison_texts)

    variants: dict[str, str] = {"empty": "", "full": full_trigger}
    for i, token in enumerate(tokens):
        variants[f"single:{i}"] = encoder.text_from_tokens([token])
    for i in range(len(tokens)):
        for j in range(i + 1, len(tokens)):
            variants[f"pair:{i},{j}"] = encoder.text_from_tokens([tokens[i], tokens[j]])

    base_questions = ["Question: " + row["question"] + "\n" for row in evaluation]
    variant_names = list(variants)
    texts = []
    for name in variant_names:
        suffix = variants[name]
        texts.extend(q + (suffix + "\n" if suffix else "") for q in base_questions)
    all_query_vectors = encoder.encode(texts)
    chunks = np.split(all_query_vectors, len(variant_names))

    # The optimizer uses five benign cluster centers. KMeans recovers the same
    # centroid geometry without fitting unused full-covariance matrices.
    centers = numpy_kmeans(np.asarray(clean_vectors), n_clusters=5, seed=0)

    ks = sorted(set([1, 2, 5, 10, args.top_k]))
    metrics = {}
    for name, vectors in zip(variant_names, chunks):
        metrics[name] = {
            "text": variants[name],
            "l_uni_centroid": l_uni(vectors, centers),
            **retrieval_metrics(vectors, clean_vectors, poison_vectors, ks),
        }

    target = f"mean_margin_at_{args.top_k}"
    empty = metrics["empty"][target]
    singles = {
        i: metrics[f"single:{i}"][target] - empty for i in range(len(tokens))
    }
    interactions = []
    for i in range(len(tokens)):
        for j in range(i + 1, len(tokens)):
            pair_effect = metrics[f"pair:{i},{j}"][target] - empty
            interaction = pair_effect - singles[i] - singles[j]
            interactions.append({
                "i": i, "j": j, "token_i": tokens[i], "token_j": tokens[j],
                "S_i": singles[i], "S_j": singles[j], "S_ij": pair_effect,
                "interaction": interaction,
            })
    interactions.sort(key=lambda row: row["interaction"], reverse=True)

    l_values = np.array([metrics[n]["l_uni_centroid"] for n in variant_names])
    m_values = np.array([metrics[n][target] for n in variant_names])
    correlation = float(np.corrcoef(l_values, m_values)[0, 1])
    discordant = []
    for a in range(len(variant_names)):
        for b in range(len(variant_names)):
            if l_values[a] > l_values[b] and m_values[a] < m_values[b]:
                discordant.append({
                    "higher_l_uni": variant_names[a], "lower_l_uni": variant_names[b],
                    "l_uni_gain": float(l_values[a] - l_values[b]),
                    "topk_margin_change": float(m_values[a] - m_values[b]),
                })
    discordant.sort(key=lambda row: row["topk_margin_change"])

    output = args.output_dir or (
        ROOT / "results/trigger_validation" /
        datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "trigger_file": str(args.trigger_file.resolve()),
        "trigger_artifact": trigger_artifact,
        "warning": (
            "Input artifact is not a completed optimized trigger."
            if trigger_artifact.get("state") != "completed" else None
        ),
        "config": {
            "encoder": ENCODER, "num_clean_keys": len(clean_vectors),
            "num_poison_keys": len(poison_vectors), "num_queries": len(evaluation),
            "source_questions": str(args.source_questions.resolve()),
            "evaluation_questions": str(args.evaluation_questions.resolve()),
            "poison_source_qids": [row["qid"] for row in sources],
            "evaluation_qids": [row["qid"] for row in evaluation],
            "seed": args.seed, "top_k": args.top_k, "ks": ks,
            "score": target,
        },
        "full_vs_empty": {
            "empty": metrics["empty"], "full": metrics["full"],
            "delta_l_uni": metrics["full"]["l_uni_centroid"] - metrics["empty"]["l_uni_centroid"],
            "delta_topk_margin": metrics["full"][target] - metrics["empty"][target],
        },
        "single_effects": [
            {"i": i, "token": tokens[i], "S_i": singles[i]} for i in range(len(tokens))
        ],
        "pair_interactions": interactions,
        "objective_alignment": {
            "pearson_l_uni_vs_topk_margin": correlation,
            "discordant_ordered_pairs": len(discordant),
            "total_ordered_pairs": len(variant_names) * (len(variant_names) - 1),
            "strongest_counterexamples": discordant[:10],
        },
        "variants": metrics,
    }
    (output / "validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# Trigger interaction and top-K validation", "",
        result["warning"] or "Completed trigger artifact.", "",
        f"Queries: {len(evaluation)}; clean keys: {len(clean_vectors)}; poison keys: {len(poison_vectors)}; K={args.top_k}.",
        f"Full poison recall@{args.top_k}: {metrics['full'][f'poison_recall_at_{args.top_k}']:.1%}.",
        f"Full mean margin@{args.top_k}: {metrics['full'][target]:.6f}; empty: {metrics['empty'][target]:.6f}.",
        f"L_uni delta full-empty: {result['full_vs_empty']['delta_l_uni']:.6f}.",
        f"Pearson(L_uni, margin@{args.top_k}) across variants: {correlation:.4f}.",
        f"Discordant ordered variant pairs: {len(discordant)}/{len(variant_names) * (len(variant_names)-1)}.",
        "", "## Strongest positive token interactions", "",
        "| i | token i | j | token j | S_i | S_j | S_ij | interaction |",
        "|---:|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in interactions[:10]:
        lines.append(
            f"| {row['i']} | `{row['token_i']}` | {row['j']} | `{row['token_j']}` | "
            f"{row['S_i']:.6f} | {row['S_j']:.6f} | {row['S_ij']:.6f} | {row['interaction']:.6f} |"
        )
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)
    print("\n".join(lines[:9]))


if __name__ == "__main__":
    main()
