"""Run the same comparison on fixtures or real StrategyQA/DPR embeddings."""
import argparse
from pathlib import Path

import torch

from algo.clustering import fit_centers
from .backends import Encoder
from .data import fixture_data, load_data
from .experiment import ARMS, compare
from .io import digest, read, write
from .quality import MeaningGuard
from .report import render
from .search import Objective


def parser():
    p = argparse.ArgumentParser(description="Trigger hierarchy: individual -> grouped -> universal, with fixed-budget controls")
    p.add_argument("command", choices=["smoke", "compare"])
    p.add_argument("--output", type=Path, default=Path("outputs/trigger_hierarchy/smoke"))
    p.add_argument("--backend", choices=["fixture", "hf"], default="fixture")
    p.add_argument("--positions", nargs="+", choices=["suffix", "prefix", "infix", "sentence"], default=["suffix", "prefix", "infix"])
    p.add_argument("--candidate-style", choices=["template", "free"], default="template")
    p.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    p.add_argument("--groups", type=int, default=2)
    p.add_argument("--budget", type=int, default=1024, help="Logical retriever text requests per arm, including hierarchy construction")
    p.add_argument("--train-size", type=int, default=8)
    p.add_argument("--validation-size", type=int, default=8)
    p.add_argument("--test-size", type=int, default=8)
    p.add_argument("--poison-count", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--margin", type=float, default=.1)
    p.add_argument("--max-words", type=int, default=3)
    p.add_argument("--max-trigger-tokens", type=int, default=12)
    p.add_argument("--meaning-threshold", type=float, default=.85)
    p.add_argument("--reference-clusters", type=int, default=5)
    p.add_argument("--model", default="facebook/dpr-ctx_encoder-single-nq-base")
    p.add_argument("--semantic-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--allow-downloads", action="store_true")
    p.add_argument("--train", type=Path, default=Path("ReAct/database/strategyqa_train_filtered.json"))
    p.add_argument("--test", type=Path, default=Path("ReAct/database/strategyqa_dev.json"))
    p.add_argument("--corpus", type=Path, default=Path("ReAct/database/strategyqa_train_paragraphs.json"))
    p.add_argument("--corpus-embeddings", type=Path, help="Reuse StrategyQA vectors.npy with its matching manifest.json")
    return p


def run(args):
    if min(args.train_size, args.validation_size, args.test_size, args.poison_count,
           args.groups, args.budget, args.top_k, args.batch_size, args.max_trigger_tokens) < 1:
        raise ValueError("All sizes and budgets must be positive")
    if args.max_words != 3:
        raise ValueError("This pilot uses three-word seeds and fixed three-word crossover; set --max-words 3")
    if args.groups > args.train_size or args.poison_count < args.groups:
        raise ValueError("Need at least one training query and poison source per group")
    if any(a in args.arms for a in ("per_query", "hierarchical_merge", "hierarchical_mix")) and args.poison_count < args.train_size:
        raise ValueError("Per-query leaves require poison-count >= train-size; total poison count stays shared across all arms")
    if any(a.startswith("hierarchical") for a in args.arms) and args.train_size < 2:
        raise ValueError("Hierarchy needs at least two training queries")
    if args.command == "smoke":
        args.backend = "fixture"
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    options = {"device": args.device, "batch_size": args.batch_size, "max_length": args.max_length,
               "local_files_only": not args.allow_downloads}
    encoder = Encoder(args.model if args.backend == "hf" else None, pooling="dpr", fixture_seed=args.seed, **options)
    semantic = Encoder(args.semantic_model if args.backend == "hf" else None, fixture_seed=args.seed + 1000, **options)
    data = fixture_data(args.train_size, args.validation_size, args.test_size, args.poison_count, args.seed) if args.backend == "fixture" else load_data(
        args.train, args.test, args.corpus, args.train_size, args.validation_size, args.test_size, args.poison_count, args.seed)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k not in {"command", "output"}}
    contract = {"version": 2, "config": config, "dataset": digest(data),
                "retriever": encoder.metadata, "semantic": semantic.metadata}
    if (args.output / "config.json").exists() and read(args.output / "config.json") != contract:
        raise ValueError("Resume configuration changed; choose a new output directory")
    if args.corpus_embeddings and args.backend == "hf":
        import hashlib
        import numpy as np
        manifest = read(args.corpus_embeddings.parent / "manifest.json")
        expected_hash = hashlib.sha256(args.corpus.read_bytes()).hexdigest()
        if manifest.get("encoder") != args.model or manifest.get("max_length") != args.max_length or manifest.get("corpus_sha256") != expected_hash or manifest.get("normalization") != "l2":
            raise ValueError("Corpus cache does not match model, corpus, normalization or max_length")
        clean = torch.from_numpy(np.array(np.load(args.corpus_embeddings, mmap_mode="r", allow_pickle=False))).float()
        if len(clean) != len(data["corpus"]) or not torch.isfinite(clean).all() or not torch.allclose(clean.norm(dim=1), torch.ones(len(clean)), atol=1e-3):
            raise ValueError("Invalid corpus embedding cache")
    else:
        clean = encoder.encode(data["corpus"])
    if not 1 <= args.top_k <= len(clean):
        raise ValueError("top-k exceeds corpus size")
    centers = fit_centers(clean, args.reference_clusters, seed=args.seed)
    shared_queries = [r["question"] for split in ("train", "validation", "test") for r in data[split]]
    encoder.encode(shared_queries)
    semantic.encode(shared_queries)
    guard = MeaningGuard(semantic, args.meaning_threshold)
    def factory(position):
        return Objective(encoder, guard, clean, centers, position=position, top_k=args.top_k,
                         margin=args.margin, max_words=args.max_words, max_tokens=args.max_trigger_tokens,
                         candidate_style=args.candidate_style)
    write(args.output / "dataset.json", {k: v for k, v in data.items() if k != "corpus"})
    write(args.output / "reference_clusters.json", {"method": "kmeans", "count": len(centers), "centers": centers.tolist()})
    result = compare(data, factory, args.output, positions=args.positions, arms=args.arms,
                     groups=args.groups, budget=args.budget, seed=args.seed, contract=contract,
                     progress=lambda message: print(message, flush=True))
    (args.output / "REPORT.md").write_text(render(result), encoding="utf-8")
    return result


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        result = run(args)
    except (ValueError, FileNotFoundError, OSError) as exc:
        p.error(str(exc))
    print((args.output / "REPORT.md").read_text(encoding="utf-8"))
