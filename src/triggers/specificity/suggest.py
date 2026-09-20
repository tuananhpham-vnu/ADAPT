"""Create a short topic-conditioned prefix for a new English query, offline."""
import argparse
import json
from pathlib import Path

import torch

from src.triggers.hierarchy.backends import Encoder
from src.triggers.hierarchy.fluency import FluencyScorer
from src.triggers.hierarchy.io import read, write
from .candidates import render, copy_metrics
from .phrases import NounPhraseExtractor
from .selection import Selector


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--query", required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--reference-dataset", type=Path, help="Optional dataset.json: train queries provide the specificity contrast")
    p.add_argument("--mode", choices=["grounded_language", "topic_heading", "generic_language"], default="grounded_language")
    p.add_argument("--candidate-cap", type=int, default=20)
    p.add_argument("--device", default="cpu")
    p.add_argument("--model-cache", default="outputs/model-cache")
    args = p.parse_args(argv)
    if not args.query.strip():
        p.error("Query must be nonempty")
    torch.set_num_threads(2)
    semantic = Encoder("sentence-transformers/all-MiniLM-L6-v2", device=args.device)
    fluency = FluencyScorer(cache_dir=args.model_cache, device=args.device)
    nouns = NounPhraseExtractor(args.model_cache, args.device)
    selector = Selector(semantic, fluency, candidate_cap=args.candidate_cap, phrase_extractor=nouns.phrases)
    references = read(args.reference_dataset)["train"] if args.reference_dataset else []
    selected = selector.fit([{"question": args.query}], args.mode, references)
    result = {"original": args.query, "trigger": selected["trigger"], "altered": render(args.query, selected["trigger"]),
              "constraint_feasible": selected["constraint_feasible"], "mode": args.mode,
              "selection": selected, "copy": copy_metrics(args.query, selected["trigger"]),
              "models": {"semantic": semantic.metadata, "fluency": fluency.metadata, "phrases": nouns.metadata}}
    if args.output:
        write(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k not in {"selection", "models"}}, indent=2))


if __name__ == "__main__":
    main()
