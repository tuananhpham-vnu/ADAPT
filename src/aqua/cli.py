"""Small CLI; each stage owns its inputs and independently resumable outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parser():
    root = argparse.ArgumentParser(description="AQuA / AuthShift research pilot")
    commands = root.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Generate matched JSONL quadruples")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--groups", type=int, default=500)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--split-by", choices=["group", "tool"], default="tool")
    collect = commands.add_parser("collect", help="Collect field activations; resumes complete shards")
    collect.add_argument("--dataset", type=Path, required=True)
    collect.add_argument("--output", type=Path, required=True)
    backend_options(collect)
    train = commands.add_parser("train", help="Train projection and calibrate on validation only")
    train.add_argument("--activations", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--rank", type=int, default=4)
    train.add_argument("--learning-rate", type=float, default=.003)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", default="cpu")
    train.add_argument("--max-false-allow", type=float, default=.05)
    train.add_argument("--equivalence-weight", type=float, default=1.)
    train.add_argument("--flip-weight", type=float, default=1.)
    train.add_argument("--authorization-weight", type=float, default=1.)
    train.add_argument("--shortcut-weight", type=float, default=.1)
    train.add_argument("--margin", type=float, default=1.)
    evaluate = commands.add_parser("evaluate", help="Replay held-out candidates and observe sandbox effects")
    evaluate.add_argument("--dataset", type=Path, required=True)
    evaluate.add_argument("--probe", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--split", choices=["validation", "test"], default="test")
    backend_options(evaluate)
    smoke = commands.add_parser("smoke", help="Offline end-to-end software check; NOT research evidence")
    smoke.add_argument("--output", type=Path, default=Path("outputs/aqua/smoke"))
    smoke.add_argument("--groups", type=int, default=40)
    smoke.add_argument("--epochs", type=int, default=30)
    smoke.add_argument("--seed", type=int, default=42)
    return root


def backend_options(command):
    command.add_argument("--backend", choices=["fixture", "huggingface"], required=True)
    command.add_argument("--model", help="HF model ID or local model directory")
    command.add_argument("--revision", help="Pin a model commit for reproducibility")
    command.add_argument("--device", default="cpu")
    command.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    command.add_argument("--layer", type=int, default=-1, help="Decoder block index; -1 is last")
    command.add_argument("--layer-path", default="model.layers", help="Module path for decoder block list")
    command.add_argument("--max-length", type=int, default=2048)
    command.add_argument("--seed", type=int, default=42)


def run(args):
    from .benchmark import build_cases, fingerprint, load_cases, save_cases
    if args.command == "prepare":
        cases = build_cases(args.groups, args.seed, args.split_by)
        if args.output.exists() and fingerprint(load_cases(args.output)) != fingerprint(cases):
            raise ValueError("Dataset already exists with different settings; use a new output path")
        save_cases(args.output, cases)
        return {"dataset": str(args.output), "groups": args.groups, "cases": len(cases),
                "fingerprint": fingerprint(cases)}

    from .backends import FixtureBackend, make_backend
    from .collection import collect
    from .evaluation import evaluate
    from .losses import LossConfig
    from .training import TrainConfig, train
    if args.command in {"collect", "evaluate"}:
        if args.backend == "huggingface" and not args.model:
            raise ValueError("--model is required for the huggingface backend")
        backend = make_backend(vars(args))
        if args.command == "collect":
            result = collect(args.dataset, args.output, backend)
            return {"activations": str(args.output), "groups": len(result["shards"]), "backend": backend.metadata}
        result = evaluate(args.dataset, args.probe, args.output, backend, args.split)
        return {"report": str(args.output / "metrics.json"), "evidence": result["evidence"],
                "metrics": result["metrics"], "representation": result["representation"]}
    if args.command == "train":
        config = TrainConfig(args.epochs, args.batch_size, args.rank, args.learning_rate,
                             args.seed, args.max_false_allow)
        losses = LossConfig(args.equivalence_weight, args.flip_weight, args.authorization_weight,
                            args.shortcut_weight, args.margin)
        artifact = train(args.activations, args.output, config, losses, args.device)
        return {"probe": str(args.output / "probe.pt"), "epochs": artifact["epochs"],
                "calibration": artifact["calibration"]}
    # The only command permitted to choose the fixture backend implicitly.
    output = args.output
    dataset = output / "authshift.jsonl"
    cases = build_cases(args.groups, args.seed, "tool")
    if dataset.exists() and fingerprint(load_cases(dataset)) != fingerprint(cases):
        raise ValueError("Smoke dataset changed; use a new output directory")
    save_cases(dataset, cases)
    backend = FixtureBackend(seed=args.seed)
    collect(dataset, output / "activations", backend)
    train(output / "activations", output / "training", TrainConfig(epochs=args.epochs, seed=args.seed))
    evaluation_output = output / f"evaluation-epoch-{args.epochs}"
    result = evaluate(dataset, output / "training/probe.pt", evaluation_output, backend)
    return {"report": str(evaluation_output / "metrics.json"), "evidence": result["evidence"],
            "metrics": result["metrics"], "representation": result["representation"]}


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    try:
        result = run(args)
    except (ValueError, FileNotFoundError) as exc:
        cli.error(str(exc))
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
