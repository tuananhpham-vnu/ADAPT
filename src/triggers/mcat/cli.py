"""Stage-oriented CLI for MCAT.

Stages mirror ``src/triggers/margin.py`` so a long Kaggle job can be cut at
any boundary and resumed:

    prepare-episodes -> index -> train -> evaluate -> report

``smoke`` runs all of them end to end against the fixture retriever on CPU, so
the artifact and resume contract can be verified without a download or a GPU.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

from src.triggers.artifacts import (
    ROOT, atomic_json, git_metadata, read_json, stable_hash, versions,
)
from src.triggers.mcat.domains import DOMAIN_NAMES
from src.triggers.mcat.episodes import Episode, EpisodeSizes, build_manifest
from src.triggers.mcat.evaluate import evaluate
from src.triggers.mcat.generator import parameter_count
from src.triggers.mcat.retrievers import build_fixture_retriever, load_dpr, DEFAULT_RETRIEVER
from src.triggers.mcat.runtime import Workspace
from src.triggers.mcat.train import TrainConfig, build_contract, train, _logits_source

DEFAULT_OUTPUT = ROOT / "outputs/mcat"


def _train_config(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        mode=args.mode, variant=args.variant, steps=args.steps,
        learning_rate=args.learning_rate, tau_start=args.tau_start, tau_end=args.tau_end,
        lambda_cpt=args.lambda_cpt, lambda_ret=args.lambda_ret, margin=args.margin,
        poison_mode=args.poison_mode, grad_clip=args.grad_clip,
        context_dim=args.context_dim, hidden=args.hidden,
        attention_pool=args.attention_pool, memory_summary_keys=args.memory_summary_keys,
        seed=args.seed,
    )


def _retriever(args: argparse.Namespace):
    if args.fixture:
        return build_fixture_retriever(max_length=args.max_length, seed=args.seed)
    return load_dpr(args.retriever_model, revision=args.retriever_revision,
                    device=args.retriever_device, max_length=args.max_length,
                    token=args.hf_token)


def _workspace(args: argparse.Namespace) -> Workspace:
    # Clean vectors depend only on (corpus, retriever), never on the training
    # arm, so a sweep can point every arm at one --cache-dir and encode once.
    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(args.output_dir) / "cache"
    return Workspace(
        retriever=_retriever(args), cache_dir=cache_dir,
        fixture=args.fixture, batch_size=args.index_batch_size,
        memory_summary_keys=args.memory_summary_keys, corpus_limit=args.corpus_limit,
    )


def _sizes(args: argparse.Namespace) -> EpisodeSizes:
    return EpisodeSizes(
        documents=args.documents, support=args.support, optimization=args.optimization,
        evaluation=args.evaluation, poison_sources=args.poison_count,
        trigger_tokens=args.trigger_tokens, retrieval_top_k=args.top_k,
    )


def _load_episodes(output_dir: Path) -> tuple[list[Episode], dict[str, Any]]:
    manifest_path = Path(output_dir) / "manifest.json"
    episodes_path = Path(output_dir) / "episodes.jsonl"
    if not manifest_path.exists() or not episodes_path.exists():
        raise FileNotFoundError(
            "stage prepare-episodes is required: manifest.json/episodes.jsonl are missing"
        )
    manifest = read_json(manifest_path)
    episodes = [
        Episode(**json.loads(line))
        for line in episodes_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    return episodes, manifest


def prepare_episodes(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes, manifest = build_manifest(
        args.domain, seed=args.seed, sizes=_sizes(args), per_split=args.per_split,
        ratios=tuple(args.ratios), corpus_limit=args.corpus_limit,
    )
    manifest |= {"environment": versions(), "git": git_metadata(),
                 "fixture": bool(args.fixture)}

    existing = output_dir / "manifest.json"
    if existing.exists() and read_json(existing).get("split_hash") != manifest["split_hash"]:
        raise ValueError("resume refused: the episode split changed")

    (output_dir / "episodes.jsonl").write_text(
        "".join(json.dumps(episode.to_json(), ensure_ascii=False) + "\n"
                for episode in episodes),
        encoding="utf-8",
    )
    atomic_json(existing, manifest)
    atomic_json(output_dir / "stages/prepare-episodes.json", {
        "state": "completed", "episodes": len(episodes),
        "outputs": ["manifest.json", "episodes.jsonl"],
    })
    print(f"prepared {len(episodes)} episodes -> {output_dir}")
    return output_dir


def index(args: argparse.Namespace) -> Path:
    episodes, _ = _load_episodes(args.output_dir)
    workspace = _workspace(args)
    covered = sorted({episode.domain for episode in episodes})
    rows = {}
    for name in covered:
        rows[name] = {
            "documents": int(workspace.document_vectors(name).shape[0]),
            "queries": int(workspace.query_vectors(name).shape[0]),
        }
    atomic_json(Path(args.output_dir) / "stages/index.json", {
        "state": "completed", "domains": rows,
        "retriever": workspace.retriever.fingerprint(),
        "cache_dir": str(workspace.cache_dir),
        "prebuilt_index": workspace.reuse_report,
    })
    print(f"indexed {rows}")
    for name, reason in workspace.reuse_report.items():
        verdict = "reused" if reason.get("reused") else f"re-encoded ({reason.get('reason')})"
        print(f"  {name}: prebuilt index {verdict}")
    return Path(args.output_dir) / "cache"


def train_stage(args: argparse.Namespace) -> Path:
    episodes, manifest = _load_episodes(args.output_dir)
    config = _train_config(args)
    workspace = _workspace(args)
    train_episodes = [episode for episode in episodes if episode.split == "train"]
    validation = [episode for episode in episodes if episode.split == "validation"]
    contract = build_contract(config, manifest, workspace.retriever)

    summary, _ = train(
        workspace, train_episodes, config, output_dir=Path(args.output_dir),
        contract=contract, resume=args.resume, validation=validation,
    )
    atomic_json(Path(args.output_dir) / "stages/train.json",
                {"state": "completed"} | summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return Path(args.output_dir) / "training.json"


def evaluate_stage(args: argparse.Namespace) -> Path:
    import torch

    episodes, manifest = _load_episodes(args.output_dir)
    config = _train_config(args)
    workspace = _workspace(args)
    checkpoint_path = Path(args.output_dir) / "checkpoint.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError("stage train is required: checkpoint.pt is missing")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected = build_contract(config, manifest, workspace.retriever)
    if checkpoint["contract"] != expected:
        raise ValueError("evaluation refused: the checkpoint was trained under another contract")

    target = [episode for episode in episodes if episode.split == args.split]
    if not target:
        raise ValueError(f"no episodes in split {args.split!r}")

    if config.mode == "direct-logit":
        # B2 has nothing to transfer: its logits belong to the episodes it was
        # fitted on.  The fair protocol is to let it optimize on the evaluation
        # episodes too, paying that online cost, and report it under
        # adapt-<split>/ so the extra compute is never lost from the comparison.
        adapt_dir = Path(args.output_dir) / f"adapt-{args.split}"
        _, modules = train(
            workspace, target, config, output_dir=adapt_dir,
            contract=dict(expected, adapted_split=args.split), resume=args.resume,
        )
    else:
        shared = _logits_source(config, workspace.retriever, target[0])
        shared.load_state_dict(checkpoint["modules"][0])
        shared.eval()
        modules = [shared] * len(target)

    summary = evaluate(workspace, modules, config, target,
                       output_dir=Path(args.output_dir), score=args.score,
                       controls=not args.no_controls)
    atomic_json(Path(args.output_dir) / "stages/evaluate.json",
                {"state": "completed", "split": args.split} | summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return Path(args.output_dir) / "evaluation.json"


def report(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir)
    manifest = read_json(output_dir / "manifest.json")
    training = read_json(output_dir / "training.json")
    evaluation = read_json(output_dir / "evaluation.json")
    lines = [
        "# MCAT run report", "",
        f"- split hash: `{manifest['split_hash'][:16]}`",
        f"- episodes: {len(manifest['episode_ids'])}",
        f"- mode: `{training['mode']}` variant `{training['variant']}`",
        f"- training steps: {training['steps']}",
        f"- round-trip valid triggers: {training['valid_triggers']}/{training['triggers']}",
        "", "## Retrieval on the held-out split", "",
        "| metric | trigger on | trigger off |", "|---|---|---|",
    ]
    for key in evaluation["trigger_on"]:
        lines.append(
            f"| {key} | {evaluation['trigger_on'][key]:.4f} "
            f"| {evaluation['trigger_off'][key]:.4f} |"
        )
    lines += ["", f"False activation: {evaluation['false_activation']:.4f}", ""]
    controls = evaluation.get("controls", {})
    if controls.get("shuffled_context", {}).get("applicable"):
        shuffled = controls["shuffled_context"]
        lines += [
            "## Mechanism controls", "",
            f"- shuffled memory context (within domain) changed the trigger in "
            f"{shuffled['changed']}/{shuffled['episodes']} episodes "
            f"({shuffled['change_rate']:.2f})",
            "  A low rate means the memory branch is inert and the conditioning "
            "claim is not supported.",
        ]
        for name, entry in sorted(shuffled.get("per_domain", {}).items()):
            lines.append(
                f"  - {name}: {entry['changed']}/{entry['episodes']} "
                f"({entry['change_rate']:.2f})"
            )
        if shuffled.get("skipped_domains"):
            lines.append(
                f"  - skipped (fewer than 2 episodes to swap within): "
                f"{', '.join(shuffled['skipped_domains'])}"
            )
        permutation = controls.get("permutation", {})
        if permutation.get("applicable"):
            lines.append(
                f"- document-order permutation left the trigger unchanged in "
                f"{permutation['stable']}/{permutation['episodes']} episodes "
                f"({permutation['stable_rate']:.2f}); pooling should make this 1.00"
            )
    path = output_dir / "REPORT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    return path


def smoke(args: argparse.Namespace) -> Path:
    """Run every stage on the fixture retriever, then verify the resume guard."""
    args.fixture = True
    prepare_episodes(args)
    index(args)
    train_stage(args)
    evaluate_stage(args)
    report(args)

    episodes, manifest = _load_episodes(args.output_dir)
    workspace = _workspace(args)
    contract = build_contract(_train_config(args), manifest, workspace.retriever)
    tampered = dict(contract, config_hash=stable_hash("different"))
    try:
        train(workspace, [e for e in episodes if e.split == "train"], _train_config(args),
              output_dir=Path(args.output_dir), contract=tampered, resume=True)  # noqa: F841
    except ValueError:
        print("resume guard: a changed contract is refused")
    else:
        raise AssertionError("resume guard did not refuse a changed contract")

    module = _logits_source(_train_config(args), workspace.retriever, episodes[0])
    print(f"smoke completed; module parameters: {parameter_count(module)}")
    return Path(args.output_dir)


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT / "run")
    parser.add_argument("--domain", action="append", choices=DOMAIN_NAMES, default=None,
                        help="repeatable; defaults to qa")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fixture", action="store_true",
                        help="use the CPU fixture retriever instead of DPR")

    episodes = parser.add_argument_group("episodes")
    episodes.add_argument("--per-split", type=int, default=4)
    episodes.add_argument("--ratios", type=float, nargs=3, default=[0.6, 0.2, 0.2])
    episodes.add_argument("--documents", type=int, default=512)
    episodes.add_argument("--support", type=int, default=32)
    episodes.add_argument("--optimization", type=int, default=64)
    episodes.add_argument("--evaluation", type=int, default=128)
    episodes.add_argument("--poison-count", type=int, default=5)
    episodes.add_argument("--trigger-tokens", type=int, default=10)
    episodes.add_argument("--top-k", type=int, default=5)
    episodes.add_argument("--corpus-limit", type=int, default=None,
                          help="truncate each domain to its first N rows; recorded "
                               "in the manifest because it changes the experiment")

    retriever = parser.add_argument_group("retriever")
    retriever.add_argument("--retriever-model", default=DEFAULT_RETRIEVER)
    retriever.add_argument("--retriever-revision", default=None)
    retriever.add_argument("--retriever-device", default="cuda:0")
    retriever.add_argument("--max-length", type=int, default=512)
    retriever.add_argument("--index-batch-size", type=int, default=32)
    retriever.add_argument("--hf-token", default=None)
    retriever.add_argument("--cache-dir", type=Path, default=None,
                           help="shared clean-vector cache; defaults to "
                                "<output-dir>/cache. Point several arms at one "
                                "directory to encode the corpus only once")

    training = parser.add_argument_group("training")
    training.add_argument("--mode", choices=("direct-logit", "universal-logit", "generator"),
                          default="generator")
    training.add_argument("--variant", default="memory+query",
                          choices=("memory+query", "query", "memory", "none"))
    training.add_argument("--steps", type=int, default=200)
    training.add_argument("--learning-rate", type=float, default=1e-3)
    training.add_argument("--tau-start", type=float, default=2.0)
    training.add_argument("--tau-end", type=float, default=0.5)
    training.add_argument("--lambda-cpt", type=float, default=0.1)
    training.add_argument("--lambda-ret", type=float, default=0.0)
    training.add_argument("--margin", type=float, default=0.1)
    training.add_argument("--poison-mode", choices=("refresh", "fixed"), default="refresh")
    training.add_argument("--grad-clip", type=float, default=5.0)
    training.add_argument("--context-dim", type=int, default=128)
    training.add_argument("--hidden", type=int, default=256)
    training.add_argument("--attention-pool", action="store_true")
    training.add_argument("--memory-summary-keys", type=int, default=128)

    evaluation = parser.add_argument_group("evaluation")
    evaluation.add_argument("--split", choices=("train", "validation", "test"),
                            default="test")
    evaluation.add_argument("--score", choices=("dot", "cosine"), default="dot")
    evaluation.add_argument("--no-controls", action="store_true")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="python -m src.triggers.mcat", description=__doc__)
    subparsers = root.add_subparsers(dest="command", required=True)
    for name, handler in (
        ("prepare-episodes", prepare_episodes), ("index", index), ("train", train_stage),
        ("evaluate", evaluate_stage), ("report", report), ("smoke", smoke),
    ):
        stage = subparsers.add_parser(name, help=handler.__doc__)
        add_common(stage)
        stage.set_defaults(func=handler)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not args.domain:
        args.domain = ["qa"]
    if args.command == "smoke":
        args.output_dir = Path(args.output_dir)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
