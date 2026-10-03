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
from src.triggers.mcat.adapt import ADAPT_METHODS, AdaptConfig
from src.triggers.mcat.costs import CostLedger
from src.triggers.mcat.drift import (
    ABLATION_KINDS, DEFAULT_GROWTH, Snapshot, assert_no_snapshot_leak, build_trajectory,
    support_similarity, trajectory_hash,
)
from src.triggers.mcat.drift_eval import (
    DRIFT_ROWS, DRIFT_SUMMARY, aggregate, load_rows, prepare_bases, run_trajectory,
)
from src.triggers.mcat.domains import DOMAIN_NAMES
from src.triggers.mcat.episodes import (
    Episode, EpisodeSizes, build_manifest, family_key,
)
from src.triggers.mcat.evaluate import evaluate
from src.triggers.mcat.generator import parameter_count
from src.triggers.mcat.encoding import POSITIONS
from src.triggers.mcat.poison import FROZEN_POISON_FILE, FrozenPoison
from src.triggers.mcat.probes import (
    GROWTH_ROWS, GROWTH_SUMMARY, POSITION_ROWS, POSITION_SUMMARY, PROBE_GROWTH,
    PROBE_SEEDS, R1_SUMMARY, GrowthProbeConfig, PositionProbeConfig, attention_mass,
    compare_runs, growth_probe, position_probe, summarize_growth, summarize_position,
)
from src.triggers.mcat.retrievers import build_fixture_retriever, load_dpr, DEFAULT_RETRIEVER
from src.triggers.mcat.runtime import Workspace
from src.triggers.mcat.train import (
    MODES, PER_EPISODE_MODES, TrainConfig, build_contract, train, _logits_source,
)

DEFAULT_OUTPUT = ROOT / "outputs/mcat"


def _train_config(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        mode=args.mode, variant=args.variant, steps=args.steps,
        learning_rate=args.learning_rate, tau_start=args.tau_start, tau_end=args.tau_end,
        lambda_cpt=args.lambda_cpt, lambda_ret=args.lambda_ret, margin=args.margin,
        poison_mode=args.poison_mode, grad_clip=args.grad_clip,
        context_dim=args.context_dim, hidden=args.hidden,
        attention_pool=args.attention_pool, memory_summary_keys=args.memory_summary_keys,
        seed=args.seed, trigger_position=args.trigger_position,
        hotflip_candidates=args.hotflip_candidates,
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

    # Priced because M3's break-even point needs C_train; without this the
    # amortization question cannot be answered at all.
    ledger = CostLedger(device=str(workspace.retriever.device))
    with ledger.phase("train"):
        summary, _ = train(
            workspace, train_episodes, config, output_dir=Path(args.output_dir),
            contract=contract, resume=args.resume, validation=validation,
        )
    atomic_json(Path(args.output_dir) / "costs-train.json", ledger.to_json())
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

    if config.mode in PER_EPISODE_MODES:
        # B1/B2 have nothing to transfer: their triggers belong to the episodes
        # they were fitted on.  The fair protocol is to let it optimize on the evaluation
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


def _adapt_config(args: argparse.Namespace) -> AdaptConfig:
    methods = tuple(args.method) if args.method else ADAPT_METHODS
    return AdaptConfig(methods=methods, steps=args.adapt_steps,
                       write_policy=args.write_policy)


def _drift_dir(args: argparse.Namespace) -> Path:
    """Where one drift configuration's artifacts live.

    Keyed by write policy, few-step budget and split, so sweeping
    ``--adapt-steps`` on validation produces parallel directories instead of
    appending rows from two budgets into one file.  Locking the budget means
    comparing those directories, which only works if they exist separately.
    """
    adapt_config = _adapt_config(args)
    name = f"{adapt_config.write_policy}-steps{adapt_config.steps}-{args.split}"
    return Path(args.output_dir) / "drift" / name


def _drift_contract(
    args: argparse.Namespace, manifest: dict[str, Any], workspace: Workspace
) -> dict[str, Any]:
    """What a drift resume must match: the run contract plus drift and adaptation.

    Kept separate from ``build_contract`` on purpose: adding the trajectory to
    the training contract would invalidate every checkpoint produced before the
    drift stage existed, for a change that does not affect training at all.
    """
    if "trajectory_hash" not in manifest:
        raise FileNotFoundError(
            "stage prepare-drift is required: the manifest has no trajectory_hash"
        )
    return build_contract(_train_config(args), manifest, workspace.retriever) | {
        "trajectory_hash": manifest["trajectory_hash"],
        "adapt_config_hash": _adapt_config(args).fingerprint(),
        "split": args.split,
    }


def _load_snapshots(output_dir: Path) -> list[Snapshot]:
    path = Path(output_dir) / "trajectories.jsonl"
    if not path.exists():
        raise FileNotFoundError("stage prepare-drift is required: trajectories.jsonl")
    return [Snapshot(**json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def prepare_drift(args: argparse.Namespace) -> Path:
    """Build one snapshot trajectory per episode and pin it into the manifest."""
    output_dir = Path(args.output_dir)
    episodes, manifest = _load_episodes(output_dir)
    workspace = _workspace(args)
    ratios = tuple(manifest["ratios"])
    ablations = tuple(args.ablation) if args.ablation else ABLATION_KINDS

    snapshots: list[Snapshot] = []
    reports: dict[str, Any] = {}
    for episode in episodes:
        assignment = workspace.assignment(episode.domain, ratios=ratios,
                                          seed=manifest["seed"])
        held = set(episode.doc_ids)
        documents = workspace.documents(episode.domain)
        pool = [row for row in documents
                if row["doc_id"] not in held
                and assignment.get(family_key(row["family"])) == episode.split]
        mixture_pool = [
            row for name in sorted(set(e.domain for e in episodes) - {episode.domain})
            for row in workspace.documents(name)
            if workspace.assignment(name, ratios=ratios, seed=manifest["seed"]).get(
                family_key(row["family"])) == episode.split
        ]
        # Similarities against Q_sup only; Q_eval stays locked (see drift.py).
        context = workspace.context_for(episode)
        scores = support_similarity(
            workspace.document_vectors(episode.domain),
            [row["doc_id"] for row in documents],
            context.support_vectors,
            [row["doc_id"] for row in pool],
        )
        built, report = build_trajectory(
            episode, pool, assignment=assignment, mixture_pool=mixture_pool,
            distractor_scores=scores, growth=tuple(args.growth), ablations=ablations,
            seed=manifest["seed"],
        )
        snapshots.extend(built)
        reports[episode.episode_id] = report

    for domain in sorted({episode.domain for episode in episodes}):
        family_of = {row["doc_id"]: row["family"]
                     for row in workspace.documents(domain)}
        assert_no_snapshot_leak(
            [s for s in snapshots if s.domain == domain], family_of,
            workspace.assignment(domain, ratios=ratios, seed=manifest["seed"]),
        )

    digest = trajectory_hash(snapshots)
    path = output_dir / "trajectories.jsonl"
    if path.exists() and manifest.get("trajectory_hash") not in (None, digest):
        raise ValueError("resume refused: the drift trajectory changed")
    path.write_text(
        "".join(json.dumps(snapshot.to_json(), ensure_ascii=False) + "\n"
                for snapshot in snapshots),
        encoding="utf-8",
    )
    manifest |= {"trajectory_hash": digest, "growth": list(args.growth),
                 "ablations": list(ablations), "trajectory_report": reports}
    atomic_json(output_dir / "manifest.json", manifest)
    atomic_json(output_dir / "stages/prepare-drift.json", {
        "state": "completed", "snapshots": len(snapshots),
        "trajectory_hash": digest, "report": reports,
    })
    print(f"prepared {len(snapshots)} snapshots for {len(episodes)} episodes")
    return path


def adapt_stage(args: argparse.Namespace) -> Path:
    """Adapt every method on every snapshot and score each one as it is produced."""
    import torch

    output_dir = Path(args.output_dir)
    episodes, manifest = _load_episodes(output_dir)
    snapshots = _load_snapshots(output_dir)
    config, adapt_config = _train_config(args), _adapt_config(args)
    workspace = _workspace(args)
    contract = _drift_contract(args, manifest, workspace)

    checkpoint = None
    checkpoint_path = output_dir / "checkpoint.pt"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        expected = build_contract(config, manifest, workspace.retriever)
        if checkpoint["contract"] != expected:
            raise ValueError(
                "adaptation refused: the checkpoint was trained under another contract")
    elif config.mode not in PER_EPISODE_MODES:
        raise FileNotFoundError("stage train is required: checkpoint.pt is missing")

    target = [episode for episode in episodes if episode.split == args.split]
    if not target:
        raise ValueError(f"no episodes in split {args.split!r}")
    wanted = {episode.episode_id for episode in target}
    trajectory = [snapshot for snapshot in snapshots if snapshot.episode_id in wanted]

    drift_dir = _drift_dir(args)
    drift_dir.mkdir(parents=True, exist_ok=True)
    config_path = drift_dir / "drift_config.json"
    if config_path.exists() and read_json(config_path) != contract:
        raise ValueError(
            "adaptation refused: this directory holds rows from another drift "
            "configuration; use a new --adapt-steps/--write-policy or a new output dir"
        )
    atomic_json(config_path, contract)

    ledger = CostLedger(device=str(workspace.retriever.device))
    with ledger.active():
        bases, frozen = prepare_bases(
            workspace, target, config, checkpoint=checkpoint, output_dir=drift_dir,
            contract=contract, ledger=ledger, resume=args.resume,
            freeze=adapt_config.write_policy == "fixed",
        )
        if frozen is None and adapt_config.write_policy == "fixed":
            frozen = FrozenPoison.load(drift_dir / FROZEN_POISON_FILE,
                                       retriever=workspace.retriever)
        rows = run_trajectory(
            workspace, episodes=target, snapshots=trajectory, config=config,
            adapt_config=adapt_config, checkpoint=checkpoint, output_dir=drift_dir,
            contract=contract, frozen_poison=frozen, score=args.score,
            resume=args.resume, ledger=ledger, bases=bases,
        )
    atomic_json(drift_dir / "costs.json", ledger.to_json())
    atomic_json(output_dir / "stages/adapt.json", {
        "state": "completed", "split": args.split, "rows": len(rows),
        "directory": str(drift_dir),
        "methods": list(adapt_config.methods), "write_policy": adapt_config.write_policy,
        "adapt_steps": adapt_config.steps,
    })
    print(f"adapted {len(rows)} (snapshot, method) rows -> {drift_dir / DRIFT_ROWS}")
    return drift_dir / DRIFT_ROWS


def evaluate_drift_stage(args: argparse.Namespace) -> Path:
    """Aggregate the drift rows: per-method means, paired intervals, amortization."""
    output_dir = Path(args.output_dir)
    drift_dir = _drift_dir(args)
    rows = load_rows(drift_dir / DRIFT_ROWS)
    if not rows:
        raise FileNotFoundError(
            f"stage adapt is required: no rows under {drift_dir}"
        )
    train_cost = None
    train_costs = output_dir / "costs-train.json"
    if train_costs.exists():
        train_cost = read_json(train_costs).get("phases", [{}])[0].get("wall_seconds")
    summary = aggregate(
        rows, baseline=args.baseline, train_cost=train_cost,
        quality_tolerance=args.quality_tolerance, iterations=args.bootstrap_iterations,
        seed=args.seed,
    )
    summary["train_cost_seconds"] = train_cost
    summary["directory"] = str(drift_dir)
    atomic_json(drift_dir / DRIFT_SUMMARY, summary)
    atomic_json(output_dir / "stages/evaluate-drift.json",
                {"state": "completed", "rows": len(rows), "baseline": args.baseline,
                 "directory": str(drift_dir)})
    print(json.dumps(summary["methods"], ensure_ascii=False, indent=2))
    return drift_dir / DRIFT_SUMMARY


def _as_written(value: Any) -> Any:
    """``value`` as it will read back from JSON.

    A probe contract holds tuples (growth levels, drift seeds, positions), and
    JSON has only arrays -- so comparing a fresh contract against the stored one
    compared a tuple with a list and refused every resume.  Round-tripping both
    sides is the comparison that means what it says.
    """
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _probe_contract(
    args: argparse.Namespace, manifest: dict[str, Any], workspace: Workspace,
    *, probe: str, config: Any,
) -> dict[str, Any]:
    """What a probe resume must match: the run contract plus the probe's own setup."""
    return _as_written(
        build_contract(_train_config(args), manifest, workspace.retriever) | {
            "split": args.split, "probe": probe, "probe_config": asdict(config),
        }
    )


def _lock_probe_config(path: Path, contract: dict[str, Any]) -> None:
    """Pin a probe directory to one configuration, or refuse to add to it."""
    if path.exists() and _as_written(read_json(path)) != contract:
        raise ValueError(
            f"probe refused: {path.parent} holds rows from another probe "
            "configuration; use a new --output-dir, or change --split / "
            "--trigger-position so this arm gets its own directory"
        )
    atomic_json(path, contract)


def _probe_dir(args: argparse.Namespace, name: str) -> Path:
    """One probe's own directory, keyed by the split and the trigger position.

    Two positions are two experiments, so their rows must not land in one file:
    a mean over a mixed file would average placements the paper reports apart.
    """
    return (Path(args.output_dir) / "probes"
            / f"{name}-{args.trigger_position}-{args.split}")


def _probe_checkpoint(args: argparse.Namespace, manifest, workspace):
    """The run's checkpoint, or ``None`` for the modes that do not need one."""
    import torch

    path = Path(args.output_dir) / "checkpoint.pt"
    config = _train_config(args)
    if not path.exists():
        if config.mode not in PER_EPISODE_MODES:
            raise FileNotFoundError(
                "stage train is required: checkpoint.pt is missing. Only --mode "
                f"{'/'.join(PER_EPISODE_MODES)} can probe without one, because they "
                "search per episode"
            )
        return None
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    expected = build_contract(config, manifest, workspace.retriever)
    if checkpoint["contract"] != expected:
        raise ValueError(
            "probe refused: the checkpoint was trained under another contract "
            "(config, split or encoder changed). Re-run train, or point --output-dir "
            "at the run that produced it"
        )
    return checkpoint


def probe_growth(args: argparse.Namespace) -> Path:
    """P0-R2: hold the trigger and poison fixed, grow the memory, measure decay."""
    output_dir = Path(args.output_dir)
    episodes, manifest = _load_episodes(output_dir)
    workspace = _workspace(args)
    config = _train_config(args)
    checkpoint = _probe_checkpoint(args, manifest, workspace)
    probe = GrowthProbeConfig(
        growth=tuple(args.growth), seeds=tuple(args.drift_seed),
        min_base_hit=args.min_base_hit, score=args.score,
    )
    target = [episode for episode in episodes if episode.split == args.split]
    if not target:
        raise ValueError(f"no episodes in split {args.split!r}")

    directory = _probe_dir(args, "growth")
    directory.mkdir(parents=True, exist_ok=True)
    contract = _probe_contract(args, manifest, workspace, probe="growth", config=probe)
    _lock_probe_config(directory / "probe_config.json", contract)

    ledger = CostLedger(device=str(workspace.retriever.device))
    with ledger.active():
        growth_probe(
            workspace, target, config, probe, checkpoint=checkpoint,
            output_dir=directory, contract=contract,
            ratios=tuple(manifest["ratios"]), split_seed=manifest["seed"],
            ledger=ledger, resume=args.resume,
        )
    # Summarized from the file, not from the rows this process produced, so a
    # resumed run reports the whole probe rather than the part it recomputed.
    summary = summarize_growth(load_rows(directory / GROWTH_ROWS),
                               iterations=args.bootstrap_iterations, seed=args.seed)
    summary["directory"] = str(directory)
    atomic_json(directory / GROWTH_SUMMARY, summary)
    atomic_json(directory / "costs.json", ledger.to_json())
    atomic_json(output_dir / "stages/probe-growth.json",
                {"state": "completed", "split": args.split,
                 "verdict": summary["verdict"], "directory": str(directory)})
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return directory / GROWTH_SUMMARY


def probe_position(args: argparse.Namespace) -> Path:
    """P1: score the trigger at the head, the tail, the middle and both ends."""
    output_dir = Path(args.output_dir)
    episodes, manifest = _load_episodes(output_dir)
    workspace = _workspace(args)
    config = _train_config(args)
    checkpoint = _probe_checkpoint(args, manifest, workspace)
    probe = PositionProbeConfig(
        positions=tuple(args.position or ("suffix", "prefix", "both", "middle")),
        mode=args.position_mode, score=args.score,
    )
    target = [episode for episode in episodes if episode.split == args.split]
    if not target:
        raise ValueError(f"no episodes in split {args.split!r}")

    directory = _probe_dir(args, f"position-{probe.mode}")
    directory.mkdir(parents=True, exist_ok=True)
    contract = _probe_contract(args, manifest, workspace, probe="position", config=probe)
    _lock_probe_config(directory / "probe_config.json", contract)

    ledger = CostLedger(device=str(workspace.retriever.device))
    with ledger.active():
        rows = position_probe(
            workspace, target, config, probe, checkpoint=checkpoint,
            output_dir=directory, contract=contract, ledger=ledger, resume=args.resume,
        )
    summary = summarize_position(load_rows(directory / POSITION_ROWS),
                                 baseline=args.position_baseline,
                                 iterations=args.bootstrap_iterations, seed=args.seed)
    if args.attention:
        import torch

        # Mechanistic evidence for the placement claim, measured on Q_sup only:
        # Q_eval stays locked, and an attention share does not need it.
        by_qid = {row["qid"]: row["question"]
                  for row in workspace.queries(target[0].domain)}
        texts = [by_qid[qid] for qid in target[0].support_qids[:args.attention_queries]]
        attention = {}
        for position in probe.positions:
            ids = _trigger_ids_of(directory, position)
            if not ids:
                attention[position] = {"supported": False, "position": position,
                                       "reason": "no scored trigger for this position"}
                continue
            attention[position] = attention_mass(
                workspace, texts, torch.tensor(ids, dtype=torch.long),
                position=position,
            )
        summary["attention"] = attention
    summary["directory"] = str(directory)
    atomic_json(directory / POSITION_SUMMARY, summary)
    atomic_json(directory / "costs.json", ledger.to_json())
    atomic_json(output_dir / "stages/probe-position.json",
                {"state": "completed", "split": args.split,
                 "best_length_matched": summary.get("best_length_matched"),
                 "directory": str(directory)})
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return directory / POSITION_SUMMARY


def _trigger_ids_of(directory: Path, position: str) -> list[int]:
    """Token ids of the first trigger scored at ``position``, from the rows on disk."""
    for row in load_rows(directory / POSITION_ROWS):
        if row.get("position") == position and row.get("applicable"):
            return list(row.get("token_ids") or [])
    return []


def probe_universal(args: argparse.Namespace) -> Path:
    """P0-R1: pair this run's evaluation against a universal-logit run's."""
    if not args.control_dir:
        raise ValueError(
            "--control-dir is required: it is the directory of the B3 "
            "universal-logit run this one is paired against"
        )
    summary = compare_runs(
        Path(args.output_dir), Path(args.control_dir), metric=args.probe_metric,
        iterations=args.bootstrap_iterations, seed=args.seed,
    )
    path = Path(args.output_dir) / R1_SUMMARY
    atomic_json(path, summary)
    atomic_json(Path(args.output_dir) / "stages/probe-universal.json",
                {"state": "completed", "verdict": summary["verdict"]})
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return path


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
        effect = shuffled.get("metric_effect", {})
        if "on_hit" in effect:
            drop = effect["on_hit"]["drop"]
            lines.append(
                f"- ASR-r on the episode's own memory: own {effect['on_hit']['own']:.4f} "
                f"vs swapped {effect['on_hit']['swapped']:.4f}, drop "
                f"{_format(drop['mean_difference'])} "
                f"[{_format(drop['ci_low'])}, {_format(drop['ci_high'])}] "
                f"-> **{effect['verdict']}**"
            )
            lines.append(
                "  This, not the change rate, is the evidence: a changed trigger "
                "with no ASR-r drop is still an inert memory branch."
            )
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
    drift_path = _drift_dir(args) / DRIFT_SUMMARY
    if drift_path.exists():
        lines += _drift_section(read_json(drift_path))
    path = output_dir / "REPORT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    return path


def _format(value: Any, digits: int = 4) -> str:
    """``null`` stays ``null``: a missing measurement is not a zero."""
    return "null" if value is None else f"{value:.{digits}f}"


def _drift_section(summary: dict[str, Any]) -> list[str]:
    baseline = summary.get("baseline")
    lines = [
        "", "## Drift and few-step adaptation", "",
        f"Metric: `{summary.get('metric')}`, baseline `{baseline}`, "
        f"{summary.get('rows')} applicable rows.", "",
        "| method | hit macro | hit micro | worst episode | false activation | "
        "wall s/row |", "|---|---|---|---|---|---|",
    ]
    for method, entry in sorted(summary.get("methods", {}).items()):
        hit = entry["hit"]
        lines.append(
            f"| `{method}` | {_format(hit['macro'])} | {_format(hit['micro'])} "
            f"| {_format(hit['worst'])} | {_format(entry['false_activation'])} "
            f"| {_format(entry['cost']['wall_seconds_per_row'], 3)} |"
        )
    paired = summary.get("paired_vs_baseline", {})
    if isinstance(paired, dict) and paired and "reason" not in paired:
        lines += ["", f"Paired difference against `{baseline}`, "
                      "bootstrapped over episodes (95% CI):", ""]
        for method, entry in sorted(paired.items()):
            interval = (f"[{_format(entry['ci_low'])}, {_format(entry['ci_high'])}]"
                        if entry.get("ci_low") is not None
                        else f"no interval ({entry.get('reason')})")
            lines.append(f"- `{method}`: {_format(entry['mean_difference'])} {interval}")
    by_kind = summary.get("by_kind", {})
    if by_kind:
        methods = sorted(summary.get("methods", {}))
        lines += ["", "| snapshot kind | " + " | ".join(f"`{m}`" for m in methods) + " |",
                  "|---" * (len(methods) + 1) + "|"]
        for kind, entry in sorted(by_kind.items()):
            lines.append(f"| {kind} | "
                         + " | ".join(_format(entry.get(method)) for method in methods)
                         + " |")
    amortization = summary.get("amortization") or {}
    if amortization.get("break_even_episodes") is not None:
        lines += ["", f"Break-even: {amortization['break_even_episodes']:.1f} snapshots, "
                      f"quality gap {_format(amortization.get('quality_gap'))} "
                      f"(comparable: {amortization.get('comparable')})."]
        if amortization.get("reason"):
            lines.append(f"  {amortization['reason']}")
    else:
        lines += ["", f"Break-even: null — {amortization.get('reason', 'not computed')}."]
    skipped = summary.get("not_applicable") or {}
    if skipped:
        lines += ["", "Not applicable:"] + [
            f"- `{method}`: {reason}" for method, reason in sorted(skipped.items())]
    return lines


def smoke(args: argparse.Namespace) -> Path:
    """Run every stage on the fixture retriever, then verify the resume guard."""
    args.fixture = True
    prepare_episodes(args)
    index(args)
    train_stage(args)
    evaluate_stage(args)
    prepare_drift(args)
    adapt_stage(args)
    evaluate_drift_stage(args)
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
    training.add_argument("--mode", choices=MODES, default="generator",
                          help="hotflip is B1, AgentPoison's discrete search per "
                               "episode; --steps then counts HotFlip iterations")
    training.add_argument("--hotflip-candidates", type=int, default=100,
                          help="hotflip: replacements scored exactly per iteration "
                               "(upstream --num-cand)")
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
    training.add_argument("--trigger-position", choices=POSITIONS, default="suffix",
                          help="where the trigger sits inside the query (P1). Part of "
                               "the config hash, so changing it invalidates a "
                               "checkpoint instead of silently rescoring it")

    evaluation = parser.add_argument_group("evaluation")
    evaluation.add_argument("--split", choices=("train", "validation", "test"),
                            default="test")
    evaluation.add_argument("--score", choices=("dot", "cosine"), default="dot")
    evaluation.add_argument("--no-controls", action="store_true")

    drift = parser.add_argument_group("drift (M3)")
    drift.add_argument("--growth", type=float, nargs="+", default=list(DEFAULT_GROWTH),
                       help="memory growth levels as fractions of the base snapshot")
    drift.add_argument("--ablation", action="append", choices=ABLATION_KINDS,
                       default=None, help="repeatable; defaults to all three")
    drift.add_argument("--method", action="append", choices=ADAPT_METHODS, default=None,
                       help="repeatable; defaults to all four adaptation methods")
    drift.add_argument("--adapt-steps", type=int, default=10,
                       help="few-step budget for warm-start; pick it on validation "
                            "and lock it before touching test")
    drift.add_argument("--write-policy", choices=("refresh", "fixed"), default="refresh",
                       help="may the attacker rewrite its poison records at each "
                            "snapshot? Not TrainConfig.poison_mode")
    drift.add_argument("--baseline", choices=ADAPT_METHODS, default="scratch",
                       help="the arm every other one is paired against")
    drift.add_argument("--bootstrap-iterations", type=int, default=10000)
    drift.add_argument("--quality-tolerance", type=float, default=0.0,
                       help="how close two arms must be before a break-even point "
                            "may be quoted as a speedup")

    probes = parser.add_argument_group("probes (P0/P1)")
    probes.add_argument("--drift-seed", type=int, nargs="+", default=list(PROBE_SEEDS),
                        help="resamples of the benign documents added at each growth "
                             "level. One seed is one data split and cannot support a "
                             "claim; the summary compares the spread across these "
                             "against the effect of growth")
    probes.add_argument("--min-base-hit", type=float, default=0.0,
                        help="skip episodes whose trigger does not reach this hit@K on "
                             "the undrifted snapshot. Measured before any growth, so "
                             "it cannot select for decay")
    probes.add_argument("--position", action="append", choices=POSITIONS, default=None,
                        help="repeatable; defaults to suffix/prefix/both/middle")
    probes.add_argument("--position-mode", choices=("transfer", "reoptimize"),
                        default="transfer",
                        help="transfer scores one trigger at every placement; "
                             "reoptimize fits a trigger per placement. Different "
                             "questions -- do not report them as one number")
    probes.add_argument("--position-baseline", choices=POSITIONS, default="suffix",
                        help="the placement every other one is paired against")
    probes.add_argument("--attention", action="store_true",
                        help="also measure the share of [CLS] attention the trigger "
                             "receives at each placement")
    probes.add_argument("--attention-queries", type=int, default=8,
                        help="how many Q_sup queries the attention share averages over")
    probes.add_argument("--control-dir", type=Path, default=None,
                        help="probe-universal: the B3 universal-logit run directory")
    probes.add_argument("--probe-metric", default="on_hit",
                        help="probe-universal: on_hit, off_hit, or any trigger_on key")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="python -m src.triggers.mcat", description=__doc__)
    subparsers = root.add_subparsers(dest="command", required=True)
    for name, handler in (
        ("prepare-episodes", prepare_episodes), ("index", index), ("train", train_stage),
        ("evaluate", evaluate_stage), ("prepare-drift", prepare_drift),
        ("adapt", adapt_stage), ("evaluate-drift", evaluate_drift_stage),
        ("probe-growth", probe_growth), ("probe-position", probe_position),
        ("probe-universal", probe_universal),
        ("report", report), ("smoke", smoke),
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
