"""Falsification probes for the open problems in ``_idea_q1_aplus/open_problems.md``.

VN — Ba phép đo rẻ, chạy TRƯỚC khi đầu tư huấn luyện lớn. Mục đích của chúng là
**bác bỏ** giả thuyết, không phải tối ưu điểm số:

``growth_probe``    P0-R2. Memory phình 25/50/100%, trigger và poison đóng băng.
                    Trigger cũ còn hoạt động không, và có bị kích hoạt ngoài ý
                    muốn không?
``position_probe``  P1. Trigger ở đầu / cuối / giữa / cả hai đầu, cùng ngân sách.
``compare_runs``    P0-R1. Ghép cặp theo episode giữa hai run đã đánh giá, ví dụ
                    ``universal-logit`` so với ``direct-logit``.

Three cheap measurements that run *before* any large training, and whose job is
to falsify a hypothesis rather than to maximize a score.  Each one answers a
question that, if answered the wrong way, means the expensive experiment should
not be run at all.

Design rules these probes are built around, because each is an easy way to
manufacture a result:

**One variable moves.**  ``growth_probe`` freezes the trigger, the poison records
and ``Q_eval``, and changes only how much benign data the memory holds.  A probe
that re-optimized the trigger per snapshot would be measuring adaptation, not
decay.

**The poison count is fixed, not the poison ratio.**  The records are written
once at ``s0`` through ``freeze_poison`` and never rewritten, so growing the
memory lowers the poison *fraction* -- which is the thing being tested.  Scaling
the record count with the memory would confound "more memory" with "more
poison", and the whole probe would say nothing.

**The trigger is chosen on the base snapshot.**  ``min_base_hit`` filters
episodes by how well the trigger works *before* any drift.  Picking the episodes
that decay the most, after seeing them decay, is the shortest path to a fake
result; a filter applied to the undrifted state cannot do that.

**Repeats are resamples, not copies.**  Several drift seeds draw different benign
documents.  The summary compares the spread across seeds against the effect of
growth, and refuses to call a drop real when the spread is larger -- one data
split is not evidence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from torch import nn

from src.triggers.artifacts import append_jsonl, read_json, stable_hash
from src.triggers.mcat.costs import CostLedger
from src.triggers.mcat.drift import Snapshot, build_trajectory
from src.triggers.mcat.drift_eval import load_rows, prepare_bases
from src.triggers.mcat.encoding import POSITIONS, POSITION_COPIES
from src.triggers.mcat.episodes import Episode, family_key
from src.triggers.mcat.evaluate import evaluate_trigger, generate_trigger
from src.triggers.mcat.poison import FrozenPoison
from src.triggers.mcat.runtime import EpisodeContext, Workspace
from src.triggers.mcat.stats import paired_bootstrap
from src.triggers.mcat.train import TrainConfig, train

GROWTH_ROWS = "probe_growth.jsonl"
GROWTH_SUMMARY = "probe_growth.json"
POSITION_ROWS = "probe_position.jsonl"
POSITION_SUMMARY = "probe_position.json"
R1_SUMMARY = "probe_universal.json"

#: Growth levels the protocol in ``open_problems.md`` fixes: +25%, +50%, +100%.
PROBE_GROWTH = (0.25, 0.50, 1.00)
#: Five resamples of the benign documents added at each level.
PROBE_SEEDS = (0, 1, 2, 3, 4)


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def hit_of(metrics: dict[str, Any]) -> float:
    """The ``hit_at_K`` value out of one metrics block, whatever K it used.

    ``retrieval_metrics`` names the key after the effective K, which shrinks when
    a snapshot holds fewer documents than K.  Reading it by prefix keeps a probe
    from going looking for ``hit_at_5`` on a snapshot that only had three.
    """
    for key, value in metrics.items():
        if key.startswith("hit_at_"):
            return float(value)
    raise KeyError(f"no hit@K metric in {sorted(metrics)}")


def _mean(values: Iterable[float]) -> float | None:
    values = [float(value) for value in values]
    return sum(values) / len(values) if values else None


def _stdev(values: Sequence[float]) -> float | None:
    """Sample standard deviation; ``None`` below two observations."""
    values = [float(value) for value in values]
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _done_keys(path: Path) -> set[str]:
    return {row["key"] for row in load_rows(path) if "key" in row}


def growth_pool(
    workspace: Workspace, episode: Episode, *, ratios: tuple[float, ...], seed: int
) -> list[dict[str, Any]]:
    """The benign documents this episode's split may add: same domain, same split.

    VN — Tài liệu lành mà episode này được phép thêm: cùng domain, cùng split, và
    chưa có trong memory. Lấy qua cùng một ``assignment`` mà ``build_episodes``
    đã dùng, nên không có đường nào lấy tài liệu của split khác.
    """
    assignment = workspace.assignment(episode.domain, ratios=tuple(ratios), seed=seed)
    held = set(episode.doc_ids)
    return [row for row in workspace.documents(episode.domain)
            if row["doc_id"] not in held
            and assignment.get(family_key(row["family"])) == episode.split]


# --------------------------------------------------------------------------- #
# P0-R2: does a frozen trigger decay as the memory grows?
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GrowthProbeConfig:
    """The P0-R2 protocol, as data so it lands in the artifact verbatim."""

    growth: tuple[float, ...] = PROBE_GROWTH
    seeds: tuple[int, ...] = PROBE_SEEDS
    #: Keep only episodes whose trigger already works at ``s0``.  Measured before
    #: any drift, so it cannot select for decay.
    min_base_hit: float = 0.0
    score: str = "dot"

    def __post_init__(self) -> None:
        if not self.growth:
            raise ValueError("the growth probe needs at least one growth level")
        if any(value <= 0 for value in self.growth):
            raise ValueError(f"growth levels must be positive, got {self.growth}")
        if not self.seeds:
            raise ValueError("the growth probe needs at least one drift seed")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError(f"drift seeds must be distinct, got {self.seeds}")
        if not 0.0 <= self.min_base_hit <= 1.0:
            raise ValueError(f"min_base_hit must be in [0, 1], got {self.min_base_hit}")

    def fingerprint(self) -> str:
        return stable_hash(asdict(self))


def probe_row_key(episode_id: str, snapshot_id: str, probe: str, contract: str) -> str:
    """Identity of one probe measurement.

    ``snapshot_id`` already carries the drift seed (``build_trajectory`` is called
    with ``tag_seed=True``), which is what keeps five resamples from collapsing
    onto one key.
    """
    return stable_hash([episode_id, snapshot_id, probe, contract])


def growth_probe(
    workspace: Workspace,
    episodes: Sequence[Episode],
    config: TrainConfig,
    probe: GrowthProbeConfig,
    *,
    checkpoint: dict[str, Any] | None,
    output_dir: Path,
    contract: dict[str, Any],
    ratios: tuple[float, ...],
    split_seed: int,
    ledger: CostLedger | None = None,
    resume: bool = False,
) -> list[dict[str, Any]]:
    """Score one frozen trigger against a growing memory, over several resamples.

    VN — Giữ nguyên trigger và poison của ``s0``, cho memory phình theo từng mức,
    rồi đo lại. Mỗi dòng ghi ngay ra đĩa và dòng đã có thì không tính lại, nên
    job bị cắt giữa chừng chạy tiếp được.

    The poison records are written once here, at ``s0``, via ``prepare_bases``
    with ``freeze=True``.  Every later measurement ranks *those* vectors, so the
    record count is constant while the memory -- and therefore the poison
    fraction -- changes.  That is the whole design: see the module docstring.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger = ledger or CostLedger(device=str(workspace.retriever.device))
    rows_path = output_dir / GROWTH_ROWS
    done = _done_keys(rows_path) if resume else set()
    fingerprint = stable_hash([contract, probe.fingerprint()])

    bases, frozen = prepare_bases(
        workspace, episodes, config, checkpoint=checkpoint, output_dir=output_dir,
        contract=contract, ledger=ledger, resume=resume, freeze=True,
    )
    if frozen is None:
        frozen = FrozenPoison.load(output_dir / "poison_s0.pt",
                                   retriever=workspace.retriever)

    rows: list[dict[str, Any]] = []

    def emit(row: dict[str, Any]) -> None:
        append_jsonl(rows_path, row)
        rows.append(row)

    for episode in episodes:
        _, trigger = bases.get(episode.episode_id, (None, None))
        if trigger is None:
            emit({"probe": "growth", "episode_id": episode.episode_id,
                  "key": probe_row_key(episode.episode_id, episode.snapshot_id,
                                       "growth", fingerprint),
                  "applicable": False,
                  "reason": "no s0 trigger; the run has no trained checkpoint"})
            continue

        keys = frozen.keys_for(episode.episode_id)
        base_context = workspace.context_for(episode, centers=False)
        base_row = _score_snapshot(
            workspace, base_context, trigger, keys, probe, config,
            snapshot=None, episode=episode, fingerprint=fingerprint,
            ledger=ledger, done=done, rows_path=rows_path,
        )
        if base_row is not None:
            rows.append(base_row)
        base_hit = (base_row or {}).get("on_hit")
        if base_hit is None:
            # Resumed: the base row is already on disk, so read the gate from it
            # rather than re-encoding an episode that was already measured.
            stored = [row for row in load_rows(rows_path)
                      if row.get("episode_id") == episode.episode_id
                      and row.get("kind") == "base"]
            base_hit = stored[0]["on_hit"] if stored else None

        if base_hit is None:
            continue
        if base_hit < probe.min_base_hit:
            # Excluded on the undrifted state, before any growth was encoded --
            # so the filter cannot have selected for decay.  Recorded, because a
            # gate that leaves no trace is indistinguishable from cherry-picking.
            emit({"probe": "growth", "episode_id": episode.episode_id,
                  "key": probe_row_key(episode.episode_id,
                                       f"{episode.episode_id}-excluded", "growth",
                                       fingerprint),
                  "applicable": False, "kind": "excluded", "on_hit": base_hit,
                  "reason": f"base hit {base_hit:.3f} < min_base_hit "
                            f"{probe.min_base_hit:.3f}"})
            continue

        pool = growth_pool(workspace, episode, ratios=ratios, seed=split_seed)
        assignment = workspace.assignment(episode.domain, ratios=tuple(ratios),
                                          seed=split_seed)
        for drift_seed in probe.seeds:
            snapshots, report = build_trajectory(
                episode, pool, assignment=assignment, growth=probe.growth,
                ablations=(), seed=drift_seed, tag_seed=True,
            )
            for snapshot in snapshots:
                if snapshot.kind == "base":
                    continue  # already scored once; it does not depend on the seed
                row = _score_snapshot(
                    workspace,
                    workspace.context_for(episode, snapshot, centers=False),
                    trigger, keys, probe, config, snapshot=snapshot, episode=episode,
                    fingerprint=fingerprint, ledger=ledger, done=done,
                    rows_path=rows_path, drift_seed=drift_seed,
                    pool_size=report.get("pool"),
                )
                if row is not None:
                    rows.append(row)
    return rows


def _score_snapshot(
    workspace: Workspace,
    context: EpisodeContext,
    trigger: dict[str, Any],
    poison_keys: torch.Tensor,
    probe: GrowthProbeConfig,
    config: TrainConfig,
    *,
    snapshot: Snapshot | None,
    episode: Episode,
    fingerprint: str,
    ledger: CostLedger,
    done: set[str],
    rows_path: Path,
    drift_seed: int | None = None,
    pool_size: int | None = None,
) -> dict[str, Any] | None:
    """One (episode, snapshot) measurement with the frozen trigger and poison."""
    snapshot_id = snapshot.snapshot_id if snapshot is not None else episode.snapshot_id
    key = probe_row_key(episode.episode_id, snapshot_id, "growth", fingerprint)
    if key in done:
        return None
    with ledger.phase(f"{snapshot_id}/growth-probe"):
        scored = evaluate_trigger(workspace, context, trigger, score=probe.score,
                                  frozen_poison=poison_keys,
                                  position=config.trigger_position)
    row = {
        "probe": "growth", "key": key, "applicable": True,
        "episode_id": episode.episode_id, "domain": episode.domain,
        "split": episode.split, "snapshot_id": snapshot_id,
        "kind": "base" if snapshot is None else snapshot.kind,
        "growth": 0.0 if snapshot is None else snapshot.growth,
        "drift_seed": drift_seed,
        "documents": context.memory_vectors.shape[0],
        # Constant by construction: the records were frozen at s0.  Carried on
        # every row so the fixed-count rule can be checked from the artifact
        # instead of taken on trust.
        "poison_records": poison_keys.shape[0],
        "poison_fraction": poison_keys.shape[0] / max(1, context.memory_vectors.shape[0]),
        "pool": pool_size,
        "trigger": scored["trigger"], "trigger_position": config.trigger_position,
        "on_hit": hit_of(scored["trigger_on"]),
        "off_hit": hit_of(scored["trigger_off"]),
        "trigger_on": scored["trigger_on"], "trigger_off": scored["trigger_off"],
    }
    append_jsonl(rows_path, row)
    return row


def summarize_growth(
    rows: Sequence[dict[str, Any]],
    *,
    iterations: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Per-level means, paired drops, and the seed-spread check that gates them.

    VN — Trả lời R2. Ba thứ phải đồng thời đúng mới gọi là "suy giảm rõ và nhất
    quán": (1) khoảng tin cậy ghép cặp của mức drop không chứa 0, (2) on_hit đơn
    điệu không tăng theo mức phình, (3) hiệu ứng lớn hơn độ lệch giữa các seed.
    Thiếu một trong ba thì verdict là ``inconclusive``, không phải "giảm".

    The verdict needs all three of: a paired interval for the drop that excludes
    zero, a mean that does not increase with growth, and an effect larger than
    the spread across drift seeds.  The third is the one that stops a single
    lucky data split from becoming a finding, and it is why the probe resamples
    at all.
    """
    usable = [row for row in rows if row.get("applicable") and "on_hit" in row]
    excluded = [row for row in rows if row.get("kind") == "excluded"]
    base = {row["episode_id"]: row for row in usable if row["kind"] == "base"}
    if not base:
        return {"verdict": "no-data", "reason": "no base snapshot rows",
                "episodes": 0, "levels": {}, "excluded_episodes": len(excluded)}

    by_level: dict[float, list[dict[str, Any]]] = {}
    for row in usable:
        if row["kind"] == "base":
            continue
        by_level.setdefault(float(row["growth"]), []).append(row)

    levels: dict[str, Any] = {}
    for growth in sorted(by_level):
        group = by_level[growth]
        # Average the seeds inside an episode first: an episode is the
        # independent unit (see stats.py), and five seeds of one episode are
        # five looks at the same trigger, not five episodes.
        per_episode: dict[str, list[float]] = {}
        per_episode_off: dict[str, list[float]] = {}
        for row in group:
            per_episode.setdefault(row["episode_id"], []).append(row["on_hit"])
            per_episode_off.setdefault(row["episode_id"], []).append(row["off_hit"])
        shared = [eid for eid in per_episode if eid in base]
        drifted = [float(_mean(per_episode[eid])) for eid in shared]
        anchor = [float(base[eid]["on_hit"]) for eid in shared]
        spreads = [value for eid in shared if (value := _stdev(per_episode[eid])) is not None]
        drop = paired_bootstrap(anchor, drifted, shared, iterations=iterations, seed=seed)
        levels[f"{growth:g}"] = {
            "growth": growth,
            "observations": len(group),
            "episodes": len(shared),
            "drift_seeds": sorted({row["drift_seed"] for row in group
                                   if row["drift_seed"] is not None}),
            "documents": _mean(row["documents"] for row in group),
            "poison_fraction": _mean(row["poison_fraction"] for row in group),
            "on_hit": _mean(drifted),
            "off_hit": _mean(_mean(per_episode_off[eid]) for eid in shared),
            # base - drifted: positive means the frozen trigger lost ground.
            "drop_vs_base": drop,
            "seed_spread": _mean(spreads),
        }

    ordered = [levels[key] for key in sorted(levels, key=lambda key: levels[key]["growth"])]
    base_on = _mean(row["on_hit"] for row in base.values())
    monotone = all(
        left["on_hit"] is not None and right["on_hit"] is not None
        and right["on_hit"] <= left["on_hit"] + 1e-12
        for left, right in zip([{"on_hit": base_on}] + ordered, ordered)
    )

    verdict, reason = "inconclusive", None
    if ordered:
        last = ordered[-1]
        drop = last["drop_vs_base"]
        significant = bool(drop.get("significant")) and (drop.get("ci_low") or 0.0) > 0
        effect = drop.get("mean_difference")
        spread = last["seed_spread"]
        beats_noise = (
            effect is not None and spread is not None and abs(effect) > spread
        )
        if significant and monotone and beats_noise:
            verdict = "decays"
            reason = ("R2 rejected: the frozen trigger loses ground monotonically, the "
                      "paired interval excludes zero, and the effect exceeds the spread "
                      "across drift seeds")
        elif significant and not beats_noise:
            verdict = "inconclusive"
            reason = (f"the drop ({effect}) does not exceed the spread across drift "
                      f"seeds ({spread}); one data split would have called this a result")
        elif not significant:
            verdict = "stable"
            reason = ("no evidence of decay in this setting: the paired interval for the "
                      "largest growth level contains zero. Holds for same-domain growth "
                      "only -- it says nothing about mixture or distractor drift")
        else:
            verdict = "inconclusive"
            reason = "the means are not monotone in the growth level"

    return {
        "verdict": verdict, "reason": reason,
        "episodes": len(base), "excluded_episodes": len(excluded),
        "excluded": [row.get("reason") for row in excluded],
        "base": {"on_hit": base_on,
                 "off_hit": _mean(row["off_hit"] for row in base.values()),
                 "documents": _mean(row["documents"] for row in base.values()),
                 "poison_fraction": _mean(row["poison_fraction"] for row in base.values())},
        "monotone": monotone,
        "levels": levels,
    }


# --------------------------------------------------------------------------- #
# P1: where in the query does the trigger belong?
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PositionProbeConfig:
    """The P1 ablation: which placements, and whether each one is re-optimized."""

    positions: tuple[str, ...] = ("suffix", "prefix", "both", "middle")
    #: ``transfer`` scores the trigger the run already holds at every position --
    #: cheap, and it isolates placement sensitivity at a fixed trigger.
    #: ``reoptimize`` fits a trigger per position, which is the comparison that
    #: says which placement is actually better.  They answer different
    #: questions and must not be reported as one number.
    mode: str = "transfer"
    score: str = "dot"

    def __post_init__(self) -> None:
        unknown = sorted(set(self.positions) - set(POSITIONS))
        if unknown:
            raise ValueError(f"unknown positions {unknown}; choose from {POSITIONS}")
        if not self.positions:
            raise ValueError("the position probe needs at least one position")
        if self.mode not in ("transfer", "reoptimize"):
            raise ValueError(f"mode must be transfer or reoptimize, got {self.mode!r}")

    def fingerprint(self) -> str:
        return stable_hash(asdict(self))


def position_probe(
    workspace: Workspace,
    episodes: Sequence[Episode],
    config: TrainConfig,
    probe: PositionProbeConfig,
    *,
    checkpoint: dict[str, Any] | None,
    output_dir: Path,
    contract: dict[str, Any],
    ledger: CostLedger | None = None,
    resume: bool = False,
) -> list[dict[str, Any]]:
    """Score the trigger at each placement, at an equal token budget.

    VN — Mỗi vị trí một dòng. ``both`` chia trigger làm hai nửa nên tổng số token
    không đổi — so sánh công bằng. ``both-repeat`` dùng gấp đôi token, nên dòng
    của nó mang ``length_matched: false`` và không được đặt cạnh các vị trí khác
    như thể cùng ngân sách.

    Poison is re-encoded at the position under test, because an attacker that
    rewrites its queries one way writes its records the same way.  Keeping an
    ``s0``-suffix poison key while moving the query's trigger would measure a
    mismatch nobody would deploy.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger = ledger or CostLedger(device=str(workspace.retriever.device))
    rows_path = output_dir / POSITION_ROWS
    done = _done_keys(rows_path) if resume else set()
    fingerprint = stable_hash([contract, probe.fingerprint()])
    rows: list[dict[str, Any]] = []

    if probe.mode == "transfer":
        # One trigger, scored everywhere.  ``prepare_bases`` is what supplies it
        # for every mode: a generator run reads its checkpoint, and
        # ``direct-logit`` pays for the per-episode search it has always needed.
        bases, _ = prepare_bases(
            workspace, episodes, config, checkpoint=checkpoint, output_dir=output_dir,
            contract=contract, ledger=ledger, resume=resume, freeze=False,
        )
        triggers = {episode.episode_id: bases.get(episode.episode_id, (None, None))[1]
                    for episode in episodes}
        arms = {position: config for position in probe.positions}
    else:
        triggers, arms = {}, {}
        for position in probe.positions:
            arm = replace(config, trigger_position=position)
            arms[position] = arm
            # A fresh search under this placement: the trigger that wins at the
            # end of the query is not the trigger that wins at the start.
            contexts = [workspace.context_for(episode) for episode in episodes]
            with ledger.phase(f"position/{position}/search"):
                _, modules = train(
                    workspace, list(episodes), arm,
                    output_dir=output_dir / position,
                    contract=dict(contract, probe_position=position), resume=resume,
                    contexts=contexts,
                )
            for module, context, episode in zip(modules, contexts, episodes):
                triggers[(position, episode.episode_id)] = generate_trigger(
                    module, arm, context, workspace.retriever)

    for position in probe.positions:
        for episode in episodes:
            key = probe_row_key(episode.episode_id, f"{position}/{probe.mode}",
                                "position", fingerprint)
            if key in done:
                continue
            trigger = (triggers.get(episode.episode_id) if probe.mode == "transfer"
                       else triggers.get((position, episode.episode_id)))
            if trigger is None:
                rows.append({"probe": "position", "position": position,
                             "episode_id": episode.episode_id, "applicable": False,
                             "reason": "no trigger for this episode; a generator run "
                                       "needs its checkpoint.pt"})
                continue
            context = workspace.context_for(episode, centers=False)
            with ledger.phase(f"position/{position}/{episode.episode_id}"):
                scored = evaluate_trigger(workspace, context, trigger,
                                          score=probe.score, position=position)
            row = {
                "probe": "position", "key": key, "applicable": True,
                "mode": probe.mode, "position": position,
                "trigger_tokens": episode.trigger_tokens,
                "tokens_spent": episode.trigger_tokens * POSITION_COPIES[position],
                "length_matched": POSITION_COPIES[position] == 1,
                "optimized_at": arms[position].trigger_position,
                "episode_id": episode.episode_id, "domain": episode.domain,
                "split": episode.split, "trigger": trigger["trigger"],
                # The ids the runtime would really emit, so the attention pass
                # places the same tokens the retrieval numbers were scored with.
                "token_ids": list(trigger["round_trip"]["re_encoded_ids"]
                                  or trigger["token_ids"]),
                "round_trip_valid": bool(trigger["round_trip"]["valid"]),
                "on_hit": hit_of(scored["trigger_on"]),
                "off_hit": hit_of(scored["trigger_off"]),
                "trigger_on": scored["trigger_on"], "trigger_off": scored["trigger_off"],
            }
            append_jsonl(rows_path, row)
            rows.append(row)
    return rows


def summarize_position(
    rows: Sequence[dict[str, Any]],
    *,
    baseline: str = "suffix",
    iterations: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Per-position means and paired differences against ``baseline``.

    Length-matched positions and ``both-repeat`` are reported in separate blocks
    on purpose: putting a double-length arm in the same ranking as the others
    would read as a placement win when it is a budget win.
    """
    usable = [row for row in rows if row.get("applicable") and "on_hit" in row]
    if not usable:
        return {"positions": {}, "baseline": baseline, "reason": "no usable rows"}
    anchor = {row["episode_id"]: row for row in usable if row["position"] == baseline}

    positions: dict[str, Any] = {}
    for position in sorted({row["position"] for row in usable}):
        group = [row for row in usable if row["position"] == position]
        shared = [row["episode_id"] for row in group if row["episode_id"] in anchor]
        difference = paired_bootstrap(
            [next(r["on_hit"] for r in group if r["episode_id"] == eid) for eid in shared],
            [anchor[eid]["on_hit"] for eid in shared],
            shared, iterations=iterations, seed=seed,
        ) if position != baseline else None
        positions[position] = {
            "episodes": len(group),
            "length_matched": all(row["length_matched"] for row in group),
            "tokens_spent": _mean(row["tokens_spent"] for row in group),
            "on_hit": _mean(row["on_hit"] for row in group),
            "off_hit": _mean(row["off_hit"] for row in group),
            "round_trip_valid_rate": _mean(float(row["round_trip_valid"]) for row in group),
            "vs_baseline": difference,
        }
    matched = {name: block for name, block in positions.items() if block["length_matched"]}
    best = max(matched, key=lambda name: matched[name]["on_hit"] or 0.0) if matched else None
    return {
        "baseline": baseline,
        "mode": usable[0].get("mode"),
        "best_length_matched": best,
        "not_length_matched": sorted(name for name, block in positions.items()
                                     if not block["length_matched"]),
        "positions": positions,
    }


def attention_mass(
    workspace: Workspace,
    texts: Sequence[str],
    trigger_ids: torch.Tensor,
    *,
    position: str = "suffix",
) -> dict[str, Any]:
    """Share of ``[CLS]`` attention that lands on the trigger, last layer.

    VN — Bằng chứng *mechanistic* cho P1: trong tổng attention mà ``[CLS]`` phát
    ra ở layer cuối, bao nhiêu phần rơi vào token trigger. Nếu "chèn vào giữa thì
    trọng số không ảnh hưởng lắm" là đúng, con số này phải thấp hẳn ở ``middle``.

    DPR pools the ``[CLS]`` row, so the attention that row sends is the channel
    through which a trigger can move the vector at all.  Returns ``supported:
    False`` rather than raising when the encoder does not expose attentions,
    because the probe it belongs to must still produce its retrieval numbers.
    """
    retriever = workspace.retriever
    model, tokenizer = retriever.model, retriever.tokenizer
    embedding = model.get_input_embeddings()
    trigger_embeds = embedding(trigger_ids.to(retriever.device))
    length = trigger_embeds.shape[0]
    spent = length * POSITION_COPIES[position]
    budget = retriever.max_length - spent - 2

    shares: list[float] = []
    # SDPA/flash kernels (the transformers 5 default) never materialize attention
    # weights, so output_attentions comes back empty.  Switch to eager for this
    # measurement only and restore afterwards: retrieval numbers stay on SDPA.
    previous = getattr(model.config, "_attn_implementation", None)
    switch = getattr(model, "set_attn_implementation", None)
    if switch is not None and previous not in (None, "eager"):
        switch("eager")
    try:
        return _attention_shares(model, tokenizer, retriever, texts, position, embedding,
                                 trigger_embeds, length, spent, budget, shares)
    finally:
        if switch is not None and previous not in (None, "eager"):
            switch(previous)


def _attention_shares(model, tokenizer, retriever, texts, position, embedding,
                      trigger_embeds, length, spent, budget, shares):
    with torch.no_grad():
        for text in texts:
            prefix = tokenizer(text, add_special_tokens=False, truncation=True,
                               max_length=budget).input_ids
            body = torch.tensor(prefix, dtype=torch.long, device=retriever.device)
            cls = torch.tensor([tokenizer.cls_token_id], device=retriever.device)
            sep = torch.tensor([tokenizer.sep_token_id], device=retriever.device)
            from src.triggers.mcat.encoding import _assemble
            sequence = _assemble(position, embedding(cls), embedding(body),
                                 embedding(sep), trigger_embeds)
            try:
                output = model(inputs_embeds=sequence.unsqueeze(0),
                               attention_mask=torch.ones(1, sequence.shape[0],
                                                         dtype=torch.long,
                                                         device=retriever.device),
                               output_attentions=True)
            except (TypeError, ValueError) as error:
                return {"supported": False, "reason": str(error), "position": position}
            attentions = getattr(output, "attentions", None)
            if not attentions:
                return {"supported": False, "position": position,
                        "reason": "the encoder returned no attentions"}
            # Last layer, mean over heads, the [CLS] query row.
            row = attentions[-1][0].mean(dim=0)[0]
            mask = _trigger_mask(position, len(prefix), length, row.shape[0],
                                 device=row.device)
            shares.append(float(row[mask].sum()))
    return {"supported": True, "position": position, "queries": len(shares),
            "cls_attention_on_trigger": _mean(shares),
            "trigger_tokens": length, "tokens_spent": spent}


def _trigger_mask(
    position: str, query_length: int, trigger_length: int, total: int, *, device
) -> torch.Tensor:
    """Which sequence slots hold trigger embeddings, for the given layout.

    Mirrors ``encoding._assemble`` slot for slot.  Kept next to the probe that
    needs it rather than inside ``encoding`` so the encoder stays free of
    analysis-only code -- but if ``_assemble`` gains a layout, this must gain the
    same one, or the attention number silently measures the wrong tokens.
    """
    mask = torch.zeros(total, dtype=torch.bool, device=device)
    if position == "suffix":
        start = 1 + query_length
        mask[start:start + trigger_length] = True
    elif position == "prefix":
        mask[1:1 + trigger_length] = True
    elif position == "middle":
        start = 1 + query_length // 2
        mask[start:start + trigger_length] = True
    elif position == "both":
        cut = (trigger_length + 1) // 2
        mask[1:1 + cut] = True
        tail = 1 + cut + query_length
        mask[tail:tail + (trigger_length - cut)] = True
    elif position == "both-repeat":
        mask[1:1 + trigger_length] = True
        tail = 1 + trigger_length + query_length
        mask[tail:tail + trigger_length] = True
    else:
        raise ValueError(f"unknown position {position!r}")
    return mask


# --------------------------------------------------------------------------- #
# P0-R1: is one universal trigger already enough?
# --------------------------------------------------------------------------- #
def compare_runs(
    treatment: Path,
    control: Path,
    *,
    metric: str = "on_hit",
    iterations: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Pair two finished evaluations by episode and bootstrap the difference.

    VN — Trả lời R1. ``treatment`` là run cần chứng minh (ví dụ ``direct-logit``
    hoặc ``generator``), ``control`` là B3 ``universal-logit``. Hai run phải đo
    trên **cùng** episode và cùng encoder; hàm này kiểm tra điều đó rồi mới ghép
    cặp, vì ghép cặp hai tập episode khác nhau thì con số vô nghĩa.

    Reads each run's ``evaluation.jsonl``.  Refuses rather than guesses when the
    two runs disagree on the trigger position or scored different episodes: a
    paired test across different inputs is not a paired test.
    """
    left_rows = load_rows(Path(treatment) / "evaluation.jsonl")
    right_rows = load_rows(Path(control) / "evaluation.jsonl")
    if not left_rows or not right_rows:
        raise FileNotFoundError(
            f"both runs need an evaluation.jsonl; got {len(left_rows)} and "
            f"{len(right_rows)} rows. Run the evaluate stage in each directory first"
        )

    def value(row: dict[str, Any]) -> float:
        return (hit_of(row["trigger_on"]) if metric == "on_hit"
                else hit_of(row["trigger_off"]) if metric == "off_hit"
                else float(row["trigger_on"][metric]))

    left = {row["episode_id"]: row for row in left_rows}
    right = {row["episode_id"]: row for row in right_rows}
    shared = sorted(set(left) & set(right))
    if not shared:
        raise ValueError(
            "the two runs share no episode id, so there is nothing to pair; they were "
            "built from different splits"
        )
    missing = sorted((set(left) | set(right)) - set(shared))

    positions = {row.get("trigger_position", "suffix")
                 for row in list(left.values()) + list(right.values())}
    if len(positions) > 1:
        raise ValueError(
            f"the runs were scored at different trigger positions {sorted(positions)}; "
            "a difference between them is placement, not method"
        )

    difference = paired_bootstrap([value(left[eid]) for eid in shared],
                                  [value(right[eid]) for eid in shared],
                                  shared, iterations=iterations, seed=seed)
    treatment_config = _config_of(treatment)
    control_config = _config_of(control)
    budget_note = None
    if treatment_config.get("steps") != control_config.get("steps"):
        budget_note = (
            f"unequal budget: {treatment_config.get('steps')} vs "
            f"{control_config.get('steps')} optimization steps. A per-episode arm given "
            "more steps than the universal arm wins on compute, not on conditioning"
        )
    significant = bool(difference.get("significant")) and (difference.get("ci_low") or 0) > 0
    return {
        "metric": metric,
        "treatment": {"directory": str(treatment), "mode": treatment_config.get("mode"),
                      "steps": treatment_config.get("steps"),
                      "mean": _mean(value(left[eid]) for eid in shared)},
        "control": {"directory": str(control), "mode": control_config.get("mode"),
                    "steps": control_config.get("steps"),
                    "mean": _mean(value(right[eid]) for eid in shared)},
        "paired": difference,
        "episodes": len(shared),
        "unpaired_episodes": missing,
        "trigger_position": sorted(positions)[0],
        "budget_warning": budget_note,
        "verdict": "conditioning-helps" if significant and not budget_note else (
            "universal-suffices" if not significant else "confounded"),
        "reason": budget_note or (
            "R1 rejected: the conditioned arm beats one universal trigger by a paired "
            "margin whose interval excludes zero" if significant else
            "no evidence that conditioning beats one universal trigger at this budget"
        ),
    }


def _config_of(directory: Path) -> dict[str, Any]:
    path = Path(directory) / "evaluation.json"
    if not path.exists():
        return {}
    return read_json(path).get("config", {})
