"""The drift experiment: every (episode, snapshot, method) scored and priced.

VN — Đây là chỗ ba mảnh của M3 ráp lại: ``drift`` cấp snapshot, ``adapt`` cấp
bốn cách sinh trigger, ``evaluate`` chấm điểm trigger đã decode trên tập query
bị khóa, ``costs`` tính tiền. Kết quả là một dòng cho mỗi ô
(episode, snapshot, method).

Hai thứ cố ý **không** gộp trung bình: dòng của snapshot ``base`` là mốc chưa
trôi nên để riêng, và hai chính sách ghi ``refresh``/``fixed`` không bao giờ
chung một con số — mỗi dòng mang theo chính sách đã sinh ra nó.

This is where the three M3 pieces meet.  ``drift`` supplies the snapshots,
``adapt`` supplies the four ways of having a trigger for one, ``evaluate``
scores the decoded trigger on the locked queries, and ``costs`` bills it.

Two things are deliberately not averaged together.  The ``base`` snapshot rows
are the no-drift reference and are reported as their own row rather than folded
into the mean, and ``refresh``/``fixed`` write policies never share a number --
each row carries the policy it was produced under.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from torch import nn

from src.triggers.artifacts import append_jsonl, read_json, stable_hash
from src.triggers.mcat.adapt import AdaptConfig, AdaptResult, adapt_all
from src.triggers.mcat.costs import CostLedger, amortization
from src.triggers.mcat.drift import Snapshot
from src.triggers.mcat.episodes import Episode
from src.triggers.mcat.evaluate import evaluate_trigger, generate_trigger
from src.triggers.mcat.poison import FrozenPoison
from src.triggers.mcat.runtime import EpisodeContext, Workspace
from src.triggers.mcat.stats import macro_micro_worst, paired_bootstrap
from src.triggers.mcat.train import TrainConfig, _logits_source, train

DRIFT_ROWS = "drift_evaluation.jsonl"
DRIFT_SUMMARY = "drift_evaluation.json"


def row_key(
    episode_id: str, snapshot_id: str, method: str, adapt_config: AdaptConfig
) -> str:
    """Identity of one measurement, including the budget it was made under.

    VN — Khóa định danh một phép đo. Có cả ngân sách few-step trong khóa, vì
    ``warm-start`` 5 bước và 50 bước là hai arm khác nhau; khóa mà bỏ qua nó thì
    lúc resume sẽ nhận nhầm dòng của ngân sách khác.

    The few-step budget is part of the key because ``warm-start`` at 5 steps and
    at 50 steps are different arms; a key that ignored it would let a resume
    hand back a row produced under another budget.
    """
    return stable_hash([episode_id, snapshot_id, method, adapt_config.fingerprint()])


def load_rows(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def base_for_episode(
    workspace: Workspace,
    episode: Episode,
    config: TrainConfig,
    *,
    checkpoint: dict[str, Any] | None,
    output_dir: Path,
    contract: dict[str, Any],
    ledger: CostLedger,
    resume: bool = False,
) -> tuple[nn.Module | None, dict[str, Any] | None, EpisodeContext]:
    """The attacker's position at ``s0``: the module it holds and what it deployed.

    VN — Vị thế của kẻ tấn công tại ``s0``: nó đang cầm module nào và đã thả
    trigger nào. Với các mode dùng chung một module thì đó là generator đã train,
    khôi phục từ checkpoint. Riêng ``direct-logit`` không có gì để chuyển sang
    snapshot mới, nên vị thế ``s0`` của nó là thứ mà một lần tìm kiếm per-episode
    thật sự sẽ tạo ra: tối ưu đầy đủ trên snapshot gốc, và vẫn bị tính tiền như
    mọi method khác.

    For the shared modes that is the trained generator, restored from the run's
    checkpoint.  ``direct-logit`` has nothing to transfer, so its ``s0`` position
    is what a per-episode search would actually have produced: a full
    optimization on the base snapshot, paid for and recorded like any other.
    """
    context = workspace.context_for(episode)
    if config.mode == "direct-logit":
        directory = Path(output_dir) / context.snapshot_id / "base-search"
        with ledger.phase(f"{context.snapshot_id}/base-search"):
            _, modules = train(
                workspace, [episode], config, output_dir=directory,
                contract=dict(contract, adapt_method="base-search",
                              adapt_snapshot=context.snapshot_id),
                resume=resume, contexts=[context],
            )
            trigger = generate_trigger(modules[0], config, context, workspace.retriever)
        return modules[0], trigger, context

    if checkpoint is None:
        return None, None, context
    module = _logits_source(config, workspace.retriever, episode)
    module.load_state_dict(checkpoint["modules"][0])
    module.eval()
    with ledger.phase(f"{context.snapshot_id}/deploy"):
        trigger = generate_trigger(module, config, context, workspace.retriever)
    return module, trigger, context


def prepare_bases(
    workspace: Workspace,
    episodes: Sequence[Episode],
    config: TrainConfig,
    *,
    checkpoint: dict[str, Any] | None,
    output_dir: Path,
    contract: dict[str, Any],
    ledger: CostLedger,
    resume: bool = False,
    freeze: bool = False,
) -> tuple[dict[str, tuple[nn.Module | None, dict[str, Any] | None]],
           FrozenPoison | None]:
    """Everything that happens at ``s0``, before any drift.

    VN — Mọi thứ xảy ra tại ``s0``, trước khi có drift. Khi bật ``freeze``, poison
    được ghi **một lần** tại đây bằng trigger của ``s0`` — đúng nghĩa chính sách
    ``fixed``. Ghi muộn hơn, theo từng snapshot, thì đó là ``refresh`` đội lốt.

    When ``freeze`` is set the poison records are written here, once, with the
    ``s0`` trigger -- which is what the ``fixed`` write policy means.  Freezing
    later, per snapshot, would silently be the ``refresh`` policy wearing the
    other name.
    """
    from src.triggers.mcat.poison import freeze_poison

    bases: dict[str, tuple[nn.Module | None, dict[str, Any] | None]] = {}
    contexts, triggers = [], []
    for episode in episodes:
        module, trigger, context = base_for_episode(
            workspace, episode, config, checkpoint=checkpoint, output_dir=output_dir,
            contract=contract, ledger=ledger, resume=resume,
        )
        bases[episode.episode_id] = (module, trigger)
        if trigger is not None:
            contexts.append(context)
            triggers.append(trigger)
    if not freeze:
        return bases, None
    if not triggers:
        raise ValueError(
            "the fixed write policy needs an s0 trigger per episode, and none was "
            "produced; check that the run has a trained checkpoint"
        )
    return bases, freeze_poison(workspace.retriever, contexts, triggers,
                                output_dir=output_dir)


def _metric_key(metrics: dict[str, Any]) -> str:
    for key in metrics:
        if key.startswith("hit_at_"):
            return key
    raise KeyError(f"no hit@K metric in {sorted(metrics)}")


def run_trajectory(
    workspace: Workspace,
    *,
    episodes: Sequence[Episode],
    snapshots: Sequence[Snapshot],
    config: TrainConfig,
    adapt_config: AdaptConfig,
    checkpoint: dict[str, Any] | None,
    output_dir: Path,
    contract: dict[str, Any],
    frozen_poison: FrozenPoison | None = None,
    score: str = "dot",
    resume: bool = False,
    ledger: CostLedger | None = None,
    bases: dict[str, tuple[nn.Module | None, dict[str, Any] | None]] | None = None,
) -> list[dict[str, Any]]:
    """Adapt and score every method on every snapshot of every episode.

    VN — Vòng chạy chính: với mỗi episode, mỗi snapshot, mỗi method — sinh trigger
    rồi chấm điểm. Mỗi dòng được ghi ngay vào ``drift_evaluation.jsonl`` và dòng
    nào đã có trên đĩa thì không tính lại, nên job bị cắt giữa chừng sẽ chạy tiếp
    chứ không làm lại từ đầu.

    Rows are appended to ``drift_evaluation.jsonl`` as they are produced and a
    row already on disk is never recomputed, so a job cut mid-trajectory
    continues instead of restarting.
    """
    output_dir = Path(output_dir)
    ledger = ledger or CostLedger(device=str(workspace.retriever.device))
    rows_path = output_dir / DRIFT_ROWS
    existing = {row["key"]: row for row in load_rows(rows_path)} if resume else {}
    if not resume and rows_path.exists():
        raise FileExistsError(f"{rows_path} exists; pass --resume to continue it")

    by_episode: dict[str, list[Snapshot]] = {}
    for snapshot in snapshots:
        by_episode.setdefault(snapshot.episode_id, []).append(snapshot)

    rows: list[dict[str, Any]] = []
    for episode in episodes:
        trajectory = by_episode.get(episode.episode_id, [])
        if not trajectory:
            continue
        if bases is not None and episode.episode_id in bases:
            base_module, base_trigger = bases[episode.episode_id]
        else:
            base_module, base_trigger, _ = base_for_episode(
                workspace, episode, config, checkpoint=checkpoint,
                output_dir=output_dir, contract=contract, ledger=ledger, resume=resume,
            )
        frozen = (None if frozen_poison is None
                  else frozen_poison.keys_for(episode.episode_id))
        for snapshot in trajectory:
            context = workspace.context_for(episode, snapshot)
            keys = {method: row_key(episode.episode_id, snapshot.snapshot_id,
                                    method, adapt_config)
                    for method in adapt_config.methods}
            pending = [method for method, key in keys.items() if key not in existing]
            done = [existing[key] for key in keys.values() if key in existing]
            rows.extend(done)
            if not pending:
                continue
            results = adapt_all(
                tuple(pending), workspace=workspace, episode=episode, context=context,
                config=config, adapt_config=adapt_config, base_module=base_module,
                base_trigger=base_trigger, output_dir=output_dir, contract=contract,
                ledger=ledger, resume=resume,
            )
            for result in results:
                row = _score(workspace, context, snapshot, result, adapt_config,
                             frozen=frozen, score=score, ledger=ledger)
                append_jsonl(rows_path, row)
                rows.append(row)
    return rows


def _score(
    workspace: Workspace,
    context: EpisodeContext,
    snapshot: Snapshot,
    result: AdaptResult,
    adapt_config: AdaptConfig,
    *,
    frozen: torch.Tensor | None,
    score: str,
    ledger: CostLedger,
) -> dict[str, Any]:
    """Score a frozen trigger, billing the deployment separately from the search.

    VN — Chấm một trigger đã chốt, và tách hóa đơn làm hai. ``cost`` là tiền
    **sinh ra** trigger — đây là phần khác nhau giữa các method, nên amortization
    dùng nó. ``deploy_cost`` là lần ghi index đi sau, mà method nào cũng phải
    trả như nhau. Tách ra thì phép kiểm tra "budget bằng nhau" mới có ý nghĩa,
    thay vì thành hiển nhiên đúng.

    ``cost`` is what producing the trigger cost and is the number the
    amortization uses, because that is what differs between methods.
    ``deploy_cost`` is the index write that follows, which every method owes
    equally -- keeping the two apart is what makes the budget check meaningful
    instead of a tautology.
    """
    row: dict[str, Any] = {
        "key": row_key(result.episode_id, result.snapshot_id, result.method,
                       adapt_config),
        "episode_id": result.episode_id, "snapshot_id": result.snapshot_id,
        "domain": snapshot.domain, "split": snapshot.split, "kind": snapshot.kind,
        "growth": snapshot.growth, "documents": len(snapshot.doc_ids),
        "method": result.method, "write_policy": adapt_config.write_policy,
        "adapt_steps": adapt_config.steps, "applicable": result.applicable,
        "reason": result.reason, "cost": result.cost,
    }
    if not result.applicable:
        return row
    phase = f"{result.snapshot_id}/{result.method}/deploy"
    with ledger.phase(phase):
        scored = evaluate_trigger(workspace, context, result.trigger, score=score,
                                  frozen_poison=frozen)
    entry = ledger.phase_named(phase)
    row["deploy_cost"] = {} if entry is None else entry.to_json()
    return row | {key: value for key, value in scored.items()
                  if key not in ("episode_id", "snapshot_id", "domain", "split")}


def aggregate(
    rows: Iterable[dict[str, Any]],
    *,
    baseline: str = "scratch",
    train_cost: float | None = None,
    quality_tolerance: float = 0.0,
    iterations: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Per-method means, paired intervals against ``baseline``, and the bill.

    VN — Tổng hợp: trung bình theo từng method, khoảng tin cậy theo cặp so với
    ``baseline``, và chi phí. Báo cáo trích khoảng tin cậy; các trung bình thô để
    đối chiếu. ``amortization`` chỉ tính khi biết ``train_cost``, vì điểm hòa vốn
    mà thiếu tiền train thì không phải điểm hòa vốn.

    The paired comparison is what the report quotes; the raw means are there to
    read it against.  ``amortization`` is only computed when ``train_cost`` is
    known, because a break-even point without the training term is not one.
    """
    usable = [row for row in rows if row.get("applicable") and "trigger_on" in row]
    summary: dict[str, Any] = {
        "rows": len(list(usable)), "baseline": baseline,
        "methods": {}, "by_kind": {}, "paired_vs_baseline": {},
        "not_applicable": {},
    }
    if not usable:
        summary["reason"] = "no applicable rows to aggregate"
        return summary

    key = _metric_key(usable[0]["trigger_on"])
    summary["metric"] = key
    methods = sorted({row["method"] for row in usable})

    for method in methods:
        subset = [row for row in usable if row["method"] == method]
        summary["methods"][method] = {
            "rows": len(subset),
            "hit": macro_micro_worst(
                [row["trigger_on"][key] for row in subset],
                [row["trigger_on"]["queries"] for row in subset],
                [row["episode_id"] for row in subset],
            ),
            "false_activation": _mean([row["false_activation"] for row in subset]),
            "round_trip_valid_rate": _mean(
                [float(row["round_trip_valid"]) for row in subset]),
            "cost": _cost_totals(subset),
            "deploy_cost": _cost_totals(subset, field="deploy_cost"),
        }

    for kind in sorted({row["kind"] for row in usable}):
        summary["by_kind"][kind] = {
            method: _mean([row["trigger_on"][key] for row in usable
                           if row["kind"] == kind and row["method"] == method])
            for method in methods
        }

    paired = {row["method"]: {} for row in usable}
    for row in usable:
        paired[row["method"]][(row["episode_id"], row["snapshot_id"])] = row
    if baseline in paired:
        for method in methods:
            if method == baseline:
                continue
            shared = sorted(set(paired[method]) & set(paired[baseline]))
            summary["paired_vs_baseline"][method] = paired_bootstrap(
                [paired[method][pair]["trigger_on"][key] for pair in shared],
                [paired[baseline][pair]["trigger_on"][key] for pair in shared],
                [pair[0] for pair in shared],
                iterations=iterations, seed=seed,
            )
    else:
        summary["paired_vs_baseline"] = {
            "reason": f"baseline {baseline!r} produced no applicable rows"}

    skipped = [row for row in rows if not row.get("applicable")]
    for row in skipped:
        summary["not_applicable"].setdefault(row["method"], row.get("reason"))

    summary["amortization"] = _amortization(
        summary, baseline=baseline, train_cost=train_cost,
        quality_tolerance=quality_tolerance,
    )
    return summary


def _amortization(
    summary: dict[str, Any], *, baseline: str, train_cost: float | None,
    quality_tolerance: float,
) -> dict[str, Any] | None:
    methods = summary["methods"]
    if train_cost is None or "generate" not in methods or baseline not in methods:
        return {"reason": "needs a training cost plus a generate and a baseline arm"}
    gap = summary.get("paired_vs_baseline", {}).get("generate", {})
    return amortization(
        train_cost=train_cost,
        online_search=methods[baseline]["cost"]["wall_seconds_per_row"],
        online_gen=methods["generate"]["cost"]["wall_seconds_per_row"],
        quality_gap=gap.get("mean_difference") if isinstance(gap, dict) else None,
        quality_tolerance=quality_tolerance,
    )


def _cost_totals(rows: Sequence[dict[str, Any]], *, field: str = "cost") -> dict[str, Any]:
    counters: dict[str, float] = {}
    wall = 0.0
    for row in rows:
        cost = row.get(field) or {}
        wall += float(cost.get("wall_seconds", 0.0))
        for name, value in (cost.get("counters") or {}).items():
            counters[name] = counters.get(name, 0.0) + float(value)
    return {"rows": len(rows), "wall_seconds": wall,
            "wall_seconds_per_row": wall / len(rows) if rows else 0.0,
            "counters": counters}


def _mean(values: Sequence[float]) -> float | None:
    values = [value for value in values if value is not None]
    return float(sum(values) / len(values)) if values else None


def load_summary(output_dir: Path) -> dict[str, Any]:
    return read_json(Path(output_dir) / DRIFT_SUMMARY)
