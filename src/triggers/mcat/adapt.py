"""Four ways to have a trigger for a snapshot, and what each one costs.

VN — Bộ nhớ agent vừa đổi. Kẻ tấn công có 4 lựa chọn để có trigger cho trạng
thái mới, file này chạy cả 4 rồi so sánh:

* ``reuse``      : xài lại trigger cũ, không tốn gì.
* ``generate``   : chạy generator một lượt (một forward). Đây là claim của MCAT.
* ``warm-start`` : lấy tham số cũ rồi tối ưu thêm vài bước. Đối thủ đáng gờm nhất.
* ``scratch``    : tối ưu lại từ đầu, đủ budget. Tốt nhất và đắt nhất.

Ba luật công bằng được ép trong code chứ không tin vào người gọi: cùng snapshot,
cùng ``Q_sup``/``Q_opt``, không ai thấy ``Q_eval``; cùng quyền ghi poison; và
``warm-start``/``scratch`` dùng lại đúng ``train.train`` nên chênh lệch chi phí là
chênh lệch method, không phải chênh lệch cách code.

When the agent's memory drifts, an attacker can do one of four things.  M3 is
the comparison between them, and it is the comparison that can end the project:

``reuse``       keep the trigger from ``s0``.  Costs nothing.  If it does not
                lose anything either, the drift was too weak to be informative
                and the rest of the table says nothing.
``generate``    one forward pass of the conditioned generator on the new
                snapshot.  This is MCAT's claim.
``warm-start``  a few optimization steps from the previous parameters.  This is
                the real competitor: if a handful of steps matches ``generate``
                at a comparable price, the generator is not necessary and
                section 14 of the plan calls for a pivot.
``scratch``     re-optimize from initialization with the full budget.  The
                quality ceiling and the price ceiling.

Fairness rules enforced here rather than trusted to the caller:

* every method sees the same snapshot, the same ``Q_sup`` and the same ``Q_opt``,
  and none of them ever sees ``Q_eval`` -- ``EpisodeContext`` does not carry it;
* every method runs under the same poison-write policy, so the budgeted resource
  is equal;
* ``warm-start`` and ``scratch`` reuse ``train.train`` unchanged, so a cost
  difference between them is a difference in method, not in implementation.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from torch import nn

from src.triggers.artifacts import stable_hash
from src.triggers.mcat.costs import CostLedger
from src.triggers.mcat.episodes import Episode
from src.triggers.mcat.evaluate import generate_trigger
from src.triggers.mcat.retrievers import Retriever
from src.triggers.mcat.runtime import EpisodeContext, Workspace
from src.triggers.mcat.train import TrainConfig, _logits_source, train

ADAPT_METHODS = ("reuse", "generate", "warm-start", "scratch")
ONLINE_METHODS = ("generate", "warm-start", "scratch")


@dataclass(frozen=True)
class AdaptConfig:
    """What the adaptation arms share; hashed into the drift contract.

    VN — Cấu hình chung của 4 arm. ``write_policy`` là quyền **ghi vào index**
    qua các snapshot (``fixed`` = giữ nguyên record đã ghi ở ``s0``). Đừng nhầm
    với ``TrainConfig.poison_mode``, thứ quyết định gradient có chạy qua phía
    poison trong một bước train hay không. Hai chuyện khác nhau, cố ý đặt hai tên.

    ``write_policy`` is the attacker's index-write permission across snapshots
    (``fixed`` = the records written at ``s0`` stay).  It is *not*
    ``TrainConfig.poison_mode``, which decides whether gradient reaches the
    poison side inside one optimization step.  Two different questions, kept
    under two different names on purpose.
    """

    methods: tuple[str, ...] = ADAPT_METHODS
    steps: int = 10               # few-step budget for warm-start
    write_policy: str = "refresh"

    def __post_init__(self) -> None:
        unknown = sorted(set(self.methods) - set(ADAPT_METHODS))
        if unknown:
            raise ValueError(f"unknown methods {unknown}; choose from {ADAPT_METHODS}")
        if not self.methods:
            raise ValueError("at least one adaptation method is required")
        if self.steps < 0:
            raise ValueError(f"steps must not be negative, got {self.steps}")
        if self.write_policy not in ("refresh", "fixed"):
            raise ValueError(
                f"write_policy must be refresh or fixed, got {self.write_policy!r}"
            )

    def fingerprint(self) -> str:
        return stable_hash(asdict(self))


@dataclass
class AdaptResult:
    """One (snapshot, method) outcome: the trigger, the module, the bill.

    VN — Kết quả của một ô (snapshot, method): trigger sinh ra, module tương ứng
    và hóa đơn chi phí. ``applicable=False`` nghĩa là method này không áp dụng
    được ở đây, và ``reason`` nói vì sao — không im lặng bỏ qua.
    """

    method: str
    episode_id: str
    snapshot_id: str
    applicable: bool
    trigger: dict[str, Any] | None = None
    module: nn.Module | None = None
    cost: dict[str, Any] = field(default_factory=dict)
    reason: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {"method": self.method, "episode_id": self.episode_id,
                "snapshot_id": self.snapshot_id, "applicable": self.applicable,
                "trigger": None if self.trigger is None else self.trigger["trigger"],
                "round_trip_valid": None if self.trigger is None
                else bool(self.trigger["round_trip"]["valid"]),
                "cost": self.cost, "reason": self.reason}


def _not_applicable(method: str, context: EpisodeContext, reason: str) -> AdaptResult:
    return AdaptResult(method=method, episode_id=context.episode.episode_id,
                       snapshot_id=context.snapshot_id, applicable=False, reason=reason)


def adapt_trigger(
    method: str,
    *,
    workspace: Workspace,
    episode: Episode,
    context: EpisodeContext,
    config: TrainConfig,
    adapt_config: AdaptConfig,
    base_module: nn.Module | None,
    base_trigger: dict[str, Any] | None,
    output_dir: Path,
    contract: dict[str, Any],
    ledger: CostLedger | None = None,
    resume: bool = False,
) -> AdaptResult:
    """Produce a trigger for ``context`` by ``method`` and bill it to ``ledger``.

    VN — Sinh trigger cho snapshot ``context`` bằng ``method``, và ghi chi phí
    vào ``ledger``. ``base_module``/``base_trigger`` là vốn liếng có từ ``s0``:
    generator đã train và trigger nó đã xuất ra trước khi bộ nhớ trôi.

    ``base_module`` and ``base_trigger`` come from the ``s0`` run: the generator
    that was trained, and the trigger it exported before the drift.
    """
    if method not in ADAPT_METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {ADAPT_METHODS}")
    retriever = workspace.retriever
    ledger = ledger or CostLedger(device=str(retriever.device))
    phase = f"{context.snapshot_id}/{method}"

    if method == "reuse":
        if base_trigger is None:
            return _not_applicable(method, context, "no s0 trigger to reuse")
        with ledger.phase(phase):
            # Costs nothing on purpose: the point of the arm is the price of
            # doing nothing, so nothing is what it does.
            trigger = deepcopy(base_trigger)
        return _result(method, context, trigger, base_module, ledger, phase)

    if method == "generate":
        if config.mode != "generator":
            return _not_applicable(
                method, context,
                f"mode {config.mode!r} has nothing to transfer to a new snapshot")
        if base_module is None:
            return _not_applicable(method, context, "no trained generator")
        with ledger.phase(phase):
            trigger = generate_trigger(base_module, config, context, retriever)
        return _result(method, context, trigger, base_module, ledger, phase)

    if method == "warm-start" and base_module is None:
        return _not_applicable(method, context, "no s0 parameters to start from")

    initial = ([deepcopy(base_module.state_dict())] if method == "warm-start" else None)
    steps = adapt_config.steps if method == "warm-start" else config.steps
    directory = Path(output_dir) / context.snapshot_id / method
    with ledger.phase(phase):
        _, modules = train(
            workspace, [episode], config, output_dir=directory,
            contract=dict(contract, adapt_method=method,
                          adapt_snapshot=context.snapshot_id),
            resume=resume, contexts=[context], initial_state=initial, steps=steps,
        )
        trigger = generate_trigger(modules[0], config, context, retriever)
    return _result(method, context, trigger, modules[0], ledger, phase)


def _result(
    method: str, context: EpisodeContext, trigger: dict[str, Any],
    module: nn.Module | None, ledger: CostLedger, phase: str,
) -> AdaptResult:
    entry = ledger.phase_named(phase)
    return AdaptResult(
        method=method, episode_id=context.episode.episode_id,
        snapshot_id=context.snapshot_id, applicable=True, trigger=trigger,
        module=module,
        cost={} if entry is None else entry.to_json(),
    )


def adapt_all(
    methods: tuple[str, ...],
    *,
    workspace: Workspace,
    episode: Episode,
    context: EpisodeContext,
    config: TrainConfig,
    adapt_config: AdaptConfig,
    base_module: nn.Module | None,
    base_trigger: dict[str, Any] | None,
    output_dir: Path,
    contract: dict[str, Any],
    ledger: CostLedger | None = None,
    resume: bool = False,
) -> list[AdaptResult]:
    """Run every method on one snapshot, in the order given.

    VN — Chạy lần lượt mọi method trên cùng một snapshot, dùng chung một ledger.
    """
    ledger = ledger or CostLedger(device=str(workspace.retriever.device))
    return [
        adapt_trigger(
            method, workspace=workspace, episode=episode, context=context,
            config=config, adapt_config=adapt_config, base_module=base_module,
            base_trigger=base_trigger, output_dir=output_dir, contract=contract,
            ledger=ledger, resume=resume,
        )
        for method in methods
    ]


def fresh_module(config: TrainConfig, retriever: Retriever, episode: Episode) -> nn.Module:
    """A module of the run's own kind, for tests and for cost calibration.

    VN — Tạo một module trắng đúng loại mà run này đang dùng (generator hay
    logits matrix, tùy ``config.mode``). Dùng cho test và để đo chi phí nền.
    """
    return _logits_source(config, retriever, episode)
