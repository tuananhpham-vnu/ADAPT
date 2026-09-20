"""What a trigger costs: counters, phase timings and the amortization point.

VN — Nửa câu hỏi của M3 là chất lượng, nửa còn lại là **giá**. Generator có thể
ngang ngửa việc tối ưu lại từng episode, nhưng nếu tiền train còn đắt hơn phần
nó tiết kiệm được thì claim amortization sụp đổ. Muốn biết thì phải đếm.

Cách đếm: một ``ContextVar`` giữ ledger đang mở, chỗ nào cần tính tiền thì gọi
``record(...)``. Không có ledger nào mở thì ``record`` không làm gì cả — đo đạc
không bao giờ được phép đổi kết quả của run.

Hai tổng phải báo riêng: **deployment-only** (một snapshot mới tốn bao nhiêu,
chưa tính train) và **lifecycle** (``C_train + C_prep + N * C_online``). Chỉ nêu
cái đầu thì method nào cũng trông như miễn phí; chỉ nêu cái sau thì giấu mất
việc tiền train chỉ trả một lần.

Half of M3's question is quality under drift; the other half is price.  A
generator that matches a per-episode search but costs more to train than it
ever saves has not earned the claim, and the only way to know is to count.

The counters live in a ``ContextVar`` rather than being threaded through every
signature.  ``encoding.py`` calls :func:`record` at the one place the retriever
is actually invoked, so an encode is counted wherever it happens -- training,
adaptation or evaluation -- and code that runs without a ledger active pays
nothing and behaves exactly as before.

Two totals have to be reported separately (``_idea/`` section 11.2):

``deployment-only``  what one new snapshot costs, training excluded.
``lifecycle``        ``C_train + C_prep + N * C_online``.

Quoting only the first makes any amortized method look free; quoting only the
second hides that the training cost is paid once.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import time
from typing import Any, Callable, Iterator, Sequence

import torch

COUNTERS = (
    "encoder_forward",     # rows pushed through the retriever
    "encoder_backward",    # backward passes that traverse the retriever
    "optimizer_steps",
    "index_writes",        # poison records written; 0 under the fixed policy
    "gmm_refits",          # fit_centers calls for a new snapshot
    "scorer_calls",        # GPT-2 / Llama feasibility calls; 0 in M3 by default
)

_ACTIVE: ContextVar["CostLedger | None"] = ContextVar("mcat_cost_ledger", default=None)


def current() -> "CostLedger | None":
    """The ledger this call stack is being charged to, if any."""
    return _ACTIVE.get()


def record(name: str, amount: float = 1) -> None:
    """Charge ``amount`` to ``name`` on every open ledger; a no-op without one.

    VN — Tính ``amount`` vào counter ``name``, cho **mọi** ledger đang mở chứ
    không chỉ ledger trong cùng. Nếu một stage mở ledger riêng mà nuốt luôn chi
    phí, ledger bao ngoài sẽ báo "stage này tốn 0" — sai một cách khó thấy.
    Không có ledger nào thì hàm này im lặng không làm gì.

    The charge walks up the stack of ledgers that are currently active, so a
    stage that opens its own ledger still reports its work to whatever opened
    one around it.  A ledger that captured charges from its callers instead of
    sharing them would silently zero out the enclosing accounting -- which is
    exactly what "the training stage costs nothing" would look like.

    Deliberately silent when no ledger is active: instrumentation must never
    change what a run computes, only what is known about it.
    """
    if name not in COUNTERS:
        raise KeyError(f"unknown counter {name!r}; known counters are {COUNTERS}")
    ledger = _ACTIVE.get()
    if ledger is not None:
        for entry in ledger.chain():
            entry.counters[name] += amount


@dataclass
class Phase:
    """One accounted stretch of work: what it cost and how long it took.

    VN — Một đoạn công việc đã được tính tiền: tốn bao nhiêu và mất bao lâu.
    """

    name: str
    wall_seconds: float
    counters: dict[str, float]
    peak_vram_bytes: int
    vram_scope: str          # "phase" for an outermost phase, "run" when nested

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name, "wall_seconds": self.wall_seconds,
                "counters": dict(self.counters), "peak_vram_bytes": self.peak_vram_bytes,
                "vram_scope": self.vram_scope}


@dataclass
class CostLedger:
    """Counters, phases and online latency samples for one run or one arm.

    VN — Sổ chi tiêu của một run hoặc một arm: các counter, các đoạn ``phase``
    đã đo, và mẫu latency online.
    """

    device: str | None = None
    counters: dict[str, float] = field(
        default_factory=lambda: {name: 0.0 for name in COUNTERS})
    phases: list[Phase] = field(default_factory=list)
    online_latency_seconds: list[float] = field(default_factory=list)
    _depth: int = 0
    _enclosing: list["CostLedger | None"] = field(default_factory=list)

    @contextmanager
    def active(self) -> Iterator["CostLedger"]:
        """Charge everything inside this block to this ledger and its callers.

        VN — Mọi thứ chạy trong khối ``with`` này được tính vào sổ này, và cả
        các sổ đang bao bên ngoài.
        """
        self._enclosing.append(_ACTIVE.get())
        token = _ACTIVE.set(self)
        try:
            yield self
        finally:
            _ACTIVE.reset(token)
            self._enclosing.pop()

    def chain(self) -> list["CostLedger"]:
        """This ledger and the ones enclosing it, each appearing once.

        VN — Danh sách sổ cần tính tiền: sổ này cộng các sổ bao ngoài, mỗi sổ
        đúng một lần. Sổ mở lồng trong chính nó không bị tính hai lần.

        A ledger opened inside itself (``phase`` within ``active``) must not be
        charged twice, so the walk stops at the first ledger it has seen.
        """
        seen: list[CostLedger] = []
        ledger: CostLedger | None = self
        while ledger is not None and not any(entry is ledger for entry in seen):
            seen.append(ledger)
            ledger = ledger._enclosing[-1] if ledger._enclosing else None
        return seen

    @contextmanager
    def phase(self, name: str) -> Iterator["CostLedger"]:
        """Time a stretch of work and attribute the counters it moved.

        VN — Đo một đoạn công việc: thời gian chạy và phần counter mà nó làm
        tăng. Phase lồng nhau thì phase ngoài **bao gồm** phase trong — đúng với
        nghĩa "chi phí của cả quá trình train".

        Nested phases are allowed and each one keeps its own delta, so an outer
        phase's counters include its inner phases -- which is what "the cost of
        training" should mean.
        """
        before = dict(self.counters)
        outermost = self._depth == 0
        if outermost:
            self._reset_peak_memory()
        self._depth += 1
        started = time.perf_counter()
        try:
            with self.active():
                yield self
        finally:
            self._depth -= 1
            elapsed = time.perf_counter() - started
            self.phases.append(Phase(
                name=name, wall_seconds=elapsed,
                counters={key: self.counters[key] - before[key] for key in COUNTERS},
                peak_vram_bytes=self._peak_memory(),
                # A nested phase cannot reset the peak without destroying its
                # parent's measurement, so its number is the run's peak, not
                # its own.  Saying so beats quietly reporting the wrong scope.
                vram_scope="phase" if outermost else "run",
            ))

    @contextmanager
    def online(self) -> Iterator[None]:
        """Time one online call, synchronizing so a GPU number means something.

        VN — Đo một lần gọi online, có đồng bộ GPU trước và sau; không đồng bộ
        thì con số trên GPU vô nghĩa vì lệnh còn đang chạy bất đồng bộ.
        """
        self._synchronize()
        started = time.perf_counter()
        try:
            yield
        finally:
            self._synchronize()
            self.online_latency_seconds.append(time.perf_counter() - started)

    def phase_named(self, name: str) -> Phase | None:
        for entry in reversed(self.phases):
            if entry.name == name:
                return entry
        return None

    def to_json(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "counters": dict(self.counters),
            "phases": [entry.to_json() for entry in self.phases],
            "online_latency": summarize_latency(self.online_latency_seconds),
            "online_samples": list(self.online_latency_seconds),
        }

    def _cuda(self) -> bool:
        return bool(self.device) and str(self.device).startswith("cuda") \
            and torch.cuda.is_available()

    def _synchronize(self) -> None:
        if self._cuda():
            torch.cuda.synchronize(self.device)

    def _reset_peak_memory(self) -> None:
        if self._cuda():
            torch.cuda.reset_peak_memory_stats(self.device)

    def _peak_memory(self) -> int:
        return int(torch.cuda.max_memory_allocated(self.device)) if self._cuda() else 0


def measure_online(
    ledger: CostLedger,
    call: Callable[[], Any],
    *,
    repeats: int = 5,
    warmup: int = 3,
) -> Any:
    """Run ``call`` ``warmup`` times unmeasured, then ``repeats`` times on the clock.

    VN — Chạy ``warmup`` lần bỏ đi rồi mới đo ``repeats`` lần. Mấy lần đầu luôn
    chậm bất thường (khởi tạo CUDA context, cấp phát lần đầu, cache lạnh); bỏ
    chúng đi chứ không tính trung bình vào.

    The discarded runs are what keep a lazy CUDA context, a first-touch
    allocation or a cold cache out of the p50.  They are discarded, not
    averaged in, and the samples they produced never reach the ledger.
    """
    if repeats < 1 or warmup < 0:
        raise ValueError("repeats must be >= 1 and warmup >= 0")
    result = None
    for _ in range(warmup):
        result = call()
    for _ in range(repeats):
        with ledger.online():
            result = call()
    return result


def percentile(values: Sequence[float], fraction: float) -> float | None:
    """Nearest-rank percentile; ``None`` for an empty sample, never 0.0.

    VN — Percentile kiểu nearest-rank. Không có mẫu nào thì trả ``None``, không
    trả 0.0: "chưa đo" và "đo được 0" là hai chuyện khác nhau.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")
    ordered = sorted(values)
    if not ordered:
        return None
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return float(ordered[index])


def summarize_latency(values: Sequence[float]) -> dict[str, Any]:
    """p50 and p95, not the mean: a long tail is the thing worth seeing.

    VN — Báo p50 và p95 chứ không báo trung bình: cái đáng nhìn là cái đuôi dài.
    """
    return {"samples": len(values), "p50": percentile(values, 0.50),
            "p95": percentile(values, 0.95)}


def break_even(train_cost: float, online_search: float, online_gen: float) -> float | None:
    """``N* = C_train / (C_online_search - C_online_gen)``, or ``None``.

    VN — Điểm hòa vốn: cần bao nhiêu snapshot thì tiền train mới được bù lại.
    Trả ``None`` khi mẫu số không dương, tức là sinh online **không** rẻ hơn tìm
    kiếm — khi đó không có điểm hòa vốn nào cả. Trả số âm rồi để ai đó đọc thành
    "hòa vốn ngay lập tức" chính là hiểu nhầm mà hàm này đang chặn.

    ``None`` whenever the denominator is not positive: if generating online is
    no cheaper than searching, there is no episode count at which training pays
    for itself.  Returning a negative number here and reading it as "breaks even
    immediately" is exactly the misreading this guard exists to prevent.
    """
    denominator = online_search - online_gen
    return train_cost / denominator if denominator > 0 else None


def lifecycle_cost(
    train_cost: float, prepare_cost: float, online_cost: float, episodes: int
) -> float:
    """``C_train + C_prep + N * C_online`` for one method over ``episodes`` uses.

    VN — Tổng chi phí trọn đời của một method khi dùng cho ``episodes`` lần.
    """
    if episodes < 0:
        raise ValueError("episodes must not be negative")
    return train_cost + prepare_cost + online_cost * episodes


def amortization(
    *,
    train_cost: float,
    online_search: float,
    online_gen: float,
    quality_gap: float | None = None,
    quality_tolerance: float = 0.0,
    episodes: int | None = None,
) -> dict[str, Any]:
    """The break-even point together with the quality it was bought at.

    VN — Điểm hòa vốn đi kèm **chất lượng đã phải đánh đổi**. ``quality_gap`` là
    hiệu metric giữa generator và baseline trên cùng tập episode. Nhanh hơn mà
    kém hơn thì không phải speedup, nên nếu gap vượt ``quality_tolerance``, kết
    quả bị đánh dấu ``comparable: false`` kèm lý do — không im lặng cho qua.

    ``quality_gap`` is the generator's metric minus the search baseline's on the
    same episodes.  A break-even point computed across a quality gap is not a
    speedup, so it is reported next to the gap and marked ``comparable: false``
    when the gap exceeds ``quality_tolerance`` -- never silently.
    """
    point = break_even(train_cost, online_search, online_gen)
    comparable = quality_gap is not None and abs(quality_gap) <= quality_tolerance
    result: dict[str, Any] = {
        "break_even_episodes": point,
        "online_search": online_search,
        "online_gen": online_gen,
        "train_cost": train_cost,
        "quality_gap": quality_gap,
        "quality_tolerance": quality_tolerance,
        "comparable": comparable,
    }
    if point is None:
        result["reason"] = "generating online is not cheaper than searching"
    elif quality_gap is None:
        result["reason"] = "no quality gap supplied; the point is not a speedup claim"
    elif not comparable:
        result["reason"] = (
            f"quality gap {quality_gap:+.4f} exceeds the tolerance "
            f"{quality_tolerance:.4f}; compare at equal quality before quoting N*"
        )
    if episodes is not None:
        result["episodes_expected"] = episodes
        result["pays_off"] = bool(point is not None and comparable and episodes >= point)
    return result
