"""Memory drift: the snapshot trajectory one episode's long-term memory follows.

VN — M0–M2 coi bộ nhớ là đứng yên: mỗi episode chỉ có một trạng thái ``s0``.
M3 cần bộ nhớ **trôi**, nên file này sinh ra một chuỗi snapshot cho mỗi episode:
phình thêm 25/50/100% tài liệu, hoặc ba biến thể ``mixture`` (thay tài liệu bằng
domain khác), ``deletion`` (xóa bớt), ``distractor`` (thêm tài liệu lành nhưng
rất giống câu hỏi).

Hai luật sống còn, ép ngay trong code:

1. Snapshot chỉ được lấy tài liệu **thuộc split của chính nó**. Lấy tài liệu
   train nhét vào bộ nhớ test là rò rỉ split, và mọi metric sẽ không thấy gì.
2. Không bao giờ chọn drift theo kết quả. Tài liệu thêm/bớt do RNG có seed quyết
   định, module này không nhìn thấy điểm số của method nào cả.

``distractor`` là ngoại lệ có chủ đích: nó cố tình chọn tài liệu giống query
nhất — nhưng chấm độ giống trên ``Q_sup``, không phải ``Q_eval``.

M0-M2 held the memory fixed, so an episode had exactly one snapshot (``-s0``).
M3 asks the question the whole amortization claim rests on: when the agent's
memory moves, is regenerating a trigger cheaper and as good as optimizing one
again?  Answering it needs more than one snapshot per episode, which is what
``build_trajectory`` produces.

Two rules decide whether these numbers mean anything, and both are enforced
here rather than left to reviewer discipline:

**A snapshot may only draw on its own split.**  ``build_trajectory`` takes the
family -> split assignment and refuses a candidate pool that reaches across it.
Growing test memory with training documents would leak the outer split through
the back door, and the leak would be invisible in every metric.

**Drift is never chosen by its effect.**  Which documents arrive, leave or get
replaced is decided by a seeded RNG keyed on ``(episode_id, kind, seed)``, and
nothing in this module can see a method's score.  Picking the documents that
happen to break a baseline is the easiest way to manufacture a result in M3, so
the selection deliberately has no access to one.

The one exception is ``distractor``, which *is* a targeted perturbation: it adds
the benign documents most similar to the episode's queries.  It takes those
similarities against ``Q_sup`` only.  Scoring them against ``Q_eval`` would make
the control an oracle over the very queries the trigger is graded on.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
from typing import Any, Sequence

import torch

from src.triggers.artifacts import stable_hash
from src.triggers.mcat.cache import select_rows
from src.triggers.mcat.episodes import Episode, family_key

KINDS = ("base", "growth", "mixture", "deletion", "distractor")
ABLATION_KINDS = ("mixture", "deletion", "distractor")
DEFAULT_GROWTH = (0.25, 0.50, 1.00)
MIXTURE_FRACTION = 0.30
DELETION_FRACTION = 0.25
DISTRACTOR_FRACTION = 0.25


@dataclass(frozen=True)
class Snapshot:
    """One state of an episode's memory, stored as an absolute document list.

    VN — Một trạng thái bộ nhớ. ``doc_ids`` lưu **danh sách đầy đủ**, không lưu
    phần chênh lệch, để đọc snapshot không cần biết lịch sử và hash luôn ổn định.

    Absolute, not a delta: the runtime resolves a snapshot without knowing its
    history, and the hash of a trajectory stays stable however it was built.
    """

    snapshot_id: str
    episode_id: str
    domain: str
    split: str
    kind: str
    growth: float
    doc_ids: list[str]
    parent_snapshot_id: str | None
    note: dict[str, Any]

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")
        if not self.doc_ids:
            raise ValueError(f"{self.snapshot_id}: a snapshot needs at least one document")
        if len(set(self.doc_ids)) != len(self.doc_ids):
            raise ValueError(f"{self.snapshot_id}: duplicate document ids")

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def base_snapshot(episode: Episode) -> Snapshot:
    """The ``-s0`` state: the memory the episode was built with.

    VN — Trạng thái gốc ``s0``: đúng bộ nhớ mà episode được dựng lên. Mốc so sánh
    khi chưa có drift nào.
    """
    return Snapshot(
        snapshot_id=episode.snapshot_id, episode_id=episode.episode_id,
        domain=episode.domain, split=episode.split, kind="base", growth=0.0,
        doc_ids=sorted(episode.doc_ids), parent_snapshot_id=None,
        note={"documents": len(episode.doc_ids)},
    )


def _rng(episode_id: str, kind: str, seed: int) -> random.Random:
    """Seeded exactly like ``episodes.build_episodes``: hash, then first 16 hex.

    VN — RNG có seed cố định theo (episode, loại drift, seed), nên cùng seed luôn
    cho cùng kết quả và không ai chọn được drift theo ý mình.
    """
    return random.Random(stable_hash([episode_id, kind, seed])[:16])


def _check_pool(
    rows: Sequence[dict[str, Any]], assignment: dict[str, str], split: str, label: str
) -> None:
    """Fail loudly when a candidate document belongs to another split.

    VN — Thấy tài liệu thuộc split khác thì báo lỗi ngay, không cảnh báo rồi chạy tiếp.
    """
    for row in rows:
        key = family_key(row["family"])
        if key not in assignment:
            raise ValueError(
                f"{label}: family {key!r} of document {row['doc_id']!r} is absent from "
                "the split assignment; the pool was built from different rows"
            )
        if assignment[key] != split:
            raise ValueError(
                f"{label}: document {row['doc_id']!r} belongs to split "
                f"{assignment[key]!r}, not {split!r}; a snapshot may not reach across "
                "the outer split"
            )


def support_similarity(
    vectors: torch.Tensor,
    order: Sequence[str],
    support: torch.Tensor,
    candidates: Sequence[str],
) -> dict[str, float]:
    """Dot product of each candidate document with the mean support query.

    VN — Chấm điểm "giống câu hỏi" cho từng tài liệu ứng viên, bằng tích vô hướng
    với trung bình vector của ``Q_sup``. Dùng để chọn distractor. Chỉ ``Q_sup``,
    tuyệt đối không dùng ``Q_eval``.

    ``support`` is ``Q_sup`` and nothing else.  The runtime scorer ranks raw dot
    products (``index/manifest.json`` records ``"normalized": false``), so this
    uses the same quantity the retrieval metrics do.
    """
    if not candidates:
        return {}
    rows = select_rows(vectors, order, candidates).to(support.device).float()
    centroid = support.float().mean(dim=0)
    return {identifier: float(score)
            for identifier, score in zip(candidates, rows @ centroid)}


def build_trajectory(
    episode: Episode,
    pool: Sequence[dict[str, Any]],
    *,
    assignment: dict[str, str],
    mixture_pool: Sequence[dict[str, Any]] = (),
    distractor_scores: dict[str, float] | None = None,
    growth: Sequence[float] = DEFAULT_GROWTH,
    ablations: Sequence[str] = ABLATION_KINDS,
    seed: int = 0,
) -> tuple[list[Snapshot], dict[str, Any]]:
    """Build ``s0`` plus one snapshot per growth level and per ablation.

    VN — Dựng cả chuỗi snapshot cho một episode: ``s0``, các mức phình, rồi các
    biến thể ablation. ``pool`` là tài liệu cùng domain, ``mixture_pool`` là tài
    liệu domain khác — cả hai đều bị kiểm tra split trước khi lấy mẫu. Cái nào
    không dựng được (hết tài liệu, run chỉ có một domain, chưa có điểm similarity)
    thì ghi lý do vào report trả về, chứ không biến mất im lặng.

    ``pool`` holds this domain's documents; ``mixture_pool`` holds another
    domain's.  Both are checked against ``assignment`` before anything is
    sampled.  Anything that cannot be built -- no spare documents, a single
    domain in the run, no support similarities -- is recorded in the returned
    report with a reason instead of being skipped silently.
    """
    for value in growth:
        if value <= 0:
            raise ValueError(f"growth levels must be positive, got {value}")
    unknown = sorted(set(ablations) - set(ABLATION_KINDS))
    if unknown:
        raise ValueError(f"unknown ablations {unknown}; choose from {ABLATION_KINDS}")

    _check_pool(pool, assignment, episode.split, "pool")
    _check_pool(mixture_pool, assignment, episode.split, "mixture_pool")

    base = sorted(episode.doc_ids)
    held = set(base)
    available = sorted({row["doc_id"] for row in pool} - held)
    foreign = sorted({row["doc_id"] for row in mixture_pool} - held)

    snapshots = [base_snapshot(episode)]
    report: dict[str, Any] = {
        "base_documents": len(base), "pool": len(available),
        "mixture_pool": len(foreign), "skipped": {},
    }

    def emit(kind: str, doc_ids: list[str], note: dict[str, Any], growth_value: float) -> None:
        snapshots.append(Snapshot(
            snapshot_id=f"{episode.episode_id}-{kind}", episode_id=episode.episode_id,
            domain=episode.domain, split=episode.split, kind=kind.split("-")[0],
            growth=growth_value, doc_ids=sorted(doc_ids),
            parent_snapshot_id=episode.snapshot_id, note=note,
        ))

    for value in growth:
        label = f"growth-{int(round(value * 100))}"
        wanted = max(1, int(round(value * len(base))))
        if not available:
            report["skipped"][label] = {"reason": "no spare documents in this split"}
            continue
        taken = min(wanted, len(available))
        added = _rng(episode.episode_id, label, seed).sample(available, taken)
        emit(label, base + added,
             {"requested": wanted, "added": taken, "truncated": taken < wanted,
              "pool": len(available)}, value)

    if "mixture" in ablations:
        wanted = max(1, int(round(MIXTURE_FRACTION * len(base))))
        if not foreign:
            report["skipped"]["mixture"] = {
                "reason": "needs documents from another domain in the same split"}
        elif len(base) <= 1:
            report["skipped"]["mixture"] = {"reason": "memory is too small to replace"}
        else:
            rng = _rng(episode.episode_id, "mixture", seed)
            removed = rng.sample(base, min(wanted, len(base) - 1))
            added = rng.sample(foreign, min(wanted, len(foreign)))
            emit("mixture", [d for d in base if d not in set(removed)] + added,
                 {"removed": len(removed), "added": len(added),
                  "fraction": MIXTURE_FRACTION}, 0.0)

    if "deletion" in ablations:
        wanted = max(1, int(DELETION_FRACTION * len(base)))
        if len(base) <= 1:
            report["skipped"]["deletion"] = {"reason": "memory is too small to delete from"}
        else:
            removed = _rng(episode.episode_id, "deletion", seed).sample(
                base, min(wanted, len(base) - 1))
            emit("deletion", [d for d in base if d not in set(removed)],
                 {"removed": len(removed), "fraction": DELETION_FRACTION}, 0.0)

    if "distractor" in ablations:
        scored = {key: value for key, value in (distractor_scores or {}).items()
                  if key in set(available)}
        wanted = max(1, math.ceil(DISTRACTOR_FRACTION * len(base)))
        if not scored:
            report["skipped"]["distractor"] = {
                "reason": "no support-query similarities for the candidate pool"}
        else:
            # Ties broken by id, never by anything a method's score could touch.
            ranked = sorted(scored, key=lambda key: (-scored[key], key))[:wanted]
            emit("distractor", base + ranked,
                 {"added": len(ranked), "requested": wanted,
                  "scored_candidates": len(scored), "selected_on": "support_queries"}, 0.0)

    return snapshots, report


def trajectory_hash(snapshots: Sequence[Snapshot]) -> str:
    return stable_hash([snapshot.to_json() for snapshot in snapshots])


def assert_no_snapshot_leak(
    snapshots: Sequence[Snapshot], family_of: dict[str, str], assignment: dict[str, str]
) -> None:
    """Independent audit: every document of every snapshot owns that split.

    VN — Kiểm tra lại lần nữa trên trajectory đã ghi ra đĩa: mọi tài liệu của mọi
    snapshot phải thuộc đúng split của snapshot đó. ``build_trajectory`` đã chặn
    rồi, nhưng hàm này còn sống sót khi ai đó sửa lại builder.

    ``build_trajectory`` already refuses a bad pool.  This is the check that
    survives a refactor of the builder, so run it over the trajectory that was
    actually written to disk.
    """
    for snapshot in snapshots:
        for doc_id in snapshot.doc_ids:
            if doc_id not in family_of:
                raise ValueError(f"{snapshot.snapshot_id}: unknown document {doc_id!r}")
            key = family_key(family_of[doc_id])
            if assignment.get(key) != snapshot.split:
                raise ValueError(
                    f"{snapshot.snapshot_id}: document {doc_id!r} maps to split "
                    f"{assignment.get(key)!r}, not {snapshot.split!r}"
                )
