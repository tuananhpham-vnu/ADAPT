"""Chống overfit prompt - phần quyết định vòng cải tiến có ý nghĩa hay không.

Xem `_guidance/12_stage1_prompt_improvement.md` mục 3. Đây là phần dễ làm sai nhất: sửa prompt để
vượt qua đúng bộ test case đã dùng để chẩn đoán nó thì `delta_Q` tăng đẹp nhưng không chứng minh
được gì.

Quy tắc bắt buộc:
- Chia test case thành **seen** (dùng để sinh bản vá) và **held-out** (chỉ dùng để đo).
- Đo lại held-out bằng bộ test case cố định (`--load-test-cases-from` phía upstream) - không để
  pipeline tự sinh lại test case cho prompt mới, vì prompt mới sẽ phân rã ra requirement khác.
- Báo cáo **cả hai** `delta_Q_seen` và `delta_Q_heldout`. Chỉ số held-out mới là bằng chứng; số seen
  dùng để đo mức overfit.
- Chấm held-out bằng judge thứ hai khác model với judge dùng ở vòng vá.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HoldoutSplit:
    seen_test_case_ids: list[str]
    heldout_test_case_ids: list[str]


def split(test_case_ids: list[str], heldout_ratio: float = 0.3, seed: int = 0) -> HoldoutSplit:
    """Chia test case thành seen/held-out. `seed` cố định để tái lập được kết quả."""
    import random
    if not 0 < heldout_ratio < 1 or len(test_case_ids) < 2 or len(set(test_case_ids)) != len(test_case_ids):
        raise ValueError("Need unique IDs, at least two cases, and ratio in (0,1)")
    ids = sorted(test_case_ids)
    random.Random(seed).shuffle(ids)
    n = max(1, min(len(ids)-1, round(len(ids)*heldout_ratio)))
    return HoldoutSplit(ids[n:], ids[:n])


@dataclass
class HoldoutReport:
    delta_q_seen: float
    delta_q_heldout: float
    delta_s: float
    non_regression_rate: float


def evaluate(before_scores: dict, after_scores: dict, split: HoldoutSplit) -> HoldoutReport:
    """Tính delta_Q_seen, delta_Q_heldout, delta_S và tỉ lệ không hồi quy.

    `before_scores`/`after_scores` là điểm theo test case id, chấm bằng judge cố định trên đúng
    một bộ test case đứng yên (không sinh lại) - xem ràng buộc ở docstring module.
    """
    import math
    import statistics
    ids = split.seen_test_case_ids + split.heldout_test_case_ids
    if len(set(ids)) != len(ids) or not split.seen_test_case_ids or not split.heldout_test_case_ids:
        raise ValueError("Splits must be nonempty and disjoint")
    if set(before_scores) != set(ids) or set(after_scores) != set(ids):
        raise ValueError("Scores must cover the exact frozen case set")
    def runs(value):
        values = value if isinstance(value,list) else [value]
        if not values or any(type(v) not in (int,float) or not math.isfinite(v) for v in values):
            raise ValueError("Scores must be finite numeric values")
        return values
    a,b = {k:runs(v) for k,v in before_scores.items()},{k:runs(v) for k,v in after_scores.items()}
    if any(len(a[k]) != len(b[k]) for k in ids):
        raise ValueError("Matched repeat counts required")
    delta = lambda keys: statistics.mean(statistics.mean(b[k])-statistics.mean(a[k]) for k in keys)
    delta_s = statistics.mean(statistics.pstdev(b[k])-statistics.pstdev(a[k]) for k in ids)
    return HoldoutReport(delta(split.seen_test_case_ids),delta(split.heldout_test_case_ids),delta_s,
                         sum(statistics.mean(b[k])>=statistics.mean(a[k]) for k in ids)/len(ids))
