"""Phân loại lỗi theo taxonomy đã kiểm chứng trên 3.519 lỗi của paper ARTEMIS.

Xem `_guidance/12_stage1_prompt_improvement.md` mục 1. Bốn nhãn:

- `ac-refuse` (37.1%) — user query dụ agent phá absolute constraint và agent nghe theo.
- `task-infer` (36.3%) — prompt mơ hồ, agent không suy ra được yêu cầu ngầm.
- `task-exec` (22.3%) — response thiếu, không đầy đủ.
- `other` (4.3%).

Kiểm tra tỉnh táo khi có dữ liệu thật: so phân phối nhãn phân loại được với 37/36/22/4 - lệch nhiều
gần như chắc chắn bộ phân loại sai chứ không phải tìm ra hiện tượng mới.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FailureLabel(str, Enum):
    AC_REFUSE = "ac-refuse"
    TASK_INFER = "task-infer"
    TASK_EXEC = "task-exec"
    OTHER = "other"


@dataclass
class FailureCase:
    """Một criterion bị trượt, đủ ngữ cảnh để phân loại và sau đó sinh bản vá."""

    user_query: str
    response: str
    criterion: str
    judge_explanation: str


def classify(case: FailureCase) -> FailureLabel:
    """Gán nhãn taxonomy cho một `FailureCase`.

    Đầu vào là bộ bốn (user_query, response, criterion, judge_explanation) - judge của upstream
    ARTEMIS đã sinh sẵn `judge_explanation` nên không cần gọi thêm model để đoán lại lý do.
    """
    text = (case.criterion + " " + case.judge_explanation).casefold()
    if any(x in text for x in ("authorization", "must not", "forbidden", "constraint", "không được", "cấm")):
        return FailureLabel.AC_REFUSE
    if any(x in text for x in ("ambiguous", "infer", "wrong tool", "intent", "mơ hồ", "ý định")):
        return FailureLabel.TASK_INFER
    if any(x in text for x in ("missing", "incomplete", "schema", "format", "thiếu", "định dạng")):
        return FailureLabel.TASK_EXEC
    return FailureLabel.OTHER


def label_distribution(cases: list[FailureCase]) -> dict[FailureLabel, float]:
    """Tính phân phối nhãn trên một tập case - dùng để đối chiếu với 37/36/22/4 của paper."""
    from collections import Counter
    counts = Counter(classify(c) for c in cases)
    return {label: counts[label]/len(cases) if cases else 0.0 for label in FailureLabel}
