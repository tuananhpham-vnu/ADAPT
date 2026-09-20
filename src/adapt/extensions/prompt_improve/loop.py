"""Vòng lặp cải tiến prompt: tham lam, nhiều vòng, dừng khi hết cải thiện.

Xem `_guidance/12_stage1_prompt_improvement.md` mục 4. Nối các module còn lại trong package này:

    phân loại lỗi (failure_classifier) -> sinh bản vá (patch_generator)
        -> chạy lại + đo held-out (holdout) -> lặp

Dừng khi `delta_Q_heldout` nhỏ hơn ngưỡng hoặc hết ngân sách vòng. Lưu lịch sử bản vá để tránh lặp
qua lặp lại giữa hai phiên bản prompt (paper gọi đây là cycling).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.adapt.extensions.prompt_improve.holdout import HoldoutReport
from src.adapt.extensions.prompt_improve.patch_generator import Patch


@dataclass
class ImproveLoopResult:
    final_prompt: str
    patches_applied: list[Patch] = field(default_factory=list)
    history: list[HoldoutReport] = field(default_factory=list)


def run(
    system_prompt: str,
    max_rounds: int = 5,
    min_delta_q_heldout: float = 0.05,
    *, evaluator=None, propose=None, split=None,
) -> ImproveLoopResult:
    """Chạy vòng lặp cải tiến prompt cho tới khi hết cải thiện hoặc hết `max_rounds`.

    Yêu cầu Stage 0 đã vendor xong `src/artemis/` để chạy lại phase 1 lấy điểm ban đầu - xem
    `_guidance/11_stage0_setup.md`.
    """
    from src.adapt.extensions.prompt_improve.holdout import evaluate
    from src.adapt.extensions.prompt_improve.patch_generator import apply_patches
    if evaluator is None or propose is None or split is None:
        raise ValueError("Provide evaluator(prompt)->scores, propose(prompt, seen_scores)->patches and a frozen split")
    if max_rounds < 0:
        raise ValueError("max_rounds must be nonnegative")
    result = ImproveLoopResult(system_prompt)
    before = evaluator(system_prompt)
    visited = {system_prompt}
    for _ in range(max_rounds):
        patches = propose(result.final_prompt,{k:before[k] for k in split.seen_test_case_ids})
        if not patches:
            break
        candidate = apply_patches(result.final_prompt,patches)
        if candidate in visited:
            break
        visited.add(candidate)
        after = evaluator(candidate)
        report = evaluate(before,after,split)
        result.history.append(report)
        if report.delta_q_heldout <= min_delta_q_heldout or report.non_regression_rate < 1:
            break
        result.final_prompt = candidate
        result.patches_applied.extend(patches)
        before = after
    return result
