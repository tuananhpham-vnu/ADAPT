"""Paired comparisons with episode-level uncertainty.

VN — Hai method luôn được đo trên **cùng** tập episode, nên đại lượng có ý nghĩa
là hiệu từng cặp, không phải khoảng cách giữa hai trung bình rời rạc.

Đơn vị lấy mẫu lại (bootstrap) là **episode**, không phải query. Các query trong
cùng một episode dùng chung một snapshot và một trigger; coi chúng là độc lập sẽ
bóp khoảng tin cậy nhỏ đi cỡ căn bậc hai số query mỗi episode, và biến nhiễu do
seed thành "kết quả". Đó là pseudo-replication — cách dễ nhất để tự bịa ra ý
nghĩa thống kê ở đây.

Two methods are always measured on the *same* episodes, so the quantity with
any power is the paired difference, not the gap between two independent means.

The resampling unit is the episode (or its family / trajectory), never the
query.  Queries inside one episode share a memory snapshot and a trigger;
treating them as independent samples would shrink every interval by roughly the
square root of the number of queries per episode and turn seed noise into a
result.  That is pseudo-replication, and it is the single easiest way to
manufacture significance here.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def paired_bootstrap(
    left: Sequence[float],
    right: Sequence[float],
    groups: Sequence[str],
    *,
    iterations: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Bootstrap the mean of ``left - right``, resampling ``groups``.

    VN — Bootstrap trung bình của hiệu ``left - right``, lấy mẫu lại theo
    ``groups``. ``groups`` là đơn vị độc lập của mỗi quan sát (thường là
    episode). Bốc cả nhóm chứ không bốc từng dòng, nên tương quan trong nhóm
    được giữ nguyên thay vì bị san phẳng.

    ``groups`` names the independent unit each observation belongs to.  Groups
    are drawn with replacement and every observation of a drawn group comes
    along, so within-group correlation is preserved instead of being averaged
    away.
    """
    if not (len(left) == len(right) == len(groups)):
        raise ValueError(
            f"paired inputs must have equal length, got {len(left)}, {len(right)}, "
            f"{len(groups)}"
        )
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    if not left:
        return {"mean_difference": None, "ci_low": None, "ci_high": None,
                "observations": 0, "groups": 0, "iterations": iterations,
                "reason": "no paired observations"}

    differences = np.asarray(left, dtype="float64") - np.asarray(right, dtype="float64")
    names = sorted(set(groups))
    index = {name: np.asarray([i for i, g in enumerate(groups) if g == name])
             for name in names}
    observed = float(differences.mean())
    if len(names) < 2:
        # One group cannot bound anything: reporting a zero-width interval here
        # would claim a precision the design does not have.
        return {"mean_difference": observed, "ci_low": None, "ci_high": None,
                "observations": len(differences), "groups": len(names),
                "iterations": iterations,
                "reason": "at least two independent groups are needed for an interval"}

    generator = np.random.default_rng(seed)
    draws = generator.integers(0, len(names), size=(iterations, len(names)))
    samples = np.empty(iterations, dtype="float64")
    for step in range(iterations):
        picked = np.concatenate([index[names[position]] for position in draws[step]])
        samples[step] = differences[picked].mean()
    low, high = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return {"mean_difference": observed, "ci_low": float(low), "ci_high": float(high),
            "observations": int(len(differences)), "groups": len(names),
            "iterations": iterations,
            "significant": bool(low > 0 or high < 0),
            "p_value": sign_flip_test(differences, groups, seed=seed)}


#: Up to this many groups the sign-flip test enumerates every assignment.
EXACT_GROUPS = 20


def sign_flip_test(
    differences: Sequence[float],
    groups: Sequence[str],
    *,
    samples: int = 100_000,
    seed: int = 0,
) -> float | None:
    """Two-sided p-value that the paired differences are centred on zero.

    VN — Kiểm định hoán vị đổi dấu, ghép cặp theo nhóm (episode). Giả thuyết
    không: hai điều kiện hoán đổi được trong từng episode, nên đổi dấu toàn bộ
    hiệu của một episode không làm thay đổi phân phối. p = tỉ lệ các phép đổi dấu
    cho |trung bình| ≥ giá trị quan sát. Với ≤ 20 episode thì liệt kê đủ 2^G phép
    đổi dấu (chính xác, không phụ thuộc seed); nhiều hơn thì lấy mẫu Monte Carlo.

    Under the null the two conditions are exchangeable inside each group, so
    flipping the sign of every difference of a group leaves the distribution
    unchanged.  The unit flipped is the group, never the observation, for the
    reason the module docstring gives.  The observed assignment is one of those
    counted, so p is never below ``1 / 2**G``.
    """
    values = np.asarray(differences, dtype="float64")
    if len(values) != len(groups):
        raise ValueError("differences and groups must have equal length")
    names = sorted(set(groups))
    if len(names) < 2:
        return None
    totals = np.asarray([values[[i for i, g in enumerate(groups) if g == name]].sum()
                         for name in names])
    observed = abs(totals.sum()) / len(values)
    if len(names) <= EXACT_GROUPS:
        codes = np.arange(2 ** len(names), dtype=np.int64)[:, None]
        signs = 1 - 2 * ((codes >> np.arange(len(names))) & 1)
    else:
        generator = np.random.default_rng(seed)
        signs = generator.choice((-1, 1), size=(samples, len(names)))
        signs[0] = 1
    flipped = np.abs(signs @ totals) / len(values)
    # A relative tolerance: sums of floats that are equal in exact arithmetic
    # must count as "at least as extreme".
    return float(np.mean(flipped >= observed - 1e-12 * max(1.0, observed)))


def holm(p_values: dict[str, float | None]) -> dict[str, float | None]:
    """Holm step-down adjustment over one family of tests (FWER).

    VN — Hiệu chỉnh Holm cho một họ kiểm định: p đã hiệu chỉnh của kiểm định thứ
    i (xếp tăng dần) là max_{j<=i} min(1, (m-j+1)·p_j). Kiểm định không có p
    (None) thì không tính vào m.
    """
    present = sorted((value, name) for name, value in p_values.items() if value is not None)
    count = len(present)
    adjusted: dict[str, float | None] = {name: None for name in p_values}
    running = 0.0
    for position, (value, name) in enumerate(present):
        running = max(running, min(1.0, (count - position) * value))
        adjusted[name] = running
    return adjusted


def macro_micro_worst(
    values: Sequence[float], weights: Sequence[float], groups: Sequence[str]
) -> dict[str, Any]:
    """Three aggregates of one metric, because they can disagree.

    VN — Ba cách tổng hợp cùng một metric, vì chúng có thể mâu thuẫn nhau:
    ``macro`` trung bình theo episode (mỗi episode một phiếu), ``micro`` có trọng
    số theo số query, ``worst`` là episode tệ nhất — thứ mà trung bình sinh ra để
    che đi.

    ``macro``  unweighted mean over episodes -- every episode counts once.
    ``micro``  weighted by the number of queries behind each episode.
    ``worst``  the weakest group, which an average is designed to hide.
    """
    if not (len(values) == len(weights) == len(groups)):
        raise ValueError("values, weights and groups must have equal length")
    if not values:
        return {"macro": None, "micro": None, "worst": None, "worst_group": None,
                "groups": 0}
    array = np.asarray(values, dtype="float64")
    weight = np.asarray(weights, dtype="float64")
    per_group: dict[str, list[float]] = {}
    for name, value in zip(groups, array):
        per_group.setdefault(name, []).append(float(value))
    group_means = {name: float(np.mean(rows)) for name, rows in per_group.items()}
    worst_group = min(group_means, key=lambda name: group_means[name])
    total = float(weight.sum())
    return {
        "macro": float(np.mean(list(group_means.values()))),
        "micro": float((array * weight).sum() / total) if total else None,
        "worst": group_means[worst_group],
        "worst_group": worst_group,
        "groups": len(group_means),
    }
