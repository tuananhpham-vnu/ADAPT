"""MCAT training objective.

Uniqueness and compactness are imported unchanged from
``algo.trigger_losses`` so the AgentPoison-compatible arm is exactly the
upstream geometry.  What is new here is the retrieval term, because the event
MCAT reports -- *at least one poison record inside top-K* -- is not the event
``algo.trigger_losses.compute_retrieval_margin_loss`` optimizes.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from algo.trigger_losses import compute_compactness_loss, compute_uniqueness_loss

__all__ = [
    "PoisonPolicy", "compute_compactness_loss", "compute_hit_at_k_margin_loss",
    "compute_uniqueness_loss", "mcat_total_loss", "score_matrix",
]


def score_matrix(queries: torch.Tensor, keys: torch.Tensor, score: str = "dot") -> torch.Tensor:
    """Similarity under the runtime scorer.

    ``index/manifest.json`` records ``"normalized": false``, so the deployed
    retriever ranks by raw dot product.  ``dot`` is therefore the default;
    ``cosine`` exists only for retrievers that normalize at index time.
    """
    if score == "dot":
        return queries @ keys.T
    if score == "cosine":
        return F.normalize(queries, p=2, dim=-1) @ F.normalize(keys, p=2, dim=-1).T
    raise ValueError(f"unknown score {score!r}; use 'dot' or 'cosine'")


def compute_hit_at_k_margin_loss(
    query_embeddings: torch.Tensor,
    clean_embeddings: torch.Tensor,
    poison_embeddings: torch.Tensor,
    *,
    top_k: int = 5,
    margin: float = 0.1,
    score: str = "dot",
) -> torch.Tensor:
    """Hinge on *at least one* poison entering top-K.

        a_i = K-th highest clean score,  b_i = highest poison score
        L   = mean(relu(margin + a_i - b_i))

    ``b_i > a_i`` displaces the weakest of the K clean keys, which is the
    condition behind ``poison_top_k_rate``.

    This is NOT ``algo.trigger_losses.compute_retrieval_margin_loss``: that one
    requires the K-th *poison* to beat the *best* clean key, i.e. full top-K
    takeover, and needs ``len(poison) >= K``.  The two are not interchangeable
    and must never be reported under the same metric name.
    """
    for name, tensor in (("query", query_embeddings), ("clean", clean_embeddings),
                         ("poison", poison_embeddings)):
        if tensor.ndim != 2:
            raise ValueError(f"{name} embeddings must be rank-2 [rows, dimensions]")
    if not 1 <= top_k <= clean_embeddings.shape[0]:
        raise ValueError(
            f"top_k must be in [1, {clean_embeddings.shape[0]}] (the number of "
            f"clean keys ranked against); got {top_k}"
        )

    clean_scores = score_matrix(query_embeddings, clean_embeddings, score)
    poison_scores = score_matrix(query_embeddings, poison_embeddings, score)
    kth_clean = clean_scores.topk(k=top_k, dim=1).values[:, -1]
    best_poison = poison_scores.max(dim=1).values
    return F.relu(float(margin) + kth_clean - best_poison).mean()


def mcat_total_loss(
    l_uni: torch.Tensor,
    l_cpt: torch.Tensor,
    l_ret: torch.Tensor,
    *,
    lambda_cpt: float = 0.1,
    lambda_ret: float = 0.0,
) -> torch.Tensor:
    """``L_uni + lambda_cpt*L_cpt + lambda_ret*L_ret``, minimization convention.

    ``lambda_ret=0`` reproduces the plain AgentPoison objective and is the
    starting point: the retrieval margin is ablated in, not assumed.

    Coherence and target probability stay out of this sum on purpose -- the
    scorers in ``algo.constraint_scorers`` run under ``no_grad()`` and act as a
    sampler and a feasibility gate, not as gradient terms.
    """
    return l_uni + float(lambda_cpt) * l_cpt + float(lambda_ret) * l_ret


@dataclass(frozen=True)
class PoisonPolicy:
    """Whether poison records may be re-encoded with the current trigger.

    ``refresh``: the attacker rewrites its B records each snapshot, so gradient
    flows through both the query side and the poison side.

    ``fixed``: poison keys were written once and frozen; only the query side
    carries gradient.  This is the stress test, and it must never silently
    borrow the refresh path's advantage.
    """

    mode: str = "refresh"

    def __post_init__(self) -> None:
        if self.mode not in ("refresh", "fixed"):
            raise ValueError(f"poison policy must be 'refresh' or 'fixed', got {self.mode!r}")

    @property
    def refresh(self) -> bool:
        return self.mode == "refresh"

    def apply(self, poison_embeddings: torch.Tensor) -> torch.Tensor:
        return poison_embeddings if self.refresh else poison_embeddings.detach()
