"""B1 -- AgentPoison's HotFlip search as a per-episode MCAT mode.

VN — Đây là baseline reviewer hỏi đầu tiên: "MCAT so với AgentPoison, cùng quyền,
cùng episode, cùng ngân sách thì sao?" (gap G3 trong
``_idea_q1_aplus/open_problems.md``). ``src/triggers/hotflip_margin.py`` đã chạy
được AgentPoison nhưng trên pipeline riêng của StrategyQA: tập query khác, poison
khác, objective khác, nên số của nó **không ghép cặp được** với số của MCAT. File
này đặt đúng thuật toán đó vào ``train.train`` như ``--mode hotflip``, nên chỉ còn
**một** biến khác giữa B1 và B2/B3/M1: bộ tối ưu.

The search is upstream's loop (``algo/trigger_optimization.py``,
``hotflip_attack``) with everything around it taken from MCAT instead of from
the StrategyQA runner:

* the objective is ``train.losses_for`` -- the same ``L_uni + lambda_cpt*L_cpt
  + lambda_ret*L_ret`` every other arm minimizes, under the same poison policy
  and trigger position.  ``--lambda-ret 0`` is upstream AgentPoison's
  ``uni : cpt = 1 : 0.1`` objective;
* the gradient is taken on ``Q_opt`` of the episode's own context, and the
  candidates are scored on the same ``Q_opt`` -- never on ``Q_eval``;
* candidates come from the retriever's ``vocab_mask``, so B1 cannot propose a
  special or ``[unused]`` token that the other arms were forbidden to use.

One iteration:

1. ``g = d l_total / d T`` at the current hard trigger ``T = W[ids]``;
2. pick a position ``p`` uniformly at random, as upstream does;
3. rank tokens by ``-W @ g[p]`` -- the first-order decrease of the loss when
   ``ids[p]`` is replaced -- and keep the top ``hotflip_candidates``;
4. score each candidate exactly (no gradient) and accept the best one only if
   it strictly lowers the loss, which is upstream's acceptance rule.

There is no relaxation anywhere, so unlike B2/M1 B1 never pays a straight-through
gap: the trigger it optimizes *is* the trigger it exports.  What it pays instead
is forward passes -- ``hotflip_candidates * (|Q_opt| + |poison|)`` per
iteration -- and those are charged to the ledger by ``encoding.py`` like any
other encode.  One HotFlip iteration and one gradient step are therefore *not*
the same budget; compare arms on ``costs-train.json``, not on ``--steps`` alone.

Two departures from upstream, both deliberate:

* the initial trigger is drawn uniformly from the allowed vocabulary rather than
  being ``[MASK] * L``.  ``[MASK]`` is a special token, so any position HotFlip
  never got round to flipping would export as a round-trip-invalid trigger, and
  B2 starts from random logits anyway;
* upstream's ``hotflip_attack(increase_loss=False)`` negates the scores and then
  masks the excluded ids with ``+inf`` before ``topk``, i.e. it *selects* the
  excluded tokens.  Upstream only ever calls it with ``increase_loss=True``, so
  the bug never fires there, but it is why this file does not import it.
"""
from __future__ import annotations

from typing import Any, Callable

import torch
from torch import nn
import torch.nn.functional as F

from src.triggers.mcat.costs import record as charge

DEFAULT_CANDIDATES = 100   # upstream ``--num-cand``


class HotFlipTrigger(nn.Module):
    """A hard trigger held as token ids -- B1's whole state.

    ``token_ids`` is a buffer, not a parameter: HotFlip is a discrete search and
    no optimizer ever touches it.  Being a buffer is what lets it ride through
    ``state_dict`` into the checkpoint and into ``warm-start`` unchanged.

    ``logits`` exposes it as a one-hot ``[L, V]`` matrix so the export, the
    evaluation and the adaptation code paths that read ``module.logits`` work
    without a B1 special case: argmax of a one-hot row is the held token.
    """

    def __init__(self, trigger_tokens: int, vocab_size: int, allowed: torch.Tensor):
        super().__init__()
        if trigger_tokens < 1:
            raise ValueError(f"trigger_tokens must be positive, got {trigger_tokens}")
        if allowed.shape != (vocab_size,):
            raise ValueError(
                f"allowed mask has shape {tuple(allowed.shape)}, expected ({vocab_size},)"
            )
        pool = allowed.nonzero(as_tuple=True)[0].cpu()
        if pool.numel() == 0:
            raise ValueError("the vocabulary mask allows no token")
        self.vocab_size = int(vocab_size)
        # Global RNG on purpose: ``seed_everything`` seeds it and the checkpoint
        # restores it, exactly as for the B2 logits drawn with ``torch.randn``.
        draw = pool[torch.randint(pool.numel(), (trigger_tokens,))]
        self.register_buffer("token_ids", draw.clone())

    @property
    def logits(self) -> torch.Tensor:
        return F.one_hot(self.token_ids, self.vocab_size).float()


def hotflip_candidates(
    gradient: torch.Tensor,
    embedding_matrix: torch.Tensor,
    allowed: torch.Tensor,
    count: int,
    *,
    exclude: int | None = None,
) -> torch.Tensor:
    """Top-``count`` replacements for one position, by first-order loss decrease.

    Replacing ``e_old`` with ``e_new`` changes the loss by about
    ``(e_new - e_old) . g``; ``e_old`` is the same for every candidate, so the
    ranking is ``-W @ g``.  Disallowed ids and ``exclude`` (the current token,
    whose swap would be a no-op) are pushed to ``-inf`` *before* ``topk``.
    """
    if gradient.ndim != 1:
        raise ValueError("gradient must be the [hidden] gradient of one position")
    if count < 1:
        raise ValueError(f"count must be positive, got {count}")
    with torch.no_grad():
        scores = -(embedding_matrix @ gradient)
        scores = scores.masked_fill(~allowed.to(scores.device), float("-inf"))
        if exclude is not None:
            scores[exclude] = float("-inf")
        available = int(torch.isfinite(scores).sum())
        if available == 0:
            return torch.empty(0, dtype=torch.long, device=scores.device)
        return scores.topk(min(count, available)).indices


def hotflip_step(
    module: HotFlipTrigger,
    context: Any,
    retriever: Any,
    config: Any,
    *,
    losses: Callable[..., dict[str, torch.Tensor]],
) -> dict[str, Any]:
    """One AgentPoison iteration on one episode; mutates ``module.token_ids``.

    ``losses`` is ``train.losses_for``, passed in rather than imported so the
    objective cannot drift away from the one the other arms minimize.

    Returns the loss terms at the trigger the gradient was taken at (the same
    convention as a gradient step's record), plus what the search did.
    """
    weight = retriever.embedding_matrix
    ids = module.token_ids
    embeds = weight[ids].detach().clone().requires_grad_(True)
    values = losses(context, embeds, retriever, config)
    (gradient,) = torch.autograd.grad(values["l_total"], embeds)
    charge("encoder_backward", 1)

    position = int(torch.randint(ids.shape[0], (1,)))
    candidates = hotflip_candidates(
        gradient[position], weight, retriever.vocab_mask, config.hotflip_candidates,
        exclude=int(ids[position]),
    )
    current = float(values["l_total"].detach())
    best_loss, best_id = current, None
    with torch.no_grad():
        for candidate in candidates.tolist():
            trial = ids.clone()
            trial[position] = candidate
            loss = float(losses(context, weight[trial], retriever, config)["l_total"])
            # Strictly lower, as upstream: a tie keeps the current trigger.
            if loss < best_loss:
                best_loss, best_id = loss, candidate
        if best_id is not None:
            module.token_ids[position] = best_id

    return {key: float(value.detach()) for key, value in values.items()} | {
        "grad_norm": float(gradient.norm()),
        "accepted": best_id is not None,
        "flip_position": position,
        "candidates": int(candidates.numel()),
        "candidate_loss": best_loss,
    }
