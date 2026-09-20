"""Straight-through Gumbel-Softmax over the retriever vocabulary.

    P  = softmax((A + G) / tau)
    P~ = P + stopgrad(onehot(argmax P) - P)
    T  = P~ @ W_embedding

The forward pass therefore sees the embedding of a real token while the
backward pass uses the relaxation's gradient.  This is a biased estimator, not
the derivative of argmax -- gains it reports must survive ``round_trip_report``
on decoded text before they are believed.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn


def allowed_vocab_mask(tokenizer, vocab_size: int | None = None) -> torch.Tensor:
    """Boolean mask of tokens a trigger may use: no special, no unused slots.

    Rows beyond ``len(tokenizer)`` exist in some checkpoints purely to pad the
    embedding matrix; they decode to nothing and are excluded too.
    """
    size = vocab_size if vocab_size is not None else len(tokenizer)
    mask = torch.ones(size, dtype=torch.bool)
    for token_id in tokenizer.all_special_ids:
        if 0 <= token_id < size:
            mask[token_id] = False
    if len(tokenizer) < size:
        mask[len(tokenizer):] = False
    for token_id in range(min(len(tokenizer), size)):
        token = tokenizer.convert_ids_to_tokens(token_id)
        if token is None or token.startswith("[unused") or token.startswith("<unused"):
            mask[token_id] = False
    return mask


def straight_through_gumbel(
    logits: torch.Tensor,
    tau: float,
    *,
    hard: bool = True,
    mask: torch.Tensor | None = None,
    noise: bool = True,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return ``P~`` of shape ``[L, V]``.

    ``hard=False`` skips the straight-through correction and returns the soft
    distribution.  ``noise=False`` drops the Gumbel perturbation, which is what
    the deterministic export path wants -- it is an explicit argument rather
    than something inferred from grad mode, so training and evaluation cannot
    silently disagree about whether the trigger was sampled.
    """
    if logits.ndim != 2:
        raise ValueError("logits must have shape [trigger_tokens, vocabulary]")
    if tau <= 0:
        raise ValueError("tau must be positive")
    scores = logits
    if mask is not None:
        # Masking before the softmax keeps the distribution normalized over the
        # allowed tokens; masking afterwards would leave probability mass behind.
        scores = scores.masked_fill(~mask.to(scores.device), float("-inf"))
    if noise:
        # Gumbel(0, 1) via -log(Exp(1)); the clamp keeps the log finite when the
        # sampler returns a subnormal draw.
        exponential = torch.empty(
            scores.shape, device=scores.device, dtype=scores.dtype
        ).exponential_(generator=generator).clamp_min(1e-20)
        scores = scores - torch.log(exponential)
    probabilities = torch.softmax(scores / tau, dim=-1)
    if not hard:
        return probabilities
    index = probabilities.argmax(dim=-1, keepdim=True)
    onehot = torch.zeros_like(probabilities).scatter_(-1, index, 1.0)
    return probabilities + (onehot - probabilities).detach()


def to_trigger_embeddings(relaxed: torch.Tensor, embedding_matrix: torch.Tensor) -> torch.Tensor:
    """``T = P~ @ W`` -- ``[L, V] @ [V, hidden]``."""
    if relaxed.shape[-1] != embedding_matrix.shape[0]:
        raise ValueError(
            f"relaxation covers {relaxed.shape[-1]} tokens but the embedding "
            f"matrix has {embedding_matrix.shape[0]} rows"
        )
    return relaxed @ embedding_matrix


class TriggerLogits(nn.Module):
    """A free ``[L, V]`` logits matrix -- baseline B2, optimized per episode.

    This is the ablation that separates the optimizer's contribution from the
    generator's conditioning.  It learns one trigger and nothing transferable.
    """

    def __init__(self, trigger_tokens: int, vocab_size: int, *, scale: float = 0.01):
        super().__init__()
        self.logits = nn.Parameter(torch.randn(trigger_tokens, vocab_size) * scale)

    def forward(
        self, tau: float, *, hard: bool = True, mask: torch.Tensor | None = None,
        noise: bool = True, generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        return straight_through_gumbel(self.logits, tau, hard=hard, mask=mask,
                                       noise=noise, generator=generator)


def hard_token_ids(logits: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Deterministic export: argmax with no Gumbel noise."""
    scores = logits
    if mask is not None:
        scores = scores.masked_fill(~mask.to(scores.device), float("-inf"))
    return scores.argmax(dim=-1).detach().cpu()


def export_hard_trigger(
    logits: torch.Tensor, tokenizer, mask: torch.Tensor | None = None
) -> dict[str, Any]:
    ids = hard_token_ids(logits, mask)
    tokens = tokenizer.convert_ids_to_tokens(ids.tolist())
    return {
        "token_ids": [int(value) for value in ids.tolist()],
        "tokens": tokens,
        "trigger": tokenizer.convert_tokens_to_string(tokens).strip(),
    }


def round_trip_report(
    trigger: dict[str, Any], tokenizer, *, query: str | None = None
) -> dict[str, Any]:
    """Decode then re-tokenize, and report what the runtime would actually see.

    A trigger that only exists as ids is not a trigger.  Whitespace, WordPiece
    merges and truncation can all change length or content on the way back, so
    every reported gain must be re-scored on ``re_encoded_ids``.
    """
    raw_ids = list(trigger["token_ids"])
    re_encoded = tokenizer(trigger["trigger"], add_special_tokens=False).input_ids
    report = {
        "raw_ids": raw_ids,
        "re_encoded_ids": re_encoded,
        "length_match": len(raw_ids) == len(re_encoded),
        "ids_match": raw_ids == re_encoded,
        "special_token_used": any(
            token in set(tokenizer.all_special_ids) for token in raw_ids
        ),
        "empty": not trigger["trigger"].strip(),
    }
    if query is not None:
        joined = f"{query} {trigger['trigger']}".strip()
        query_only = tokenizer(query, add_special_tokens=False).input_ids
        report["joined_ids"] = tokenizer(joined, add_special_tokens=False).input_ids
        report["joined_delta"] = len(report["joined_ids"]) - len(query_only)
        # A trigger that re-tokenizes to a different length inside the query
        # shifts the retriever's input; that must be visible, not averaged away.
        report["joined_length_match"] = report["joined_delta"] == len(re_encoded)
    report["valid"] = (
        report["length_match"] and not report["special_token_used"] and not report["empty"]
    )
    return report
