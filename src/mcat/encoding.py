"""Retriever encoding paths for MCAT.

``encode_with_trigger_embeddings`` generalizes
``algo.agentpoison_margin._encode_triggered``: the trigger arrives as a
``[L, hidden]`` tensor that carries gradient back to the generator instead of
as token ids.  The surrounding assembly -- ``[CLS] prefix trigger [SEP]``,
manual right padding and the attention mask -- is kept byte-for-byte identical
so a relaxed run and the HotFlip baseline score the same text the same way.

The retriever is frozen with ``requires_grad_(False)``.  Deliberately no
``no_grad()`` wrapper: that would sever the path back to the generator.
"""
from __future__ import annotations

from typing import Sequence

import torch


def freeze_retriever(model) -> None:
    """Freeze every retriever parameter and drop any stale gradient."""
    model.eval()
    model.requires_grad_(False)
    model.zero_grad(set_to_none=True)


def trigger_embeddings_from_ids(model, trigger_ids: torch.Tensor) -> torch.Tensor:
    """Embedding rows for hard token ids -- the reference the relaxation targets."""
    return model.get_input_embeddings()(trigger_ids)


def encode_plain(
    model, tokenizer, texts: Sequence[str], *, device: str, max_length: int
) -> torch.Tensor:
    encoded = tokenizer(list(texts), padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    return model(**encoded).pooler_output


def encode_with_trigger_embeddings(
    model,
    tokenizer,
    texts: Sequence[str],
    trigger_embeds: torch.Tensor,
    *,
    device: str,
    max_length: int,
) -> torch.Tensor:
    """Encode ``texts`` with ``trigger_embeds`` appended before ``[SEP]``.

    ``trigger_embeds`` is ``[L, hidden]`` and shared across the batch, which is
    what makes one trigger universal over an episode's queries.
    """
    if trigger_embeds.ndim != 2:
        raise ValueError("trigger_embeds must have shape [trigger_tokens, hidden]")
    embedding = model.get_input_embeddings()
    trigger_length = trigger_embeds.shape[0]
    budget = max_length - trigger_length - 2
    if budget < 1:
        raise ValueError(
            f"max_length={max_length} leaves no room for {trigger_length} trigger "
            "tokens plus [CLS] and [SEP]"
        )

    pieces, masks = [], []
    for text in texts:
        prefix = tokenizer(text, add_special_tokens=False, truncation=True,
                           max_length=budget).input_ids
        head = torch.tensor([tokenizer.cls_token_id] + prefix, device=device)
        tail = torch.tensor([tokenizer.sep_token_id], device=device)
        pieces.append(torch.cat(
            (embedding(head), trigger_embeds.to(device), embedding(tail)), dim=0
        ))

    longest = max(piece.shape[0] for piece in pieces)
    padded = []
    for piece in pieces:
        pad = torch.zeros(longest - piece.shape[0], piece.shape[1],
                          device=device, dtype=piece.dtype)
        padded.append(torch.cat((piece, pad), dim=0))
        masks.append([1] * piece.shape[0] + [0] * pad.shape[0])
    inputs = torch.stack(padded)
    attention = torch.tensor(masks, device=device)
    return model(inputs_embeds=inputs, attention_mask=attention).pooler_output
