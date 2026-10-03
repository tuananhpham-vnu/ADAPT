"""Retriever encoding paths for MCAT.

``encode_with_trigger_embeddings`` generalizes
``src.triggers.margin._encode_triggered``: the trigger arrives as a
``[L, hidden]`` tensor that carries gradient back to the generator instead of
as token ids.  At the default ``position="suffix"`` the assembly is
``[CLS] prefix trigger [SEP]`` with manual right padding, byte-for-byte what it
has always been, so a relaxed run and the HotFlip baseline still score the same
text the same way.

VN — ``position`` là biến của P1 trong ``_idea_q1_aplus/open_problems.md``: chèn
trigger vào đầu / cuối / giữa / cả hai đầu. Trước đây chỉ có ``suffix``, nên mọi
số ASR đã có đều là số của một vị trí duy nhất.

``position`` is the P1 variable:

``suffix``       ``[CLS] query trigger [SEP]``  -- the historical default
``prefix``       ``[CLS] trigger query [SEP]``
``middle``       ``[CLS] query[:h] trigger query[h:] [SEP]``, ``h = len//2``
``both``         ``[CLS] T1 query T2 [SEP]``, the trigger split in half
``both-repeat``  ``[CLS] trigger query trigger [SEP]``, the whole trigger twice

``both`` and ``both-repeat`` answer different questions and must not be
conflated.  ``both`` keeps the token budget of every other position, so a
difference in score is a difference in *placement*.  ``both-repeat`` spends
twice the trigger tokens, so it is not length-matched and any gain it shows is
confounded with length -- it is here to bound what two copies can buy, and it
should never be compared against the others as if the budget were equal.

The retriever is frozen with ``requires_grad_(False)``.  Deliberately no
``no_grad()`` wrapper: that would sever the path back to the generator.
"""
from __future__ import annotations

from typing import Sequence

import torch

from src.triggers.mcat.costs import record

POSITIONS = ("suffix", "prefix", "middle", "both", "both-repeat")

#: How many copies of the trigger each position writes into one sequence.  Only
#: ``both-repeat`` spends more than one, and the token budget below has to know.
POSITION_COPIES = {"suffix": 1, "prefix": 1, "middle": 1, "both": 1, "both-repeat": 2}


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
    # The one place plain encoding happens, so it is the one place it is
    # counted; without an active ledger this costs nothing (see costs.record).
    record("encoder_forward", len(texts))
    encoded = tokenizer(list(texts), padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    return model(**encoded).pooler_output


def _assemble(
    position: str,
    cls_embeds: torch.Tensor,
    query_embeds: torch.Tensor,
    sep_embeds: torch.Tensor,
    trigger_embeds: torch.Tensor,
) -> torch.Tensor:
    """Lay out one sequence for ``position``; the only place placement is decided.

    VN — Chỗ duy nhất quyết định trigger nằm ở đâu. Mọi vị trí đi qua đây, nên
    không có nhánh nào lặng lẽ dùng layout khác.

    Every position goes through this function, so no caller can quietly use a
    different layout than the one its artifacts claim.
    """
    if position == "suffix":
        parts = (cls_embeds, query_embeds, trigger_embeds, sep_embeds)
    elif position == "prefix":
        parts = (cls_embeds, trigger_embeds, query_embeds, sep_embeds)
    elif position == "middle":
        half = query_embeds.shape[0] // 2
        parts = (cls_embeds, query_embeds[:half], trigger_embeds,
                 query_embeds[half:], sep_embeds)
    elif position == "both":
        # Split, not duplicated: the total trigger budget matches every other
        # position, so a difference here is placement and not length.
        cut = (trigger_embeds.shape[0] + 1) // 2
        parts = (cls_embeds, trigger_embeds[:cut], query_embeds,
                 trigger_embeds[cut:], sep_embeds)
    elif position == "both-repeat":
        parts = (cls_embeds, trigger_embeds, query_embeds, trigger_embeds, sep_embeds)
    else:
        raise ValueError(f"position must be one of {POSITIONS}, got {position!r}")
    return torch.cat([part for part in parts if part.shape[0]], dim=0)


def encode_with_trigger_embeddings(
    model,
    tokenizer,
    texts: Sequence[str],
    trigger_embeds: torch.Tensor,
    *,
    device: str,
    max_length: int,
    position: str = "suffix",
) -> torch.Tensor:
    """Encode ``texts`` with ``trigger_embeds`` placed according to ``position``.

    ``trigger_embeds`` is ``[L, hidden]`` and shared across the batch, which is
    what makes one trigger universal over an episode's queries.  ``position``
    defaults to ``"suffix"``, which reproduces the pre-P1 assembly exactly.

    VN — ``position`` mặc định là ``suffix`` để tái lập đúng hành vi cũ; mọi giá
    trị khác là một arm khác của ablation P1 và phải được ghi vào contract, nếu
    không thì trigger export ra sẽ bị chấm ở vị trí khác lúc tối ưu.
    """
    if trigger_embeds.ndim != 2:
        raise ValueError("trigger_embeds must have shape [trigger_tokens, hidden]")
    if position not in POSITIONS:
        raise ValueError(f"position must be one of {POSITIONS}, got {position!r}")
    embedding = model.get_input_embeddings()
    trigger_length = trigger_embeds.shape[0]
    if position == "both" and trigger_length < 2:
        raise ValueError(
            f"position 'both' splits the trigger in half and needs at least 2 tokens, "
            f"got {trigger_length}; use 'both-repeat' to place a 1-token trigger twice"
        )
    # ``both-repeat`` writes the trigger twice, so it reserves twice the room.
    # Charging it the single-copy budget would silently truncate the query and
    # turn a length confound into a query-truncation confound as well.
    spent = trigger_length * POSITION_COPIES[position]
    budget = max_length - spent - 2
    if budget < 1:
        raise ValueError(
            f"max_length={max_length} leaves no room for {spent} trigger "
            f"tokens (position {position!r}) plus [CLS] and [SEP]"
        )

    # Counted only once the call is known to be well formed, so a rejected
    # configuration never shows up as work the retriever did.
    record("encoder_forward", len(texts))
    trigger_embeds = trigger_embeds.to(device)
    pieces, masks = [], []
    for text in texts:
        prefix = tokenizer(text, add_special_tokens=False, truncation=True,
                           max_length=budget).input_ids
        head = torch.tensor([tokenizer.cls_token_id], device=device)
        body = torch.tensor(prefix, dtype=torch.long, device=device)
        tail = torch.tensor([tokenizer.sep_token_id], device=device)
        pieces.append(_assemble(
            position, embedding(head), embedding(body), embedding(tail), trigger_embeds,
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
