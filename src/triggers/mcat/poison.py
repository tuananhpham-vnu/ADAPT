"""Poison keys as they were written into the index, and the fixed-write policy.

VN — Có hai nghĩa của chữ "fixed", đừng lẫn:

* ``objectives.PoisonPolicy`` nói về **một bước train**: gradient có chạy qua
  phía poison hay không.
* File này nói về **quyền ghi index qua các snapshot**. ``refresh`` = kẻ tấn công
  được viết lại B record ở mỗi snapshot. ``fixed`` = chỉ viết một lần ở ``s0``,
  sau đó không đụng nữa.

Nếu nhánh ``fixed`` vẫn lặng lẽ encode lại poison bằng trigger mới, nó mượn đúng
lợi thế của nhánh ``refresh`` và mọi số phía dưới thành vô giá trị. Nên key của
``s0`` được ghi ra file một lần, và lúc đánh giá thì **đọc lại** chứ không tính lại.

``objectives.PoisonPolicy`` decides whether gradient reaches the poison side
*within* one optimization step.  Under drift the word "fixed" means something
stronger, and the difference is the whole point of the stress test:

``refresh``  the attacker may rewrite its B records at every snapshot, so the
             poison keys are re-encoded with whatever trigger is current.
``fixed``    the records were written once, at ``s0``, with the ``s0`` trigger,
             and are never touched again.  Later snapshots rank *those* vectors.

Re-encoding poison with the snapshot's own trigger while calling the arm
``fixed`` would hand it the refresh arm's advantage and make every number below
it meaningless -- so the frozen keys live in an artifact on disk, written once,
and the evaluator reads them rather than recomputing them.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from src.triggers.artifacts import atomic_torch
from src.triggers.mcat.costs import record
from src.triggers.mcat.encoding import encode_with_trigger_embeddings
from src.triggers.mcat.retrievers import Retriever
from src.triggers.mcat.runtime import EpisodeContext

FROZEN_POISON_FILE = "poison_s0.pt"
SCHEMA_VERSION = 1


def encode_poison_keys(
    context: EpisodeContext, trigger_ids: torch.Tensor, retriever: Retriever
) -> torch.Tensor:
    """The keys the attacker's B records would carry under ``trigger_ids``.

    VN — Encode B record poison của episode với trigger đang xét, ra vector key
    mà retriever sẽ dùng để xếp hạng.
    """
    with torch.no_grad():
        embeds = retriever.model.get_input_embeddings()(trigger_ids.to(retriever.device))
        return encode_with_trigger_embeddings(
            retriever.model, retriever.tokenizer, context.poison_texts, embeds,
            device=retriever.device, max_length=retriever.max_length,
        )


def runtime_trigger_ids(trigger: dict[str, Any], device: str) -> torch.Tensor:
    """The ids the runtime would really produce: re-encoded first, raw as fallback.

    VN — Lấy đúng dãy token id mà runtime thật sẽ sinh ra: ưu tiên bản đã decode
    rồi tokenize lại, không có thì mới dùng id thô của optimizer.
    """
    ids = trigger["round_trip"]["re_encoded_ids"] or trigger["token_ids"]
    return torch.tensor(ids, device=device)


@dataclass
class FrozenPoison:
    """The index as it stood after the attacker's one write at ``s0``.

    VN — Ảnh chụp index sau lần ghi duy nhất ở ``s0``: key poison, trigger đã
    dùng, và fingerprint của encoder để không ai load nhầm key của model khác.
    """

    retriever: dict[str, Any]
    triggers: dict[str, list[int]]
    keys: dict[str, torch.Tensor]
    snapshot_ids: dict[str, str]

    def keys_for(self, episode_id: str, *, device: str | None = None) -> torch.Tensor:
        if episode_id not in self.keys:
            raise KeyError(
                f"no frozen poison for episode {episode_id!r}; it was written for "
                f"{sorted(self.keys)[:3]}... Run the s0 freeze before a fixed-policy "
                "evaluation instead of letting the poison be re-encoded"
            )
        keys = self.keys[episode_id]
        return keys if device is None else keys.to(device)

    def save(self, path: Path) -> Path:
        path = Path(path)
        atomic_torch(path, {
            "schema_version": SCHEMA_VERSION, "retriever": self.retriever,
            "triggers": self.triggers, "snapshot_ids": self.snapshot_ids,
            "keys": {key: value.detach().cpu() for key, value in self.keys.items()},
        })
        return path

    @classmethod
    def load(cls, path: Path, *, retriever: Retriever | None = None) -> "FrozenPoison":
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"{path}: frozen poison schema {payload.get('schema_version')!r} is not "
                f"{SCHEMA_VERSION}"
            )
        if retriever is not None and payload["retriever"] != retriever.fingerprint():
            raise ValueError(
                f"{path}: the frozen poison was written by another encoder "
                f"({payload['retriever']}); its keys live in a different space"
            )
        return cls(retriever=payload["retriever"], triggers=payload["triggers"],
                   keys=payload["keys"], snapshot_ids=payload["snapshot_ids"])


def freeze_poison(
    retriever: Retriever,
    contexts: Sequence[EpisodeContext],
    triggers: Iterable[dict[str, Any]],
    *,
    output_dir: Path,
) -> FrozenPoison:
    """Write the poison records once, at ``s0``, and persist their keys.

    VN — Ghi poison đúng một lần ở ``s0`` rồi lưu key xuống ``poison_s0.pt``.
    ``contexts`` bắt buộc là context của ``s0``; ghi ở snapshot đã trôi thì đó
    không còn là chính sách ``fixed`` nữa.

    ``contexts`` must be the ``s0`` contexts: a record written against a drifted
    snapshot is not the record the fixed policy describes.
    """
    keys, token_ids, snapshots = {}, {}, {}
    for context, trigger in zip(contexts, triggers):
        episode = context.episode
        if context.snapshot_id != episode.snapshot_id:
            raise ValueError(
                f"{episode.episode_id}: poison may only be frozen at the base snapshot, "
                f"got {context.snapshot_id!r}"
            )
        ids = runtime_trigger_ids(trigger, retriever.device)
        # The one write the fixed policy ever makes, charged like any other.
        record("index_writes", len(context.poison_texts))
        keys[episode.episode_id] = encode_poison_keys(context, ids, retriever).detach().cpu()
        token_ids[episode.episode_id] = ids.detach().cpu().tolist()
        snapshots[episode.episode_id] = context.snapshot_id
    frozen = FrozenPoison(retriever.fingerprint(), token_ids, keys, snapshots)
    frozen.save(Path(output_dir) / FROZEN_POISON_FILE)
    return frozen
