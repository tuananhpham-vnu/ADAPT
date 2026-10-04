"""P0-R2, the two non-benign drifts: memory that carries a trigger.

VN — Lần chạy 3 cho thấy memory **lành** (kể cả tài liệu gần vùng trigger nhất)
không làm trigger cũ suy giảm. Còn hai kịch bản có thể làm được:

``self``   agent tự ghi các tương tác **đã bị kích hoạt** vào memory: bản ghi =
           một query cùng split + **chính trigger của mình**, mang nhãn lành.
           Chúng rơi đúng vào vùng trigger và tranh top-k với poison.
``rival``  kẻ tấn công khác chèn poison của họ: trigger + poison đã tối ưu độc
           lập của **episode khác** cùng split, cùng domain.

Trigger và poison của mình đóng băng ở ``s0`` (như ``growth_probe``); chỉ có các
bản ghi thêm vào thay đổi. Ai được thêm do RNG có seed quyết định, lặp nhiều seed.

The benign-growth probes found nothing: the trigger's region of embedding space
is empty of clean documents and stays empty however they are chosen.  What can
still reach that region is a record that itself carries a trigger.  ``self``
adds records written with *this* attacker's frozen trigger -- the agent logging
its own triggered interactions as ordinary experience -- and counts them as
competitors, not poison.  ``rival`` adds the frozen poison of other attackers,
each the full s0 search of another episode in the same split.

Only the added records move; the trigger, the attacker's own poison keys and
``Q_eval`` are fixed at ``s0``.  Which records arrive is a seeded draw, repeated
over several seeds, and nothing here can see a score before choosing.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import random
from typing import Any, Sequence

import torch

from src.triggers.artifacts import append_jsonl, stable_hash
from src.triggers.mcat.costs import CostLedger
from src.triggers.mcat.drift_eval import load_rows, prepare_bases
from src.triggers.mcat.encoding import encode_with_trigger_embeddings
from src.triggers.mcat.episodes import Episode, family_key
from src.triggers.mcat.evaluate import evaluate_trigger
from src.triggers.mcat.poison import FrozenPoison, runtime_trigger_ids
from src.triggers.mcat.probes import hit_of, probe_row_key
from src.triggers.mcat.runtime import Workspace
from src.triggers.mcat.train import TrainConfig

SCENARIOS = ("self", "rival")
CONTAMINATION_ROWS = "probe_contamination.jsonl"
CONTAMINATION_SUMMARY = "probe_contamination.json"
DEFAULT_LEVELS = {"self": (1, 2, 5, 10, 25), "rival": (1, 3, 7, 15)}


@dataclass(frozen=True)
class ContaminationConfig:
    """The protocol, as data so it lands in the artifact verbatim."""

    scenario: str = "self"
    #: ``self``: records added.  ``rival``: other attackers added.
    levels: tuple[int, ...] = DEFAULT_LEVELS["self"]
    seeds: tuple[int, ...] = (0, 1, 2)
    score: str = "dot"

    def __post_init__(self) -> None:
        if self.scenario not in SCENARIOS:
            raise ValueError(f"scenario must be one of {SCENARIOS}, got {self.scenario!r}")
        if not self.levels or any(level < 1 for level in self.levels):
            raise ValueError(f"levels must be positive counts, got {self.levels}")
        if list(self.levels) != sorted(set(self.levels)):
            raise ValueError(f"levels must be strictly increasing, got {self.levels}")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError(f"seeds must be distinct and non-empty, got {self.seeds}")

    def fingerprint(self) -> str:
        return stable_hash(asdict(self))


def self_pool(workspace: Workspace, episode: Episode, *, ratios, seed: int) -> list[str]:
    """Query texts the agent could have met and logged: same split, not this episode's.

    VN — Những query mà agent có thể đã gặp rồi ghi vào memory: cùng split, và
    không phải query nào của chính episode (đặc biệt không phải ``Q_eval``).
    """
    assignment = workspace.assignment(episode.domain, ratios=tuple(ratios), seed=seed)
    own = set(episode.support_qids + episode.optimization_qids + episode.eval_qids
              + episode.poison_source_qids)
    return [row["question"] for row in workspace.queries(episode.domain)
            if row["qid"] not in own
            and assignment.get(family_key(row["family"])) == episode.split]


def self_records(
    workspace: Workspace, texts: Sequence[str], trigger: dict[str, Any], *, position: str,
) -> torch.Tensor:
    """Keys of logged interactions: each query text written with the frozen trigger."""
    retriever = workspace.retriever
    with torch.no_grad():
        embeds = retriever.model.get_input_embeddings()(
            runtime_trigger_ids(trigger, retriever.device))
        return torch.cat([
            encode_with_trigger_embeddings(
                retriever.model, retriever.tokenizer, list(texts[start:start + 64]), embeds,
                device=retriever.device, max_length=retriever.max_length, position=position)
            for start in range(0, len(texts), 64)], dim=0)


def contamination_probe(
    workspace: Workspace,
    episodes: Sequence[Episode],
    config: TrainConfig,
    probe: ContaminationConfig,
    *,
    checkpoint: dict[str, Any] | None,
    output_dir: Path,
    contract: dict[str, Any],
    ratios: tuple[float, ...],
    split_seed: int,
    ledger: CostLedger | None = None,
    resume: bool = False,
) -> list[dict[str, Any]]:
    """Score each frozen s0 trigger as trigger-carrying records join its memory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger = ledger or CostLedger(device=str(workspace.retriever.device))
    rows_path = output_dir / CONTAMINATION_ROWS
    done = {row["key"] for row in load_rows(rows_path) if "key" in row} if resume else set()
    fingerprint = stable_hash([contract, probe.fingerprint()])
    position = config.trigger_position

    bases, frozen = prepare_bases(
        workspace, episodes, config, checkpoint=checkpoint, output_dir=output_dir,
        contract=contract, ledger=ledger, resume=resume, freeze=True,
    )
    if frozen is None:
        frozen = FrozenPoison.load(output_dir / "poison_s0.pt", retriever=workspace.retriever)
    attackers = [episode.episode_id for episode in episodes
                 if bases.get(episode.episode_id, (None, None))[1] is not None]
    if probe.scenario == "rival" and probe.levels[-1] > len(attackers) - 1:
        raise ValueError(
            f"rival level {probe.levels[-1]} needs that many other attackers, but only "
            f"{len(attackers) - 1} other episodes have an s0 trigger")

    rows: list[dict[str, Any]] = []

    def score(episode, context, trigger, keys, *, level, drift_seed, added, kind):
        snapshot_id = (episode.snapshot_id if kind == "base"
                       else f"{episode.episode_id}-{probe.scenario}-{level}-d{drift_seed}")
        key = probe_row_key(episode.episode_id, snapshot_id, "contamination", fingerprint)
        if key in done:
            return
        with ledger.phase(f"{snapshot_id}/contamination-probe"):
            scored = evaluate_trigger(workspace, context, trigger, score=probe.score,
                                      frozen_poison=keys, position=position)
        row = {
            "probe": "contamination", "scenario": probe.scenario, "key": key,
            "applicable": True, "episode_id": episode.episode_id,
            "domain": episode.domain, "split": episode.split,
            "snapshot_id": snapshot_id, "kind": kind,
            # summarize_growth groups on ``growth``: here it is the count added.
            "growth": float(level), "drift_seed": drift_seed,
            "documents": context.memory_vectors.shape[0], "added": added,
            "poison_records": keys.shape[0],
            "poison_fraction": keys.shape[0] / max(1, context.memory_vectors.shape[0]),
            "trigger": scored["trigger"], "trigger_position": position,
            "on_hit": hit_of(scored["trigger_on"]), "off_hit": hit_of(scored["trigger_off"]),
            "trigger_on": scored["trigger_on"], "trigger_off": scored["trigger_off"],
        }
        append_jsonl(rows_path, row)
        rows.append(row)

    for episode in episodes:
        _, trigger = bases.get(episode.episode_id, (None, None))
        if trigger is None:
            continue
        keys = frozen.keys_for(episode.episode_id)
        base = workspace.context_for(episode, centers=False)
        score(episode, base, trigger, keys, level=0, drift_seed=None, added=[], kind="base")

        if probe.scenario == "self":
            pool = self_pool(workspace, episode, ratios=ratios, seed=split_seed)
            if len(pool) < probe.levels[-1]:
                raise ValueError(f"{episode.episode_id}: {len(pool)} candidate queries, "
                                 f"level {probe.levels[-1]} needs more")
        else:
            pool = sorted(other for other in attackers if other != episode.episode_id)

        for drift_seed in probe.seeds:
            rng = random.Random(stable_hash([episode.episode_id, probe.scenario,
                                             drift_seed])[:16])
            # One draw per seed, then nested prefixes: level 5 extends level 2,
            # so a change between levels is the added records and nothing else.
            order = rng.sample(pool, probe.levels[-1])
            if probe.scenario == "self":
                extra_all = self_records(workspace, order, trigger, position=position)
            for level in probe.levels:
                chosen = order[:level]
                if probe.scenario == "self":
                    extra = extra_all[:level]
                    added = [f"logged-{index}" for index in range(level)]
                else:
                    extra = torch.cat([frozen.keys_for(other) for other in chosen], dim=0)
                    added = list(chosen)
                memory = torch.cat([base.memory_vectors, extra.to(base.memory_vectors.device)])
                context = replace(base, memory_vectors=memory)
                score(episode, context, trigger, keys, level=level, drift_seed=drift_seed,
                      added=added, kind=probe.scenario)
    return rows
