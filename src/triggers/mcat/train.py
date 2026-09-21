"""Training loops for the direct-logit baselines and the generator.

Three modes share one optimization step and differ only in where the ``[L, V]``
logits come from:

``direct-logit``     B2 -- a fresh logits matrix per episode, nothing shared.
``universal-logit``  B3 -- one logits matrix fitted across all train episodes.
``generator``        M1/B4/B5/B6 -- logits produced from the episode's context.

Comparing them is the whole point: B2 measures what the optimizer alone buys,
B3 whether one universal trigger already suffices, and only the gap above both
can be attributed to conditioning.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn

from src.triggers.artifacts import (
    append_jsonl, atomic_json, atomic_torch, read_json, restore_rng, rng_state,
    seed_everything, stable_hash,
)
from src.triggers.mcat.costs import record as charge
from src.triggers.mcat.encoding import encode_with_trigger_embeddings
from src.triggers.mcat.episodes import Episode
from src.triggers.mcat.generator import TriggerGenerator, sample_memory_keys
from src.triggers.mcat.objectives import (
    PoisonPolicy, compute_compactness_loss, compute_hit_at_k_margin_loss,
    compute_uniqueness_loss, mcat_total_loss,
)
from src.triggers.mcat.relaxation import (
    TriggerLogits, export_hard_trigger, round_trip_report, straight_through_gumbel,
    to_trigger_embeddings,
)
from src.triggers.mcat.retrievers import Retriever
from src.triggers.mcat.runtime import EpisodeContext, Workspace

MODES = ("direct-logit", "universal-logit", "generator")


@dataclass(frozen=True)
class TrainConfig:
    mode: str = "generator"
    variant: str = "memory+query"
    steps: int = 200
    learning_rate: float = 1e-3
    tau_start: float = 2.0
    tau_end: float = 0.5
    lambda_cpt: float = 0.1
    lambda_ret: float = 0.0
    margin: float = 0.1
    poison_mode: str = "refresh"
    grad_clip: float = 5.0
    context_dim: int = 128
    hidden: int = 256
    attention_pool: bool = False
    memory_summary_keys: int = 128
    seed: int = 0

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")


def temperature(config: TrainConfig, step: int, total: int | None = None) -> float:
    """Linear anneal from ``tau_start`` to ``tau_end`` over the run.

    ``total`` overrides ``config.steps`` for a shortened budget, so a few-step
    adaptation still reaches ``tau_end`` at its own last step instead of
    stopping while the relaxation is still hot.
    """
    total = config.steps if total is None else total
    if total <= 1:
        return config.tau_end
    fraction = step / (total - 1)
    return config.tau_start + (config.tau_end - config.tau_start) * fraction


def losses_for(
    context: EpisodeContext,
    trigger_embeds: torch.Tensor,
    retriever: Retriever,
    config: TrainConfig,
) -> dict[str, torch.Tensor]:
    """One forward pass: triggered queries, triggered poison, three loss terms."""
    policy = PoisonPolicy(config.poison_mode)
    query = encode_with_trigger_embeddings(
        retriever.model, retriever.tokenizer, context.optimization_texts, trigger_embeds,
        device=retriever.device, max_length=retriever.max_length,
    )
    poison_source = trigger_embeds if policy.refresh else trigger_embeds.detach()
    poison = encode_with_trigger_embeddings(
        retriever.model, retriever.tokenizer, context.poison_texts, poison_source,
        device=retriever.device, max_length=retriever.max_length,
    )
    poison = policy.apply(poison)

    top_k = min(context.episode.retrieval_top_k, context.clean_keys.shape[0])
    l_uni = compute_uniqueness_loss(query, context.reference_centers)
    l_cpt = compute_compactness_loss(query)
    l_ret = compute_hit_at_k_margin_loss(
        query, context.clean_keys, poison, top_k=top_k, margin=config.margin,
    )
    total = mcat_total_loss(l_uni, l_cpt, l_ret,
                            lambda_cpt=config.lambda_cpt, lambda_ret=config.lambda_ret)
    return {"l_uni": l_uni, "l_cpt": l_cpt, "l_ret": l_ret, "l_total": total}


def _logits_source(
    config: TrainConfig, retriever: Retriever, episode: Episode
) -> nn.Module:
    if config.mode == "generator":
        return TriggerGenerator(
            embedding_dim=retriever.hidden_size, vocab_size=retriever.vocab_size,
            trigger_tokens=episode.trigger_tokens, context_dim=config.context_dim,
            hidden=config.hidden, variant=config.variant,
            attention_pool=config.attention_pool,
        ).to(retriever.device)
    return TriggerLogits(episode.trigger_tokens, retriever.vocab_size).to(retriever.device)


def _logits_for(
    module: nn.Module, config: TrainConfig, context: EpisodeContext,
    *, generator: torch.Generator | None = None,
) -> torch.Tensor:
    if config.mode != "generator":
        return module.logits
    memory = sample_memory_keys(
        context.memory_vectors, config.memory_summary_keys, generator=generator
    )
    return module(memory, context.support_vectors)


def _trigger_embeddings(
    logits: torch.Tensor, retriever: Retriever, tau: float, *, noise: bool = True
) -> torch.Tensor:
    relaxed = straight_through_gumbel(logits, tau, mask=retriever.vocab_mask, noise=noise)
    return to_trigger_embeddings(relaxed, retriever.embedding_matrix)


def export_for(
    module: nn.Module, config: TrainConfig, context: EpisodeContext, retriever: Retriever
) -> dict[str, Any]:
    """Freeze the trigger: argmax, decode, re-tokenize, then re-score the text."""
    with torch.no_grad():
        logits = _logits_for(module, config, context)
        trigger = export_hard_trigger(logits, retriever.tokenizer, retriever.vocab_mask)
        report = round_trip_report(
            trigger, retriever.tokenizer,
            query=context.optimization_texts[0] if context.optimization_texts else None,
        )
        # Score the ids the runtime would actually produce, not the ids the
        # optimizer held -- that difference is where relaxation gains vanish.
        runtime_ids = torch.tensor(report["re_encoded_ids"] or trigger["token_ids"],
                                   device=retriever.device)
        embeds = retriever.model.get_input_embeddings()(runtime_ids)
        values = losses_for(context, embeds, retriever, config)
    return {
        "episode_id": context.episode.episode_id,
        "snapshot_id": context.snapshot_id,
        "split": context.episode.split,
        "domain": context.episode.domain,
        "trigger": trigger["trigger"],
        "token_ids": trigger["token_ids"],
        "tokens": trigger["tokens"],
        "round_trip": report,
        "metrics": {key: float(value) for key, value in values.items()},
    }


def train(
    workspace: Workspace,
    episodes: list[Episode],
    config: TrainConfig,
    *,
    output_dir: Path,
    contract: dict[str, Any],
    resume: bool = False,
    validation: list[Episode] | None = None,
    on_step: Callable[[int, dict[str, Any]], None] | None = None,
    contexts: list[EpisodeContext] | None = None,
    initial_state: list[dict[str, Any]] | None = None,
    steps: int | None = None,
) -> tuple[dict[str, Any], list[nn.Module]]:
    """Fit ``config.mode`` over ``episodes`` and write the run's artifacts.

    Returns the summary and the fitted modules, one per episode.  For the
    shared modes every entry is the same object; for ``direct-logit`` they are
    genuinely independent, which is what makes it a per-episode baseline.

    Three optional arguments exist for M3's adaptation methods, and keeping them
    here rather than writing a second loop is deliberate: warm-start and
    re-optimization must use the exact optimization this file already defines,
    or a cost comparison between them compares two implementations instead of
    two methods.

    ``contexts``       materialize the episodes at a drifted snapshot instead of
                       at ``s0``.
    ``initial_state``  seed the modules from an earlier snapshot's parameters --
                       the warm start.  The optimizer moments are *not* carried
                       across snapshots; a fresh Adam is the conservative
                       reading, and it can only understate warm-start.
    ``steps``          a shorter budget than ``config.steps`` for the few-step
                       arm, recorded in the summary so the budget is visible.
    """
    if not episodes:
        raise ValueError("no training episodes were supplied")
    output_dir = Path(output_dir)
    retriever = workspace.retriever
    seed_everything(config.seed)
    total_steps = config.steps if steps is None else steps
    if total_steps < 0:
        raise ValueError(f"steps must not be negative, got {total_steps}")

    if contexts is None:
        contexts = [workspace.context_for(episode) for episode in episodes]
    elif len(contexts) != len(episodes):
        raise ValueError(
            f"got {len(contexts)} contexts for {len(episodes)} episodes"
        )
    if config.mode == "direct-logit":
        # B2 owns one independent logits matrix per episode; nothing is shared,
        # which is exactly what makes it a per-episode search baseline.
        modules = [_logits_source(config, retriever, episode) for episode in episodes]
    else:
        shared = _logits_source(config, retriever, episodes[0])
        modules = [shared] * len(episodes)

    if initial_state is not None:
        if len(initial_state) != len(modules):
            raise ValueError(
                f"got {len(initial_state)} initial states for {len(modules)} modules"
            )
        for module, state in zip(modules, initial_state):
            module.load_state_dict(state)

    parameters = list(dict.fromkeys(
        parameter for module in modules for parameter in module.parameters()
    ))
    optimizer = torch.optim.Adam(parameters, lr=config.learning_rate)

    checkpoint_path = output_dir / "checkpoint.pt"
    start, history = 0, []
    if checkpoint_path.exists():
        if not resume:
            raise FileExistsError(f"{checkpoint_path} exists; pass --resume")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint["contract"] != contract:
            raise ValueError(
                "resume refused: the run contract changed (config, episodes or retriever)"
            )
        for module, state in zip(modules, checkpoint["modules"]):
            module.load_state_dict(state)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start, history = checkpoint["next_step"], checkpoint["history"]
        restore_rng(checkpoint["rng"])

    for step in range(start, total_steps):
        tau = temperature(config, step, total_steps)
        optimizer.zero_grad(set_to_none=True)
        totals: dict[str, float] = {}
        for module, context in zip(modules, contexts):
            logits = _logits_for(module, config, context)
            embeds = _trigger_embeddings(logits, retriever, tau)
            values = losses_for(context, embeds, retriever, config)
            (values["l_total"] / len(contexts)).backward()
            # One backward per context: the path runs through the frozen
            # retriever, which is what makes it the expensive half of a step.
            charge("encoder_backward", 1)
            for key, value in values.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach()) / len(contexts)
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, config.grad_clip)

        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"step {step}: gradient norm is {float(grad_norm)}")
        optimizer.step()
        charge("optimizer_steps", 1)

        record = {"step": step, "tau": tau, "grad_norm": float(grad_norm)} | totals
        append_jsonl(output_dir / "metrics.jsonl", record)
        history.append(record)
        if on_step is not None:
            on_step(step, record)
        atomic_torch(checkpoint_path, {
            "schema_version": 1, "contract": contract,
            "modules": [module.state_dict() for module in modules],
            "optimizer": optimizer.state_dict(), "next_step": step + 1,
            "history": history, "rng": rng_state(),
        })

    triggers = [export_for(module, config, context, retriever)
                for module, context in zip(modules, contexts)]
    if validation:
        triggers.extend(
            export_for(modules[0], config, workspace.context_for(episode), retriever)
            for episode in validation
        )
    for trigger in triggers:
        append_jsonl(output_dir / "triggers.jsonl", trigger)

    summary = {
        "mode": config.mode, "variant": config.variant, "steps": total_steps,
        "episodes": [episode.episode_id for episode in episodes],
        "final": history[-1] if history else None,
        "valid_triggers": sum(1 for trigger in triggers if trigger["round_trip"]["valid"]),
        "triggers": len(triggers),
    }
    atomic_json(output_dir / "training.json", {"config": asdict(config)} | summary)
    return summary, modules


def build_contract(
    config: TrainConfig, manifest: dict[str, Any], retriever: Retriever
) -> dict[str, Any]:
    """What a resume must match: configuration, episode split and encoder."""
    return {
        "config_hash": stable_hash(asdict(config)),
        "split_hash": manifest["split_hash"],
        "retriever": stable_hash(retriever.fingerprint()),
    }


def load_summary(output_dir: Path) -> dict[str, Any]:
    return read_json(Path(output_dir) / "training.json")
