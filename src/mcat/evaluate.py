"""Zero-shot evaluation on held-out episodes, plus the mechanism controls.

Everything here scores the *decoded* trigger: the export path re-tokenizes the
text and the retrieval metrics run on those ids.  A gain that exists only on
the relaxed objective never reaches this file.

Two controls decide whether conditioning does any work:

``shuffled-context``  swap one episode's memory summary for another's and keep
                      the queries. If the metrics do not move, the generator
                      is ignoring memory and the conditioning claim fails.
``permutation``       permute the document order. The output must not move,
                      because the pooling is permutation invariant.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.mcat.artifacts import atomic_json, append_jsonl
from src.mcat.encoding import encode_plain, encode_with_trigger_embeddings
from src.mcat.episodes import Episode
from src.mcat.objectives import score_matrix
from src.mcat.relaxation import export_hard_trigger, round_trip_report
from src.mcat.retrievers import Retriever
from src.mcat.runtime import EpisodeContext, Workspace
from src.mcat.train import TrainConfig, _logits_for


def retrieval_metrics(
    queries: torch.Tensor,
    clean_keys: torch.Tensor,
    poison_keys: torch.Tensor,
    *,
    top_k: int,
    score: str = "dot",
) -> dict[str, float]:
    """Rank poison against clean keys exactly as the runtime scorer would."""
    clean_scores = score_matrix(queries, clean_keys, score)
    poison_scores = score_matrix(queries, poison_keys, score)
    top_k = min(top_k, clean_keys.shape[0])

    kth_clean = clean_scores.topk(k=top_k, dim=1).values[:, -1]
    best_poison = poison_scores.max(dim=1).values
    margin = best_poison - kth_clean
    hit = (margin > 0).float()

    # Occupancy: how many of the K slots the poison records actually take.
    combined = torch.cat((clean_scores, poison_scores), dim=1)
    is_poison = torch.zeros(combined.shape[1], dtype=torch.bool, device=combined.device)
    is_poison[clean_scores.shape[1]:] = True
    order = combined.argsort(dim=1, descending=True)[:, :top_k]
    occupancy = is_poison[order].float().sum(dim=1)

    full_order = combined.argsort(dim=1, descending=True)
    poison_rank = torch.stack([
        (is_poison[row].nonzero(as_tuple=True)[0].min() + 1).float() for row in full_order
    ])
    return {
        f"hit_at_{top_k}": float(hit.mean()),
        f"poison_occupancy_at_{top_k}": float((occupancy / top_k).mean()),
        "mean_margin": float(margin.mean()),
        "p10_margin": float(margin.quantile(0.10)),
        "worst_margin": float(margin.min()),
        "mean_poison_rank": float(poison_rank.mean()),
        "mrr": float((1.0 / poison_rank).mean()),
        "queries": int(queries.shape[0]),
    }


def _poison_keys(
    context: EpisodeContext, trigger_ids: torch.Tensor, retriever: Retriever
) -> torch.Tensor:
    embeds = retriever.model.get_input_embeddings()(trigger_ids)
    return encode_with_trigger_embeddings(
        retriever.model, retriever.tokenizer, context.poison_texts, embeds,
        device=retriever.device, max_length=retriever.max_length,
    )


def evaluate_trigger(
    workspace: Workspace,
    context: EpisodeContext,
    trigger: dict[str, Any],
    *,
    score: str = "dot",
) -> dict[str, Any]:
    """Score one frozen trigger on this episode's locked evaluation queries."""
    retriever = workspace.retriever
    texts = workspace.eval_texts(context.episode)
    ids = trigger["round_trip"]["re_encoded_ids"] or trigger["token_ids"]
    trigger_ids = torch.tensor(ids, device=retriever.device)
    top_k = context.episode.retrieval_top_k

    with torch.no_grad():
        poison = _poison_keys(context, trigger_ids, retriever)
        embeds = retriever.model.get_input_embeddings()(trigger_ids)
        triggered = encode_with_trigger_embeddings(
            retriever.model, retriever.tokenizer, texts, embeds,
            device=retriever.device, max_length=retriever.max_length,
        )
        # Trigger off: the poison records are already written, but the query
        # carries no trigger.  Anything retrieved here is a false activation.
        untriggered = encode_plain(
            retriever.model, retriever.tokenizer, texts,
            device=retriever.device, max_length=retriever.max_length,
        )
        on = retrieval_metrics(triggered, context.clean_keys, poison,
                               top_k=top_k, score=score)
        off = retrieval_metrics(untriggered, context.clean_keys, poison,
                                top_k=top_k, score=score)
    return {
        "episode_id": context.episode.episode_id,
        "domain": context.episode.domain,
        "split": context.episode.split,
        "trigger": trigger["trigger"],
        "round_trip_valid": bool(trigger["round_trip"]["valid"]),
        "trigger_on": on,
        "trigger_off": off,
        "false_activation": off[f"hit_at_{min(top_k, context.clean_keys.shape[0])}"],
    }


def generate_trigger(
    module: nn.Module, config: TrainConfig, context: EpisodeContext, retriever: Retriever,
    *, memory_override: torch.Tensor | None = None,
) -> dict[str, Any]:
    """One deterministic forward pass; ``memory_override`` drives the controls."""
    if memory_override is not None:
        context = EpisodeContext(
            episode=context.episode, memory_vectors=memory_override,
            support_vectors=context.support_vectors,
            reference_centers=context.reference_centers,
            optimization_texts=context.optimization_texts,
            poison_texts=context.poison_texts,
        )
    with torch.no_grad():
        logits = _logits_for(module, config, context)
        trigger = export_hard_trigger(logits, retriever.tokenizer, retriever.vocab_mask)
        trigger["round_trip"] = round_trip_report(trigger, retriever.tokenizer)
    return trigger


def shuffled_context_control(
    modules: list[nn.Module],
    config: TrainConfig,
    contexts: list[EpisodeContext],
    retriever: Retriever,
) -> dict[str, Any]:
    """Rotate memory summaries between episodes, holding queries fixed.

    Identical triggers before and after the swap mean the memory branch is
    inert -- the finding that would sink the conditioning claim, so it is
    reported as a number rather than left to inspection.

    Swaps stay **within a domain**.  Handing a StrategyQA episode an EhrAgent
    snapshot is a distribution shift a generator could notice without having
    learned anything about memory state, which would inflate the control.  The
    honest question is whether it distinguishes two snapshots of the same kind.
    """
    if config.mode != "generator":
        return {"applicable": False, "reason": "needs a generator"}

    by_domain: dict[str, list[int]] = {}
    for index, context in enumerate(contexts):
        by_domain.setdefault(context.episode.domain, []).append(index)
    usable = {name: group for name, group in by_domain.items() if len(group) >= 2}
    if not usable:
        return {"applicable": False,
                "reason": "needs >= 2 episodes from one domain to swap within"}

    rows, per_domain = [], {}
    for name, group in sorted(usable.items()):
        changed = 0
        for position, index in enumerate(group):
            context = contexts[index]
            other = contexts[group[(position + 1) % len(group)]]
            original = generate_trigger(modules[index], config, context, retriever)
            swapped = generate_trigger(modules[index], config, context, retriever,
                                       memory_override=other.memory_vectors)
            differs = original["token_ids"] != swapped["token_ids"]
            changed += int(differs)
            rows.append({"episode_id": context.episode.episode_id, "domain": name,
                         "swapped_with": other.episode.episode_id,
                         "original": original["trigger"], "swapped": swapped["trigger"],
                         "changed": differs})
        per_domain[name] = {"episodes": len(group), "changed": changed,
                            "change_rate": changed / len(group)}

    episodes = sum(entry["episodes"] for entry in per_domain.values())
    changed = sum(entry["changed"] for entry in per_domain.values())
    skipped = sorted(set(by_domain) - set(usable))
    return {"applicable": True, "episodes": episodes, "changed": changed,
            "change_rate": changed / episodes, "per_domain": per_domain,
            "skipped_domains": skipped, "rows": rows}


def permutation_control(
    modules: list[nn.Module],
    config: TrainConfig,
    contexts: list[EpisodeContext],
    retriever: Retriever,
    *,
    seed: int = 0,
) -> dict[str, Any]:
    """Permuting the snapshot must leave the trigger unchanged."""
    if config.mode != "generator":
        return {"applicable": False, "reason": "needs a generator"}
    generator = torch.Generator().manual_seed(seed)
    stable = 0
    for module, context in zip(modules, contexts):
        index = torch.randperm(context.memory_vectors.shape[0], generator=generator)
        original = generate_trigger(module, config, context, retriever)
        permuted = generate_trigger(module, config, context, retriever,
                                    memory_override=context.memory_vectors[index])
        stable += int(original["token_ids"] == permuted["token_ids"])
    return {"applicable": True, "episodes": len(contexts), "stable": stable,
            "stable_rate": stable / len(contexts) if contexts else 0.0}


def evaluate(
    workspace: Workspace,
    modules: list[nn.Module],
    config: TrainConfig,
    episodes: list[Episode],
    *,
    output_dir: Path,
    score: str = "dot",
    controls: bool = True,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    retriever = workspace.retriever
    contexts = [workspace.context_for(episode) for episode in episodes]
    if len(modules) != len(contexts):
        raise ValueError(
            f"expected one module per episode, got {len(modules)} for {len(contexts)}"
        )

    rows = []
    for module, context in zip(modules, contexts):
        trigger = generate_trigger(module, config, context, retriever)
        row = evaluate_trigger(workspace, context, trigger, score=score)
        append_jsonl(output_dir / "evaluation.jsonl", row)
        rows.append(row)

    keys = [key for key in rows[0]["trigger_on"] if key != "queries"] if rows else []
    summary: dict[str, Any] = {
        "mode": config.mode, "variant": config.variant, "score": score,
        "episodes": len(rows),
        "trigger_on": {key: _mean(row["trigger_on"][key] for row in rows) for key in keys},
        "trigger_off": {key: _mean(row["trigger_off"][key] for row in rows) for key in keys},
        "false_activation": _mean(row["false_activation"] for row in rows),
        "round_trip_valid_rate": _mean(float(row["round_trip_valid"]) for row in rows),
    }
    if controls:
        summary["controls"] = {
            "shuffled_context": shuffled_context_control(modules, config, contexts, retriever),
            "permutation": permutation_control(modules, config, contexts, retriever),
        }
    atomic_json(output_dir / "evaluation.json", {"config": asdict(config)} | summary)
    return summary


def _mean(values) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0
