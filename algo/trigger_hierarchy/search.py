"""Budgeted bounded discrete search, with the existing AgentPoison losses."""
from dataclasses import dataclass
from collections import Counter
import random

import torch

from algo.trigger_losses import compute_compactness_loss, compute_retrieval_margin_loss, compute_uniqueness_loss
from .text import SEEDS, proposals, insert


@dataclass
class Budget:
    limit: int
    used: int = 0

    def spend(self, amount):
        if self.used + amount > self.limit:
            return False
        self.used += amount
        return True


class Objective:
    def __init__(self, encoder, guard, clean, centers, *, position, top_k=5, margin=.1,
                 max_words=3, max_tokens=12, candidate_style="template"):
        self.encoder, self.guard, self.clean, self.centers = encoder, guard, clean, centers
        self.position, self.top_k, self.margin = position, top_k, margin
        self.max_words, self.max_tokens = max_words, max_tokens
        self.candidate_style = candidate_style

    def valid(self, trigger):
        return 0 < len(trigger.split()) <= self.max_words and self.encoder.token_count(trigger) <= self.max_tokens

    def score(self, trigger, rows, sources, budget):
        if not self.valid(trigger):
            return None
        if not budget.spend(len(rows) + len(sources)):
            return None
        original = [r["question"] for r in rows]
        altered = [insert(q, trigger, self.position) for q in original]
        query = self.encoder.encode(altered)
        poison = self.encoder.encode([insert(r["question"], trigger, self.position) for r in sources])
        quality = self.guard.check(original, altered)
        valid = all(q["meaning_proxy_pass"] for q in quality)
        uni = compute_uniqueness_loss(query, self.centers)
        compact = compute_compactness_loss(query)
        margin = compute_retrieval_margin_loss(query, self.clean, poison, self.top_k, self.margin)
        loss = float(uni + .1 * compact + margin)
        return {"trigger": trigger, "loss": loss, "valid": valid,
                "semantic_similarity": sum(q["semantic_similarity"] for q in quality) / len(quality)}


def optimize(objective, rows, sources, budget, *, seed, initial=None, parents=()):
    rng = random.Random(seed)
    best, fallback, evaluations = None, None, 0
    queue = [(t, "parent_seed" if initial else "seed") for t in dict.fromkeys([*(initial or SEEDS), *parents])]
    history, offered = [], Counter()
    attempted, stalled = set(), 0
    cost = len(rows) + len(sources)
    while budget.used + cost <= budget.limit:
        if not queue:
            base = best["trigger"] if best else (fallback["trigger"] if fallback else rng.choice(SEEDS))
            queue = list(proposals(base, rng, objective.max_words, parents, objective.candidate_style))
        trigger, origin = queue.pop(0)
        offered[origin] += 1
        if trigger in attempted or not objective.valid(trigger):
            stalled += 1
            if stalled > 100:
                break
            continue
        attempted.add(trigger)
        result = objective.score(trigger, rows, sources, budget)
        if result is None:
            continue
        evaluations += 1
        stalled = 0
        improved = bool(result["valid"] and (best is None or result["loss"] < best["loss"]))
        result["origin"] = origin
        history.append({**result, "improved_feasible_best": improved, "budget_used": budget.used})
        if fallback is None or result["loss"] < fallback["loss"]:
            fallback = result
        if result["valid"] and (best is None or result["loss"] < best["loss"]):
            best = result
    if fallback is None:
        raise ValueError("Budget too small to evaluate any valid-length trigger")
    chosen = best or fallback
    return {**chosen, "constraint_feasible": best is not None, "evaluations": evaluations,
            "budget_used": budget.used, "budget_limit": budget.limit, "history": history,
            "proposals_offered": dict(offered), "origins_evaluated": dict(Counter(r["origin"] for r in history))}


def allocate(total, groups):
    if groups < 1 or total < groups:
        raise ValueError("Budget must cover all groups")
    return [total // groups + int(i < total % groups) for i in range(groups)]
