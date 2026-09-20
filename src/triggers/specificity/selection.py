"""Bounded language search: relevance, group coverage, fluency and preservation."""
import math
import time

import torch

from src.triggers.hierarchy.text import protected_tokens, SEEDS
from .candidates import pool, render, words, phrases


class Selector:
    def __init__(self, encoder, fluency, *, candidate_cap=20, meaning_threshold=.85, ppl_limit=1.5,
                 phrase_extractor=phrases, min_grounded_relevance=.15, position="prefix"):
        if candidate_cap < len(SEEDS):
            raise ValueError("Candidate cap must cover the generic seeds")
        if not 0 <= meaning_threshold <= 1 or not math.isfinite(ppl_limit) or ppl_limit <= 0:
            raise ValueError("Invalid language constraints")
        self.encoder, self.fluency = encoder, fluency
        self.cap, self.threshold, self.ppl_limit = candidate_cap, meaning_threshold, ppl_limit
        self.phrase_extractor, self.min_grounded_relevance = phrase_extractor, min_grounded_relevance
        self.position = position

    def fit(self, rows, mode, counter_rows=()):
        started = time.perf_counter()
        queries = [r["question"] for r in rows]
        negatives = [r["question"] for r in counter_rows if r["question"] not in queries]
        qv = self.encoder.encode(queries)
        candidates = [t for t in pool(queries, mode, self.phrase_extractor) if len(words(t)) <= 3 and self.encoder.token_count(t) <= 12]
        if not candidates:
            raise ValueError("No valid candidates")
        tv = self.encoder.encode(candidates)
        similarity = tv @ qv.T
        contrast = similarity.mean(1) - (tv @ self.encoder.encode(negatives).T).mean(1) if negatives else torch.zeros(len(tv))
        screening = similarity.mean(1) + .25 * similarity.min(1).values + .25 * contrast
        order = sorted(range(len(candidates)), key=lambda i: (-float(screening[i]), candidates[i]))
        # Keep generic seeds available as fallbacks; the remainder is screened by
        # topic relevance before expensive full-sentence scoring.
        seeds = [i for i, t in enumerate(candidates) if t in SEEDS]
        chosen = list(dict.fromkeys([*seeds, *order]))[:self.cap]
        rendered = [render(q, candidates[i], position=self.position) for i in chosen for q in queries]
        av = self.encoder.encode(rendered).reshape(len(chosen), len(queries), -1)
        preservation = (av * qv.unsqueeze(0)).sum(-1)
        lm = self.fluency.score(rendered)
        original_lm = self.fluency.score(queries)
        history = []
        for rank, i in enumerate(chosen):
            delta = sum(lm[rank * len(queries) + j]["nll"] - original_lm[j]["nll"] for j in range(len(queries))) / len(queries)
            kept = all(protected_tokens(q) == protected_tokens(render(q, candidates[i], position=self.position)) for q in queries)
            coverage = candidates[i] in SEEDS or float(similarity[i].min()) >= self.min_grounded_relevance
            feasible = kept and coverage and float(preservation[rank].min()) >= self.threshold and delta <= math.log(self.ppl_limit)
            history.append({"trigger": candidates[i], "relevance": float(similarity[i].mean()),
                            "worst_relevance": float(similarity[i].min()), "contrast": float(contrast[i]),
                            "min_preservation": float(preservation[rank].min()), "nll_delta": delta,
                            "score": float(screening[i]) - .25 * max(0., delta), "feasible": feasible,
                            "contextual_coverage_pass": coverage})
        eligible = [h for h in history if h["feasible"]]
        best = max(eligible or history, key=lambda h: (h["score"], h["trigger"]))
        return {**best, "constraint_feasible": bool(eligible), "candidate_pool_size": len(candidates),
                "candidates_scored": len(chosen), "modified_texts_scored": len(rendered),
                "seconds": time.perf_counter() - started, "history": history}
