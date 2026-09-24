"""Grounding, exact rendering, disjoint evaluation and language-search contracts."""
import tempfile
import unittest
from pathlib import Path

import torch

from src.triggers.specificity.candidates import phrases, pool, render, copy_metrics, words
from src.triggers.specificity.selection import Selector
from src.triggers.specificity.phrases import noun_spans
from src.triggers.specificity.__main__ import fresh_partition, choose_on_validation
from src.triggers.hierarchy.backends import Encoder
from src.triggers.hierarchy.io import write
from src.triggers.hierarchy.text import protected_tokens, SEEDS


class SpecificityTests(unittest.TestCase):
    def test_grounded_spans_do_not_cross_stopwords_or_copy_negation_numbers(self):
        q = "Did Tony Blair not visit London in 2011?"
        p = phrases(q)
        self.assertIn("Tony Blair", p)
        self.assertNotIn("Blair visit", p)
        self.assertTrue(all(not protected_tokens(t) for t in p))
        self.assertTrue(all(len(words(t)) <= 3 for t in pool([q], "grounded_language")))

    def test_renderer_preserves_query_exactly_and_distinguishes_format_control(self):
        q = "Can Tony Blair enter?"
        self.assertEqual(render(q, "About Tony Blair"), "About Tony Blair: Can Tony Blair enter?")
        self.assertEqual(render(q, "use useful details", "raw"), "use useful details Can Tony Blair enter?")
        self.assertEqual(render(q, "Tony Blair").split(": ", 1)[1], q)

    def test_copy_measure_does_not_hide_topic_repetition(self):
        m = copy_metrics("Can Tony Blair enter?", "About Tony Blair")
        self.assertEqual(m["trigger_content_copy_fraction"], 1.)
        self.assertEqual(m["copied_content_words"], 2)

    def test_fresh_holdout_excludes_prior_evaluated_queries_and_strips_answers(self):
        previous = {k: [{"qid": k, "question": f"Question {k}?"}] for k in ("train", "validation", "test", "sources")}
        rows = [r for split in previous.values() for r in split]
        rows += [{"qid": "duplicate_text", "question": "Question test?"}]
        rows += [{"qid": str(i), "question": f"New topic {i}?", "answer": True} for i in range(5)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.json"
            write(path, rows)
            result = fresh_partition(previous, path, 3, 42)
            self.assertEqual(result, fresh_partition(previous, path, 3, 42))
            self.assertTrue(all(set(r) == {"qid", "question"} for r in result))
            self.assertTrue(all(r["qid"].isdigit() for r in result))

    def test_validation_selection_rejects_high_score_with_bad_fluency(self):
        metrics = {"trigger_query_cosine": .5, "specificity_contrast": 0., "nll_delta": 0.,
                   "preservation_proxy_pass_rate": 1., "geometric_mean_ppl_ratio": 1.}
        results = {}
        for scope in ("universal", "semantic", "per_query"):
            results[f"{scope}/good"] = {"metrics": metrics}
            results[f"{scope}/bad"] = {"metrics": {**metrics, "trigger_query_cosine": .9, "geometric_mean_ppl_ratio": 3.}}
        choices, details = choose_on_validation(results)
        self.assertTrue(all(name.endswith("good") for name in choices.values()))
        self.assertTrue(all(r["validation_constraints_met"] for r in details.values()))

    def test_selector_budget_and_fallback_are_explicit(self):
        torch.set_num_threads(2)
        class FlatLM:
            def score(self, texts):
                return [{"nll": 1., "ppl": 2.718, "scored_tokens": 3} for _ in texts]
        selector = Selector(Encoder(fixture_seed=42), FlatLM(), candidate_cap=8, meaning_threshold=1.)
        result = selector.fit([{"question": "Can Tony Blair enter London?"}], "grounded_language")
        self.assertEqual(result["candidates_scored"], 8)
        self.assertEqual(result["modified_texts_scored"], 8)
        self.assertFalse(result["constraint_feasible"])
        self.assertEqual(len(result["history"]), 8)

    def test_pos_filter_rejects_predicates_and_partial_names(self):
        q = "Is chaff produced near Tony Blair?"
        import re
        tags = ["AUX", "NOUN", "VERB", "ADP", "PROPN", "PROPN"]
        tagged = [{"start": w.start(), "end": w.end(), "tag": tag} for w, tag in zip(re.finditer(r"\w+", q), tags)]
        spans = noun_spans(q, tagged)
        self.assertIn("chaff", spans)
        self.assertIn("Tony Blair", spans)
        self.assertNotIn("chaff produced", spans)
        self.assertNotIn("Blair", spans)

    def test_group_candidate_cannot_narrow_to_one_unrelated_member(self):
        class ControlledEncoder:
            def encode(self, texts):
                vectors = []
                for t in texts:
                    if t.endswith("Alpha question?") or "Alpha" in t:
                        vectors.append([1., 0.])
                    elif t.endswith("Beta question?") or "Beta" in t:
                        vectors.append([0., 1.])
                    else:
                        vectors.append([-.1, .9949874])
                return torch.tensor(vectors)
            def token_count(self, text):
                return len(text.split())
        class FlatLM:
            def score(self, texts):
                return [{"nll": 1.} for _ in texts]
        selector = Selector(ControlledEncoder(), FlatLM(), phrase_extractor=lambda q: [q.split()[0]])
        result = selector.fit([{"question": "Alpha question?"}, {"question": "Beta question?"}], "grounded_language")
        self.assertIn(result["trigger"], SEEDS)
        topics = [h for h in result["history"] if h["trigger"] not in SEEDS]
        self.assertTrue(topics)
        self.assertTrue(all(not h["contextual_coverage_pass"] for h in topics))


if __name__ == "__main__":
    unittest.main()
