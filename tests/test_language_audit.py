"""Language metrics: padding, pairing, and specificity controls without downloads."""
import math
import unittest

import torch

from src.triggers.hierarchy.fluency import causal_nll
from src.triggers.hierarchy.language_audit import relevance_scores, paired_comparison, extract_records


class LanguageAuditTests(unittest.TestCase):
    def test_padding_and_first_token_do_not_affect_nll(self):
        ids = torch.tensor([[0, 1, 2, 0], [1, 0, 0, 0]])
        mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
        logits = torch.zeros(2, 4, 3)
        nll, count = causal_nll(logits, ids, mask)
        torch.testing.assert_close(nll, torch.full((2,), math.log(3)))
        self.assertEqual(count.tolist(), [2, 1])
        logits[0, 2:] = 99
        logits[1, 1:] = -20
        torch.testing.assert_close(causal_nll(logits, ids, mask)[0], nll)

    def test_next_token_alignment(self):
        ids = torch.tensor([[0, 1, 2]])
        logits = torch.zeros(1, 3, 3)
        logits[0, 0, 1] = 10
        logits[0, 1, 2] = 10
        self.assertLess(float(causal_nll(logits, ids, torch.ones_like(ids))[0][0]), .001)
        with self.assertRaises(ValueError):
            causal_nll(torch.zeros(1, 1, 3), torch.zeros(1, 1, dtype=torch.long), torch.ones(1, 1))

    def test_matched_relevance_and_constant_universal_control(self):
        queries = torch.eye(3)
        matched, contrast = relevance_scores(queries, queries)
        self.assertEqual(matched, [1., 1., 1.])
        self.assertEqual(contrast, [1., 1., 1.])
        _, universal_contrast = relevance_scores(queries, torch.tensor([[1., 0., 0.]] * 3))
        self.assertAlmostEqual(sum(universal_contrast), 0.)

    def test_pairing_uses_query_ids_not_row_order(self):
        rows = [{"qid": str(i), "trigger_query_cosine": i, "specificity_contrast": i, "nll_delta": i} for i in range(3)]
        result = paired_comparison(rows, list(reversed(rows)))
        for metric in result.values():
            self.assertEqual(metric["paired_bootstrap_95_interval"], [0., 0.])
        with self.assertRaises(ValueError):
            paired_comparison(rows, rows[:2])

    def test_train_specific_uses_learned_assignment(self):
        artifact = {"position": "suffix", "bank": {"triggers": ["first", "second"], "training_assignments": [1, 0]}}
        data = {"train": [{"qid": "a", "question": "Question A?"}, {"qid": "b", "question": "Question B?"}]}
        rows = extract_records(artifact, data, "train")
        self.assertEqual([r["trigger"] for r in rows], ["second", "first"])
        self.assertEqual(rows[0]["altered"], "Question A? second")


if __name__ == "__main__":
    unittest.main()
