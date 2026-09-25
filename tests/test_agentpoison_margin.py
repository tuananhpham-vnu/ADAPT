import json
from pathlib import Path
import tempfile
import unittest

import torch

from src.triggers.scorers import (
    coherence_sampling_probabilities, geometric_target_probability, select_candidate,
)
from src.triggers.losses import (
    compute_compactness_loss, compute_retrieval_margin_loss,
    compute_total_loss, compute_uniqueness_loss,
)
from src.triggers.margin import main
from src.triggers.clustering import fit_centers


class TriggerLossTests(unittest.TestCase):
    def test_loss_signs_and_graph(self):
        query = torch.tensor([[0.0, 0.0], [2.0, 0.0]], requires_grad=True)
        centers = torch.tensor([[0.0, 0.0]])
        self.assertAlmostEqual(float(compute_uniqueness_loss(query, centers).detach()), -1.0)
        self.assertAlmostEqual(float(compute_compactness_loss(query).detach()), 1.0)
        total = compute_total_loss(
            compute_uniqueness_loss(query, centers), compute_compactness_loss(query),
            torch.tensor(0.0), .1, 0,
        )
        total.backward()
        self.assertIsNotNone(query.grad)

    def test_kth_poison_margin(self):
        query = torch.tensor([[1.0, 0.0]])
        clean = torch.tensor([[.85, (1 - .85 ** 2) ** .5], [0.0, 1.0]])
        poison_scores = [1.0, .98, .96, .94, .92]
        poison = torch.tensor([[score, (1 - score ** 2) ** .5] for score in poison_scores])
        self.assertEqual(float(compute_retrieval_margin_loss(query, clean, poison, 1, .1)), 0)
        self.assertGreater(float(compute_retrieval_margin_loss(query, clean, poison, 5, .1)), 0)

    def test_gmm_centers_match_upstream_configuration(self):
        try:
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("scikit-learn is not installed in the lightweight test venv")
        torch.manual_seed(4)
        vectors = torch.cat((torch.randn(20, 2) - 3, torch.randn(20, 2) + 3))
        centers = fit_centers(vectors, count=2, seed=0)
        self.assertEqual(tuple(centers.shape), (2, 2))
        self.assertTrue(torch.allclose(
            centers[centers[:, 0].argsort()],
            torch.tensor([[-3.0, -3.0], [3.0, 3.0]]),
            atol=1.0,
        ))

    def test_constraints(self):
        probabilities = coherence_sampling_probabilities(torch.tensor([1.0, 3.0]))
        self.assertGreater(float(probabilities[0]), float(probabilities[1]))
        logs = torch.log(torch.tensor([[.25, 1.0]]))
        self.assertAlmostEqual(float(geometric_target_probability(logs)), .5)
        self.assertIsNone(select_candidate(1.0, .85, [.5], [.79], .8))
        self.assertEqual(select_candidate(1.0, .7, [.5], [.75], .8), 0)


class StagedSmokeTests(unittest.TestCase):
    def test_smoke_creates_complete_contract_and_resumes(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            argv = ["smoke", "--output-dir", str(output), "--smoke-fixture",
                    "--num-iter", "2", "--batch-size", "4",
                    "--replacement-candidates", "20", "--subsample-candidates", "5"]
            main(argv)
            expected = {"config.json", "split.json", "checkpoint.pt", "metrics.jsonl",
                        "trigger.json", "retrieval_evaluation.json", "agent_evaluation.json", "REPORT.md"}
            self.assertTrue(expected.issubset({path.name for path in output.iterdir()}))
            self.assertTrue((output / "index/checkpoint.json").exists())
            self.assertEqual(json.loads((output / "trigger.json").read_text())["state"], "completed")
            self.assertEqual(len((output / "metrics.jsonl").read_text().splitlines()), 2)
            main(argv)
            self.assertEqual(len((output / "metrics.jsonl").read_text().splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
