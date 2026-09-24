"""Contracts for fixed-budget hierarchy experiments; no model downloads."""
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import torch

from src.triggers.clustering import assign, fit_centers
from src.triggers.hierarchy.backends import Encoder
from src.triggers.hierarchy.data import fixture_data, load_data
from src.triggers.hierarchy.evaluation import evaluate_bank, routed_groups
from src.triggers.hierarchy.experiment import compare, fit_bank
from src.triggers.hierarchy.hierarchy import (build_tree, converged_level, effect_signatures,
                                               merge_signatures)
from src.triggers.hierarchy.io import write
from src.triggers.hierarchy.quality import MeaningGuard
from src.triggers.hierarchy.search import Budget, Objective
from src.triggers.hierarchy.text import candidates, insert, proposals, actual_position, SEEDS


class ClusteringTests(unittest.TestCase):
    def test_fits_means_instead_of_sampled_points(self):
        x = torch.tensor([[0., 0.], [0., 2.], [10., 0.], [10., 2.]])
        centers = fit_centers(x, 2, seed=3)
        centers = centers[centers[:, 0].argsort()]
        torch.testing.assert_close(centers, torch.tensor([[0., 1.], [10., 1.]]))
        torch.testing.assert_close(fit_centers(x, 2, seed=3), fit_centers(x, 2, seed=3))

    def test_effect_tree_merges_similar_effects_first(self):
        signatures = torch.tensor([[1., 0.], [.99, .01], [-1., 0.]])
        signatures = torch.nn.functional.normalize(signatures, dim=1)
        tree = build_tree(signatures)
        self.assertEqual(set(tree[0]["leaves"]), {0, 1})
        self.assertEqual(set(tree[-1]["leaves"]), {0, 1, 2})


class TextTests(unittest.TestCase):
    def test_insert_positions_and_sentence_fallback(self):
        q = "Read record 12. Is the road closed?"
        self.assertEqual(insert(q, "use relevant information", "sentence"),
                         "Read record 12. use relevant information Is the road closed?")
        self.assertTrue(insert(q, "use relevant information", "suffix").startswith(q))
        self.assertTrue(insert("Is it closed?", "use relevant information", "sentence").endswith("Is it closed?"))

    def test_mix_respects_fixed_word_cap(self):
        values = list(candidates("use relevant information", random.Random(1), 3, ["check useful details"]))
        self.assertTrue(values)
        self.assertTrue(all(len(t.split()) <= 3 for t in values))

    def test_infix_is_actually_inside_a_single_sentence(self):
        q = "Is this bicycle allowed on the main road?"
        self.assertEqual(actual_position(q, "infix"), "infix")
        self.assertIn("use relevant information", insert(q, "use relevant information", "infix"))
        self.assertEqual(actual_position(q, "sentence"), "prefix")

    def test_template_mutation_and_crossover_origins(self):
        candidates = list(proposals("use relevant information", random.Random(1), 3, ["check useful details"]))
        self.assertEqual(candidates[0][1], "crossover")
        self.assertTrue(any(t not in {"use relevant information", "check useful details"} and kind == "crossover" for t, kind in candidates))
        slots = [set(s.split()[i] for s in SEEDS) for i in range(3)]
        for t, _ in candidates:
            self.assertTrue(all(w in slots[i] for i, w in enumerate(t.split())))

    def test_protected_values_and_independent_semantics(self):
        guard = MeaningGuard(Encoder(fixture_seed=7), threshold=0.)
        result = guard.check(["Is record 12 not closed?"], ["Is record 13 not closed?"])[0]
        self.assertFalse(result["meaning_proxy_pass"])
        self.assertFalse(result["protected_tokens_unchanged"])


class HierarchyExperimentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.data = fixture_data(4, 4, 4, 4)
        self.encoder = Encoder(fixture_seed=42)
        self.guard = MeaningGuard(Encoder(fixture_seed=1042), .7)
        clean = self.encoder.encode(self.data["corpus"])
        # The fixture gives four poison keys, so a group owns at most two and the
        # per_query arm owns exactly one.  Upstream-strict top_k would then ask the
        # 5th poison key to win a hinge that has no 5th key.  Clamping lowers only the
        # training hinge, never the reported hit@top_k, which is what lets every arm
        # run on the same small fixture.
        self.objective = Objective(self.encoder, self.guard, clean, fit_centers(clean, 3), position="suffix",
                                   top_k_policy="clamp")

    def test_all_arms_share_poison_count_and_budget_cap(self):
        for arm in ("universal", "random", "semantic", "per_query", "hierarchical_merge", "hierarchical_mix"):
            with self.subTest(arm=arm):
                bank = fit_bank(arm, self.data, self.objective, 256, 2, 42)
                self.assertLessEqual(bank["budget_used"], 256)
                report = evaluate_bank(bank, self.data["test"], self.data["sources"], self.objective)
                self.assertEqual(len(report["poison_keys"]), 4)
                self.assertEqual(report["metrics"]["n"], 4)
                self.assertLessEqual(report["metrics"]["joint_success_rate"], report["metrics"]["retrieval_rate"])
                if "hierarchy" in bank:
                    self.assertGreater(bank["hierarchy"]["signature_budget_used"], 0)
                    self.assertEqual(len(bank["hierarchy"]["merges"]), 3)
                    for cut in bank["hierarchy"]["cuts"].values():
                        self.assertEqual(len(cut["source_groups"]), 4)
                        self.assertEqual(set(cut["source_groups"]), set(range(len(cut["triggers"]))))

    def test_per_query_heldout_routing_does_not_optimize_test(self):
        bank = fit_bank("per_query", self.data, self.objective, 128, 2, 42)
        unseen = self.data["test"]
        embeddings = self.encoder.encode([r["question"] for r in unseen])
        expected = assign(embeddings, torch.tensor(bank["centers"]))
        torch.testing.assert_close(routed_groups(bank, unseen, embeddings), expected)

    def test_budget_charges_effect_construction_and_refuses_overrun(self):
        budget = Budget(2)
        with self.assertRaisesRegex(ValueError, "signatures"):
            effect_signatures(self.encoder, ["use relevant information"], self.data["train"][:2], "suffix", budget)
        self.assertEqual(budget.used, 0)

    def test_completed_arms_resume_without_training(self):
        with tempfile.TemporaryDirectory() as folder:
            kwargs = dict(positions=["suffix"], arms=["universal"], groups=2, budget=128, seed=42, contract={"test": 1})
            first = compare(self.data, lambda p: self.objective, Path(folder), **kwargs)
            with patch("src.triggers.hierarchy.experiment.fit_bank", side_effect=AssertionError("resumed training")):
                second = compare(self.data, lambda p: self.objective, Path(folder), **kwargs)
            self.assertEqual(first, second)
            with self.assertRaisesRegex(ValueError, "changed"):
                compare(self.data, lambda p: self.objective, Path(folder), **{**kwargs, "contract": {"test": 2}})

    def test_bottom_up_arms_report_every_level_and_stop_on_training_loss(self):
        for arm in ("hierarchical_merge", "hierarchical_query", "hierarchical_both"):
            with self.subTest(arm=arm):
                bank = fit_bank(arm, self.data, self.objective, 256, 2, 42)
                hierarchy = bank["hierarchy"]
                # Four leaves means a level for every merge depth, leaves through root.
                self.assertEqual(sorted(int(k) for k in hierarchy["loss_curve"]), [1, 2, 3, 4])
                self.assertEqual(set(hierarchy["loss_curve"]), set(hierarchy["cuts"]))
                stopping = hierarchy["stopping"]
                self.assertIn(stopping["converged_level"], [int(k) for k in hierarchy["loss_curve"]])
                self.assertEqual(stopping["basis"],
                                 {"hierarchical_merge": "effect", "hierarchical_query": "query",
                                  "hierarchical_both": "both"}[arm])

    def test_stopping_reads_training_loss_only_and_halts_when_merging_costs(self):
        # Loss flat from 4 to 3, then a jump well past the tolerance: stop at 3.
        curve = {"4": 1.0, "3": 1.02, "2": 1.6, "1": 1.7}
        self.assertEqual(converged_level(curve, .05), 3)
        # A tolerance wide enough to absorb every rise merges all the way to the root.
        self.assertEqual(converged_level(curve, 1.0), 1)
        # Loss that keeps falling never triggers the rule.
        self.assertEqual(converged_level({"3": 2.0, "2": 1.5, "1": 1.0}, 0), 1)

    def test_merge_basis_selects_which_side_is_clustered(self):
        effect = torch.tensor([[1., 0.], [0., 1.]])
        queries = torch.tensor([[0., 2.], [3., 0.]])
        self.assertTrue(torch.allclose(merge_signatures("effect", effect, queries), effect))
        self.assertTrue(torch.allclose(merge_signatures("query", effect, queries),
                                       torch.tensor([[0., 1.], [1., 0.]])))
        self.assertEqual(merge_signatures("both", effect, queries).shape, (2, 4))
        with self.assertRaisesRegex(ValueError, "basis"):
            merge_signatures("nearest", effect, queries)

    def test_false_activation_separates_the_trigger_from_question_shaped_keys(self):
        bank = fit_bank("universal", self.data, self.objective, 256, 2, 42)
        report = evaluate_bank(bank, self.data["test"], self.data["sources"], self.objective)
        metrics = report["metrics"]
        # The baseline reuses the same sources with no trigger, so whatever it already
        # retrieves cannot be charged to the trigger.
        self.assertLessEqual(metrics["false_activation_attributable"], metrics["false_activation"])
        for record in report["records"]:
            if record["false_activation_attributable"]:
                self.assertTrue(record["false_activation_without_trigger"])
                self.assertFalse(record["false_activation_untriggered_keys"])

    def test_data_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write(root / "train.json", [{"qid": "a", "question": "Question?"}])
            write(root / "test.json", [{"qid": "b", "question": "Question?"}])
            write(root / "corpus.json", {"doc": {"content": "Evidence"}})
            with self.assertRaisesRegex(ValueError, "overlap"):
                load_data(root / "train.json", root / "test.json", root / "corpus.json", 1, 1, 1, 1, 42)


if __name__ == "__main__":
    unittest.main()
