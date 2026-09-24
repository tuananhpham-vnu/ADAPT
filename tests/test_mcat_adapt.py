"""Adaptation methods, paired statistics and the drift stages.

The comparison M3 exists to make is only worth anything if the four arms are
run under equal conditions, so most of these check fairness rather than
behaviour: equal write budgets, a warm start that really starts warm, a
baseline that cannot silently disappear, and an interval that resamples
episodes instead of queries.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

import torch

from src.triggers.mcat.adapt import (
    ADAPT_METHODS, AdaptConfig, adapt_all, adapt_trigger, fresh_module,
)
from src.triggers.mcat.cli import main
from src.triggers.mcat.costs import CostLedger
from src.triggers.mcat.drift import Snapshot
from src.triggers.mcat.drift_eval import DRIFT_ROWS, DRIFT_SUMMARY, aggregate, load_rows
from src.triggers.mcat.episodes import Episode
from src.triggers.mcat.retrievers import build_fixture_retriever, fixture_text
from src.triggers.mcat.runtime import EpisodeContext, Workspace
from src.triggers.mcat.stats import macro_micro_worst, paired_bootstrap
from src.triggers.mcat.train import TrainConfig

MAX_LENGTH = 32


class AdaptConfigTests(unittest.TestCase):
    def test_defaults_cover_every_method(self):
        self.assertEqual(AdaptConfig().methods, ADAPT_METHODS)

    def test_invalid_configuration_is_refused(self):
        with self.assertRaises(ValueError):
            AdaptConfig(methods=("teleport",))
        with self.assertRaises(ValueError):
            AdaptConfig(methods=())
        with self.assertRaises(ValueError):
            AdaptConfig(steps=-1)
        with self.assertRaises(ValueError):
            AdaptConfig(write_policy="sometimes")

    def test_the_fingerprint_follows_the_few_step_budget(self):
        # The budget is a free parameter, so a run that changes it must not be
        # able to resume on top of a run that used another one.
        self.assertNotEqual(AdaptConfig(steps=5).fingerprint(),
                            AdaptConfig(steps=10).fingerprint())
        self.assertNotEqual(AdaptConfig(write_policy="fixed").fingerprint(),
                            AdaptConfig(write_policy="refresh").fingerprint())


class AdaptMethodTests(unittest.TestCase):
    """One episode, one drifted snapshot, four ways to have a trigger for it."""

    def setUp(self):
        self.retriever = build_fixture_retriever(max_length=MAX_LENGTH)
        self.directory = Path(tempfile.mkdtemp(prefix="mcat-adapt-"))
        self.addCleanup(shutil.rmtree, self.directory, True)
        self.config = TrainConfig(mode="generator", variant="memory+query", steps=2,
                                  context_dim=8, hidden=16, memory_summary_keys=8)
        self.adapt_config = AdaptConfig(steps=1)
        self.documents = [{"doc_id": f"d-{i:03d}", "text": fixture_text(i),
                           "family": f"fam-{i}"} for i in range(12)]
        self.queries = [{"qid": qid, "question": fixture_text(index),
                         "family": f"q-{index}"}
                        for index, qid in enumerate(["s0", "s1", "o0", "e0", "p0", "p1"])]
        self.workspace = Workspace(retriever=self.retriever, cache_dir=self.directory,
                                   fixture=True)
        torch.manual_seed(0)
        self.workspace._documents["qa"] = self.documents
        self.workspace._queries["qa"] = self.queries
        self.workspace._doc_vectors["qa"] = torch.randn(12, self.retriever.hidden_size)
        self.workspace._query_vectors["qa"] = torch.randn(6, self.retriever.hidden_size)
        self.episode = Episode(
            episode_id="qa-test-000", domain="qa", split="test",
            snapshot_id="qa-test-000-s0",
            doc_ids=[row["doc_id"] for row in self.documents[:8]],
            support_qids=["s0", "s1"], optimization_qids=["o0"], eval_qids=["e0"],
            poison_source_qids=["p0", "p1"], budget_poison=2, trigger_tokens=3,
            retrieval_top_k=3,
        )
        self.snapshot = Snapshot(
            snapshot_id="qa-test-000-growth-50", episode_id="qa-test-000", domain="qa",
            split="test", kind="growth", growth=0.5,
            doc_ids=[row["doc_id"] for row in self.documents],
            parent_snapshot_id="qa-test-000-s0", note={},
        )
        torch.manual_seed(0)
        self.base_module = fresh_module(self.config, self.retriever, self.episode)
        base_context = self.workspace.context_for(self.episode)
        from src.triggers.mcat.evaluate import generate_trigger
        self.base_trigger = generate_trigger(self.base_module, self.config,
                                             base_context, self.retriever)
        self.context = self.workspace.context_for(self.episode, self.snapshot)

    def _adapt(self, method, **overrides):
        options = dict(
            workspace=self.workspace, episode=self.episode, context=self.context,
            config=self.config, adapt_config=self.adapt_config,
            base_module=self.base_module, base_trigger=self.base_trigger,
            output_dir=self.directory / "adapt", contract={"test": True},
            ledger=CostLedger(device="cpu"),
        )
        options.update(overrides)
        return adapt_trigger(method, **options)

    def test_reuse_returns_the_s0_trigger_and_spends_nothing(self):
        result = self._adapt("reuse")
        self.assertTrue(result.applicable)
        self.assertEqual(result.trigger["token_ids"], self.base_trigger["token_ids"])
        self.assertEqual(set(result.cost["counters"].values()), {0.0})

    def test_generate_runs_one_forward_and_no_optimizer_step(self):
        result = self._adapt("generate")
        self.assertTrue(result.applicable)
        self.assertEqual(result.cost["counters"]["optimizer_steps"], 0.0)
        self.assertEqual(result.cost["counters"]["encoder_backward"], 0.0)

    def test_scratch_pays_the_full_step_budget(self):
        result = self._adapt("scratch")
        self.assertEqual(result.cost["counters"]["optimizer_steps"], self.config.steps)

    def test_warm_start_pays_only_the_few_step_budget(self):
        result = self._adapt("warm-start")
        self.assertEqual(result.cost["counters"]["optimizer_steps"],
                         self.adapt_config.steps)
        self.assertLess(result.cost["counters"]["optimizer_steps"],
                        self.config.steps)

    def test_warm_start_with_no_steps_is_exactly_generate(self):
        # Starting warm means starting from the s0 parameters: with a zero
        # budget the two arms have to coincide, or the warm start is not warm.
        warm = self._adapt("warm-start", adapt_config=AdaptConfig(steps=0))
        generated = self._adapt("generate")
        self.assertEqual(warm.trigger["token_ids"], generated.trigger["token_ids"])

    def test_generate_is_not_applicable_without_a_transferable_module(self):
        result = self._adapt("generate", base_module=None)
        self.assertFalse(result.applicable)
        self.assertIn("no trained generator", result.reason)

    def test_generate_is_not_applicable_to_a_per_episode_baseline(self):
        result = self._adapt("generate", config=TrainConfig(mode="direct-logit", steps=2))
        self.assertFalse(result.applicable)
        self.assertIn("nothing to transfer", result.reason)

    def test_reuse_without_an_s0_trigger_says_so(self):
        result = self._adapt("reuse", base_trigger=None)
        self.assertFalse(result.applicable)

    def test_an_unknown_method_is_refused(self):
        with self.assertRaises(ValueError):
            self._adapt("teleport")

    def test_every_method_is_billed_to_one_ledger(self):
        ledger = CostLedger(device="cpu")
        results = adapt_all(
            ADAPT_METHODS, workspace=self.workspace, episode=self.episode,
            context=self.context, config=self.config, adapt_config=self.adapt_config,
            base_module=self.base_module, base_trigger=self.base_trigger,
            output_dir=self.directory / "all", contract={"test": True}, ledger=ledger,
        )
        self.assertEqual([r.method for r in results], list(ADAPT_METHODS))
        self.assertTrue(all(r.applicable for r in results))
        self.assertEqual(len(ledger.phases), len(ADAPT_METHODS))
        # No method may write to the index while merely producing a trigger.
        self.assertEqual(ledger.counters["index_writes"], 0.0)

    def test_no_method_is_handed_the_locked_queries(self):
        self.assertFalse(hasattr(self.context, "eval_texts"))
        locked = self.workspace.eval_texts(self.episode)
        self.assertNotIn(locked[0], self.context.optimization_texts)


class StatsTests(unittest.TestCase):
    def test_a_constant_difference_has_an_interval_around_it(self):
        left = [0.6, 0.7, 0.8, 0.9]
        right = [0.5, 0.6, 0.7, 0.8]
        result = paired_bootstrap(left, right, ["a", "b", "c", "d"],
                                  iterations=500, seed=0)
        self.assertAlmostEqual(result["mean_difference"], 0.1, places=6)
        self.assertLessEqual(result["ci_low"], 0.1)
        self.assertGreaterEqual(result["ci_high"], 0.1)

    def test_identical_arms_produce_an_interval_containing_zero(self):
        values = [0.1, 0.4, 0.9, 0.2]
        result = paired_bootstrap(values, values, ["a", "b", "c", "d"],
                                  iterations=500, seed=0)
        self.assertEqual(result["mean_difference"], 0.0)
        self.assertLessEqual(result["ci_low"], 0.0)
        self.assertGreaterEqual(result["ci_high"], 0.0)
        self.assertFalse(result["significant"])

    def test_resampling_follows_the_group_not_the_row(self):
        # Ten rows but one group: the interval must refuse to exist rather than
        # pretend ten independent observations.
        result = paired_bootstrap([1.0] * 10, [0.0] * 10, ["only"] * 10,
                                  iterations=100, seed=0)
        self.assertIsNone(result["ci_low"])
        self.assertIn("two independent groups", result["reason"])

    def test_mismatched_inputs_are_refused(self):
        with self.assertRaises(ValueError):
            paired_bootstrap([1.0], [1.0, 2.0], ["a", "b"])
        with self.assertRaises(ValueError):
            paired_bootstrap([1.0], [1.0], ["a"], alpha=1.5)

    def test_an_empty_comparison_is_reported_not_invented(self):
        result = paired_bootstrap([], [], [])
        self.assertIsNone(result["mean_difference"])
        self.assertEqual(result["groups"], 0)

    def test_macro_micro_and_worst_can_disagree(self):
        # One large weak episode and one small strong one: the micro average
        # follows the weak one, the macro does not, and worst names it.
        summary = macro_micro_worst([0.2, 0.9], [100, 1], ["big", "small"])
        self.assertAlmostEqual(summary["macro"], 0.55, places=6)
        self.assertLess(summary["micro"], 0.25)
        self.assertEqual(summary["worst_group"], "big")

    def test_an_empty_aggregate_is_null(self):
        summary = macro_micro_worst([], [], [])
        self.assertIsNone(summary["macro"])


class AggregateTests(unittest.TestCase):
    def _row(self, method, episode, snapshot, hit, **extra):
        return {"method": method, "episode_id": episode, "snapshot_id": snapshot,
                "kind": "growth", "applicable": True, "round_trip_valid": True,
                "false_activation": 0.0,
                "trigger_on": {"hit_at_3": hit, "queries": 10},
                "cost": {"wall_seconds": 1.0, "counters": {}},
                "deploy_cost": {"wall_seconds": 0.1, "counters": {"index_writes": 2}},
                **extra}

    def test_a_missing_baseline_is_named_not_silently_skipped(self):
        rows = [self._row("generate", "e0", "s1", 0.5)]
        summary = aggregate(rows, baseline="scratch", iterations=50)
        self.assertIn("reason", summary["paired_vs_baseline"])

    def test_not_applicable_rows_keep_their_reason(self):
        rows = [self._row("scratch", "e0", "s1", 0.5),
                {"method": "generate", "applicable": False,
                 "reason": "nothing to transfer"}]
        summary = aggregate(rows, iterations=50)
        self.assertEqual(summary["not_applicable"]["generate"], "nothing to transfer")

    def test_amortization_is_withheld_without_a_training_cost(self):
        rows = [self._row("scratch", "e0", "s1", 0.5),
                self._row("generate", "e0", "s1", 0.5)]
        summary = aggregate(rows, iterations=50)
        self.assertIn("reason", summary["amortization"])
        self.assertNotIn("break_even_episodes", summary["amortization"])

    def test_an_empty_input_reports_why(self):
        summary = aggregate([], iterations=50)
        self.assertIn("no applicable rows", summary["reason"])


class DriftStageTests(unittest.TestCase):
    """The CLI stages, end to end on the fixture retriever."""

    def setUp(self):
        self.directory = Path(tempfile.mkdtemp(prefix="mcat-drift-stage-"))
        self.addCleanup(shutil.rmtree, self.directory, True)

    def _drift_dir(self, policy="refresh", steps=1, split="test"):
        return self.directory / "drift" / f"{policy}-steps{steps}-{split}"

    def _argv(self, command, *extra):
        return [
            command, "--output-dir", str(self.directory), "--fixture",
            "--domain", "qa", "--corpus-limit", "300", "--per-split", "2",
            "--documents", "16", "--support", "3", "--optimization", "4",
            "--evaluation", "6", "--poison-count", "2", "--trigger-tokens", "3",
            "--top-k", "3", "--steps", "2", "--max-length", "32",
            "--adapt-steps", "1", "--bootstrap-iterations", "100", *extra,
        ]

    def _run(self, *commands, extra=()):
        for command in commands:
            self.assertEqual(main(self._argv(command, *extra)), 0, command)

    def test_the_drift_stages_produce_their_artifacts(self):
        # `report` still quotes the zero-shot evaluation, so that stage runs too.
        self._run("prepare-episodes", "index", "train", "evaluate", "prepare-drift",
                  "adapt", "evaluate-drift", "report")
        for name in ("trajectories.jsonl", "costs-train.json"):
            self.assertTrue((self.directory / name).exists(), f"{name} is missing")
        for name in (DRIFT_ROWS, DRIFT_SUMMARY, "costs.json", "drift_config.json"):
            self.assertTrue((self._drift_dir() / name).exists(), f"{name} is missing")
        manifest = json.loads((self.directory / "manifest.json").read_text("utf-8"))
        self.assertIn("trajectory_hash", manifest)
        self.assertIn("Drift and few-step", (self.directory / "REPORT.md").read_text("utf-8"))

    def test_adapting_requires_a_trajectory(self):
        self._run("prepare-episodes", "index", "train")
        with self.assertRaises(FileNotFoundError):
            main(self._argv("adapt"))

    def test_changing_the_drift_levels_is_refused(self):
        self._run("prepare-episodes", "index", "train", "prepare-drift")
        with self.assertRaises(ValueError):
            main(self._argv("prepare-drift", "--growth", "0.25"))

    def test_rerunning_adapt_without_resume_is_refused(self):
        self._run("prepare-episodes", "index", "train", "prepare-drift", "adapt")
        with self.assertRaises(FileExistsError):
            main(self._argv("adapt"))

    def test_resuming_adapt_recomputes_nothing(self):
        self._run("prepare-episodes", "index", "train", "prepare-drift", "adapt")
        before = load_rows(self._drift_dir() / DRIFT_ROWS)
        self._run("adapt", extra=("--resume",))
        after = load_rows(self._drift_dir() / DRIFT_ROWS)
        self.assertEqual(len(before), len(after))
        self.assertEqual([row["key"] for row in before], [row["key"] for row in after])

    def test_every_method_pays_the_same_write_budget(self):
        self._run("prepare-episodes", "index", "train", "prepare-drift", "adapt")
        rows = [row for row in load_rows(self._drift_dir() / DRIFT_ROWS)
                if row.get("applicable")]
        writes = {}
        for row in rows:
            counters = row["deploy_cost"]["counters"]
            writes.setdefault(row["method"], set()).add(counters.get("index_writes"))
        self.assertGreater(len(writes), 1)
        # One value, shared by every method: the budgeted resource is equal.
        self.assertEqual({value for values in writes.values() for value in values},
                         {2.0})

    def test_the_fixed_policy_freezes_the_poison_and_says_so(self):
        self._run("prepare-episodes", "index", "train", "prepare-drift")
        self._run("adapt", extra=("--write-policy", "fixed"))
        self.assertTrue((self._drift_dir("fixed") / "poison_s0.pt").exists())
        rows = [row for row in load_rows(self._drift_dir("fixed") / DRIFT_ROWS)
                if row.get("applicable")]
        self.assertTrue(rows)
        for row in rows:
            self.assertEqual(row["write_policy"], "fixed")
            self.assertEqual(row["poison_source"], "frozen")
            # Nothing is written to the index after s0 under this policy.
            self.assertEqual(row["deploy_cost"]["counters"].get("index_writes"), 0.0)

    def test_every_snapshot_of_every_episode_is_covered(self):
        self._run("prepare-episodes", "index", "train", "prepare-drift", "adapt")
        snapshots = [json.loads(line) for line in
                     (self.directory / "trajectories.jsonl").read_text("utf-8").splitlines()]
        rows = load_rows(self._drift_dir() / DRIFT_ROWS)
        test_snapshots = {s["snapshot_id"] for s in snapshots if s["split"] == "test"}
        self.assertEqual({row["snapshot_id"] for row in rows}, test_snapshots)
        self.assertEqual(len(rows), len(test_snapshots) * len(ADAPT_METHODS))


if __name__ == "__main__":
    unittest.main()
