"""hit@1/2/3/5 beside hit@K, the ood growth arm, and shared s0 searches.

The metric is checked on hand-built vectors whose rankings are known; the two
CLI tests run on the CPU fixture retriever and check the protocol only.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

import torch

from src.triggers.mcat.evaluate import retrieval_metrics
from src.triggers.mcat.probes import GrowthProbeConfig, hit_of, summarize_growth


class HitCurveTests(unittest.TestCase):
    def test_each_k_counts_queries_whose_poison_is_in_the_first_k(self):
        # One query along x.  Clean scores 5, 4, 3, 2, 1; the poison scores 3.5,
        # so it ranks 3rd: out of top-1 and top-2, inside top-3 and top-5.
        query = torch.tensor([[1.0, 0.0]])
        clean = torch.tensor([[5.0, 0], [4.0, 0], [3.0, 0], [2.0, 0], [1.0, 0]])
        poison = torch.tensor([[3.5, 0.0]])
        metrics = retrieval_metrics(query, clean, poison, top_k=5)
        self.assertEqual(metrics["hit_curve"], {"1": 0.0, "2": 0.0, "3": 1.0, "5": 1.0})
        self.assertEqual(metrics["hit_at_5"], 1.0)

    def test_hit_at_K_is_still_the_only_hit_at_key(self):
        # hit_of reads the first ``hit_at_`` key; the curve must not shadow it.
        query = torch.tensor([[1.0, 0.0]])
        clean = torch.tensor([[5.0, 0], [4.0, 0], [3.0, 0]])
        metrics = retrieval_metrics(query, clean, torch.tensor([[4.5, 0.0]]), top_k=3)
        self.assertEqual([key for key in metrics if key.startswith("hit_at_")], ["hit_at_3"])
        self.assertEqual(hit_of(metrics), 1.0)
        self.assertEqual(set(metrics["hit_curve"]), {"1", "2", "3"})


class GrowthSummaryCurveTests(unittest.TestCase):
    def _row(self, episode, kind, growth, curve, seed=None):
        return {"applicable": True, "episode_id": episode, "kind": kind, "growth": growth,
                "drift_seed": seed, "documents": 10, "poison_fraction": 0.1,
                "on_hit": curve["5"], "off_hit": 0.0,
                "trigger_on": {"hit_at_5": curve["5"], "hit_curve": curve}}

    def test_each_level_reports_the_paired_drop_per_k(self):
        full = {"1": 1.0, "2": 1.0, "3": 1.0, "5": 1.0}
        moved = {"1": 0.5, "2": 0.8, "3": 1.0, "5": 1.0}
        rows = []
        for episode in ("a", "b", "c"):
            rows.append(self._row(episode, "base", 0.0, full))
            rows.append(self._row(episode, "ood", 1.0, moved, seed=0))
        summary = summarize_growth(rows, iterations=200, selection="ood")
        curve = summary["levels"]["1"]["hit_curve"]
        self.assertAlmostEqual(curve["1"]["drop_vs_base"]["mean_difference"], 0.5)
        self.assertAlmostEqual(curve["5"]["drop_vs_base"]["mean_difference"], 0.0)
        self.assertEqual(summary["base"]["hit_curve"]["1"], 1.0)

    def test_ood_draws_several_seeds_unlike_the_ranked_selections(self):
        GrowthProbeConfig(selection="ood", seeds=(0, 1, 2))
        with self.assertRaisesRegex(ValueError, "deterministic"):
            GrowthProbeConfig(selection="support", seeds=(0, 1))


class SharedSearchCliTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp(prefix="mcat-curve-"))
        self.addCleanup(shutil.rmtree, self.directory, True)

    def _main(self, command, *extra):
        from src.triggers.mcat.cli import main
        argv = [
            command, "--output-dir", str(self.directory), "--fixture",
            "--domain", "qa", "--corpus-limit", "300", "--per-split", "3",
            "--documents", "16", "--support", "3", "--optimization", "4",
            "--evaluation", "6", "--poison-count", "2", "--trigger-tokens", "4",
            "--top-k", "3", "--steps", "2", "--max-length", "32",
            "--mode", "direct-logit", *extra,
        ]
        self.assertEqual(main(argv), 0, command)

    def test_ood_growth_and_contamination_share_one_s0_search(self):
        shared = self.directory / "base-search"
        self._main("prepare-episodes")
        self._main("index")
        self._main("probe-growth", "--split", "test", "--growth-selection", "ood",
                   "--growth", "0.5", "1.0", "--drift-seed", "0", "1",
                   "--bootstrap-iterations", "200", "--base-search-dir", str(shared))
        searched = sorted(path.parent.name for path in shared.glob("*/checkpoint.pt"))
        self.assertTrue(searched)
        stamps = {path: path.stat().st_mtime_ns for path in shared.glob("*/checkpoint.pt")}

        self._main("probe-contamination", "--split", "test", "--scenario", "self",
                   "--contamination-level", "1", "2", "--drift-seed", "0",
                   "--bootstrap-iterations", "200", "--base-search-dir", str(shared))
        # The second probe resumed the finished searches instead of redoing them.
        self.assertEqual({path: path.stat().st_mtime_ns
                          for path in shared.glob("*/checkpoint.pt")}, stamps)
        probes = self.directory / "probes"
        self.assertFalse(list(probes.rglob("base-search")))

        rows = [json.loads(line) for line in
                (probes / "growth_ood-suffix-test/probe_growth.jsonl")
                .read_text(encoding="utf-8").splitlines()]
        ood = [row for row in rows if row.get("kind") == "ood"]
        episodes = {row["episode_id"] for row in rows if row.get("kind") == "base"}
        self.assertEqual(len(ood), len(episodes) * 2 * 2)
        self.assertTrue(all("hit_curve" in row["trigger_on"] for row in ood))
        base_docs = {row["episode_id"]: row["documents"] for row in rows
                     if row.get("kind") == "base"}
        for row in ood:
            self.assertEqual(row["documents"],
                             base_docs[row["episode_id"]] + round(row["growth"] * 16))


if __name__ == "__main__":
    unittest.main()
