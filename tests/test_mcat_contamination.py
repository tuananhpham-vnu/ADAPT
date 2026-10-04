"""probe-contamination: trigger-carrying records join a frozen attacker's memory.

Runs on the CPU fixture retriever.  The numbers are meaningless; what is checked
is the protocol -- what gets added, what stays frozen, and that every row can be
told apart on resume.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

from src.triggers.mcat.cli import main
from src.triggers.mcat.contamination import CONTAMINATION_ROWS, ContaminationConfig


class ContaminationConfigTests(unittest.TestCase):
    def test_an_unknown_scenario_is_refused(self):
        with self.assertRaisesRegex(ValueError, "scenario"):
            ContaminationConfig(scenario="benign")

    def test_levels_must_be_increasing_positive_counts(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            ContaminationConfig(levels=(0, 1))
        with self.assertRaisesRegex(ValueError, "increasing"):
            ContaminationConfig(levels=(5, 2))

    def test_seeds_must_be_distinct(self):
        with self.assertRaisesRegex(ValueError, "distinct"):
            ContaminationConfig(seeds=(0, 0))


class ContaminationCliTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp(prefix="mcat-contam-"))
        self.addCleanup(shutil.rmtree, self.directory, True)

    def _argv(self, command, *extra):
        return [
            command, "--output-dir", str(self.directory), "--fixture",
            "--domain", "qa", "--corpus-limit", "300", "--per-split", "3",
            "--documents", "16", "--support", "3", "--optimization", "4",
            "--evaluation", "6", "--poison-count", "2", "--trigger-tokens", "4",
            "--top-k", "3", "--steps", "2", "--max-length", "32",
            "--mode", "direct-logit", *extra,
        ]

    def _run(self, command, *extra):
        self.assertEqual(main(self._argv(command, *extra)), 0, command)

    def _probe(self, scenario, levels):
        self._run("prepare-episodes")
        self._run("index")
        self._run("probe-contamination", "--split", "test", "--scenario", scenario,
                  "--contamination-level", *map(str, levels), "--drift-seed", "0", "1",
                  "--bootstrap-iterations", "200")
        directory = self.directory / f"probes/contam_{scenario}-suffix-test"
        rows = [json.loads(line) for line in
                (directory / CONTAMINATION_ROWS).read_text(encoding="utf-8").splitlines()]
        summary = json.loads((directory / "probe_contamination.json").read_text("utf-8"))
        return rows, summary

    def _check_common(self, rows, levels):
        episodes = {row["episode_id"] for row in rows}
        # 1 base + levels x 2 seeds per episode.
        self.assertEqual(len(rows), (1 + 2 * len(levels)) * len(episodes))
        self.assertEqual(len({row["key"] for row in rows}), len(rows),
                         "two measurements share a key, so a resume would confuse them")
        # The attacker's own poison is frozen at s0: its count never moves.
        self.assertEqual(len({row["poison_records"] for row in rows}), 1)

    def test_self_adds_exactly_the_requested_records(self):
        rows, summary = self._probe("self", (1, 3))
        self._check_common(rows, (1, 3))
        base = {row["episode_id"]: row["documents"] for row in rows if row["kind"] == "base"}
        for row in rows:
            if row["kind"] == "self":
                self.assertEqual(row["documents"], base[row["episode_id"]] + row["growth"])
        self.assertEqual(summary["scenario"], "self")
        if summary["verdict"] == "stable":
            # The conclusion is scoped to this scenario, not to "growth".
            self.assertIn("triggered interactions", summary["reason"])

    def test_rival_adds_other_attackers_poison_never_its_own(self):
        rows, summary = self._probe("rival", (1, 2))
        self._check_common(rows, (1, 2))
        base = {row["episode_id"]: row["documents"] for row in rows if row["kind"] == "base"}
        for row in rows:
            if row["kind"] != "rival":
                continue
            self.assertNotIn(row["episode_id"], row["added"])
            self.assertEqual(len(row["added"]), row["growth"])
            # poison-count 2 per rival attacker.
            self.assertEqual(row["documents"], base[row["episode_id"]] + 2 * row["growth"])
        self.assertEqual(summary["scenario"], "rival")

    def test_levels_are_nested_within_a_seed(self):
        rows, _ = self._probe("rival", (1, 2))
        by = {(row["episode_id"], row["drift_seed"], row["growth"]): row["added"]
              for row in rows if row["kind"] == "rival"}
        for (episode, seed, level), added in by.items():
            if level == 2.0:
                self.assertEqual(by[(episode, seed, 1.0)], added[:1])

    def test_rival_refuses_more_attackers_than_exist(self):
        self._run("prepare-episodes")
        self._run("index")
        with self.assertRaisesRegex(ValueError, "other attackers"):
            main(self._argv("probe-contamination", "--split", "test", "--scenario",
                            "rival", "--contamination-level", "5", "--drift-seed", "0"))


if __name__ == "__main__":
    unittest.main()
