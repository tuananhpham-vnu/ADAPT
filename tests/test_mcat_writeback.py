"""probe-writeback: the agent logs its triggered interactions under a write policy.

The policy logic is checked on hand-built 2-d vectors whose rankings are known;
the CLI test runs on the CPU fixture retriever, where the numbers are
meaningless and only the protocol is checked.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

import torch

from src.triggers.mcat.cli import main
from src.triggers.mcat.writeback import (
    WRITEBACK_ROWS, WritebackConfig, simulate_stream, summarize_writeback,
)

# Triggered queries point along y; clean memory along x, so it scores 0 against
# them.  The seed poison scores 0.5 and a logged triggered query scores 1.
CLEAN = torch.tensor([[1.0, 0.0]] * 3)
POISON = torch.tensor([[0.0, 0.5]])
STREAM = torch.tensor([[0.0, 1.0]] * 4)


def run(policy):
    return simulate_stream(STREAM, CLEAN, POISON, policy=policy, top_k=1,
                           checkpoints=(1, 4))


class SimulateStreamTests(unittest.TestCase):
    def test_none_writes_nothing_and_every_interaction_fires(self):
        state = run("none")[4]
        self.assertEqual(state.fired, [True] * 4)
        self.assertEqual((state.benign.shape[0], state.malicious.shape[0]), (0, 0))

    def test_log_outcome_turns_fired_interactions_into_malicious_records(self):
        state = run("log_outcome")[4]
        self.assertEqual(state.fired, [True] * 4)
        self.assertEqual((state.benign.shape[0], state.malicious.shape[0]), (0, 4))

    def test_verified_drops_fired_interactions(self):
        state = run("verified")[4]
        self.assertEqual(state.fired, [True] * 4)
        self.assertEqual((state.benign.shape[0], state.malicious.shape[0]), (0, 0))

    def test_corrected_logs_benign_and_the_next_interaction_is_blocked(self):
        states = run("corrected")
        self.assertEqual(states[1].fired, [True])
        # The first benign log outscores the seed poison for every later query.
        self.assertEqual(states[4].fired, [True, False, False, False])
        self.assertEqual(states[4].benign.shape[0], 4)

    def test_snapshots_are_taken_after_the_checkpointed_write(self):
        self.assertEqual(run("log_outcome")[1].malicious.shape[0], 1)

    def test_a_checkpoint_beyond_the_stream_is_refused(self):
        with self.assertRaisesRegex(ValueError, "exceeds"):
            simulate_stream(STREAM, CLEAN, POISON, policy="none", top_k=1,
                            checkpoints=(5,))

    def test_an_unknown_policy_is_refused(self):
        with self.assertRaisesRegex(ValueError, "policy"):
            run("forget")


class WritebackConfigTests(unittest.TestCase):
    def test_unknown_and_duplicate_policies_are_refused(self):
        with self.assertRaisesRegex(ValueError, "drawn from"):
            WritebackConfig(policies=("none", "forget"))
        with self.assertRaisesRegex(ValueError, "distinct"):
            WritebackConfig(policies=("none", "none"))

    def test_levels_and_seed_poison_are_validated(self):
        with self.assertRaisesRegex(ValueError, "increasing"):
            WritebackConfig(levels=(10, 5))
        with self.assertRaisesRegex(ValueError, "seed_poison"):
            WritebackConfig(seed_poison=0)


class SummaryTests(unittest.TestCase):
    def _row(self, episode, kind, on, *, policy="none", level=0, seed=None, cleanup=0.0):
        return {"episode_id": episode, "kind": kind, "policy": policy,
                "interactions": level, "drift_seed": seed, "on_hit": on, "off_hit": 0.0,
                "cleanup_hit": cleanup, "malicious_written": 0, "benign_written": 0,
                "stream_fire_rate": 1.0,
                "trigger_on": {"poison_occupancy_at_5": 0.5}}

    def test_a_large_paired_drop_reads_as_dilution(self):
        rows = []
        for index, episode in enumerate(["a", "b", "c", "d"]):
            rows.append(self._row(episode, "base", 1.0))
            for seed in (0, 1):
                rows.append(self._row(episode, "stream", 0.5 + 0.01 * index,
                                      policy="corrected", level=25, seed=seed))
        verdict = summarize_writeback(rows, iterations=200)["policies"]["corrected"]["verdict"]
        self.assertEqual(verdict["direction"], "dilutes")
        self.assertFalse(verdict["persists_after_cleanup"])


class WritebackCliTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp(prefix="mcat-writeback-"))
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

    def test_every_policy_level_and_seed_gets_one_row_plus_one_base(self):
        for command in ("prepare-episodes", "index"):
            self.assertEqual(main(self._argv(command)), 0, command)
        self.assertEqual(main(self._argv(
            "probe-writeback", "--split", "test", "--writeback-level", "1", "3",
            "--drift-seed", "0", "1", "--seed-poison", "1",
            "--bootstrap-iterations", "200")), 0)
        directory = self.directory / "probes/writeback_p1-suffix-test"
        rows = [json.loads(line) for line in
                (directory / WRITEBACK_ROWS).read_text(encoding="utf-8").splitlines()]
        episodes = {row["episode_id"] for row in rows}
        self.assertEqual(len(rows), len(episodes) * (1 + 4 * 2 * 2))
        self.assertTrue(all(row["seed_poison_records"] == 1 for row in rows))
        self.assertEqual(len({row["key"] for row in rows}), len(rows))
        for row in rows:
            written = row["benign_written"] + row["malicious_written"]
            if row["policy"] in ("none", "verified"):
                self.assertEqual(row["malicious_written"], 0)
            if row["policy"] in ("log_outcome", "corrected"):
                self.assertEqual(written, row["interactions"])
        summary = json.loads((directory / "probe_writeback.json").read_text(encoding="utf-8"))
        self.assertEqual(set(summary["policies"]),
                         {"none", "log_outcome", "verified", "corrected"})


if __name__ == "__main__":
    unittest.main()
