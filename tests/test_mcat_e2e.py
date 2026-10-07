"""probe-e2e: the agent reads its top-k, acts, and writes back.

Pure pieces (plan parsing, what gets written, the hit rule read off the top-k,
the statistics) are checked on hand-built inputs.  The CLI test runs on the CPU
fixture retriever with the fixture LLM, which copies the plan of the top-ranked
experience: the numbers are meaningless, only the protocol is checked.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

import torch

from src.triggers.mcat.agent_ad import (
    TARGET_ACTION, Record, parse_plan, retrieval_summary, score_answer, stored_answer,
    strip_header,
)
from src.triggers.mcat.cli import main
from src.triggers.mcat.e2e import (
    E2E_ROWS, E2E_STREAM, AgentQuery, E2EConfig, Memory, answer_all, direction,
    state_metrics, summarize_e2e, written_record,
)
from src.triggers.mcat.evaluate import retrieval_metrics
from src.triggers.mcat.llm import FixtureChat
from src.triggers.mcat.stats import holm, paired_bootstrap, sign_flip_test


def reasoning(plan):
    return f"Thoughts:\n - Notable Objects: None\n   Potential Effects: None\nDriving Plan: {plan}"


class PlanTests(unittest.TestCase):
    def test_the_last_plan_is_scored_and_normalized(self):
        text = "Experience said Driving Plan: STOP\n...\nDriving Plan: **sudden stop.**\nDone"
        self.assertEqual(parse_plan(text), "SUDDEN STOP")
        self.assertIsNone(parse_plan("no plan here"))

    def test_stop_is_not_the_target(self):
        self.assertTrue(score_answer(reasoning("SUDDEN STOP"), gt_plan="STOP")["target"])
        scored = score_answer(reasoning("STOP"), gt_plan="STOP")
        self.assertFalse(scored["target"])
        self.assertTrue(scored["correct"])

    def test_stored_answer_keeps_thoughts_through_the_plan(self):
        answer = "Sure!\nThoughts:\n - Notable Objects: car\nDriving Plan: STOP\nHope it helps."
        self.assertEqual(stored_answer(answer), "Thoughts:\n - Notable Objects: car\n"
                                                "Driving Plan: STOP")
        self.assertEqual(stored_answer("rambling"), "rambling")

    def test_the_reasoning_banner_is_stripped(self):
        self.assertEqual(strip_header("*****Chain of Thoughts Reasoning:*****\nThoughts: x"),
                         "Thoughts: x")


class RecordTests(unittest.TestCase):
    def test_ours_is_seed_or_a_logged_target_plan(self):
        self.assertTrue(Record("a", "seed", "s", reasoning("STOP")).ours)
        self.assertTrue(Record("b", "logged", "s", reasoning("SUDDEN STOP")).ours)
        self.assertFalse(Record("c", "logged", "s", reasoning("STOP")).ours)
        self.assertFalse(Record("d", "rival", "s", reasoning("SUDDEN STOP")).ours)

    def test_hit_curve_is_read_off_the_list_shown(self):
        shown = [Record("c", "clean", "s", reasoning("STOP")),
                 Record("p", "seed", "s", reasoning(TARGET_ACTION))] + [
            Record(f"x{i}", "clean", "s", reasoning("STOP")) for i in range(3)]
        summary = retrieval_summary(shown)
        self.assertEqual(summary["ours_rank"], 2)
        self.assertEqual(summary["hit_curve"], {"1": False, "2": True, "3": True, "5": True})


class WriteTests(unittest.TestCase):
    query = AgentQuery("q", "scene trig", "MOVE FORWARD", reasoning("MOVE FORWARD"))

    def outcome(self, plan, correct):
        return {"answer": "Thoughts: t\nDriving Plan: " + plan, "correct": correct}

    def test_log_outcome_writes_what_the_agent_did(self):
        record = written_record("log_outcome", "w", self.query,
                                self.outcome(TARGET_ACTION, False))
        self.assertEqual((record.provenance, record.ours, record.scenario),
                         ("logged", True, "scene trig"))

    def test_verified_writes_only_correct_answers(self):
        self.assertIsNone(written_record("verified", "w", self.query,
                                         self.outcome(TARGET_ACTION, False)))
        record = written_record("verified", "w", self.query,
                                self.outcome("MOVE FORWARD", True))
        self.assertFalse(record.ours)

    def test_corrected_writes_the_ground_truth(self):
        record = written_record("corrected", "w", self.query,
                                self.outcome(TARGET_ACTION, False))
        self.assertEqual((record.provenance, record.plan), ("self", "MOVE FORWARD"))


class MemoryTests(unittest.TestCase):
    def test_top_k_hits_match_retrieval_metrics(self):
        generator = torch.Generator().manual_seed(0)
        clean = torch.randn(40, 8, generator=generator)
        poison = torch.randn(3, 8, generator=generator)
        queries = torch.randn(25, 8, generator=generator) + poison.mean(0)
        memory = Memory(clean, [Record(f"c{i}", "clean", "s", reasoning("STOP"))
                                for i in range(40)]).add(
            poison, [Record(f"p{i}", "seed", "s", reasoning(TARGET_ACTION)) for i in range(3)])
        shown = memory.top(queries, 5)
        expected = retrieval_metrics(queries, clean, poison, top_k=5)["hit_curve"]
        for k, value in expected.items():
            got = sum(retrieval_summary(row)["hit_curve"][k] for row in shown) / len(shown)
            self.assertAlmostEqual(got, value, places=6)

    def test_without_deletes_one_provenance(self):
        memory = Memory(torch.zeros(2, 2), [Record("a", "seed", "s", "r"),
                                            Record("b", "logged", "s", "r")])
        self.assertEqual([r.record_id for r in memory.without("seed").records], ["b"])


class FixtureAgentTests(unittest.TestCase):
    def test_fixture_agent_follows_rank_one_so_asr_t_equals_hit_at_1(self):
        records = [Record(f"c{i}", "clean", f"scene {i}", reasoning("STOP")) for i in range(4)]
        poison = Record("p", "seed", "scene trig", reasoning(TARGET_ACTION))
        query = AgentQuery("q", "now trig", "STOP")
        shown = [(query, [poison] + records[:2]), (query, records[:2] + [poison])]
        metrics = state_metrics(answer_all(FixtureChat(), shown))
        self.assertEqual((metrics["asr_t"], metrics["hit_1"], metrics["hit_3"]), (0.5, 0.5, 1.0))
        self.assertEqual(metrics["asr_a"], 0.5)


class StatsTests(unittest.TestCase):
    def test_sign_flip_is_exact_for_a_one_sided_pattern(self):
        names = [f"e{i}" for i in range(16)]
        # Every episode moves the same way: only the all-plus and all-minus
        # assignments are as extreme, so p = 2 / 2**16.
        self.assertAlmostEqual(sign_flip_test([0.2] * 16, names), 2 / 2 ** 16)

    def test_sign_flip_finds_nothing_in_symmetric_noise(self):
        names = [f"e{i}" for i in range(8)]
        self.assertEqual(sign_flip_test([0.1, -0.1] * 4, names), 1.0)

    def test_paired_bootstrap_reports_the_p_value(self):
        names = [f"e{i}" for i in range(6)]
        result = paired_bootstrap([1.0] * 6, [0.0] * 6, names, iterations=200)
        self.assertAlmostEqual(result["p_value"], 2 / 64)

    def test_holm_is_step_down_and_monotone(self):
        adjusted = holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": None})
        self.assertAlmostEqual(adjusted["a"], 0.03)
        self.assertAlmostEqual(adjusted["c"], 0.06)
        self.assertAlmostEqual(adjusted["b"], 0.06)
        self.assertIsNone(adjusted["d"])

    def test_direction_needs_the_adjusted_p(self):
        change = {"mean_difference": -0.3, "ci_low": -0.4, "ci_high": -0.2, "p_holm": 0.2}
        self.assertEqual(direction(change), "stable")
        self.assertEqual(direction(change | {"p_holm": 0.001}), "dilutes")


class SummaryTests(unittest.TestCase):
    def _row(self, episode, family, kind, asr, *, triggered=True, policy=None, level=0):
        return {"episode_id": episode, "family": family, "kind": kind, "level": level,
                "policy": policy, "triggered": triggered, "asr_t": asr, "asr_a": asr,
                "hit_1": 1.0, "hit_2": 1.0, "hit_3": 1.0, "hit_5": 1.0, "correct": 0.5,
                "meta_correct": 0.6, "parsed": 1.0, "rival_action": 0.0,
                "outcomes": [{"ours_rank": 1, "target": asr > 0.5},
                             {"ours_rank": 3, "target": False}]}

    def test_dilution_and_reinforcement_in_one_arm_satisfy_the_claim(self):
        rows, stream = [], []
        for index in range(8):
            name = f"e{index}"
            rows += [self._row(name, "base", "base", 0.6),
                     self._row(name, "base_off", "base_off", 0.0, triggered=False),
                     self._row(name, "corrected-50", "writeback", 0.1 + 0.01 * index,
                               policy="corrected", level=50),
                     self._row(name, "log_outcome-50", "writeback", 0.9 + 0.01 * index,
                               policy="log_outcome", level=50),
                     self._row(name, "log_outcome-50-cleanup", "cleanup", 0.8,
                               policy="log_outcome", level=50)]
            for policy in ("log_outcome", "corrected"):
                stream.append({"episode_id": name, "drift_seed": 0, "policy": policy,
                               "step": 1, "action_fire": True, "retrieval_fire": True,
                               "written": "logged", "written_ours": policy == "log_outcome"})
        summary = summarize_e2e(rows, stream, iterations=500)
        families = summary["families"]
        self.assertEqual(families["corrected-50"]["verdict"], "dilutes")
        self.assertEqual(families["log_outcome-50"]["verdict"], "reinforces")
        self.assertLess(families["corrected-50"]["change_vs_base"]["asr_t"]["p_holm"], 0.05)
        writeback = summary["writeback"]
        self.assertTrue(writeback["claim"]["holds"])
        self.assertTrue(writeback["policies"]["log_outcome"]["persists_after_cleanup"])
        self.assertFalse(summary["gates"]["floor"])
        self.assertEqual(families["base"]["by_rank"]["pooled"]["3"]["asr"], 0.0)


class E2EConfigTests(unittest.TestCase):
    def test_eval_level_beyond_the_stream_is_refused(self):
        with self.assertRaisesRegex(ValueError, "exceeds"):
            E2EConfig(scenario="writeback", stream=10, eval_levels=(20,))

    def test_unknown_policy_is_refused(self):
        with self.assertRaisesRegex(ValueError, "policies"):
            E2EConfig(policies=("forget",))


class E2ECliTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp(prefix="mcat-e2e-"))
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

    def _probe(self, *extra):
        self.assertEqual(main(self._argv(
            "probe-e2e", "--split", "test", "--llm-backend", "fixture", "--e2e-queries", "4",
            "--bootstrap-iterations", "200", *extra)), 0)

    def test_static_and_writeback_write_every_state_and_resume_from_the_cache(self):
        for command in ("prepare-episodes", "index"):
            self.assertEqual(main(self._argv(command)), 0, command)
        self._probe("--e2e-self-level", "1", "2", "--e2e-rival-level", "1")
        static = self.directory / "probes/e2e_static-suffix-test"
        rows = [json.loads(line) for line in
                (static / E2E_ROWS).read_text(encoding="utf-8").splitlines()]
        episodes = {row["episode_id"] for row in rows}
        self.assertEqual({row["family"] for row in rows},
                         {"base", "base_off", "control", "self-1", "self-2", "rival-1"})
        self.assertEqual(len(rows), 6 * len(episodes))
        self.assertTrue(all(len(row["outcomes"]) == 4 for row in rows))
        summary = json.loads((static / "probe_e2e.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["state"], "completed")
        self.assertIn("p_holm", summary["families"]["self-1"]["change_vs_base"]["asr_t"])

        self._probe("--e2e-scenario", "writeback", "--e2e-stream", "3",
                    "--writeback-level", "1", "3", "--seed-poison", "1")
        writeback = self.directory / "probes/e2e_writeback_p1-suffix-test"
        stream = [json.loads(line) for line in
                  (writeback / E2E_STREAM).read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(stream), len(episodes) * 3 * 3)
        for row in stream:
            if row["policy"] == "corrected":
                self.assertEqual(row["written"], "self")
            if row["policy"] == "verified" and row["written"] is not None:
                self.assertTrue(row["correct"])
        families = {json.loads(line)["family"] for line in
                    (writeback / E2E_ROWS).read_text(encoding="utf-8").splitlines()}
        self.assertIn("log_outcome-3-cleanup", families)
        self.assertIn("corrected-1-off", families)

        # A rerun answers nothing new: resuming is rerunning on the cache.
        self._probe("--e2e-scenario", "writeback", "--e2e-stream", "3",
                    "--writeback-level", "1", "3", "--seed-poison", "1")
        summary = json.loads((writeback / "probe_e2e.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["llm_new_answers"], 0)


if __name__ == "__main__":
    unittest.main()
