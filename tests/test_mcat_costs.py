"""Cost accounting: what gets counted, where, and what must not be quoted.

Two failure modes these cover.  The first is instrumentation that changes the
run -- a counter that raises, or that behaves differently when no ledger is
active.  The second is the reporting trap: a break-even point computed across a
quality gap, or a negative one read as "pays off immediately".
"""
from __future__ import annotations

import unittest

import torch

from src.triggers.mcat.costs import (
    COUNTERS, CostLedger, amortization, break_even, current, lifecycle_cost,
    measure_online, percentile, record, summarize_latency,
)
from src.triggers.mcat.encoding import encode_plain, encode_with_trigger_embeddings
from src.triggers.mcat.retrievers import build_fixture_retriever, fixture_text

MAX_LENGTH = 32


class RecordingTests(unittest.TestCase):
    def test_recording_without_a_ledger_is_a_no_op(self):
        self.assertIsNone(current())
        record("encoder_forward", 10)  # must not raise and must not store

    def test_an_unknown_counter_is_refused(self):
        with self.assertRaises(KeyError):
            record("gpu_hours_we_wish_we_had", 1)

    def test_an_inner_ledger_shares_its_charges_with_the_outer_one(self):
        # A stage that opens its own ledger must not blank out the accounting
        # that wraps it; that failure reads as "this stage costs nothing".
        outer, inner = CostLedger(), CostLedger()
        with outer.active():
            self.assertIs(current(), outer)
            with inner.active():
                record("optimizer_steps", 1)
                self.assertIs(current(), inner)
            self.assertIs(current(), outer)
            record("optimizer_steps", 1)
        self.assertIsNone(current())
        self.assertEqual(inner.counters["optimizer_steps"], 1)
        self.assertEqual(outer.counters["optimizer_steps"], 2)

    def test_a_ledger_opened_inside_itself_is_charged_once(self):
        ledger = CostLedger()
        with ledger.active():
            with ledger.phase("inner"):
                record("optimizer_steps", 1)
        self.assertEqual(ledger.counters["optimizer_steps"], 1)

    def test_every_counter_starts_at_zero(self):
        ledger = CostLedger()
        self.assertEqual(sorted(ledger.counters), sorted(COUNTERS))
        self.assertEqual(set(ledger.counters.values()), {0.0})


class EncodingIsCountedTests(unittest.TestCase):
    """The retriever is the expensive part, so every call through it is charged."""

    def setUp(self):
        self.retriever = build_fixture_retriever(max_length=MAX_LENGTH)
        self.texts = [fixture_text(index) for index in range(4)]

    def _encode_plain(self):
        return encode_plain(self.retriever.model, self.retriever.tokenizer, self.texts,
                            device=self.retriever.device, max_length=MAX_LENGTH)

    def _encode_triggered(self, trigger_tokens=3):
        embeds = self.retriever.model.get_input_embeddings()(
            torch.tensor([6] * trigger_tokens))
        return encode_with_trigger_embeddings(
            self.retriever.model, self.retriever.tokenizer, self.texts, embeds,
            device=self.retriever.device, max_length=MAX_LENGTH,
        )

    def test_plain_encoding_charges_one_row_per_text(self):
        ledger = CostLedger()
        with ledger.active():
            self._encode_plain()
        self.assertEqual(ledger.counters["encoder_forward"], len(self.texts))

    def test_triggered_encoding_charges_one_row_per_text(self):
        ledger = CostLedger()
        with ledger.active():
            self._encode_triggered()
        self.assertEqual(ledger.counters["encoder_forward"], len(self.texts))

    def test_a_refused_configuration_is_not_counted_as_work(self):
        ledger = CostLedger()
        with ledger.active():
            with self.assertRaises(ValueError):
                # No room for the trigger plus [CLS]/[SEP]: nothing is encoded.
                self._encode_triggered(trigger_tokens=MAX_LENGTH)
        self.assertEqual(ledger.counters["encoder_forward"], 0)

    def test_encoding_outside_a_ledger_still_works(self):
        result = self._encode_plain()
        self.assertEqual(result.shape[0], len(self.texts))


class PhaseTests(unittest.TestCase):
    def test_a_phase_records_only_its_own_delta(self):
        ledger = CostLedger()
        with ledger.active():
            record("optimizer_steps", 5)
            with ledger.phase("train"):
                record("optimizer_steps", 2)
        phase = ledger.phase_named("train")
        self.assertEqual(phase.counters["optimizer_steps"], 2)
        self.assertEqual(ledger.counters["optimizer_steps"], 7)

    def test_an_outer_phase_includes_its_nested_phases(self):
        ledger = CostLedger()
        with ledger.phase("lifecycle"):
            with ledger.phase("train"):
                record("encoder_backward", 3)
            record("encoder_forward", 1)
        self.assertEqual(ledger.phase_named("train").counters["encoder_backward"], 3)
        outer = ledger.phase_named("lifecycle")
        self.assertEqual(outer.counters["encoder_backward"], 3)
        self.assertEqual(outer.counters["encoder_forward"], 1)

    def test_the_vram_scope_is_stated_rather_than_assumed(self):
        ledger = CostLedger()
        with ledger.phase("outer"):
            with ledger.phase("inner"):
                pass
        self.assertEqual(ledger.phase_named("outer").vram_scope, "phase")
        # A nested phase cannot reset the peak without destroying its parent's
        # measurement, so it says its number covers the whole run.
        self.assertEqual(ledger.phase_named("inner").vram_scope, "run")

    def test_a_phase_survives_an_exception_and_still_accounts_for_it(self):
        ledger = CostLedger()
        with self.assertRaises(RuntimeError):
            with ledger.phase("adapt"):
                record("optimizer_steps", 1)
                raise RuntimeError("boom")
        self.assertEqual(ledger.phase_named("adapt").counters["optimizer_steps"], 1)
        self.assertIsNone(current())

    def test_a_phase_is_serializable(self):
        ledger = CostLedger(device="cpu")
        with ledger.phase("train"):
            record("encoder_forward", 2)
        payload = ledger.to_json()
        self.assertEqual(payload["phases"][0]["counters"]["encoder_forward"], 2)
        self.assertEqual(payload["phases"][0]["peak_vram_bytes"], 0)
        self.assertEqual(payload["online_latency"]["samples"], 0)


class LatencyTests(unittest.TestCase):
    def test_warmup_samples_never_reach_the_ledger(self):
        ledger, calls = CostLedger(), []
        measure_online(ledger, lambda: calls.append(1), repeats=4, warmup=3)
        self.assertEqual(len(calls), 7)
        self.assertEqual(len(ledger.online_latency_seconds), 4)

    def test_the_last_result_is_returned(self):
        ledger, counter = CostLedger(), iter(range(100))
        self.assertEqual(measure_online(ledger, lambda: next(counter),
                                        repeats=1, warmup=0), 0)

    def test_an_empty_budget_is_refused(self):
        with self.assertRaises(ValueError):
            measure_online(CostLedger(), lambda: None, repeats=0)

    def test_percentiles_are_reported_not_the_mean(self):
        values = [1.0, 2.0, 3.0, 4.0, 100.0]
        self.assertEqual(percentile(values, 0.50), 3.0)
        self.assertEqual(percentile(values, 0.95), 100.0)
        summary = summarize_latency(values)
        self.assertEqual(summary["samples"], 5)
        self.assertNotIn("mean", summary)

    def test_an_empty_sample_is_null_not_zero(self):
        # Zero latency and no measurement are different claims.
        self.assertIsNone(percentile([], 0.5))
        self.assertIsNone(summarize_latency([])["p50"])

    def test_an_invalid_fraction_is_refused(self):
        with self.assertRaises(ValueError):
            percentile([1.0], 1.5)


class AmortizationTests(unittest.TestCase):
    def test_break_even_is_the_ratio_when_generating_is_cheaper(self):
        self.assertEqual(break_even(100.0, 10.0, 2.0), 12.5)

    def test_there_is_no_break_even_when_generating_is_not_cheaper(self):
        # The guard that stops a negative number being read as "immediately".
        self.assertIsNone(break_even(100.0, 2.0, 10.0))
        self.assertIsNone(break_even(100.0, 5.0, 5.0))

    def test_lifecycle_adds_training_to_the_online_cost(self):
        self.assertEqual(lifecycle_cost(100.0, 5.0, 2.0, 10), 125.0)
        self.assertEqual(lifecycle_cost(100.0, 5.0, 2.0, 0), 105.0)
        with self.assertRaises(ValueError):
            lifecycle_cost(1.0, 0.0, 1.0, -1)

    def test_a_point_bought_at_a_quality_gap_is_not_comparable(self):
        result = amortization(train_cost=100.0, online_search=10.0, online_gen=2.0,
                              quality_gap=-0.20, quality_tolerance=0.02)
        self.assertEqual(result["break_even_episodes"], 12.5)
        self.assertFalse(result["comparable"])
        self.assertIn("equal quality", result["reason"])

    def test_a_point_at_equal_quality_is_comparable(self):
        result = amortization(train_cost=100.0, online_search=10.0, online_gen=2.0,
                              quality_gap=0.01, quality_tolerance=0.02, episodes=20)
        self.assertTrue(result["comparable"])
        self.assertTrue(result["pays_off"])

    def test_too_few_episodes_do_not_pay_off(self):
        result = amortization(train_cost=100.0, online_search=10.0, online_gen=2.0,
                              quality_gap=0.0, quality_tolerance=0.02, episodes=5)
        self.assertFalse(result["pays_off"])

    def test_a_missing_quality_gap_is_stated_not_assumed(self):
        result = amortization(train_cost=100.0, online_search=10.0, online_gen=2.0)
        self.assertFalse(result["comparable"])
        self.assertIn("not a speedup claim", result["reason"])

    def test_no_break_even_carries_its_reason(self):
        result = amortization(train_cost=100.0, online_search=2.0, online_gen=10.0,
                              quality_gap=0.0, episodes=1000)
        self.assertIsNone(result["break_even_episodes"])
        self.assertFalse(result["pays_off"])
        self.assertIn("not cheaper", result["reason"])


class PipelineAccountingTests(unittest.TestCase):
    """The counters have to sit on the real path, not only in unit tests.

    A ledger is a ``ContextVar``, so wrapping the CLI call is enough to charge
    everything the stages do -- which is also the check that the instrumentation
    was not left somewhere the pipeline never reaches.
    """

    def setUp(self):
        from pathlib import Path
        import shutil
        import tempfile

        self.directory = Path(tempfile.mkdtemp(prefix="mcat-costs-"))
        self.addCleanup(shutil.rmtree, self.directory, True)

    def _argv(self, command):
        return [command, "--output-dir", str(self.directory), "--fixture",
                "--domain", "qa", "--corpus-limit", "300", "--per-split", "2",
                "--documents", "16", "--support", "3", "--optimization", "4",
                "--evaluation", "6", "--poison-count", "2", "--trigger-tokens", "3",
                "--top-k", "3", "--steps", "2", "--max-length", "32"]

    def test_a_run_charges_training_and_writes_to_the_ledger(self):
        from src.triggers.mcat.cli import main

        ledger = CostLedger(device="cpu")
        with ledger.active():
            main(self._argv("prepare-episodes"))
            with ledger.phase("index"):
                main(self._argv("index"))
            with ledger.phase("train"):
                main(self._argv("train"))
            with ledger.phase("evaluate"):
                main(self._argv("evaluate"))

        train_phase = ledger.phase_named("train")
        # Two train episodes, two steps: one backward per episode per step and
        # one optimizer step per step, shared across episodes.
        self.assertEqual(train_phase.counters["optimizer_steps"], 2)
        self.assertEqual(train_phase.counters["encoder_backward"], 4)
        self.assertGreater(train_phase.counters["encoder_forward"], 0)
        self.assertGreater(ledger.phase_named("index").counters["encoder_forward"], 0)

        # Refreshing poison at evaluation is the attacker's index write: two
        # test episodes carrying two records each.
        self.assertEqual(ledger.phase_named("evaluate").counters["index_writes"], 4)
        # One GMM fit per distinct snapshot, never per context materialization.
        self.assertGreater(ledger.counters["gmm_refits"], 0)
        self.assertEqual(ledger.counters["scorer_calls"], 0)

    def test_counters_are_not_charged_when_no_ledger_is_active(self):
        from src.triggers.mcat.cli import main

        for command in ("prepare-episodes", "index", "train"):
            main(self._argv(command))
        self.assertIsNone(current())


if __name__ == "__main__":
    unittest.main()
