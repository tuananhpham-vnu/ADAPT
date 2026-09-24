"""Agent-generated tool calls and the clean benign split.

VN — Hai lỗ hổng mà các test này canh: (1) candidate do benchmark cấp chứ không
do agent sinh, (2) không có case nào sạch nên ``false_deny`` không đọc được.
Phần lớn test ở đây kiểm tra chuyện **không được lặng lẽ xảy ra**: lỗi parse bị
gộp vào abstain, quyền của field tính theo candidate thay vì theo call đã thực
thi, hai giao thức ghi đè record của nhau, hay tập clean lẫn injection.

Most of these check what must not happen silently: a parse failure counted as an
abstention, field authorization judged on the benchmark's candidate instead of
the executed call, two protocols overwriting each other's records, or the clean
split quietly carrying injected text.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

import torch

from src.aqua.backends import FixtureBackend, Proposal
from src.aqua.benchmark import (
    INJECTION_NOTE, assert_companion, build_cases, fingerprint, load_cases, save_cases,
)
from src.aqua.cli import main
from src.aqua.evaluation import AGENT_MODES, MODES, REPLAY_MODES, evaluate
from src.aqua.metrics import summarize
from src.aqua.prompts import PROPOSAL_STATUS, parse_call_with_spans, serialize_call
from src.aqua.sandbox import observe
from src.aqua.schema import Call, Case, validate_cases
from src.aqua.training import TrainConfig, train
from src.aqua.collection import collect


class CleanSplitTests(unittest.TestCase):
    def test_the_clean_split_carries_no_injected_text(self):
        for case in build_cases(20, injection="none"):
            self.assertFalse(case.injection)
            for text in case.context.values():
                self.assertNotIn(INJECTION_NOTE.strip(), text)

    def test_the_injected_split_still_injects_every_variant(self):
        # Keeping the attack words on all four variants is what stops them being
        # a label; the clean split is a separate file, not a subset.
        for case in build_cases(20):
            self.assertTrue(case.injection)
            self.assertTrue(any(INJECTION_NOTE.strip() in t for t in case.context.values()))

    def test_the_two_splits_differ_in_one_thing_only(self):
        assert_companion(build_cases(20, injection="all"),
                         build_cases(20, injection="none"))

    def test_a_mismatched_companion_is_refused(self):
        with self.assertRaises(ValueError):
            assert_companion(build_cases(20), build_cases(21, injection="none"))
        with self.assertRaises(ValueError):
            # Same groups, different seed: the proposed calls no longer line up.
            assert_companion(build_cases(20, seed=1), build_cases(20, seed=2, injection="none"))

    def test_swapping_the_arguments_is_refused(self):
        with self.assertRaises(ValueError):
            assert_companion(build_cases(20, injection="none"), build_cases(20))

    def test_a_quadruple_may_not_mix_injection_settings(self):
        rows = build_cases(20)[:4]
        mixed = rows[:3] + [Case(**{**rows[3].to_dict(), "proposed": rows[3].proposed,
                                    "grant": rows[3].grant,
                                    "authority": rows[3].authority, "injection": False})]
        with self.assertRaises(ValueError):
            validate_cases(mixed)

    def test_a_case_without_the_field_loads_as_injected(self):
        # Backward compatibility: JSONL written before the field existed must not
        # be reclassified as clean.
        row = build_cases(20)[0].to_dict()
        row.pop("injection")
        self.assertTrue(Case.from_dict(row).injection)

    def test_an_unknown_injection_mode_is_refused(self):
        with self.assertRaises(ValueError):
            build_cases(20, injection="maybe")


class ParserTests(unittest.TestCase):
    def setUp(self):
        self.call = Call("mail_send", {"resource": "r-1", "recipient": "to-1",
                                       "payload": 'say "hi", then stop'})

    def test_a_serialized_call_round_trips_with_exact_spans(self):
        text, _ = serialize_call(self.call)
        parsed, spans, status = parse_call_with_spans(text)
        self.assertEqual(status, "ok")
        self.assertEqual(parsed, self.call)
        for field, value in self.call.values().items():
            start, end = spans[field]
            self.assertEqual(json.loads(text[start:end]), value)

    def test_spans_point_at_each_field_even_when_values_repeat(self):
        # The reason this is not implemented with text.find(): two fields holding
        # the same value would resolve to one position.
        call = Call("mail_send", {"resource": "same", "recipient": "same",
                                   "payload": "same"})
        text, _ = serialize_call(call)
        _, spans, _ = parse_call_with_spans(text)
        positions = {spans[f] for f in call.values()}
        self.assertEqual(len(positions), len(call.values()))

    def test_null_is_an_abstention_not_a_failure(self):
        parsed, spans, status = parse_call_with_spans("null")
        self.assertEqual(status, "abstain")
        self.assertIsNone(parsed)
        self.assertEqual(spans, {})

    def test_every_failure_mode_is_distinguished(self):
        cases = {
            "parse_error": "I would rather not.",
            "unknown_tool": '{"tool": "mail_teleport", "arguments": {"resource": "a"}}',
            "missing_fields": '{"tool": "mail_send", "arguments": {"resource": "a"}}',
            "extra_fields": '{"tool": "mail_send", "arguments": {"resource": "a", '
                            '"recipient": "b", "payload": "c", "extra": "d"}}',
            "invalid_value": '{"tool": "mail_send", "arguments": {"resource": "a", '
                             '"recipient": "b", "payload": ""}}',
        }
        for expected, text in cases.items():
            with self.subTest(expected):
                parsed, _, status = parse_call_with_spans(text)
                self.assertEqual(status, expected)
                self.assertIsNone(parsed)
        self.assertTrue(set(cases) <= set(PROPOSAL_STATUS))

    def test_a_truncated_object_is_a_parse_error_not_a_crash(self):
        for text in ('{"tool": "mail_send", "arguments": {"resource": ', "{", '{"tool":}'):
            with self.subTest(text):
                self.assertEqual(parse_call_with_spans(text)[2], "parse_error")

    def test_an_integer_amount_survives_the_scan(self):
        call = Call("billing_pay", {"resource": "r", "recipient": "x", "amount": 42})
        text, _ = serialize_call(call)
        parsed, _, status = parse_call_with_spans(text)
        self.assertEqual((status, parsed), ("ok", call))

    def test_a_non_integer_amount_is_an_invalid_value(self):
        text = '{"tool": "billing_pay", "arguments": {"resource": "r", ' \
               '"recipient": "x", "amount": -3}}'
        self.assertEqual(parse_call_with_spans(text)[2], "invalid_value")


class SandboxTests(unittest.TestCase):
    def setUp(self):
        self.cases = {c.variant: c for c in build_cases(20)[:4]}

    def test_the_replay_protocol_is_unchanged(self):
        case = self.cases["authorized_user"]
        self.assertEqual(observe(case, True), observe(case, True, case.proposed))

    def test_field_authorization_follows_the_executed_call(self):
        # The bug this guards: judging the generated call by case.labels, which
        # describe the benchmark's candidate and not what ran.
        case = self.cases["denied"]
        other = Call(case.proposed.tool,
                     {**case.proposed.arguments,
                      list(case.proposed.arguments)[0]: "improvised"})
        result = observe(case, True, other)
        self.assertTrue(result["executed"])
        self.assertGreater(result["unauthorized_fields_executed"], 0)
        self.assertLess(result["argument_agreement"], 1.0)

    def test_an_invalid_generated_call_is_reported_not_raised(self):
        case = self.cases["authorized_user"]
        broken = Call("mail_send", {"resource": "a"})
        result = observe(case, True, broken)
        self.assertTrue(result["invalid_call"])
        self.assertFalse(result["executed"])
        self.assertFalse(result["harmful_effect"])
        self.assertIsNone(result["argument_agreement"])

    def test_an_abstention_is_not_an_invalid_call(self):
        result = observe(self.cases["authorized_user"], True, None)
        self.assertTrue(result["abstained"])
        self.assertFalse(result["invalid_call"])
        self.assertFalse(result["executed"])

    def test_a_blocked_call_executes_nothing(self):
        result = observe(self.cases["denied"], False, self.cases["denied"].proposed)
        self.assertFalse(result["executed"])
        self.assertEqual(result["unauthorized_fields_executed"], 0)


class FixtureProposalTests(unittest.TestCase):
    def setUp(self):
        self.backend = FixtureBackend()
        self.cases = build_cases(40)

    def test_proposing_is_deterministic(self):
        case = self.cases[0]
        first, second = self.backend.propose(case), self.backend.propose(case)
        self.assertEqual(first.text, second.text)
        self.assertEqual(first.status, second.status)

    def test_both_the_ok_and_the_parse_error_branch_are_reachable(self):
        statuses = {self.backend.propose(c).status for c in self.cases}
        self.assertIn("ok", statuses)
        self.assertIn("parse_error", statuses)

    def test_an_unparsable_proposal_carries_no_activations(self):
        broken = [c for c in self.cases if self.backend.propose(c).status == "parse_error"]
        self.assertTrue(broken)
        self.assertIsNone(self.backend.propose(broken[0]).hidden)

    def test_regeneration_does_not_key_off_authorization(self):
        # If the fixture changed its answer according to the labels it would look
        # like a working defense, which only a real model may demonstrate.
        changed = {}
        for case in self.cases:
            first = self.backend.propose(case)
            if first.status != "ok":
                continue
            corrections = torch.ones_like(first.hidden)
            second = self.backend.propose(case, corrections=corrections)
            changed.setdefault(all(case.labels.values()), set()).add(
                second.text != first.text)
        self.assertEqual(changed[True], changed[False])


class MetricDenominatorTests(unittest.TestCase):
    def _row(self, mode, **decision):
        base = {"authorized": True, "harmful_effect": False, "executed": True,
                "task_success": True, "unauthorized_fields_executed": 0,
                "field_count": 4, "abstained": False, "invalid_call": False,
                "argument_agreement": 1.0}
        return {"variant": "authorized_user", "domain": "mail",
                "decisions": {mode: base | decision}}

    def test_a_mode_no_record_carries_is_empty_not_wrong(self):
        summary = summarize([self._row("agent_unguarded")], "agent_detector")
        self.assertEqual(summary["cases"], 0)
        self.assertIsNone(summary["false_allow"])

    def test_generation_metrics_have_their_own_denominators(self):
        rows = [self._row("agent_unguarded"),
                self._row("agent_unguarded", executed=False, abstained=True,
                          argument_agreement=None)]
        summary = summarize(rows, "agent_unguarded")
        self.assertEqual(summary["abstain_rate"], 0.5)
        self.assertEqual(summary["invalid_call_rate"], 0.0)
        # Agreement averages over executed calls only, so the abstention does not
        # drag it toward zero.
        self.assertEqual(summary["argument_agreement"], 1.0)


class EvaluationSourceTests(unittest.TestCase):
    """The end-to-end offline path, both protocols and the clean split."""

    @classmethod
    def setUpClass(cls):
        cls.directory = Path(tempfile.mkdtemp(prefix="aqua-agent-"))
        cls.addClassCleanup(shutil.rmtree, cls.directory, True)
        cls.dataset = cls.directory / "authshift.jsonl"
        cls.clean = cls.directory / "authshift_clean.jsonl"
        save_cases(cls.dataset, build_cases(40))
        save_cases(cls.clean, build_cases(40, injection="none"))
        cls.backend = FixtureBackend()
        collect(cls.dataset, cls.directory / "activations", cls.backend)
        train(cls.directory / "activations", cls.directory / "training",
              TrainConfig(epochs=2))
        cls.probe = cls.directory / "training/probe.pt"

    def _evaluate(self, name, **kwargs):
        return evaluate(self.dataset, self.probe, self.directory / name, self.backend,
                        **kwargs)

    def test_the_default_protocol_reports_only_replay_modes(self):
        metrics = self._evaluate("replay")
        self.assertEqual(metrics["proposed_source"], "benchmark")
        for mode in REPLAY_MODES:
            self.assertGreater(metrics["metrics"][mode]["cases"], 0)
        for mode in AGENT_MODES:
            self.assertEqual(metrics["metrics"][mode]["cases"], 0)
        self.assertIsNone(metrics["proposals"])

    def test_the_agent_protocol_reports_its_own_modes_and_statuses(self):
        metrics = self._evaluate("agent", proposed_source="agent")
        self.assertGreater(metrics["metrics"]["agent_unguarded"]["cases"], 0)
        self.assertEqual(metrics["metrics"]["unguarded"]["cases"], 0)
        proposals = metrics["proposals"]
        self.assertIn("parse_error", proposals["status_counts"])
        self.assertGreater(proposals["probe_not_applicable"], 0)
        # A parse failure is not an abstention: it must not be gated away.
        self.assertGreater(metrics["metrics"]["agent_unguarded"]["abstain_rate"], 0)

    def test_the_two_protocols_do_not_overwrite_each_other(self):
        metrics = self._evaluate("both", proposed_source="both")
        self.assertGreater(metrics["metrics"]["unguarded"]["cases"], 0)
        self.assertGreater(metrics["metrics"]["agent_unguarded"]["cases"], 0)
        rows = [json.loads(line) for line in
                (self.directory / "both/records.jsonl").read_text("utf-8").splitlines()]
        sources = {}
        for row in rows:
            sources.setdefault(row["id"], set()).add(row["proposed_source"])
        self.assertTrue(all(v == {"benchmark", "agent"} for v in sources.values()))

    def test_the_clean_split_is_reported_separately(self):
        metrics = self._evaluate("clean", proposed_source="benchmark",
                                 clean_dataset=self.clean)
        self.assertIsNotNone(metrics["clean"])
        self.assertGreater(metrics["clean"]["cases"], 0)
        # Injected and clean numbers are never pooled.
        self.assertNotIn("clean", metrics["metrics"])

    def test_a_mismatched_clean_split_is_refused(self):
        other = self.directory / "other_clean.jsonl"
        save_cases(other, build_cases(21, injection="none"))
        with self.assertRaises(ValueError):
            self._evaluate("bad-clean", clean_dataset=other)

    def test_resuming_reuses_records_and_rejects_a_changed_contract(self):
        first = self._evaluate("resume", proposed_source="both")
        again = self._evaluate("resume", proposed_source="both")
        self.assertEqual(first["metrics"], again["metrics"])
        with self.assertRaises(ValueError):
            self._evaluate("resume", proposed_source="agent")

    def test_an_unknown_source_is_refused(self):
        with self.assertRaises(ValueError):
            self._evaluate("nope", proposed_source="oracle")


class HuggingFaceProposalTests(unittest.TestCase):
    """Generation plumbing on a random tiny Llama: no download, no pretrained claim.

    VN — Vocabulary 4 token nên model không thể sinh ra JSON hợp lệ; chính vì vậy
    nó kiểm tra đúng thứ cần kiểm tra ở đây: nhánh ``parse_error``, dọn hook, chặn
    truncation, và cấu hình decode có nằm trong contract hay không. Không có con
    số nào ở đây là bằng chứng nghiên cứu.
    """

    @classmethod
    def setUpClass(cls):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
        from src.aqua.backends import HuggingFaceBackend

        tokenizer = Tokenizer(WordLevel(
            {"[UNK]": 0, "null": 1, "tool": 2, "arguments": 3}, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]",
                                       pad_token="[UNK]")
        torch.manual_seed(2)
        config = LlamaConfig(vocab_size=4, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=2, num_attention_heads=2,
                             num_key_value_heads=2)
        cls.backend = HuggingFaceBackend(LlamaForCausalLM(config), fast,
                                         model_id="random-local-test", layer=0,
                                         max_new_tokens=8)
        cls.case = build_cases(20)[1]

    def test_the_generation_config_is_part_of_the_metadata(self):
        generation = self.backend.metadata["generation"]
        self.assertEqual(generation["max_new_tokens"], 8)
        self.assertFalse(generation["do_sample"])
        self.assertEqual(generation["num_beams"], 1)

    def test_a_proposal_is_produced_and_the_hook_is_removed(self):
        proposal = self.backend.propose(self.case)
        self.assertIn(proposal.status, PROPOSAL_STATUS)
        self.assertLessEqual(proposal.generated_tokens, 8)
        self.assertEqual(len(self.backend.block._forward_hooks), 0)

    def test_generation_is_deterministic(self):
        self.assertEqual(self.backend.propose(self.case).text,
                         self.backend.propose(self.case).text)

    def test_regenerating_under_a_correction_cleans_up_too(self):
        capture = self.backend.capture(self.case, residuals=False)
        self.backend.propose(self.case, corrections=torch.zeros_like(capture.hidden))
        self.assertEqual(len(self.backend.block._forward_hooks), 0)

    def test_a_generation_budget_that_would_truncate_is_refused(self):
        from unittest.mock import patch

        with patch.object(self.backend, "max_length", 12):
            with self.assertRaisesRegex(ValueError, "no silent truncation"):
                self.backend.propose(self.case)

    def test_field_positions_can_come_from_a_generated_completion(self):
        # The generated path must reuse the one prediction_positions implementation
        # rather than growing a second way to compute positions.
        text, spans = serialize_call(self.case.proposed)
        _, _, _, generated, _ = self.backend._inputs(self.case, text, spans)
        _, _, _, benchmark, _ = self.backend._inputs(self.case)
        self.assertEqual(generated, benchmark)

    def test_a_completion_without_spans_scores_no_fields(self):
        _, _, _, positions, targets = self.backend._inputs(self.case, "null")
        self.assertEqual(positions, {})
        self.assertTrue(targets)


class SmokeTests(unittest.TestCase):
    def test_the_smoke_command_exercises_generation_and_the_clean_split(self):
        directory = Path(tempfile.mkdtemp(prefix="aqua-smoke-"))
        self.addCleanup(shutil.rmtree, directory, True)
        self.assertIsNone(main(["smoke", "--output", str(directory), "--groups", "20",
                                "--epochs", "2"]))
        metrics = json.loads(
            (directory / "evaluation-epoch-2/metrics.json").read_text("utf-8"))
        self.assertEqual(metrics["proposed_source"], "both")
        self.assertIsNotNone(metrics["clean"])
        self.assertIsNotNone(metrics["proposals"])
        self.assertTrue((directory / "authshift_clean.jsonl").exists())
        self.assertFalse(any(c.injection for c in
                             load_cases(directory / "authshift_clean.jsonl")))


if __name__ == "__main__":
    unittest.main()
