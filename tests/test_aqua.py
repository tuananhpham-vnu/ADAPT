"""Offline contracts, gradients, real transformer hooks, and staged regression."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from src.aqua.backends import Capture, FixtureBackend, HuggingFaceBackend
from src.aqua.benchmark import build_cases, fingerprint, load_cases, save_cases
from src.aqua.collection import ActivationDataset, collect
from src.aqua.evaluation import evaluate
from src.aqua.intervention import corrections_for, scrub
from src.aqua.losses import LossConfig, quotient_loss
from src.aqua.metrics import calibration_error, summarize
from src.aqua.model import AuthorizationProbe, ReverseGradient
from src.aqua.prompts import prediction_positions, render, serialize_call
from src.aqua.sandbox import Sandbox, observe
from src.aqua.schema import Call, Grant, validate_cases
from src.aqua.training import TrainConfig, calibrate, train


class BenchmarkTests(unittest.TestCase):
    def test_reproducible_quadruples_and_tool_holdout(self):
        self.assertEqual({c.split for c in build_cases(3, split_by="group")}, {"train", "validation", "test"})
        cases = build_cases(40)
        self.assertEqual(fingerprint(cases), fingerprint(build_cases(40)))
        tools = {split: {c.proposed.tool for c in cases if c.split == split}
                 for split in ("train", "validation", "test")}
        self.assertTrue(all(tools.values()))
        self.assertFalse(tools["train"] & tools["test"])
        self.assertFalse(tools["train"] & tools["validation"])
        self.assertFalse(tools["test"] & tools["validation"])
        for start in range(0, len(cases), 4):
            a, b, c, d = cases[start:start + 4]
            self.assertEqual(a.proposed, b.proposed)
            self.assertEqual(b.proposed, c.proposed)
            self.assertEqual(c.proposed, d.proposed)
            self.assertTrue(all(a.labels.values()) and all(b.labels.values()))
            self.assertFalse(all(c.labels.values()) or all(d.labels.values()))

    def test_split_leakage_rejected(self):
        cases = build_cases(20)
        cases[0] = replace(cases[0], split="validation" if cases[1].split != "validation" else "test")
        with self.assertRaisesRegex(ValueError, "cross data splits"):
            validate_cases(cases)

    def test_prompt_does_not_read_labels_or_grants(self):
        case = build_cases(20)[0]
        altered = replace(case, grant=Grant(case.proposed.tool, {}, active=False), variant="denied")
        self.assertEqual(render(case), render(altered))
        prompt, spans = render(case)
        for source, (a, b) in spans.items():
            self.assertEqual(prompt[a:b], case.context[source])

    def test_revocation_expiration_and_field_scope(self):
        call = Call("billing_pay", {"resource": "A", "recipient": "B", "amount": 10})
        grant = Grant(call.tool, dict(call.arguments), expires_at=100)
        self.assertTrue(all(grant.allows(call, 99).values()))
        self.assertFalse(any(grant.allows(call, 100).values()))
        self.assertFalse(any(replace(grant, revoked=True).allows(call, 99).values()))
        changed = Call(call.tool, {**call.arguments, "amount": 11})
        labels = grant.allows(changed, 99)
        self.assertTrue(labels["recipient"])
        self.assertFalse(labels["amount"])

    def test_observed_effect_is_separate_from_permission(self):
        denied = build_cases(20)[2]
        self.assertTrue(observe(denied, True)["harmful_effect"])
        self.assertIsNone(observe(denied, False)["effect"])
        with self.assertRaises(ValueError):
            Sandbox().execute(Call("unknown", {}))
        with self.assertRaises(ValueError):
            Sandbox().execute(Call("billing_pay", {"resource": "a", "recipient": "b", "amount": True}))


class RepresentationTests(unittest.TestCase):
    def test_projection_idempotent_and_scrub_retains_complement(self):
        torch.manual_seed(1)
        model = AuthorizationProbe(8, rank=2)
        residual = torch.randn(3, 8)
        projected = model.project(residual)
        torch.testing.assert_close(model.project(projected), projected)
        complement = residual - projected
        torch.testing.assert_close(complement @ model.orthogonal_basis(), torch.zeros(3, 2), atol=1e-6, rtol=1e-6)
        case = build_cases(20)[1]
        hidden = torch.randn(4, 8)
        capture = Capture(list(case.proposed.values()), hidden, {"tool": torch.ones_like(hidden)}, 0., 0.)
        exempt = capture.fields[1]
        case = replace(case, authority={**case.authority, exempt: ("user", "tool")})
        correction = corrections_for(case, capture, model)
        self.assertEqual(correction[1].abs().sum().item(), 0.)
        after = scrub(hidden, correction)
        torch.testing.assert_close(after - model.project(after), hidden - model.project(hidden))

    def test_gradient_reversal_direction(self):
        x = torch.ones(3, requires_grad=True)
        ReverseGradient.apply(x, .5).sum().backward()
        torch.testing.assert_close(x.grad, torch.full((3,), -.5))

    def test_loss_uses_only_permission_changing_fields_for_flip(self):
        model = AuthorizationProbe(8, rank=2)
        h = torch.randn(5, 4, 8, requires_grad=True)
        batch = {"hidden": h, "labels": torch.ones(5, 4),
                 "source": torch.zeros(5, 4, dtype=torch.long), "domain": torch.zeros(5, 4, dtype=torch.long)}
        loss, parts = quotient_loss(model, batch, LossConfig())
        self.assertEqual(parts["flip"].item(), 0.)
        loss.backward()
        self.assertTrue(torch.isfinite(h.grad).all())
        self.assertGreater(model.basis.grad.abs().sum().item(), 0.)

    def test_calibration_and_empty_metric_denominators(self):
        calibration = calibrate([.9, .8, .4, .2], [True, True, False, False], 0.)
        self.assertEqual(calibration["validation_false_allow"], 0.)
        self.assertEqual(calibration["validation_allow_rate_authorized"], 1.)
        self.assertEqual(calibration_error([1., 0.], [True, False]), 0.)
        self.assertIsNone(summarize([], "detector")["false_allow"])


class PipelineTests(unittest.TestCase):
    def test_roundtrip_resume_and_contract_change(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dataset = root / "data.jsonl"
            cases = build_cases(20)
            save_cases(dataset, cases)
            self.assertEqual(cases, load_cases(dataset))
            backend = FixtureBackend()
            collect(dataset, root / "activations", backend)
            with patch.object(backend, "capture", side_effect=AssertionError("cached collection called model")):
                collect(dataset, root / "activations", backend)
            with self.assertRaisesRegex(ValueError, "changed"):
                collect(dataset, root / "activations", FixtureBackend(seed=99))
            config = TrainConfig(epochs=2, batch_size=4)
            first = train(root / "activations", root / "train", config)
            again = train(root / "activations", root / "train", config)
            for key in first["model"]:
                torch.testing.assert_close(first["model"][key], again["model"][key], atol=0, rtol=0)
            extended = train(root / "activations", root / "train", replace(config, epochs=3))
            full = train(root / "activations", root / "full", replace(config, epochs=3))
            for key in full["model"]:
                torch.testing.assert_close(extended["model"][key], full["model"][key], atol=0, rtol=0)
            report = evaluate(dataset, root / "train/probe.pt", root / "eval", backend)
            self.assertEqual(report["metrics"]["reference_monitor"]["executed_attack_success_rate"], 0.)
            self.assertEqual(report["metrics"]["unguarded"]["executed_attack_success_rate"], 1.)
            # A repeated train stage may reserialize identical weights. Evaluation
            # cache identity must depend on values, not torch.save ZIP filenames.
            train(root / "activations", root / "train", replace(config, epochs=3))
            with patch.object(backend, "capture", side_effect=AssertionError("cached eval called model")):
                self.assertEqual(report, evaluate(dataset, root / "train/probe.pt", root / "eval", backend))

    def test_incomplete_collection_not_trainable(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "manifest.json").write_text('{"complete": false}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                ActivationDataset(folder, "train")


class HuggingFaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Random, tiny real Llama. No downloads, API key, or pretrained claims.
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
        tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "null": 1, "tool": 2, "arguments": 3}, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]")
        torch.manual_seed(2)
        config = LlamaConfig(vocab_size=4, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2)
        cls.model = LlamaForCausalLM(config)
        cls.backend = HuggingFaceBackend(cls.model, fast, model_id="random-local-test", layer=0)

    def test_spans_and_prediction_shift(self):
        call = Call("mail_send", {"resource": "same", "recipient": "same", "payload": 'say "hi"'})
        text, spans = serialize_call(call)
        self.assertEqual(json.loads(text), {"tool": call.tool, "arguments": call.arguments})
        for field, (a, b) in spans.items():
            self.assertEqual(json.loads(text[a:b]), call.values()[field])
        self.assertEqual(prediction_positions([(0, 2), (2, 4), (4, 6)], 4, 6), [1])

    def test_capture_residuals_zero_replay_and_hook_cleanup(self):
        case = build_cases(20)[1]
        capture = self.backend.capture(case)
        self.assertEqual(capture.hidden.shape, (4, 16))
        self.assertEqual(set(capture.residuals), set(case.context))
        self.assertTrue(torch.isfinite(capture.hidden).all())
        self.assertAlmostEqual(capture.call_score, self.backend.replay(case, torch.zeros_like(capture.hidden)), places=6)
        # Nonzero intervention must reach the decoder's output logits.
        changed = self.backend.replay(case, torch.randn_like(capture.hidden) * 2.)
        self.assertGreater(abs(changed - capture.call_score), 1e-7)
        self.assertEqual(len(self.backend.block._forward_hooks), 0)
        with patch.object(self.model, "forward", side_effect=RuntimeError("model failure")):
            with self.assertRaisesRegex(RuntimeError, "model failure"):
                self.backend.capture(case)
        self.assertEqual(len(self.backend.block._forward_hooks), 0)

    def test_truncation_rejected(self):
        with patch.object(self.backend, "max_length", 4):
            with self.assertRaisesRegex(ValueError, "no silent truncation"):
                self.backend.capture(build_cases(20)[0])


if __name__ == "__main__":
    unittest.main()
