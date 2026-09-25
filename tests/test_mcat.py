"""Offline CPU checks for the MCAT episode, encoding and relaxation contracts.

Everything here runs without a GPU and without downloading a checkpoint: the
retriever is the fixture encoder: a randomly initialized two-layer BERT over a
45-token vocabulary.
That is enough to verify the properties that are expensive to get wrong -- the
gradient path, the encode parity and the margin definition.
"""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from src.triggers.losses import compute_retrieval_margin_loss
from src.triggers.mcat.domains import AD_MISSING, load_domain
from src.triggers.mcat.encoding import (
    encode_with_trigger_embeddings, freeze_retriever, trigger_embeddings_from_ids,
)
from src.triggers.mcat.episodes import (
    Episode, EpisodeSizes, assert_no_family_leak, build_episodes, build_manifest,
    scale_sizes,
)
from src.triggers.mcat.objectives import (
    compute_hit_at_k_margin_loss, mcat_total_loss, PoisonPolicy,
)
from src.triggers.mcat.retrievers import build_fixture_retriever
from src.triggers.mcat.relaxation import (
    TriggerLogits, allowed_vocab_mask, export_hard_trigger, hard_token_ids,
    round_trip_report, straight_through_gumbel, to_trigger_embeddings,
)

ROOT = Path(__file__).resolve().parents[1]
MAX_LENGTH = 32


def tiny_retriever():
    """The shared CPU fixture encoder, unwrapped for the low-level tests."""
    retriever = build_fixture_retriever(max_length=MAX_LENGTH)
    return retriever.model, retriever.tokenizer


class SyntheticDomain:
    """Two families per split with predictable ids, for fast episode checks."""

    name = "synthetic"

    def __init__(self, families: int = 30, per_family: int = 12):
        self._documents = [
            {"doc_id": f"d{family:02d}-{index:02d}", "text": f"document {family} {index}",
             "family": f"fam{family:02d}"}
            for family in range(families) for index in range(per_family)
        ]
        self._queries = [
            {"qid": f"q{family:02d}-{index:02d}", "question": f"question {family} {index}",
             "family": f"fam{family:02d}"}
            for family in range(families) for index in range(per_family)
        ]

    def documents(self):
        return list(self._documents)

    def queries(self):
        return list(self._queries)

    def fingerprint(self):
        return {"name": self.name}


SMALL = EpisodeSizes(documents=8, support=3, optimization=4, evaluation=5,
                     poison_sources=2, trigger_tokens=4, retrieval_top_k=3)


class EpisodeTests(unittest.TestCase):
    def test_query_groups_stay_disjoint(self):
        episodes, _ = build_episodes(SyntheticDomain(), seed=0, sizes=SMALL, per_split=3)
        self.assertEqual(len(episodes), 9)
        for episode in episodes:
            episode.assert_disjoint()

    def test_disjointness_is_actually_checked(self):
        episode = Episode(
            episode_id="e", domain="d", split="train", snapshot_id="s",
            doc_ids=["a"], support_qids=["q1"], optimization_qids=["q1"],
            eval_qids=["q2"], poison_source_qids=["q3"],
            budget_poison=1, trigger_tokens=4, retrieval_top_k=3,
        )
        with self.assertRaises(ValueError):
            episode.assert_disjoint()

    def test_no_family_crosses_the_outer_split(self):
        domain = SyntheticDomain()
        episodes, _ = build_episodes(domain, seed=1, sizes=SMALL, per_split=3)
        assert_no_family_leak(episodes, domain)

    def test_split_hash_is_deterministic_and_seed_sensitive(self):
        domain = SyntheticDomain()
        first, _ = build_episodes(domain, seed=0, sizes=SMALL, per_split=2)
        again, _ = build_episodes(domain, seed=0, sizes=SMALL, per_split=2)
        other, _ = build_episodes(domain, seed=7, sizes=SMALL, per_split=2)
        self.assertEqual([e.to_json() for e in first], [e.to_json() for e in again])
        self.assertNotEqual([e.to_json() for e in first], [e.to_json() for e in other])

    def test_thin_domain_is_scaled_and_recorded_not_padded(self):
        scaled, notes = scale_sizes(EpisodeSizes(), documents=40, queries=60)
        self.assertEqual(notes["documents"], {"requested": 512, "available": 40})
        self.assertEqual(scaled.documents, 40)
        self.assertLessEqual(scaled.queries_needed, 60)
        self.assertEqual(notes["queries"]["available"], 60)

    def test_scaling_leaves_a_sufficient_domain_untouched(self):
        scaled, notes = scale_sizes(SMALL, documents=100, queries=100)
        self.assertEqual(notes, {})
        self.assertEqual(scaled, SMALL)


class DomainTests(unittest.TestCase):
    def test_agentdriver_reports_the_missing_corpus(self):
        domain = load_domain("ad")
        with self.assertRaises(FileNotFoundError) as caught:
            domain.documents()
        self.assertIn("data_samples_train.json", str(caught.exception))
        self.assertIn("agentdriver/data/finetune", str(caught.exception))
        self.assertIn("split.json", AD_MISSING)

    @unittest.skipUnless((ROOT / "ReAct/database/strategyqa_train_paragraphs.json").exists(),
                         "StrategyQA corpus is not vendored")
    def test_strategyqa_documents_and_queries_carry_families(self):
        domain = load_domain("qa")
        documents, queries = domain.documents(), domain.queries()
        self.assertGreater(len(documents), 1000)
        self.assertTrue(all(row["family"] for row in documents))
        self.assertTrue(all(row["family"] for row in queries))
        qids = [row["qid"] for row in queries]
        self.assertEqual(len(qids), len(set(qids)))

    @unittest.skipUnless((ROOT / "ReAct/database/strategyqa_train_paragraphs.json").exists(),
                         "StrategyQA corpus is not vendored")
    def test_real_manifest_pins_inputs_and_splits(self):
        episodes, manifest = build_manifest(["qa"], seed=0, sizes=SMALL, per_split=2)
        self.assertEqual(len(episodes), 6)
        self.assertIn("corpus_sha256", manifest["domains"]["qa"])
        self.assertEqual(len(manifest["split_hash"]), 64)
        assert_no_family_leak(episodes, load_domain("qa"))


class EncodingTests(unittest.TestCase):
    def setUp(self):
        self.model, self.tokenizer = tiny_retriever()
        freeze_retriever(self.model)
        self.texts = ["w1 w2 w3", "w4", "w5 w6 w7 w8 w9"]
        self.trigger_ids = torch.tensor([10, 11, 12, 13])

    def _id_path(self) -> torch.Tensor:
        """The same sequence assembled from ids, as the runtime would build it."""
        pieces = []
        for text in self.texts:
            prefix = self.tokenizer(text, add_special_tokens=False, truncation=True,
                                    max_length=MAX_LENGTH - len(self.trigger_ids) - 2).input_ids
            pieces.append([self.tokenizer.cls_token_id] + prefix
                          + self.trigger_ids.tolist() + [self.tokenizer.sep_token_id])
        longest = max(len(piece) for piece in pieces)
        ids = torch.tensor([piece + [0] * (longest - len(piece)) for piece in pieces])
        mask = torch.tensor([[1] * len(piece) + [0] * (longest - len(piece)) for piece in pieces])
        return self.model(input_ids=ids, attention_mask=mask).pooler_output

    def test_embedding_path_matches_the_id_path(self):
        embeds = trigger_embeddings_from_ids(self.model, self.trigger_ids)
        actual = encode_with_trigger_embeddings(
            self.model, self.tokenizer, self.texts, embeds,
            device="cpu", max_length=MAX_LENGTH,
        )
        torch.testing.assert_close(actual, self._id_path(), rtol=1e-5, atol=1e-6)

    def test_gradient_reaches_the_trigger_and_not_the_retriever(self):
        embeds = trigger_embeddings_from_ids(self.model, self.trigger_ids).clone()
        embeds.requires_grad_(True)
        output = encode_with_trigger_embeddings(
            self.model, self.tokenizer, self.texts, embeds,
            device="cpu", max_length=MAX_LENGTH,
        )
        output.pow(2).mean().backward()
        self.assertIsNotNone(embeds.grad)
        self.assertTrue(torch.isfinite(embeds.grad).all())
        self.assertGreater(float(embeds.grad.abs().sum()), 0)
        self.assertIsNone(self.model.get_input_embeddings().weight.grad)

    def test_trigger_longer_than_the_budget_is_rejected(self):
        embeds = torch.zeros(MAX_LENGTH, self.model.config.hidden_size)
        with self.assertRaises(ValueError):
            encode_with_trigger_embeddings(
                self.model, self.tokenizer, self.texts, embeds,
                device="cpu", max_length=MAX_LENGTH,
            )

    def test_batch_of_mixed_lengths_masks_the_padding(self):
        embeds = trigger_embeddings_from_ids(self.model, self.trigger_ids)
        both = encode_with_trigger_embeddings(
            self.model, self.tokenizer, self.texts, embeds,
            device="cpu", max_length=MAX_LENGTH,
        )
        alone = encode_with_trigger_embeddings(
            self.model, self.tokenizer, self.texts[1:2], embeds,
            device="cpu", max_length=MAX_LENGTH,
        )
        # Padding must not change a row's encoding.
        torch.testing.assert_close(both[1:2], alone, rtol=1e-5, atol=1e-6)


class RelaxationTests(unittest.TestCase):
    def setUp(self):
        self.model, self.tokenizer = tiny_retriever()
        freeze_retriever(self.model)
        self.vocab = self.model.get_input_embeddings().weight.shape[0]
        self.mask = allowed_vocab_mask(self.tokenizer, self.vocab)

    def test_mask_excludes_special_tokens(self):
        for token_id in self.tokenizer.all_special_ids:
            self.assertFalse(bool(self.mask[token_id]), f"token {token_id} should be masked")
        self.assertGreater(int(self.mask.sum()), 10)

    def test_straight_through_forward_is_one_hot(self):
        logits = torch.randn(4, self.vocab)
        relaxed = straight_through_gumbel(logits, tau=0.5, mask=self.mask)
        torch.testing.assert_close(relaxed.sum(dim=-1), torch.ones(4))
        self.assertEqual(int((relaxed == 1).sum()), 4)

    def test_hard_forward_equals_the_embedding_of_the_chosen_token(self):
        logits = torch.randn(4, self.vocab)
        relaxed = straight_through_gumbel(logits, tau=0.5, mask=self.mask)
        weight = self.model.get_input_embeddings().weight
        actual = to_trigger_embeddings(relaxed, weight)
        expected = weight[relaxed.argmax(dim=-1)]
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_masked_tokens_are_never_selected(self):
        logits = torch.zeros(6, self.vocab)
        logits[:, self.tokenizer.cls_token_id] = 50.0
        relaxed = straight_through_gumbel(logits, tau=0.5, mask=self.mask)
        self.assertFalse(
            bool((relaxed.argmax(dim=-1) == self.tokenizer.cls_token_id).any())
        )

    def test_gradient_flows_into_the_logits(self):
        module = TriggerLogits(4, self.vocab)
        weight = self.model.get_input_embeddings().weight
        embeds = to_trigger_embeddings(module(tau=1.0, mask=self.mask), weight)
        encode_with_trigger_embeddings(
            self.model, self.tokenizer, ["w1 w2"], embeds,
            device="cpu", max_length=MAX_LENGTH,
        ).pow(2).mean().backward()
        self.assertIsNotNone(module.logits.grad)
        self.assertTrue(torch.isfinite(module.logits.grad).all())
        self.assertGreater(float(module.logits.grad.abs().sum()), 0)
        self.assertIsNone(self.model.get_input_embeddings().weight.grad)

    def test_export_is_deterministic_and_noise_free(self):
        logits = torch.randn(4, self.vocab)
        first = export_hard_trigger(logits, self.tokenizer, self.mask)
        again = export_hard_trigger(logits, self.tokenizer, self.mask)
        self.assertEqual(first, again)
        self.assertEqual(first["token_ids"], hard_token_ids(logits, self.mask).tolist())

    def test_round_trip_accepts_a_clean_trigger(self):
        trigger = {"token_ids": [10, 11, 12], "trigger": "w5 w6 w7"}
        report = round_trip_report(trigger, self.tokenizer, query="w1 w2")
        self.assertTrue(report["length_match"], report)
        self.assertTrue(report["valid"], report)
        self.assertFalse(report["special_token_used"])

    def test_round_trip_flags_a_special_token(self):
        trigger = {"token_ids": [self.tokenizer.cls_token_id, 11],
                   "trigger": self.tokenizer.cls_token + " w6"}
        report = round_trip_report(trigger, self.tokenizer)
        self.assertTrue(report["special_token_used"])
        self.assertFalse(report["valid"])

    def test_round_trip_flags_an_empty_trigger(self):
        report = round_trip_report({"token_ids": [], "trigger": "   "}, self.tokenizer)
        self.assertTrue(report["empty"])
        self.assertFalse(report["valid"])


def _unit_rows(scores):
    """Rows on the unit circle whose dot with [1, 0] equals the given score."""
    return torch.tensor([[score, (1 - score ** 2) ** 0.5] for score in scores])


class ObjectiveTests(unittest.TestCase):
    def setUp(self):
        self.query = torch.tensor([[1.0, 0.0]])
        self.clean = _unit_rows([0.9, 0.8, 0.7])
        self.poison = _unit_rows([0.75, 0.5])

    def test_hit_at_k_is_not_the_full_takeover_loss(self):
        """The poison enters top-3 but does not beat the best clean key."""
        hit = compute_hit_at_k_margin_loss(
            self.query, self.clean, self.poison, top_k=3, margin=0.0,
        )
        takeover = compute_retrieval_margin_loss(
            self.query, self.clean, self.poison, 1, 0.0,
        )
        self.assertEqual(float(hit), 0.0)
        self.assertGreater(float(takeover), 0.0)

    def test_hit_at_k_is_positive_when_no_poison_reaches_top_k(self):
        loss = compute_hit_at_k_margin_loss(
            self.query, self.clean, self.poison, top_k=1, margin=0.0,
        )
        self.assertAlmostEqual(float(loss), 0.9 - 0.75, places=5)

    def test_hit_at_k_rejects_an_out_of_range_k(self):
        with self.assertRaises(ValueError):
            compute_hit_at_k_margin_loss(self.query, self.clean, self.poison, top_k=9)

    def test_hit_at_k_defaults_to_the_unnormalized_runtime_scorer(self):
        scaled_clean = self.clean * 3.0
        dot = compute_hit_at_k_margin_loss(
            self.query, scaled_clean, self.poison, top_k=3, margin=0.0,
        )
        cosine = compute_hit_at_k_margin_loss(
            self.query, scaled_clean, self.poison, top_k=3, margin=0.0, score="cosine",
        )
        # Scaling the clean keys changes dot-product ranking but not cosine;
        # a silent normalization would hide exactly this failure.
        self.assertGreater(float(dot), 0.0)
        self.assertEqual(float(cosine), 0.0)

    def test_total_loss_defaults_to_plain_agentpoison(self):
        l_uni, l_cpt, l_ret = torch.tensor(-2.0), torch.tensor(3.0), torch.tensor(5.0)
        self.assertAlmostEqual(float(mcat_total_loss(l_uni, l_cpt, l_ret)), -2.0 + 0.3)
        self.assertAlmostEqual(
            float(mcat_total_loss(l_uni, l_cpt, l_ret, lambda_ret=0.5)), -2.0 + 0.3 + 2.5
        )

    def test_fixed_policy_detaches_the_poison_side(self):
        poison = torch.randn(3, 4, requires_grad=True)
        self.assertTrue(PoisonPolicy("refresh").apply(poison).requires_grad)
        self.assertFalse(PoisonPolicy("fixed").apply(poison).requires_grad)
        with self.assertRaises(ValueError):
            PoisonPolicy("sometimes")


if __name__ == "__main__":
    unittest.main()
