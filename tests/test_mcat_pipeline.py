"""Generator, retrieval-metric and end-to-end stage checks for MCAT.

``tests/test_mcat.py`` covers the low-level contracts (episodes, encoding,
relaxation, objectives).  This file covers what sits on top of them: the
conditioned generator's invariances, the metrics the report quotes, and the
stage pipeline with its resume guards.  All of it runs on CPU against the
fixture retriever.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import torch

from src.triggers.mcat.cache import (
    PREBUILT_QA_VECTORS, load_prebuilt_qa_vectors, select_rows,
)
from src.triggers.mcat.cli import main
from src.triggers.mcat.evaluate import retrieval_metrics, shuffled_context_control
from src.triggers.mcat.generator import (
    SetEncoder, TriggerGenerator, parameter_count, sample_memory_keys,
)
from src.triggers.mcat.retrievers import build_fixture_retriever, fixture_text

MAX_LENGTH = 32


def unit_rows(scores):
    """Rows on the unit circle whose dot with [1, 0] equals the given score."""
    return torch.tensor([[score, (1 - score ** 2) ** 0.5] for score in scores])


class GeneratorTests(unittest.TestCase):
    def setUp(self):
        self.retriever = build_fixture_retriever(max_length=MAX_LENGTH)
        torch.manual_seed(0)
        self.memory = torch.randn(20, self.retriever.hidden_size)
        self.support = torch.randn(6, self.retriever.hidden_size)

    def _generator(self, variant="memory+query", **overrides):
        torch.manual_seed(0)
        return TriggerGenerator(
            embedding_dim=self.retriever.hidden_size,
            vocab_size=self.retriever.vocab_size,
            trigger_tokens=4, context_dim=8, hidden=16, variant=variant, **overrides,
        )

    def test_output_shape_is_trigger_tokens_by_vocabulary(self):
        logits = self._generator()(self.memory, self.support)
        self.assertEqual(tuple(logits.shape), (4, self.retriever.vocab_size))

    def test_pooling_is_permutation_invariant(self):
        generator = self._generator()
        order = torch.randperm(self.memory.shape[0])
        torch.testing.assert_close(
            generator(self.memory, self.support),
            generator(self.memory[order], self.support), rtol=1e-5, atol=1e-6,
        )

    def test_attention_pooling_is_permutation_invariant_too(self):
        generator = self._generator(attention_pool=True)
        order = torch.randperm(self.memory.shape[0])
        torch.testing.assert_close(
            generator(self.memory, self.support),
            generator(self.memory[order], self.support), rtol=1e-5, atol=1e-6,
        )

    def test_memory_changes_the_output_of_the_conditioned_variant(self):
        generator = self._generator()
        self.assertFalse(torch.allclose(
            generator(self.memory, self.support),
            generator(torch.randn_like(self.memory), self.support),
        ))

    def test_query_only_variant_ignores_memory(self):
        generator = self._generator("query")
        torch.testing.assert_close(
            generator(self.memory, self.support),
            generator(torch.randn_like(self.memory), self.support),
        )

    def test_memory_only_variant_ignores_queries(self):
        generator = self._generator("memory")
        torch.testing.assert_close(
            generator(self.memory, self.support),
            generator(self.memory, torch.randn_like(self.support)),
        )

    def test_unconditional_variant_ignores_both(self):
        generator = self._generator("none")
        torch.testing.assert_close(
            generator(self.memory, self.support),
            generator(torch.randn_like(self.memory), torch.randn_like(self.support)),
        )

    def test_every_variant_has_the_same_capacity(self):
        # If a dropped branch also dropped parameters, B4/B5/B6 would confound
        # conditioning with model size and the ablation would prove nothing.
        counts = {
            variant: parameter_count(self._generator(variant))
            for variant in ("memory+query", "query", "memory", "none")
        }
        self.assertEqual(len(set(counts.values())), 1, counts)

    def test_missing_required_context_is_an_error(self):
        with self.assertRaises(ValueError):
            self._generator("memory+query")(None, self.support)
        with self.assertRaises(ValueError):
            TriggerGenerator(embedding_dim=4, vocab_size=8, trigger_tokens=2,
                             variant="nonsense")

    def test_memory_key_sampling_is_bounded(self):
        generator = torch.Generator().manual_seed(3)
        sampled = sample_memory_keys(self.memory, 5, generator=generator)
        self.assertEqual(sampled.shape[0], 5)
        self.assertIs(sample_memory_keys(self.memory, 999), self.memory)

    def test_set_encoder_rejects_a_non_set_input(self):
        with self.assertRaises(ValueError):
            SetEncoder(4, 8, 4)(torch.randn(2, 3, 4))


class RetrievalMetricTests(unittest.TestCase):
    def test_metrics_match_an_explicit_ranking(self):
        queries = torch.tensor([[1.0, 0.0]])
        clean = unit_rows([0.9, 0.8, 0.7, 0.6])
        poison = unit_rows([0.85, 0.75])
        metrics = retrieval_metrics(queries, clean, poison, top_k=3)
        # Ranking: 0.9 clean, 0.85 poison, 0.8 clean -> one of three slots.
        self.assertEqual(metrics["hit_at_3"], 1.0)
        self.assertAlmostEqual(metrics["poison_occupancy_at_3"], 1 / 3, places=5)
        self.assertAlmostEqual(metrics["mean_poison_rank"], 2.0, places=5)
        self.assertAlmostEqual(metrics["mrr"], 0.5, places=5)

    def test_a_poison_below_the_cutoff_is_not_a_hit(self):
        queries = torch.tensor([[1.0, 0.0]])
        metrics = retrieval_metrics(queries, unit_rows([0.9, 0.8, 0.7]),
                                    unit_rows([0.5]), top_k=3)
        self.assertEqual(metrics["hit_at_3"], 0.0)
        self.assertEqual(metrics["poison_occupancy_at_3"], 0.0)
        self.assertLess(metrics["mean_margin"], 0.0)

    def test_full_takeover_fills_every_slot(self):
        queries = torch.tensor([[1.0, 0.0]])
        metrics = retrieval_metrics(queries, unit_rows([0.5, 0.4]),
                                    unit_rows([0.95, 0.9]), top_k=2)
        self.assertEqual(metrics["poison_occupancy_at_2"], 1.0)
        self.assertEqual(metrics["mean_poison_rank"], 1.0)


class PipelineTests(unittest.TestCase):
    """Every stage on the fixture retriever, plus the resume guards."""

    def setUp(self):
        self.directory = Path(tempfile.mkdtemp(prefix="mcat-pipeline-"))
        self.addCleanup(shutil.rmtree, self.directory, True)

    def _argv(self, command, *extra):
        return [
            command, "--output-dir", str(self.directory), "--fixture",
            "--domain", "qa", "--corpus-limit", "300", "--per-split", "2",
            "--documents", "16", "--support", "3", "--optimization", "4",
            "--evaluation", "6", "--poison-count", "2", "--trigger-tokens", "3",
            "--top-k", "3", "--steps", "2", "--max-length", "32", *extra,
        ]

    def _run_through(self, *commands, extra=()):
        for command in commands:
            self.assertEqual(main(self._argv(command, *extra)), 0, command)

    def test_stages_produce_the_full_artifact_set(self):
        self._run_through("prepare-episodes", "index", "train", "evaluate", "report")
        for name in ("manifest.json", "episodes.jsonl", "checkpoint.pt", "metrics.jsonl",
                     "triggers.jsonl", "training.json", "evaluation.json",
                     "evaluation.jsonl", "REPORT.md"):
            self.assertTrue((self.directory / name).exists(), f"{name} is missing")

    def test_direct_logit_adapts_on_the_evaluated_split(self):
        extra = ("--mode", "direct-logit")
        self._run_through("prepare-episodes", "index", "train", "evaluate", extra=extra)
        # B2 cannot transfer, so evaluation must pay for its own optimization
        # and that cost has to leave a trace on disk.
        self.assertTrue((self.directory / "adapt-test/metrics.jsonl").exists())

    def test_training_refuses_to_resume_without_the_flag(self):
        self._run_through("prepare-episodes", "index", "train")
        with self.assertRaises(FileExistsError):
            main(self._argv("train"))

    def test_changing_the_episode_split_is_refused(self):
        self._run_through("prepare-episodes")
        with self.assertRaises(ValueError):
            main(self._argv("prepare-episodes", "--per-split", "3"))

    def test_evaluation_refuses_a_checkpoint_from_another_contract(self):
        self._run_through("prepare-episodes", "index", "train")
        with self.assertRaises(ValueError):
            main(self._argv("evaluate", "--lambda-ret", "0.5"))

    def test_evaluation_requires_a_trained_checkpoint(self):
        self._run_through("prepare-episodes", "index")
        with self.assertRaises(FileNotFoundError):
            main(self._argv("evaluate"))

    def test_locked_queries_never_reach_the_training_groups(self):
        self._run_through("prepare-episodes")
        rows = (self.directory / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertTrue(rows)
        for line in rows:
            episode = json.loads(line)
            locked = set(episode["eval_qids"])
            used = set(episode["support_qids"] + episode["optimization_qids"]
                       + episode["poison_source_qids"])
            self.assertFalse(locked & used, episode["episode_id"])

    def test_the_manifest_records_a_truncated_corpus(self):
        self._run_through("prepare-episodes")
        manifest = json.loads((self.directory / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["corpus_limit"], 300)
        self.assertTrue(manifest["fixture"])


class FixtureTests(unittest.TestCase):
    def test_fixture_text_stays_inside_the_fixture_vocabulary(self):
        retriever = build_fixture_retriever(max_length=MAX_LENGTH)
        for index in range(20):
            ids = retriever.tokenizer(fixture_text(index),
                                      add_special_tokens=False).input_ids
            self.assertNotIn(retriever.tokenizer.unk_token_id, ids,
                             f"fixture_text({index}) fell out of vocabulary")


class PrebuiltIndexTests(unittest.TestCase):
    """The shipped DPR index may only be reused when it genuinely matches."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="mcat-prebuilt-"))
        self.addCleanup(shutil.rmtree, self.root, True)
        self.directory = self.root / PREBUILT_QA_VECTORS
        self.directory.mkdir(parents=True)
        self.retriever = build_fixture_retriever(max_length=MAX_LENGTH)
        self.doc_ids = ["a-1", "b-1", "c-1"]
        # Stored in a different order than the domain emits, exactly like the
        # real index, and with one recognisable row per document.
        self.stored = ["c-1", "a-1", "b-1"]
        self.vectors = np.array([[3.0, 3.0], [1.0, 1.0], [2.0, 2.0]], dtype="float32")
        np.save(self.directory / "vectors.npy", self.vectors)

    def _manifest(self, **overrides):
        manifest = {
            "corpus_sha256": "hash", "encoder": self.retriever.name,
            "encoder_revision": None, "max_length": self.retriever.max_length,
            "normalization": "none", "paragraph_ids": self.stored,
        }
        manifest.update(overrides)
        (self.directory / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )

    def _load(self):
        return load_prebuilt_qa_vectors(self.root, self.retriever, "hash", self.doc_ids)

    def test_rows_are_permuted_into_the_requested_order(self):
        self._manifest()
        matrix, reason = self._load()
        self.assertTrue(reason["reused"], reason)
        self.assertTrue(reason["reordered"])
        # a-1 -> [1,1], b-1 -> [2,2], c-1 -> [3,3]; storing order would give
        # every document its neighbour's vector.
        np.testing.assert_allclose(
            matrix, np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype="float32")
        )

    def test_a_normalized_index_is_refused(self):
        self._manifest(normalization="l2")
        matrix, reason = self._load()
        self.assertIsNone(matrix)
        self.assertEqual(reason["reason"], "prebuilt index is normalized")

    def test_a_different_corpus_is_refused(self):
        self._manifest(corpus_sha256="other")
        matrix, reason = self._load()
        self.assertIsNone(matrix)
        self.assertIn("corpus_sha256", reason["mismatches"])

    def test_a_different_encoder_is_refused(self):
        self._manifest(encoder="some/other-encoder")
        matrix, reason = self._load()
        self.assertIsNone(matrix)
        self.assertIn("encoder", reason["mismatches"])

    def test_partial_coverage_is_refused(self):
        self._manifest(paragraph_ids=["a-1", "b-1"])
        matrix, reason = self._load()
        self.assertIsNone(matrix)
        self.assertEqual(reason["reason"], "paragraph_ids do not cover the corpus")

    def test_a_missing_index_is_reported_not_raised(self):
        matrix, reason = self._load()
        self.assertIsNone(matrix)
        self.assertEqual(reason["reason"], "no prebuilt index on disk")

    def test_the_repository_index_is_refused_because_it_is_l2_normalized(self):
        """Guards the real file: it is l2-normalized, so it must not be reused."""
        from src.triggers.artifacts import ROOT, read_json

        manifest_path = ROOT / PREBUILT_QA_VECTORS / "manifest.json"
        if not manifest_path.exists():
            self.skipTest("the prebuilt StrategyQA index is not vendored")
        self.assertEqual(read_json(manifest_path).get("normalization"), "l2")


class SelectRowsTests(unittest.TestCase):
    def test_rows_are_gathered_by_identifier(self):
        vectors = np.array([[0.0], [1.0], [2.0]], dtype="float32")
        gathered = select_rows(vectors, ["x", "y", "z"], ["z", "x"])
        self.assertEqual(gathered.flatten().tolist(), [2.0, 0.0])

    def test_an_unknown_identifier_is_an_error(self):
        vectors = np.array([[0.0]], dtype="float32")
        with self.assertRaises(KeyError):
            select_rows(vectors, ["x"], ["missing"])


class ShuffledContextControlTests(unittest.TestCase):
    """The control must swap snapshots within a domain, never across them."""

    def setUp(self):
        from src.triggers.mcat.train import TrainConfig

        self.retriever = build_fixture_retriever(max_length=MAX_LENGTH)
        self.config = TrainConfig(mode="generator", variant="memory+query")
        torch.manual_seed(0)
        self.generator = TriggerGenerator(
            embedding_dim=self.retriever.hidden_size,
            vocab_size=self.retriever.vocab_size,
            trigger_tokens=3, context_dim=8, hidden=16,
        )

    def _context(self, episode_id, domain):
        from src.triggers.mcat.episodes import Episode
        from src.triggers.mcat.runtime import EpisodeContext

        torch.manual_seed(abs(hash(episode_id)) % 1000)
        episode = Episode(
            episode_id=episode_id, domain=domain, split="test",
            snapshot_id=episode_id, doc_ids=["d"], support_qids=["s"],
            optimization_qids=["o"], eval_qids=["e"], poison_source_qids=["p"],
            budget_poison=1, trigger_tokens=3, retrieval_top_k=3,
        )
        return EpisodeContext(
            episode=episode,
            memory_vectors=torch.randn(12, self.retriever.hidden_size),
            support_vectors=torch.randn(4, self.retriever.hidden_size),
            reference_centers=torch.randn(3, self.retriever.hidden_size),
            optimization_texts=[fixture_text(1)], poison_texts=[fixture_text(2)],
        )

    def _run(self, contexts):
        return shuffled_context_control(
            [self.generator] * len(contexts), self.config, contexts, self.retriever
        )

    def test_swaps_never_cross_a_domain_boundary(self):
        contexts = [self._context("qa-0", "qa"), self._context("qa-1", "qa"),
                    self._context("ehr-0", "ehr"), self._context("ehr-1", "ehr")]
        result = self._run(contexts)
        self.assertTrue(result["applicable"])
        for row in result["rows"]:
            self.assertEqual(row["domain"], row["swapped_with"].split("-")[0])
        self.assertEqual(set(result["per_domain"]), {"qa", "ehr"})

    def test_a_domain_with_one_episode_is_skipped_not_swapped_across(self):
        contexts = [self._context("qa-0", "qa"), self._context("qa-1", "qa"),
                    self._context("ehr-0", "ehr")]
        result = self._run(contexts)
        self.assertEqual(result["skipped_domains"], ["ehr"])
        self.assertEqual(result["episodes"], 2)
        self.assertTrue(all(row["domain"] == "qa" for row in result["rows"]))

    def test_no_swappable_pair_reports_inapplicable(self):
        result = self._run([self._context("qa-0", "qa"), self._context("ehr-0", "ehr")])
        self.assertFalse(result["applicable"])
        self.assertIn("same domain", result["reason"].replace("one domain", "same domain"))


if __name__ == "__main__":
    unittest.main()
