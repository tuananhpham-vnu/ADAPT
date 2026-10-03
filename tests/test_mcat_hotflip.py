"""B1 -- AgentPoison's HotFlip search as ``--mode hotflip``.

What has to hold for B1 to be a fair baseline next to B2/B3/M1: it draws from
the same vocabulary, minimizes the same objective, never accepts a flip that
does not lower that objective, and runs through every stage the other arms do.
All of it on CPU against the fixture retriever.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest

import torch

from src.triggers.mcat.cli import main
from src.triggers.mcat.episodes import Episode
from src.triggers.mcat.hotflip import HotFlipTrigger, hotflip_candidates, hotflip_step
from src.triggers.mcat.retrievers import build_fixture_retriever, fixture_text
from src.triggers.mcat.runtime import EpisodeContext
from src.triggers.mcat.train import (
    TrainConfig, _logits_source, build_contract, losses_for, train,
)
from src.triggers.artifacts import stable_hash

MAX_LENGTH = 32


def _context(retriever, episode_id="qa-0", seed=0):
    generator = torch.Generator().manual_seed(seed)
    hidden = retriever.hidden_size
    episode = Episode(
        episode_id=episode_id, domain="qa", split="train", snapshot_id=episode_id,
        doc_ids=["d"], support_qids=["s"], optimization_qids=["o"], eval_qids=["e"],
        poison_source_qids=["p"], budget_poison=2, trigger_tokens=3, retrieval_top_k=3,
    )
    return EpisodeContext(
        episode=episode,
        memory_vectors=torch.randn(12, hidden, generator=generator),
        support_vectors=torch.randn(4, hidden, generator=generator),
        reference_centers=torch.randn(3, hidden, generator=generator),
        optimization_texts=[fixture_text(seed + 1), fixture_text(seed + 5)],
        poison_texts=[fixture_text(seed + 2), fixture_text(seed + 9)],
    )


class CandidateTests(unittest.TestCase):
    def test_never_proposes_a_disallowed_or_the_current_token(self):
        torch.manual_seed(0)
        weight = torch.randn(10, 4)
        allowed = torch.tensor([False, True, True, True, False, True, True, True, True, True])
        picked = hotflip_candidates(torch.randn(4), weight, allowed, 20, exclude=3)
        self.assertEqual(len(picked), int(allowed.sum()) - 1)
        self.assertTrue(all(allowed[i] and i != 3 for i in picked.tolist()))

    def test_ranks_by_first_order_loss_decrease(self):
        weight = torch.eye(4)
        gradient = torch.tensor([0.5, -2.0, 1.0, -1.0])
        picked = hotflip_candidates(gradient, weight, torch.ones(4, dtype=torch.bool), 2)
        # -W @ g = [-0.5, 2.0, -1.0, 1.0]: token 1 lowers the loss most, then 3.
        self.assertEqual(picked.tolist(), [1, 3])

    def test_upstream_masking_bug_is_not_reproduced(self):
        # Upstream masks with +inf after negating; the excluded token would win.
        weight = torch.eye(3)
        gradient = torch.tensor([-100.0, 1.0, 1.0])
        allowed = torch.tensor([False, True, True])
        picked = hotflip_candidates(gradient, weight, allowed, 1)
        self.assertNotEqual(int(picked[0]), 0)


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.retriever = build_fixture_retriever(max_length=MAX_LENGTH)
        self.config = TrainConfig(mode="hotflip", steps=6, hotflip_candidates=8)
        self.context = _context(self.retriever)

    def test_initial_trigger_uses_only_allowed_tokens(self):
        torch.manual_seed(0)
        module = _logits_source(self.config, self.retriever, self.context.episode)
        self.assertIsInstance(module, HotFlipTrigger)
        self.assertTrue(bool(self.retriever.vocab_mask[module.token_ids].all()))
        self.assertEqual(list(module.parameters()), [])

    def test_a_step_never_raises_the_loss(self):
        torch.manual_seed(0)
        module = _logits_source(self.config, self.retriever, self.context.episode)
        for _ in range(5):
            before = module.token_ids.clone()
            result = hotflip_step(module, self.context, self.retriever, self.config,
                                  losses=losses_for)
            with torch.no_grad():
                after = float(losses_for(self.context, self.retriever.embedding_matrix[
                    module.token_ids], self.retriever, self.config)["l_total"])
            if result["accepted"]:
                self.assertLess(after, result["l_total"])
                self.assertEqual(int((before != module.token_ids).sum()), 1)
            else:
                self.assertTrue(torch.equal(before, module.token_ids))
                self.assertAlmostEqual(after, result["l_total"], places=5)

    def test_training_is_monotone_and_resumable(self):
        workspace = SimpleNamespace(retriever=self.retriever)
        contexts = [self.context, _context(self.retriever, "qa-1", seed=3)]
        episodes = [context.episode for context in contexts]
        with tempfile.TemporaryDirectory() as directory:
            summary, modules = train(workspace, episodes, self.config,
                                     output_dir=Path(directory), contract={"c": 1},
                                     contexts=contexts)
            self.assertIsNot(modules[0], modules[1])
            self.assertEqual(summary["triggers"], 2)

            history = torch.load(Path(directory) / "checkpoint.pt",
                                 weights_only=False)["history"]
            losses = [row["l_total"] for row in history]
            for earlier, later in zip(losses, losses[1:]):
                self.assertLessEqual(later, earlier + 1e-6)

            # Resuming a finished run is a no-op that keeps the searched ids.
            _, resumed = train(workspace, episodes, self.config,
                               output_dir=Path(directory), contract={"c": 1},
                               contexts=contexts, resume=True)
            for left, right in zip(modules, resumed):
                self.assertTrue(torch.equal(left.token_ids, right.token_ids))


class ContractTests(unittest.TestCase):
    def test_other_modes_keep_their_pre_hotflip_hash(self):
        retriever = build_fixture_retriever(max_length=MAX_LENGTH)
        manifest = {"split_hash": "x"}
        for mode in ("direct-logit", "universal-logit", "generator"):
            config = TrainConfig(mode=mode)
            legacy = asdict(config)
            legacy.pop("hotflip_candidates")
            self.assertEqual(build_contract(config, manifest, retriever)["config_hash"],
                             stable_hash(legacy))

    def test_candidate_count_is_part_of_a_hotflip_contract(self):
        retriever = build_fixture_retriever(max_length=MAX_LENGTH)
        manifest = {"split_hash": "x"}
        hashes = {build_contract(TrainConfig(mode="hotflip", hotflip_candidates=n),
                                 manifest, retriever)["config_hash"] for n in (10, 100)}
        self.assertEqual(len(hashes), 2)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp(prefix="mcat-hotflip-"))
        self.addCleanup(shutil.rmtree, self.directory, True)

    def _argv(self, command, *extra):
        return [
            command, "--output-dir", str(self.directory), "--fixture",
            "--domain", "qa", "--corpus-limit", "300", "--per-split", "2",
            "--documents", "16", "--support", "3", "--optimization", "4",
            "--evaluation", "6", "--poison-count", "2", "--trigger-tokens", "3",
            "--top-k", "3", "--steps", "2", "--max-length", "32",
            "--mode", "hotflip", "--hotflip-candidates", "5", *extra,
        ]

    def test_hotflip_runs_every_stage_and_searches_on_the_evaluated_split(self):
        for command in ("prepare-episodes", "index", "train", "evaluate", "report"):
            self.assertEqual(main(self._argv(command)), 0, command)
        # Like B2, B1 transfers nothing: evaluation pays for its own search.
        self.assertTrue((self.directory / "adapt-test/metrics.jsonl").exists())
        self.assertTrue((self.directory / "evaluation.json").exists())


if __name__ == "__main__":
    unittest.main()
