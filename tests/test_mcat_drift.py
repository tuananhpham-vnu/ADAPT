"""Memory-drift contracts: trajectories, snapshot-aware runtime, frozen poison.

These are the M3 checks that have to hold before any adaptation method is even
written, because each of them is a way to produce numbers that look like a
result and are not one:

* a snapshot that quietly borrows documents from another split;
* drift that was chosen for its effect rather than by seed;
* a ``fixed`` poison policy that re-encodes its records anyway and so inherits
  the ``refresh`` arm's advantage.

Everything runs on CPU against the fixture retriever.
"""
from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import torch

from src.triggers.mcat.drift import (
    DEFAULT_GROWTH, Snapshot, assert_no_snapshot_leak, base_snapshot, build_trajectory,
    support_similarity, trajectory_hash,
)
from src.triggers.mcat.episodes import Episode, assignment_for_rows
from src.triggers.mcat.evaluate import _poison_keys
from src.triggers.mcat.poison import (
    FROZEN_POISON_FILE, FrozenPoison, encode_poison_keys, freeze_poison,
)
from src.triggers.mcat.retrievers import build_fixture_retriever, fixture_text
from src.triggers.mcat.runtime import EpisodeContext, Workspace

MAX_LENGTH = 32


def make_episode(doc_ids, *, episode_id="qa-test-000", split="test", domain="qa"):
    return Episode(
        episode_id=episode_id, domain=domain, split=split,
        snapshot_id=f"{episode_id}-s0", doc_ids=sorted(doc_ids),
        support_qids=["q-sup-0", "q-sup-1"], optimization_qids=["q-opt-0"],
        eval_qids=["q-eval-0"], poison_source_qids=["q-poison-0", "q-poison-1"],
        budget_poison=2, trigger_tokens=3, retrieval_top_k=3,
    )


def rows(prefix, count, *, family=None, start=0):
    return [{"doc_id": f"{prefix}-{index:03d}", "text": fixture_text(index),
             "family": family or f"{prefix}-fam-{index:03d}"}
            for index in range(start, start + count)]


class TrajectoryTests(unittest.TestCase):
    """``build_trajectory``: split safety, determinism and the shape of drift."""

    def setUp(self):
        self.base_rows = rows("base", 8)
        self.pool_rows = rows("pool", 12)
        self.episode = make_episode([row["doc_id"] for row in self.base_rows])
        self.assignment = {row["family"]: "test"
                           for row in self.base_rows + self.pool_rows}

    def _build(self, **overrides):
        options = {"assignment": self.assignment, "seed": 0, "ablations": ()}
        options.update(overrides)
        return build_trajectory(self.episode, self.pool_rows, **options)

    def _by_id(self, snapshots):
        return {snapshot.snapshot_id: snapshot for snapshot in snapshots}

    def test_the_first_snapshot_is_the_episode_itself(self):
        snapshots, _ = self._build()
        self.assertEqual(snapshots[0].snapshot_id, self.episode.snapshot_id)
        self.assertEqual(snapshots[0].kind, "base")
        self.assertEqual(snapshots[0].doc_ids, sorted(self.episode.doc_ids))
        self.assertIsNone(snapshots[0].parent_snapshot_id)

    def test_growth_adds_the_requested_fraction_and_keeps_the_old_memory(self):
        snapshots, _ = self._build()
        by_id = self._by_id(snapshots)
        for level in DEFAULT_GROWTH:
            snapshot = by_id[f"{self.episode.episode_id}-growth-{int(level * 100)}"]
            self.assertEqual(len(snapshot.doc_ids),
                             len(self.episode.doc_ids) + round(level * 8))
            # Drift is growth, not replacement: nothing may disappear.
            self.assertTrue(set(self.episode.doc_ids) <= set(snapshot.doc_ids))
            self.assertEqual(snapshot.growth, level)
            self.assertEqual(snapshot.parent_snapshot_id, self.episode.snapshot_id)

    def test_a_document_from_another_split_is_refused(self):
        stranger = rows("train", 1, start=99)
        assignment = dict(self.assignment, **{stranger[0]["family"]: "train"})
        with self.assertRaises(ValueError) as caught:
            build_trajectory(self.episode, self.pool_rows + stranger,
                             assignment=assignment, seed=0, ablations=())
        self.assertIn("may not reach across", str(caught.exception))

    def test_a_family_missing_from_the_assignment_is_refused(self):
        # A pool built from different rows than the split map is a silent way to
        # smuggle documents in, so the unknown family has to raise.
        orphan = rows("orphan", 1)
        with self.assertRaises(ValueError) as caught:
            build_trajectory(self.episode, self.pool_rows + orphan,
                             assignment=self.assignment, seed=0, ablations=())
        self.assertIn("absent from", str(caught.exception))

    def test_the_base_snapshot_helper_matches_the_first_snapshot(self):
        snapshots, _ = self._build()
        self.assertEqual(base_snapshot(self.episode).to_json(), snapshots[0].to_json())

    def test_selection_is_seeded_not_arbitrary(self):
        first, _ = self._build(seed=0)
        again, _ = self._build(seed=0)
        other, _ = self._build(seed=1)
        self.assertEqual([s.doc_ids for s in first], [s.doc_ids for s in again])
        self.assertNotEqual([s.doc_ids for s in first], [s.doc_ids for s in other])

    def test_deletion_removes_a_quarter_and_stays_a_subset(self):
        snapshots, _ = self._build(growth=(), ablations=("deletion",))
        snapshot = self._by_id(snapshots)[f"{self.episode.episode_id}-deletion"]
        self.assertTrue(set(snapshot.doc_ids) < set(self.episode.doc_ids))
        self.assertEqual(len(snapshot.doc_ids), 8 - 2)

    def test_mixture_needs_a_second_domain_and_says_so_when_it_is_absent(self):
        _, report = self._build(growth=(), ablations=("mixture",))
        self.assertIn("mixture", report["skipped"])
        self.assertIn("another domain", report["skipped"]["mixture"]["reason"])

    def test_mixture_swaps_in_foreign_documents(self):
        foreign = rows("ehr", 6)
        assignment = dict(self.assignment, **{row["family"]: "test" for row in foreign})
        snapshots, report = build_trajectory(
            self.episode, self.pool_rows, assignment=assignment, mixture_pool=foreign,
            growth=(), ablations=("mixture",), seed=0,
        )
        snapshot = self._by_id(snapshots)[f"{self.episode.episode_id}-mixture"]
        self.assertEqual(report["skipped"], {})
        added = set(snapshot.doc_ids) - set(self.episode.doc_ids)
        self.assertTrue(added <= {row["doc_id"] for row in foreign})
        self.assertTrue(set(self.episode.doc_ids) - set(snapshot.doc_ids))

    def test_distractor_takes_the_documents_closest_to_the_support_queries(self):
        scores = {row["doc_id"]: float(index) for index, row in enumerate(self.pool_rows)}
        snapshots, _ = self._build(growth=(), ablations=("distractor",),
                                   distractor_scores=scores)
        snapshot = self._by_id(snapshots)[f"{self.episode.episode_id}-distractor"]
        added = sorted(set(snapshot.doc_ids) - set(self.episode.doc_ids))
        best = sorted(scores, key=lambda key: -scores[key])[:len(added)]
        self.assertEqual(added, sorted(best))
        self.assertEqual(snapshot.note["selected_on"], "support_queries")

    def test_distractor_without_similarities_is_reported_not_invented(self):
        _, report = self._build(growth=(), ablations=("distractor",))
        self.assertIn("distractor", report["skipped"])

    def test_an_empty_pool_is_reported_per_level(self):
        _, report = build_trajectory(self.episode, [], assignment=self.assignment,
                                     seed=0, ablations=())
        self.assertEqual(sorted(report["skipped"]), ["growth-100", "growth-25", "growth-50"])

    def test_invalid_configuration_is_refused(self):
        with self.assertRaises(ValueError):
            self._build(growth=(0.0,))
        with self.assertRaises(ValueError):
            self._build(ablations=("nonsense",))

    def test_the_trajectory_hash_follows_the_drift_levels(self):
        first, _ = self._build(growth=(0.25,))
        other, _ = self._build(growth=(0.50,))
        self.assertNotEqual(trajectory_hash(first), trajectory_hash(other))

    def test_a_snapshot_needs_documents_and_no_duplicates(self):
        with self.assertRaises(ValueError):
            Snapshot("s", "e", "qa", "test", "growth", 0.25, [], None, {})
        with self.assertRaises(ValueError):
            Snapshot("s", "e", "qa", "test", "growth", 0.25, ["a", "a"], None, {})

    def test_the_audit_catches_a_doctored_trajectory(self):
        snapshots, _ = self._build()
        family_of = {row["doc_id"]: row["family"]
                     for row in self.base_rows + self.pool_rows}
        assert_no_snapshot_leak(snapshots, family_of, self.assignment)

        leaked = Snapshot("bad", self.episode.episode_id, "qa", "test", "growth", 0.25,
                          ["train-099"], self.episode.snapshot_id, {})
        with self.assertRaises(ValueError):
            assert_no_snapshot_leak([leaked], {"train-099": "train-fam"},
                                    {"train-fam": "train"})

    def test_support_similarity_ranks_by_the_support_centroid(self):
        vectors = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0]], dtype="float32")
        support = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        scores = support_similarity(vectors, ["a", "b", "c"], support, ["a", "b", "c"])
        self.assertEqual(max(scores, key=scores.get), "c")
        self.assertEqual(support_similarity(vectors, ["a"], support, []), {})

    def test_the_assignment_helper_matches_the_episode_builder(self):
        assignment = assignment_for_rows(
            self.base_rows, [{"family": "q-fam"}], (0.6, 0.2, 0.2), 0
        )
        self.assertEqual(set(assignment.values()) - {"train", "validation", "test"}, set())


class SnapshotRuntimeTests(unittest.TestCase):
    """``Workspace`` must resolve a snapshot without disturbing the queries."""

    def setUp(self):
        self.retriever = build_fixture_retriever(max_length=MAX_LENGTH)
        self.documents = rows("base", 6) + rows("pool", 6)
        self.queries = [
            {"qid": qid, "question": fixture_text(index), "family": f"q-{index}"}
            for index, qid in enumerate(
                ["q-sup-0", "q-sup-1", "q-opt-0", "q-eval-0", "q-poison-0", "q-poison-1"]
            )
        ]
        torch.manual_seed(0)
        self.workspace = Workspace(retriever=self.retriever, cache_dir=Path("."),
                                   fixture=True)
        # Prefill the caches so the test never touches a corpus on disk.
        self.workspace._documents["qa"] = self.documents
        self.workspace._queries["qa"] = self.queries
        self.workspace._doc_vectors["qa"] = torch.randn(
            len(self.documents), self.retriever.hidden_size)
        self.workspace._query_vectors["qa"] = torch.randn(
            len(self.queries), self.retriever.hidden_size)
        self.episode = make_episode([row["doc_id"] for row in self.documents[:6]])
        self.snapshot = Snapshot(
            snapshot_id=f"{self.episode.episode_id}-growth-50",
            episode_id=self.episode.episode_id, domain="qa", split="test", kind="growth",
            growth=0.5, doc_ids=sorted(self.episode.doc_ids
                                       + [row["doc_id"] for row in self.documents[6:9]]),
            parent_snapshot_id=self.episode.snapshot_id, note={},
        )

    def test_the_default_context_is_the_base_snapshot(self):
        context = self.workspace.context_for(self.episode)
        self.assertEqual(context.snapshot_id, self.episode.snapshot_id)
        self.assertEqual(context.memory_vectors.shape[0], len(self.episode.doc_ids))

    def test_a_snapshot_changes_the_memory_and_nothing_else(self):
        base = self.workspace.context_for(self.episode)
        drifted = self.workspace.context_for(self.episode, self.snapshot)
        self.assertEqual(drifted.snapshot_id, self.snapshot.snapshot_id)
        self.assertEqual(drifted.memory_vectors.shape[0], len(self.snapshot.doc_ids))
        self.assertEqual(base.optimization_texts, drifted.optimization_texts)
        self.assertEqual(base.poison_texts, drifted.poison_texts)
        torch.testing.assert_close(base.support_vectors, drifted.support_vectors)

    def test_reference_centers_are_keyed_on_the_snapshot_not_the_episode(self):
        # Replacement, not growth: the fixture stands in for the GMM by taking
        # the first rows, so the snapshot has to actually move them for this to
        # distinguish a per-snapshot cache from a per-episode one.
        replaced = Snapshot(
            snapshot_id=f"{self.episode.episode_id}-mixture",
            episode_id=self.episode.episode_id, domain="qa", split="test",
            kind="mixture", growth=0.0,
            doc_ids=sorted(row["doc_id"] for row in self.documents[6:]),
            parent_snapshot_id=self.episode.snapshot_id, note={},
        )
        base = self.workspace.context_for(self.episode)
        drifted = self.workspace.context_for(self.episode, replaced)
        # Sharing centers would let the drifted run optimize against the
        # geometry of the memory it no longer has.
        self.assertFalse(torch.allclose(base.reference_centers, drifted.reference_centers))
        self.assertEqual(
            sorted(self.workspace._centers),
            sorted([self.episode.snapshot_id, replaced.snapshot_id]),
        )

    def test_a_snapshot_from_another_episode_is_refused(self):
        stranger = Snapshot("x", "qa-test-999", "qa", "test", "growth", 0.25,
                            ["base-000"], None, {})
        with self.assertRaises(ValueError):
            self.workspace.context_for(self.episode, stranger)

    def test_the_split_assignment_is_cached_per_seed(self):
        first = self.workspace.assignment("qa", ratios=(0.6, 0.2, 0.2), seed=0)
        again = self.workspace.assignment("qa", ratios=(0.6, 0.2, 0.2), seed=0)
        self.assertIs(first, again)
        self.assertIsNot(first, self.workspace.assignment("qa", ratios=(0.6, 0.2, 0.2),
                                                          seed=1))


class FrozenPoisonTests(unittest.TestCase):
    """A ``fixed`` write policy must survive drift without being re-encoded."""

    def setUp(self):
        self.retriever = build_fixture_retriever(max_length=MAX_LENGTH)
        self.directory = Path(tempfile.mkdtemp(prefix="mcat-poison-"))
        self.addCleanup(shutil.rmtree, self.directory, True)
        self.episode = make_episode(["base-000", "base-001"])
        torch.manual_seed(0)
        self.context = EpisodeContext(
            episode=self.episode,
            memory_vectors=torch.randn(4, self.retriever.hidden_size),
            support_vectors=torch.randn(2, self.retriever.hidden_size),
            reference_centers=torch.randn(2, self.retriever.hidden_size),
            optimization_texts=[fixture_text(1)],
            poison_texts=[fixture_text(2), fixture_text(3)],
        )
        self.trigger = self._trigger([6, 7, 8])
        self.other_trigger = self._trigger([9, 10, 11])

    def _trigger(self, ids):
        return {"trigger": " ".join(f"w{i}" for i in ids), "token_ids": ids,
                "round_trip": {"re_encoded_ids": ids, "valid": True}}

    def _drifted(self):
        drifted = EpisodeContext(
            episode=self.episode,
            memory_vectors=torch.randn(9, self.retriever.hidden_size),
            support_vectors=self.context.support_vectors,
            reference_centers=self.context.reference_centers,
            optimization_texts=self.context.optimization_texts,
            poison_texts=self.context.poison_texts,
            snapshot_id=f"{self.episode.episode_id}-growth-50",
        )
        return drifted

    def test_freezing_writes_the_artifact_and_pins_the_encoder(self):
        frozen = freeze_poison(self.retriever, [self.context], [self.trigger],
                               output_dir=self.directory)
        self.assertTrue((self.directory / FROZEN_POISON_FILE).exists())
        self.assertEqual(frozen.retriever, self.retriever.fingerprint())
        self.assertEqual(frozen.triggers[self.episode.episode_id], [6, 7, 8])
        self.assertEqual(frozen.snapshot_ids[self.episode.episode_id],
                         self.episode.snapshot_id)

    def test_fixed_keys_are_identical_after_drift_while_refresh_moves(self):
        frozen = freeze_poison(self.retriever, [self.context], [self.trigger],
                               output_dir=self.directory)
        drifted = self._drifted()
        ids = torch.tensor(self.other_trigger["token_ids"], device=self.retriever.device)

        fixed = _poison_keys(drifted, ids, self.retriever,
                             frozen=frozen.keys_for(self.episode.episode_id))
        refreshed = _poison_keys(drifted, ids, self.retriever)
        # Bit-identical, not merely close: the fixed arm ranks the very vectors
        # that were written at s0.
        self.assertTrue(torch.equal(
            fixed.cpu(), frozen.keys_for(self.episode.episode_id).cpu()))
        self.assertFalse(torch.allclose(fixed.cpu(), refreshed.cpu()))

    def test_refreshing_with_the_same_trigger_reproduces_the_frozen_keys(self):
        # Guards the freeze path itself: if these differed, "fixed" would be
        # measuring an encoding artifact rather than the write policy.
        keys = encode_poison_keys(
            self.context, torch.tensor(self.trigger["token_ids"]), self.retriever)
        frozen = freeze_poison(self.retriever, [self.context], [self.trigger],
                               output_dir=self.directory)
        torch.testing.assert_close(
            keys.cpu(), frozen.keys_for(self.episode.episode_id), rtol=0, atol=0)

    def test_a_record_count_mismatch_is_refused(self):
        frozen = freeze_poison(self.retriever, [self.context], [self.trigger],
                               output_dir=self.directory)
        ids = torch.tensor(self.trigger["token_ids"], device=self.retriever.device)
        with self.assertRaises(ValueError):
            _poison_keys(self.context, ids, self.retriever,
                         frozen=frozen.keys_for(self.episode.episode_id)[:1])

    def test_poison_may_only_be_frozen_at_the_base_snapshot(self):
        with self.assertRaises(ValueError) as caught:
            freeze_poison(self.retriever, [self._drifted()], [self.trigger],
                          output_dir=self.directory)
        self.assertIn("base snapshot", str(caught.exception))

    def test_an_unknown_episode_names_the_missing_freeze(self):
        frozen = freeze_poison(self.retriever, [self.context], [self.trigger],
                               output_dir=self.directory)
        with self.assertRaises(KeyError):
            frozen.keys_for("qa-test-404")

    def test_loading_refuses_keys_from_another_encoder(self):
        freeze_poison(self.retriever, [self.context], [self.trigger],
                      output_dir=self.directory)
        path = self.directory / FROZEN_POISON_FILE
        FrozenPoison.load(path, retriever=self.retriever)
        other = build_fixture_retriever(max_length=MAX_LENGTH, hidden=8, seed=1)
        with self.assertRaises(ValueError) as caught:
            FrozenPoison.load(path, retriever=other)
        self.assertIn("another encoder", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
