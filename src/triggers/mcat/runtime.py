"""Glue between episodes, the vector cache and the frozen retriever.

``Workspace`` owns everything that is expensive and shared: the domain rows,
the clean vectors and the per-snapshot reference centers.  Each episode then
hands the training loop one ``EpisodeContext`` holding exactly the tensors and
texts one optimization step needs -- and nothing drawn from ``Q_eval``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from src.triggers.clustering import fit_centers
from src.triggers.artifacts import ROOT, stable_hash
from src.triggers.mcat.cache import encode_corpus, load_prebuilt_qa_vectors, select_rows
from src.triggers.mcat.costs import record
from src.triggers.mcat.domains import Domain, limit_rows, load_domain
from src.triggers.mcat.drift import Snapshot
from src.triggers.mcat.episodes import Episode, assignment_for_rows
from src.triggers.mcat.retrievers import Retriever, fixture_text


@dataclass
class EpisodeContext:
    """One episode at one snapshot. Q_eval texts are deliberately absent.

    ``snapshot_id`` defaults to the episode's own ``-s0`` state, so a run that
    never drifts behaves exactly as it did before M3.  Every metrics row carries
    it, because under drift "which episode" no longer identifies a measurement.
    """

    episode: Episode
    memory_vectors: torch.Tensor
    support_vectors: torch.Tensor
    reference_centers: torch.Tensor
    optimization_texts: list[str]
    poison_texts: list[str]
    snapshot_id: str = ""

    def __post_init__(self) -> None:
        if not self.snapshot_id:
            self.snapshot_id = self.episode.snapshot_id

    @property
    def clean_keys(self) -> torch.Tensor:
        """The keys a poison record has to out-rank: this snapshot's memory."""
        return self.memory_vectors


@dataclass
class Workspace:
    retriever: Retriever
    cache_dir: Path
    fixture: bool = False
    batch_size: int = 32
    memory_summary_keys: int = 128
    corpus_limit: int | None = None
    reuse_report: dict[str, Any] = field(default_factory=dict)
    _domains: dict[str, Domain] = field(default_factory=dict)
    _documents: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    _queries: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    _doc_vectors: dict[str, torch.Tensor] = field(default_factory=dict)
    _query_vectors: dict[str, torch.Tensor] = field(default_factory=dict)
    _centers: dict[str, torch.Tensor] = field(default_factory=dict)
    _assignments: dict[str, dict[str, str]] = field(default_factory=dict)

    def domain(self, name: str) -> Domain:
        if name not in self._domains:
            self._domains[name] = load_domain(name)
        return self._domains[name]

    def documents(self, name: str) -> list[dict[str, Any]]:
        if name not in self._documents:
            rows = limit_rows(self.domain(name).documents(), self.corpus_limit)
            if self.fixture:
                rows = [row | {"text": fixture_text(index)} for index, row in enumerate(rows)]
            self._documents[name] = rows
        return self._documents[name]

    def queries(self, name: str) -> list[dict[str, Any]]:
        if name not in self._queries:
            rows = limit_rows(self.domain(name).queries(), self.corpus_limit)
            if self.fixture:
                rows = [row | {"question": fixture_text(index + 1)}
                        for index, row in enumerate(rows)]
            self._queries[name] = rows
        return self._queries[name]

    def _vectors(self, name: str, kind: str) -> torch.Tensor:
        store = self._doc_vectors if kind == "documents" else self._query_vectors
        if name in store:
            return store[name]
        rows = self.documents(name) if kind == "documents" else self.queries(name)
        texts = [row["text"] if kind == "documents" else row["question"] for row in rows]
        matrix = None
        if kind == "documents" and name == "qa" and not self.fixture and not self.corpus_limit:
            fingerprint = self.domain(name).fingerprint()
            matrix, reason = load_prebuilt_qa_vectors(
                ROOT, self.retriever, fingerprint.get("corpus_sha256", ""),
                [row["doc_id"] for row in rows],
            )
            self.reuse_report[name] = reason
        elif kind == "documents" and name == "qa":
            self.reuse_report[name] = {
                "reused": False,
                "reason": "fixture retriever" if self.fixture else "corpus limit is set",
            }
        if matrix is None:
            matrix = encode_corpus(
                self.retriever, texts, directory=self.cache_dir / name,
                kind=kind, batch_size=self.batch_size,
            )
        tensor = torch.from_numpy(matrix[:].astype("float32"))
        store[name] = tensor
        return tensor

    def document_vectors(self, name: str) -> torch.Tensor:
        return self._vectors(name, "documents")

    def query_vectors(self, name: str) -> torch.Tensor:
        return self._vectors(name, "queries")

    def assignment(
        self, name: str, *, ratios: tuple[float, float, float], seed: int
    ) -> dict[str, str]:
        """The family -> split map for a domain, cached per (ratios, seed).

        A drift trajectory has to know which documents its split owns, and it
        must derive that the same way ``build_episodes`` did.  Recomputing it
        here from the cached rows costs nothing and keeps one definition.
        """
        key = stable_hash([name, list(ratios), seed])
        if key not in self._assignments:
            self._assignments[key] = assignment_for_rows(
                self.documents(name), self.queries(name), tuple(ratios), seed
            )
        return self._assignments[key]

    def reference_centers(self, snapshot_id: str, memory: torch.Tensor) -> torch.Tensor:
        """GMM means over this snapshot's memory, cached per snapshot id.

        Five full-covariance components with ``random_state=0``, matching
        ``src.triggers.clustering.fit_centers`` so the uniqueness term is the upstream
        one rather than a lookalike.

        Keyed on the snapshot, not the episode: two snapshots of one episode hold
        different memory and therefore different centers.  Sharing them would let
        a drifted run optimize against the geometry of the state before the drift.
        """
        if snapshot_id not in self._centers:
            # A cache hit is not a refit; only the miss costs anything, and
            # under drift each new snapshot pays for exactly one.
            record("gmm_refits", 1)
            count = min(5, memory.shape[0])
            if self.fixture:
                # The fixture exists to exercise plumbing on a machine without
                # scikit-learn; it is NOT the upstream geometry, so it may only
                # ever be used for smoke runs.  Real runs take fit_centers.
                centers = memory[:count].detach().float().cpu()
            else:
                centers = fit_centers(memory, count=count, seed=0)
            self._centers[snapshot_id] = centers
        return self._centers[snapshot_id]

    def context_for(
        self, episode: Episode, snapshot: Snapshot | None = None
    ) -> EpisodeContext:
        """Materialize ``episode`` at ``snapshot`` (its own ``-s0`` when omitted).

        Only the memory moves.  ``Q_sup``, ``Q_opt``, the poison sources and
        ``Q_eval`` are properties of the episode and stay fixed across a
        trajectory: drifting the query distribution at the same time would make
        it impossible to attribute any change to memory state.
        """
        if snapshot is not None and snapshot.episode_id != episode.episode_id:
            raise ValueError(
                f"snapshot {snapshot.snapshot_id!r} belongs to episode "
                f"{snapshot.episode_id!r}, not {episode.episode_id!r}"
            )
        doc_ids = snapshot.doc_ids if snapshot is not None else episode.doc_ids
        snapshot_id = snapshot.snapshot_id if snapshot is not None else episode.snapshot_id

        documents, queries = self.documents(episode.domain), self.queries(episode.domain)
        doc_order = [row["doc_id"] for row in documents]
        query_order = [row["qid"] for row in queries]
        text_of = {row["qid"]: row["question"] for row in queries}

        memory = select_rows(self.document_vectors(episode.domain), doc_order, doc_ids)
        support = select_rows(
            self.query_vectors(episode.domain), query_order, episode.support_qids
        )
        device = self.retriever.device
        return EpisodeContext(
            episode=episode,
            memory_vectors=memory.to(device),
            support_vectors=support.to(device),
            reference_centers=self.reference_centers(snapshot_id, memory).to(device),
            optimization_texts=[text_of[qid] for qid in episode.optimization_qids],
            poison_texts=[text_of[qid] for qid in episode.poison_source_qids],
            snapshot_id=snapshot_id,
        )

    def eval_texts(self, episode: Episode) -> list[str]:
        """Q_eval, available only to the evaluator after the trigger is frozen."""
        text_of = {row["qid"]: row["question"] for row in self.queries(episode.domain)}
        return [text_of[qid] for qid in episode.eval_qids]
