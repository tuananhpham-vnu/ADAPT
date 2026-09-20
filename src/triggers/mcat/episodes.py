"""Episodes, snapshots and the family-level outer split.

One episode is the unit the generator is conditioned on and evaluated over:

    E = (D, Q_sup, Q_opt, Q_eval, P, B, L)

``Q_sup`` builds the conditioning context, ``Q_opt`` carries the training loss
and ``Q_eval`` is locked until the trigger is frozen.  Episodes never straddle
the outer split because families -- not individual rows -- are partitioned
first, so paraphrases of one base task stay on one side.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Any, Iterable

from src.triggers.artifacts import stable_hash
from src.triggers.mcat.domains import Domain, limit_rows, load_domain

SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class EpisodeSizes:
    """Pilot defaults from the MCAT plan; shrunk per domain when data is thin."""

    documents: int = 512
    support: int = 32
    optimization: int = 64
    evaluation: int = 128
    poison_sources: int = 5
    trigger_tokens: int = 10
    retrieval_top_k: int = 5

    @property
    def queries_needed(self) -> int:
        return self.support + self.optimization + self.evaluation + self.poison_sources


@dataclass(frozen=True)
class Episode:
    episode_id: str
    domain: str
    split: str
    snapshot_id: str
    doc_ids: list[str]
    support_qids: list[str]
    optimization_qids: list[str]
    eval_qids: list[str]
    poison_source_qids: list[str]
    budget_poison: int
    trigger_tokens: int
    retrieval_top_k: int

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    def assert_disjoint(self) -> None:
        """Q_sup, Q_opt, Q_eval and the poison sources must not overlap."""
        groups = {
            "support": set(self.support_qids),
            "optimization": set(self.optimization_qids),
            "eval": set(self.eval_qids),
            "poison": set(self.poison_source_qids),
        }
        names = list(groups)
        for index, left in enumerate(names):
            for right in names[index + 1:]:
                shared = groups[left] & groups[right]
                if shared:
                    raise ValueError(
                        f"{self.episode_id}: {left} and {right} share "
                        f"{len(shared)} qids, e.g. {sorted(shared)[:3]}"
                    )


def family_key(value: str) -> str:
    return " ".join(str(value).casefold().split())


def assign_families(
    families: Iterable[str], ratios: tuple[float, float, float], seed: int
) -> dict[str, str]:
    """Map every family key to one split. Deterministic given ``seed``."""
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1, got {ratios}")
    keys = sorted({family_key(value) for value in families})
    random.Random(seed).shuffle(keys)
    train_end = int(len(keys) * ratios[0])
    validation_end = train_end + int(len(keys) * ratios[1])
    groups = {"train": keys[:train_end], "validation": keys[train_end:validation_end],
              "test": keys[validation_end:]}
    return {key: split for split, members in groups.items() for key in members}


def scale_sizes(
    sizes: EpisodeSizes, documents: int, queries: int
) -> tuple[EpisodeSizes, dict[str, Any]]:
    """Shrink an episode to what a small domain can supply without reusing rows.

    Returns the scaled sizes plus a note recording every reduction, so a thin
    domain stays visible in the manifest instead of being silently padded.
    """
    notes: dict[str, Any] = {}
    scaled = sizes
    if documents < sizes.documents:
        notes["documents"] = {"requested": sizes.documents, "available": documents}
        scaled = EpisodeSizes(**{**asdict(scaled), "documents": max(1, documents)})
    if queries < sizes.queries_needed:
        factor = queries / sizes.queries_needed
        shrunk = {
            "support": max(2, int(sizes.support * factor)),
            "optimization": max(2, int(sizes.optimization * factor)),
            "evaluation": max(2, int(sizes.evaluation * factor)),
            "poison_sources": max(1, int(sizes.poison_sources * factor)),
        }
        notes["queries"] = {"requested": sizes.queries_needed, "available": queries,
                            "scaled_to": shrunk}
        scaled = EpisodeSizes(**{**asdict(scaled), **shrunk})
    return scaled, notes


def build_episodes(
    domain: Domain,
    *,
    seed: int,
    sizes: EpisodeSizes = EpisodeSizes(),
    per_split: int = 4,
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
    corpus_limit: int | None = None,
) -> tuple[list[Episode], dict[str, Any]]:
    documents = limit_rows(domain.documents(), corpus_limit)
    queries = limit_rows(domain.queries(), corpus_limit)
    families = [row["family"] for row in documents] + [row["family"] for row in queries]
    assignment = assign_families(families, ratios, seed)

    episodes: list[Episode] = []
    report: dict[str, Any] = {"scaling": {}, "counts": {}}
    for split in SPLIT_NAMES:
        split_docs = [row for row in documents if assignment[family_key(row["family"])] == split]
        split_queries = [row for row in queries if assignment[family_key(row["family"])] == split]
        scaled, notes = scale_sizes(sizes, len(split_docs), len(split_queries))
        if notes:
            report["scaling"][split] = notes
        report["counts"][split] = {"documents": len(split_docs), "queries": len(split_queries)}
        if not split_docs or len(split_queries) < scaled.queries_needed:
            raise ValueError(
                f"{domain.name}/{split}: only {len(split_docs)} documents and "
                f"{len(split_queries)} queries; the scaled episode needs "
                f"{scaled.queries_needed} queries"
            )
        for index in range(per_split):
            rng = random.Random(stable_hash([domain.name, split, index, seed])[:16])
            doc_sample = rng.sample(split_docs, min(scaled.documents, len(split_docs)))
            query_sample = rng.sample(split_queries, scaled.queries_needed)
            cursor = 0

            def take(count: int) -> list[str]:
                nonlocal cursor
                chunk = query_sample[cursor:cursor + count]
                cursor += count
                return [row["qid"] for row in chunk]

            episode = Episode(
                episode_id=f"{domain.name}-{split}-{index:03d}",
                domain=domain.name,
                split=split,
                snapshot_id=f"{domain.name}-{split}-{index:03d}-s0",
                doc_ids=sorted(row["doc_id"] for row in doc_sample),
                support_qids=take(scaled.support),
                optimization_qids=take(scaled.optimization),
                eval_qids=take(scaled.evaluation),
                poison_source_qids=take(scaled.poison_sources),
                budget_poison=scaled.poison_sources,
                trigger_tokens=scaled.trigger_tokens,
                retrieval_top_k=scaled.retrieval_top_k,
            )
            episode.assert_disjoint()
            episodes.append(episode)
    return episodes, report


def build_manifest(
    domain_names: Iterable[str],
    *,
    seed: int,
    sizes: EpisodeSizes = EpisodeSizes(),
    per_split: int = 4,
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
    corpus_limit: int | None = None,
) -> tuple[list[Episode], dict[str, Any]]:
    episodes: list[Episode] = []
    manifest: dict[str, Any] = {
        "schema_version": 1, "seed": seed, "ratios": list(ratios),
        "per_split": per_split, "sizes": asdict(sizes),
        "corpus_limit": corpus_limit, "domains": {},
    }
    for name in domain_names:
        domain = load_domain(name)
        domain_episodes, report = build_episodes(
            domain, seed=seed, sizes=sizes, per_split=per_split, ratios=ratios,
            corpus_limit=corpus_limit,
        )
        episodes.extend(domain_episodes)
        manifest["domains"][name] = domain.fingerprint() | report
    manifest["episode_ids"] = [episode.episode_id for episode in episodes]
    manifest["split_hash"] = stable_hash([episode.to_json() for episode in episodes])
    return episodes, manifest


def assert_no_family_leak(
    episodes: list[Episode], domain: Domain, *, corpus_limit: int | None = None
) -> None:
    """Fail if one family supplies rows to two different splits."""
    queries = limit_rows(domain.queries(), corpus_limit)
    documents = limit_rows(domain.documents(), corpus_limit)
    family_of = {row["qid"]: family_key(row["family"]) for row in queries}
    family_of |= {row["doc_id"]: family_key(row["family"]) for row in documents}
    seen: dict[str, str] = {}
    for episode in episodes:
        if episode.domain != domain.name:
            continue
        members = (episode.doc_ids + episode.support_qids + episode.optimization_qids
                   + episode.eval_qids + episode.poison_source_qids)
        for member in members:
            family = family_of[member]
            if seen.setdefault(family, episode.split) != episode.split:
                raise ValueError(
                    f"family {family!r} appears in both {seen[family]} and {episode.split}"
                )
