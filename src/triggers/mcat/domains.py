"""Per-agent-domain adapters that expose a memory corpus and a query pool.

An episode needs three things from a domain: documents that make up the benign
long-term memory, queries drawn from the clean distribution, and a ``family``
label per item.  ``family`` is the unit the outer split is taken on, so that
paraphrases and near-duplicates of one base task can never straddle
train/validation/test (see ``_idea/memory_conditioned_generator_Q1_A_star.md``
section 5).
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from src.triggers.artifacts import ROOT, read_json, sha256_file

QA_CORPUS = ROOT / "ReAct/database/strategyqa_train_paragraphs.json"
QA_QUERIES = ROOT / "ReAct/database/strategyqa_train_filtered.json"
EHR_CORPUS = ROOT / "EhrAgent/database/ehr_logs/logs_final"
EHR_QUERIES = ROOT / "EhrAgent/database/ehr_logs/eicu_ac.json"
AD_CORPUS = ROOT / "agentdriver/data/finetune/data_samples_train.json"

AD_MISSING = (
    "AgentDriver memory is not vendored in this repository: {path} is absent "
    "(agentdriver/data/ only carries split.json). Fetch data_samples_train.json "
    "from the upstream AgentDriver release into agentdriver/data/finetune/ "
    "before selecting --domain ad."
)


@dataclass(frozen=True)
class Domain:
    """A memory corpus plus the clean query pool retrieved against it."""

    name: str
    corpus_path: Path
    query_path: Path | None

    def documents(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    def queries(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    def fingerprint(self) -> dict[str, Any]:
        """Hashes that pin this domain's inputs into a run manifest."""
        result: dict[str, Any] = {"name": self.name, "corpus": str(self.corpus_path)}
        if self.corpus_path.is_file():
            result["corpus_sha256"] = sha256_file(self.corpus_path)
        elif self.corpus_path.is_dir():
            files = sorted(self.corpus_path.glob("*.txt"))
            result["corpus_sha256"] = _hash_of_many(files)
            result["corpus_files"] = len(files)
        if self.query_path is not None and self.query_path.is_file():
            result["query_sha256"] = sha256_file(self.query_path)
        return result


def limit_rows(rows: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    """Truncate a domain to its first ``limit`` rows.

    Episode building and the vector cache must apply the same limit or an
    episode will reference ids the cache never encoded, so both call this.
    The limit is recorded in the manifest: a truncated corpus is a different
    experiment, not a smaller version of the same one.
    """
    return rows if not limit else rows[:limit]


def _hash_of_many(paths: list[Path]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def _dedupe_by_text(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    """Drop repeats under the same normalization ``agentpoison_margin`` uses."""
    seen: set[str] = set()
    result = []
    for row in rows:
        normalized = " ".join(str(row[key]).casefold().split())
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(row)
    return result


class StrategyQADomain(Domain):
    """ReAct / StrategyQA: Wikipedia paragraphs keyed by ``<title>-<index>``."""

    def documents(self) -> list[dict[str, Any]]:
        raw = read_json(self.corpus_path)
        rows = [
            {"doc_id": doc_id, "text": value["content"],
             "family": value.get("title") or doc_id.rsplit("-", 1)[0]}
            for doc_id, value in raw.items()
        ]
        rows.sort(key=lambda row: row["doc_id"])
        return rows

    def queries(self) -> list[dict[str, Any]]:
        raw = read_json(self.query_path)
        rows = [
            {"qid": str(row["qid"]), "question": row["question"],
             "family": row.get("term") or str(row["qid"])}
            for row in raw
        ]
        rows.sort(key=lambda row: row["qid"])
        return _dedupe_by_text(rows, "question")


class EhrAgentDomain(Domain):
    """EhrAgent: solved-trajectory memory records, eICU questions as queries."""

    def documents(self) -> list[dict[str, Any]]:
        from algo.utils import load_ehr_memory

        # load_ehr_memory dedupes through a set of tuples, so its order varies
        # with PYTHONHASHSEED.  Sort before assigning ids to keep runs stable.
        records = sorted(load_ehr_memory(str(self.corpus_path)),
                         key=lambda item: item["question"].strip())
        return [
            {"doc_id": f"ehr-{index:04d}", "text": record["question"].strip(),
             "knowledge": record["knowledge"], "family": _ehr_family(record["question"])}
            for index, record in enumerate(records)
        ]

    def queries(self) -> list[dict[str, Any]]:
        raw = read_json(self.query_path)
        rows = [
            {"qid": f"eicu-{index:04d}", "question": row["question"],
             "family": row.get("q_tag") or row.get("template") or row["question"]}
            for index, row in enumerate(raw)
        ]
        return _dedupe_by_text(rows, "question")


def _ehr_family(question: str) -> str:
    """Coarse template key: the first few tokens of the question stem."""
    return " ".join(question.casefold().split()[:4])


class AgentDriverDomain(Domain):
    """AgentDriver: not vendored here; fails loudly with a fetch instruction."""

    def _require(self) -> None:
        if not self.corpus_path.exists():
            raise FileNotFoundError(AD_MISSING.format(path=self.corpus_path))

    def documents(self) -> list[dict[str, Any]]:
        self._require()
        raw = json.loads(self.corpus_path.read_text(encoding="utf-8"))
        rows = [
            {"doc_id": str(row.get("token", index)),
             "text": json.dumps(row.get("ego_states", row), ensure_ascii=False),
             "family": str(row.get("scene_token", row.get("token", index)))}
            for index, row in enumerate(raw)
        ]
        rows.sort(key=lambda row: row["doc_id"])
        return rows

    def queries(self) -> list[dict[str, Any]]:
        self._require()
        return [
            {"qid": row["doc_id"], "question": row["text"], "family": row["family"]}
            for row in self.documents()
        ]


_REGISTRY = {
    "qa": lambda: StrategyQADomain("qa", QA_CORPUS, QA_QUERIES),
    "ehr": lambda: EhrAgentDomain("ehr", EHR_CORPUS, EHR_QUERIES),
    "ad": lambda: AgentDriverDomain("ad", AD_CORPUS, None),
}

DOMAIN_NAMES = tuple(_REGISTRY)


def load_domain(name: str) -> Domain:
    if name not in _REGISTRY:
        raise ValueError(f"unknown domain {name!r}; choose from {DOMAIN_NAMES}")
    return _REGISTRY[name]()
