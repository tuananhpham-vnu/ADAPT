"""Clean-vector cache for memory snapshots and clean queries.

Benign vectors never change while a trigger is optimized, so they are encoded
once per (corpus, retriever) pair and memory-mapped afterwards.  Re-encoding
them inside the training loop is the single easiest way to turn a one-hour run
into a ten-hour one.

Writes are resumable in the same shape as ``src.triggers.margin.index``:
an ``.npy`` written through ``open_memmap`` plus a checkpoint recording how
many rows are already valid.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from src.triggers.artifacts import atomic_json, read_json, stable_hash
from src.triggers.mcat.encoding import encode_plain
from src.triggers.mcat.retrievers import Retriever

PREBUILT_QA_VECTORS = "ReAct/database/embeddings/agentpoison_dpr"


def cache_key(retriever: Retriever, kind: str, texts: Sequence[str]) -> str:
    """Identity of a cached matrix: retriever, what was encoded, and the text."""
    return stable_hash({
        "retriever": retriever.fingerprint(), "kind": kind,
        "rows": len(texts), "texts": stable_hash(list(texts)),
    })


def encode_corpus(
    retriever: Retriever,
    texts: Sequence[str],
    *,
    directory: Path,
    kind: str,
    batch_size: int = 32,
    resume: bool = True,
) -> np.ndarray:
    """Encode ``texts`` into ``<directory>/<kind>.npy``, resuming if possible."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{kind}.npy"
    checkpoint_path = directory / f"{kind}.checkpoint.json"
    key = cache_key(retriever, kind, texts)

    next_row = 0
    if resume and target.exists() and checkpoint_path.exists():
        checkpoint = read_json(checkpoint_path)
        if checkpoint.get("cache_key") != key:
            # A changed corpus or retriever invalidates every row; start over
            # rather than blending vectors from two different encoders.
            next_row = 0
        else:
            next_row = int(checkpoint.get("next_row", 0))
            if next_row >= len(texts):
                return np.load(target, mmap_mode="r")

    shape = (len(texts), retriever.hidden_size)
    mode = "r+" if target.exists() and next_row else "w+"
    vectors = np.lib.format.open_memmap(target, mode=mode, dtype="float32", shape=shape)
    for start in range(next_row, len(texts), batch_size):
        stop = min(start + batch_size, len(texts))
        with torch.no_grad():
            batch = encode_plain(
                retriever.model, retriever.tokenizer, list(texts[start:stop]),
                device=retriever.device, max_length=retriever.max_length,
            )
        vectors[start:stop] = batch.detach().cpu().numpy()
        vectors.flush()
        atomic_json(checkpoint_path, {"cache_key": key, "next_row": stop,
                                      "total_rows": len(texts)})
    atomic_json(directory / f"{kind}.manifest.json", {
        "cache_key": key, "rows": len(texts), "dimensions": retriever.hidden_size,
        "normalized": False, "retriever": retriever.fingerprint(),
    })
    return np.load(target, mmap_mode="r")


def load_prebuilt_qa_vectors(
    root: Path, retriever: Retriever, corpus_sha256: str, doc_ids: Sequence[str]
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Reuse ``ReAct/database/embeddings/agentpoison_dpr`` when it truly matches.

    Returns ``(matrix, reason)``; the matrix is ``None`` whenever reuse is
    refused, and ``reason`` always says why, so the index stage can report it
    rather than silently spending an hour re-encoding -- or, worse, silently
    reusing vectors that do not correspond to this run.

    Three checks earn the reuse:

    * the corpus hash, encoder name, revision and ``max_length`` all agree;
    * the index is **not** normalized -- the shipped one is ``l2``, while the
      objective ranks by raw dot product against GMM centers fitted on raw
      vectors, so mixing them would change the geometry, not just the scorer;
    * the stored ``paragraph_ids`` cover exactly ``doc_ids``.  They are stored
      in a different order than the domain emits, so the rows are permuted into
      ``doc_ids`` order before being handed back.  Skipping that step would
      pair every document with another document's vector.
    """
    directory = Path(root) / PREBUILT_QA_VECTORS
    manifest_path, vectors_path = directory / "manifest.json", directory / "vectors.npy"
    if not manifest_path.exists() or not vectors_path.exists():
        return None, {"reused": False, "reason": "no prebuilt index on disk"}

    manifest = read_json(manifest_path)
    mismatches = {
        key: (manifest.get(key), expected)
        for key, expected in (
            ("corpus_sha256", corpus_sha256),
            ("encoder", retriever.name),
            ("max_length", retriever.max_length),
        )
        if manifest.get(key) != expected
    }
    if retriever.revision is not None and manifest.get("encoder_revision") != retriever.revision:
        mismatches["encoder_revision"] = (manifest.get("encoder_revision"), retriever.revision)
    if mismatches:
        return None, {"reused": False, "reason": "manifest mismatch",
                      "mismatches": {k: list(v) for k, v in mismatches.items()}}

    normalization = manifest.get("normalization")
    if normalization not in (None, "none", False):
        return None, {
            "reused": False, "reason": "prebuilt index is normalized",
            "normalization": normalization,
            "detail": "MCAT ranks raw dot products against centers fitted on raw "
                      "vectors; reusing an l2-normalized index would change the geometry",
        }

    stored = list(manifest.get("paragraph_ids") or [])
    if set(stored) != set(doc_ids):
        return None, {"reused": False, "reason": "paragraph_ids do not cover the corpus",
                      "stored": len(stored), "wanted": len(doc_ids)}

    vectors = np.load(vectors_path, mmap_mode="r")
    if vectors.shape[0] != len(stored):
        return None, {"reused": False, "reason": "vector count differs from paragraph_ids",
                      "rows": int(vectors.shape[0]), "ids": len(stored)}

    position = {identifier: index for index, identifier in enumerate(stored)}
    order = np.asarray([position[identifier] for identifier in doc_ids], dtype=np.int64)
    return np.asarray(vectors[order], dtype="float32"), {
        "reused": True, "rows": len(doc_ids), "reordered": not np.array_equal(
            order, np.arange(len(doc_ids))
        ),
    }


def select_rows(vectors: np.ndarray, order: Sequence[str], wanted: Sequence[str]) -> torch.Tensor:
    """Gather the rows for ``wanted`` ids out of a matrix laid out as ``order``."""
    position = {identifier: index for index, identifier in enumerate(order)}
    missing = [identifier for identifier in wanted if identifier not in position]
    if missing:
        raise KeyError(f"{len(missing)} ids are absent from the cache, e.g. {missing[:3]}")
    rows = np.asarray([position[identifier] for identifier in wanted], dtype=np.int64)
    return torch.from_numpy(np.asarray(vectors[rows], dtype="float32"))


def summarize_cache(directory: Path) -> dict[str, Any]:
    directory = Path(directory)
    return {
        path.stem.replace(".manifest", ""): read_json(path)
        for path in sorted(directory.glob("*.manifest.json"))
    }
