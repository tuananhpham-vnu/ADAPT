"""Audit clustering/projection on cached embeddings; no model or attack calls.

Example:
    python scripts/audit_embedding_clusters.py --embeddings vectors.npy --output audit.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import sklearn
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits


def audit(path, *, sample_size=3000, dimensions=(2, 32, 128), clusters=5, seed=42):
    vectors = np.load(path, mmap_mode="r", allow_pickle=False)
    if vectors.ndim != 2 or min(vectors.shape) < 2:
        raise ValueError("Expected a nonempty [samples, dimensions] matrix")
    if sample_size < 30 or clusters < 2:
        raise ValueError("Need sample_size >= 30 and clusters >= 2")
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(vectors), min(sample_size, len(vectors)), replace=False)
    data = np.array(vectors[selected], dtype=np.float32)
    if not np.isfinite(data).all() or np.any(np.linalg.norm(data, axis=1) == 0):
        raise ValueError("Embeddings must be finite and nonzero")
    data = normalize(data)
    cut = int(.8 * len(data))
    train, heldout = data[:cut], data[cut:]
    if len(train) < max(10, clusters) or len(heldout) < 3:
        raise ValueError("Too few embeddings for the train/held-out audit")
    if any(d < 1 or d > min(train.shape) for d in dimensions):
        raise ValueError("PCA dimensions exceed the sampled training matrix rank bound")
    k = min(10, len(train))
    reference_neighbors = NearestNeighbors(n_neighbors=k, metric="cosine").fit(train).kneighbors(heldout, return_distance=False)
    rows, reference_labels = [], None
    for dimension in dict.fromkeys([data.shape[1], *dimensions]):
        started = time.perf_counter()
        if dimension == data.shape[1]:
            train_low, heldout_low, variance = train, heldout, 1.
        else:
            projection = PCA(n_components=dimension, svd_solver="randomized", random_state=seed)
            train_low = projection.fit_transform(train)
            heldout_low = projection.transform(heldout)
            variance = float(projection.explained_variance_ratio_.sum())
        # Normalize after PCA so all neighbor comparisons still use cosine.
        train_low, heldout_low = normalize(train_low), normalize(heldout_low)
        estimator = MiniBatchKMeans(n_clusters=clusters, random_state=seed,
                                    n_init=3, batch_size=256).fit(train_low)
        labels = estimator.predict(heldout_low)
        neighbors = NearestNeighbors(n_neighbors=k, metric="cosine").fit(train_low).kneighbors(heldout_low, return_distance=False)
        recall = float(np.mean([len(set(a) & set(b)) / k for a, b in zip(reference_neighbors, neighbors)]))
        if reference_labels is None:
            reference_labels = labels
        distinct = len(set(labels))
        silhouette = float(silhouette_score(heldout, labels, metric="cosine")) if 1 < distinct < len(labels) else None
        rows.append({"dimensions": dimension, "explained_variance": variance,
                     "neighbor_recall_at_10_vs_original": recall,
                     "cluster_ari_vs_original": float(adjusted_rand_score(reference_labels, labels)),
                     "silhouette_in_original_space": silhouette,
                     "heldout_cluster_sizes": np.bincount(labels, minlength=clusters).tolist(),
                     "fit_and_audit_seconds": time.perf_counter() - started})
    # Online update of a fixed number of centers, distinct from adaptive-K clustering.
    online = MiniBatchKMeans(n_clusters=clusters, random_state=seed, n_init=3, batch_size=256)
    batches = list(np.array_split(train, max(1, len(train) // max(256, clusters))))
    online.partial_fit(batches[0])
    initial = online.predict(heldout)
    for batch in batches[1:]:
        online.partial_fit(batch)
    updated = online.predict(heldout)
    return {
        "embeddings": str(path), "source_shape": list(vectors.shape), "seed": seed,
        "sample_sha256": hashlib.sha256(data.tobytes()).hexdigest(),
        "versions": {"numpy": np.__version__, "scikit_learn": sklearn.__version__},
        "sample_size": len(data), "train_size": len(train), "heldout_size": len(heldout),
        "clusters": clusters, "results": rows,
        "online_fixed_k": {"batches": len(batches), "first_vs_final_ari": float(adjusted_rand_score(initial, updated)),
                           "heldout_assignment_change_rate": float(np.mean(initial != updated))},
        "limitations": [
            "Geometry audit only; no trigger training, attack success, or naturalness measurement.",
            "If input is a document cache, these are document clusters, not query/task groups.",
            "PCA and clustering fit only the sampled training partition; heldout tests geometry.",
            "Timing includes PCA, clustering and neighbor/silhouette evaluation, not encoder inference.",
            "One seed and a corpus subset; repeat seeds before drawing a general conclusion.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=3000)
    parser.add_argument("--dimensions", type=int, nargs="+", default=[2, 32, 128])
    parser.add_argument("--clusters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        with threadpool_limits(limits=2):
            result = audit(args.embeddings, sample_size=args.sample_size, dimensions=args.dimensions,
                           clusters=args.clusters, seed=args.seed)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
