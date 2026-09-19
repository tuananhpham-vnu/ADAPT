"""GMM reference centers matching ``algo/trigger_optimization.py``."""
from __future__ import annotations

import torch


def fit_centers(vectors, count: int = 5, seed: int = 0) -> torch.Tensor:
    """Fit full-covariance GMM on every raw DPR embedding and return its means.

    This deliberately mirrors the upstream optimizer: five components,
    ``covariance_type='full'`` and ``random_state=0``.  There is no sampling or
    normalization before fitting.
    """
    x = torch.as_tensor(vectors).detach().float().cpu()
    if x.ndim != 2 or not len(x) or not torch.isfinite(x).all():
        raise ValueError("Expected nonempty finite [rows, dimensions] embeddings")
    if count < 1 or count > len(x):
        raise ValueError("count must be in [1, number of embeddings]")
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError as exc:
        raise RuntimeError(
            "scikit-learn is required for upstream-compatible GMM clustering"
        ) from exc
    gmm = GaussianMixture(
        n_components=count,
        covariance_type="full",
        random_state=seed,
    )
    gmm.fit(x.numpy())
    return torch.from_numpy(gmm.means_).to(dtype=x.dtype)
