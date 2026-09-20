"""GMM reference centers matching ``algo/trigger_optimization.py``, plus routing."""
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


def assign(vectors, centers) -> torch.Tensor:
    """Nearest-center index for every row: LongTensor of shape [rows].

    Euclidean distance, because the centers this routes against are GMM means
    fitted on raw embeddings and are not unit-norm.  For the per-query arm the
    centers are the L2-normalized clean queries themselves, where nearest by
    Euclidean distance and nearest by cosine agree.

    Restored after commit ``6caa991a`` dropped it; the signature comes from its
    two call sites, ``hierarchy/experiment.py`` (training labels) and
    ``hierarchy/evaluation.py`` (routing unseen queries).
    """
    x = torch.as_tensor(vectors).detach().float().cpu()
    c = torch.as_tensor(centers).detach().float().cpu()
    if x.ndim != 2 or not len(x) or not torch.isfinite(x).all():
        raise ValueError("Expected nonempty finite [rows, dimensions] vectors")
    if c.ndim != 2 or not len(c) or not torch.isfinite(c).all():
        raise ValueError("Expected nonempty finite [centers, dimensions] centers")
    if x.shape[1] != c.shape[1]:
        raise ValueError("Vectors and centers must share the embedding dimension")
    return torch.cdist(x, c).argmin(dim=1).long()
