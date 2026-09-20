"""Differentiable losses used by the staged AgentPoison optimizer.

All public functions follow a minimization convention and return scalar tensors
without detaching the computation graph.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_uniqueness_loss(
    query_embeddings: torch.Tensor, cluster_centers: torch.Tensor
) -> torch.Tensor:
    """Negative mean distance from every query to every clean cluster center."""
    if query_embeddings.ndim != 2:
        raise ValueError("query_embeddings must have shape [queries, dimensions]")
    if cluster_centers.ndim == 3 and cluster_centers.shape[0] == 1:
        cluster_centers = cluster_centers[0]
    if cluster_centers.ndim != 2:
        raise ValueError("cluster_centers must have shape [centers, dimensions]")
    
    # Match upstream AgentPoison: Euclidean distance in the original DPR
    # embedding space. Only the retrieval-margin loss uses cosine normalization.
    return -torch.cdist(query_embeddings, cluster_centers).mean()


def compute_compactness_loss(query_embeddings: torch.Tensor) -> torch.Tensor:
    """Mean L2 distance of query embeddings from their centroid."""
    if query_embeddings.ndim != 2:
        raise ValueError("query_embeddings must have shape [queries, dimensions]")
    centroid = query_embeddings.mean(dim=0, keepdim=True)
    return torch.linalg.vector_norm(query_embeddings - centroid, dim=1).mean()


def compute_retrieval_margin_loss(
    query_embeddings: torch.Tensor,
    clean_embeddings: torch.Tensor,
    poison_embeddings: torch.Tensor,
    top_k: int = 1,
    margin: float = 0.1,
) -> torch.Tensor:
    """Hinge loss requiring the K-th poison to beat the best clean key by margin."""
    if query_embeddings.ndim != 2 or clean_embeddings.ndim != 2 or poison_embeddings.ndim != 2:
        raise ValueError("query, clean and poison embeddings must be rank-2 tensors")
    
    if not 1 <= top_k <= poison_embeddings.shape[0]:
        raise ValueError("top_k must be between 1 and the number of poison embeddings")
    
    query = F.normalize(query_embeddings, p=2, dim=-1)
    clean = F.normalize(clean_embeddings, p=2, dim=-1)
    poison = F.normalize(poison_embeddings, p=2, dim=-1)
    
    clean_scores = query @ clean.T
    poison_scores = query @ poison.T
    
    best_clean = clean_scores.max(dim=1).values
    kth_poison = poison_scores.topk(
        k=top_k,
        dim=1
    ).values[:, -1]
    
    loss_per_query = F.relu(float(margin) + best_clean - kth_poison)
    return loss_per_query.mean()


def compute_total_loss(
    l_uni: torch.Tensor,
    l_cpt: torch.Tensor,
    l_margin: torch.Tensor,
    lambda_cpt: float = 0.1,
    margin_weight: float = 0.0,
) -> torch.Tensor:
    """Combine retrieval losses; coherence and target constraints stay external."""
    return l_uni + float(lambda_cpt) * l_cpt + float(margin_weight) * l_margin
