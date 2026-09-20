"""The context-conditioned trigger generator and its ablation variants.

    c_m = f_phi({E(d) : d in D})          memory summary
    c_q = f_psi({E(q) : q in Q_sup})      clean-query summary
    h_l = MLP([c_m ; c_q ; p_l])
    A   = h_l W_out^T + b                 logits, [L, V]

Both summaries pool a *set*: no positional encoding over document order, so
permuting the snapshot cannot change the trigger.  That property is what makes
the shuffled-context control in ``evaluate`` meaningful, and it is asserted in
the tests rather than assumed.

The variants answer "is the conditioning doing anything?": ``memory+query`` is
the method, ``query``/``memory`` drop one branch (B5/B6), and ``none`` is an
unconditional generator (B4).  Every variant keeps the same modules and the
same parameter count -- a dropped branch reads a learned constant pseudo-set
instead of the episode's vectors -- so the comparison isolates information
rather than model size.
"""
from __future__ import annotations

from typing import Iterable

import torch
from torch import nn

VARIANTS = ("memory+query", "query", "memory", "none")


class SetEncoder(nn.Module):
    """Permutation-invariant pooling over a set of embeddings (DeepSets).

    ``attention=True`` swaps mean pooling for a learned attention weighting,
    which is still permutation invariant because the scores depend only on each
    element, never on its index.
    """

    def __init__(self, input_dim: int, hidden: int, output: int, *, attention: bool = False):
        super().__init__()
        self.element = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.attention = nn.Linear(hidden, 1) if attention else None
        self.readout = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, output),
        )

    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        if vectors.ndim != 2:
            raise ValueError("expected a [rows, dimensions] set of embeddings")
        encoded = self.element(vectors)
        if self.attention is None:
            pooled = encoded.mean(dim=0)
        else:
            weights = torch.softmax(self.attention(encoded).squeeze(-1), dim=0)
            pooled = weights @ encoded
        return self.readout(pooled)


class TriggerGenerator(nn.Module):
    """Maps an episode's context to a ``[L, V]`` logits matrix."""

    def __init__(
        self,
        *,
        embedding_dim: int,
        vocab_size: int,
        trigger_tokens: int,
        context_dim: int = 128,
        hidden: int = 256,
        variant: str = "memory+query",
        attention_pool: bool = False,
        constant_rows: int = 8,
    ):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
        self.variant = variant
        self.trigger_tokens = trigger_tokens
        self.vocab_size = vocab_size
        self.uses_memory = variant in ("memory+query", "memory")
        self.uses_query = variant in ("memory+query", "query")

        self.memory_encoder = SetEncoder(embedding_dim, hidden, context_dim,
                                         attention=attention_pool)
        self.query_encoder = SetEncoder(embedding_dim, hidden, context_dim,
                                        attention=attention_pool)
        # What a dropped branch reads instead of the episode's own vectors.
        self.constant_memory = nn.Parameter(torch.randn(constant_rows, embedding_dim) * 0.02)
        self.constant_query = nn.Parameter(torch.randn(constant_rows, embedding_dim) * 0.02)

        self.positions = nn.Parameter(torch.randn(trigger_tokens, context_dim) * 0.02)
        self.trunk = nn.Sequential(
            nn.Linear(3 * context_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.head = nn.Linear(hidden, vocab_size)

    def context(
        self, memory_vectors: torch.Tensor | None, query_vectors: torch.Tensor | None
    ) -> torch.Tensor:
        if self.uses_memory and memory_vectors is None:
            raise ValueError(f"variant {self.variant!r} needs memory vectors")
        if self.uses_query and query_vectors is None:
            raise ValueError(f"variant {self.variant!r} needs support-query vectors")
        memory = memory_vectors if self.uses_memory else self.constant_memory
        query = query_vectors if self.uses_query else self.constant_query
        return torch.cat(
            (self.memory_encoder(memory), self.query_encoder(query)), dim=-1
        )

    def forward(
        self,
        memory_vectors: torch.Tensor | None = None,
        query_vectors: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pooled = self.context(memory_vectors, query_vectors)
        repeated = pooled.unsqueeze(0).expand(self.trigger_tokens, -1)
        hidden = self.trunk(torch.cat((repeated, self.positions), dim=-1))
        return self.head(hidden)


def sample_memory_keys(
    vectors: torch.Tensor, count: int, *, generator: torch.Generator | None = None
) -> torch.Tensor:
    """A seeded subset of the snapshot's keys, so summarizing stays affordable."""
    if count >= vectors.shape[0]:
        return vectors
    index = torch.randperm(vectors.shape[0], generator=generator)[:count]
    return vectors[index]


def gmm_summary(vectors: torch.Tensor, components: int = 5, seed: int = 0) -> torch.Tensor:
    """Benign reference centers as a set: ``[components, dimensions + 2]``.

    Each row carries a center plus its mixture weight and dispersion.  Rows are
    sorted by weight so the summary does not depend on the arbitrary order
    scikit-learn assigns to components.

    These are a *benign reference*, not query routers and not five mandatory
    triggers -- the distinction ``_idea/group_conditioned_triggers.md`` insists
    on keeping separate in code.
    """
    from sklearn.mixture import GaussianMixture

    array = vectors.detach().float().cpu().numpy()
    model = GaussianMixture(n_components=min(components, len(array)),
                            covariance_type="full", random_state=seed).fit(array)
    means = torch.from_numpy(model.means_).float()
    weights = torch.from_numpy(model.weights_).float().unsqueeze(-1)
    dispersion = torch.from_numpy(
        model.covariances_.diagonal(axis1=1, axis2=2).sum(axis=1)
    ).float().unsqueeze(-1)
    rows = torch.cat((means, weights, dispersion), dim=-1)
    return rows[weights.squeeze(-1).argsort(descending=True)]


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def trainable_parameters(modules: Iterable[nn.Module]):
    for module in modules:
        yield from (p for p in module.parameters() if p.requires_grad)
