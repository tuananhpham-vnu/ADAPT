"""Low-rank orthogonal authorization projection with conditional adversaries."""
from __future__ import annotations

import torch
from torch import nn


class ReverseGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, scale):
        ctx.scale = scale
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.scale * gradient, None


class AuthorizationProbe(nn.Module):
    def __init__(self, hidden_size, rank=4, domains=5):
        super().__init__()
        if not 1 <= rank <= hidden_size:
            raise ValueError("rank must be between 1 and hidden_size")
        self.config = {"hidden_size": hidden_size, "rank": rank, "domains": domains}
        self.basis = nn.Parameter(torch.randn(hidden_size, rank) / hidden_size ** .5)
        self.classifier = nn.Linear(rank, 1)
        # Condition nuisance predictions on authorization to suppress within-label
        # shortcuts without forcing removal of the authorization label itself.
        self.source_adversary = nn.Linear(rank + 1, 4)
        self.domain_adversary = nn.Linear(rank + 1, domains)
        self.register_buffer("center", torch.zeros(hidden_size))
        self.register_buffer("scale", torch.ones(()))

    def orthogonal_basis(self):
        return torch.linalg.qr(self.basis, mode="reduced").Q

    def encode(self, hidden):
        return ((hidden - self.center) / self.scale) @ self.orthogonal_basis()

    def forward(self, hidden):
        return self.classifier(self.encode(hidden)).squeeze(-1)

    def project(self, residual):
        """P_A r in original hidden-state units, not rank-dimensional coordinates."""
        basis = self.orthogonal_basis()
        return (residual @ basis) @ basis.T

    def nuisance_logits(self, encoded, labels, strength=1.):
        reversed_z = ReverseGradient.apply(encoded, strength)
        conditional = torch.cat([reversed_z, labels.unsqueeze(-1)], dim=-1)
        return self.source_adversary(conditional), self.domain_adversary(conditional)
