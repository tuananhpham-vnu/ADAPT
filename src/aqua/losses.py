"""Minimization losses for [matched fields, four variants, dimensions]."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class LossConfig:
    equivalence: float = 1.
    flip: float = 1.
    authorization: float = 1.
    shortcut: float = .1
    margin: float = 1.

    def __post_init__(self):
        if any(not math.isfinite(v) or v < 0 for v in vars(self).values()) or self.margin == 0:
            raise ValueError("Loss weights must be finite/nonnegative; margin must be positive")


def quotient_loss(model, batch, config: LossConfig):
    h, labels = batch["hidden"], batch["labels"]
    if h.ndim != 3 or h.shape[1] != 4 or labels.shape != h.shape[:2] or h.shape[0] == 0:
        raise ValueError("Expected a nonempty [fields, 4, hidden] matched batch")
    z = model.encode(h)
    equivalence = (z[:, 0] - z[:, 1]).square().sum(-1).mean()
    distances = torch.linalg.vector_norm(z[:, :1] - z[:, 2:], dim=-1)
    changed = labels[:, :1] != labels[:, 2:]
    hinges = F.relu(config.margin - distances).square()
    flip = (hinges * changed).sum() / changed.sum().clamp_min(1)
    logits = model.classifier(z).squeeze(-1)
    authorization = F.binary_cross_entropy_with_logits(logits, labels)
    source_logits, domain_logits = model.nuisance_logits(z, labels)
    source = F.cross_entropy(source_logits.flatten(0, 1), batch["source"].flatten())
    domain = F.cross_entropy(domain_logits.flatten(0, 1), batch["domain"].flatten())
    total = (config.equivalence * equivalence + config.flip * flip +
             config.authorization * authorization + config.shortcut * (source + domain))
    return total, {"equivalence": equivalence, "flip": flip,
                   "authorization": authorization, "source": source, "domain": domain}
