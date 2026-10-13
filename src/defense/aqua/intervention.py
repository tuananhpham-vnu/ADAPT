"""Field-wise causal scrubbing. Authorization metadata comes from the harness."""
from __future__ import annotations

import torch


def corrections_for(case, capture, model):
    correction = torch.zeros_like(capture.hidden)
    for source, residual in capture.residuals.items():
        if residual.shape != capture.hidden.shape:
            raise ValueError("Residual and field activation shapes must match")
        denied = torch.tensor([source not in case.authority[f] for f in capture.fields], dtype=torch.bool)
        correction[denied] += model.project(residual)[denied]
    return correction


def scrub(hidden, correction):
    if hidden.shape != correction.shape:
        raise ValueError("Correction and hidden state shapes must match")
    return hidden - correction
