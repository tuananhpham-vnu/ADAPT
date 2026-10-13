"""Mini-batch training, train-only scaling, validation calibration, exact resume."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .collection import ActivationDataset, collate_quadruples
from .io import write_json, write_tensor
from .losses import LossConfig, quotient_loss
from .model import AuthorizationProbe


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 30
    batch_size: int = 16
    rank: int = 4
    learning_rate: float = .003
    seed: int = 42
    max_false_allow: float = .05

    def __post_init__(self):
        if min(self.epochs, self.batch_size, self.rank) < 1:
            raise ValueError("epochs, batch_size and rank must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if not 0 <= self.max_false_allow <= 1:
            raise ValueError("max_false_allow must be in [0, 1]")


def calibrate(scores, labels, max_false_allow):
    """Call-level min-field probability; tune exclusively on validation cases."""
    if not scores or len(scores) != len(labels) or not any(labels) or all(labels):
        raise ValueError("Calibration requires both authorized and unauthorized validation calls")
    positives, negatives = sum(labels), len(labels) - sum(labels)
    candidates = sorted({0., 1. + 1e-6, *scores, *(s + 1e-7 for s in scores)})
    options = []
    for threshold in candidates:
        fa = sum(s >= threshold and not y for s, y in zip(scores, labels)) / negatives
        utility = sum(s >= threshold and y for s, y in zip(scores, labels)) / positives
        if fa <= max_false_allow:
            options.append((utility, -fa, threshold))
    utility, negative_fa, threshold = max(options)
    return {"threshold": threshold, "validation_false_allow": -negative_fa,
            "validation_allow_rate_authorized": utility, "max_false_allow": max_false_allow}


def normalize_from_train(model, dataset):
    total = torch.zeros(model.config["hidden_size"], dtype=torch.float64)
    squared, count = 0., 0
    for row in dataset:
        h = row["hidden"].double().flatten(0, 1)
        total += h.sum(0)
        squared += h.square().sum().item()
        count += h.shape[0]
    center = total / count
    variance = (squared / count - center.square().sum()).clamp_min(1e-8)
    model.center.copy_(center.float())
    # Scalar normalization preserves the Euclidean projection geometry.
    model.scale.copy_(variance.sqrt().float())


def train(directory, output, config=TrainConfig(), losses=LossConfig(), device="cpu"):
    train_data, validation = ActivationDataset(directory, "train"), ActivationDataset(directory, "validation")
    architecture = {"hidden_size": train_data[0]["hidden"].shape[-1], "rank": config.rank,
                    "domains": len(train_data.manifest["domains"])}
    settings = asdict(config)
    settings.pop("epochs")  # Extend a run without changing its optimization contract.
    contract = {"version": 1, "collection": train_data.manifest["contract"],
                "settings": settings, "losses": asdict(losses), "architecture": architecture,
                "device": device}
    output = Path(output)
    checkpoint_path = output / "checkpoint.pt"
    torch.manual_seed(config.seed)
    model = AuthorizationProbe(**architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    start, history = 0, []
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
        if checkpoint["contract"] != contract:
            raise ValueError("Training contract changed; use a new output directory")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        torch.set_rng_state(checkpoint["rng"])
        start, history = checkpoint["epoch"], checkpoint["history"]
        if config.epochs < start:
            raise ValueError(f"Checkpoint already has {start} epochs; cannot resume backwards")
    else:
        normalize_from_train(model, train_data)
    for epoch in range(start, config.epochs):
        generator = torch.Generator().manual_seed(config.seed + epoch)
        loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True,
                            generator=generator, collate_fn=collate_quadruples)
        model.train()
        sums, count = {}, 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss, parts = quotient_loss(model, batch, losses)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            size = batch["hidden"].shape[0]
            count += size
            for key, value in {"total": loss, **parts}.items():
                sums[key] = sums.get(key, 0.) + float(value.detach()) * size
        history.append({"epoch": epoch + 1, **{k: v / count for k, v in sums.items()}})
        write_tensor(checkpoint_path, {"contract": contract, "model": model.state_dict(),
                                      "optimizer": optimizer.state_dict(), "epoch": epoch + 1,
                                      "rng": torch.get_rng_state(), "history": history})
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for row in validation:
            probabilities = model(row["hidden"].to(device)).sigmoid()
            scores.extend(probabilities.min(dim=1).values.tolist())
            labels.extend(row["labels"].bool().all(dim=1).tolist())
    calibration = calibrate(scores, labels, config.max_false_allow)
    artifact = {"version": 1, "contract": contract, "model": model.cpu().state_dict(),
                "calibration": calibration, "epochs": config.epochs}
    write_tensor(output / "probe.pt", artifact)
    write_json(output / "training.json", {"contract": contract, "history": history, "calibration": calibration})
    return artifact


def load_probe(path):
    checkpoint = torch.load(path, weights_only=True, map_location="cpu")
    model = AuthorizationProbe(**checkpoint["contract"]["architecture"])
    model.load_state_dict(checkpoint["model"])
    return model.eval(), checkpoint
