"""One atomic tensor shard per quadruple: bounded RAM and resumable collection."""
from __future__ import annotations

from pathlib import Path

import torch

from .benchmark import fingerprint, load_cases
from .io import digest, read_json, write_json, write_tensor
from .schema import SOURCES, VARIANTS


def collect(dataset, output, backend):
    cases = load_cases(dataset)
    output = Path(output)
    contract = {"version": 1, "dataset": fingerprint(cases), "backend": backend.metadata}
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and read_json(manifest_path)["contract"] != contract:
        raise ValueError("Collection configuration/dataset changed; use a new output directory")
    groups = {}
    for case in cases:
        groups.setdefault(case.group, []).append(case)
    manifest = {"contract": contract, "shards": []}
    # Write the contract before work starts, so interrupted runs cannot mix models.
    if not manifest_path.exists():
        write_json(manifest_path, manifest)
    domains = sorted({c.domain for c in cases})
    for group, rows in sorted(groups.items()):
        rows.sort(key=lambda c: VARIANTS.index(c.variant))
        name = digest(group)[:24] + ".pt"
        path = output / "shards" / name
        key = digest([contract, [c.to_dict() for c in rows]])
        if path.exists():
            shard = torch.load(path, map_location="cpu", weights_only=True)
            if shard.get("key") != key:
                raise ValueError(f"Stale/corrupt shard: {path}")
        else:
            captures = [backend.capture(c, residuals=False) for c in rows]
            fields = captures[0].fields
            if any(c.fields != fields for c in captures):
                raise ValueError(f"{group}: inconsistent field order")
            shard = {"key": key, "group": group, "split": rows[0].split, "fields": fields,
                     "hidden": torch.stack([c.hidden for c in captures]),
                     "labels": torch.tensor([[c.labels[f] for f in fields] for c in rows], dtype=torch.float32),
                     "source": torch.tensor([[SOURCES.index(c.provenance[f]) for f in fields] for c in rows]),
                     "domain": torch.full((4, len(fields)), domains.index(rows[0].domain)),
                     "case_ids": [c.id for c in rows]}
            if not torch.isfinite(shard["hidden"]).all():
                raise ValueError(f"{group}: nonfinite activations")
            write_tensor(path, shard)
        manifest["shards"].append({"file": "shards/" + name, "group": group, "split": rows[0].split})
    manifest["domains"] = domains
    manifest["complete"] = True
    write_json(manifest_path, manifest)
    return manifest


class ActivationDataset(torch.utils.data.Dataset):
    def __init__(self, directory, split):
        self.directory = Path(directory)
        self.manifest = read_json(self.directory / "manifest.json")
        if not self.manifest.get("complete"):
            raise ValueError("Collection incomplete; resume collect first")
        self.entries = [s for s in self.manifest["shards"] if s["split"] == split]
        if not self.entries:
            raise ValueError(f"No {split} activation shards")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        path = self.directory / self.entries[index]["file"]
        if not path.resolve().is_relative_to(self.directory.resolve()):
            raise ValueError("Shard path escapes activation directory")
        return torch.load(path, map_location="cpu", weights_only=True)


def collate_quadruples(rows):
    # Each field is a matched quadruple; variable field counts can share a batch.
    return {key: torch.cat([r[key].transpose(0, 1) for r in rows], dim=0)
            for key in ("hidden", "labels", "source", "domain")}
