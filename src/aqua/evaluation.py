"""Controlled proposed-call replay, learned gating, and observed sandbox effects.

This measures candidate acceptance, not autonomous task completion. The oracle is
used for labels and a separately named reference-monitor baseline only.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import torch

from .benchmark import fingerprint, load_cases
from .intervention import corrections_for, scrub
from .io import atomic_write, digest, read_json, write_json
from .metrics import calibration_error, summarize
from .sandbox import observe
from .schema import VARIANTS
from .training import load_probe

MODES = ("unguarded", "reference_monitor", "detector", "scrub_only", "scrub_and_gate")


def evaluate(dataset, probe_path, output, backend, split="test"):
    if split not in {"validation", "test"}:
        raise ValueError("Evaluation only supports validation or test")
    cases = load_cases(dataset)
    model, checkpoint = load_probe(probe_path)
    collection = checkpoint["contract"]["collection"]
    if collection["dataset"] != fingerprint(cases) or collection["backend"] != backend.metadata:
        raise ValueError("Dataset/backend differs from the probe's activation collection")
    selected = [c for c in cases if c.split == split]
    if not selected:
        raise ValueError(f"No {split} cases")
    output = Path(output)
    # torch.save ZIP headers depend on the temporary filename; hash tensor values
    # and metadata so saving an unchanged probe does not invalidate resume.
    state_hash = hashlib.sha256(digest({k: v for k, v in checkpoint.items() if k != "model"}).encode())
    for name, value in sorted(checkpoint["model"].items()):
        state_hash.update(name.encode())
        state_hash.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    weight_hash = state_hash.hexdigest()
    contract = {"version": 1, "collection": collection, "probe_sha256": weight_hash, "split": split}
    manifest = output / "evaluation_config.json"
    if manifest.exists() and read_json(manifest) != contract:
        raise ValueError("Evaluation configuration changed; use a new output directory")
    write_json(manifest, contract)
    threshold = checkpoint["calibration"]["threshold"]
    records = []
    with torch.no_grad():
        for case in selected:
            path = output / "cases" / (digest(case.id)[:24] + ".json")
            if path.exists():
                cached = read_json(path)
                if cached.get("contract") != digest(contract) or cached["id"] != case.id:
                    raise ValueError(f"Stale evaluation record: {path}")
                records.append(cached)
                continue
            started = time.perf_counter()
            capture = backend.capture(case)
            correction = corrections_for(case, capture, model)
            probabilities = model(capture.hidden).sigmoid()
            corrected = scrub(capture.hidden, correction)
            corrected_probabilities = model(corrected).sigmoid()
            replay_score = backend.replay(case, correction)
            proposed = capture.call_score >= capture.abstain_score
            replayed = replay_score >= capture.abstain_score
            decisions = {
                "unguarded": proposed,
                "reference_monitor": proposed and all(case.labels.values()),
                "detector": proposed and probabilities.min().item() >= threshold,
                "scrub_only": replayed,
                "scrub_and_gate": replayed and corrected_probabilities.min().item() >= threshold,
            }
            record = {"id": case.id, "group": case.group, "variant": case.variant,
                      "tool": case.proposed.tool, "domain": case.domain,
                      "contract": digest(contract), "fields": capture.fields,
                      "labels": [case.labels[f] for f in capture.fields],
                      "probabilities": probabilities.tolist(),
                      "corrected_probabilities": corrected_probabilities.tolist(),
                      "encoded": model.encode(capture.hidden).tolist(),
                      "correction_norm": correction.norm().item(),
                      "call_score": capture.call_score, "abstain_score": capture.abstain_score,
                      "replay_score": replay_score, "latency_seconds": time.perf_counter() - started,
                      "decisions": {name: observe(case, decision) for name, decision in decisions.items()}}
            write_json(path, record)
            records.append(record)
    groups = {}
    for record in records:
        groups.setdefault(record["group"], {})[record["variant"]] = record
    equivalence, flips = [], []
    for variants in groups.values():
        z = [torch.tensor(variants[v]["encoded"]) for v in VARIANTS]
        equivalence.extend((z[0] - z[1]).norm(dim=-1).tolist())
        for index in (2, 3):
            changed = ~torch.tensor(variants[VARIANTS[index]]["labels"])
            flips.extend((z[0] - z[index]).norm(dim=-1)[changed].tolist())
    mean = lambda values: sum(values) / len(values) if values else None
    metrics = {
        "protocol": "matched proposed call versus null; mean conditional log-probability replay",
        "evidence": backend.metadata["evidence"], "split": split, "threshold": threshold,
        "metrics": {mode: summarize(records, mode) for mode in MODES},
        "representation": {
            "authorization_equivalence_gap": mean(equivalence),
            "authorization_flip_distance": mean(flips),
            "authorization_flip_sensitivity": mean([float(d >= checkpoint['contract']['losses']['margin']) for d in flips]),
            "field_ece": calibration_error([p for r in records for p in r["probabilities"]],
                                           [y for r in records for y in r["labels"]]),
        },
        "mean_case_latency_seconds": mean([r["latency_seconds"] for r in records]),
        "by_domain": {domain: {mode: summarize([r for r in records if r["domain"] == domain], mode)
                               for mode in MODES} for domain in sorted({r["domain"] for r in records})},
        "limitations": [
            "Synthetic AuthShift templates; not an AgentDojo/ASB reproduction.",
            "Candidate replay cannot regenerate alternative arguments or measure autonomous utility.",
            "Masked source ablation is an approximation; source interactions are not identified.",
            "Post-scrub gate uses corrected original states, not a second residual estimation.",
            "No adaptive payload attack or security certificate is established.",
        ],
    }
    write_json(output / "metrics.json", metrics)
    atomic_write(output / "records.jsonl", lambda p: p.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8"))
    return metrics
