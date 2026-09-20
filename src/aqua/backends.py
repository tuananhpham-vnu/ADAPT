"""Backend interface, explicit tensor fixture, and white-box HF decoder adapter."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Protocol

import torch

from .io import digest
from .prompts import prediction_positions, render, serialize_call


@dataclass
class Capture:
    fields: list[str]
    hidden: torch.Tensor                 # [fields, hidden]
    residuals: dict[str, torch.Tensor]   # source -> [fields, hidden]
    call_score: float                    # mean conditional log probability
    abstain_score: float


class Backend(Protocol):
    metadata: dict

    def capture(self, case, *, residuals=True) -> Capture: ...
    def replay(self, case, corrections: torch.Tensor) -> float: ...


class FixtureBackend:
    """Label-encoded synthetic vectors ONLY for software checks, not LLM evidence."""
    def __init__(self, hidden_size=16, seed=42):
        if hidden_size < 4:
            raise ValueError("Fixture hidden_size must be >= 4")
        self.hidden_size, self.seed = hidden_size, seed
        self.metadata = {"backend": "fixture", "hidden_size": hidden_size, "seed": seed,
                         "evidence": "synthetic software test; labels encoded in activations"}

    def capture(self, case, *, residuals=True):
        fields = list(case.proposed.values())
        generator = torch.Generator().manual_seed(int(digest([case.group, self.seed])[:12], 16))
        hidden = torch.randn(len(fields), self.hidden_size, generator=generator) * .03
        hidden[:, 0] = torch.tensor([1. if case.labels[f] else -1. for f in fields])
        hidden[:, 1] = float(case.variant == "authorized_user") * .4
        residual = torch.zeros_like(hidden)
        residual[:, 0] = .2
        return Capture(fields, hidden, {s: residual.clone() for s in case.context} if residuals else {}, -.2, -.4)

    def replay(self, case, corrections):
        # Artificial decoder response, deliberately identified as a fixture.
        return -.2 - float(corrections.square().mean())


class HuggingFaceBackend:
    """Hooks a decoder block; supports models exposing a sequence at layer_path.

    Full-context replay uses no KV cache. Source ablation masks source tokens at
    fixed positions. Both collection and intervention use the same block output.
    """
    def __init__(self, model, tokenizer, *, model_id, layer=-1, layer_path="model.layers",
                 max_length=2048, revision=None):
        if not tokenizer.is_fast:
            raise ValueError("A fast tokenizer with offset mappings is required")
        self.model, self.tokenizer = model.eval(), tokenizer
        self.max_length = max_length
        blocks = model.get_submodule(layer_path)
        if not -len(blocks) <= layer < len(blocks):
            raise ValueError(f"layer must index {len(blocks)} decoder blocks")
        self.layer = layer % len(blocks)
        self.block = blocks[self.layer]
        self.device = model.get_input_embeddings().weight.device
        self.metadata = {"backend": "huggingface", "model": model_id, "revision": revision,
                         "resolved_revision": getattr(model.config, "_commit_hash", None),
                         "layer": self.layer, "layer_path": layer_path, "max_length": max_length,
                         "dtype": str(next(model.parameters()).dtype),
                         "hidden_size": model.config.hidden_size,
                         "evidence": "teacher-forced candidate replay; not free-generation evaluation"}

    @classmethod
    def load(cls, model_id, *, device="cpu", dtype="float32", revision=None, **kwargs):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, torch_dtype=getattr(torch, dtype), trust_remote_code=False)
        model.to(device)
        return cls(model, tokenizer, model_id=model_id, revision=revision, **kwargs)

    def _inputs(self, case, completion=None):
        prompt, sources = render(case, self.tokenizer)
        call, spans = serialize_call(case.proposed)
        completion = call if completion is None else completion
        batch = self.tokenizer(prompt + completion, return_tensors="pt",
                               return_offsets_mapping=True, add_special_tokens=False)
        offsets = batch.pop("offset_mapping")[0].tolist()
        if len(offsets) > self.max_length:
            raise ValueError(f"{case.id}: {len(offsets)} tokens exceed max_length={self.max_length}; no silent truncation")
        inputs = {k: v.to(self.device) for k, v in batch.items() if k in {"input_ids", "attention_mask"}}
        positions = {f: prediction_positions(offsets, len(prompt) + a, len(prompt) + b)
                     for f, (a, b) in spans.items()} if completion == call else {}
        targets = prediction_positions(offsets, len(prompt), len(prompt) + len(completion))
        return inputs, offsets, sources, positions, targets

    @contextmanager
    def _hook(self, callback):
        handle = self.block.register_forward_hook(callback)
        try:
            yield
        finally:
            handle.remove()

    def _forward(self, inputs, positions, targets, corrections=None):
        captured = []

        def hook(module, arguments, output):
            state = output[0] if isinstance(output, tuple) else output
            if corrections is not None:
                state = state.clone()
                # A tokenizer can merge a boundary; average corrections on shared positions.
                delta, counts = torch.zeros_like(state), torch.zeros_like(state[..., :1])
                for row, token_ids in enumerate(positions.values()):
                    delta[:, token_ids] += corrections[row].to(state)
                    counts[:, token_ids] += 1
                state = state - delta / counts.clamp_min(1)
            if positions:
                captured.append(torch.stack([state[0, p].float().mean(0) for p in positions.values()]).detach().cpu())
            return (state, *output[1:]) if isinstance(output, tuple) else state

        with self._hook(hook), torch.no_grad():
            output = self.model(**inputs, use_cache=False, return_dict=True)
        logits = output.logits[0, targets].float()
        labels = inputs["input_ids"][0, torch.tensor(targets, device=self.device) + 1]
        score = logits.log_softmax(-1).gather(1, labels[:, None]).mean().item()
        if not torch.isfinite(torch.tensor(score)):
            raise ValueError("Model returned a nonfinite candidate score")
        return captured[0] if captured else None, score

    def capture(self, case, *, residuals=True):
        inputs, offsets, sources, positions, targets = self._inputs(case)
        hidden, score = self._forward(inputs, positions, targets)
        source_residuals = {}
        for source, (start, end) in (sources.items() if residuals else []):
            ablated = {k: v.clone() for k, v in inputs.items()}
            tokens = [i for i, (a, b) in enumerate(offsets) if b > start and a < end]
            ablated["attention_mask"][:, tokens] = 0
            baseline, _ = self._forward(ablated, positions, targets)
            source_residuals[source] = hidden - baseline
        null_score = 0.
        if residuals:
            null_inputs, _, _, _, null_targets = self._inputs(case, "null")
            _, null_score = self._forward(null_inputs, {}, null_targets)
        return Capture(list(positions), hidden, source_residuals, score, null_score)

    def replay(self, case, corrections):
        inputs, _, _, positions, targets = self._inputs(case)
        _, score = self._forward(inputs, positions, targets, corrections)
        return score


def make_backend(config):
    if config["backend"] == "fixture":
        return FixtureBackend(config.get("hidden_size", 16), config.get("seed", 42))
    if config["backend"] != "huggingface":
        raise ValueError("backend must be fixture or huggingface")
    return HuggingFaceBackend.load(config["model"], device=config.get("device", "cpu"),
                                  dtype=config.get("dtype", "float32"), layer=config.get("layer", -1),
                                  layer_path=config.get("layer_path", "model.layers"),
                                  max_length=config.get("max_length", 2048), revision=config.get("revision"))
