"""Backend interface, explicit tensor fixture, and white-box HF decoder adapter."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Protocol

import torch

from .io import digest
from .prompts import parse_call_with_spans, prediction_positions, render, serialize_call


@dataclass
class Capture:
    fields: list[str]
    hidden: torch.Tensor                 # [fields, hidden]
    residuals: dict[str, torch.Tensor]   # source -> [fields, hidden]
    call_score: float                    # mean conditional log probability
    abstain_score: float


@dataclass
class Proposal:
    """A tool call the model produced itself, plus why it is or is not usable.

    VN — Khác hẳn ``Capture``: ``Capture`` chấm một candidate do benchmark cấp,
    còn đây là thứ agent **tự quyết định làm**. ``status`` phân biệt sáu kiểu hỏng
    (xem ``prompts.PROPOSAL_STATUS``); ``call is None`` không đồng nghĩa với
    abstain, phải đọc ``status`` mới biết là abstain hợp lệ hay parse lỗi.

    Unlike ``Capture``, which scores a candidate the benchmark supplied, this is
    what the agent decided to do.  ``call is None`` does not mean abstention on
    its own -- ``status`` distinguishes a valid ``null`` from a parse failure.
    """

    text: str
    call: object | None                  # schema.Call when status == "ok"
    status: str
    fields: list[str]
    hidden: torch.Tensor | None          # [fields, hidden]; None when unparsable
    residuals: dict[str, torch.Tensor]
    generated_tokens: int


class Backend(Protocol):
    metadata: dict

    def capture(self, case, *, residuals=True) -> Capture: ...
    def replay(self, case, corrections: torch.Tensor) -> float: ...
    def propose(self, case, *, corrections=None, residuals=True) -> Proposal: ...


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

    def propose(self, case, *, corrections=None, residuals=True):
        """Deterministic stand-in generation, for plumbing only.

        VN — Fixture phải đi qua **cả hai** nhánh (parse được và parse lỗi) để test
        chạm tới nhánh lỗi mà không cần model. Hai điều cố ý KHÔNG làm:

        * lỗi parse chọn theo hash của group, không theo nhãn quyền — nếu buộc vào
          authorized thì fixture sẽ tự tạo ra tương quan "model hỏng đúng lúc";
        * khi có ``corrections``, việc đổi hành vi cũng chọn theo tính chẵn lẻ của
          group, **không** theo việc case có được phép hay không. Nếu buộc vào
          quyền, fixture sẽ trông như causal scrubbing hoạt động, và đó là kết luận
          mà chỉ model thật mới được phép đưa ra.

        Both the parse failure and the regeneration change key off the group
        hash, never off authorization: tying either to the labels would let the
        fixture manufacture the appearance of a working defense.
        """
        token = int(digest([case.group, self.seed])[:12], 16)
        capture = self.capture(case, residuals=residuals)
        if token % 10 == 0:
            return Proposal(text="I will not answer in JSON.", call=None,
                            status="parse_error", fields=[], hidden=None,
                            residuals=capture.residuals, generated_tokens=7)
        if corrections is not None and token % 2 == 0:
            return Proposal(text="null", call=None, status="abstain", fields=[],
                            hidden=None, residuals=capture.residuals,
                            generated_tokens=1)
        text, _ = serialize_call(case.proposed)
        call, _, status = parse_call_with_spans(text)
        return Proposal(text=text, call=call, status=status, fields=capture.fields,
                        hidden=capture.hidden, residuals=capture.residuals,
                        generated_tokens=len(text.split()))


class HuggingFaceBackend:
    """Hooks a decoder block; supports models exposing a sequence at layer_path.

    Full-context replay uses no KV cache. Source ablation masks source tokens at
    fixed positions. Both collection and intervention use the same block output.
    """
    def __init__(self, model, tokenizer, *, model_id, layer=-1, layer_path="model.layers",
                 max_length=2048, revision=None, max_new_tokens=64):
        if not tokenizer.is_fast:
            raise ValueError("A fast tokenizer with offset mappings is required")
        self.model, self.tokenizer = model.eval(), tokenizer
        self.max_length, self.max_new_tokens = max_length, max_new_tokens
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
                         # VN — Cấu hình decode nằm trong metadata, nên nó nằm trong
                         # contract: đổi max_new_tokens là đổi thí nghiệm.
                         # In the contract, so changing decoding forces a new output dir.
                         "generation": {"max_new_tokens": max_new_tokens, "do_sample": False,
                                        "num_beams": 1, "stop": "}}"},
                         "evidence": "teacher-forced candidate replay; not free-generation evaluation"}

    @classmethod
    def load(cls, model_id, *, device="cpu", dtype="float32", revision=None, **kwargs):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, torch_dtype=getattr(torch, dtype), trust_remote_code=False)
        model.to(device)
        return cls(model, tokenizer, model_id=model_id, revision=revision, **kwargs)

    def _inputs(self, case, completion=None, spans=None):
        """Tokenize prompt+completion and locate the field prediction positions.

        VN — ``spans`` cho phép chấm activation của một completion **do model sinh
        ra** thay vì candidate của benchmark. Span luôn đi qua cùng
        ``prediction_positions``, nên không có con đường thứ hai để tính vị trí:
        hai đường tính vị trí là hai cách lệch nhau âm thầm.

        ``spans`` lets the field positions come from a generated completion
        instead of the benchmark candidate, while still going through the one
        ``prediction_positions`` implementation.
        """
        prompt, sources = render(case, self.tokenizer)
        call, call_spans = serialize_call(case.proposed)
        if completion is None:
            completion, spans = call, call_spans
        batch = self.tokenizer(prompt + completion, return_tensors="pt",
                               return_offsets_mapping=True, add_special_tokens=False)
        offsets = batch.pop("offset_mapping")[0].tolist()
        if len(offsets) > self.max_length:
            raise ValueError(f"{case.id}: {len(offsets)} tokens exceed max_length={self.max_length}; no silent truncation")
        inputs = {k: v.to(self.device) for k, v in batch.items() if k in {"input_ids", "attention_mask"}}
        positions = {f: prediction_positions(offsets, len(prompt) + a, len(prompt) + b)
                     for f, (a, b) in (spans or {}).items()}
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

    def propose(self, case, *, corrections=None, residuals=True):
        """Let the model write the call, then measure the call it wrote.

        VN — Hai lượt, và phải là hai lượt:

        1. **Sinh** greedy (``do_sample=False``, 1 beam) từ prompt. Cấu hình decode
           đi vào ``metadata`` nên nó nằm trong contract — đổi ``max_new_tokens``
           là phải dùng thư mục output mới, giống như đổi layer hay dtype.
        2. **Chấm** ``prompt + text`` với ``use_cache=False`` và hook trên đúng
           decoder block mà ``capture`` dùng. Nếu lượt 2 dùng block khác hay quy
           ước vị trí khác thì activation của hai giao thức không so được.

        ``corrections`` là biến thể ``scrub_regenerate_twopass``: correction ước
        lượng trên lượt sinh trước được trừ ở **mọi token mới**. Đây là **xấp xỉ**
        — correction thuộc về một chuỗi khác chuỗi đang sinh — và phải được báo
        cáo dưới tên riêng, không được gọi là causal scrubbing đúng nghĩa.

        Generation is greedy and its config is part of ``metadata`` and therefore
        of the contract.  The scoring pass reuses ``capture``'s block and
        position convention.  With ``corrections`` the delta is subtracted at
        every newly generated position, which is an approximation because the
        correction was estimated on a different continuation.
        """
        prompt, _ = render(case, self.tokenizer)
        batch = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(self.device) for k, v in batch.items()
                  if k in {"input_ids", "attention_mask"}}
        length = int(inputs["input_ids"].shape[1])
        if length + self.max_new_tokens > self.max_length:
            raise ValueError(
                f"{case.id}: prompt of {length} tokens plus {self.max_new_tokens} new "
                f"tokens exceeds max_length={self.max_length}; no silent truncation")

        delta = None if corrections is None else corrections.mean(0)
        with self._generation_hook(delta, length), torch.no_grad():
            produced = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                num_beams=1, pad_token_id=self.tokenizer.pad_token_id
                or self.tokenizer.eos_token_id)
        new_tokens = produced[0, length:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        text = text.split("}}")[0] + "}}" if "}}" in text else text
        call, spans, status = parse_call_with_spans(text)
        if call is None:
            return Proposal(text=text, call=None, status=status, fields=[], hidden=None,
                            residuals={}, generated_tokens=int(new_tokens.shape[0]))

        scored, offsets, sources, positions, targets = self._inputs(case, text, spans)
        hidden, _ = self._forward(scored, positions, targets)
        source_residuals = {}
        for source, (start, end) in (sources.items() if residuals else []):
            ablated = {k: v.clone() for k, v in scored.items()}
            tokens = [i for i, (a, b) in enumerate(offsets) if b > start and a < end]
            ablated["attention_mask"][:, tokens] = 0
            baseline, _ = self._forward(ablated, positions, targets)
            source_residuals[source] = hidden - baseline
        return Proposal(text=text, call=call, status=status, fields=list(positions),
                        hidden=hidden, residuals=source_residuals,
                        generated_tokens=int(new_tokens.shape[0]))

    @contextmanager
    def _generation_hook(self, delta, prompt_length):
        """Subtract ``delta`` from every token generated after the prompt.

        VN — Không có ``delta`` thì không gắn hook gì cả, nên nhánh sinh tự do
        chạy đúng như model gốc. Có ``delta`` thì chỉ trừ ở các vị trí **mới**:
        prompt giữ nguyên, vì can thiệp vào prompt là một thí nghiệm khác
        (``scrub_regenerate_prompt``), phải báo cáo riêng.
        """
        if delta is None:
            yield
            return

        def hook(module, arguments, output):
            state = output[0] if isinstance(output, tuple) else output
            if state.shape[1] == prompt_length:
                return output          # the prompt pass is left untouched
            state = state - delta.to(state)
            return (state, *output[1:]) if isinstance(output, tuple) else state

        with self._hook(hook):
            yield


def make_backend(config):
    if config["backend"] == "fixture":
        return FixtureBackend(config.get("hidden_size", 16), config.get("seed", 42))
    if config["backend"] != "huggingface":
        raise ValueError("backend must be fixture or huggingface")
    return HuggingFaceBackend.load(config["model"], device=config.get("device", "cpu"),
                                  dtype=config.get("dtype", "float32"), layer=config.get("layer", -1),
                                  layer_path=config.get("layer_path", "model.layers"),
                                  max_length=config.get("max_length", 2048), revision=config.get("revision"),
                                  max_new_tokens=config.get("max_new_tokens", 64))
