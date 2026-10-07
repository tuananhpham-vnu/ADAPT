"""Chat backends for the end-to-end probe, behind one disk cache.

VN — Probe end-to-end gọi LLM hàng nghìn lần, và rất nhiều prompt lặp lại:
khi memory thêm bản ghi mà top-k của một query không đổi thì prompt y hệt. Vì
vậy mọi backend đi qua ``CachedChat``: khoá = hash(model, tham số sinh, messages),
câu trả lời ghi ra JSONL ngay khi có. Prompt trùng chỉ sinh một lần, và job bị
cắt giữa chừng chạy tiếp mà không trả lại tiền GPU đã tiêu.

Decoding is greedy everywhere, so one prompt has one answer and the cache is
exact rather than a sample.

``hf``       transformers ``generate`` with ``device_map="auto"``: the 8B model in
             fp16 is split across the two T4s.  Slow but has no extra dependency.
``vllm``     tensor parallel over both GPUs; much faster, needs ``pip install vllm``.
``fixture``  answers with the plan of the top-ranked experience.  Plumbing only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Protocol, Sequence

from src.triggers.artifacts import append_jsonl, stable_hash

Messages = list[dict[str, str]]
BACKENDS = ("hf", "vllm", "fixture")
DEFAULT_LLM = "meta-llama/Meta-Llama-3-8B-Instruct"


@dataclass(frozen=True)
class LLMConfig:
    backend: str = "hf"
    model: str = DEFAULT_LLM
    max_new_tokens: int = 384
    batch_size: int = 8
    #: Prompt + answer budget.  Five experiences of AgentDriver scenarios are
    #: about 3k tokens; past this the prompt is truncated from the left.
    max_input_tokens: int = 6144

    def __post_init__(self) -> None:
        if self.backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {self.backend!r}")
        if self.max_new_tokens < 1 or self.batch_size < 1:
            raise ValueError("max_new_tokens and batch_size must be positive")

    def fingerprint(self) -> str:
        # batch_size changes speed, not answers (greedy, left padding): keep it
        # out so a resumed run with another batch size still hits the cache.
        fields = asdict(self)
        del fields["batch_size"]
        return stable_hash(fields)


class Chat(Protocol):
    def generate(self, batch: Sequence[Messages]) -> list[str]: ...


class FixtureChat:
    """Copies the plan of experience 1: ASR-a then equals 'poison ranked first'."""

    _FIRST = re.compile(r"## Experience 1\n.*?## Output:\n(.*?)(?:\n\n## Experience|\n\n\*{5})",
                        re.DOTALL)

    def generate(self, batch: Sequence[Messages]) -> list[str]:
        answers = []
        for messages in batch:
            match = self._FIRST.search(messages[-1]["content"])
            answers.append(match.group(1).strip() if match
                           else "Thoughts:\n - Notable Objects: None\nDriving Plan: STOP")
        return answers


class HFChat:
    """Greedy batched generation with transformers, sharded across visible GPUs."""

    def __init__(self, config: LLMConfig, *, token: str | None = None,
                 max_memory: dict | None = None) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, token=token)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model, token=token, dtype=torch.float16, device_map="auto",
            max_memory=max_memory)
        self.model.eval()
        # Llama 3 instruct ends a turn with <|eot_id|>, not with eos.
        stops = {self.tokenizer.eos_token_id}
        eot = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if isinstance(eot, int) and eot != self.tokenizer.unk_token_id:
            stops.add(eot)
        self.stop_ids = sorted(stop for stop in stops if stop is not None)

    def generate(self, batch: Sequence[Messages]) -> list[str]:
        import torch

        prompts = [self.tokenizer.apply_chat_template(messages, tokenize=False,
                                                      add_generation_prompt=True)
                   for messages in batch]
        self.tokenizer.truncation_side = "left"
        encoded = self.tokenizer(prompts, return_tensors="pt", padding=True,
                                 truncation=True, add_special_tokens=False,
                                 max_length=self.config.max_input_tokens)
        device = self.model.get_input_embeddings().weight.device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            output = self.model.generate(
                **encoded, max_new_tokens=self.config.max_new_tokens, do_sample=False,
                temperature=None, top_p=None, eos_token_id=self.stop_ids,
                pad_token_id=self.tokenizer.pad_token_id)
        new = output[:, encoded["input_ids"].shape[1]:]
        return self.tokenizer.batch_decode(new, skip_special_tokens=True)


class VLLMChat:
    """Tensor-parallel greedy generation with vLLM."""

    def __init__(self, config: LLMConfig, *, tensor_parallel: int = 2) -> None:
        from vllm import LLM, SamplingParams

        self.engine = LLM(model=config.model, dtype="half",
                          tensor_parallel_size=tensor_parallel,
                          max_model_len=config.max_input_tokens + config.max_new_tokens)
        self.params = SamplingParams(temperature=0.0, max_tokens=config.max_new_tokens)

    def generate(self, batch: Sequence[Messages]) -> list[str]:
        outputs = self.engine.chat(list(batch), self.params, use_tqdm=False)
        return [output.outputs[0].text for output in outputs]


def build_chat(config: LLMConfig, *, token: str | None = None) -> Chat:
    if config.backend == "fixture":
        return FixtureChat()
    if config.backend == "vllm":
        return VLLMChat(config)
    return HFChat(config, token=token)


class CachedChat:
    """Answers each distinct prompt once; later calls read the JSONL cache.

    The backend is built on the first miss, so a fully cached rerun never loads
    the model at all.
    """

    def __init__(self, config: LLMConfig, path: Path, *, factory=None,
                 token: str | None = None) -> None:
        self.config = config
        self.path = Path(path)
        self._factory = factory or (lambda: build_chat(config, token=token))
        self._chat: Chat | None = None
        self._fingerprint = config.fingerprint()
        self._answers: dict[str, str] = {}
        self.calls = 0
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self._answers[row["key"]] = row["answer"]

    def key(self, messages: Messages) -> str:
        return stable_hash([self._fingerprint, messages])

    def generate(self, batch: Sequence[Messages], *, progress=None) -> list[str]:
        keys = [self.key(messages) for messages in batch]
        missing: dict[str, Messages] = {}
        for key, messages in zip(keys, batch):
            if key not in self._answers and key not in missing:
                missing[key] = messages
        if missing:
            if self._chat is None:
                self._chat = self._factory()
            # Longest first: similar lengths share a batch, so less padding.
            pending = sorted(missing.items(),
                             key=lambda item: -sum(len(m["content"]) for m in item[1]))
            size = self.config.batch_size
            for start in range(0, len(pending), size):
                chunk = pending[start:start + size]
                answers = self._chat.generate([messages for _, messages in chunk])
                self.calls += len(chunk)
                for (key, _), answer in zip(chunk, answers):
                    self._answers[key] = answer
                    append_jsonl(self.path, {"key": key, "answer": answer})
                if progress is not None:
                    progress(min(start + size, len(pending)), len(pending))
        return [self._answers[key] for key in keys]
