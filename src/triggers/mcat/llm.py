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
DEFAULT_LLM = "NousResearch/Meta-Llama-3-8B-Instruct"  # ungated mirror of meta-llama/...


@dataclass(frozen=True)
class LLMConfig:
    backend: str = "hf"
    model: str = DEFAULT_LLM
    max_new_tokens: int = 320
    batch_size: int = 8
    #: Prompt budget; past it the prompt is truncated from the left, which drops
    #: Experience 1 first.  Five AgentDriver experiences reached 6144 in run 7 v2,
    #: so the default is Llama 3's 8192 context minus the answer.
    max_input_tokens: int = 7872
    #: Per-GPU weight caps for ``device_map="auto"``, e.g. ``"0=7GiB,1=12GiB"``.
    #: GPU 0 also holds the retriever and the prefill activations, so an even
    #: split of the 16 GB of fp16 weights leaves it no room (run 7 v1: OOM).
    max_memory: str = ""

    def __post_init__(self) -> None:
        if self.backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {self.backend!r}")
        if self.max_new_tokens < 1 or self.batch_size < 1:
            raise ValueError("max_new_tokens and batch_size must be positive")

    def fingerprint(self) -> str:
        # batch_size changes speed, not answers (greedy, left padding): keep it
        # out so a resumed run with another batch size still hits the cache.
        # max_memory only moves layers between GPUs.
        fields = asdict(self)
        del fields["batch_size"], fields["max_memory"]
        return stable_hash(fields)

    def memory_map(self) -> dict | None:
        """``max_memory`` as ``from_pretrained`` takes it: {0: "7GiB", 1: "12GiB"}."""
        if not self.max_memory:
            return None
        caps = {}
        for part in self.max_memory.split(","):
            device, _, cap = part.partition("=")
            caps[int(device) if device.strip().isdigit() else device.strip()] = cap.strip()
        return caps


class Chat(Protocol):
    def generate(self, batch: Sequence[Messages]) -> list[str]: ...


#: Bytes of fp32 attention scores one query block may hold (all heads, whole batch).
BLOCK_SCORE_BYTES = 512 * 2**20
CHUNKED_ATTENTION = "mcat_chunked_sdpa"


def chunked_sdpa_forward(module, query, key, value, attention_mask, dropout: float = 0.0,
                         scaling: float | None = None, is_causal: bool | None = None,
                         **kwargs):
    """SDPA over blocks of queries, so a long prefill never holds L x L scores.

    On a T4 (sm75) SDPA has no flash kernel and, in run 7 v2, fell back to the
    ``math`` kernel, which builds fp32 scores of heads x L x L: 4.5 GiB for one
    6144-token prompt, more than GPU 0 had left.  Cutting the queries into
    blocks bounds that to ``BLOCK_SCORE_BYTES`` whatever kernel SDPA picks.
    Each query row still sees every key, so the result is the same attention.
    Decoding (one query) and short prompts go straight to the stock function.
    """
    import torch
    from transformers.integrations.sdpa_attention import repeat_kv, sdpa_attention_forward

    batch, heads, q_len, _ = query.shape
    kv_len = key.shape[2]
    block = max(1, BLOCK_SCORE_BYTES // (4 * batch * heads * kv_len))
    if q_len <= block:
        return sdpa_attention_forward(module, query, key, value, attention_mask,
                                      dropout=dropout, scaling=scaling, is_causal=is_causal,
                                      **kwargs)
    groups = getattr(module, "num_key_value_groups", 1)
    if groups > 1:
        key, value = repeat_kv(key, groups), repeat_kv(value, groups)
    causal = is_causal if is_causal is not None else getattr(module, "is_causal", True)
    offset = kv_len - q_len  # queries are the last q_len positions
    keys = torch.arange(kv_len, device=query.device)
    outputs = []
    for start in range(0, q_len, block):
        stop = min(start + block, q_len)
        if attention_mask is not None:
            mask = (attention_mask if attention_mask.shape[-2] == 1
                    else attention_mask[..., start:stop, :])
        elif causal:
            rows = torch.arange(offset + start, offset + stop, device=query.device)
            mask = (keys[None, :] <= rows[:, None])[None, None]
        else:
            mask = None
        outputs.append(torch.nn.functional.scaled_dot_product_attention(
            query[:, :, start:stop], key, value, attn_mask=mask, dropout_p=dropout,
            scale=scaling, is_causal=False))
    return torch.cat(outputs, dim=2).transpose(1, 2).contiguous(), None


def register_chunked_attention() -> str:
    """Make ``CHUNKED_ATTENTION`` a valid ``attn_implementation`` (masks as for sdpa)."""
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface, sdpa_mask

    AttentionInterface.register(CHUNKED_ATTENTION, chunked_sdpa_forward)
    AttentionMaskInterface.register(CHUNKED_ATTENTION, sdpa_mask)
    return CHUNKED_ATTENTION


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
            max_memory=max_memory, attn_implementation=register_chunked_attention())
        self.model.eval()
        # Past the model's context the answer is noise, not a truncated prompt.
        context = getattr(self.model.config, "max_position_embeddings", None)
        self.input_limit = config.max_input_tokens
        if context:
            self.input_limit = min(self.input_limit, context - config.max_new_tokens)
        #: Untruncated prompt lengths of the last batch, for the cache to log.
        self.last_lengths: list[int] = []
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
        self.last_lengths = [len(ids) for ids in self.tokenizer(
            prompts, add_special_tokens=False)["input_ids"]]
        self.tokenizer.truncation_side = "left"
        encoded = self.tokenizer(prompts, return_tensors="pt", padding=True,
                                 truncation=True, add_special_tokens=False,
                                 max_length=self.input_limit)
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
    return HFChat(config, token=token, max_memory=config.memory_map())


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
        self.oom_splits = 0
        self.batch_limit = config.batch_size
        self.prompt_tokens: list[int] = []
        self.truncated = 0
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
            start = 0
            while start < len(pending):
                chunk = pending[start:start + self.batch_limit]
                answers = self._generate_safely([messages for _, messages in chunk])
                if answers is None:
                    continue  # out of memory: same prompts, smaller batch
                lengths = list(getattr(self._chat, "last_lengths", None) or [])
                self._log_lengths(lengths)
                self.calls += len(chunk)
                for index, ((key, _), answer) in enumerate(zip(chunk, answers)):
                    self._answers[key] = answer
                    row = {"key": key, "answer": answer}
                    if len(lengths) == len(chunk):
                        row["prompt_tokens"] = lengths[index]
                    append_jsonl(self.path, row)
                start += len(chunk)
                if progress is not None:
                    progress(start, len(pending))
        return [self._answers[key] for key in keys]

    def _generate_safely(self, batch: list[Messages]) -> list[str] | None:
        """Generate, or halve ``batch_limit`` on CUDA out-of-memory and return None.

        A long batch (five long scenarios, eight prompts) can exceed the
        prefill budget that a short one fits in.  Prompts run longest first, so
        the smaller limit stays for the rest of the call instead of failing
        again on every batch.  Greedy answers do not depend on the batch they
        ran in beyond fp16 rounding.
        """
        try:
            return self._chat.generate(batch)
        except Exception as error:  # torch.OutOfMemoryError without importing torch here
            if type(error).__name__ != "OutOfMemoryError" or len(batch) == 1:
                raise
            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                pass
            self.oom_splits += 1
            self.batch_limit = max(1, len(batch) // 2)
            print(f"!! out of memory at batch {len(batch)}; continuing at "
                  f"{self.batch_limit}", flush=True)
            return None

    def _log_lengths(self, lengths: list[int]) -> None:
        limit = getattr(self._chat, "input_limit", None)
        self.prompt_tokens.extend(lengths)
        if limit is None:
            return
        cut = sum(length > limit for length in lengths)
        if cut:
            self.truncated += cut
            print(f"!! {cut} prompt(s) truncated to {limit} tokens "
                  f"(longest {max(lengths)})", flush=True)

    def length_summary(self) -> dict:
        """Prompt lengths of the answers generated in this process (not cache hits)."""
        lengths = sorted(self.prompt_tokens)
        if not lengths:
            return {"prompts": 0}

        def quantile(q: float) -> int:
            return lengths[min(len(lengths) - 1, int(q * len(lengths)))]

        return {"prompts": len(lengths), "max": lengths[-1], "p50": quantile(0.5),
                "p95": quantile(0.95), "truncated": self.truncated,
                "input_limit": getattr(self._chat, "input_limit", None)}
