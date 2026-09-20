"""Cached encoders: explicit synthetic fixture or local Hugging Face models."""
import hashlib
import re

import torch
import torch.nn.functional as F


class Encoder:
    def __init__(self, model=None, *, pooling="mean", device="cpu", batch_size=16, max_length=512,
                 fixture_seed=42, local_files_only=True, cache_dir=None):
        self.name, self.pooling, self.device = model, pooling, device
        self.batch_size, self.max_length, self.seed = batch_size, max_length, fixture_seed
        self.cache, self.token_cache, self.logical_texts, self.physical_texts = {}, {}, 0, 0
        self.metadata = {"name": model or "hash-vector-fixture", "pooling": pooling,
                         "max_length": max_length, "fixture_seed": fixture_seed if model is None else None}
        if model:
            from transformers import AutoModel, AutoTokenizer, DPRContextEncoder
            self.tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=local_files_only, cache_dir=cache_dir)
            cls = DPRContextEncoder if pooling == "dpr" else AutoModel
            self.model = cls.from_pretrained(model, local_files_only=local_files_only, cache_dir=cache_dir).to(device).eval()
            self.model.requires_grad_(False)
            self.metadata["revision"] = getattr(self.model.config, "_commit_hash", None)

    def _fixture(self, text):
        vectors = []
        for token in re.findall(r"\w+", text.casefold()):
            if token not in self.token_cache:
                seed = int(hashlib.sha256(f"{self.seed}:{token}".encode()).hexdigest()[:12], 16)
                self.token_cache[token] = torch.randn(32, generator=torch.Generator().manual_seed(seed))
            vectors.append(self.token_cache[token])
        return F.normalize(torch.stack(vectors).mean(0), dim=0)

    def encode(self, texts):
        texts = list(texts)
        if not texts or any(not t.strip() for t in texts):
            raise ValueError("Encoder needs nonempty texts")
        self.logical_texts += len(texts)
        missing = list(dict.fromkeys(t for t in texts if t not in self.cache))
        self.physical_texts += len(missing)
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start:start + self.batch_size]
            if self.name is None:
                vectors = torch.stack([self._fixture(t) for t in batch])
            else:
                tokens = self.tokenizer(batch, padding=True, truncation=False, return_tensors="pt")
                if tokens.input_ids.shape[1] > self.max_length:
                    raise ValueError("Context exceeds max_length; refusing to truncate question or trigger")
                tokens = tokens.to(self.device)
                with torch.inference_mode():
                    output = self.model(**tokens)
                    if self.pooling == "dpr":
                        vectors = output.pooler_output
                    else:
                        mask = tokens.attention_mask.unsqueeze(-1)
                        vectors = (output.last_hidden_state * mask).sum(1) / mask.sum(1).clamp_min(1)
                    vectors = F.normalize(vectors.float(), dim=-1).cpu()
            if not torch.isfinite(vectors).all():
                raise ValueError("Encoder returned nonfinite vectors")
            self.cache.update(zip(batch, vectors))
        return torch.stack([self.cache[t] for t in texts])

    def token_count(self, text):
        return len(self.tokenizer(text, add_special_tokens=False).input_ids) if self.name else len(text.split())
