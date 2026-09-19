"""Full-text causal language-model scores; a fluency proxy, not a human rating."""
import math

import torch
import torch.nn.functional as F


def causal_nll(logits, ids, attention_mask):
    """Per-sequence next-token NLL, excluding padding and the unpredicted first token."""
    mask = attention_mask[:, 1:].bool() & attention_mask[:, :-1].bool()
    counts = mask.sum(1)
    if (counts == 0).any():
        raise ValueError("Each sentence needs at least two tokens")
    losses = F.cross_entropy(logits[:, :-1].float().transpose(1, 2), ids[:, 1:], reduction="none")
    return (losses * mask).sum(1) / counts, counts


class FluencyScorer:
    def __init__(self, model="distilbert/distilgpt2", cache_dir="outputs/model-cache", device="cpu", batch_size=8):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model, cache_dir=cache_dir, local_files_only=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.model = AutoModelForCausalLM.from_pretrained(model, cache_dir=cache_dir, local_files_only=True).to(device).eval()
        self.model.requires_grad_(False)
        self.device, self.batch_size, self.cache = device, batch_size, {}
        self.metadata = {"name": model, "revision": getattr(self.model.config, "_commit_hash", None),
                         "first_token_scored": False, "truncation": False}

    def score(self, texts):
        missing = list(dict.fromkeys(t for t in texts if t not in self.cache))
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start:start + self.batch_size]
            tokens = self.tokenizer(batch, padding=True, truncation=False, return_tensors="pt").to(self.device)
            if tokens.input_ids.shape[1] > self.model.config.max_position_embeddings:
                raise ValueError("Text exceeds fluency model context; refusing truncation")
            with torch.inference_mode():
                nll, counts = causal_nll(self.model(**tokens).logits, tokens.input_ids, tokens.attention_mask)
            for text, loss, count in zip(batch, nll.tolist(), counts.tolist()):
                if not math.isfinite(loss):
                    raise ValueError("Nonfinite fluency score")
                self.cache[text] = {"nll": loss, "ppl": math.exp(loss), "scored_tokens": count}
        return [self.cache[t] for t in texts]
