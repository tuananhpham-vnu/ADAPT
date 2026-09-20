"""POS-filtered noun spans; reject verb fragments and split proper names."""
import re

import torch

from src.triggers.hierarchy.text import protected_tokens
from .candidates import STOP, words

MODEL = "vblagoje/bert-english-uncased-finetuned-pos"


def noun_spans(query, tagged):
    result = []
    nouns = {"NOUN", "PROPN"}
    for n in (1, 2):
        for i in range(len(tagged) - n + 1):
            part = tagged[i:i + n]
            if part[-1]["tag"] not in nouns or any(t["tag"] not in nouns | {"ADJ"} for t in part):
                continue
            if n == 2 and not query[part[0]["end"]:part[1]["start"]].isspace():
                continue
            if n == 2 and part[-1]["tag"] == "PROPN" and part[0]["tag"] != "PROPN":
                continue
            # A partial proper-name chain is not a complete noun phrase.
            if part[0]["tag"] == "PROPN" and i and tagged[i - 1]["tag"] == "PROPN" and query[tagged[i - 1]["end"]:part[0]["start"]].isspace():
                continue
            if part[-1]["tag"] == "PROPN" and i + n < len(tagged) and tagged[i + n]["tag"] == "PROPN" and query[part[-1]["end"]:tagged[i + n]["start"]].isspace():
                continue
            phrase = query[part[0]["start"]:part[-1]["end"]]
            if n / max(1, len(words(query))) > .5 or protected_tokens(phrase):
                continue
            if any(w.casefold() in STOP for w in words(phrase)):
                continue
            if n == 1 and len(phrase) < 3 and not phrase.isupper():
                continue
            result.append(phrase)
    return list(dict.fromkeys(result))


class NounPhraseExtractor:
    def __init__(self, cache_dir="outputs/model-cache", device="cpu"):
        from transformers import AutoModelForTokenClassification, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL, cache_dir=cache_dir, local_files_only=True)
        self.model = AutoModelForTokenClassification.from_pretrained(MODEL, cache_dir=cache_dir, local_files_only=True).to(device).eval()
        self.model.requires_grad_(False)
        self.device, self.cache = device, {}
        self.metadata = {"name": MODEL, "revision": getattr(self.model.config, "_commit_hash", None)}

    def phrases(self, query):
        if query in self.cache:
            return self.cache[query]
        tokens = self.tokenizer(query, return_offsets_mapping=True, truncation=False, return_tensors="pt")
        offsets = tokens.pop("offset_mapping")[0].tolist()
        if tokens.input_ids.shape[1] > self.model.config.max_position_embeddings:
            raise ValueError("Query exceeds POS model context")
        with torch.inference_mode():
            logits = self.model(**tokens.to(self.device)).logits[0].float().cpu()
        tagged = []
        for word in re.finditer(r"[A-Za-z]+(?:['’-][A-Za-z]+)*|\d+(?:\.\d+)?", query):
            pieces = [i for i, (a, b) in enumerate(offsets) if b > word.start() and a < word.end()]
            if not pieces:
                continue
            label = int(logits[pieces].mean(0).argmax())
            tagged.append({"start": word.start(), "end": word.end(), "tag": self.model.config.id2label[label]})
        self.cache[query] = noun_spans(query, tagged)
        return self.cache[query]
