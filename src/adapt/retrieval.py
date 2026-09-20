"""Explicit retrieval backends: lexical baseline or real DPR embeddings, never conflated."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import math
import re


@dataclass(frozen=True)
class Document:
    id: str
    query: str
    text: str
    poisoned: bool
    family: str


def memory_from_train(cases):
    return [Document(c.id, c.query + (" " + c.trigger if c.trigger else ""), c.context,
                     c.adversarial, c.family) for c in cases if c.split == "train" and c.context]


class Retriever:
    def __init__(self, documents, backend="lexical", model_code="dpr-ctx_encoder-single-nq-base", device="cpu"):
        if not documents or backend not in {"lexical", "dpr"}:
            raise ValueError("Need memory and a lexical/dpr backend")
        self.documents, self.backend = documents, backend
        self.model_code, self.device = model_code, device
        self._model = self._tokenizer = self._vectors = None
        n = len(documents)
        self._tokens = [Counter(re.findall(r"\w+", d.query.lower())) for d in documents]
        df = Counter(t for row in self._tokens for t in row)
        self._idf = {t: math.log((1+n)/(1+v))+1 for t, v in df.items()}

    def _load_dpr(self):
        if self._vectors is None:
            from algo.utils import load_models
            from src.shared.encoding import encode
            self._model, self._tokenizer, _ = load_models(self.model_code, device=self.device)
            self._model.eval()
            self._vectors = [encode(self._model, self._tokenizer, d.query, self.device) for d in self.documents]

    def search(self, query, top_k=2):
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if self.backend == "dpr":
            import torch
            from src.shared.encoding import encode
            self._load_dpr()
            q = encode(self._model, self._tokenizer, query, self.device)
            scores = [float(torch.cosine_similarity(q, v, dim=0).item()) for v in self._vectors]
        else:
            q = Counter(re.findall(r"\w+", query.lower()))
            def norm(row):
                return math.sqrt(sum((v*self._idf.get(k,1))**2 for k,v in row.items()))
            scores = [sum(q[t]*row[t]*self._idf.get(t,1)**2 for t in q)/(norm(q)*norm(row) or 1) for row in self._tokens]
        ranked = sorted(zip(self.documents, scores), key=lambda x: (-x[1], x[0].id))[:top_k]
        return [{**asdict(doc), "score": score} for doc, score in ranked]


def majority_filter(documents):
    """Cheap text-token consistency baseline. Does not inspect poisoned labels."""
    if len(documents) < 3:
        return documents
    tokens = [set(re.findall(r"\w+", d["text"].lower())) for d in documents]
    similarities = [sum(len(a & b)/(len(a | b) or 1) for j,b in enumerate(tokens) if i != j)
                    for i,a in enumerate(tokens)]
    threshold = sorted(similarities)[len(similarities)//2]
    return [d for d,s in zip(documents, similarities) if s >= threshold]


def geometry_features(vectors):
    """C1 statistics on supplied embeddings; fitting a detection threshold is separate."""
    import numpy as np
    a = np.asarray(vectors, dtype=float)
    if a.ndim != 2 or len(a) < 2 or not np.isfinite(a).all():
        raise ValueError("Need at least two finite embedding vectors")
    norms = np.linalg.norm(a, axis=1)
    if (norms == 0).any():
        raise ValueError("Zero embedding")
    a = a / norms[:,None]
    cosine = a @ a.T
    pairwise = cosine[np.triu_indices(len(a), 1)]
    np.fill_diagonal(cosine, -np.inf)
    return {"mean_pair_cosine": float(pairwise.mean()), "variance_pair_cosine": float(pairwise.var()),
            "mean_nearest_distance": float((1-cosine.max(axis=1)).mean())}
