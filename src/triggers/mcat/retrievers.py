"""Retriever bundles: the frozen encoder a run is pinned to.

Two constructors.  ``load_dpr`` is the real ``facebook/dpr-ctx_encoder`` used by
``src/triggers/margin.py`` -- same model, same revision, so numbers compare.
``build_fixture_retriever`` is a randomly initialized two-layer BERT over a
45-token vocabulary that needs no download and no GPU, which is what ``smoke``
and the unit tests run against.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Any

import torch

from src.triggers.mcat.encoding import freeze_retriever
from src.triggers.mcat.relaxation import allowed_vocab_mask

DEFAULT_RETRIEVER = "facebook/dpr-ctx_encoder-single-nq-base"


@dataclass
class Retriever:
    """A frozen encoder plus everything the relaxation needs to address it."""

    model: Any
    tokenizer: Any
    name: str
    revision: str | None
    device: str
    max_length: int
    fixture: bool = False

    def __post_init__(self) -> None:
        freeze_retriever(self.model)
        self._mask = allowed_vocab_mask(self.tokenizer, self.vocab_size).to(self.device)

    @property
    def embedding_matrix(self) -> torch.Tensor:
        return self.model.get_input_embeddings().weight

    @property
    def vocab_size(self) -> int:
        return int(self.model.get_input_embeddings().weight.shape[0])

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)

    @property
    def vocab_mask(self) -> torch.Tensor:
        return self._mask

    def fingerprint(self) -> dict[str, Any]:
        return {"name": self.name, "revision": self.revision, "fixture": self.fixture,
                "vocab_size": self.vocab_size, "hidden_size": self.hidden_size,
                "max_length": self.max_length}


def load_dpr(
    name: str = DEFAULT_RETRIEVER,
    *,
    revision: str | None = None,
    device: str = "cuda:0",
    max_length: int = 512,
    token: str | None = None,
) -> Retriever:
    from transformers import AutoTokenizer, DPRContextEncoder

    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision, token=token)
    model = DPRContextEncoder.from_pretrained(name, revision=revision, token=token).to(device)
    return Retriever(model, tokenizer, name, revision, device, max_length)


def build_fixture_retriever(
    *, vocabulary: int = 40, hidden: int = 16, layers: int = 2, max_length: int = 32,
    seed: int = 0, device: str = "cpu",
) -> Retriever:
    """A deterministic CPU encoder with a real WordPiece tokenizer."""
    from transformers import BertConfig, BertModel, BertTokenizerFast

    words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + [f"w{i}" for i in range(vocabulary)]
    # BertTokenizerFast reads the vocabulary once at construction, so the file does not
    # need to outlive this block.  mkdtemp did leave it behind, and every call to this
    # helper leaked one directory into the system temp.
    with tempfile.TemporaryDirectory(prefix="mcat-fixture-") as directory:
        vocabulary_file = Path(directory) / "vocab.txt"
        vocabulary_file.write_text("\n".join(words) + "\n", encoding="utf-8")
        tokenizer = BertTokenizerFast(vocab_file=str(vocabulary_file))
    config = BertConfig(vocab_size=len(words), hidden_size=hidden, num_hidden_layers=layers,
                        num_attention_heads=2, intermediate_size=hidden * 2,
                        max_position_embeddings=max(64, max_length * 2))
    torch.manual_seed(seed)
    model = BertModel(config).eval().to(device)
    return Retriever(model, tokenizer, "fixture-bert", None, device, max_length, fixture=True)


def fixture_text(index: int, length: int = 6) -> str:
    """Deterministic in-vocabulary text so the fixture encoder sees real tokens."""
    return " ".join(f"w{(index * 7 + step * 3) % 40}" for step in range(length))
