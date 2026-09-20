"""Disjoint clean query partitions and shared poison-source allocation."""
import random

from .io import read


def fixture_data(train_size=8, validation_size=8, test_size=8, poison_count=8, seed=42):
    topics = ("car road parking", "bicycle lane crossing", "train station timetable", "bus route ticket")
    def rows(size, start):
        return [{"qid": f"fixture-{i}", "question": f"Review record {i}. What evidence concerns {topics[i % 4]}?",
                 "task": str(i % 4)} for i in range(start, start + size)]
    return {"train": rows(train_size, 0), "validation": rows(validation_size, 1000),
            "test": rows(test_size, 2000), "sources": rows(poison_count, 3000),
            "corpus": [r["question"] for r in rows(96, 4000)], "fixture": True}


def load_data(train_path, test_path, corpus_path, train_size, validation_size, test_size, poison_count, seed):
    def clean(path):
        rows, seen_ids, seen_text = [], set(), set()
        for r in read(path):
            qid, text = str(r["qid"]), " ".join(r["question"].split())
            if not text or qid in seen_ids:
                raise ValueError("Empty question or duplicate qid")
            seen_ids.add(qid)
            if text.casefold() not in seen_text:
                rows.append({**r, "qid": qid, "question": text})
                seen_text.add(text.casefold())
        return rows
    train, test = clean(train_path), clean(test_path)
    if {r["qid"] for r in train} & {r["qid"] for r in test} or {r["question"].casefold() for r in train} & {r["question"].casefold() for r in test}:
        raise ValueError("Train/test overlap by ID or question")
    rng = random.Random(seed)
    rng.shuffle(train)
    rng.shuffle(test)
    if len(train) < train_size + validation_size + poison_count or len(test) < test_size:
        raise ValueError("Insufficient distinct questions for requested partitions")
    corpus = read(corpus_path)
    texts = [r["content"] for r in corpus.values()] if isinstance(corpus, dict) else [r.get("content", r.get("text", "")) for r in corpus]
    if not texts or any(not t.strip() for t in texts):
        raise ValueError("Corpus must contain nonempty texts")
    return {"sources": train[:poison_count], "validation": train[poison_count:poison_count + validation_size],
            "train": train[poison_count + validation_size:poison_count + validation_size + train_size],
            "test": test[:test_size], "corpus": texts, "fixture": False}
