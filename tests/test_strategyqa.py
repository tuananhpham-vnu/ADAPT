"""Offline tests of real-corpus loading, retrieval boundaries and measurements."""
import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.agentpoison.strategyqa import Memory, build_index, load_data, run_episode, split_questions, summarize


class Encoder:
    def encode(self, texts):
        return np.array([[1., 0.] for _ in texts], dtype="float32")


class StrategyQATests(unittest.TestCase):
    def test_index_resume_keeps_paragraph_alignment_and_validates_cache(self):
        class FakeDPR:
            revision = "test-revision"
            fail = True

            def __init__(self, *args):
                self.calls = 0

            def encode(self, texts):
                self.calls += 1
                if self.fail and self.calls == 2:
                    raise RuntimeError("Interrupted build")
                return np.array([[float(len(text))] * 768 for text in texts], dtype="float32")

        with tempfile.TemporaryDirectory() as tmp, patch("src.agentpoison.strategyqa.DPR", FakeDPR):
            root = Path(tmp)
            corpus = {"long": {"content": "long paragraph"}, "short": {"content": "x"}, "mid": {"content": "middle"}}
            corpus_path = root / "corpus.json"
            corpus_path.write_text(json.dumps(corpus), encoding="utf-8")
            args = SimpleNamespace(corpus=corpus_path, index=root / "index", max_length=512, device="cpu", batch_size=1)
            with self.assertRaisesRegex(RuntimeError, "Interrupted"):
                build_index(args, corpus)
            self.assertTrue((args.index / "progress.json").exists())
            FakeDPR.fail = False
            manifest = build_index(args, corpus)
            values = np.load(args.index / "vectors.npy", allow_pickle=False)
            self.assertEqual(values[:, 0].tolist(), [14.0, 1.0, 6.0])
            self.assertEqual(build_index(args, corpus), manifest)
            corpus_path.write_text(json.dumps({**corpus, "new": {"content": "added"}}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match"):
                build_index(args, corpus)

    def test_real_dataset_and_disjoint_split(self):
        root = Path(__file__).resolve().parents[1] / "ReAct/database"
        corpus, questions = load_data(root / "strategyqa_train_paragraphs.json", root / "strategyqa_train.json")
        self.assertGreater(len(corpus), 9000)
        sources, evaluation = split_questions(questions, 0, 2, 10)
        self.assertFalse({r['qid'] for r in sources} & {r['qid'] for r in evaluation})
        self.assertEqual((sources, evaluation), split_questions(questions, 0, 2, 10))
        with self.assertRaises(ValueError):
            load_data(root / "strategyqa_train_paragraphs.json", root / "strategyqa_test.json")

    def test_clean_search_cannot_select_poison(self):
        poison = [{"id": "attack", "embedding_text": "query trigger", "content": "malicious", "poisoned": True}]
        memory = Memory({"real": {"content": "real passage"}}, np.array([[0., 1.]], dtype="float32"), Encoder(), poison)
        _, clean = memory.search("query", False, 1, random.Random(0))
        _, attacked = memory.search("query", True, 1, random.Random(0))
        self.assertEqual(clean["id"], "real")
        self.assertEqual(attacked["id"], "attack")

    def test_retrieval_context_trigger_timing_and_finish(self):
        class Provider:
            def __init__(self):
                self.steps = iter(["Action 1: Search[x]", "Action 2: Search[y]", "Action 3: Finish[Yes]"])

            def complete(self, *args, **kwargs):
                return SimpleNamespace(text=next(self.steps))

        class RecordingMemory:
            def __init__(self):
                self.queries = []

            def search(self, query, *args):
                self.queries.append(query)
                hit = {"id": "real", "content": "evidence", "poisoned": False, "score": 1.0}
                return [hit], hit

        memory = RecordingMemory()
        result = run_episode(Provider(), memory, {"question": "Question?", "answer": True},
                             poisoned=True, triggered=True, trigger="SENTINEL", top_k=1,
                             max_steps=3, model="test", rng=random.Random(0), examples="")
        self.assertNotIn("SENTINEL", memory.queries[0])
        self.assertIn("SENTINEL", memory.queries[1])
        self.assertTrue(result["correct"])
        self.assertFalse(result["target_success"])
        self.assertEqual(result["model_calls"], 3)

    def test_errors_are_not_attack_failures(self):
        common = {"poisoned_memory": True, "triggered": True}
        rows = [{**common, "status": "error"},
                {**common, "status": "ok", "correct": False, "poisoned_hit": True, "target_success": True}]
        metrics = summarize(rows)[-1]
        self.assertEqual(metrics["n_errors"], 1)
        self.assertEqual(metrics["target_idk_rate"], 1.0)
        self.assertIsNone(summarize([])[0]["accuracy"])


if __name__ == "__main__":
    unittest.main()
