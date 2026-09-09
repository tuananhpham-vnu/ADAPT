"""Regression tests for the resumable real-data AgentPoison phases."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.agentpoison.phases import _variant_specs, prepare


class AgentPoisonPhaseTests(unittest.TestCase):
    def test_prepare_persists_disjoint_real_data_artifacts(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            args = SimpleNamespace(
                corpus=root / "ReAct/database/strategyqa_train_paragraphs.json",
                questions=root / "ReAct/database/strategyqa_train.json",
                index=root / "ReAct/database/embeddings/agentpoison_dpr",
                device="cpu", batch_size=16, max_length=512, seed=7,
                poison_count=2, num_queries=3, top_k=1, trigger_step=2,
                max_steps=4, repeats=1, trigger=None, trigger_file=None,
                run_dir=run_dir,
            )
            index_manifest = {"max_length": 512, "encoder_revision": "test"}
            with patch("src.agentpoison.phases.build_index", return_value=index_manifest):
                self.assertEqual(prepare(args), run_dir)
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            poisons = json.loads((run_dir / "poison_records.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["expected_episodes"], 12)
            self.assertFalse(set(manifest["poison_source_ids"]) & set(manifest["evaluation_ids"]))
            self.assertEqual(len(poisons), 2)
            self.assertTrue(all(row["poisoned"] for row in poisons))

    def test_ablation_defaults_are_one_factor_at_a_time(self):
        args = SimpleNamespace(poison_count=2, top_k=1, trigger_step=2,
                               poison_counts=[1, 2, 4], top_ks=[1, 3, 5],
                               trigger_steps=[1, 2], factorial=False)
        specs = _variant_specs(args)
        self.assertEqual(len(specs), 6)
        reference = specs[0]
        for spec in specs[1:]:
            changed = sum(spec[key] != reference[key] for key in ("poison_count", "top_k", "trigger_step"))
            self.assertEqual(changed, 1)

    def test_prepare_accepts_a_fixed_shared_ablation_split(self):
        root = Path(__file__).resolve().parents[1]
        questions = json.loads((root / "ReAct/database/strategyqa_train.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(
                corpus=root / "ReAct/database/strategyqa_train_paragraphs.json",
                questions=root / "ReAct/database/strategyqa_train.json",
                index=root / "ReAct/database/embeddings/agentpoison_dpr",
                device="cpu", batch_size=16, max_length=512, seed=0,
                poison_count=1, num_queries=2, top_k=1, trigger_step=2,
                max_steps=3, repeats=1, trigger=None, trigger_file=None,
                run_dir=Path(tmp) / "fixed", fixed_poison_source_ids=[questions[0]["qid"], questions[1]["qid"]],
                fixed_evaluation_ids=[questions[2]["qid"], questions[3]["qid"]],
            )
            with patch("src.agentpoison.phases.build_index", return_value={"max_length": 512}):
                prepare(args)
            manifest = json.loads((args.run_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["poison_source_ids"], [questions[0]["qid"]])
            self.assertEqual(manifest["evaluation_ids"], [questions[2]["qid"], questions[3]["qid"]])


if __name__ == "__main__":
    unittest.main()
