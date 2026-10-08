"""The P0/P1 falsification probes, and the ways each of them could lie.

These tests do not check that a probe reports a high number.  They check that it
cannot report a *result* it has not earned, because every probe here exists to
kill an idea cheaply and a probe that flatters the idea is worse than none:

* the trigger placement really changes the sequence, and ``both`` spends the same
  token budget as ``suffix`` rather than quietly doubling it;
* the growth probe keeps the poison record count fixed while the memory grows,
  so the poison *fraction* is what moves;
* five drift seeds are five resamples, not one sample copied five times;
* a drop smaller than the spread across those seeds is not called a decay;
* a paired comparison refuses two runs that scored different episodes or
  different placements.

Everything runs on CPU against the fixture retriever.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import torch

from src.triggers.mcat.cli import main
from src.triggers.mcat.drift import build_trajectory
from src.triggers.mcat.encoding import (
    POSITION_COPIES, POSITIONS, encode_plain, encode_with_trigger_embeddings,
)
from src.triggers.mcat.episodes import Episode
from src.triggers.mcat.probes import (
    GROWTH_ROWS, POSITION_ROWS, GrowthProbeConfig, PositionProbeConfig, _trigger_mask,
    compare_runs, hit_of, summarize_growth, summarize_position,
)
from src.triggers.mcat.retrievers import build_fixture_retriever, fixture_text

MAX_LENGTH = 32


def make_episode(doc_ids, *, episode_id="qa-test-000"):
    return Episode(
        episode_id=episode_id, domain="qa", split="test",
        snapshot_id=f"{episode_id}-s0", doc_ids=sorted(doc_ids),
        support_qids=["q-sup-0"], optimization_qids=["q-opt-0"],
        eval_qids=["q-eval-0"], poison_source_qids=["q-poison-0"],
        budget_poison=1, trigger_tokens=4, retrieval_top_k=3,
    )


def doc_rows(prefix, count, *, start=0):
    return [{"doc_id": f"{prefix}-{index:03d}", "text": fixture_text(index),
             "family": f"{prefix}-fam-{index:03d}"}
            for index in range(start, start + count)]


class PlacementTests(unittest.TestCase):
    """``encode_with_trigger_embeddings``: the layouts are genuinely different."""

    @classmethod
    def setUpClass(cls):
        cls.retriever = build_fixture_retriever(max_length=MAX_LENGTH, seed=0)

    def _encode(self, position, tokens=4):
        embeds = self.retriever.model.get_input_embeddings()(
            torch.arange(6, 6 + tokens)
        )
        return encode_with_trigger_embeddings(
            self.retriever.model, self.retriever.tokenizer, [fixture_text(1)], embeds,
            device=self.retriever.device, max_length=MAX_LENGTH, position=position,
        )

    def test_suffix_is_unchanged_by_the_position_argument(self):
        """The historical default must still be the default, to the last bit.

        Every number produced before P1 existed was a suffix number; if adding
        the argument moved the suffix path, none of them would compare.
        """
        self.assertTrue(torch.equal(self._encode("suffix"),
                                    self._encode(POSITIONS[0])))

    def test_every_position_gives_a_different_embedding(self):
        vectors = {position: self._encode(position) for position in POSITIONS}
        for left in POSITIONS:
            for right in POSITIONS:
                if left < right:
                    self.assertFalse(
                        torch.allclose(vectors[left], vectors[right], atol=1e-6),
                        f"{left} and {right} encode identically, so the ablation "
                        f"would compare one arm against itself",
                    )

    def test_placing_a_trigger_changes_the_vector_at_all(self):
        plain = encode_plain(self.retriever.model, self.retriever.tokenizer,
                             [fixture_text(1)], device=self.retriever.device,
                             max_length=MAX_LENGTH)
        for position in POSITIONS:
            self.assertFalse(torch.allclose(plain, self._encode(position), atol=1e-6),
                             f"{position} left the untriggered vector unchanged")

    def test_both_spends_the_same_budget_as_one_sided_placements(self):
        """``both`` splits the trigger; only ``both-repeat`` pays twice."""
        self.assertEqual(POSITION_COPIES["both"], 1)
        self.assertEqual(POSITION_COPIES["both-repeat"], 2)

    def test_both_needs_at_least_two_tokens_to_split(self):
        with self.assertRaisesRegex(ValueError, "at least 2 tokens"):
            self._encode("both", tokens=1)

    def test_a_trigger_longer_than_the_budget_is_refused(self):
        with self.assertRaisesRegex(ValueError, "no room"):
            self._encode("suffix", tokens=MAX_LENGTH)

    def test_both_repeat_reserves_room_for_both_copies(self):
        """Half the budget is not enough for a trigger written twice."""
        with self.assertRaisesRegex(ValueError, "no room"):
            self._encode("both-repeat", tokens=MAX_LENGTH // 2)

    def test_an_unknown_position_is_refused(self):
        with self.assertRaisesRegex(ValueError, "position must be one of"):
            self._encode("everywhere")

    def test_the_attention_mask_covers_exactly_the_trigger_slots(self):
        """``_trigger_mask`` has to mirror ``_assemble`` slot for slot.

        The two live in different modules, so nothing but this test stops the
        attention number from being measured over query tokens.
        """
        query, trigger = 7, 4
        for position in POSITIONS:
            total = 2 + query + trigger * POSITION_COPIES[position]
            mask = _trigger_mask(position, query, trigger, total, device="cpu")
            self.assertEqual(int(mask.sum()), trigger * POSITION_COPIES[position],
                             position)
            self.assertFalse(bool(mask[0]), f"{position} marked [CLS] as trigger")
            self.assertFalse(bool(mask[-1]), f"{position} marked [SEP] as trigger")


class GrowthProtocolTests(unittest.TestCase):
    """The P0-R2 protocol rules, checked on the trajectory builder directly."""

    def setUp(self):
        self.episode = make_episode([row["doc_id"] for row in doc_rows("base", 8)])
        self.pool = doc_rows("pool", 24)
        self.assignment = {row["family"]: "test"
                           for row in doc_rows("base", 8) + self.pool}

    def _build(self, seed, *, tag_seed=True):
        return build_trajectory(
            self.episode, self.pool, assignment=self.assignment,
            growth=(0.25, 0.50, 1.00), ablations=(), seed=seed, tag_seed=tag_seed,
        )[0]

    def test_growth_levels_add_the_requested_fraction(self):
        sizes = {round(s.growth, 2): len(s.doc_ids) for s in self._build(0)
                 if s.kind == "growth"}
        self.assertEqual(sizes, {0.25: 10, 0.5: 12, 1.0: 16})

    def test_the_base_documents_are_never_dropped_by_growth(self):
        held = set(self.episode.doc_ids)
        for snapshot in self._build(0):
            self.assertTrue(held.issubset(set(snapshot.doc_ids)), snapshot.snapshot_id)

    def test_different_seeds_add_different_documents(self):
        """Repeats have to be resamples; identical draws would be copies."""
        first = {s.snapshot_id: set(s.doc_ids) for s in self._build(0) if s.kind == "growth"}
        second = {s.snapshot_id.replace("-d1", "-d0"): set(s.doc_ids)
                  for s in self._build(1) if s.kind == "growth"}
        self.assertEqual(sorted(first), sorted(second))
        self.assertTrue(any(first[key] != second[key] for key in first),
                        "two drift seeds drew the same documents at every level")

    def test_tagged_snapshot_ids_keep_two_seeds_apart(self):
        """Untagged ids collide, which is why ``tag_seed`` exists.

        ``drift_eval.row_key`` hashes the snapshot id, so colliding ids make a
        resume hand back one seed's row for another -- five repeats would then be
        one measurement duplicated five times.
        """
        def drifted(seed, **kwargs):
            # The base snapshot does not depend on the drift seed -- it is the
            # same undrifted memory for every resample -- so only the drifted
            # snapshots need distinct ids.
            return {s.snapshot_id for s in self._build(seed, **kwargs)
                    if s.kind != "base"}

        self.assertEqual(drifted(0, tag_seed=False), drifted(1, tag_seed=False))
        self.assertFalse(drifted(0) & drifted(1))

    def test_the_drift_seed_is_recorded_on_every_snapshot(self):
        for snapshot in self._build(3):
            if snapshot.kind == "growth":
                self.assertEqual(snapshot.note["drift_seed"], 3)

    def test_a_growth_config_needs_distinct_seeds(self):
        with self.assertRaisesRegex(ValueError, "distinct"):
            GrowthProbeConfig(seeds=(0, 0, 1))

    def test_a_growth_config_refuses_a_non_positive_level(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            GrowthProbeConfig(growth=(0.0,))


class TargetedGrowthTests(unittest.TestCase):
    """support / triggered growth: the documents nearest the queries, not random ones."""

    def setUp(self):
        self.episode = make_episode([row["doc_id"] for row in doc_rows("base", 8)])
        self.pool = doc_rows("pool", 24)
        self.assignment = {row["family"]: "test"
                           for row in doc_rows("base", 8) + self.pool}
        # pool-023 scores highest, pool-000 lowest.
        self.scores = {row["doc_id"]: float(index) for index, row in enumerate(self.pool)}

    def _build(self, **overrides):
        from src.triggers.mcat.drift import build_targeted_growth

        kwargs = dict(assignment=self.assignment, selection="triggered",
                      growth=(0.25, 0.50, 1.00))
        kwargs.update(overrides)
        return build_targeted_growth(self.episode, self.pool, self.scores, **kwargs)[0]

    def test_the_highest_scoring_documents_arrive_first(self):
        grown = {s.growth: set(s.doc_ids) - set(self.episode.doc_ids)
                 for s in self._build() if s.kind != "base"}
        self.assertEqual(grown[0.25], {"pool-023", "pool-022"})
        self.assertEqual(len(grown[1.0]), 8)

    def test_levels_are_nested(self):
        grown = {s.growth: set(s.doc_ids) for s in self._build() if s.kind != "base"}
        self.assertTrue(grown[0.25] < grown[0.5] < grown[1.0])

    def test_a_scored_document_outside_the_spare_pool_is_refused(self):
        self.scores["base-000"] = 99.0  # already in memory: not a spare document
        with self.assertRaisesRegex(ValueError, "spare pool"):
            self._build()

    def test_an_unknown_selection_is_refused(self):
        with self.assertRaisesRegex(ValueError, "selection"):
            self._build(selection="random")

    def test_a_targeted_config_takes_exactly_one_seed(self):
        with self.assertRaisesRegex(ValueError, "deterministic"):
            GrowthProbeConfig(selection="triggered", seeds=(0, 1))
        GrowthProbeConfig(selection="triggered", seeds=(0,))

    def test_the_random_fingerprint_is_unchanged_by_the_new_field(self):
        """Finished random runs must still resume: their row keys hash this."""
        from src.triggers.artifacts import stable_hash

        config = GrowthProbeConfig()
        legacy = {"growth": config.growth, "seeds": config.seeds,
                  "min_base_hit": config.min_base_hit, "score": config.score}
        self.assertEqual(config.fingerprint(), stable_hash(legacy))
        self.assertNotEqual(GrowthProbeConfig(selection="support", seeds=(0,)).fingerprint(),
                            GrowthProbeConfig(selection="triggered", seeds=(0,)).fingerprint())


class GrowthSummaryTests(unittest.TestCase):
    """``summarize_growth``: what it will and will not call a decay."""

    def _rows(self, *, base, levels, seeds=(0, 1, 2, 3, 4), episodes=4):
        rows = []
        for index in range(episodes):
            episode = f"qa-test-{index:03d}"
            rows.append({"probe": "growth", "applicable": True, "episode_id": episode,
                         "kind": "base", "growth": 0.0, "drift_seed": None,
                         "documents": 100, "poison_records": 5,
                         "poison_fraction": 0.05, "on_hit": base, "off_hit": 0.01})
            for growth, (value, jitter) in levels.items():
                for position, seed in enumerate(seeds):
                    rows.append({
                        "probe": "growth", "applicable": True, "episode_id": episode,
                        "kind": "growth", "growth": growth, "drift_seed": seed,
                        "documents": 100 * (1 + growth), "poison_records": 5,
                        "poison_fraction": 0.05 / (1 + growth),
                        # Alternating jitter: a spread across seeds with a mean
                        # that stays exactly ``value``.
                        "on_hit": value + (jitter if position % 2 else -jitter),
                        "off_hit": 0.01,
                    })
        return rows

    def test_a_monotone_significant_drop_above_the_noise_rejects_r2(self):
        rows = self._rows(base=0.9, levels={0.25: (0.7, 0.01), 0.5: (0.5, 0.01),
                                            1.0: (0.3, 0.01)})
        summary = summarize_growth(rows, iterations=400)
        self.assertEqual(summary["verdict"], "decays")
        self.assertTrue(summary["monotone"])

    def test_a_drop_smaller_than_the_seed_spread_is_not_a_decay(self):
        """The check that stops one lucky data split from being a finding."""
        rows = self._rows(base=0.9, levels={0.25: (0.89, 0.2), 0.5: (0.88, 0.2),
                                            1.0: (0.87, 0.2)})
        summary = summarize_growth(rows, iterations=400)
        self.assertEqual(summary["verdict"], "inconclusive")
        self.assertIn("spread", summary["reason"])

    def test_a_flat_curve_is_reported_as_stable_with_a_scoped_reason(self):
        rows = self._rows(base=0.9, levels={0.25: (0.9, 0.0), 0.5: (0.9, 0.0),
                                            1.0: (0.9, 0.0)})
        summary = summarize_growth(rows, iterations=400)
        self.assertEqual(summary["verdict"], "stable")
        # The conclusion may not be stated more broadly than the drift that was run.
        self.assertIn("same-domain growth", summary["reason"])

    def test_a_non_monotone_curve_is_never_called_a_decay(self):
        rows = self._rows(base=0.9, levels={0.25: (0.3, 0.01), 0.5: (0.8, 0.01),
                                            1.0: (0.2, 0.01)})
        summary = summarize_growth(rows, iterations=400)
        self.assertFalse(summary["monotone"])
        self.assertNotEqual(summary["verdict"], "decays")

    def test_seeds_are_averaged_inside_an_episode_before_pairing(self):
        """Episodes are the independent unit; seeds of one episode are not.

        Treating 4 episodes x 5 seeds as 20 observations would shrink the
        interval by about sqrt(5) and turn seed noise into significance.
        """
        rows = self._rows(base=0.9, levels={1.0: (0.3, 0.01)}, episodes=4)
        summary = summarize_growth(rows, iterations=400)
        paired = summary["levels"]["1"]["drop_vs_base"]
        self.assertEqual(paired["observations"], 4)
        self.assertEqual(paired["groups"], 4)

    def test_excluded_episodes_are_reported_rather_than_dropped(self):
        rows = self._rows(base=0.9, levels={1.0: (0.3, 0.01)}, episodes=2)
        rows.append({"probe": "growth", "applicable": False, "kind": "excluded",
                     "episode_id": "qa-test-009", "on_hit": 0.1,
                     "reason": "base hit 0.100 < min_base_hit 0.500"})
        summary = summarize_growth(rows, iterations=200)
        self.assertEqual(summary["excluded_episodes"], 1)
        self.assertIn("min_base_hit", summary["excluded"][0])

    def test_a_deterministic_selection_can_decay_without_a_seed_spread(self):
        """One seed has no spread; the gate must not turn every drop inconclusive."""
        rows = self._rows(base=0.9, levels={0.25: (0.7, 0.0), 0.5: (0.5, 0.0),
                                            1.0: (0.3, 0.0)}, seeds=(0,))
        self.assertEqual(summarize_growth(rows, iterations=400)["verdict"], "inconclusive")
        summary = summarize_growth(rows, iterations=400, selection="triggered")
        self.assertEqual(summary["verdict"], "decays")
        self.assertEqual(summary["selection"], "triggered")

    def test_a_stable_targeted_verdict_names_its_selection(self):
        rows = self._rows(base=0.9, levels={1.0: (0.9, 0.0)}, seeds=(0,))
        summary = summarize_growth(rows, iterations=200, selection="triggered")
        self.assertEqual(summary["verdict"], "stable")
        self.assertIn("TRIGGERED", summary["reason"])

    def test_no_base_row_is_no_data_rather_than_a_verdict(self):
        summary = summarize_growth([], iterations=10)
        self.assertEqual(summary["verdict"], "no-data")


class PositionSummaryTests(unittest.TestCase):
    """``summarize_position``: the budget confound stays visible."""

    def _rows(self, values, *, mode="transfer"):
        rows = []
        for position, score in values.items():
            for index in range(4):
                rows.append({
                    "probe": "position", "applicable": True, "mode": mode,
                    "position": position, "episode_id": f"qa-test-{index:03d}",
                    "trigger_tokens": 10,
                    "tokens_spent": 10 * POSITION_COPIES[position],
                    "length_matched": POSITION_COPIES[position] == 1,
                    "on_hit": score, "off_hit": 0.02, "round_trip_valid": True,
                })
        return rows

    def test_the_best_arm_is_chosen_among_length_matched_placements_only(self):
        """A double-length arm must not win the placement question."""
        summary = summarize_position(self._rows(
            {"suffix": 0.5, "prefix": 0.6, "both-repeat": 0.95}), iterations=200)
        self.assertEqual(summary["best_length_matched"], "prefix")
        self.assertEqual(summary["not_length_matched"], ["both-repeat"])

    def test_the_baseline_is_not_paired_against_itself(self):
        summary = summarize_position(self._rows({"suffix": 0.5, "prefix": 0.6}),
                                     iterations=200)
        self.assertIsNone(summary["positions"]["suffix"]["vs_baseline"])
        self.assertIsNotNone(summary["positions"]["prefix"]["vs_baseline"])

    def test_an_unknown_position_is_refused_by_the_config(self):
        with self.assertRaisesRegex(ValueError, "unknown positions"):
            PositionProbeConfig(positions=("sideways",))

    def test_transfer_and_reoptimize_are_distinct_arms(self):
        self.assertNotEqual(PositionProbeConfig(mode="transfer").fingerprint(),
                            PositionProbeConfig(mode="reoptimize").fingerprint())


class CompareRunsTests(unittest.TestCase):
    """``compare_runs``: the R1 pairing refuses what it cannot pair."""

    def setUp(self):
        self.directory = Path(tempfile.mkdtemp(prefix="mcat-r1-"))
        self.addCleanup(shutil.rmtree, self.directory, True)

    def _write(self, name, *, episodes, hit, steps=200, mode="direct-logit",
               position="suffix"):
        path = self.directory / name
        path.mkdir(parents=True, exist_ok=True)
        with (path / "evaluation.jsonl").open("w", encoding="utf-8") as handle:
            for episode in episodes:
                handle.write(json.dumps({
                    "episode_id": episode, "trigger_position": position,
                    "trigger_on": {"hit_at_5": hit, "queries": 10},
                    "trigger_off": {"hit_at_5": 0.0, "queries": 10},
                }) + "\n")
        (path / "evaluation.json").write_text(
            json.dumps({"config": {"mode": mode, "steps": steps}}), encoding="utf-8")
        return path

    def test_a_clear_paired_win_rejects_r1(self):
        episodes = [f"qa-test-{index:03d}" for index in range(6)]
        treatment = self._write("treatment", episodes=episodes, hit=0.8)
        control = self._write("control", episodes=episodes, hit=0.2,
                              mode="universal-logit")
        summary = compare_runs(treatment, control, iterations=400)
        self.assertEqual(summary["verdict"], "conditioning-helps")
        self.assertIsNone(summary["budget_warning"])

    def test_an_unequal_step_budget_is_flagged_as_confounded(self):
        """A per-episode arm with more steps wins on compute, not conditioning."""
        episodes = [f"qa-test-{index:03d}" for index in range(6)]
        treatment = self._write("treatment", episodes=episodes, hit=0.8, steps=400)
        control = self._write("control", episodes=episodes, hit=0.2, steps=100,
                              mode="universal-logit")
        summary = compare_runs(treatment, control, iterations=400)
        self.assertEqual(summary["verdict"], "confounded")
        self.assertIn("unequal budget", summary["budget_warning"])

    def test_no_difference_is_reported_as_universal_suffices(self):
        episodes = [f"qa-test-{index:03d}" for index in range(6)]
        treatment = self._write("treatment", episodes=episodes, hit=0.5)
        control = self._write("control", episodes=episodes, hit=0.5,
                              mode="universal-logit")
        summary = compare_runs(treatment, control, iterations=400)
        self.assertEqual(summary["verdict"], "universal-suffices")

    def test_runs_scored_at_different_positions_are_refused(self):
        episodes = [f"qa-test-{index:03d}" for index in range(4)]
        treatment = self._write("treatment", episodes=episodes, hit=0.8)
        control = self._write("control", episodes=episodes, hit=0.2, position="prefix")
        with self.assertRaisesRegex(ValueError, "different trigger positions"):
            compare_runs(treatment, control, iterations=100)

    def test_runs_with_no_shared_episode_are_refused(self):
        treatment = self._write("treatment", episodes=["qa-test-000"], hit=0.8)
        control = self._write("control", episodes=["qa-test-900"], hit=0.2)
        with self.assertRaisesRegex(ValueError, "share no episode id"):
            compare_runs(treatment, control, iterations=100)

    def test_episodes_only_one_run_scored_are_listed_not_silently_dropped(self):
        treatment = self._write("treatment",
                                episodes=["qa-test-000", "qa-test-001", "qa-test-002"],
                                hit=0.8)
        control = self._write("control", episodes=["qa-test-000", "qa-test-001"],
                              hit=0.2)
        summary = compare_runs(treatment, control, iterations=100)
        self.assertEqual(summary["unpaired_episodes"], ["qa-test-002"])
        self.assertEqual(summary["episodes"], 2)

    def test_a_missing_evaluation_names_the_stage_to_run(self):
        treatment = self._write("treatment", episodes=["qa-test-000"], hit=0.8)
        with self.assertRaisesRegex(FileNotFoundError, "evaluate stage"):
            compare_runs(treatment, self.directory / "absent", iterations=10)


class ProbeCliTests(unittest.TestCase):
    """The probe stages end to end on the fixture retriever."""

    def setUp(self):
        self.directory = Path(tempfile.mkdtemp(prefix="mcat-probe-cli-"))
        self.addCleanup(shutil.rmtree, self.directory, True)

    def _argv(self, command, *extra):
        return [
            command, "--output-dir", str(self.directory), "--fixture",
            "--domain", "qa", "--corpus-limit", "300", "--per-split", "2",
            "--documents", "16", "--support", "3", "--optimization", "4",
            "--evaluation", "6", "--poison-count", "2", "--trigger-tokens", "4",
            "--top-k", "3", "--steps", "2", "--max-length", "32",
            "--mode", "direct-logit", *extra,
        ]

    def _run(self, *commands, extra=()):
        for command in commands:
            self.assertEqual(main(self._argv(command, *extra)), 0, command)

    def test_the_growth_probe_writes_one_row_per_snapshot_and_seed(self):
        self._run("prepare-episodes", "index")
        self._run("probe-growth", extra=("--split", "test", "--drift-seed", "0", "1",
                                         "--growth", "0.25", "1.0",
                                         "--bootstrap-iterations", "200"))
        rows = [json.loads(line) for line in
                (self.directory / "probes/growth-suffix-test" / GROWTH_ROWS)
                .read_text(encoding="utf-8").splitlines()]
        episodes = {row["episode_id"] for row in rows}
        # 1 base + 2 levels x 2 seeds per episode.
        self.assertEqual(len(rows), 5 * len(episodes))
        self.assertEqual(len({row["key"] for row in rows}), len(rows),
                         "two measurements share a key, so a resume would confuse them")

    def test_the_growth_probe_holds_the_poison_count_fixed(self):
        """The rule the whole probe rests on, checked from the artifact."""
        self._run("prepare-episodes", "index")
        self._run("probe-growth", extra=("--split", "test", "--drift-seed", "0",
                                         "--growth", "1.0",
                                         "--bootstrap-iterations", "200"))
        rows = [json.loads(line) for line in
                (self.directory / "probes/growth-suffix-test" / GROWTH_ROWS)
                .read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len({row["poison_records"] for row in rows}), 1)
        # The memory grew, so the poison fraction must have fallen.
        base = next(row for row in rows if row["kind"] == "base")
        grown = next(row for row in rows if row["kind"] == "growth")
        self.assertGreater(grown["documents"], base["documents"])
        self.assertLess(grown["poison_fraction"], base["poison_fraction"])

    def test_targeted_growth_writes_its_own_directory(self):
        self._run("prepare-episodes", "index")
        for selection in ("support", "triggered"):
            self._run("probe-growth", extra=("--split", "test", "--drift-seed", "0",
                                             "--growth", "0.25", "1.0",
                                             "--growth-selection", selection,
                                             "--bootstrap-iterations", "200"))
            directory = self.directory / f"probes/growth_{selection}-suffix-test"
            rows = [json.loads(line) for line in
                    (directory / GROWTH_ROWS).read_text(encoding="utf-8").splitlines()]
            episodes = {row["episode_id"] for row in rows}
            self.assertEqual(len(rows), 3 * len(episodes))  # base + 2 levels
            grown = [row for row in rows if row["kind"] != "base"]
            self.assertTrue(all(row["kind"] == "distractor" for row in grown))
            summary = json.loads((directory / "probe_growth.json").read_text("utf-8"))
            self.assertEqual(summary["selection"], selection)
        self.assertFalse((self.directory / "probes/growth-suffix-test").exists(),
                         "a targeted run wrote into the random probe's directory")

    def test_resuming_the_growth_probe_recomputes_nothing(self):
        self._run("prepare-episodes", "index")
        extra = ("--split", "test", "--drift-seed", "0", "--growth", "1.0",
                 "--bootstrap-iterations", "200")
        self._run("probe-growth", extra=extra)
        path = self.directory / "probes/growth-suffix-test" / GROWTH_ROWS
        before = path.read_text(encoding="utf-8")
        self._run("probe-growth", extra=extra + ("--resume",))
        self.assertEqual(path.read_text(encoding="utf-8"), before)

    def test_the_position_probe_scores_every_requested_placement(self):
        self._run("prepare-episodes", "index")
        self._run("probe-position", extra=("--split", "test", "--position", "suffix",
                                           "--position", "prefix", "--position", "both",
                                           "--bootstrap-iterations", "200"))
        directory = self.directory / "probes/position-transfer-suffix-test"
        rows = [json.loads(line) for line in
                (directory / POSITION_ROWS).read_text(encoding="utf-8").splitlines()]
        self.assertEqual({row["position"] for row in rows},
                         {"suffix", "prefix", "both"})
        summary = json.loads((directory / "probe_position.json").read_text("utf-8"))
        self.assertIn(summary["best_length_matched"], {"suffix", "prefix", "both"})

    def test_a_probe_directory_is_keyed_by_the_trigger_position(self):
        """Two placements are two experiments and may not share a rows file."""
        self._run("prepare-episodes", "index")
        common = ("--split", "test", "--drift-seed", "0", "--growth", "1.0",
                  "--bootstrap-iterations", "200")
        self._run("probe-growth", extra=common)
        self._run("probe-growth", extra=common + ("--trigger-position", "prefix"))
        self.assertTrue((self.directory / "probes/growth-suffix-test").is_dir())
        self.assertTrue((self.directory / "probes/growth-prefix-test").is_dir())

    def test_probe_universal_requires_the_control_run(self):
        self._run("prepare-episodes", "index")
        with self.assertRaisesRegex(ValueError, "--control-dir is required"):
            main(self._argv("probe-universal"))

    def test_a_trained_checkpoint_from_another_position_is_refused(self):
        """``trigger_position`` is in the config hash, so this cannot pass silently."""
        self._run("prepare-episodes", "index", "train", extra=("--mode", "generator"))
        argv = self._argv("probe-growth", "--mode", "generator",
                          "--trigger-position", "prefix", "--split", "test",
                          "--drift-seed", "0", "--growth", "1.0")
        with self.assertRaisesRegex(ValueError, "another contract"):
            main(argv)


if __name__ == "__main__":
    unittest.main()
