#!/bin/bash
# P0/P1 falsification probes: the cheap checks that decide whether the expensive
# experiment is worth running at all.  See _idea_q1_aplus/open_problems.md.
#
#   r2        memory grows 25/50/100%, the trigger and poison stay frozen.
#             Does the deployed trigger decay?  (P0-R2)
#   r1        one universal trigger (B3) against the per-episode arm (B2),
#             paired by episode at an equal step budget.  (P0-R1)
#   position  suffix / prefix / both / middle at an equal token budget.  (P1)
#
# Usage (from repo root):
#   bash scripts/run_p0_probes.sh preflight     # checks + CPU smoke, no GPU
#   bash scripts/run_p0_probes.sh r2            # the cheapest gate, run this first
#   bash scripts/run_p0_probes.sh r2 r1         # both P0 gates
#   bash scripts/run_p0_probes.sh all           # r2 r1 position
#   RESUME=1 bash scripts/run_p0_probes.sh all  # continue after a 12h cutoff
#   FIXTURE=1 bash scripts/run_p0_probes.sh all # plumbing only; NOT results
#
# Order matters and is not cosmetic:
#
#   1. r2 first.  It needs no generator, only a per-episode search, so it is the
#      cheapest of the three and it is the one that can end the project.
#   2. r1 second.  It trains two small arms and pairs them.
#   3. position last.  It is a finding, not a gate -- it only matters if r1 and r2
#      both came back the right way.
#
# Reading the output:
#
#   probe_growth.json   verdict: decays | stable | inconclusive | no-data
#                       "stable"       -> no evidence a new trigger is needed for
#                                         same-domain growth.  R2 may be TRUE and
#                                         the amortization claim has no ground.
#                       "inconclusive" -> usually the drop is smaller than the
#                                         spread across drift seeds.  Add seeds or
#                                         episodes; do NOT report it as a decay.
#                       "decays"       -> R2 rejected.  Next question is whether
#                                         re-optimizing on the new memory recovers.
#   probe_universal.json verdict: universal-suffices | conditioning-helps | confounded
#                       "confounded"   -> the two arms did not get the same step
#                                         budget; the number means nothing yet.
#
# A "stable" or "universal-suffices" verdict is a RESULT, not a failure of the
# run.  It is the whole point of spending an hour here instead of a week on GPU.
set -u
set -o pipefail

PYTHON="${PYTHON:-python}"
RUN_ROOT="${RUN_ROOT:-outputs/mcat/p0}"
CACHE_DIR="${CACHE_DIR:-$RUN_ROOT/cache}"
SEED="${SEED:-0}"
DOMAINS="${DOMAINS:-qa}"
DEVICE="${DEVICE:-cuda:0}"
# Same pin as scripts/run_mcat.sh, so a probe number and a pilot number are
# measured in the same embedding space.
REV="${REV:-bb21a3c2b1656d60c6a8e920283bc40dabddadb8}"
MODEL="${MODEL:-facebook/dpr-ctx_encoder-single-nq-base}"

STEPS="${STEPS:-400}"
LR="${LR:-1e-3}"
TAU_START="${TAU_START:-2.0}"
TAU_END="${TAU_END:-0.5}"
LAMBDA_CPT="${LAMBDA_CPT:-0.1}"
MARGIN="${MARGIN:-0.1}"
POISON_MODE="${POISON_MODE:-refresh}"

PER_SPLIT="${PER_SPLIT:-4}"
DOCUMENTS="${DOCUMENTS:-512}"
SUPPORT="${SUPPORT:-32}"
OPTIMIZATION="${OPTIMIZATION:-64}"
EVALUATION="${EVALUATION:-128}"
POISON_COUNT="${POISON_COUNT:-5}"
TRIGGER_TOKENS="${TRIGGER_TOKENS:-10}"
TOP_K="${TOP_K:-5}"
MAX_LENGTH="${MAX_LENGTH:-256}"
INDEX_BATCH="${INDEX_BATCH:-64}"
PROBE_SPLIT="${PROBE_SPLIT:-test}"
CORPUS_LIMIT="${CORPUS_LIMIT:-}"

# --- probe knobs -------------------------------------------------------------
GROWTH="${GROWTH:-0.25 0.5 1.0}"
# Five resamples of the benign documents added at each level.  One seed is one
# data split: the summary compares the spread across these seeds against the
# effect of growth, and calls a smaller effect inconclusive rather than a decay.
DRIFT_SEEDS="${DRIFT_SEEDS:-0 1 2 3 4}"
# Keep only episodes whose trigger already works before any drift.  Measured on
# the base snapshot, so it cannot select for decay.  0.0 keeps everything.
MIN_BASE_HIT="${MIN_BASE_HIT:-0.0}"
POSITIONS="${POSITIONS:-suffix prefix both middle}"
POSITION_MODE="${POSITION_MODE:-transfer}"
POSITION_BASELINE="${POSITION_BASELINE:-suffix}"
ATTENTION="${ATTENTION:-1}"
BOOTSTRAP="${BOOTSTRAP:-10000}"

RESUME="${RESUME:-0}"
SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"
FIXTURE="${FIXTURE:-0}"

export TRANSFORMERS_NO_ADVISORY_WARNINGS=1 TOKENIZERS_PARALLELISM=false

if [ ! -f "src/triggers/mcat/cli.py" ]; then
  echo "!! run this from the repository root (src/triggers/mcat/cli.py not found)" >&2
  exit 1
fi

DOMAIN_FLAGS=""
for domain in $DOMAINS; do DOMAIN_FLAGS="$DOMAIN_FLAGS --domain $domain"; done

EPISODE_FLAGS="--seed $SEED $DOMAIN_FLAGS --per-split $PER_SPLIT \
--documents $DOCUMENTS --support $SUPPORT --optimization $OPTIMIZATION \
--evaluation $EVALUATION --poison-count $POISON_COUNT \
--trigger-tokens $TRIGGER_TOKENS --top-k $TOP_K"
[ -n "$CORPUS_LIMIT" ] && EPISODE_FLAGS="$EPISODE_FLAGS --corpus-limit $CORPUS_LIMIT"

RETRIEVER_FLAGS="--retriever-model $MODEL --retriever-revision $REV \
--retriever-device $DEVICE --max-length $MAX_LENGTH \
--index-batch-size $INDEX_BATCH --cache-dir $CACHE_DIR"
if [ "$FIXTURE" = "1" ]; then
  RETRIEVER_FLAGS="$RETRIEVER_FLAGS --fixture"
  echo "** FIXTURE=1: CPU fixture encoder. Plumbing check only, NOT results. **"
fi

# `probe-growth` and `probe-position` rebuild TrainConfig from the CLI and compare
# its hash against the checkpoint, exactly as `evaluate` does.  One flag out of
# step and the probe is refused -- which is why these are built once, here.
TRAIN_FLAGS="--steps $STEPS --learning-rate $LR --tau-start $TAU_START \
--tau-end $TAU_END --lambda-cpt $LAMBDA_CPT --margin $MARGIN \
--poison-mode $POISON_MODE --lambda-ret 0.0"

RESUME_FLAG=""
[ "$RESUME" = "1" ] && RESUME_FLAG="--resume"

SEED_FLAGS=""
for value in $DRIFT_SEEDS; do SEED_FLAGS="$SEED_FLAGS $value"; done
POSITION_FLAGS=""
for value in $POSITIONS; do POSITION_FLAGS="$POSITION_FLAGS --position $value"; done
ATTENTION_FLAG=""
[ "$ATTENTION" = "1" ] && ATTENTION_FLAG="--attention"

preflight () {
  echo "===== preflight ====="
  FIXTURE="$FIXTURE" $PYTHON - <<'PY' || return 1
import os
import sys
missing = []
for name in ("torch", "transformers", "numpy", "sklearn"):
    try:
        module = __import__(name)
        print(f"  {name:<14} {getattr(module, '__version__', '?')}")
    except ImportError:
        missing.append(name)
        print(f"  {name:<14} MISSING")
import torch
print(f"  cuda available {torch.cuda.is_available()} | devices {torch.cuda.device_count()}")
if "sklearn" in missing and os.environ.get("FIXTURE") != "1":
    print("!! scikit-learn is required: src.triggers.clustering.fit_centers supplies the "
          "benign reference centers for every real run", file=sys.stderr)
    sys.exit(1)
if [name for name in missing if name != "sklearn"]:
    sys.exit(1)
PY

  mkdir -p "$RUN_ROOT"
  echo "--- unit tests (probes + pipeline) ---"
  local tests="$RUN_ROOT/_preflight-tests.log"
  if ! $PYTHON -m unittest tests.test_mcat_probes tests.test_mcat_drift \
       tests.test_mcat_pipeline > "$tests" 2>&1; then
    tail -25 "$tests"
    echo "!! unit tests failed; see $tests" >&2
    return 1
  fi
  grep -E "^(Ran [0-9]+ test|OK|FAILED)" "$tests" | tail -2

  echo "--- CPU smoke: all three probes on the fixture encoder ---"
  local smoke="$RUN_ROOT/_smoke"
  local log="$RUN_ROOT/_preflight-smoke.log"
  rm -rf "$smoke"
  local small="--output-dir $smoke --fixture --domain qa --corpus-limit 400 \
--per-split 3 --documents 24 --support 4 --optimization 6 --evaluation 8 \
--poison-count 2 --trigger-tokens 4 --top-k 3 --max-length 32 --steps 3 \
--mode direct-logit"
  (
    set -e
    $PYTHON -m src.triggers.mcat prepare-episodes $small
    $PYTHON -m src.triggers.mcat index            $small
    $PYTHON -m src.triggers.mcat probe-growth     $small --split test \
      --drift-seed 0 1 --growth 0.25 1.0 --bootstrap-iterations 200
    $PYTHON -m src.triggers.mcat probe-position   $small --split test \
      --position suffix --position prefix --position both --attention \
      --attention-queries 2 --bootstrap-iterations 200
  ) > "$log" 2>&1
  # NOT `) || {...}`: a subshell on the left of || runs with set -e suppressed, so
  # a failing stage would be skipped over instead of stopping the run.
  if [ $? -ne 0 ]; then
    tail -25 "$log"
    echo "!! smoke failed; see $log" >&2
    return 1
  fi
  echo "===== preflight ok ====="
}

# One shared run directory per arm: episodes, index and checkpoint are reused by
# every probe that reads them, so the corpus is encoded once.
arm_dir () { echo "$RUN_ROOT/$1/seed_$SEED"; }

# prepare_arm <arm> <mode> [with_train]
#
# ``with_train`` is explicit rather than inferred from the mode.  The probes reach
# a trigger two different ways and they need different preparation:
#
#   probe-growth / probe-position under direct-logit search per episode inside the
#   probe, so a train stage would be compute spent on a checkpoint nothing reads.
#
#   `evaluate` refuses to run without checkpoint.pt whatever the mode, so the R1
#   comparison -- which goes through `evaluate` -- does need one.
prepare_arm () {
  local arm="$1" mode="$2" with_train="${3:-0}"
  local out
  out=$(arm_dir "$arm")
  local common="--output-dir $out $EPISODE_FLAGS $RETRIEVER_FLAGS"
  $PYTHON -m src.triggers.mcat prepare-episodes $common $RESUME_FLAG
  $PYTHON -m src.triggers.mcat index            $common
  if [ "$with_train" = "1" ] || [ "$mode" != "direct-logit" ]; then
    $PYTHON -m src.triggers.mcat train $common $TRAIN_FLAGS --mode "$mode" $RESUME_FLAG
  fi
}

probe_r2 () {
  echo ""
  echo ">> P0-R2: frozen trigger against a growing memory"
  local out log
  out=$(arm_dir b2)
  log="$RUN_ROOT/r2.console"
  local common="--output-dir $out $EPISODE_FLAGS $RETRIEVER_FLAGS"
  (
    set -e
    prepare_arm b2 direct-logit
    $PYTHON -m src.triggers.mcat probe-growth $common $TRAIN_FLAGS \
      --mode direct-logit --split "$PROBE_SPLIT" --growth $GROWTH \
      --drift-seed $SEED_FLAGS --min-base-hit "$MIN_BASE_HIT" \
      --bootstrap-iterations "$BOOTSTRAP" $RESUME_FLAG
  ) > "$log" 2>&1
  if [ $? -ne 0 ]; then
    echo "!! r2 failed; last lines of $log:"; tail -20 "$log"; return 1
  fi
  echo "   done -> $out/probes/growth-suffix-$PROBE_SPLIT/probe_growth.json"
}

probe_r1 () {
  echo ""
  echo ">> P0-R1: one universal trigger (b3) against per-episode (b2)"
  local log="$RUN_ROOT/r1.console"
  local treatment control
  treatment=$(arm_dir b2)
  control=$(arm_dir b3)
  (
    set -e
    # Both arms need checkpoint.pt here, because R1 is measured through
    # `evaluate` and that stage requires one for every mode.
    prepare_arm b2 direct-logit 1
    prepare_arm b3 universal-logit
    # Equal step budget on both sides: the whole comparison is worthless if the
    # per-episode arm is given more optimization than the universal one.
    $PYTHON -m src.triggers.mcat evaluate --output-dir "$treatment" \
      $EPISODE_FLAGS $RETRIEVER_FLAGS $TRAIN_FLAGS --mode direct-logit \
      --split "$PROBE_SPLIT" $RESUME_FLAG
    $PYTHON -m src.triggers.mcat evaluate --output-dir "$control" \
      $EPISODE_FLAGS $RETRIEVER_FLAGS $TRAIN_FLAGS --mode universal-logit \
      --split "$PROBE_SPLIT" $RESUME_FLAG
    $PYTHON -m src.triggers.mcat probe-universal --output-dir "$treatment" \
      $EPISODE_FLAGS $RETRIEVER_FLAGS $TRAIN_FLAGS --mode direct-logit \
      --control-dir "$control" --bootstrap-iterations "$BOOTSTRAP"
  ) > "$log" 2>&1
  if [ $? -ne 0 ]; then
    echo "!! r1 failed; last lines of $log:"; tail -20 "$log"; return 1
  fi
  echo "   done -> $treatment/probe_universal.json"
}

probe_position () {
  echo ""
  echo ">> P1: trigger placement at an equal token budget ($POSITION_MODE)"
  local out log
  out=$(arm_dir b2)
  log="$RUN_ROOT/position.console"
  local common="--output-dir $out $EPISODE_FLAGS $RETRIEVER_FLAGS"
  (
    set -e
    prepare_arm b2 direct-logit
    $PYTHON -m src.triggers.mcat probe-position $common $TRAIN_FLAGS \
      --mode direct-logit --split "$PROBE_SPLIT" $POSITION_FLAGS \
      --position-mode "$POSITION_MODE" --position-baseline "$POSITION_BASELINE" \
      $ATTENTION_FLAG --bootstrap-iterations "$BOOTSTRAP" $RESUME_FLAG
  ) > "$log" 2>&1
  if [ $? -ne 0 ]; then
    echo "!! position failed; last lines of $log:"; tail -20 "$log"; return 1
  fi
  echo "   done -> $out/probes/position-$POSITION_MODE-suffix-$PROBE_SPLIT/probe_position.json"
}

summarize () {
  echo ""
  echo "===== P0 verdicts ($RUN_ROOT, seed $SEED, split $PROBE_SPLIT) ====="
  RUN_ROOT="$RUN_ROOT" SEED="$SEED" SPLIT="$PROBE_SPLIT" \
  POSITION_MODE="$POSITION_MODE" $PYTHON - <<'PY'
import json, os
from pathlib import Path

root = Path(os.environ["RUN_ROOT"])
seed, split = os.environ["SEED"], os.environ["SPLIT"]
mode = os.environ["POSITION_MODE"]
b2 = root / "b2" / f"seed_{seed}"


def load(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


growth = load(b2 / "probes" / f"growth-suffix-{split}" / "probe_growth.json")
if growth:
    print(f"  R2  verdict: {growth['verdict']}")
    print(f"      {growth.get('reason')}")
    base = growth.get("base") or {}
    print(f"      base on_hit {base.get('on_hit')!r}  off_hit {base.get('off_hit')!r}"
          f"  docs {base.get('documents')!r}")
    for key, level in growth.get("levels", {}).items():
        paired = level["drop_vs_base"]
        print(f"      +{float(key) * 100:>5.0f}%  on_hit {level['on_hit']!r:>22}"
              f"  drop {paired['mean_difference']!r:>22}"
              f"  ci [{paired['ci_low']!r}, {paired['ci_high']!r}]"
              f"  seed_spread {level['seed_spread']!r}")
else:
    print("  R2  (not run)")

universal = load(b2 / "probe_universal.json")
if universal:
    print(f"  R1  verdict: {universal['verdict']}")
    print(f"      {universal.get('reason')}")
    print(f"      treatment {universal['treatment']['mean']!r} vs control "
          f"{universal['control']['mean']!r}  paired "
          f"{universal['paired']['mean_difference']!r}")
else:
    print("  R1  (not run)")

position = load(b2 / "probes" / f"position-{mode}-suffix-{split}" / "probe_position.json")
if position:
    print(f"  P1  best length-matched placement: {position.get('best_length_matched')}")
    for name, block in sorted(position.get("positions", {}).items()):
        flag = "" if block["length_matched"] else "  (NOT length-matched)"
        print(f"      {name:<12} on_hit {block['on_hit']!r:>22}"
              f"  off_hit {block['off_hit']!r}{flag}")
    for name, block in sorted((position.get("attention") or {}).items()):
        if block.get("supported"):
            print(f"      {name:<12} [CLS] attention on trigger "
                  f"{block['cls_attention_on_trigger']!r}")
else:
    print("  P1  (not run)")
PY
}

PROBES=("$@")
[ ${#PROBES[@]} -eq 0 ] && PROBES=(r2)
if [ "${PROBES[0]}" = "all" ]; then
  PROBES=(r2 r1 position)
fi
if [ "${PROBES[0]}" = "preflight" ]; then
  preflight || exit 1
  exit 0
fi

for probe in "${PROBES[@]}"; do
  case "$probe" in
    r2|r1|position) ;;
    *) echo "!! unknown probe: $probe (choose r2, r1, position, all, preflight)" >&2
       exit 1 ;;
  esac
done

if [ "$SKIP_PREFLIGHT" != "1" ]; then
  preflight || { echo "!! preflight failed; nothing was run on the GPU" >&2; exit 1; }
fi

# RUN_ROOT too: preflight is the only other place that makes it, and
# SKIP_PREFLIGHT=1 would leave every probe with nowhere to write its console.
mkdir -p "$CACHE_DIR" "$RUN_ROOT"
failed=()
for probe in "${PROBES[@]}"; do
  case "$probe" in
    r2)       probe_r2       || failed+=(r2) ;;
    r1)       probe_r1       || failed+=(r1) ;;
    position) probe_position || failed+=(position) ;;
  esac
done

summarize

if [ ${#failed[@]} -gt 0 ]; then
  echo "!! failed probes: ${failed[*]}" >&2
  exit 1
fi
echo "===== complete; artifacts under $RUN_ROOT ====="
