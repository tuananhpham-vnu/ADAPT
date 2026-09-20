#!/bin/bash
# MCAT M0-M2 pilot: episode preparation, generator training and the baseline arms.
#
# Runs src/mcat over StrategyQA (+ EhrAgent) on a single GPU. Every arm shares one
# clean-vector cache, so the corpus is encoded once rather than once per arm.
#
# Usage (from repo root):
#   bash scripts/run_mcat_kaggle.sh                  # preflight + the main arm (m1)
#   bash scripts/run_mcat_kaggle.sh m1 b4 b5 b6      # pick arms
#   bash scripts/run_mcat_kaggle.sh all              # every arm
#   bash scripts/run_mcat_kaggle.sh preflight        # checks only, touches no GPU
#   RESUME=1 bash scripts/run_mcat_kaggle.sh all     # continue after a 12h cutoff
#   STEPS=50 DOMAINS="qa" bash scripts/run_mcat_kaggle.sh m1     # quick shakedown
#
# Arms (see _idea/memory_conditioned_generator_Q1_A_star.md section 9):
#   m1      memory+query generator ............ the method
#   b2      direct logits, per episode ........ what the optimizer alone buys
#   b3      one universal logits matrix ....... is a single trigger already enough
#   b4      unconditional generator ........... is the network just memorizing one answer
#   b5      query-only generator .............. how much comes from the query distribution
#   b6      memory-only generator ............. how much comes from the memory state
#   margin  m1 with the retrieval hinge on .... ablates lambda_ret
#
# Notes:
#  * Needs ONE GPU. Unlike algo/agentpoison_margin.py there is no GPT-2 or Llama
#    scorer at M0-M2, so do not reserve two.
#  * scikit-learn is required: algo.clustering.fit_centers supplies the benign
#    reference centers. Without it `train` stops, by design.
#  * --domain ad will fail: agentdriver/data/finetune/data_samples_train.json is
#    not vendored. Do not describe results as covering three agent domains.
set -u
# Without pipefail a failing `cmd | tail` reports tail's status, so a broken test
# run would sail straight through the preflight and into the GPU stages.
set -o pipefail

PYTHON="${PYTHON:-python}"
RUN_ROOT="${RUN_ROOT:-outputs/mcat/pilot}"
CACHE_DIR="${CACHE_DIR:-$RUN_ROOT/cache}"
SEED="${SEED:-0}"
DOMAINS="${DOMAINS:-qa ehr}"
DEVICE="${DEVICE:-cuda:0}"
# Pinned to the checkpoint that built ReAct/database/embeddings/agentpoison_dpr,
# so these numbers stay comparable with the AgentPoison baseline.
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
EVAL_SPLIT="${EVAL_SPLIT:-test}"
CORPUS_LIMIT="${CORPUS_LIMIT:-}"
RESUME="${RESUME:-0}"
SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"
# FIXTURE=1 runs every arm against the CPU fixture encoder: no download, no GPU,
# no scikit-learn. It verifies the plumbing and the train/evaluate flag contract.
# Its NUMBERS ARE NOT RESULTS -- the reference centers are the first rows of the
# snapshot rather than a fitted GMM, so the geometry is not AgentPoison's.
FIXTURE="${FIXTURE:-0}"

export TRANSFORMERS_NO_ADVISORY_WARNINGS=1 TOKENIZERS_PARALLELISM=false

if [ ! -f "src/mcat/cli.py" ]; then
  echo "!! run this from the repository root (src/mcat/cli.py not found)" >&2
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
  echo "** FIXTURE=1: CPU fixture encoder. Plumbing check only, not results. **"
fi

# Shared training hyperparameters. `evaluate` rebuilds TrainConfig from the CLI
# and compares its hash against the checkpoint, so train and evaluate MUST be
# given exactly the same flags or the run is refused. Building them once here is
# what keeps that true.
TRAIN_FLAGS="--steps $STEPS --learning-rate $LR --tau-start $TAU_START \
--tau-end $TAU_END --lambda-cpt $LAMBDA_CPT --margin $MARGIN \
--poison-mode $POISON_MODE"

RESUME_FLAG=""
[ "$RESUME" = "1" ] && RESUME_FLAG="--resume"

arm_flags () {
  case "$1" in
    m1)     echo "--mode generator --variant memory+query --lambda-ret 0.0" ;;
    b2)     echo "--mode direct-logit --lambda-ret 0.0" ;;
    b3)     echo "--mode universal-logit --lambda-ret 0.0" ;;
    b4)     echo "--mode generator --variant none --lambda-ret 0.0" ;;
    b5)     echo "--mode generator --variant query --lambda-ret 0.0" ;;
    b6)     echo "--mode generator --variant memory --lambda-ret 0.0" ;;
    margin) echo "--mode generator --variant memory+query --lambda-ret 1.0" ;;
    *)      return 1 ;;
  esac
}

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
    print("!! scikit-learn is required: algo.clustering.fit_centers supplies the "
          "benign reference centers for every real run", file=sys.stderr)
    sys.exit(1)
if [name for name in missing if name != "sklearn"]:
    sys.exit(1)
PY

  mkdir -p "$RUN_ROOT"

  echo "--- unit tests ---"
  local tests="$RUN_ROOT/_preflight-tests.log"
  if ! $PYTHON -m unittest tests.test_mcat tests.test_mcat_pipeline > "$tests" 2>&1; then
    tail -20 "$tests"
    echo "!! unit tests failed; see $tests" >&2
    return 1
  fi
  # The CLI stages inside the pipeline tests print JSON into this same log, so
  # pick the unittest summary out rather than trusting the last line.
  grep -E "^(Ran [0-9]+ test|OK|FAILED)" "$tests" | tail -2

  echo "--- CPU smoke (fixture retriever, no download) ---"
  local smoke="$RUN_ROOT/_smoke"
  local smoke_log="$RUN_ROOT/_preflight-smoke.log"
  rm -rf "$smoke"
  if ! $PYTHON -m src.mcat smoke --output-dir "$smoke" --fixture \
      --domain qa --corpus-limit 400 --per-split 2 \
      --documents 24 --support 4 --optimization 6 --evaluation 8 \
      --poison-count 2 --trigger-tokens 4 --top-k 3 --steps 3 --max-length 32 \
      > "$smoke_log" 2>&1; then
    tail -20 "$smoke_log"
    echo "!! smoke failed; see $smoke_log" >&2
    return 1
  fi
  # The resume guard is the point of the smoke run, so show that it fired.
  grep -F "resume guard" "$smoke_log" || {
    echo "!! smoke did not exercise the resume guard" >&2
    return 1
  }
  echo "===== preflight ok ====="
}

run_arm () {
  local arm="$1"
  local flags
  flags=$(arm_flags "$arm") || { echo "!! unknown arm: $arm" >&2; return 1; }
  local out="$RUN_ROOT/$arm/seed_$SEED"
  local log="$RUN_ROOT/$arm.console"
  mkdir -p "$RUN_ROOT"

  echo ""
  echo ">> arm [$arm] -> $out"
  echo "   $flags"
  local common="--output-dir $out $EPISODE_FLAGS $RETRIEVER_FLAGS"

  # A brace group runs in the current shell, so `exit 1` inside it would kill
  # the whole sweep instead of failing this one arm. A subshell is what is meant.
  (
    set -e
    $PYTHON -m src.mcat prepare-episodes $common $RESUME_FLAG
    $PYTHON -m src.mcat index            $common
    $PYTHON -m src.mcat train            $common $TRAIN_FLAGS $flags $RESUME_FLAG
    $PYTHON -m src.mcat evaluate         $common $TRAIN_FLAGS $flags --split "$EVAL_SPLIT" $RESUME_FLAG
    $PYTHON -m src.mcat report           $common
  ) > "$log" 2>&1
  local status=$?
  if [ $status -ne 0 ]; then
    echo "!! arm [$arm] failed; last lines of $log:"
    tail -15 "$log"
    return 1
  fi
  echo "   done -> $out/REPORT.md"
}

summarize () {
  echo ""
  echo "===== summary ($RUN_ROOT, seed $SEED, split $EVAL_SPLIT) ====="
  RUN_ROOT="$RUN_ROOT" SEED="$SEED" $PYTHON - "$@" <<'PY'
import json, os, sys
from pathlib import Path

root, seed = Path(os.environ["RUN_ROOT"]), os.environ["SEED"]
header = f"  {'arm':<8} {'hit@K':>7} {'occ@K':>7} {'margin':>9} {'false-act':>10} {'rt-ok':>6}  controls"
print(header)
print("  " + "-" * (len(header) - 2))
for arm in sys.argv[1:]:
    path = root / arm / f"seed_{seed}" / "evaluation.json"
    if not path.exists():
        print(f"  {arm:<8} {'(no evaluation.json)':>7}")
        continue
    data = json.loads(path.read_text(encoding="utf-8"))
    on = data["trigger_on"]
    hit = next((v for k, v in on.items() if k.startswith("hit_at_")), float("nan"))
    occ = next((v for k, v in on.items() if k.startswith("poison_occupancy_at_")), float("nan"))
    controls = data.get("controls", {}).get("shuffled_context", {})
    if controls.get("applicable"):
        note = f"shuffled-context changed {controls['changed']}/{controls['episodes']}"
    else:
        note = f"shuffled-context n/a ({controls.get('reason', '-')})"
    print(f"  {arm:<8} {hit:>7.3f} {occ:>7.3f} {on['mean_margin']:>9.3f} "
          f"{data['false_activation']:>10.3f} {data['round_trip_valid_rate']:>6.2f}  {note}")
print()
print("  Read these before quoting any of it:")
print("   * a low shuffled-context change rate means the memory branch is inert")
print("     and the conditioning claim is NOT supported -- a valid negative result.")
print("   * rt-ok below 1.00 means triggers broke on decode; loss gains are then void.")
print("   * false-act is only meaningful when the snapshot is much larger than K;")
print("     read it per domain in evaluation.jsonl, not as this average.")
PY
}

ARMS=("$@")
[ ${#ARMS[@]} -eq 0 ] && ARMS=("m1")
if [ "${ARMS[0]}" = "all" ]; then
  ARMS=(m1 b2 b3 b4 b5 b6 margin)
fi

if [ "${ARMS[0]}" = "preflight" ]; then
  preflight || exit 1
  exit 0
fi

for arm in "${ARMS[@]}"; do
  arm_flags "$arm" > /dev/null || { echo "!! unknown arm: $arm" >&2; exit 1; }
done

if [ "$SKIP_PREFLIGHT" != "1" ]; then
  preflight || { echo "!! preflight failed; nothing was run on the GPU" >&2; exit 1; }
fi

mkdir -p "$CACHE_DIR"
failed=()
for arm in "${ARMS[@]}"; do
  run_arm "$arm" || failed+=("$arm")
done

summarize "${ARMS[@]}"

if [ ${#failed[@]} -gt 0 ]; then
  echo "!! failed arms: ${failed[*]}" >&2
  exit 1
fi
echo "===== complete; artifacts under $RUN_ROOT ====="
