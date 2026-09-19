#!/usr/bin/env bash
set -euo pipefail

STAGE=${1:-smoke}
ARM=${2:-baseline}
PYTHON=${PYTHON:-python}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/agentpoison_margin}
EXPERIMENT=${EXPERIMENT:-full}
SEED=${SEED:-0}
MARGIN_WEIGHT=${MARGIN_WEIGHT:-0.5}
TARGET_MICROBATCH=${TARGET_MICROBATCH:-4}
POISON_COUNT=${POISON_COUNT:-5}

case "$ARM" in
  baseline) ARM_WEIGHT=0.0; ARM_K=1 ;;
  margin-k1) ARM_WEIGHT=$MARGIN_WEIGHT; ARM_K=1 ;;
  margin-k5) ARM_WEIGHT=$MARGIN_WEIGHT; ARM_K=5 ;;
  *) echo "Unknown arm: $ARM" >&2; exit 2 ;;
esac

run_stage() {
  local stage=$1 arm=$2 weight=$3 topk=$4
  local run_dir="$OUTPUT_ROOT/$EXPERIMENT/$arm/seed_$SEED"
  local resume=()
  [[ -d "$run_dir" ]] && resume=(--resume)
  "$PYTHON" -m algo.agentpoison_margin "$stage" \
    --output-dir "$run_dir" --seed "$SEED" --margin-weight "$weight" \
    --margin-top-k "$topk" --poison-count "$POISON_COUNT" \
    --target-microbatch-size "$TARGET_MICROBATCH" "${resume[@]}"
}

if [[ "$STAGE" == smoke ]]; then
  EXPERIMENT=smoke
  for spec in "baseline:0.0:1" "margin-k1:$MARGIN_WEIGHT:1"; do
    IFS=: read -r smoke_arm smoke_weight smoke_k <<< "$spec"
    run_dir="$OUTPUT_ROOT/$EXPERIMENT/$smoke_arm/seed_$SEED"
    resume=(); [[ -d "$run_dir" ]] && resume=(--resume)
    "$PYTHON" -m algo.agentpoison_margin smoke \
      --output-dir "$run_dir" --seed "$SEED" --smoke-fixture \
      --num-iter 2 --num-grad-iter 2 --batch-size 4 \
      --replacement-candidates 20 --subsample-candidates 5 \
      --target-microbatch-size 2 --margin-weight "$smoke_weight" \
      --margin-top-k "$smoke_k" --poison-count "$POISON_COUNT" "${resume[@]}"
  done
elif [[ "$STAGE" =~ ^(prepare|index|optimize|retrieval-eval|report)$ ]]; then
  run_stage "$STAGE" "$ARM" "$ARM_WEIGHT" "$ARM_K"
elif [[ "$STAGE" == agent-eval ]]; then
  : "${AGENT_RECORDS:?Set AGENT_RECORDS to the inference JSONL path}"
  run_dir="$OUTPUT_ROOT/$EXPERIMENT/$ARM/seed_$SEED"
  resume=(); [[ -d "$run_dir" ]] && resume=(--resume)
  "$PYTHON" -m algo.agentpoison_margin agent-eval \
    --output-dir "$run_dir" --seed "$SEED" --margin-weight "$ARM_WEIGHT" \
    --margin-top-k "$ARM_K" --poison-count "$POISON_COUNT" \
    --agent-records "$AGENT_RECORDS" "${resume[@]}"
elif [[ "$STAGE" =~ ^(baseline|margin-k1|margin-k5)$ ]]; then
  exec "$0" optimize "$STAGE"
else
  echo "Usage: $0 {smoke|prepare|index|optimize|retrieval-eval|agent-eval|report} [baseline|margin-k1|margin-k5]" >&2
  exit 2
fi
