#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-pilot}
RUNNER="$(dirname "$0")/run_optimization_validation.sh"
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/agentpoison_margin}

case "$MODE" in
  pilot)
    for k in 1 5; do
      for weight in 0.1 0.5 1.0; do
        for stage in prepare index optimize retrieval-eval; do
          SEED=99 EXPERIMENT="pilot/alpha_$weight" MARGIN_WEIGHT=$weight OUTPUT_ROOT="$OUTPUT_ROOT" \
            bash "$RUNNER" "$stage" "margin-k$k"
        done
      done
    done
    ;;
  full)
    : "${MARGIN_WEIGHT_K1:?Set MARGIN_WEIGHT_K1 after pilot}"
    : "${MARGIN_WEIGHT_K5:?Set MARGIN_WEIGHT_K5 after pilot}"
    for seed in 0 1 2; do
      for stage in prepare index optimize retrieval-eval; do
        SEED=$seed EXPERIMENT=full OUTPUT_ROOT="$OUTPUT_ROOT" bash "$RUNNER" "$stage" baseline
        SEED=$seed EXPERIMENT=full MARGIN_WEIGHT=$MARGIN_WEIGHT_K1 OUTPUT_ROOT="$OUTPUT_ROOT" bash "$RUNNER" "$stage" margin-k1
        SEED=$seed EXPERIMENT=full MARGIN_WEIGHT=$MARGIN_WEIGHT_K5 OUTPUT_ROOT="$OUTPUT_ROOT" bash "$RUNNER" "$stage" margin-k5
      done
    done
    ;;
  resume) echo "Rerun the exact interrupted stage; resume is automatic." ;;
  report) find "$OUTPUT_ROOT" -name retrieval_evaluation.json -print ;;
  *) echo "Usage: $0 {pilot|full|resume|report}" >&2; exit 2 ;;
esac
