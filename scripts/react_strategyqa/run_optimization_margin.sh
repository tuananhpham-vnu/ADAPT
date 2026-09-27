# *tuananhpham-vnu*
# Step 1 (margin variant): same search and flags as run_optimization.sh, run through
# src/triggers/hotflip_margin.py, which minimizes loss = -score + MARGIN_WEIGHT * l_margin.
# Writes results/qa/ap_margin/<timestamp>/{trigger.json,checkpoint.pt,metrics.jsonl}.
#
#   bash scripts/react_strategyqa/run_optimization_margin.sh                  # MARGIN_WEIGHT=0.1, same as the compactness weight
#   MARGIN_WEIGHT=0 bash scripts/react_strategyqa/run_optimization_margin.sh  # baseline: upstream objective, l_margin still logged
#   MARGIN_TOP_K=2 bash scripts/react_strategyqa/run_optimization_margin.sh
#
# Evaluate with the usual step 2, pointing it at the run directory:
#   bash scripts/react_strategyqa/run_inference_llama2.sh results/qa/ap_margin/<timestamp>
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m src.triggers.hotflip_margin \
  --agent qa --algo ap \
  --model dpr-ctx_encoder-single-nq-base \
  --save-dir ./results \
  --ppl-filter \
  --exclude-special \
  --trigger-tokens 10 \
  --golden-trigger \
  --batch-size 64 --micro-batch-size 16 \
  --num-iter 100 --patience 5 \
  --margin-weight "${MARGIN_WEIGHT:-0.1}" \
  --margin-value "${MARGIN_VALUE:-0.1}" \
  --margin-top-k "${MARGIN_TOP_K:-1}" \
  --wandb --plot \
  "$@"
# *tuananhpham-vnu*
