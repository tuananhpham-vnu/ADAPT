# *tuananhpham-vnu*
# Step 1 (margin variant): same search and flags as run_optimization.sh, run through
# src/triggers/hotflip_margin.py, which minimizes the normalized convex combination
#   loss = W_UNI*l_uni + W_CPT*l_cpt + W_MARGIN*l_margin   (weights renormalized to sum 1)
# Writes results/qa/ap_margin/<timestamp>_u.._c.._m../{trigger.json,checkpoint.pt,metrics.jsonl}.
#
#   bash scripts/react_strategyqa/run_optimization_margin.sh                        # 0.8 / 0.1 / 0.1
#   W_UNI=1 W_CPT=0.1 W_MARGIN=0 bash scripts/react_strategyqa/run_optimization_margin.sh   # upstream ratio
#   W_UNI=0.6 W_MARGIN=0.3 bash scripts/react_strategyqa/run_optimization_margin.sh
#
# Evaluate with the usual step 2, pointing it at the run directory:
#   bash scripts/react_strategyqa/run_inference_llama2.sh results/qa/ap_margin/<run>
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m src.triggers.hotflip_margin \
  --agent qa --algo ap \
  --model dpr-ctx_encoder-single-nq-base \
  --save-dir ./results \
  --ppl-filter \
  --exclude-special \
  --golden-trigger \
  --batch-size 64 --micro-batch-size 16 \
  --num-iter 100 --patience 5 \
  --w-uni "${W_UNI:-0.8}" \
  --w-cpt "${W_CPT:-0.1}" \
  --w-margin "${W_MARGIN:-0.1}" \
  --margin-value "${MARGIN_VALUE:-0.1}" \
  --margin-top-k "${MARGIN_TOP_K:-1}" \
  --wandb --plot \
  "$@"
# *tuananhpham-vnu*
