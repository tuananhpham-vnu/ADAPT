#!/bin/bash
# Specificity pilot on StrategyQA: universal vs group vs per-query trigger language.
#
# Answers one question only: at each level of trigger specificity, how well does the
# inserted prefix stay on-topic with the query and how much does it damage fluency?
# It does NOT measure ASR. Retrieval numbers come out of stage `bank` as a by-product;
# downstream action success is not evaluated anywhere in this script.
#
# Usage (from repo root, on a Kaggle GPU notebook with internet ON for stage `models`):
#   bash scripts/run_specificity_ap.sh all           # every stage, in order
#   bash scripts/run_specificity_ap.sh preflight     # checks only, no GPU, no network
#   bash scripts/run_specificity_ap.sh models index  # pick stages
#   TRAIN_SIZE=12 TEST_SIZE=48 bash scripts/run_specificity_ap.sh all   # smaller/cheaper
#
# Interrupted run (Kaggle 12h cutoff, killed session): re-run the SAME command. Finished
# stages are skipped via their marker under $RUN_ROOT/state, the bank stage reuses each
# completed arm, and the language stage reloads its per-query fits from fit_cache.json.
# FORCE=1 redoes everything; FORCE=bank redoes one stage. Logs land in $RUN_ROOT/logs.
#
# Stages (full explanation in _guidance/21_specificity_pipeline_stages_agentpoison.md):
#   preflight  dependencies, repo root, data files present. Touches no GPU.
#   models     download the six models into the two caches the code reads from.
#   index      encode the 9,251 StrategyQA paragraphs once with DPR -> vectors.npy.
#   bank       optimize one trigger bank per specificity level (universal / semantic
#              groups / per query) at prefix position, under one shared request cap.
#   language   score those banks for query relevance and fluency on FRESH queries,
#              plus the query-adaptive arm that picks a prefix per incoming query.
#   summary    print the comparison table and the caveats that go with it.
#
# Notes:
#  * Stage `models` is the only one needing network. Kaggle: Settings -> Internet ON.
#  * One GPU is enough. Everything here is inference; nothing is fine-tuned.
#  * scikit-learn is required: src.triggers.clustering.fit_centers fits the GMM
#    reference centers for the semantic arm and the benign uniqueness term.
#  * Re-running a stage with CHANGED settings is refused by the contract check in
#    both CLIs. Change RUN_ROOT (or delete the directory) for a new configuration.
#  * Group routing here is GMM on clean train embeddings, which produces unbalanced
#    clusters. That is a known weakness, not a design goal; see the guidance doc.
set -u
set -o pipefail

PYTHON="${PYTHON:-python}"
RUN_ROOT="${RUN_ROOT:-outputs/specificity/kaggle}"
BANK_DIR="${BANK_DIR:-$RUN_ROOT/bank}"
LANG_DIR="${LANG_DIR:-$RUN_ROOT/language}"
MODEL_CACHE="${MODEL_CACHE:-outputs/model-cache}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
STATE_DIR="${STATE_DIR:-$RUN_ROOT/state}"
# FORCE=1 redoes every stage; FORCE=<stage> redoes just that one.
FORCE="${FORCE:-0}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-42}"
LANGUAGE_SEED="${LANGUAGE_SEED:-2026}"

# Three disjoint StrategyQA question files (verified: zero shared question text):
#   strategyqa_train_filtered.json  2,803 unique, LABELLED  -> train + poison carriers
#   strategyqa_test.json              489 unique, NO labels -> validation
#   strategyqa_dev.json               229 unique, LABELLED  -> test
# The unlabelled pool goes to validation on purpose: validation only freezes variants
# on language metrics and never needs an answer. Test keeps its labels so downstream
# answer accuracy and action ASR stay measurable later on the SAME test questions.
# TEST_SIZE (stage bank) and FRESH_TEST_SIZE (stage language) both draw from the test
# file and must not overlap, so TEST_SIZE + FRESH_TEST_SIZE <= 229.
# Defaults consume both evaluation files completely: all 489 validation questions and
# all 229 test questions, split 80 for the bank stage and 149 for the language stage.
# TRAIN_SIZE cannot be scaled the same way -- per_query fits one trigger per training
# query and POISON_COUNT grows with it, so 24 queries already plant 120 poisoned keys.
TRAIN_SIZE="${TRAIN_SIZE:-24}"
VALIDATION_SIZE="${VALIDATION_SIZE:-489}"
TEST_SIZE="${TEST_SIZE:-80}"
FRESH_TEST_SIZE="${FRESH_TEST_SIZE:-149}"
# The retrieval hinge in src/triggers/losses.py needs TOP_K poison keys inside the group
# being optimized. per_query is the binding arm: it owns POISON_COUNT/TRAIN_SIZE sources
# per trigger, so POISON_COUNT >= TOP_K * TRAIN_SIZE, otherwise that arm dies with
# "top_k must be between 1 and the number of poison embeddings".
# TOP_K=5 keeps hit@5, comparable with the earlier DPR runs and with how the RAG
# poisoning papers report. The price is 5x more poisoned keys in the corpus, so the
# preflight prints the poisoning ratio and false activation has to be read with it.
TOP_K="${TOP_K:-5}"
POISON_COUNT="${POISON_COUNT:-$((TOP_K * TRAIN_SIZE))}"
GROUP_COUNT="${GROUP_COUNT:-4}"
# Logical retriever-text requests per arm. Split evenly across that arm's groups, so
# per_query gives each of its TRAIN_SIZE triggers BUDGET/TRAIN_SIZE requests.
BUDGET="${BUDGET:-1536}"
CANDIDATE_CAP="${CANDIDATE_CAP:-20}"

DATA_DIR="${DATA_DIR:-ReAct/database}"
TRAIN_FILE="$DATA_DIR/strategyqa_train_filtered.json"
# Set VALIDATION_FILE to "" to fall back to the old behaviour (validation carved out of
# the train pool). Swap VALIDATION_FILE and TEST_FILE if you would rather have the large
# unlabelled pool as test and accept that downstream cannot be scored on it.
VALIDATION_FILE="${VALIDATION_FILE:-$DATA_DIR/strategyqa_test.json}"
TEST_FILE="${TEST_FILE:-$DATA_DIR/strategyqa_dev.json}"
CORPUS_FILE="$DATA_DIR/strategyqa_train_paragraphs.json"
# Only the index stage reads this one; src.agentpoison.strategyqa loads the question
# file before it builds the cache, even though indexing itself only needs the corpus.
QUESTION_FILE="$DATA_DIR/strategyqa_train.json"
INDEX_DIR="${INDEX_DIR:-$DATA_DIR/embeddings/agentpoison_dpr}"
DPR_MODEL="${DPR_MODEL:-facebook/dpr-ctx_encoder-single-nq-base}"
MAX_LENGTH="${MAX_LENGTH:-512}"
BATCH_SIZE="${BATCH_SIZE:-32}"

export TRANSFORMERS_NO_ADVISORY_WARNINGS=1 TOKENIZERS_PARALLELISM=false

if [ ! -f "src/triggers/specificity/__main__.py" ]; then
  echo "!! run this from the repository root (src/triggers/specificity/__main__.py not found)" >&2
  exit 1
fi
mkdir -p "$RUN_ROOT"

# ---------------------------------------------------------------- stage: preflight
# Fails before anything expensive starts. Checks the four things that actually break
# on a fresh Kaggle session: missing packages, missing data files, no GPU, wrong cwd.
stage_preflight () {
  echo "===== preflight ====="
  $PYTHON - <<'PY' || return 1
import sys
missing = []
for name in ("torch", "transformers", "numpy", "sklearn"):
    try:
        module = __import__(name)
        print(f"  {name:<14} {getattr(module, '__version__', '?')}")
    except ImportError:
        missing.append(name)
        print(f"  {name:<14} MISSING")
if missing:
    print(f"!! install first: {' '.join(missing)}", file=sys.stderr)
    sys.exit(1)
import torch
print(f"  cuda available {torch.cuda.is_available()} | devices {torch.cuda.device_count()}")
PY

  local ok=0
  for path in "$TRAIN_FILE" "$TEST_FILE" "$CORPUS_FILE" "$QUESTION_FILE" ${VALIDATION_FILE:+"$VALIDATION_FILE"}; do
    if [ -f "$path" ]; then
      echo "  data ok        $path"
    else
      echo "!! missing data file: $path" >&2
      ok=1
    fi
  done
  [ $ok -eq 0 ] || return 1

  # The two dev-drawn splits must fit inside the dev file, and per_query needs one
  # poison source per training query. Both are refused later; catch them here instead.
  TRAIN_SIZE="$TRAIN_SIZE" VALIDATION_SIZE="$VALIDATION_SIZE" TEST_SIZE="$TEST_SIZE" \
  FRESH_TEST_SIZE="$FRESH_TEST_SIZE" POISON_COUNT="$POISON_COUNT" GROUP_COUNT="$GROUP_COUNT" \
  TOP_K="$TOP_K" CORPUS_FILE="$CORPUS_FILE" \
  TRAIN_FILE="$TRAIN_FILE" TEST_FILE="$TEST_FILE" VALIDATION_FILE="$VALIDATION_FILE" \
  $PYTHON - <<'PY' || return 1
import json, os, sys
size = lambda key: int(os.environ[key])
train = json.load(open(os.environ["TRAIN_FILE"], encoding="utf-8"))
dev = json.load(open(os.environ["TEST_FILE"], encoding="utf-8"))
validation_file = os.environ.get("VALIDATION_FILE") or ""
validation = json.load(open(validation_file, encoding="utf-8")) if validation_file else None
need_train = size("TRAIN_SIZE") + size("POISON_COUNT") + (0 if validation else size("VALIDATION_SIZE"))
need_test = size("TEST_SIZE") + size("FRESH_TEST_SIZE")
labelled = lambda rows: "answer" in rows[0]
print(f"  train pool     {len(train)} rows, need {need_train}, labelled={labelled(train)}")
print(f"  test pool      {len(dev)} rows, need {need_test} (bank test + fresh language test), "
      f"labelled={labelled(dev)}")
problems = []
if validation is not None:
    print(f"  valid pool     {len(validation)} rows, need {size('VALIDATION_SIZE')}, "
          f"labelled={labelled(validation)}")
    if size("VALIDATION_SIZE") > len(validation):
        problems.append("validation pool too small")
if not labelled(dev):
    problems.append("the test pool has no answer labels: downstream accuracy and action "
                    "ASR can never be scored on these questions")
if need_train > len(train):
    problems.append("train pool too small")
if need_test > len(dev):
    problems.append("test pool too small: lower TEST_SIZE or FRESH_TEST_SIZE")
per_trigger = size("POISON_COUNT") // max(1, size("TRAIN_SIZE"))
corpus = len(json.load(open(os.environ["CORPUS_FILE"], encoding="utf-8")))
print(f"  per_query      {per_trigger} poison key(s) per trigger, TOP_K={size('TOP_K')}")
print(f"  poisoning      {size('POISON_COUNT')}/{corpus} paragraphs = "
      f"{100 * size('POISON_COUNT') / corpus:.2f}% of the corpus")
if size("TOP_K") > per_trigger:
    problems.append(f"TOP_K must be <= {per_trigger}: raise POISON_COUNT to "
                    f"{size('TOP_K') * size('TRAIN_SIZE')}, or lower TOP_K")
if size("GROUP_COUNT") > size("TRAIN_SIZE"):
    problems.append("GROUP_COUNT must not exceed TRAIN_SIZE")
for p in problems:
    print(f"!! {p}", file=sys.stderr)
sys.exit(1 if problems else 0)
PY
  echo "===== preflight ok ====="
}

# ------------------------------------------------------------------- stage: models
# Every loader in src/triggers runs with local_files_only=True, so nothing downloads
# implicitly later. Two caches are in play and they are NOT interchangeable:
#   default HF cache  -> DPR (retriever) and MiniLM-L6 (selection semantics)
#   outputs/model-cache -> distilgpt2, gpt2, paraphrase-MiniLM-L3, POS tagger
# Fetching each name into both caches costs a few hundred MB and removes the whole
# class of "works on my machine, local_files_only fails on Kaggle" errors.
stage_models () {
  echo "===== models (needs internet) ====="
  MODEL_CACHE="$MODEL_CACHE" DPR_MODEL="$DPR_MODEL" $PYTHON - <<'PY' || return 1
import os
from transformers import (AutoModel, AutoModelForCausalLM, AutoModelForTokenClassification,
                          AutoTokenizer, DPRContextEncoder)
cache = os.environ["MODEL_CACHE"]
models = [
    (os.environ["DPR_MODEL"], DPRContextEncoder),                       # retriever under attack
    ("sentence-transformers/all-MiniLM-L6-v2", AutoModel),              # selection relevance + meaning guard
    ("distilbert/distilgpt2", AutoModelForCausalLM),                    # selection fluency
    ("sentence-transformers/paraphrase-MiniLM-L3-v2", AutoModel),       # audit relevance (unseen weights)
    ("openai-community/gpt2", AutoModelForCausalLM),                    # audit fluency (unseen weights)
    ("vblagoje/bert-english-uncased-finetuned-pos", AutoModelForTokenClassification),  # noun-phrase filter
]
for name, loader in models:
    for kwargs in ({}, {"cache_dir": cache}):
        AutoTokenizer.from_pretrained(name, **kwargs)
        loader.from_pretrained(name, **kwargs)
    print(f"  cached {name}", flush=True)
PY
  echo "===== models ok ====="
}

# -------------------------------------------------------------------- stage: index
# Encodes all 9,251 paragraphs once with DPR, L2-normalized, into vectors.npy plus a
# manifest that pins corpus hash / encoder / max_length. Stage `bank` refuses to start
# if any of those three disagree, which is what stops a stale cache from silently
# changing the retrieval numbers. Reuses an existing matching index instead of redoing it.
# The encoder here is pinned in src/agentpoison/strategyqa.py (ENCODER), so overriding
# DPR_MODEL without editing that constant makes stage `bank` reject the cache.
stage_index () {
  echo "===== index ====="
  $PYTHON -m src.agentpoison.strategyqa index \
    --corpus "$CORPUS_FILE" --questions "$QUESTION_FILE" --index "$INDEX_DIR" \
    --device "$DEVICE" --batch-size "$BATCH_SIZE" --max-length "$MAX_LENGTH" || return 1
  ls -la "$INDEX_DIR"
  echo "===== index ok ====="
}

# --------------------------------------------------------------------- stage: bank
# The trigger banks themselves. Three arms, all at PREFIX position, all under the same
# --budget of logical retriever requests:
#   universal  1 trigger for all TRAIN_SIZE training queries
#   semantic   GROUP_COUNT triggers; queries grouped by GMM on clean DPR embeddings,
#              unseen queries routed to the nearest clean center
#   per_query  TRAIN_SIZE triggers, one per training query, routed by nearest neighbour
# Prefix only: stage `language` reads $BANK_DIR/prefix/*.json and nothing else, so
# paying for suffix and infix here would be wasted compute.
stage_bank () {
  echo "===== bank ====="
  $PYTHON -m src.triggers.hierarchy compare \
    --backend hf --output "$BANK_DIR" \
    --arms universal semantic per_query --positions prefix \
    --groups "$GROUP_COUNT" --budget "$BUDGET" \
    --train-size "$TRAIN_SIZE" --validation-size "$VALIDATION_SIZE" \
    --test-size "$TEST_SIZE" --poison-count "$POISON_COUNT" --top-k "$TOP_K" \
    --seed "$SEED" --device "$DEVICE" --batch-size "$BATCH_SIZE" \
    --model "$DPR_MODEL" --max-length "$MAX_LENGTH" \
    --train "$TRAIN_FILE" --test "$TEST_FILE" --corpus "$CORPUS_FILE" \
    ${VALIDATION_FILE:+--validation "$VALIDATION_FILE"} \
    --corpus-embeddings "$INDEX_DIR/vectors.npy" || return 1
  echo "===== bank ok -> $BANK_DIR/REPORT.md ====="
}

# ----------------------------------------------------------------- stage: language
# The actual specificity evaluation. Takes FRESH_TEST_SIZE dev queries that appear in
# none of the bank splits, routes them with the FROZEN clean centers from stage `bank`,
# and scores six variants per scope plus the query-adaptive arm:
#   legacy_raw / legacy_heading  the stage-`bank` trigger, old and punctuated rendering
#   generic_language             re-picked shared seed, language objective only
#   grounded_language            candidates drawn from train content, bank frozen at test
#   topic_heading                bare topic, no About/Regarding -- the word-repetition control
#   per_query/query_adaptive     a prefix chosen for each incoming query at inference
# Variants are frozen on the 16 validation queries BEFORE the fresh test is scored, and
# the reported numbers come from two audit models that never took part in selection.
stage_language () {
  echo "===== language ====="
  $PYTHON -m src.triggers.specificity \
    --source "$BANK_DIR" --output "$LANG_DIR" \
    --fresh-test-size "$FRESH_TEST_SIZE" --seed "$LANGUAGE_SEED" \
    --candidate-cap "$CANDIDATE_CAP" --phrase-mode pos \
    --device "$DEVICE" --model-cache "$MODEL_CACHE" || return 1
  echo "===== language ok -> $LANG_DIR/REPORT.md ====="
}

# ------------------------------------------------------------------- stage: summary
# One table, audit-model scores, fresh test split only.
stage_summary () {
  echo ""
  echo "===== summary ($LANG_DIR, fresh test, audit models) ====="
  LANG_DIR="$LANG_DIR" BANK_DIR="$BANK_DIR" $PYTHON - <<'PY'
import json, os
from pathlib import Path

lang, bank = Path(os.environ["LANG_DIR"]), Path(os.environ["BANK_DIR"])
results = lang / "results.json"
if not results.exists():
    raise SystemExit(f"  no results at {results}; run the language stage first")
data = json.loads(results.read_text(encoding="utf-8"))
test = data["audit_models"]["test"]
choices = json.loads((lang / "selection.json").read_text(encoding="utf-8"))["choices"]

print(f"  {'variant':<32} {'relevance':>10} {'ppl ratio':>10} {'meaning ok':>11}")
print("  " + "-" * 66)
for name, row in test.items():
    m = row["metrics"]
    print(f"  {name:<32} {m['trigger_query_cosine']:>10.4f} "
          f"{m['geometric_mean_ppl_ratio']:>10.3f} {m['preservation_proxy_pass_rate']:>11.3f}")
print()
print(f"  frozen on validation: {choices}")

report = bank / "REPORT.md"
if report.exists():
    print()
    print("  retrieval side (from stage bank, prefix position):")
    for line in report.read_text(encoding="utf-8").splitlines():
        if line.startswith("| prefix"):
            print("   " + line)
print()
print("  Read before quoting any of it:")
print("   * relevance is cosine to the original query, not a probability of being on topic;")
print("     ppl ratio is GPT-2 on the whole sentence, 1.000 = the untouched query.")
print("   * query_adaptive scores high largely by REPEATING a topic word from the query.")
print("     Compare it against topic_heading before calling that an improvement.")
print("   * meaning ok is a similarity + number/negation proxy, not entailment and not")
print("     human review. It does not certify the answer is unchanged.")
print("   * none of this is ASR. A fluent trigger that never gets retrieved is still useless.")
PY
}

STAGES=("$@")
[ ${#STAGES[@]} -eq 0 ] && STAGES=("all")
if [ "${STAGES[0]}" = "all" ]; then
  STAGES=(preflight models index bank language summary)
fi

for stage in "${STAGES[@]}"; do
  case "$stage" in
    preflight|models|index|bank|language|summary) ;;
    *) echo "!! unknown stage: $stage (preflight models index bank language summary all)" >&2; exit 1 ;;
  esac
done

# Every stage writes a timestamped log and, once it succeeds, a marker under state/.
# A rerun of the same command then skips the finished stages, which is what makes a
# 12h cutoff or a killed session recoverable without redoing the GPU work.
#   FORCE=1          re-run a stage even if its marker exists
#   FORCE=bank       re-run only that stage
# The marker is written ONLY after the stage exits 0, so an interrupted stage is never
# recorded as finished. `preflight` and `summary` are cheap and always re-run.
run_stage () {
  local stage="$1"
  local marker="$STATE_DIR/$stage.done"
  local log="$LOG_DIR/$stage.log"
  mkdir -p "$STATE_DIR" "$LOG_DIR"

  case "$stage" in
    preflight|summary) ;;
    *)
      if [ -f "$marker" ] && [ "$FORCE" != "1" ] && [ "$FORCE" != "$stage" ]; then
        echo ">> stage [$stage] already done ($(cat "$marker")); FORCE=$stage to redo"
        return 0
      fi ;;
  esac

  echo ""
  echo ">> stage [$stage] -> $log"
  {
    echo "===== $stage started $(date -u '+%Y-%m-%dT%H:%M:%SZ') ====="
    echo "     run_root=$RUN_ROOT seed=$SEED train=$TRAIN_SIZE groups=$GROUP_COUNT"
    echo "     budget=$BUDGET top_k=$TOP_K poison=$POISON_COUNT fresh_test=$FRESH_TEST_SIZE"
  } >> "$log"

  # tee would hand back its own exit status, so read the stage's from PIPESTATUS.
  set -o pipefail
  "stage_$stage" 2>&1 | tee -a "$log"
  local status=${PIPESTATUS[0]}
  if [ $status -ne 0 ]; then
    echo "!! stage [$stage] failed (exit $status); full log in $log" >&2
    echo "===== $stage FAILED exit $status $(date -u '+%Y-%m-%dT%H:%M:%SZ') =====" >> "$log"
    return $status
  fi
  echo "===== $stage ok $(date -u '+%Y-%m-%dT%H:%M:%SZ') =====" >> "$log"
  date -u '+%Y-%m-%dT%H:%M:%SZ' > "$marker"
  return 0
}

for stage in "${STAGES[@]}"; do
  run_stage "$stage" || { echo "!! later stages were not run" >&2; exit 1; }
done

echo ""
echo "===== complete; artifacts under $RUN_ROOT ====="
echo "      logs         $LOG_DIR"
echo "      checkpoints  $STATE_DIR"
