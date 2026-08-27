# ADAPT / AgentPoison pipeline.
# Chạy `make` để xem danh sách target.
#
# Biến ghi đè được:  make opt-qa MODEL=bge-large-en NUM_ITER=200
#                    make run-qa-adv BACKBONE=llama3

VENV      ?= .venv-adapt
ifeq ($(OS),Windows_NT)
PY        := $(VENV)/Scripts/python.exe
else
PY        := $(VENV)/bin/python
endif

AGENT     ?= qa
ALGO      ?= ap
MODEL     ?= dpr-ctx_encoder-single-nq-base
EMBEDDER  ?= dpr
BACKBONE  ?= gpt
RESULTS   ?= ./results
OUT       ?= ./result
NUM_ITER  ?= 1000
NUM_CAND  ?= 100
BATCH     ?= 64
NUM_Q     ?= -1

OPT_FLAGS ?= --ppl_filter --exclude_special --golden_trigger --num_adv_passage_tokens 10

.DEFAULT_GOAL := help

## help: liệt kê target
help:
	@$(PY) scripts/make_help.py Makefile

# ---------- setup ----------

## venv: tạo virtualenv bằng uv (Python 3.11)
venv:
	uv venv --python 3.11 $(VENV)

## install: cài deps core (qa + ehr + tối ưu trigger)
install:
	uv pip install --python $(PY) -r requirements.txt

## install-ad: cài thêm deps của Agent-Driver
install-ad:
	uv pip install --python $(PY) -r requirements-agentdriver.txt

## torch-cu121: cài lại torch bản CUDA 12.1 (mặc định pip cho bản CPU)
torch-cu121:
	uv pip install --python $(PY) --index-url https://download.pytorch.org/whl/cu121 torch

## check: in phiên bản torch và tình trạng CUDA
check:
	$(PY) -c "import torch;print('torch',torch.__version__,'| cuda',torch.cuda.is_available())"

# ---------- tối ưu trigger ----------

## opt: tối ưu trigger cho AGENT (mặc định qa). Vài giờ trên 1 GPU.
opt:
	$(PY) algo/trigger_optimization.py --agent $(AGENT) --algo $(ALGO) --model $(MODEL) \
	  --save_dir $(RESULTS) --num_iter $(NUM_ITER) --num_cand $(NUM_CAND) \
	  --per_gpu_eval_batch_size $(BATCH) $(OPT_FLAGS)

## opt-qa / opt-ehr / opt-ad: tối ưu cho từng agent
opt-qa:
	$(MAKE) opt AGENT=qa
opt-ehr:
	$(MAKE) opt AGENT=ehr
opt-ad:
	$(MAKE) opt AGENT=ad

## opt-fast: bản rút gọn ~10 phút để smoke test pipeline
opt-fast:
	$(MAKE) opt AGENT=$(AGENT) NUM_ITER=5 NUM_CAND=20 BATCH=16 RESULTS=$(RESULTS)/demo_fast

## trigger: in trigger cuối của lần tối ưu gần nhất
trigger:
	$(PY) scripts/show_trigger.py --agent $(AGENT) --algo $(ALGO) --save_dir $(RESULTS)

# ---------- inference ----------
# Nhớ dán trigger vào script inference trước khi chạy nhánh adv.

## run-qa-benign / run-qa-adv: ReAct StrategyQA
run-qa-benign: | outdirs
	$(PY) ReAct/run_strategyqa_gpt3.5.py --model $(EMBEDDER) --algo $(ALGO) --backbone $(BACKBONE) --task_type benign --save_dir $(OUT)/ReAct
run-qa-adv: | outdirs
	$(PY) ReAct/run_strategyqa_gpt3.5.py --model $(EMBEDDER) --algo $(ALGO) --backbone $(BACKBONE) --task_type adv --save_dir $(OUT)/ReAct

## run-ehr-benign / run-ehr-adv: EhrAgent (NUM_Q=-1 là chạy hết dataset)
run-ehr-benign: | outdirs
	$(PY) EhrAgent/ehragent/main.py --backbone $(BACKBONE) --model $(EMBEDDER) --algo $(ALGO) --num_questions $(NUM_Q) --save_dir $(OUT)/Ehragent
run-ehr-adv: | outdirs
	$(PY) EhrAgent/ehragent/main.py --backbone $(BACKBONE) --model $(EMBEDDER) --algo $(ALGO) --num_questions $(NUM_Q) --attack --save_dir $(OUT)/Ehragent

## run-ad: Agent-Driver (cần dữ liệu nuScenes, xem _guidance/04)
run-ad:
	$(PY) agentdriver/execution/inference.py

# ---------- evaluation ----------

## eval-qa: chấm cả hai nhánh của ReAct
eval-qa:
	$(PY) ReAct/eval.py -p $(OUT)/ReAct/$(EMBEDDER)-$(ALGO)-benign.jsonl
	$(PY) ReAct/eval.py -p $(OUT)/ReAct/$(EMBEDDER)-$(ALGO)-adv.jsonl

## eval-ehr: chấm cả hai nhánh của EhrAgent (cần GPU)
eval-ehr:
	$(PY) EhrAgent/ehragent/eval.py -p $(OUT)/Ehragent/$(BACKBONE)/$(ALGO)_benign_$(EMBEDDER).json
	$(PY) EhrAgent/ehragent/eval.py -p $(OUT)/Ehragent/$(BACKBONE)/$(ALGO)_trigger_$(EMBEDDER).json

## eval-embedder: chỉ đánh giá retriever, không tốn API
eval-embedder:
	$(PY) embedder/eval_embed_contrastive.py
	$(PY) embedder/eval_embed_classification.py

# ---------- tiện ích ----------

## sweep: chạy lưới ablation (cần bash + nvidia-smi)
sweep:
	PYTHON=$(PY) NUM_ITER=$(NUM_ITER) NUM_CAND=$(NUM_CAND) bash scripts/run_ablation_sweep.sh

outdirs:
	$(PY) -c "import os;[os.makedirs(d,exist_ok=True) for d in ('$(OUT)/ReAct','$(OUT)/Ehragent/$(BACKBONE)')]"

## clean-out: xoá kết quả inference (script append, phải xoá trước khi chạy lại)
clean-out:
	$(PY) -c "import shutil;shutil.rmtree('$(OUT)',ignore_errors=True)"

## clean-cache: xoá cache embedding DB (bắt buộc khi đổi embedder)
clean-cache:
	$(PY) -c "import shutil;shutil.rmtree('data/memory',ignore_errors=True);shutil.rmtree('EhrAgent/database/embedding',ignore_errors=True)"

.PHONY: help venv install install-ad torch-cu121 check opt opt-qa opt-ehr opt-ad \
        opt-fast trigger run-qa-benign run-qa-adv run-ehr-benign run-ehr-adv run-ad \
        eval-qa eval-ehr eval-embedder sweep outdirs clean-out clean-cache
