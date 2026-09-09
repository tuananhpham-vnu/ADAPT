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

# ---------- ARTEMIS: kiểm thử prompt của MAS (xem _guidance/10-16) ----------

ARTEMIS_DIR ?= src/artemis
SUT         ?= $(ARTEMIS_DIR)/benchmarks/test_system/6.simple_travel_planner_langgraph
ART_OUT     ?= ./output

## artemis-config: in cấu hình ba vai model và cảnh báo nếu judge trùng test
artemis-config:
	$(PY) -c "from src.config import get_settings,check_role_separation as c; s=get_settings(); print('internal',s.role('internal')); print('test    ',s.role('test')); print('judge   ',s.role('judge')); print('n_run',s.n_run,'n_judge',s.n_judge); w=c(); print('CANH BAO:',w) if w else print('hai vai da tach')"

## artemis-vendor: kéo code ARTEMIS gốc vào src/artemis (tạo commit)
artemis-vendor:
	git subtree add --prefix=$(ARTEMIS_DIR) https://github.com/hype1524/MultiAgentTesting.git HEAD --squash

## artemis-baseline: chạy phase 1 trên SUT để lấy baseline Q/S
artemis-baseline:
	$(PY) $(ARTEMIS_DIR)/run_pipeline.py --folder $(SUT) --phase1-only

# ---------- ADAPT: cải tiến của dự án ----------
ARGS ?=

# Real StrategyQA corpus (separate from the six-record banking demo).
AP_PROVIDER ?= deepseek
AP_DEVICE   ?= cpu
AP_QUERIES  ?= 10
AP_BATCH    ?= 16
AP_INDEX    ?= ReAct/database/embeddings/agentpoison_dpr
AP_RUN      ?=
AP_FLAGS    = --provider $(AP_PROVIDER) --device $(AP_DEVICE) --num-queries $(AP_QUERIES) --batch-size $(AP_BATCH) --index "$(AP_INDEX)"

## agentpoison-check: validate real corpus, labels, split and call budget (offline)
agentpoison-check:
	$(PY) -m src.agentpoison.strategyqa check $(AP_FLAGS) $(ARGS)

## agentpoison-index: build/resume DPR index for the complete real corpus
agentpoison-index:
	$(PY) -m src.agentpoison.strategyqa index $(AP_FLAGS) $(ARGS)

## agentpoison: index if needed, then run real StrategyQA + live LLM in four conditions
agentpoison:
	$(PY) -m src.agentpoison.strategyqa run $(AP_FLAGS) $(ARGS)

## agentpoison-report: rebuild report from AP_RUN without calling models
agentpoison-report:
	$(PY) -m src.agentpoison.strategyqa report --output "$(AP_RUN)"

## agentpoison-demo: demo 2x2 với seed trigger (gọi model thật)
agentpoison-demo:
	$(PY) -m src.agentpoison.run_agentpoison_demo $(ARGS)

## agentpoison-prepare: phase 1, khóa split/config và tạo poison artifacts
agentpoison-prepare:
	$(PY) -m src.agentpoison.phases prepare $(AP_FLAGS) $(ARGS)

## agentpoison-optimize: phase 0, tối ưu và xuất trigger.json (cần CUDA)
agentpoison-optimize:
	$(PY) -m src.agentpoison.phases optimize $(ARGS)

## agentpoison-retrieve: phase 2, đo retrieval thật offline cho run đã prepare
agentpoison-retrieve:
	$(PY) -m src.agentpoison.phases retrieve --run-dir "$(AP_RUN)" $(ARGS)

## agentpoison-infer: phase 3, chạy ReAct + LLM thật và có thể resume
agentpoison-infer:
	$(PY) -m src.agentpoison.phases infer --run-dir "$(AP_RUN)" --provider $(AP_PROVIDER) $(ARGS)

## agentpoison-evaluate: phase 4, chấm lại artifacts mà không gọi model
agentpoison-evaluate:
	$(PY) -m src.agentpoison.phases evaluate --run-dir "$(AP_RUN)"

## agentpoison-all: chạy liên tiếp prepare, retrieve, infer, evaluate
agentpoison-all:
	$(PY) -m src.agentpoison.phases all $(AP_FLAGS) $(ARGS)

## agentpoison-ablate: ablation one-factor-at-a-time trên corpus thật
agentpoison-ablate:
	$(PY) -m src.agentpoison.phases ablate $(AP_FLAGS) $(ARGS)

## adapt-plan: ước lượng ngân sách, không gọi model
adapt-plan:
	$(PY) -m src.adapt --plan $(ARGS)

## adapt-fixture: kiểm thử pipeline bằng backend tổng hợp
adapt-fixture:
	$(PY) -m src.adapt --backend fixture $(ARGS)

## adapt-live: chạy cải tiến trên model thật
adapt-live:
	$(PY) -m src.adapt --backend live $(ARGS)

## gate-demo: pilot kiểm tra quyền thực thi tool
gate-demo:
	$(PY) -m src.adapt.run_gate_demo $(ARGS)

# Các target improve-loop/robustness-eval cũ trỏ tới module chưa tồn tại.
# prompt_improve là API callback; pipeline CLI hiện tại là src.adapt.

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
        eval-qa eval-ehr eval-embedder sweep outdirs clean-out clean-cache \
        artemis-config artemis-vendor artemis-baseline agentpoison-demo adapt-plan adapt-fixture adapt-live gate-demo

.PHONY: agentpoison-check agentpoison-index agentpoison agentpoison-report
.PHONY: agentpoison-optimize agentpoison-prepare agentpoison-retrieve agentpoison-infer agentpoison-evaluate agentpoison-all agentpoison-ablate
