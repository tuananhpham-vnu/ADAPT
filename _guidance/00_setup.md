# Chuẩn bị

Repo là nhánh phát triển từ AgentPoison: tối ưu trigger backdoor trên bộ nhớ RAG
của 3 agent — `qa` (ReAct/StrategyQA), `ehr` (EhrAgent/eICU), `ad` (Agent-Driver).

Dữ liệu của `qa` và `ehr` có sẵn trong repo. `ad` thì chưa — `agentdriver/data/`
mới chỉ có `split.json`.

Thư mục `src/` không thuộc pipeline này (code sót của project khác), bỏ qua.

## Môi trường

Dùng `uv` + venv, không cần conda:

```powershell
make venv          # uv venv --python 3.11 .venv-adapt
make install       # requirements.txt
make torch-cu121   # nếu có GPU NVIDIA — pip mặc định cài torch bản CPU
make check         # in torch version + cuda.is_available()
make install-ad    # chỉ khi chạy Agent-Driver
```

Không có `make` thì gõ tay:

```powershell
uv venv --python 3.11 .venv-adapt
uv pip install --python .venv-adapt/Scripts/python.exe -r requirements.txt
uv pip install --python .venv-adapt/Scripts/python.exe --index-url https://download.pytorch.org/whl/cu121 torch
```

`environment.yml` là file conda gốc của upstream AgentPoison, giữ lại để tham
chiếu pin version — không cần dùng. `.venv` hiện có trong repo là env của project
khác (autogen-agentchat, camel-ai), đừng cài đè lên đó.

Thiếu gói thì cứ cài theo `ImportError`; `requirements.txt` liệt kê phần core, các
script phụ (visualization, fine-tune) có thể cần thêm.

## `.env`

Copy `.env.example` → `.env`. Tối thiểu cần `OPENAI_API_KEY` (backbone GPT của
ReAct/EhrAgent). Thêm khi dùng đến: `REPLICATE_API_TOKEN` (backbone LLaMA-3 API),
`LLAMA2_CHAT_7B_PATH` (target model cho `--target_gradient_guidance`),
`EMBEDDER_CKPT_DIR` (checkpoint embedder tự train), `FINETUNE_PLANNER_NAME` +
`NUSCENES_DATAROOT` (Agent-Driver).

## Cần biết trước khi chạy

- Bước tối ưu trigger bắt buộc GPU NVIDIA, device hard-code `cuda:0`
  ([trigger_optimization.py:415](algo/trigger_optimization.py#L415)). ~15GB VRAM ở
  batch 64, giảm `BATCH` nếu thiếu.
- Inference với backbone `gpt` chạy được trên CPU, chỉ tốn tiền API.
- Chạy từ repo root, đường dẫn trong code là relative. Agent-Driver cần thêm
  `$env:PYTHONPATH = $PWD`.
- Lần chạy đầu build cache embedding cho 20k passage, mất 30–60 phút. Cache nằm ở
  `data/memory/` (qa) và `EhrAgent/database/embedding/` (ehr). Đổi embedder thì
  `make clean-cache`.

## Makefile

`make` (không tham số) in danh sách target. Ghi đè biến trên dòng lệnh:

```powershell
make opt-qa MODEL=bge-large-en NUM_ITER=200
make run-ehr-adv BACKBONE=llama3 NUM_Q=50
```

Biến hay dùng: `AGENT` (qa/ehr/ad), `MODEL` (mã embedder cho bước tối ưu),
`EMBEDDER` (dpr/ance/bge/realm cho bước inference), `BACKBONE` (gpt/llama3),
`NUM_ITER`, `NUM_CAND`, `BATCH`, `NUM_Q`, `VENV`, `OUT`, `RESULTS`.
