# Chuẩn bị

Repo là nhánh phát triển từ AgentPoison: tối ưu trigger backdoor trên bộ nhớ RAG
của 3 agent — `qa` (ReAct/StrategyQA), `ehr` (EhrAgent/eICU), `ad` (Agent-Driver).

Dữ liệu của `qa` và `ehr` có sẵn trong repo. `ad` thì chưa — `agentdriver/data/`
mới chỉ có `split.json`.

Thư mục `src/` không thuộc pipeline này (code sót của project khác), bỏ qua.

## Môi trường

```powershell
conda env create -f environment.yml
conda activate agentpoison
```

Thiếu gói thì cài theo `ImportError`, hay gặp `jsonlines`, `ag2`, `sentence-transformers`.

## `.env`

Copy `.env.example` → `.env`. Tối thiểu cần `OPENAI_API_KEY` (backbone GPT của
ReAct/EhrAgent). Thêm khi dùng đến: `REPLICATE_API_TOKEN` (backbone LLaMA-3 API),
`LLAMA2_CHAT_7B_PATH` (target model cho `--target_gradient_guidance`),
`EMBEDDER_CKPT_DIR` (checkpoint embedder tự train), `FINETUNE_PLANNER_NAME` +
`NUSCENES_DATAROOT` (Agent-Driver).

## Cần biết trước khi chạy

- Bước tối ưu trigger bắt buộc GPU NVIDIA, device hard-code `cuda:0`
  ([trigger_optimization.py:415](algo/trigger_optimization.py#L415)). ~15GB VRAM ở
  batch 64, giảm `-b` nếu thiếu.
- Inference với backbone `gpt` chạy được trên CPU, chỉ tốn tiền API.
- Chạy từ repo root, đường dẫn trong code là relative. Agent-Driver cần thêm
  `$env:PYTHONPATH = $PWD`.
- Lần chạy đầu build cache embedding cho 20k passage, mất 30–60 phút. Cache nằm ở
  `data/memory/embeddings_<model_code>.pkl` (qa) và
  `EhrAgent/database/embedding/` (ehr). Đổi embedder thì xoá cache cũ.
