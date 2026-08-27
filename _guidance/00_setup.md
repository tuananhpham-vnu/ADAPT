# 0. Cài đặt & chuẩn bị chung (đọc trước tất cả các guide khác)

Repo này là một nhánh phát triển từ **AgentPoison**: tối ưu trigger (backdoor)
trên bộ nhớ RAG của 3 agent —

| Mã agent | Agent | Entry point inference | Dataset kèm repo? |
|---|---|---|---|
| `qa` | ReAct-StrategyQA | `ReAct/run_strategyqa_gpt3.5.py`, `ReAct/run_strategyqa_llama3_api.py` | ✅ `ReAct/database/` |
| `ehr` | EhrAgent (eICU) | `EhrAgent/ehragent/main.py` | ✅ `EhrAgent/database/` |
| `ad` | Agent-Driver (nuScenes) | `agentdriver/execution/inference.py` | ❌ chỉ có `agentdriver/data/split.json` |

Thư mục `src/` (main.py rỗng + `src/providers/`) **không** thuộc pipeline này —
nó là code còn sót của một project khác; đừng dùng khi chạy các guide bên dưới.

## 0.1 Môi trường

```powershell
conda env create -f environment.yml
conda activate agentpoison
```

Hoặc dùng `.venv` sẵn có trong repo. Nếu thiếu gói, cài theo `ImportError`
(hay gặp: `jsonlines`, `ag2`/`autogen`, `sentence-transformers`, `python-dotenv`).

## 0.2 `.env`

Copy `.env.example` → `.env` (đã git-ignore) và điền:

| Biến | Cần cho |
|---|---|
| `OPENAI_API_KEY` | backbone GPT của ReAct và EhrAgent, embedding ADA |
| `REPLICATE_API_TOKEN` | backbone LLaMA-3-70B API (ReAct / EhrAgent) |
| `LLAMA2_CHAT_7B_PATH` | white-box target model cho `--target_gradient_guidance` |
| `EMBEDDER_CKPT_DIR` | checkpoint embedder fine-tune (mặc định `RAG/embedder`) |
| `FINETUNE_PLANNER_NAME` | planner GPT-3.5 đã fine-tune của Agent-Driver |
| `NUSCENES_DATAROOT` | chỉ cho visualization của Agent-Driver |
| `WANDB_PROJECT`, `WANDB_ENTITY` | chỉ khi chạy tối ưu với cờ `-w` |

Tất cả script load `.env` tự động qua `python-dotenv`.

## 0.3 Yêu cầu phần cứng

- **Tối ưu trigger (`algo/trigger_optimization.py`) bắt buộc có GPU NVIDIA**:
  device được hard-code `cuda:0` ([trigger_optimization.py:415](algo/trigger_optimization.py#L415)).
  Với DPR + GPT-2 PPL filter, cần ~15 GB VRAM ở batch 64 (giảm `-b` nếu thiếu).
- **Inference (ReAct / EhrAgent với backbone `gpt`)**: chạy được trên CPU, chỉ tốn API cost.
- **Backbone `llama3` local** (`ReAct/run_strategyqa_gpt3.5.py -b llama3`) cần GPU tải Llama-3-8B.

## 0.4 Luôn chạy từ **repo root**

Nhiều đường dẫn là relative (`ReAct/database/...`, `EhrAgent/database/...`,
`data/memory/`). Với Agent-Driver còn cần `PYTHONPATH`:

```powershell
$env:PYTHONPATH = $PWD
```

## 0.5 Cache embedding

Lần chạy đầu, DB embeddings được build và cache lại:

- `qa` → `data/memory/embeddings_<model_code>.pkl`
- `ehr` → `EhrAgent/database/embedding/embeddings_<model_code>.pkl`

Lần đầu mất vài chục phút (20k passages); các lần sau load từ cache gần như tức thì.
Xoá file `.pkl` nếu đổi embedder hoặc đổi dataset.
