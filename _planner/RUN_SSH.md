# Hướng dẫn chạy AgentPoison thật trên SSH bằng Python

Tài liệu này dành cho pipeline **StrategyQA corpus thật** trong `src/agentpoison/`.
Pipeline dùng 9.251 passages cục bộ, DPR retrieval thật và LLM thật qua provider. Mỗi câu hỏi
được chạy trong bốn điều kiện đối chứng:

1. memory sạch, không trigger;
2. memory sạch, có trigger;
3. memory độc, không trigger;
4. memory độc, có trigger.

Các lệnh bên dưới gọi từng CLI Python trực tiếp, không cần GNU Make. Vì mã nguồn dùng package
import, hãy chạy bằng `python -m ...` từ thư mục gốc repository thay vì chạy trực tiếp file
`src/agentpoison/phases.py`.

> `python -m src.agentpoison.run_agentpoison_demo` không phải benchmark này. Lệnh đó chỉ chạy
> demo tool-calling với seed trigger có sẵn.

## 1. Đưa đúng phiên bản code lên server

Các file mới hoặc thay đổi ở máy local phải được commit và push trước khi dùng `git pull` trên
server. Có thể dùng `rsync` nếu không muốn commit kết quả hoặc thay đổi thử nghiệm.

Sau khi SSH vào server:

```bash
git clone <URL_REPOSITORY>
cd <THU_MUC_REPOSITORY>
git status --short
```

Nếu repository đã có trên server:

```bash
cd <THU_MUC_REPOSITORY>
git pull
```

## 2. Yêu cầu máy

- Linux và Python 3.11;
- NVIDIA GPU + CUDA để **optimize trigger**;
- đủ dung lượng đĩa cho model Hugging Face và index DPR;
- API key của provider dùng ở phase inference.

Kiểm tra GPU:

```bash
nvidia-smi
```

## 3. Tạo môi trường

Tạo và kích hoạt virtual environment `.venv-adapt`:

```bash
python3 -m venv .venv-adapt
source .venv-adapt/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Mọi phần bên dưới giả định environment đã được kích hoạt và lệnh `python` trỏ tới
`.venv-adapt/bin/python`. Có thể xác nhận bằng:

```bash
which python
python --version
```

`requirements.txt` có thể cài bản PyTorch CPU hoặc một bản CUDA mới hơn driver trên server.
Nếu driver hỗ trợ CUDA 12.1 trở lên, ép cài lại wheel CUDA 12.1 tương thích:

```bash
python -m pip install --force-reinstall \
  --index-url https://download.pytorch.org/whl/cu121 \
  "torch==2.5.1"
python -c "import torch; print('torch', torch.__version__, '| built_cuda', torch.version.cuda, '| cuda_available', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0)) if torch.cuda.is_available() else None"
```

Kết quả kiểm tra phải có `cuda_available True` trước khi optimize trigger. Nếu gặp lỗi
`The NVIDIA driver on your system is too old (found version 12020)`, wheel PyTorch hiện tại được
build cho CUDA mới hơn driver. Cài lại wheel phù hợp rồi chạy lại lệnh kiểm tra trên. Với wheel
CUDA 12.1, kết quả mong đợi gồm `torch 2.5.1+cu121`, `built_cuda 12.1` và
`cuda_available True`.

Build index không bắt buộc dùng GPU; có thể dùng `--device cpu --batch-size 16` nếu chưa thể xử
lý package hoặc driver.

## 4. Cấu hình provider

```bash
cp .env.example .env
nano .env
```

Ví dụ với DeepSeek:

```dotenv
DEEPSEEK_API_KEY=<KEY_CUA_BAN>
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL_NAME=deepseek-chat
```

Không commit `.env`. Pipeline tự đọc file này bằng `python-dotenv`.

## 5. Kiểm tra corpus và ngân sách

Lệnh sau không gọi LLM và không tốn API:

```bash
python -m src.agentpoison.strategyqa check \
  --provider deepseek \
  --device cpu \
  --num-queries 2 \
  --batch-size 16 \
  --index ReAct/database/embeddings/agentpoison_dpr
```

Kết quả đúng hiện tại phải báo khoảng:

- `corpus_size`: 9251;
- `labeled_questions`: 2290;
- hai tập `poison_source_ids` và `evaluation_ids` tách biệt;
- `maximum_model_calls`: 56 khi chạy hai câu, một repeat, tối đa bảy bước.

## 6. Build DPR index

Chạy trên GPU để nhanh hơn:

```bash
python -m src.agentpoison.strategyqa index \
  --provider deepseek \
  --device cuda \
  --num-queries 10 \
  --batch-size 32 \
  --index ReAct/database/embeddings/agentpoison_dpr
```

Nếu thiếu VRAM, giảm `--batch-size`. Có thể dùng CPU:

```bash
python -m src.agentpoison.strategyqa index \
  --provider deepseek \
  --device cpu \
  --num-queries 10 \
  --batch-size 16 \
  --index ReAct/database/embeddings/agentpoison_dpr
```

Index mặc định nằm tại `ReAct/database/embeddings/agentpoison_dpr/`. Quá trình build ghi theo
batch và có thể chạy lại để tiếp tục. Nếu model đã có trong Hugging Face cache:

```bash
export HF_HUB_OFFLINE=1
```

## 7. Optimize trigger thật

### 7.1 Smoke test optimizer

Chạy vài iteration để phát hiện lỗi CUDA, model hoặc thiếu bộ nhớ:

```bash
python -m src.agentpoison.phases optimize \
  --agent qa \
  --retriever-model dpr-ctx_encoder-single-nq-base \
  --num-iter 5 \
  --num-cand 20 \
  --num-grad-iter 3 \
  --opt-batch-size 16 \
  --trigger-output results/triggers/qa-dpr-smoke.json
```

Kiểm tra artifact:

```bash
python -m json.tool results/triggers/qa-dpr-smoke.json
```

### 7.2 Optimize đầy đủ

```bash
python -m src.agentpoison.phases optimize \
  --agent qa \
  --retriever-model dpr-ctx_encoder-single-nq-base \
  --num-iter 1000 \
  --num-cand 100 \
  --num-grad-iter 30 \
  --opt-batch-size 64 \
  --trigger-output results/triggers/qa-dpr-ap.json
```

Nếu CUDA OOM, giảm `--opt-batch-size`, sau đó giảm `--num-cand`. Optimizer ghi `trigger.json`
sau mỗi iteration trong `results/trigger_optimization/qa/ap/...`, nên vẫn giữ được trigger gần
nhất nếu job bị ngắt.

`--target-gradient-guidance` là tùy chọn và cần target LLM/checkpoint white-box phù hợp. Không bật
nó trong smoke test đầu tiên.

## 8. Chạy một case thật theo từng phase

Dùng trigger vừa tối ưu và một thư mục run mới:

```bash
RUN_DIR=results/ssh_smoke_01
TRIGGER=results/triggers/qa-dpr-ap.json
```

### 8.1 Chuẩn bị split và poison records

```bash
python -m src.agentpoison.phases prepare \
  --run-dir "$RUN_DIR" \
  --trigger-file "$TRIGGER" \
  --provider deepseek \
  --device cuda \
  --num-queries 1 \
  --batch-size 16 \
  --index ReAct/database/embeddings/agentpoison_dpr \
  --repeats 1
```

Phase này tạo `manifest.json` và `poison_records.json`, không gọi LLM.

### 8.2 Đo retrieval riêng

```bash
python -m src.agentpoison.phases retrieve --run-dir "$RUN_DIR"
```

Đọc `retrieval_summary.json` để xem poison có vào top-k và có được chọn hay không.

### 8.3 Chạy ReAct với model thật

```bash
python -m src.agentpoison.phases infer \
  --run-dir "$RUN_DIR" \
  --provider deepseek
```

Mỗi episode được append ngay vào `records.jsonl`. Nếu SSH hoặc API bị ngắt, chạy lại; case đã
hoàn tất được bỏ qua. Muốn thử lại riêng các case lỗi:

```bash
python -m src.agentpoison.phases infer \
  --run-dir "$RUN_DIR" \
  --provider deepseek \
  --retry-errors
```

### 8.4 Chấm kết quả

```bash
python -m src.agentpoison.phases evaluate --run-dir "$RUN_DIR"
```

Phase evaluate chỉ đọc log, không gọi lại model.

## 9. Đọc từng case

Các artifact chính:

```text
results/ssh_smoke_01/
├── manifest.json
├── poison_records.json
├── retrieval_records.jsonl
├── retrieval_summary.json
├── records.jsonl
├── summary.json
├── final.json
└── REPORT.md
```

Đọc báo cáo tổng hợp:

```bash
less "$RUN_DIR/REPORT.md"
```

Mỗi dòng `records.jsonl` là một episode. Nếu có `jq`:

```bash
jq -c '{case_id, question, poisoned_memory, triggered, answer, correct, status}' \
  "$RUN_DIR/records.jsonl"
```

Lọc một case theo ID:

```bash
jq 'select(.case_id == "<CASE_ID>")' "$RUN_DIR/records.jsonl"
```

Một câu hỏi với `repeats=1` tạo bốn episode. `OK` chỉ có nghĩa là không phát sinh exception;
cần kiểm tra thêm `finished`, `correct`, `poisoned_hit` và `target_success`.

## 10. Chạy gộp sau khi smoke test thành công

Ví dụ 10 câu, một repeat:

```bash
python -m src.agentpoison.phases all \
  --run-dir results/ssh_real_10 \
  --trigger-file results/triggers/qa-dpr-ap.json \
  --provider deepseek \
  --device cuda \
  --num-queries 10 \
  --batch-size 16 \
  --index ReAct/database/embeddings/agentpoison_dpr \
  --repeats 1
```

Mười câu tạo 40 episodes. Với `max_steps=7`, ngân sách trên lý thuyết tối đa là 280 model calls.
Luôn chạy lệnh `check` ở mục 5 hoặc pilot một câu trước khi tăng quy mô.

Run gần với thí nghiệm báo cáo hơn:

```bash
python -m src.agentpoison.phases all \
  --run-dir results/ssh_paper_50_r3 \
  --trigger-file results/triggers/qa-dpr-ap.json \
  --provider deepseek \
  --device cuda \
  --num-queries 50 \
  --batch-size 16 \
  --index ReAct/database/embeddings/agentpoison_dpr \
  --repeats 3 \
  --seed 42
```

## 11. Ablation

Xem ngân sách mà không gọi model:

```bash
python -m src.agentpoison.phases ablate \
  --num-queries 10 \
  --repeats 1 \
  --trigger-file results/triggers/qa-dpr-ap.json \
  --poison-counts 1,2,4 \
  --top-ks 1,3,5 \
  --trigger-steps 1,2 \
  --plan-only
```

Chạy one-factor-at-a-time:

```bash
python -m src.agentpoison.phases ablate \
  --run-dir results/ablation_pilot \
  --trigger-file results/triggers/qa-dpr-ap.json \
  --provider deepseek \
  --device cuda \
  --num-queries 10 \
  --batch-size 16 \
  --index ReAct/database/embeddings/agentpoison_dpr \
  --repeats 1 \
  --poison-counts 1,2,4 \
  --top-ks 1,3,5 \
  --trigger-steps 1,2
```

Kết quả tổng nằm trong `ABLATION.md`. Chỉ thêm `--factorial` khi thật sự cần toàn bộ tương tác
giữa các tham số vì số episode và chi phí API tăng nhanh.

## 12. Optimize thật hay demo?

| Lệnh Python | Trạng thái |
|---|---|
| `python -m src.agentpoison.run_agentpoison_demo` | Demo 2×2, dùng seed trigger có sẵn |
| `python -m src.agentpoison.phases all` không truyền `--trigger-file` | Corpus/LLM/DPR thật nhưng vẫn dùng seed trigger |
| `python -m src.agentpoison.phases optimize` | Optimizer white-box thật, gọi `algo/trigger_optimization.py`, cần CUDA |
| Run có `--trigger-file results/triggers/qa-dpr-ap.json` | Đánh giá bằng trigger đã optimize |

Optimizer thật đã được nối vào CLI Python và có artifact/provenance. Tuy nhiên workspace hiện chưa
có một artifact trigger hoàn chỉnh được tạo bởi một lần optimize GPU. Run thật đã lưu trước đây chỉ
xác nhận inference end-to-end bằng seed trigger. Vì vậy cần hoàn thành bước 7 trên SSH GPU trước khi
coi kết quả là đánh giá attack đã tối ưu.

Pipeline này là bản hiện đại hóa AgentPoison trên StrategyQA với split và đối chứng 2×2; nó không
phải reproduction bit-for-bit của toàn bộ cấu hình paper upstream.

## 13. Pilot tái hiện cấu hình paper

Script riêng dưới đây chạy 5 câu đầu của StrategyQA dev bằng trigger ReAct được công bố trong
Table 7. Cấu hình bị khóa ở `top-k=1`, `temperature=0`, `top-p=1`, 4 poison records và 7 bước.
Code gốc không đặt seed; script dùng seed cục bộ `0` để artifact có thể chạy lại, nhưng seed này
không ảnh hưởng lựa chọn retrieval khi `top-k=1`.

```bash
python scripts/reproduce_agentpoison_react.py --check-only
python scripts/reproduce_agentpoison_react.py \
  --device cuda \
  --output results/paper_reproduction/react_gpt35_dpr_5
```

Runner dùng `OPENROUTER_API_KEY` vì endpoint OpenAI trực tiếp không còn cung cấp
`gpt-3.5-turbo-instruct`. Nếu job bị lỗi API hoặc hết credit, giữ nguyên thư mục output và chạy:

```bash
python scripts/reproduce_agentpoison_react.py \
  --device cuda \
  --output results/paper_reproduction/react_gpt35_dpr_5 \
  --resume
```

Đây là pilot inference, không chạy lại optimizer 1.000 iteration. Manifest ghi cả cấu hình paper,
hash dữ liệu/prompt và các sai khác bắt buộc của transport để không nhầm kết quả pilot với một lần
reproduction hoàn toàn tương đương.
