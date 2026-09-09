# Hướng dẫn chạy AgentPoison thật trên SSH bằng Makefile

Tài liệu này dành cho pipeline **StrategyQA corpus thật** trong `src/agentpoison/`.
Pipeline dùng 9.251 passages cục bộ, DPR retrieval thật và LLM thật qua provider.
Mỗi câu hỏi được chạy trong bốn điều kiện đối chứng:

1. memory sạch, không trigger;
2. memory sạch, có trigger;
3. memory độc, không trigger;
4. memory độc, có trigger.

> `make agentpoison-demo` không phải benchmark này. Target đó chỉ chạy demo tool-calling
> với seed trigger có sẵn.

## 1. Đưa đúng phiên bản code lên server

Các file mới hoặc thay đổi ở máy local phải được commit và push trước khi dùng `git pull`
trên server. Có thể thay bằng `rsync` nếu không muốn commit kết quả/thay đổi thử nghiệm.

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

- Linux, GNU Make và Python 3.11;
- NVIDIA GPU + CUDA để **optimize trigger**;
- khoảng trống đĩa cho model Hugging Face và index DPR;
- API key của provider dùng ở phase inference.

Kiểm tra GPU:

```bash
nvidia-smi
```

## 3. Tạo môi trường

Repo dùng virtual environment `.venv-adapt`:

```bash
make venv
make install
```

`requirements.txt` có thể cài bản PyTorch CPU. Nếu server dùng CUDA 12.1, cài lại bản
CUDA bằng target sau:

```bash
make torch-cu121
make check
```

Kết quả `make check` phải có `cuda True` trước khi optimize trigger. Nếu CUDA của server
không tương thích CUDA 12.1, hãy cài wheel PyTorch phù hợp với server thay vì dùng target
`torch-cu121`.

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
make agentpoison-check AP_QUERIES=2
```

Kết quả đúng hiện tại phải báo khoảng:

- `corpus_size`: 9251;
- `labeled_questions`: 2290;
- hai tập `poison_source_ids` và `evaluation_ids` tách biệt;
- `maximum_model_calls`: 56 khi chạy hai câu, một repeat, tối đa bảy bước.

## 6. Build DPR index

Chạy trên GPU để nhanh hơn:

```bash
make agentpoison-index AP_DEVICE=cuda AP_BATCH=32
```

Nếu thiếu VRAM, giảm `AP_BATCH`. Có thể dùng CPU:

```bash
make agentpoison-index AP_DEVICE=cpu AP_BATCH=16
```

Index mặc định nằm tại `ReAct/database/embeddings/agentpoison_dpr/`. Quá trình build
ghi theo batch và có thể chạy lại để tiếp tục. Nếu model đã có trong Hugging Face cache:

```bash
export HF_HUB_OFFLINE=1
```

## 7. Optimize trigger thật

### 7.1 Smoke test optimizer

Chạy vài iteration để phát hiện lỗi CUDA, model hoặc thiếu bộ nhớ:

```bash
make agentpoison-optimize ARGS="--agent qa \
  --retriever-model dpr-ctx_encoder-single-nq-base \
  --num-iter 5 --num-cand 20 --num-grad-iter 3 \
  --opt-batch-size 16 \
  --trigger-output results/triggers/qa-dpr-smoke.json"
```

Kiểm tra artifact:

```bash
python -m json.tool results/triggers/qa-dpr-smoke.json
```

### 7.2 Optimize đầy đủ

```bash
make agentpoison-optimize ARGS="--agent qa \
  --retriever-model dpr-ctx_encoder-single-nq-base \
  --num-iter 1000 --num-cand 100 --num-grad-iter 30 \
  --opt-batch-size 64 \
  --trigger-output results/triggers/qa-dpr-ap.json"
```

Nếu CUDA OOM, giảm `--opt-batch-size`, sau đó giảm `--num-cand`. Optimizer ghi
`trigger.json` sau mỗi iteration trong `results/trigger_optimization/qa/ap/...`, nên
vẫn giữ được trigger gần nhất nếu job bị ngắt.

`--target-gradient-guidance` là tùy chọn và cần target LLM/checkpoint white-box phù hợp.
Không bật nó trong smoke test đầu tiên.

## 8. Chạy một case thật theo từng phase

Dùng trigger vừa tối ưu và một thư mục run mới:

```bash
RUN_DIR=results/ssh_smoke_01
TRIGGER=results/triggers/qa-dpr-ap.json
```

### 8.1 Chuẩn bị split và poison records

```bash
make agentpoison-prepare AP_DEVICE=cuda AP_QUERIES=1 \
  ARGS="--run-dir $RUN_DIR --trigger-file $TRIGGER --repeats 1"
```

Phase này tạo `manifest.json` và `poison_records.json`, không gọi LLM.

### 8.2 Đo retrieval riêng

```bash
make agentpoison-retrieve AP_RUN="$RUN_DIR"
```

Đọc `retrieval_summary.json` để xem poison có vào top-k và có được chọn hay không.

### 8.3 Chạy ReAct với model thật

```bash
make agentpoison-infer AP_RUN="$RUN_DIR" AP_PROVIDER=deepseek
```

Mỗi episode được append ngay vào `records.jsonl`. Nếu SSH hoặc API bị ngắt, chạy lại;
case đã hoàn tất được bỏ qua. Muốn thử lại riêng các case lỗi:

```bash
make agentpoison-infer AP_RUN="$RUN_DIR" AP_PROVIDER=deepseek \
  ARGS="--retry-errors"
```

### 8.4 Chấm kết quả

```bash
make agentpoison-evaluate AP_RUN="$RUN_DIR"
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

Một câu hỏi với `repeats=1` tạo bốn episode. `OK` chỉ có nghĩa là không phát sinh
exception; cần kiểm tra thêm `finished`, `correct`, `poisoned_hit` và `target_success`.

## 10. Chạy gộp sau khi smoke test thành công

Ví dụ 10 câu, một repeat:

```bash
make agentpoison-all AP_DEVICE=cuda AP_QUERIES=10 AP_PROVIDER=deepseek \
  ARGS="--run-dir results/ssh_real_10 \
  --trigger-file results/triggers/qa-dpr-ap.json \
  --repeats 1"
```

Mười câu tạo 40 episodes. Với `max_steps=7`, ngân sách trên lý thuyết tối đa là 280
model calls. Luôn chạy `agentpoison-check` hoặc pilot một câu trước khi tăng quy mô.

Run gần với thí nghiệm báo cáo hơn:

```bash
make agentpoison-all AP_DEVICE=cuda AP_QUERIES=50 AP_PROVIDER=deepseek \
  ARGS="--run-dir results/ssh_paper_50_r3 \
  --trigger-file results/triggers/qa-dpr-ap.json \
  --repeats 3 --seed 42"
```

## 11. Ablation

Xem ngân sách mà không gọi model:

```bash
make agentpoison-ablate AP_QUERIES=10 \
  ARGS="--repeats 1 --trigger-file results/triggers/qa-dpr-ap.json --plan-only"
```

Chạy one-factor-at-a-time:

```bash
make agentpoison-ablate AP_DEVICE=cuda AP_QUERIES=10 AP_PROVIDER=deepseek \
  ARGS="--run-dir results/ablation_pilot \
  --repeats 1 --trigger-file results/triggers/qa-dpr-ap.json \
  --poison-counts 1,2,4 --top-ks 1,3,5 --trigger-steps 1,2"
```

Kết quả tổng nằm trong `ABLATION.md`. Chỉ thêm `--factorial` khi thật sự cần toàn bộ
tương tác giữa các tham số vì số episode và chi phí API tăng nhanh.

## 12. Optimize thật hay demo?

Repo hiện có cả hai loại:

| Thành phần | Trạng thái |
|---|---|
| `make agentpoison-demo` | Demo 2×2, dùng seed trigger có sẵn |
| `make agentpoison-all` không truyền `--trigger-file` | Corpus/LLM/DPR thật nhưng vẫn dùng seed trigger |
| `make agentpoison-optimize` | Optimizer white-box thật, gọi `algo/trigger_optimization.py`, cần CUDA |
| Run có `--trigger-file results/triggers/qa-dpr-ap.json` | Đánh giá bằng trigger đã optimize |

Optimizer thật đã được nối vào CLI/Makefile và có artifact/provenance. Tuy nhiên workspace
hiện chưa có một artifact trigger hoàn chỉnh được tạo bởi một lần optimize GPU. Run thật
đã lưu trước đây chỉ xác nhận inference end-to-end bằng seed trigger. Vì vậy cần hoàn thành
bước 7 trên SSH GPU trước khi coi kết quả là đánh giá attack đã tối ưu.

Pipeline này là bản hiện đại hóa AgentPoison trên StrategyQA với split và đối chứng 2×2;
nó không phải reproduction bit-for-bit của toàn bộ cấu hình paper upstream.
