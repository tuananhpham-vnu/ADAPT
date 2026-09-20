# AgentPoison + retrieval-margin: kế hoạch triển khai và vận hành

Tài liệu này là hợp đồng kỹ thuật cho thí nghiệm StrategyQA so sánh AgentPoison gốc
với AgentPoison có thêm retrieval-margin loss. Hai nhánh phải dùng cùng dữ liệu, seed,
candidate search, coherence constraint và target constraint; biến độc lập duy nhất là
`margin_weight` (và `margin_top_k` ở hai thí nghiệm K=1/K=5).

## 1. Bài toán tối ưu

Optimizer dùng convention **loss càng nhỏ càng tốt**:

```text
L_uni    = -mean(||E(q + trigger) - cluster_center||_2)
L_cpt    =  mean(||E(q + trigger) - mean(E(q + trigger))||_2)
L_margin =  mean(relu(delta + best_clean_similarity - kth_poison_similarity))
L_total  =  L_uni + lambda_cpt * L_cpt + margin_weight * L_margin
```

Hai constraint không được cộng vào `L_total`:

- `L_coh`: GPT-2 contextual negative log-likelihood. HotFlip sinh `m=500` token,
  sau đó sample không hoàn lại `s=100` token theo `softmax(-L_coh / temperature)`.
- `L_tar = -target_probability`: Llama-3 teacher-force chuỗi
  `Finish[I don't know]`. Candidate khả thi khi `target_probability >= 0.8`, tương
  đương `L_tar <= -0.8`.

## 2. Input

| Input | Mặc định | Vai trò |
|---|---|---|
| Clean corpus | `ReAct/database/strategyqa_train_paragraphs.json` | Clean retrieval keys; GMM 5 components được fit trên toàn bộ raw DPR embeddings như code gốc |
| Optimization questions | `ReAct/database/strategyqa_train_filtered.json` | Poison sources, optimization, validation |
| Final dev | `ReAct/database/strategyqa_dev.json` | Chỉ dùng sau khi khóa hyperparameter |
| Retriever | `facebook/dpr-ctx_encoder-single-nq-base` | Embedding và HotFlip gradient |
| Coherence LM | `openai-community/gpt2` | `L_coh` |
| Target LM | `meta-llama/Meta-Llama-3-8B-Instruct` | `L_tar` white-box |
| Initial trigger | `Make efficient tool calls.` | Đúng 5 DPR WordPiece, không có special token |
| Target completion | `Finish[I don't know]` | Target action của poison payload |

Trước mỗi run, optimizer deduplicate optimization questions theo normalized text,
kiểm tra train/dev không overlap, chọn 5 poison-source questions theo seed, loại chúng
khỏi optimization/validation, rồi lưu QID vào `split.json`. Các arm cùng seed phải
dùng cùng split file.

## 3. CLI

Entry point validation riêng (không sửa optimizer upstream):

```bash
python -m algo.agentpoison_margin optimize \
  --agent qa --algo ap \
  --retriever-model dpr-ctx_encoder-single-nq-base \
  --retriever-device cuda:0 \
  --coherence-model openai-community/gpt2 --coherence-device cuda:0 \
  --target-model meta-llama/Meta-Llama-3-8B-Instruct --target-device cuda:1 \
  --target-load-in-4bit --target-prob-threshold 0.8 \
  --target-microbatch-size 4 \
  --replacement-candidates 500 --subsample-candidates 100 \
  --coherence-temperature 1.0 \
  --lambda-cpt 0.1 --margin-weight 0.0 --margin-value 0.1 \
  --margin-top-k 1 --num-iter 1000 --num-grad-iter 30 \
  --batch-size 64 --trigger-tokens 5 \
  --initial-trigger "Make efficient tool calls." \
  --poison-count 5 --seed 0 --output-dir <run-dir>
```

`--num-cand` là alias deprecated cho `--subsample-candidates`. `--resume` chỉ chấp
nhận checkpoint có config hash và split hash giống run hiện tại.

## 4. Data flow mỗi iteration

1. Reset gradient hook.
2. Lấy tối đa 30 batch, mỗi batch 64 câu hỏi.
3. Encode triggered queries; tính `L_uni`, `L_cpt` và, nếu được bật, `L_margin`.
4. Backward `L_total`; lấy gradient tại đúng năm trigger-token positions.
5. Chọn ngẫu nhiên một position; HotFlip lấy top `m=500` replacements.
6. GPT-2 tính contextual `L_coh`, rồi sample `s=100` candidates.
7. Tính lại tất cả loss retrieval cho 100 candidates trên cùng evaluation batches.
8. Giữ candidates cải thiện `L_total`, sort tăng dần theo loss.
9. Llama-3 chấm target probability theo thứ tự và dừng ở candidate feasible đầu tiên.
10. Nếu trigger hiện tại đã feasible thì không nhận candidate infeasible. Khi đang
    bootstrap, chỉ nhận candidate vừa cải thiện objective vừa tăng target probability.
11. Ghi một dòng `metrics.jsonl`, cập nhật `trigger.json`, rồi atomic-save checkpoint.

## 5. GPU layout

Kaggle T4 x2 dùng cố định:

- `cuda:0`: DPR, clean embeddings và GPT-2.
- `cuda:1`: Llama-3-8B-Instruct NF4 4-bit, double quantization, FP16 compute.

Không dùng `device_map="auto"`, vì nó có thể chiếm cả hai GPU. Target prompts được
teacher-force theo microbatch (mặc định 4), không gọi `generate()` trong optimizer.

## 6. Output contract

```text
outputs/agentpoison_margin/<experiment>/<arm>/seed_<n>/
|-- config.json
|-- split.json
|-- checkpoint.pt
|-- metrics.jsonl
|-- trigger.json
|-- retrieval_evaluation.json
|-- agent_evaluation.json
`-- REPORT.md
```

`config.json` chứa CLI, loss definitions, hashes dữ liệu, model revisions, phiên bản
Torch/Transformers/CUDA/bitsandbytes, GPU names, git commit và dirty status.

`checkpoint.pt` chứa trigger IDs, next iteration, best/current metrics và toàn bộ RNG
states. `trigger.json` là `running` trong checkpoint trung gian và chỉ thành
`completed` sau đủ `num_iter`.

Mỗi dòng metrics tối thiểu có: `iteration`, `token_position`, `trigger`, `l_uni`,
`l_cpt`, `l_margin`, `l_total`, `l_coh`, `perplexity`, `l_tar`,
`target_probability`, `target_feasible`, `candidate_count_m`, `candidate_count_s`
và `updated`.

## 7. Runners

```bash
bash scripts/react_strategyqa/run_optimization_validation.sh smoke
bash scripts/react_strategyqa/run_optimization_validation.sh baseline
bash scripts/react_strategyqa/run_optimization_validation.sh margin-k1
bash scripts/react_strategyqa/run_optimization_validation.sh margin-k5
```

Biến override: `SEED`, `MARGIN_WEIGHT`, `OUTPUT_ROOT`, `TARGET_MICROBATCH`, `PYTHON`.

`run_margin_comparison.sh pilot` chạy alpha `{0.1,0.5,1.0}` cho K `{1,5}` với
seed 99. Chọn alpha trên validation train-only theo thứ tự: target feasible, recall@K,
P10 margin, mean margin, rồi coherence thấp hơn. `full` chạy baseline, margin@1 và
margin@5 với seeds 0/1/2. `report` chỉ đọc artifacts.

## 8. Evaluation

Final evaluation báo cáo riêng từng seed và mean/std cho:

- `L_uni`, `L_cpt`, `L_margin`, `L_coh`, `L_tar`;
- recall@1/@5, mean/P10/worst margin, poison rank và MRR;
- ASR-r, ASR-a, ASR-t, clean ACC và false activation;
- paired delta giữa margin arm và baseline trên cùng seed/QID.

Inference nhận `--trigger-file` và `--split-file`; không yêu cầu sửa source để dán
WordPiece. Mỗi trigger chạy bốn điều kiện clean/poison memory x trigger off/on.

## 9. Nghiệm thu

Unit tests phải kiểm tra dấu/gradient của loss, K-th margin, sampling coherence,
teacher-forced target probability, target gate, gradient reset, five-token trigger,
split disjoint và deterministic resume.

Smoke trên Kaggle chỉ đạt khi hai model group nằm đúng GPU, không OOM, không có
`input()`, hoàn thành hai iterations và sinh đủ config/split/checkpoint/metrics/trigger.
Full experiment chỉ đạt khi có đủ 9 completed runs, paired split đúng, không dùng dev
để chọn alpha và có report cho cả K=1/K=5.

## 10. Runbook theo stage cho giới hạn Kaggle 12 giờ

Mọi output mới nằm dưới `outputs/agentpoison_margin/`. Một run cụ thể có dạng
`outputs/agentpoison_margin/<experiment>/<arm>/seed_<n>/`. Không đổi `OUTPUT_ROOT`,
`EXPERIMENT`, `SEED`, arm hoặc hyperparameter giữa lần chạy đầu và lần resume.

| Stage | Input bắt buộc | Output/Checkpoint | Có gọi model? |
|---|---|---|---|
| `prepare` | 3 JSON corpus/train/dev, seed, arm config | `config.json`, `split.json`, `stages/prepare.json` | Không |
| `index` | `config.json`, `split.json`, corpus | `index/clean_embeddings.npy`, `index/manifest.json`, `index/checkpoint.json` | DPR; checkpoint mỗi batch |
| `optimize` | config, split, clean index | `checkpoint.pt`, `metrics.jsonl`, `trigger.json`, `stages/optimize.json` | DPR + GPT-2 + Llama; checkpoint mỗi iteration |
| `retrieval-eval` | completed trigger, split, clean index | `retrieval_evaluation.json` | DPR, không gọi agent LLM |
| inference ngoài runner | `trigger.json`, `split.json` | JSONL có bốn điều kiện 2×2 | Có, có thể chạy ở Kaggle session riêng |
| `agent-eval` | JSONL inference qua `AGENT_RECORDS`; mỗi dòng có `poisoned_memory`, `triggered`, `correct`, `target_success`/`target_idk`, `poisoned_retrieval`/`poison_selected` | `agent_evaluation.json` | Không |
| `report` | toàn bộ artifact trên | `REPORT.md` | Không |

`split.json` là input bất biến cho mọi stage sau. `checkpoint.pt` chứa `next_iteration`,
trigger IDs, current/best metrics, RNG Python/NumPy/Torch/CUDA, `config_hash` và
`split_hash`. Resume từ chối ngay nếu một trong hai hash thay đổi. `index/checkpoint.json`
chứa `next_row`, nên stage index tiếp tục từ batch corpus cuối đã ghi.

Chạy từng stage (ví dụ baseline):

```bash
SEED=0 EXPERIMENT=full bash scripts/react_strategyqa/run_optimization_validation.sh prepare baseline
SEED=0 EXPERIMENT=full bash scripts/react_strategyqa/run_optimization_validation.sh index baseline
SEED=0 EXPERIMENT=full bash scripts/react_strategyqa/run_optimization_validation.sh optimize baseline
SEED=0 EXPERIMENT=full bash scripts/react_strategyqa/run_optimization_validation.sh retrieval-eval baseline

# Sau khi inference đã sinh records.jsonl trong một session khác:
SEED=0 EXPERIMENT=full AGENT_RECORDS=/kaggle/working/records.jsonl \
  bash scripts/react_strategyqa/run_optimization_validation.sh agent-eval baseline
SEED=0 EXPERIMENT=full bash scripts/react_strategyqa/run_optimization_validation.sh report baseline
```

Lặp lại cùng một lệnh stage sẽ tự thêm `--resume`. Để chuyển checkpoint qua các Kaggle
session, download/upload nguyên thư mục `seed_<n>`; không chỉ copy riêng `checkpoint.pt`.
Stage sau fail-fast nếu artifact của stage trước thiếu.

Smoke CPU cục bộ chạy cả baseline và margin-k1, 2 iteration, không tải model:

```bash
bash scripts/react_strategyqa/run_optimization_validation.sh smoke
```

Trên Windows không có Bash/WSL, dùng trực tiếp:

```powershell
.\.venv-adapt\Scripts\python.exe -m algo.agentpoison_margin smoke `
  --output-dir outputs/agentpoison_margin/smoke/baseline/seed_0 `
  --smoke-fixture --num-iter 2 --num-grad-iter 2 --batch-size 4 `
  --replacement-candidates 20 --subsample-candidates 5
```

Smoke fixture chỉ kiểm tra loss, split, stage dependency, atomic checkpoint, resume và
schema artifact. Trước full run vẫn phải chạy một smoke model thật trên Kaggle để xác
nhận hai GPU, quyền truy cập Llama và bộ nhớ 4-bit.
