# 3. Attack — tối ưu trigger và tiêm vào agent

> Đọc [00_setup.md](00_setup.md) trước. Bước tối ưu **bắt buộc có GPU NVIDIA**.
> Chỉ dùng trên dataset/agent nội bộ của nghiên cứu này.

## 3.1 Ý tưởng

`algo/trigger_optimization.py` tìm một chuỗi token ngắn (mặc định 10 token) sao cho:

1. Query có trigger bị đẩy về **một vùng riêng** trong không gian embedding của
   retriever → item độc luôn được kéo lên top-k (tối ưu MMD + khoảng cách tới
   cluster centroid, giảm variance).
2. Query **không** có trigger giữ nguyên hành vi → agent vẫn hoạt động bình thường.
3. Chuỗi trigger vẫn "tự nhiên" (perplexity thấp) → khó bị lọc bằng PPL detector.

Khởi tạo bằng golden trigger nếu bật `--golden_trigger`
([trigger_optimization.py:428](algo/trigger_optimization.py#L428)):
`ad` → `"Be careful and drive safe."`, `qa` → `"Make efficient calls."`, `ehr` → `"Be careful"`.
Không bật thì khởi tạo bằng `[MASK]` × n.

## 3.2 Tham số CLI

| Cờ | Mặc định | Ý nghĩa |
|---|---|---|
| `--agent, -a` | `ad` | `ad` / `qa` / `ehr` |
| `--algo, -al` | `ap` | `ap` (AgentPoison) hoặc `badchain` |
| `--model, -m` | `classification_user-ckpt-500` | mã embedder, xem `algo/config.py` |
| `--num_iter, -n` | 1000 | số vòng tối ưu |
| `--num_grad_iter, -g` | 30 | số bước tích luỹ gradient |
| `--num_cand, -c` | 100 | số token ứng viên mỗi vòng (HotFlip) |
| `--per_gpu_eval_batch_size, -b` | 64 | batch (giảm nếu OOM) |
| `--num_adv_passage_tokens, -t` | 10 | độ dài trigger |
| `--golden_trigger, -gt` | off | khởi tạo từ golden trigger |
| `--ppl_filter, -ppl` | off | lọc ứng viên bằng GPT-2 perplexity (tàng hình) |
| `--exclude_special` | off | cấm chọn `[CLS]/[SEP]/[MASK]/[unused*]` |
| `--target_gradient_guidance, -gg` | off | dùng loss của target LLM để hướng cập nhật |
| `--use_gpt, -u` | off | dùng GPT-3.5 làm target thay cho Llama-2-7b local |
| `--asr_threshold, -at` | 0.5 | ngưỡng ASR khi bật `-gg` |
| `--coh_sample` | off | lấy mẫu ứng viên theo `softmax(-log ppl / T)` (Eq.10 / Alg.1 dòng 7) |
| `--coh_temperature` | 1.0 | T cho `--coh_sample` (nhỏ = tham lam về PPL thấp) |
| `--coh_select_weight` | 0.0 | trọng số coherence khi chọn ứng viên |
| `--plot, -p` | off | vẽ PCA quá trình dịch chuyển embedding |
| `--report_to_wandb, -w` | off | log lên W&B (`WANDB_PROJECT` / `WANDB_ENTITY`) |

Mã embedder hợp lệ: `dpr-ctx_encoder-single-nq-base`, `ance-dpr-question-multi`,
`bge-large-en`, `realm-cc-news-pretrained-embedder`, `realm-orqa-nq-openqa`, `ada`,
và các checkpoint fine-tune (`classification_user-ckpt-500`,
`contrastive_user-random-ckpt-300`, …) — nhóm sau cần checkpoint dưới `$EMBEDDER_CKPT_DIR`.

## 3.3 Chạy tối ưu (cấu hình paper)

Có sẵn 3 script gói đúng cấu hình chuẩn:

```bash
bash scripts/react_strategyqa/run_optimization.sh   # agent qa
bash scripts/ehragent/run_optimization.sh           # agent ehr
bash scripts/agent_driver/run_optimization.sh       # agent ad
```

Cả ba đều tương đương:

```powershell
python algo/trigger_optimization.py `
  --agent qa --algo ap --model dpr-ctx_encoder-single-nq-base `
  --save_dir ./results --ppl_filter --target_gradient_guidance `
  --asr_threshold 0.5 --num_adv_passage_tokens 10 --golden_trigger -w -p
```

Kết quả: `results/<agent>/<algo>/<timestamp>/stdout.txt`
(stdout bị redirect vào file — terminal sẽ im lặng, đây là hành vi bình thường).

Theo dõi và lấy trigger:

```powershell
Get-Content -Wait -Tail 20 ./results/qa/ap/<timestamp>/stdout.txt
Select-String -Path ./results/qa/ap/*/stdout.txt -Pattern "Current adv_passage" | Select-Object -Last 1
```

## 3.4 Tiêm trigger vào agent

Trigger được **dán tay** vào script inference (mỗi file có comment
`##### Put your trigger tokens here #####`):

| Agent | File | Dòng |
|---|---|---|
| ReAct (GPT) | [ReAct/run_strategyqa_gpt3.5.py:115](ReAct/run_strategyqa_gpt3.5.py#L115) | `trigger_token_list = [...]` |
| ReAct (LLaMA-3) | [ReAct/run_strategyqa_llama3_api.py](ReAct/run_strategyqa_llama3_api.py) | tương tự |
| EhrAgent | [EhrAgent/ehragent/main.py:91](EhrAgent/ehragent/main.py#L91) | `trigger_token_list = [...]` |

Cách trigger được chèn khác nhau theo agent:
- **ReAct**: nối vào `current_context` ở bước suy luận thứ 2 của vòng ReAct
  ([run_strategyqa_gpt3.5.py:199](ReAct/run_strategyqa_gpt3.5.py#L199)); chỉ có hiệu lực khi `--task_type adv`.
- **EhrAgent**: nối vào cuối câu hỏi ([main.py:155](EhrAgent/ehragent/main.py#L155)); chỉ có hiệu lực khi có cờ `--attack`.

## 3.5 Chạy agent dưới tấn công

```powershell
mkdir result\ReAct -Force | Out-Null
python ReAct/run_strategyqa_gpt3.5.py --model dpr --task_type adv --algo ap
python ReAct/run_strategyqa_llama3_api.py --model dpr --task_type adv --algo ap

mkdir result\Ehragent\gpt -Force | Out-Null
python EhrAgent/ehragent/main.py --backbone gpt --model dpr --algo ap --attack
python EhrAgent/ehragent/main.py --backbone llama3 --model dpr --algo ap --attack
```

Bỏ `--task_type adv` / `--attack` để lấy nhánh benign đối chứng.
Chấm điểm: xem [02_evaluation.md](02_evaluation.md).

## 3.6 Baseline so sánh

`--algo badchain` chạy baseline BadChain trên cùng đường ống, dùng để đối chiếu với `ap`.

## 3.7 Ablation sweep hàng loạt

[scripts/run_ablation_sweep.sh](scripts/run_ablation_sweep.sh) chạy lưới
(agent × embedder × setting), tự xếp job lên GPU còn trống:

```bash
PYTHON=$(which python) NUM_ITER=1000 NUM_CAND=100 MAX_JOBS=4 MIN_FREE_MB=15000 \
  bash scripts/run_ablation_sweep.sh
```

- Sửa mảng `GRID` trong script để chọn tổ hợp cần chạy.
- Mặc định `NUM_ITER=30`, `NUM_CAND=50` (nhanh); paper-scale là `1000` / `100`.
- Output: `results/ppl_ablation/sweep/<label>/` + `<label>.console`; cuối script in
  bảng tóm tắt số iteration, số lỗi, và trigger cuối.
- Các dòng `ad_*` chỉ chạy khi đã có dữ liệu Agent-Driver.
- Script cần bash + `nvidia-smi` (trên Windows: WSL hoặc Git Bash + driver NVIDIA).

## 3.8 Kinh nghiệm chỉnh tham số

| Vấn đề | Cách xử lý |
|---|---|
| ASR-r thấp | tăng `--num_iter`, tăng `--num_cand`, tăng `-t` (trigger dài hơn) |
| Trigger vô nghĩa, dễ bị PPL detector bắt | bật `--ppl_filter` + `--coh_sample --coh_temperature 0.5`, hoặc tăng `--coh_select_weight` |
| Trigger chứa `[unused0]`, token rác | bật `--exclude_special` |
| Retrieval thành công nhưng agent không hành động sai (ASR-a thấp) | bật `--target_gradient_guidance` (thêm `--use_gpt` nếu không có Llama-2 local) |
| `CUDA OOM` | giảm `-b` (64 → 32 → 16), giảm `--num_cand` |
| ACC nhánh benign tụt | trigger quá "xâm lấn" — giảm `-t`, siết `--asr_threshold` |
