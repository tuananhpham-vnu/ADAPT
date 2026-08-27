# 1. Demo nhanh (~15–30 phút)

Mục tiêu: chứng minh **toàn bộ vòng lặp chạy được** (tối ưu trigger → tiêm vào
agent → đo ASR) với chi phí nhỏ nhất. Không nhằm tái lập số liệu paper.

> Đọc [00_setup.md](00_setup.md) trước. Mọi lệnh chạy từ repo root.
> Dùng agent `qa` (ReAct-StrategyQA) vì dataset đã có sẵn trong repo.

## Bước 1 — Smoke test: tối ưu trigger cực ngắn (GPU, ~5–10 phút)

```powershell
python algo/trigger_optimization.py `
  --agent qa `
  --algo ap `
  --model dpr-ctx_encoder-single-nq-base `
  --num_iter 5 `
  --num_cand 20 `
  --num_grad_iter 3 `
  --per_gpu_eval_batch_size 16 `
  --num_adv_passage_tokens 10 `
  --golden_trigger `
  --ppl_filter `
  --exclude_special `
  --save_dir ./results/demo_fast
```

Lưu ý quan trọng: script **redirect stdout vào file**, terminal sẽ im lặng.
Theo dõi tiến độ ở:

```powershell
Get-ChildItem -Recurse ./results/demo_fast -Filter stdout.txt |
  Sort-Object LastWriteTime | Select-Object -Last 1 | Get-Content -Wait -Tail 20
```

Đường dẫn kết quả: `results/demo_fast/qa/ap/<timestamp>/stdout.txt`.

Điều cần thấy:
- `Init adv_passage [...]` — khởi tạo từ golden trigger `"Make efficient calls."`
- các dòng `Iteration: k` và `Current adv_passage ...` — trigger đang được cập nhật.

Lấy trigger cuối cùng:

```powershell
Select-String -Path ./results/demo_fast/qa/ap/*/stdout.txt -Pattern "Current adv_passage" |
  Select-Object -Last 1
```

Nếu bước này chạy trót lọt → pipeline tối ưu OK. **5 iteration không đủ để có
trigger mạnh**; số thật xem [03_attack.md](03_attack.md).

## Bước 2 — Dán trigger vào agent

Mở [ReAct/run_strategyqa_gpt3.5.py:115](ReAct/run_strategyqa_gpt3.5.py#L115) và
thay list placeholder bằng token vừa lấy:

```python
##### Put your trigger tokens here #####
trigger_token_list = ['make', 'efficient', 'calls', ...]   # copy từ stdout.txt
```

Token đặc biệt `[CLS] [SEP] [MASK]` sẽ tự bị lọc ở dòng ngay dưới.

## Bước 3 — Chạy inference vài chục câu rồi dừng

```powershell
mkdir result\ReAct -Force | Out-Null   # script không tự tạo thư mục output
python ReAct/run_strategyqa_gpt3.5.py --model dpr --algo ap --task_type adv --save_dir ./result/ReAct
```

Script append từng dòng vào `result/ReAct/dpr-ap-adv.jsonl` **ngay sau mỗi câu**,
nên có thể `Ctrl+C` sau ~30–50 dòng mà file kết quả vẫn đọc được.

> File mở ở chế độ `"a"` (append). Xoá file cũ trước mỗi lần demo lại, nếu không
> số liệu của các lần chạy sẽ bị cộng dồn.

## Bước 4 — Đo ngay

```powershell
python ReAct/eval.py -p ./result/ReAct/dpr-ap-adv.jsonl
```

In ra `Accuracy`, `ASR-r`, `ASR-a`, `ASR-t`. Ý nghĩa từng chỉ số:
xem [02_evaluation.md](02_evaluation.md).

## Biến thể demo nhanh khác

| Muốn demo | Lệnh |
|---|---|
| Baseline sạch (không trigger) để so sánh | `python ReAct/run_strategyqa_gpt3.5.py --model dpr --algo ap --task_type benign` |
| Agent y tế thay vì QA | `python EhrAgent/ehragent/main.py --backbone gpt --model dpr --algo ap --attack --num_questions 10` |
| Không có GPU | Bỏ Bước 1, giữ nguyên trigger placeholder hoặc dùng golden trigger `"Make efficient calls."`, chạy thẳng Bước 3–4 (chỉ demo được đường ống, ASR sẽ thấp) |

## Lỗi hay gặp

| Triệu chứng | Nguyên nhân / cách xử lý |
|---|---|
| Terminal không in gì khi tối ưu | Bình thường — stdout đã bị redirect, xem `stdout.txt` |
| `CUDA out of memory` | Giảm `-b` xuống 16/8, giảm `--num_cand` |
| Lần chạy đầu treo rất lâu ở phần load DB | Đang build cache embedding 20k passage — chờ, lần sau nhanh |
| `openai.AuthenticationError` | Thiếu `OPENAI_API_KEY` trong `.env` |
| `ModuleNotFoundError: algo` | Đang chạy không từ repo root |
