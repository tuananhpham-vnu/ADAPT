# Demo nhanh

Chạy hết vòng lặp (tối ưu trigger → tiêm → đo) trong ~20 phút để kiểm tra pipeline
còn sống. Không dùng số này để báo cáo. Dùng agent `qa` vì dataset có sẵn.

## 1. Tối ưu trigger, iteration cực ngắn

```powershell
python algo/trigger_optimization.py `
  --agent qa --algo ap --model dpr-ctx_encoder-single-nq-base `
  --num_iter 5 --num_cand 20 --num_grad_iter 3 --per_gpu_eval_batch_size 16 `
  --golden_trigger --ppl_filter --exclude_special `
  --save_dir ./results/demo_fast
```

Script redirect stdout vào file nên terminal sẽ không in gì — theo dõi ở
`results/demo_fast/qa/ap/<timestamp>/stdout.txt`:

```powershell
Get-ChildItem -Recurse ./results/demo_fast -Filter stdout.txt |
  Sort-Object LastWriteTime | Select-Object -Last 1 | Get-Content -Wait -Tail 20
```

Thấy `Init adv_passage [...]` rồi các dòng `Iteration:` kèm `Current adv_passage`
đổi dần là được. Lấy trigger cuối:

```powershell
Select-String -Path ./results/demo_fast/qa/ap/*/stdout.txt -Pattern "Current adv_passage" |
  Select-Object -Last 1
```

## 2. Dán trigger vào agent

Sửa `trigger_token_list` ở [ReAct/run_strategyqa_gpt3.5.py:115](ReAct/run_strategyqa_gpt3.5.py#L115),
thay list placeholder bằng token vừa lấy.

## 3. Chạy inference rồi Ctrl+C

```powershell
mkdir result\ReAct -Force | Out-Null
python ReAct/run_strategyqa_gpt3.5.py --model dpr --algo ap --task_type adv
```

Mỗi câu xong là append ngay một dòng vào `result/ReAct/dpr-ap-adv.jsonl`, nên cứ
để chạy 30–50 dòng rồi Ctrl+C. File mở chế độ append — xoá file cũ trước mỗi lần
demo lại, không thì số của hai lần chạy cộng dồn.

## 4. Đo

```powershell
python ReAct/eval.py -p ./result/ReAct/dpr-ap-adv.jsonl
```

## Khi vướng

- Terminal im lặng lúc tối ưu: bình thường, xem `stdout.txt`.
- `CUDA out of memory`: giảm `-b` còn 8, giảm `--num_cand`.
- Treo lâu ở đoạn load DB: đang build cache embedding lần đầu.
- `ModuleNotFoundError: algo`: không chạy từ repo root.

Không có GPU thì bỏ bước 1, giữ nguyên golden trigger `"Make efficient calls."`
và chạy thẳng bước 3–4 — chỉ demo được đường ống, ASR sẽ thấp.
