# AgentPoison theo từng phase và ablation thật

Chạy từ thư mục gốc repo. Trên Windows dùng `make.ps1`; mọi tham số sau target
được chuyển thẳng vào Python runner.

## 0. Kiểm tra môi trường và dữ liệu

```powershell
.\.venv-adapt\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
.\make.ps1 agentpoison-check --num-queries 10
```

Corpus mặc định là 9.251 passages trong StrategyQA. `check` xác nhận nhãn, split,
checksum và ngân sách call, không gọi model. Đặt key provider trong `.env`.

## 1. Tối ưu trigger (GPU, chạy một lần)

```powershell
.\make.ps1 agentpoison-optimize `
  --agent qa `
  --retriever-model dpr-ctx_encoder-single-nq-base `
  --num-iter 1000 --num-cand 100 --num-grad-iter 30 `
  --opt-batch-size 64 `
  --trigger-output results/triggers/qa-dpr-ap.json
```

Smoke test GPU: thay bằng `--num-iter 5 --num-cand 20 --num-grad-iter 3
--opt-batch-size 16`. Giảm batch trước nếu OOM. Phase này cần NVIDIA CUDA; không
có GPU thì bỏ qua và dùng seed trigger mặc định. Optimizer ghi `trigger.json` sau
mỗi iteration, gồm chuỗi decode đúng, token list, score và provenance.

## 2. Build/reuse index DPR thật

```powershell
.\make.ps1 agentpoison-index --device cpu --batch-size 16
```

Index tự resume theo batch và được kiểm checksum. Nếu model đã nằm trong cache,
có thể tránh các HEAD request tới Hugging Face:

```powershell
$env:HF_HUB_OFFLINE = '1'
```

## 3. Chuẩn bị attack artifacts

```powershell
.\make.ps1 agentpoison-prepare `
  --run-dir results/ap_exp_01 `
  --num-queries 50 --repeats 3 --seed 42 `
  --poison-count 2 --top-k 1 --trigger-step 2 `
  --trigger-file results/triggers/qa-dpr-ap.json
```

Phase này khóa config/split trong `manifest.json` và tạo `poison_records.json`.
Không sửa corpus gốc. Nếu bỏ `--trigger-file`, runner ghi rõ đang dùng seed chưa
tối ưu để tránh nhầm kết quả pilot với kết quả paper-scale.

## 4. Đo riêng retrieval, không gọi LLM

```powershell
.\make.ps1 agentpoison-retrieve --run-dir results/ap_exp_01
```

Đọc `retrieval_summary.json` và `retrieval_records.jsonl`. Hai số chính là
`poison_top_k_rate` và `poison_selected_rate`. So nhánh poisoned+triggered với
poisoned+untriggered để phát hiện false activation.

## 5. Chạy ReAct với LLM thật

```powershell
.\make.ps1 agentpoison-infer `
  --run-dir results/ap_exp_01 `
  --provider deepseek
```

Mỗi câu có bốn đối chứng: clean/poison memory × trigger off/on. Runner ghi từng
episode ngay vào `records.jsonl`. Chạy lại cùng lệnh sẽ bỏ qua case đã có; để chạy
lại các case API lỗi:

```powershell
.\make.ps1 agentpoison-infer --run-dir results/ap_exp_01 `
  --provider deepseek --retry-errors
```

## 6. Evaluate, không gọi lại model

```powershell
.\make.ps1 agentpoison-evaluate --run-dir results/ap_exp_01
```

Đọc `REPORT.md`, `summary.json`, `final.json`. Báo cáo tách accuracy, poison vào
top-k, poison được chọn, target IDK, số episode lỗi/chưa finish và số model calls.

## 7. Ablation study

Xem ngân sách trước:

```powershell
.\make.ps1 agentpoison-ablate --num-queries 50 --repeats 3 --plan-only
```

Pilot one-factor-at-a-time:

```powershell
.\make.ps1 agentpoison-ablate `
  --run-dir results/ablation_pilot `
  --num-queries 10 --repeats 1 --max-steps 5 `
  --poison-counts 1,2,4 --top-ks 1,3,5 --trigger-steps 1,2 `
  --trigger-file results/triggers/qa-dpr-ap.json `
  --provider deepseek
```

Run báo cáo nên tăng `--num-queries` và `--repeats`. Thêm `--factorial` chỉ khi
cần mọi tương tác giữa các factor; chi phí tăng theo tích Descartes. Tất cả
variant dùng cùng evaluation IDs và cùng poison-source pool. Tiếp tục run bị ngắt:

```powershell
.\make.ps1 agentpoison-ablate `
  --run-dir results/ablation_pilot `
  --num-queries 10 --repeats 1 --max-steps 5 `
  --poison-counts 1,2,4 --top-ks 1,3,5 --trigger-steps 1,2 `
  --trigger-file results/triggers/qa-dpr-ap.json `
  --provider deepseek --resume --retry-errors
```

Giữ nguyên toàn bộ tham số khi resume. Kết quả tổng là `ABLATION.md` và
`ablation_summary.json`; mỗi variant có đủ artifact của phases 3–6.

## Cách đọc để cải tiến

- `Poison top-k` cao nhưng `Poison selected` thấp: policy chọn top-k đang làm yếu
  attack; kiểm tra k hoặc thay chiến lược chọn.
- `Poison selected` cao nhưng `Attack IDK` thấp: payload/prompt chưa đủ mạnh; tập
  trung vào nội dung poison hoặc target-gradient guidance.
- `False select` cao: trigger thiếu tính đặc hiệu; tăng chất lượng tối ưu, đổi số
  poison hoặc thêm loss phạt query sạch.
- Clean ACC giảm: trigger/payload làm nhiễu hành vi gốc; ưu tiên cấu hình có attack
  lift dương nhưng giữ clean ACC.

