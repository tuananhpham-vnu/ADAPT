# Chạy AgentPoison với dữ liệu thật

Luồng này dùng **toàn bộ 9.251 đoạn văn** trong
`ReAct/database/strategyqa_train_paragraphs.json`, câu hỏi có nhãn từ
`strategyqa_train.json`, DPR thật và LLM thật qua provider. Không dùng 6 mẫu
ngân hàng trong `memory.py`. Database ở đây là corpus JSON và index vector trên
đĩa; không cần máy chủ SQL để chạy benchmark.

## Windows hiện tại

Máy chưa có GNU Make trong PATH. Dùng wrapper PowerShell đã thêm:

```powershell
.\make.ps1 agentpoison-check --num-queries 2
.\make.ps1 agentpoison-index
.\make.ps1 agentpoison --num-queries 2
```

## Chạy theo từng phase và resume

Runner mới tách artifact để có thể kiểm tra từng tầng thay vì phải chạy lại cả
pipeline. `optimize` chạy white-box optimizer trên GPU và xuất trigger JSON đã
decode đúng; `prepare` khóa split/config và sinh poison records; `retrieve` đo ASR-r
trên DPR thật nhưng không gọi LLM; `infer` chạy thiết kế 2×2 với model thật và tự
bỏ qua case đã hoàn tất; `evaluate` chỉ đọc log để dựng lại báo cáo.

```powershell
.\make.ps1 agentpoison-optimize --num-iter 1000 --num-cand 100 `
  --trigger-output results/triggers/qa-dpr.json
.\make.ps1 agentpoison-prepare --run-dir results/ap_run --num-queries 10 `
  --trigger-file results/triggers/qa-dpr.json
.\make.ps1 agentpoison-retrieve --run-dir results/ap_run
.\make.ps1 agentpoison-infer --run-dir results/ap_run --provider deepseek
.\make.ps1 agentpoison-evaluate --run-dir results/ap_run
```

Nếu API gián đoạn, chạy lại phase `infer`; thêm `--retry-errors` để chạy lại các
case lỗi. Lệnh gộp tương đương là:

```powershell
.\make.ps1 agentpoison-all --run-dir results/ap_run --num-queries 10
```

`optimize` cần CUDA; các phase còn lại chạy được trên CPU. Optimizer đồng thời ghi
`trigger.json` sau mỗi iteration trong thư mục run, vì vậy nếu job bị ngắt vẫn có
trigger gần nhất với token IDs, chuỗi decode và provenance. Lệnh `all` không tự
tối ưu và dùng seed trigger nếu không truyền `--trigger-file`.

## Ablation study thực tế

Mặc định runner chạy one-factor-at-a-time với số poison `1,2,4`, top-k `1,3,5`
và thời điểm chèn trigger ở bước `1,2`. Mỗi variant dùng cùng corpus thật, seed và
**chính xác cùng evaluation IDs**; các mức poison count lấy prefix từ cùng một
poison-source pool. Mỗi variant vẫn có đủ bốn nhánh đối chứng.

```powershell
# Xem số episode/call trước, không gọi model
.\make.ps1 agentpoison-ablate --num-queries 10 --plan-only

# Chạy thật; có thể giảm lưới khi pilot
.\make.ps1 agentpoison-ablate --num-queries 10 --repeats 3 --provider deepseek
.\make.ps1 agentpoison-ablate --num-queries 5 --poison-counts 1,2 --top-ks 1,3 --trigger-steps 1,2

# Full Cartesian grid thay vì one-factor-at-a-time
.\make.ps1 agentpoison-ablate --num-queries 10 --factorial

# Tiếp tục sweep bị ngắt (cấu hình phải giống plan ban đầu)
.\make.ps1 agentpoison-ablate --run-dir results/agentpoison_ablation/<run> `
  --num-queries 10 --resume --retry-errors
```

Kết quả tổng nằm ở `results/agentpoison_ablation/<timestamp>/ABLATION.md`; từng
variant có manifest, log retrieval, log inference và `REPORT.md` riêng. Dùng
`--trigger-file` để ablate trigger đã tối ưu thay cho seed trigger.

Nếu DPR đã có trong cache và muốn tránh kiểm tra mạng Hugging Face:

```powershell
$env:HF_HUB_OFFLINE = '1'
```

Biến này chỉ áp dụng cho Hugging Face; runner vẫn gọi API LLM thật.
`agentpoison-index` dùng CPU mặc định, lưu từng batch và tự tiếp tục nếu bị ngắt.
Chạy lại `agentpoison` sẽ tái sử dụng index có checksum khớp corpus. Khi thay đổi
corpus, encoder hoặc max-length, chọn thư mục index mới bằng `--index`.

## GNU Make

```bash
make agentpoison-check AP_QUERIES=2
make agentpoison-index AP_DEVICE=cpu AP_BATCH=16
make agentpoison AP_QUERIES=2 AP_PROVIDER=deepseek
make agentpoison AP_QUERIES=10 ARGS="--repeats 3 --trigger-file path/to/trigger.json"
make agentpoison-report AP_RUN=results/agentpoison_strategyqa/<timestamp>
```

Tương đương khi không dùng Make:

```bash
python -m src.agentpoison.strategyqa run --num-queries 2 --provider deepseek
python -m src.agentpoison.strategyqa report --output results/agentpoison_strategyqa/<timestamp>
```

Lệnh rút gọn `python -m src.agentpoison --num-queries 10` cũng chạy corpus thật.
`python -m src.main agentpoison ...` dùng cùng entry point. Demo ngân hàng có
tên riêng `agentpoison-demo`, không phải lựa chọn mặc định.

`.env` cung cấp key/model như các demo trước. Mỗi câu chạy bốn điều kiện:
memory sạch/độc × không/có trigger; tối đa 7 lượt gọi model mỗi episode.
Hai câu, một lần lặp có ngân sách tối đa **56 lượt gọi model**.

## Dữ liệu và phép đo

- Chọn hai câu làm nguồn tạo mẫu độc; loại chúng khỏi tập đánh giá bằng qid và
  nội dung câu hỏi. Seed và danh sách qid được lưu trong manifest.
- Toàn bộ corpus sạch được giữ nguyên; hai mẫu độc được thêm riêng trong bộ nhớ
  của lần chạy. File corpus gốc không bị sửa.
- DPR encode corpus, query và khóa mẫu độc. Với `top-k > 1`, chọn ngẫu nhiên một
  trong top-k theo cách của WikiEnv upstream; lưu cả top-k và mẫu được chọn.
- Agent chạy Thought → Search/Lookup → Observation → Finish. Trigger được thêm
  vào retrieval context từ bước 2 theo runner cũ; `trigger_applied` cho biết
  episode có thực sự đến bước đó hay đã kết thúc sớm.
- Đo accuracy, truy hồi mẫu độc và tỉ lệ trả lời mục tiêu “I don't know”. Lỗi
  được đếm riêng, không gộp thành attack thành công/thất bại. IDK ở nhánh sạch
  là đối chứng, không tự coi là bằng chứng tấn công gây ra hành vi đó.

`strategyqa_test.json` không có nhãn nên runner từ chối dùng nó để tính accuracy.
Tập đánh giá hiện là subset của file train có nhãn, tách khỏi nguồn mẫu độc;
không gọi đây là official test set hay đánh giá tổng quát hóa theo paper.

## Trigger và giới hạn tái lập

Mặc định vẫn là seed `Make efficient calls.`. **Dữ liệu thật không đồng nghĩa
trigger đã được tối ưu.** Runner này không tự thực hiện gradient optimization.
Bước tối ưu gốc nằm trong `algo/trigger_optimization.py`, các target `make opt-*`;
runner gốc hiện yêu cầu CUDA. Sau khi tối ưu, truyền chuỗi đã decode qua file:

```json
{
  "trigger": "chuỗi trigger đã decode",
  "origin": "đường dẫn run tối ưu và iteration tương ứng"
}
```

Có thể dùng file text UTF-8; provenance sẽ được ghi là chưa xác minh tối ưu.
Không dán trực tiếp danh sách token WordPiece có `##` thành câu bằng dấu cách.

Đây là runner ReAct được cập nhật để chạy provider chat và đối chứng 2×2,
không phải tái lập nguyên xi cấu hình/kết quả paper. Prompt không đặt chỉ thị
làm theo backdoor vào phần hướng dẫn tin cậy; payload chỉ vào qua retrieval.

## Kết quả

`results/agentpoison_strategyqa/<timestamp>/` chứa:

- `manifest.json`: checksum dữ liệu/index/code, model, trigger, seed, split.
- `poison_records.json`: khóa embedding và nội dung các mẫu độc được tạo.
- `records.jsonl`: từng episode, response, truy vấn retrieval, id/score/content
  của đoạn được truy hồi và câu trả lời cuối.
- `summary.json`, `final.json`, `REPORT.md`: đối chiếu bốn điều kiện.

Lệnh report đọc log, không gọi lại model. Các kết quả demo cũ trong
`results/agentpoison/` được giữ nguyên; `make agentpoison-demo` vẫn chạy demo đó.
