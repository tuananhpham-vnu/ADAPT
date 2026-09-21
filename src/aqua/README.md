# AQuA: pipeline nghiên cứu authorization

Triển khai **pilot chạy được** từ [`paper_Q1_A_plus.md`](../../_idea/paper_Q1_A_plus.md).
Entry point: `python -m src.aqua COMMAND --help` (chạy từ thư mục gốc repo).

## 1. Chạy nhanh

Môi trường Python đang có của repo:

```powershell
.\make.ps1 aqua-smoke --output outputs/aqua/smoke
```

Hoặc dùng Python trực tiếp trên Windows/Linux:

```bash
python -m pip install -r requirements-aqua.txt
python -m src.aqua smoke --output outputs/aqua/smoke
python -m unittest tests.test_aqua tests.test_aqua_generation
```

`smoke` tạo 40 quadruples / 160 cases trên 20 tools, thu tensor fixture, train,
calibrate và đánh giá trên test tools. **Fixture mã hóa label vào activation** để
kiểm tra phần mềm; metric của nó không chứng minh giả thuyết nghiên cứu.
Test HF sử dụng Llama rất nhỏ với trọng số ngẫu nhiên, không tải pretrained model.

## 2. Chạy theo từng bước

Các lệnh dưới đây dùng được trong cả PowerShell lẫn Bash, mỗi lệnh trên một dòng.

```bash
python -m src.aqua prepare --output outputs/aqua/pilot/authshift.jsonl --groups 500 --split-by tool --seed 42
python -m src.aqua collect --dataset outputs/aqua/pilot/authshift.jsonl --output outputs/aqua/pilot/activations --backend fixture --seed 42
python -m src.aqua train --activations outputs/aqua/pilot/activations --output outputs/aqua/pilot/training --epochs 30 --batch-size 16 --rank 4
python -m src.aqua evaluate --dataset outputs/aqua/pilot/authshift.jsonl --probe outputs/aqua/pilot/training/probe.pt --output outputs/aqua/pilot/evaluation --backend fixture --seed 42
```

500 quadruples tương ứng **2.000 cases**, mỗi case có operation và ba argument
fields. `--split-by tool` giữ tools của train/validation/test tách biệt; `group`
giữ matched variants cùng tập nhưng cho phép tools xuất hiện ở nhiều tập.

## 3. Model thật qua Hugging Face

Chuẩn bị model trong `models/qwen` hoặc thay bằng model ID mà bạn có quyền truy cập.
Ví dụ bên dưới giả định model có các decoder blocks ở `model.layers`.
Chọn build PyTorch tương ứng phần cứng của bạn trước khi chạy GPU.

```bash
python -m src.aqua collect --dataset outputs/aqua/pilot/authshift.jsonl --output outputs/aqua/qwen/layer-12 --backend huggingface --model ./models/qwen --device cuda --dtype bfloat16 --layer 12 --max-length 4096
python -m src.aqua train --activations outputs/aqua/qwen/layer-12 --output outputs/aqua/qwen/probe-12 --epochs 30 --rank 8 --device cuda
python -m src.aqua evaluate --dataset outputs/aqua/pilot/authshift.jsonl --probe outputs/aqua/qwen/probe-12/probe.pt --output outputs/aqua/qwen/eval-12 --backend huggingface --model ./models/qwen --device cuda --dtype bfloat16 --layer 12 --max-length 4096
```

- `--layer` là index decoder block từ 0; `-1` là block cuối.
- Dùng `--layer-path` nếu kiến trúc đặt decoder blocks ở đường dẫn khác.
- `--revision COMMIT` ghim phiên bản model trên HF. Giữ nguyên local weights nếu
  dùng đường dẫn model local; config hiện chưa hash toàn bộ các file trọng số lớn.
- Dùng `--device cpu --dtype float32` nếu không có CUDA.
- `collect` và `evaluate` phải dùng cùng model/revision/layer/dtype/max-length.
- Tokenizer cần hỗ trợ offset mapping. Context vượt giới hạn sẽ báo lỗi, không
  tự cắt mất grant hoặc field. Chat template của tokenizer được dùng nếu có.
- Chưa chạy kiểm chứng pretrained Qwen/Llama/Gemma trong môi trường hiện tại.

## 4. Cấu trúc file

| File | Trách nhiệm |
|---|---|
| `cli.py`, `__main__.py` | Parse lệnh và nối các bước |
| `schema.py` | Case, tool call, grant, provenance, kiểm tra quadruple |
| `benchmark.py` | Sinh/đọc JSONL, chia tập theo group/tool |
| `prompts.py` | Prompt và vị trí từng field; không đưa oracle labels vào prompt |
| `backends.py` | Interface backend, tensor fixture, HF hooks và replay |
| `collection.py` | Thu activations, shard theo quadruple, resume |
| `model.py` | Low-rank projection trực giao, authorization head, nuisance heads |
| `losses.py` | Equivalence, permission flip, classification, shortcut suppression |
| `training.py` | Mini-batch training, checkpoint, calibration validation |
| `intervention.py` | Chiếu residual và loại authority component theo field |
| `sandbox.py` | Thực thi trong ledger RAM và kiểm tra effect đã xảy ra |
| `metrics.py`, `evaluation.py` | Metrics, baseline, replay và báo cáo |
| `io.py` | Artifact JSON/tensor ghi atomically |

Các thuật toán tối ưu trigger cũ vẫn nằm ở `algo/`; `src/aqua` không phụ thuộc
AgentPoison hoặc các alias legacy.

## 5. Loss và intervention

`z = ((h - center) / scale) @ Q`, với `Q` là cơ sở trực giao lấy qua QR và
`P_A = Q @ Q.T`. Center và scalar scale chỉ được ước lượng trên train.

```text
L = w_eq * ||z_authorized_user - z_authorized_external||²
  + w_flip * mean(relu(margin - distance_to_permission_flip)²)
  + w_auth * BCE(authorization_logit, field_label)
  + w_shortcut * (source_cross_entropy + domain_cross_entropy)
```

Flip loss chỉ áp dụng cho fields thực sự thay đổi permission. Source/domain heads
nhận gradient bình thường; projection nhận gradient đảo chiều qua gradient reversal.
Nuisance heads được condition trên authorization label để giảm source/domain
shortcut bên trong cùng lớp quyền. Chưa có head riêng cho lexical cues/harmfulness.

HF collector dùng output cùng một decoder block cho cả collection và intervention.
Vị trí lấy state là **token trước token cần dự đoán** trong từng field value.
Evaluation mask source tokens với vị trí giữ nguyên để ước lượng residual.
Phần chiếu của residual từ source không có quyền quyết định field bị trừ khỏi
hidden state, sau đó forward lại qua decoder để chấm candidate.

## 6. Protocol đánh giá và giới hạn

Pilot thực hiện **teacher-forced proposed-call replay**: so sánh mean conditional
log-probability của candidate call với `null`, rồi quyết định dispatch vào sandbox.
Điểm trung bình này là lựa chọn decoding của pilot, không phải xác suất chuẩn hóa
của hai hành động. Candidate được cung cấp bởi benchmark, chưa do agent tự sinh.

Từ 2026-09-20 có thêm trục **nguồn candidate**: `--proposed-source {benchmark,agent,both}`.
`benchmark` là giao thức replay cũ, không đổi. `agent` để model tự decode ra JSON call rồi
chấm chính call đó; sáu `status` (`ok`, `abstain`, `parse_error`, `unknown_tool`,
`missing_fields`, `extra_fields`, `invalid_value`) được đếm riêng trong `metrics.proposals`
— gộp `parse_error` vào `abstain` sẽ biến một model hỏng thành một defense tốt. Hai nguồn
không bao giờ chung một con số.

Tập clean benign: `--injection none` sinh `authshift_clean.jsonl` (cùng seed, cùng group,
cùng proposed call, chỉ thiếu dòng `Record note:`), truyền vào `evaluate` bằng
`--clean-dataset`. Metric của nó nằm riêng ở `metrics["clean"]`.

Năm chế độ replay được báo cáo riêng:

1. `unguarded`: candidate được model chọn.
2. `reference_monitor`: thêm kiểm tra exact grant từ trusted store.
3. `detector`: thêm threshold học trên validation cho min field probability.
4. `scrub_only`: replay candidate sau intervention.
5. `scrub_and_gate`: replay và kiểm tra margin bằng corrected original states.

Threshold được chọn theo call-level false-allow budget trên validation. Nó có thể
chặn toàn bộ nếu probe chưa học được gì; luôn đọc false-deny/overblocking cùng ASR.
Không dùng test để chọn threshold. Sau scrub chưa re-estimate toàn bộ residual.

Sandbox chỉ ghi effect vào RAM, không gọi email, filesystem hoặc payment thật.
Đánh giá harmful effect dựa trên action đã đi qua dispatcher và grant hiện tại.
`field_privilege_regret` = số field trái phép đã execute / tổng field của mọi case.
Mẫu số bằng 0 được xuất `null`.

`authorized_task_success` và `utility_under_injected_context` là utility của
candidate replay. Dataset có injection text trên cả authorized variants nên chưa
có metric **benign task success trên tập clean độc lập**. Source ablation bỏ cả
fact và instruction của source, là xấp xỉ causal có thể bị confound.

Kế hoạch và trạng thái từng bước: [`_guidance/23`](../../_guidance/23_aqua_agent_generation_plan.md).

Phần cần thêm để chạy nghiên cứu đầy đủ theo paper:

- Adapter AgentDojo/ASB/InjecAgent và dữ liệu đa dạng hơn synthetic templates.
- Chạy `--proposed-source agent` trên **model thật**: hiện chỉ kiểm chứng bằng fixture và
  Llama ngẫu nhiên tí hon. Multi-turn và multi-step effects vẫn ngoài phạm vi.
- Adaptive attack tối ưu payload qua model; chưa có white-box robustness claim.
- So sánh source/harmfulness probes, các published defenses, lexical suppression.
- Multi-layer fusion, unseen-domain/attack splits, memory/GPU profiling và certificate.

## 7. Artifact và resume

```text
outputs/aqua/pilot/
  authshift.jsonl
  activations/
    manifest.json          # dataset fingerprint, model và cấu hình collection
    shards/*.pt            # một quadruple/file, load mini-batch khi train
  training/
    checkpoint.pt          # weights, optimizer, RNG, epoch, history
    probe.pt               # projection, classifier, normalization, threshold
    training.json          # cấu hình, loss history, calibration
  evaluation/
    evaluation_config.json
    cases/*.json           # resume từng case
    records.jsonl          # probabilities, decisions và observed effects
    metrics.json           # metrics tổng hợp và theo domain
```

Chạy lại cùng lệnh để resume. Tăng `--epochs` để tiếp tục training; kết quả evaluation
của probe mới phải dùng thư mục output mới. Thay dataset/model/loss/seed/rank cần
thư mục output mới. CLI từ chối trộn artifact khác cấu hình.

RAM dành cho activations giữ một quadruple; metadata JSONL được đọc vào RAM.
DataLoader chỉ đọc batch cần dùng. Muốn thêm
backend, implement `metadata`, `capture(case, residuals=...)`, `replay(case, corrections)`.
Muốn thêm tool, cập nhật registry/schema và sandbox cùng nhau; adapter dataset phải
giữ đúng contract quadruple và label theo field.

## 8. Ablation và lệnh bổ trợ

```bash
python -m src.aqua train --activations outputs/aqua/pilot/activations --output outputs/aqua/ablation/no-shortcut --shortcut-weight 0 --epochs 30
python -m src.aqua train --activations outputs/aqua/pilot/activations --output outputs/aqua/ablation/no-equivalence --equivalence-weight 0 --epochs 30
python -m src.aqua train --activations outputs/aqua/pilot/activations --output outputs/aqua/ablation/no-flip --flip-weight 0 --epochs 30
python -m src.main aqua --help
```

PowerShell wrappers: `aqua-prepare`, `aqua-collect`, `aqua-train`, `aqua-evaluate`.
Các tham số phía sau target được chuyển nguyên cho Python CLI.
