# P0 lần chạy 3 — R2 với memory thêm vào có chủ đích (2026-10-04)

Thiết lập và tiêu chí được chốt **trước** khi chạy ở `open_problems.md` §"Lần chạy 3".
Lần 1: `p0_falsification_results.md`. Lần 2: `p0_run2_agentdriver_results.md`.

Kernel `dainn98s/adapt-mcat-p0-r2-targeted` v1, T4×2, chạy 04:41 → 09:16 (≈4.6 h), cả hai
lane xong (`stages: preflight ok, probes ok`). Code = `origin/tanh` `97e9800d` + overlay
đúng bằng nội dung commit `e60ca9e7`. Artifact: `outputs/kaggle/mcat-p0-r2t/`.

Thiết lập giống hệt lần 2: AgentDriver, trigger 6 token (`suffix`), 16 episode test,
2000 tài liệu + 5 poison đóng băng ở `s0`, 1000 query eval / episode, top-5, 400 step.
Trigger ở `s0` của hai lane trùng nhau (search tất định), ví dụ episode đầu:
`keynote snake takeover shady sultan nation`.

Chỉ khác: tài liệu lành thêm vào (+25 / 50 / 100%) là những tài liệu **gần query nhất**
trong phần dư của split test, chấm bằng max dot product với `Q_sup` (32 query):

- `support` — với `Q_sup` **chưa** gắn trigger (distractor tự nhiên)
- `triggered` — với `Q_sup` **đã** gắn trigger (chặn trên cho memory lành)

## Verdict theo tiêu chí đã chốt

| selection | verdict | drop `hit@5` ở +100% | CI 95% | ≥ ngưỡng 0.05? |
|---|---|---|---|---|
| `support` | `stable` | 0.00056 | [0.0000, 0.0013] | không |
| `triggered` | `stable` | 0.00056 | [0.0000, 0.0013] | không |

Theo tiêu chí 4: **`triggered` không suy giảm đáng kể → R2 đúng kể cả ở kịch bản xấu
nhất cho memory lành → bỏ hướng generator cho amortization theo memory drift.**

## Số chi tiết

### `support`

| growth | n | docs | hit@5 | mean margin | p10 | worst | off_hit |
|---|---|---|---|---|---|---|---|
| base | 16 | 2000 | 0.9844 | 20.65 | 8.98 | 0.70 | 0.0000 |
| +25% | 16 | 2500 | 0.9839 | 20.52 | 8.82 | 0.56 | 0.0000 |
| +50% | 16 | 3000 | 0.9838 | 20.50 | 8.80 | 0.55 | 0.0000 |
| +100% | 16 | 4000 | 0.9838 | 20.45 | 8.78 | 0.52 | 0.0000 |

### `triggered` (chặn trên)

| growth | n | docs | hit@5 | mean margin | p10 | worst | off_hit |
|---|---|---|---|---|---|---|---|
| base | 16 | 2000 | 0.9844 | 20.65 | 8.98 | 0.70 | 0.0000 |
| +25% | 16 | 2500 | 0.9838 | 20.51 | 8.81 | 0.57 | 0.0000 |
| +50% | 16 | 3000 | 0.9838 | 20.50 | 8.80 | 0.57 | 0.0000 |
| +100% | 16 | 4000 | 0.9838 | 20.44 | 8.75 | 0.46 | 0.0000 |

### So với thêm ngẫu nhiên (lần 2, cùng thiết lập)

| +100% | mean margin | worst margin | hit@5 |
|---|---|---|---|
| random (lần 2) | 20.46 | 0.49 | 0.9840 |
| support | 20.45 | 0.52 | 0.9838 |
| triggered | 20.44 | 0.46 | 0.9838 |

Ba cách thêm memory cho **gần như cùng một kết quả**. Ngay cả 2000 tài liệu lành gần vùng
trigger nhất cũng chỉ kéo margin trung bình xuống ~0.2 trên ~20.

## Diễn giải

- Trigger tạo ra một vùng embedding gần như **trống** tài liệu lành. Encoder và các
  passage sạch không bao giờ dịch chuyển; chỉ query-có-trigger và adv passage được kéo
  vào vùng đó. Tài liệu lành mới, dù chọn gần vùng đó nhất có thể, vẫn cách poison quá
  xa (margin ~20) để chen vào top-5.
- `hit@5` chững ở 0.9838 từ +25% trở đi: vài query vốn đã sát ngưỡng thì trượt sớm, phần
  còn lại không bị ảnh hưởng. `mean_poison_rank` (7.2 → ~14.8) tăng là do **chính các
  query đã trượt** bị đẩy xuống sâu hơn, không phải do thêm query trượt.
- Hệ quả cho MCAT: luận điểm "memory đổi → trigger cũ hỏng → cần sinh trigger mới rẻ"
  **không có cơ sở** với memory lành, trong thiết lập này.

## Giới hạn (không được nói rộng hơn số liệu)

- "Chặn trên" chỉ trong phạm vi **2520 tài liệu dư của split test**, chấm trên `Q_sup`
  (không phải `Q_eval`). Một corpus lớn hơn nhiều có thể có tài liệu gần hơn.
- Chỉ memory **lành**. Chưa thử memory chứa token của trigger (agent tự ghi tương tác bị
  trigger vào memory) hay poison của kẻ tấn công khác — hai kịch bản duy nhất còn có thể
  làm trigger suy giảm.
- Một domain (AgentDriver), một độ dài trigger (6), poison đóng băng, 1 seed tìm trigger.
- Không lưu điểm của tài liệu được thêm (`min_added_score` chỉ nằm trong snapshot note,
  không ghi ra artifact) → không đo trực tiếp được khoảng cách giữa tài liệu thêm vào và
  poison; chỉ suy ra qua margin.

## Nguồn số liệu

Đường dẫn tương đối với `outputs/kaggle/mcat-p0-r2t/mcat_p0_r2t/`.

| Số | Lấy từ |
|---|---|
| Verdict, drop, CI | `{support,triggered}/b2/seed_0/probes/growth_{sel}-suffix-test/probe_growth.json` |
| Bảng chi tiết | trung bình theo `growth` của `probe_growth.jsonl` (64 dòng = 16 × 4) mỗi lane |
| Hàng "random (lần 2)" | `outputs/kaggle/mcat-p0-ad/mcat_p0_ad/p0/b2/seed_0/probes/growth-suffix-test/probe_growth.jsonl` |
| Thời gian chạy, trạng thái | `../mcat_p0_r2t_summary.json`; log `logs/r2_{support,triggered}.log` |
