# P0 lần chạy 2 — AgentDriver, trigger 6 token (2026-10-04)

Thiết lập và tiêu chí đọc được chốt **trước** khi chạy ở `open_problems.md` §"Lần chạy 2".
Lần chạy 1 (qa, chạm trần) vẫn giữ nguyên ở `p0_falsification_results.md`.

Kernel `dainn98s/adapt-mcat-p0-ad` v1, T4×2, DPR `facebook/dpr-ctx_encoder-single-nq-base`
(rev `bb21a3c`). Code = clone `ad22e9fc` + overlay đúng bằng nội dung commit `a43e3156`.
Artifact: `outputs/kaggle/mcat-p0-ad/`.

| | Giá trị |
|---|---|
| Domain | `ad` — key `ego + perception`, `MAX_LENGTH=512` |
| Trigger | **6 token**, vị trí `suffix` |
| Episode test | 16 (đơn vị bootstrap) |
| Memory / episode | 2000 tài liệu lành + 5 poison (`poison_fraction` 0.0025), top-5 |
| Query eval / episode | 1000 (trên 1379 query của split test) |
| Tối ưu | 400 step, 16 query tối ưu, poison `refresh` |
| R2 drift | +25 / 50 / 100% (2500 / 3000 / 4000 tài liệu), 5 drift seed |

## Trạng thái chạy

Kernel bị Kaggle dừng ở giới hạn 12 h (`CANCEL_ACKNOWLEDGED`) trước khi ghi summary.
Các probe ghi artifact riêng nên vẫn đọc được:

| Probe | Trạng thái |
|---|---|
| R2 | ✅ xong |
| R1 | ✅ xong |
| P1 `transfer` | ✅ xong (kèm attention) |
| P1 `reoptimize` | ❌ dở — vị trí `suffix` mới tới step 13/400. **Chưa có số.** |

## Kết quả theo tiêu chí đã chốt

| | Verdict từ code | Cờ trần (base `hit@5` ≥ 0.95 ở cả B2, B3) | Đọc theo tiêu chí |
|---|---|---|---|
| **R2** | `stable` | **có** (base 0.984) | chạm trần — không phân biệt được |
| **R1** | `universal-suffices` | **có** (B2 0.978, B3 0.996) | chạm trần — không phân biệt được |

Theo luật 3: R2 chỉ bị bác khi `decays`, R1 chỉ khi `conditioning-helps`. **Không có cái
nào bị bác bỏ.** Không train generator.

### R2 — memory phình, trigger + poison đóng băng

| growth | n | docs | on_hit | mean margin | p10 margin | worst margin | off_hit |
|---|---|---|---|---|---|---|---|
| base | 16 | 2000 | 0.9844 | 20.65 | 8.98 | 0.70 | 0.0000 |
| +25% | 80 | 2500 | 0.9844 | 20.58 | 8.92 | 0.63 | 0.0000 |
| +50% | 80 | 3000 | 0.9842 | 20.54 | 8.88 | 0.58 | 0.0000 |
| +100% | 80 | 4000 | 0.9840 | 20.46 | 8.79 | 0.49 | 0.0000 |

- Drop của `on_hit` ở +100%: **0.0004**, CI [0.0000, 0.0009], seed_spread 0.0002.
- Margin giảm đơn điệu nhưng chỉ ~1% khi memory **gấp đôi**. Trigger cũ vẫn chạy gần
  như y nguyên.
- `on_hit` theo episode ở base: 14/16 episode ≥ 0.98; hai episode yếu nhất 0.917 và 0.851.

### R1 — B2 per-episode vs B3 universal, cùng 400 step

| arm | hit@5 | mean margin | p10 margin | worst margin | MRR | false act. | round-trip |
|---|---|---|---|---|---|---|---|
| B2 `direct-logit` | 0.9781 | 15.58 | 7.17 | −0.16 | 0.975 | 0.0000 | 1.00 |
| B3 `universal-logit` | **0.9959** | 14.43 | 7.57 | **1.41** | **0.994** | 0.0000 | 1.00 |

- Hiệu ghép cặp B2 − B3 trên `hit@5`: **−0.0178, CI [−0.0359, −0.0032], có ý nghĩa** —
  trigger **phổ quát tốt hơn** trigger per-episode, ngược chiều với điều MCAT cần.
- B2 chỉ hơn ở mean margin (+1.15, thắng 9/16 episode), nhưng thua ở worst margin và hit:
  B2 có những episode tụt (0.868, 0.924, 0.929) mà B3 không có (thấp nhất 0.977).
- Cách đọc khả dĩ: B3 tối ưu trên query của 16 episode train, B2 trên 16 query của một
  episode → B2 overfit. Không có dấu hiệu thông tin riêng của memory giúp ích.

### P1 `transfer` — một trigger (tối ưu ở suffix) chấm ở mọi vị trí, cùng 6 token

| vị trí | on_hit | vs suffix (paired) | off_hit | [CLS] attention vào trigger |
|---|---|---|---|---|
| suffix | 0.9844 | — | 0.0000 | 0.219 |
| prefix | 1.0000 | +0.016 [0.0001, 0.038] ✱ | 0.0000 | 0.241 |
| both | 1.0000 | +0.016 [0.0001, 0.038] ✱ | 0.0000 | **0.268** |
| middle | **0.9031** | **−0.081 [−0.129, −0.034] ✱** | 0.0000 | **0.165** |

✱ = khoảng tin cậy không chứa 0. Attention đo trên 8 query/vị trí (layer cuối, trung
bình các head).

- **Đây là phát hiện duy nhất không chạm trần:** chèn vào **giữa** query làm ASR tụt
  rõ (−8.1 điểm), và đúng là vị trí đó nhận ít attention từ `[CLS]` nhất. Thứ tự
  attention (both > prefix > suffix > middle) khớp với thứ tự ASR.
- Đây là chế độ `transfer` (trigger không được tối ưu lại cho vị trí mới). Câu "vị trí
  nào thực sự tốt hơn" cần `reoptimize`, mà lần này chưa chạy xong.
- Attention chỉ trên 8 query → mới là tín hiệu, chưa đủ để claim.

## So với lần chạy 1

| | Lần 1 (qa, 10 token, 512 docs) | Lần 2 (ad, 6 token, 2000 docs) |
|---|---|---|
| Base `hit@5` | 1.000 | 0.984 |
| Mean margin (R2 base) | 52.3 | 20.7 |
| False activation | 0.25 – 0.69 | **0.000** |
| R1 | universal ≥ per-episode | universal **>** per-episode (có ý nghĩa) |

- Bài toán khó hơn (margin giảm ~2.5×) nhưng **vẫn chạm trần**.
- False activation cao ở lần 1 **không xuất hiện** trên AgentDriver: là đặc thù của
  cách dựng poison cho qa, không phải lỗi chung của pipeline.

## Nguồn số liệu (đối chiếu từ artifact)

Đường dẫn tương đối với `outputs/kaggle/mcat-p0-ad/mcat_p0_ad/`.

| Số | Lấy từ |
|---|---|
| Verdict R2, drop + CI, seed_spread | `p0/b2/seed_0/probes/growth-suffix-test/probe_growth.json` |
| Bảng R2 | trung bình theo `growth` của `probe_growth.jsonl` (256 dòng = 16 + 3×16×5) |
| Verdict R1, paired −0.0178 [−0.0359, −0.0032] | `p0_r1/b2/seed_0/probe_universal.json` |
| Bảng R1 | trung bình 16 dòng của `p0_r1/{b2,b3}/seed_0/evaluation.jsonl` |
| Bảng P1 + attention | `p0/b2/seed_0/probes/position-transfer-suffix-test/probe_position.json` |
| Tiến độ `reoptimize` | `p0/b2/seed_0/probes/position-reoptimize-suffix-test/suffix/metrics.jsonl` (14 dòng) |
| Kiểm tra attention eager | `logs/attention_check.log` (`eager attentions 12`) |

## Kết luận

1. Hai lần chạy, hai domain, hai độ dài trigger: **cả hai đều chạm trần**, và ở lần 2
   trigger phổ quát còn **thắng có ý nghĩa** trigger per-episode. Không có bằng chứng
   nào ủng hộ memory-conditioning hay amortization trong chế độ white-box này.
2. Phát hiện chắc nhất hiện có là **P1: trigger ở giữa query yếu hơn rõ rệt**, kèm bằng
   chứng attention cùng chiều.
3. Nếu còn muốn tìm chế độ mà MCAT có chỗ đứng thì phải giảm thêm sức mạnh trigger
   (2–4 token) hoặc thêm ràng buộc (stealth / perplexity), và lại chốt trước khi chạy.
