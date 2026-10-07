# P0 lần chạy 6 — chạy lại 2–5 với hit@1/2/3/5, thêm arm out-of-domain (2026-10-06)

Thiết lập và tiêu chí được chốt **trước** khi chạy ở `open_problems.md` §"Lần chạy 5" và
§"Lần chạy 6". Lần 5 (kernel `adapt-mcat-p0-writeback` v1) lỗi preflight vì CRLF, không có số
liệu, nên được gộp vào lần này.

Kernel `dainn98s/adapt-mcat-p0-rerun` v2, T4×2, chạy 2026-10-05 16:33 → 21:01 (≈4.5 h), preflight
và cả 8 probe đều `ok`. Artifact: `outputs/kaggle/mcat-p0-rerun/`.

Thiết lập chung: AgentDriver, DPR, trigger 6 token (`suffix`), 2000 tài liệu, 5 poison đóng
băng ở `s0`, 16 episode test, 1000 query eval / episode, top-5, 400 step. Mọi `off_hit` = **0**
(không có kích hoạt nhầm khi query không mang trigger, ở mọi arm và mọi mức).

## 1. Kiểm tra tái lập (tiêu chí 1): khớp tuyệt đối

So từng dòng (theo `snapshot_id`) với artifact của lần 2–4:

| arm | dòng | trigger trùng | max \|Δ hit@5\| | max \|Δ off_hit\| | max \|Δ margin\| |
|---|---|---|---|---|---|
| random (lần 2) | 256 | 256/256 | 0 | 0 | 0 |
| support (lần 3) | 64 | 64/64 | 0 | 0 | 0 |
| triggered (lần 3) | 64 | 64/64 | 0 | 0 | 0 |
| self (lần 4) | 256 | 256/256 | 0 | 0 | 0 |
| rival (lần 4) | 208 | 208/208 | 0 | 0 | 0 |

Ba thay đổi (cache query, dùng chung lần tìm `s0`, thêm `hit_curve`) **không làm đổi một con số
nào**. Các bảng của lần 2–4 vẫn giữ nguyên giá trị; lần này chỉ thêm hit@1/2/3.

## 2. Memory benign: in-domain và out-of-domain

Base (chung mọi arm): hit@1 0.9813 · hit@2 0.9831 · hit@3 0.9839 · hit@5 0.9844 · margin 20.65.

| arm | +100% (4000 docs): hit@1 / hit@2 / hit@3 / hit@5 | drop hit@5 (CI) | drop hit@1 (CI) | margin | verdict |
|---|---|---|---|---|---|
| random in-domain (5 seed) | 0.9811 / 0.9826 / 0.9833 / 0.9840 | 0.0004 [0, 0.0009] | 0.0002 [0, 0.0004] | 20.46 | `stable` |
| support (gần `Q_sup` sạch) | 0.9811 / 0.9824 / 0.9831 / 0.9838 | 0.0006 [0, 0.0013] | 0.0002 [0, 0.0005] | 20.45 | `stable` |
| triggered (gần `Q_sup` có trigger) | 0.9811 / 0.9824 / 0.9831 / 0.9838 | 0.0006 [0, 0.0013] | 0.0003 [0, 0.0006] | 20.44 | `stable` |
| **ood: benign QA (5 seed)** | **0.9813 / 0.9831 / 0.9839 / 0.9844** | **0** [0, 0] | **0** [0, 0] | **20.65** | `stable` |

**OOD so với in-domain** (tiêu chí 3: hiệu ghép cặp `drop_ood − drop_random`, bootstrap theo episode):

| mức | hit@5 | hit@1 | mean margin |
|---|---|---|---|
| +25% | −0.0000 [−0.0000, 0.0000] | −0.0001 [−0.0001, 0.0000] | **−0.065** [−0.074, −0.056] |
| +50% | −0.0001 [−0.0003, 0.0000] | −0.0001 [−0.0001, 0.0000] | **−0.109** [−0.126, −0.095] |
| +100% | −0.0004 [−0.0009, 0.0000] | −0.0002 [−0.0004, 0.0000] | **−0.185** [−0.216, −0.161] |

- Theo hit@5 và hit@1: CI chạm 0, nên ghi là **không phân biệt được**. Cả hai arm đều không làm
  ASR giảm.
- Theo margin: OOD **yếu hơn có ý nghĩa**, đúng dự đoán đã chốt. Thực ra margin của OOD **không
  đổi một chút nào** (20.65 ở mọi mức và mọi seed). Nghĩa là không một tài liệu QA nào lọt vào
  top-5 của bất kỳ query mang trigger nào.

Trả lời câu hỏi đặt ra:
- Thêm benign in-domain làm ASR giảm **0.04%** ở +100%, tức không đáng kể.
- Thêm OOD làm giảm **0%**: yếu hơn in-domain, nhưng cả hai đều ở mức bằng 0 về thực tế.

## 3. Memory adversarial: hit@k cho thấy thứ hit@5 che mất

### `self`: bản ghi mang trigger của mình, nhãn lành (`decays`)

| bản ghi | hit@1 | hit@2 | hit@3 | hit@5 | drop hit@1 (CI) | drop hit@5 (CI) |
|---|---|---|---|---|---|---|
| 0 | 0.981 | 0.983 | 0.984 | 0.984 | — | — |
| 1 | **0.840** | 0.983 | 0.984 | 0.984 | **0.142** [0.105, 0.183] | 0.000 |
| 2 | 0.721 | 0.934 | 0.983 | 0.984 | 0.260 [0.200, 0.326] | 0.000 |
| 5 | 0.501 | 0.740 | 0.876 | 0.980 | 0.481 [0.410, 0.547] | 0.004 [0.001, 0.010] |
| 10 | 0.345 | 0.550 | 0.714 | 0.875 | 0.636 [0.565, 0.703] | 0.109 [0.063, 0.162] |
| 25 | 0.198 | 0.327 | 0.433 | 0.598 | 0.783 [0.721, 0.842] | 0.386 [0.306, 0.466] |

Mức nhỏ nhất mà drop ≥ 0.05 (tiêu chí 4): hit@1 ở **1** bản ghi; hit@2 ở 5; hit@3 ở 5; hit@5 ở
10. Đúng dự đoán: bản ghi mang trigger **chiếm hạng 1 trước**, rồi mới đẩy poison ra khỏi top-5.
Chỉ **một** bản ghi đã làm hit@1 tụt 14 điểm.

### `rival`: poison của kẻ tấn công khác (`decays`, hit@5 nhỏ)

| kẻ tấn công | poison lạ | hit@1 | hit@5 | drop hit@1 (CI) | drop hit@5 (CI) |
|---|---|---|---|---|---|
| 1 | 5 | 0.979 | 0.984 | 0.002 [0.001, 0.004] | 0.000 |
| 3 | 15 | 0.969 | 0.984 | 0.013 [0.004, 0.023] | 0.000 |
| 7 | 35 | 0.933 | 0.983 | 0.048 [0.018, 0.088] | 0.001 [0, 0.004] |
| 15 | 75 | **0.889** | 0.980 | **0.092** [0.039, 0.157] | 0.004 [0.000, 0.010] |

Lần 4 kết luận `rival` "nhỏ" dựa trên hit@5. Theo hit@1 thì 15 kẻ tấn công làm tụt **9.2 điểm**
(vượt ngưỡng 0.05 ở mức 15). Với agent chỉ lấy top-1, trigger của người khác **có** ảnh hưởng.

## 4. Agent tự ghi lại theo chính sách ghi (lần 5, vòng kín)

Mỗi chính sách chạy trên cùng stream 50 tương tác có trigger, 3 seed. "Thay đổi" = hit@5 ở mức
50 trừ base (âm = yếu đi), CI ghép cặp theo episode.

### Arm chính, 5 poison (base hit@5 0.984)

| chính sách | hit@1 / hit@5 sau 50 lần | thay đổi hit@5 (CI) | verdict | `cleanup_hit` | bản ghi độc / lành đã ghi |
|---|---|---|---|---|---|
| `none` | 0.981 / 0.984 | 0 | `stable` | 0 | 0 / 0 |
| `log_outcome` | 0.999 / **1.000** | +0.016 [0.000, 0.038] | `significant-but-small` (chạm trần) | **1.000** | 49.9 / 0.1 |
| `verified` | 0.910 / 0.960 | −0.025 [−0.062, 0.000] | `stable` | 0 | 0 / 1.2 |
| `corrected` | 0.096 / **0.360** | **−0.625** [−0.702, −0.541] | `dilutes` | 0 | 0 / 50 |

### Arm ngoài trần, 1 poison (base hit@5 0.931)

| chính sách | hit@1 / hit@5 sau 50 lần | thay đổi hit@5 (CI) | verdict | `cleanup_hit` | bản ghi độc / lành đã ghi |
|---|---|---|---|---|---|
| `none` | 0.925 / 0.931 | 0 | `stable` | 0 | 0 / 0 |
| `log_outcome` | 0.994 / **0.994** | **+0.063** [0.022, 0.115] | **`reinforces`** | **0.994** | 49.6 / 0.4 |
| `verified` | 0.658 / 0.818 | **−0.114** [−0.221, −0.031] | `dilutes` | 0 | 0 / 5.8 |
| `corrected` | 0.022 / **0.104** | **−0.827** [−0.899, −0.743] | `dilutes` | 0 | 0 / 50 |

Đối chiếu với các dự đoán đã ghi trước:

| | dự đoán | kết quả |
|---|---|---|
| H1 | `corrected` → `dilutes` | ✅ cả hai arm (−0.63 và −0.83) |
| H2 | `verified` → `stable` | ✅ arm 5 poison; ❌ **arm 1 poison: `dilutes`** (−0.11) |
| H3 | `log_outcome` không `dilutes`; arm ngoài trần `reinforces`; tồn tại sau khi dọn | ✅ cả ba |
| Claim chính (tiêu chí 3) | cùng arm có policy `dilutes` và policy không `dilutes`, CI không chồng nhau | ✅ cả hai arm: `corrected` [−0.70, −0.54] so với `log_outcome` [0.000, 0.038]; và [−0.90, −0.74] so với [0.022, 0.115] |

**Claim chính đứng được: chính sách ghi quyết định chiều của hiệu ứng.**

Diễn giải:
1. **`log_outcome`** (agent lưu mọi kết quả làm kinh nghiệm): mỗi lần trigger nổ sinh thêm một
   bản ghi độc. Sau 50 lần đã có khoảng 50 bản ghi độc, tức gấp 10 lần poison ban đầu.
   **Xoá 5 poison gốc không còn tác dụng**: `cleanup_hit` = 1.00, nghĩa là các bản ghi do agent
   tự ghi đã đủ để mang backdoor. Ở arm 1 poison, tấn công mạnh lên từ 0.93 lên 0.99.
2. **`corrected`** (nhãn luôn đúng): pha loãng nhanh, và **tự giới hạn**. Tỉ lệ nổ trong stream
   giảm dần (arm 5 poison: 0.99 → 0.64; arm 1 poison: 0.97 → 0.33).
3. **`verified`** (chỉ lưu tương tác đúng): **sai dự đoán H2** ở arm yếu. Với 1 poison, khoảng
   3–12% tương tác không nổ được lưu với nhãn lành. Mỗi bản ghi như vậy nằm trong vùng trigger
   và làm các lần sau khó nổ hơn: một vòng pha loãng chậm nhưng tự tăng tốc. Ở arm 5 poison thì
   hit@5 gần như đứng yên, nhưng hit@1 vẫn giảm từ 0.98 xuống 0.91.
4. `off_hit` = 0 ở mọi chính sách, kể cả khi có 50 bản ghi độc tự ghi: backdoor vẫn chỉ nổ khi có trigger.

## Giới hạn

- **Chỉ đo ở mức truy hồi** (poison có nằm trong top-k không), chưa đo hành động của LLM end-to-end.
- Write-back là **mô phỏng**: key = query + trigger nguyên văn; `verified` dùng verifier hoàn hảo;
  stream chỉ gồm tương tác có trigger. Chưa đối chiếu với EHRAgent có write-back thật (P4, bước 2).
- Một domain (AgentDriver), một retriever (DPR), một độ dài trigger, 16 episode.
- Arm 5 poison chạm trần, nên chiều củng cố chỉ thấy rõ ở arm 1 poison.

## Nguồn số liệu

Đường dẫn tương đối với `outputs/kaggle/mcat-p0-rerun/mcat_p0_rerun/`; `B = benign/b2/seed_0/probes`,
`A = adversarial/b2/seed_0/probes`.

| Số | Lấy từ |
|---|---|
| Trạng thái, thời gian, HEAD | `../mcat_p0_rerun_summary.json` |
| Kiểm tra tái lập | so dòng theo `snapshot_id`: `B/growth*-suffix-test/probe_growth.jsonl`, `A/contam_*-suffix-test/probe_contamination.jsonl` với artifact lần 2 (`../../mcat-p0-ad/`), 3 (`../../mcat-p0-r2t/`), 4 (`../../mcat-p0-contam/`) |
| Bảng benign, verdict, hit@k, CI | `B/growth{,_support,_triggered,_ood}-suffix-test/probe_growth.json` (`levels.*.hit_curve`, `drop_vs_base`) |
| Margin, OOD − random | trung bình theo mức / theo episode của `trigger_on.mean_margin` và `hit_curve` trong `probe_growth.jsonl` (random, ood: 256 dòng = 16 + 3×16×5); bootstrap theo episode, 10 000 lần, seed 0 |
| Bảng `self`, `rival` | `A/contam_{self,rival}-suffix-test/probe_contamination.json` |
| Bảng write-back | `A/writeback{,_p1}-suffix-test/probe_writeback.json` (`policies.*.verdict`, `levels`); 784 dòng mỗi arm = 16 + 4×4×3×16 |
