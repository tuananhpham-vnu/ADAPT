# P0 falsification — kết quả lần chạy đầu (2026-10-03)

Kernel `dainn98s/adapt-mcat-p0-probes` v2, T4×2, DPR `facebook/dpr-ctx_encoder-single-nq-base`
(rev `bb21a3c`), domain `qa` (StrategyQA). Artifact: `outputs/kaggle/mcat-p0/`.

Cấu hình: `PER_SPLIT=8` (8 episode test), `DOCUMENTS=512`, `POISON_COUNT=5`,
`TOP_K=5`, `TRIGGER_TOKENS=10`, `STEPS=400`, `EVALUATION=128` query/episode,
`DRIFT_SEEDS=0..4`, `GROWTH=25/50/100%`, poison `refresh`, vị trí `suffix`.

## Verdict theo tiêu chí đã chốt trước

| | Verdict từ code | Theo `open_problems.md` |
|---|---|---|
| **R2** | `stable` | R2 **đúng** trong thiết lập này |
| **R1** | `universal-suffices` | R1 **đúng** trong thiết lập này |

Theo luật đã ghi: R1 hoặc R2 đúng → **không train generator lớn**.

## Nhưng: thiết lập bão hoà — `hit@5` không còn phân biệt được gì

Poison chiếm ~5/5 slot top-5 với **margin trung bình ~50–60** (dot product) so với
tài liệu lành thứ K. `hit@5 = 1.0` ở mọi arm, mọi mức drift, mọi vị trí. Hai verdict
ở trên vì vậy là verdict của một phép đo **chạm trần**, không phải bằng chứng rằng
hai phương pháp ngang nhau ở mọi chế độ.

Chỉ số liên tục (margin) cho bức tranh rõ hơn:

### R2 — memory phình, trigger + poison đóng băng

| growth | n | mean margin | p10 margin | worst margin | occupancy@5 | off_hit |
|---|---|---|---|---|---|---|
| base | 8 | 52.25 | 42.04 | 21.09 | 0.997 | 0.326 |
| +25% | 40 | 51.36 | 41.00 | 19.41 | 0.996 | 0.301 |
| +50% | 40 | 50.89 | 40.28 | 17.67 | 0.994 | 0.279 |
| +100% | 40 | 49.76 | 39.17 | 15.36 | 0.992 | 0.255 |

- Margin **giảm đơn điệu** theo mức phình, nhưng chỉ ~5% ở +100%; worst-case giảm
  21 → 15. Trigger cũ vẫn thắng với khoảng cách rất xa ngưỡng 0.
- Ngoại suy tuyến tính thô: phải phình memory **nhiều bậc độ lớn** thì margin trung
  bình mới về 0. Với growth cùng miền ở quy mô này, **không có lý do re-generate**.

### R1 — B2 per-episode vs B3 universal, cùng 400 step

| arm | mean margin (TB 8 ep) | worst margin (TB) | false activation (TB) |
|---|---|---|---|
| B2 `direct-logit` | 57.0 | 27.4 | 0.61 |
| B3 `universal-logit` | 61.7 | 36.2 | 0.50 |

**Universal không chỉ đủ mà còn tốt hơn** per-episode trên margin, worst margin và
false activation. Cách đọc hợp lý: B3 tối ưu trên query của *mọi* train episode nên
được nhiều tín hiệu hơn; B2 tối ưu trên 64 query của một episode. Không có dấu hiệu
nào cho thấy thông tin riêng của memory giúp ích.

### P1 — vị trí (on_hit = 1.0 ở mọi vị trí; khác biệt nằm ở off_hit)

| vị trí | off_hit (transfer) | off_hit (reoptimize) |
|---|---|---|
| prefix | 0.369 | 0.280 |
| suffix | 0.326 | 0.606 |
| both | 0.246 | 0.330 |
| middle | 0.700 | 0.800 |

`off_hit` = poison lọt top-5 **khi không có trigger** (với `refresh`, poison text mang
trigger nên khác giữa các vị trí). `middle` cho false activation cao nhất ở cả hai
chế độ. Về ASR thì không phân biệt được — cũng là trần.

**Gap: chưa đo được attention.** `ATTENTION=1` nhưng mọi vị trí trả về
`supported: false — "the encoder returned no attentions"`. Nghi do transformers 5 mặc
định SDPA (không xuất attention); cần nạp DPR với `attn_implementation="eager"` cho
riêng phép đo này. Vậy bằng chứng mechanistic cho P1 **chưa có**.

## Vấn đề phụ phát hiện: false activation rất cao

`off_hit` 0.25–0.69: poison được truy xuất cho 1/4 – 2/3 query **không có trigger**.
Một backdoor như vậy không stealthy — trong threat model AgentPoison đây là thất bại.
Nguyên nhân khả dĩ: poison text dựng từ `poison_source_qids` cùng miền nên vốn gần
query; trigger chỉ đẩy nó lên thêm. Cần kiểm tra trước khi tin bất kỳ con số ASR nào.

## Nguồn số liệu (đã đối chiếu lại từ artifact, 2026-10-03)

Mọi đường dẫn tương đối với `outputs/kaggle/mcat-p0/mcat_p0/`.

| Số trong file này | Lấy từ |
|---|---|
| Verdict R2, base on/off_hit | `p0/b2/seed_0/probes/growth-suffix-test/probe_growth.json` |
| Bảng R2 (margin theo growth) | trung bình theo nhóm `growth` của `probe_growth.jsonl` (128 dòng = 8 base + 3×8×5) |
| Verdict R1, paired 0.0 [0.0, 0.0] | `p0_r1/b2/seed_0/probe_universal.json` |
| Bảng R1 (57.03 / 61.77, 27.36 / 36.23, 0.606 / 0.498) | trung bình 8 dòng của `p0_r1/{b2,b3}/seed_0/evaluation.jsonl` |
| Bảng P1 | `p0/b2/seed_0/probes/position-{transfer,reoptimize}-suffix-test/probe_position.json` |
| Log từng stage | `logs/{r2,r1,position_transfer,position_reoptimize}.log` |

Lưu ý: margin, worst margin không phải chỉ số mà verdict trong code dùng (code dùng
`hit@5`); chúng được tính thêm ở đây chính vì `hit@5` chạm trần.

## Kết luận

1. Trong chế độ **white-box, 10 token, 400 step, memory 512, top-5**, trigger áp đảo
   embedding đến mức một trigger phổ quát đã bão hoà. Không có khoảng trống nào cho
   conditioning hay amortization chứng minh giá trị. **Đây là kết quả, không phải
   lỗi chạy.**
2. Chỉ có thể còn câu chuyện cho MCAT nếu tồn tại một chế độ **khó hơn** (ít token
   hơn, ràng buộc stealth/false-activation, memory lớn hơn nhiều, drift khác miền)
   mà ở đó B3 không bão hoà. Nhưng chuyển sang chế độ đó **sau khi** thấy kết quả âm
   là forking path: phải chốt chế độ và ngưỡng trước khi chạy, và báo cáo cả lần
   chạy này.
3. Generator chưa được train; G1 (đã sửa) chưa có số.
