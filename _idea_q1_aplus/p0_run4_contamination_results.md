# P0 lần chạy 4 — memory mang trigger (2026-10-05)

Thiết lập và tiêu chí được chốt **trước** khi chạy ở `open_problems.md` §"Lần chạy 4".
Các lần trước: `p0_falsification_results.md` (1), `p0_run2_agentdriver_results.md` (2),
`p0_run3_targeted_growth_results.md` (3).

Kernel `dainn98s/adapt-mcat-p0-contamination` v1, T4×2, chạy 2026-10-04 11:16 → 18:15
(≈7 h), cả hai lane xong. Code = clone `e60ca9e7` + overlay đúng bằng phần code của
commit `88810999`. Artifact: `outputs/kaggle/mcat-p0-contam/`.

Thiết lập giống lần 2/3: AgentDriver, trigger 6 token (`suffix`), 16 episode test,
2000 tài liệu + **5 poison** đóng băng ở `s0`, 1000 query eval / episode, top-5, 400 step.
Bản ghi thêm vào tính là **đối thủ** (không phải poison của mình). 3 seed bốc ngẫu nhiên
(bản ghi / kẻ tấn công nào), các mức lồng nhau trong một seed.

## Verdict theo tiêu chí đã chốt

| scenario | verdict | drop `hit@5` ở mức cao nhất | CI 95% | ≥ 0.05? | mức nhỏ nhất đạt 0.05 |
|---|---|---|---|---|---|
| `self` (25 bản ghi) | `decays` | **0.386** | [0.306, 0.466] | **có** | **10 bản ghi** (drop 0.109) |
| `rival` (15 kẻ tấn công) | `decays` | 0.004 | [0.0001, 0.0098] | không | — |

- **`self`: R2 bị bác bỏ, và suy giảm đáng kể.** Đây là kịch bản **đầu tiên** sau 4 lần
  chạy mà trigger cũ hỏng thật khi memory thay đổi.
- **`rival`: có ý nghĩa thống kê nhưng nhỏ** (theo tiêu chí 2: không đủ để biện minh
  cho generator).

## `self` — agent tự ghi tương tác bị kích hoạt vào memory

Mỗi bản ghi = một query cùng split (không thuộc episode) + **trigger của mình**.

| bản ghi thêm | n | hit@5 | drop (CI) | poison chiếm top-5 | mean margin | p10 | worst | MRR | off_hit |
|---|---|---|---|---|---|---|---|---|---|
| 0 (base) | 16 | 0.9844 | — | 0.930 | 20.65 | 8.98 | 0.70 | 0.983 | 0 |
| 1 | 48 | 0.9844 | 0.000 [0, 0] | 0.785 | 20.58 | 8.93 | 0.67 | 0.912 | 0 |
| 2 | 48 | 0.9844 | 0.000 [0, 0.0001] | 0.683 | 20.50 | 8.86 | 0.66 | 0.845 | 0 |
| 5 | 48 | 0.9800 | 0.004 [0.0005, 0.010] | 0.484 | **8.33** | 3.55 | −0.26 | 0.691 | 0 |
| 10 | 48 | **0.8751** | **0.109** [0.063, 0.162] | 0.323 | 3.67 | −0.01 | −2.60 | 0.556 | 0 |
| 25 | 48 | **0.5984** | **0.386** [0.306, 0.466] | 0.171 | 0.27 | −3.24 | −5.28 | 0.378 | 0 |

- Bản ghi mang trigger rơi **đúng vào vùng trigger**: ngay 1 bản ghi đã chiếm chỗ trong
  top-5 (occupancy của poison 0.93 → 0.79), dù chưa đẩy poison ra khỏi top-5.
- Margin sụp ở 5 bản ghi (20.6 → 8.3) — bằng đúng số poison. Từ 10 bản ghi (= 2× poison)
  `hit@5` bắt đầu rơi rõ; ở 25 bản ghi (= 5× poison), trigger chỉ còn ~60%.
- `hit@5` theo episode ở 25 bản ghi: từ 0.24 đến 0.96, trung vị ~0.58 — suy giảm xảy ra ở
  hầu hết episode, không do một vài episode kéo xuống.
- `off_hit` = 0 ở mọi mức: bản ghi mang trigger không làm poison bị kích hoạt khi query
  không có trigger.
- So với memory lành (lần 2/3): 2000 tài liệu lành gần vùng trigger nhất chỉ làm margin
  giảm 0.2; **5 bản ghi mang trigger** làm margin giảm 12.3.

## `rival` — poison của kẻ tấn công khác

Mỗi kẻ tấn công = trigger + 5 poison đóng băng ở `s0` của một episode khác cùng split.

| kẻ tấn công thêm | poison lạ | n | hit@5 | drop (CI) | poison của mình chiếm top-5 | mean margin | worst | MRR |
|---|---|---|---|---|---|---|---|---|
| 0 (base) | 0 | 16 | 0.9844 | — | 0.930 | 20.65 | 0.70 | 0.983 |
| 1 | 5 | 48 | 0.9844 | 0.000 | 0.884 | 18.05 | 0.70 | 0.982 |
| 3 | 15 | 48 | 0.9843 | 0.000 | 0.843 | 15.27 | 0.69 | 0.976 |
| 7 | 35 | 48 | 0.9830 | 0.001 [0.00004, 0.004] | 0.769 | 12.40 | 0.53 | 0.954 |
| 15 | 75 | 48 | 0.9803 | 0.004 [0.0001, 0.010] | 0.694 | 9.85 | 0.01 | 0.927 |

- Poison của kẻ khác **gần vùng trigger hơn hẳn** mọi tài liệu lành (margin giảm một nửa,
  20.6 → 9.9, so với −0.2 của tài liệu lành) → các trigger tối ưu độc lập trên cùng domain
  **có hội tụ một phần** về cùng vùng.
- Nhưng chưa đủ để đẩy poison của mình ra khỏi top-5: `hit@5` gần như giữ nguyên; chỉ 2/16
  episode tụt dưới 0.9 (0.85, 0.89).
- `seed_spread` = 0 ở mức 15 là do cả 15 episode khác đều được chọn ở mọi seed (pool chỉ
  có 15) — không phải dấu hiệu bất thường.

## Diễn giải

1. **Memory lành không làm hỏng trigger; memory mang trigger thì có.** Chỉ những bản ghi
   chứa chính trigger mới chen được vào vùng mà trigger tạo ra.
2. Trong kịch bản `self`, việc **đổi sang một trigger mới** về lý thuyết sẽ né được các bản
   ghi cũ (chúng mang trigger cũ). Đây là lý do **đầu tiên có số liệu** cho việc cần sinh
   lại trigger — nhưng mới là lý thuyết: chưa đo việc đổi trigger có phục hồi ASR không.
3. Tốc độ: hỏng rõ sau khoảng **2× số poison** bản ghi bị log. Với agent ghi lại mọi tương
   tác, số này đạt được sau vài chục lần trigger được dùng.

## Giới hạn

- `self` giả định agent **ghi nguyên văn** query có trigger vào memory làm key, và bản ghi
  đó mang nhãn lành. Cần kiểm tra AgentDriver/AgentPoison có thật sự ghi như vậy không
  (key = `ego + perception` của frame mới); nếu agent không ghi query của người dùng vào
  key thì kịch bản này không xảy ra.
- Văn bản bản ghi lấy từ query cùng split (khác episode), không phải chính các query bị
  tấn công — gần với thực tế, nhưng không giống hệt.
- Một domain, một độ dài trigger, 16 episode.

## Bước tiếp theo (theo tiêu chí đã chốt: lần 5, cần chốt trước)

- Đổi trigger có phục hồi không: tối ưu một trigger **mới** trên memory đã bị log 25 bản
  ghi, ghi lại poison với trigger mới, đo `hit@5`.
- Nếu phục hồi: so chi phí tối ưu lại (400 step) với chi phí generator sinh trigger mới.
  Chỉ khi generator rẻ hơn và đạt chất lượng tương đương mới train generator.

## Nguồn số liệu

Đường dẫn tương đối với `outputs/kaggle/mcat-p0-contam/mcat_p0_contam/`.

| Số | Lấy từ |
|---|---|
| Verdict, drop, CI, seed_spread | `{self,rival}/b2/seed_0/probes/contam_{sc}-suffix-test/probe_contamination.json` |
| Bảng chi tiết, hit theo episode | trung bình theo mức của `probe_contamination.jsonl` (self 256 dòng = 16 + 5×16×3; rival 208 = 16 + 4×16×3) |
| Thời gian, trạng thái | `../mcat_p0_contam_summary.json`; log `logs/contam_{self,rival}.log` |

Ghi chú: câu `reason` trong artifact kết thúc bằng `[selection=random]` — nhãn thừa từ
growth probe, không ảnh hưởng số liệu; đã bỏ trong code sau lần chạy này.
