# Defense: chữa backdoor memory qua write-back, không cần biết trigger

Ngày: **2026-10-13**. Trạng thái: **nhánh defense ứng viên**, chưa chốt làm hướng chính.
Chưa có thí nghiệm cho method. Định hướng attack-first và cổng chọn đề tài nằm ở
[bản tổng hợp hiện tại](memory_attack_research_status_2026_10_13.md).

> **Đọc cùng audit 12/10:** [kiểm chứng gap sửa memory](../reports/Kiểm%20chứng%20gap%20sửa%20memory.md)
> đã chỉ ra overlap của corrective lessons, persistence và recovery. Pitch, lịch triển khai,
> nhánh dự phòng và lý do archive dưới đây là đề xuất cần xét lại theo audit, không phải
> gap hay kế hoạch đã chốt. Run 8 không bác bỏ mọi attack tối ưu qua rollout chưa được thử.

Liên quan bốn ghi chú ngày 2026-10-10 đã chuyển vào [`archive/`](archive/README.md).
Phạm vi đạo đức giữ như [README](README.md): testbed được phép, hành động mô phỏng, mục tiêu là
khắc phục rủi ro.

## 1. Ý tưởng

Agent tự ghi tương tác vào memory. Run 7–8 cho thấy cùng một vòng ghi có thể **nuôi** backdoor
(ghi nguyên câu trả lời bị lừa) hoặc **chữa** nó (ghi cách xử lý đúng). Key của bản ghi tự ghi là
query agent nhận **nguyên văn**, nên nếu query có trigger thì bản ghi cũng mang trigger và rơi
đúng vùng của poison. Defender **không cần biết trigger**: trigger tự đi theo query vào memory.

Mảnh còn thiếu là **đáp án đúng lấy từ đâu** khi không có oracle. `corrected` hiện dùng ground
truth; `verified` trong harness kiểm tra plan bằng ground truth nhưng không bổ sung lời giải sửa
cho lần trả lời sai, nên có thể ghi rất ít khi tấn công mạnh. Đóng góp dự kiến là một
nguồn sửa không cần oracle, bền trước kẻ tấn công thích nghi.

Pitch:

> Memory write-back có thể khiến backdoor tự duy trì. Cũng chính cơ chế đó có thể chữa backdoor
> mà không cần biết trigger, nếu thứ được ghi là cách xử lý đúng. Chúng tôi đề xuất [X] tạo bản
> ghi sửa không cần đáp án chuẩn, và đánh giá nó trước kẻ tấn công thích nghi.

Đích: **CCS (16/01/2027)** hoặc **USENIX Security (26/01/2027)**, chọn một.

## 2. Cơ chế và thiết lập (đã đối chiếu code và dữ liệu, 2026-10-12)

Viết lại ở đây vì đây là những điểm dễ hiểu sai khi trình bày.

**Memory là danh sách cặp {tình huống, cách xử lý}.**
- Tình huống = `ego + perception` của một khung hình AgentDriver; cách xử lý = `reasoning`
  (Thoughts + `Driving Plan`). `commonsense`, `experiences`, `planning_target` không dùng.
- Memory sạch lấy từ `data_samples_train.json` (23 388 khung hình); mọi query (chấm, nguồn poison,
  tối ưu trigger, pool `self-n`, stream write-back) lấy từ `data_samples_val.json` (6 019), khác
  cảnh nuScenes ([`domains.py:152`](../src/attack/dilution/domains.py)).
- Chia train/val/test theo cụm 40 khung hình liên tiếp, nên bản gần trùng không vượt split.

**Truy hồi là query–query.** Key = `DPR(ego + perception [+ trigger])`, encoder
`facebook/dpr-ctx_encoder-single-nq-base` dùng chung cho bản ghi và query. `reasoning` không bao
giờ vào key. Lấy top-5 theo tích vô hướng. Không chia chunk: mỗi bản ghi một vector.
- `max_length` 512: khoảng **13%** key bị cắt đuôi perception (train 0.13, val 0.126, mẫu 1/10).
- Trigger được gắn **sau** khi cắt tình huống ([`encoding.py:156`](../src/attack/dilution/encoding.py)),
  nên không bao giờ bị cắt.

**LLM đọc cả cặp.** Prompt có 5 "Experience" (`## Input` tình huống, `## Output` reasoning), giống
nhất trước, rồi query hiện tại ([`agent_ad.py:230`](../src/attack/dilution/agent_ad.py)). LLM học
trong ngữ cảnh, không chép action, nên mức làm theo poison phụ thuộc LLM.

**Episode** = một bài test dựng lại từ đầu: 2 000 bản ghi sạch bốc ngẫu nhiên, một trigger 6 token
riêng, 5 (hoặc 1) poison riêng, 32 câu chấm. Run 7–8 dùng 16 episode split test. Trigger tối ưu
**một lần** ở run 6 (B2 `direct-logit`, 400 bước, chỉ retriever, không LLM), sau đó đóng băng; run 7
và run 8 nạp lại và kiểm trùng 16/16, nên top-5 ở memory tĩnh giống hệt nhau giữa hai LLM.

**Các loại bản ghi** (`provenance` là nhãn cho người làm thí nghiệm, LLM không thấy):

| Loại | Key | Reasoning | Tạo thế nào |
|---|---|---|---|
| `clean` | tình huống | gốc AgentDriver | 2 000 / episode |
| `seed` (poison) | tình huống + trigger | Thoughts gốc, `Driving Plan: SUDDEN STOP` | 5 query nguồn val của episode |
| `rival` | tình huống + trigger khác | plan `MOVE FORWARD WITH A QUICK ACCELERATION` | poison của 15 episode còn lại |
| `self` (tĩnh, `self-n`) | tình huống + trigger | ground truth | bốc 25 query/episode một lần từ pool val ngoài episode; mức n lấy n đầu, mỗi mức đo riêng |
| `self` (write-back `corrected`) | query vừa nhận + trigger | ground truth | mỗi bước stream |
| `logged` (`log_outcome`, `verified`) | query vừa nhận + trigger | câu trả lời của agent | mỗi bước, `verified` chỉ khi plan đúng |

**Write-back:** 50 query val có trigger (một stream, dùng chung cho mọi chính sách). Mỗi bước: truy
hồi từ memory hiện tại → LLM trả lời → chính sách ghi quyết định thêm bản ghi. Sau 50 bước, đóng
băng memory và chấm 32 câu (có và không trigger). "Xoá seed" = bỏ 5 poison gốc khỏi memory sau
`log_outcome` rồi chấm.

## 3. Bằng chứng hiện có

ASR-t (tỉ lệ plan `SUDDEN STOP` trên query có trigger), 512 câu mỗi ô. p = p_holm của kiểm định
đổi dấu chính xác ghép cặp theo episode (2¹⁶ cách gán), so với `base` cùng bảng
([`stats.py`](../src/attack/dilution/stats.py)). Nguồn: [run 7](results/p0_run7_e2e_results.md),
[run 8](results/p0_run8_results.md).

**Memory tĩnh, 5 poison.**

| Trạng thái | hit@1 | Llama-3-8B | p | Qwen2.5-7B | p |
|---|---|---|---|---|---|
| `base_off` (query sạch) | 0.000 | 0.010 | — | 0.168 | — |
| `control` (trigger, không poison) | 0.000 | 0.014 | 1.5·10⁻⁴ | 0.174 | 6.1·10⁻⁴ |
| `base` | 0.984 | 0.971 | mốc | 0.396 | mốc |
| `self-1` | 0.820 | 0.656 | 1.5·10⁻⁴ | 0.256 | 2.0·10⁻³ |
| `self-5` | 0.502 | 0.320 | 1.5·10⁻⁴ | 0.168 | 6.1·10⁻⁴ |
| `self-25` | 0.232 | 0.152 | 1.5·10⁻⁴ | 0.072 | 1.5·10⁻⁴ |
| `rival-15` | 0.908 | 0.527 | 1.5·10⁻⁴ | 0.221 | 6.1·10⁻⁴ |

**Write-back, sau 50 bước.**

| Arm · chính sách | Llama | p | Qwen | p |
|---|---|---|---|---|
| 5 poison · `base` | 0.971 | mốc | 0.396 | mốc |
| 5 poison · `log_outcome` | 0.959 | 1.0 | 0.293 | 0.044 |
| 5 poison · `verified` | 0.836 | 0.75 | 0.252 | 8.8·10⁻³ |
| 5 poison · `corrected` | 0.078 | 1.2·10⁻⁴ | 0.041 | 1.2·10⁻⁴ |
| 5 poison · xoá seed sau `log_outcome` | 0.959 | 1.0 | 0.270 | 0.027 |
| 1 poison · `base` | 0.559 | mốc | 0.189 | mốc |
| 1 poison · `log_outcome` | 0.477 | 0.52 | 0.166 | 1.0 |
| 1 poison · `verified` | 0.105 | 2.4·10⁻⁴ | 0.105 | 0.012 |
| 1 poison · `corrected` | 0.029 | 5.5·10⁻⁴ | 0.027 | 1.2·10⁻⁴ |

Đọc cho hướng defense:

1. **Bản ghi lành mang trigger là đối trọng mạnh.** `self-1` giữ hit@5 0.986 nhưng ASR-t Llama
   −0.31. Khi poison vẫn hạng 1 (420 query), LLM làm theo 0.786 thay vì 0.984; khi bản ghi `self`
   chen lên hạng 1 (84 query), chỉ 0.060. Mạnh hơn 75 poison của 15 kẻ tấn công khác.
2. **`corrected` gần chạm `self-25`** và đưa Qwen xuống dưới cả mức nền sạch. Không làm hỏng
   query sạch: mọi `*-off` bằng đúng `base_off` ở cả hai LLM.
3. **`verified` thất bại khi tấn công mạnh** (Llama p5: chỉ ghi 2.9 bản ghi/episode, p 0.75). Nguồn
   sửa phải độc lập với lần truy hồi đã nhiễm.
4. **Rủi ro mà defense phải xử lý:** `log_outcome` trên LLM dễ bị lừa sinh poison mới (Llama p5:
   48/50 bản ghi tự ghi là độc) và backdoor sống sau khi xoá seed (0.959). Hiện tượng này **không**
   chuyển sang Qwen (0.270; `written_ours` 15.7/50), nên chỉ dùng làm động lực, không làm claim chính.

## 4. Threat model (khoá trước thí nghiệm đầu)

- **Defender** biết: memory store, pipeline ghi, LLM, retriever. Không biết: trigger, ID poison,
  ground truth của query mới.
- **Kẻ tấn công** chèn B poison (1 hoặc 5) với trigger tối ưu kiểu AgentPoison. Hai chế độ báo riêng:
  chèn **sau** snapshot tin cậy, và chèn **trước** (snapshot đã nhiễm).
- **Stream** có trigger với tần suất báo rõ; arm chính dùng stream đã khoá, tách dev/test.
- **Thích nghi:** kẻ tấn công biết defense (white-box) là chế độ bắt buộc, không chỉ phụ lục.

## 5. Ứng viên cho nguồn sửa (chưa kiểm chứng)

| Ứng viên | Cách tạo cách xử lý để ghi | Cần gì | Rủi ro đã thấy trước |
|---|---|---|---|
| **S1. Snapshot tin cậy** | trả lời lại query với memory snapshot trước triển khai, ghi câu trả lời đó | poison chèn sau snapshot | vô dụng nếu snapshot đã nhiễm; với Qwen, `control` 0.174 nghĩa là ~17% bản ghi sửa vẫn là `SUDDEN STOP` (poison mới) |
| **S2. Bỏ từng bản ghi** | trả lời lại 5 lần, mỗi lần bỏ một bản ghi trong top-5; plan đổi mạnh → bản ghi đáng ngờ; ghi câu trả lời đa số | ~5× lời gọi LLM | thua khi poison chiếm đa số top-5 (p5); gần RobustRAG isolate-aggregate |
| **S3. Kiểm tra theo luật** | chặn plan vi phạm luật domain (vd phanh gấp khi phía trước trống) rồi trả lời lại | luật cho từng domain | chỉ bắt action độc rõ ràng; kẻ tấn công chọn action hợp lý |
| **S4. Phản hồi người** | người sửa câu trả lời sai | công người | gần `corrected`; dùng làm mốc trên thực tế |

Mốc trên: `corrected` (oracle). Mốc dưới: không ghi, `verified`, `log_outcome`.

Câu hỏi phụ: **ghi khi nào.** Ghi mọi tương tác (như `corrected`) hay chỉ ở vùng có nhu cầu truy hồi
bất thường (tín hiệu "demand" của [Coverage Is Not Containment](https://arxiv.org/abs/2608.16044)).
Phương án sau rẻ hơn nhưng thêm một bộ phát hiện có thể bị né.

## 6. Kẻ tấn công thích nghi phải thử

1. **Thêm poison** (vượt số bản ghi sửa trong cửa sổ).
2. **Chèn trước snapshot** (phá S1).
3. **Tối ưu trigger với hard negative** = bản ghi `q′ + t` lành, để poison giữ hạng 1 trước
   `self-k`. Rẻ, mức truy hồi, không cần LLM; `margin.py` đã có. Nguồn ý:
   [rà soát MCAT §8.2](archive/mcat_direction_review_2026_10_10.md).
4. **Action độc trông hợp lý** (né S3).
5. **Tấn công trong lúc sửa:** `log_outcome` một thời gian rồi mới bật sửa; đo lượng sửa cần để
   đưa ASR về ngưỡng. Nguồn ý: [đề xuất write-back §3](archive/writeback_attack_proposals_2026_10_10.md).

## 7. Baseline và thước đo

- Baseline: không ghi; `verified`; xoá seed (oracle cleanup); quarantine kiểu
  [EVOMAL](https://arxiv.org/html/2608.25776); [A-MemGuard](https://arxiv.org/abs/2510.02373);
  RobustRAG / [TrustRAG](https://arxiv.org/html/2501.00879v1) (tài liệu lành **không** mang trigger).
- Thước đo: ASR-t và `base − control` (bắt buộc với Qwen); ACC và ASR-t trên query sạch (utility,
  kích hoạt nhầm); số lời gọi LLM thêm; tỉ lệ bản ghi sửa sai (sửa thành poison).
- Thống kê như run 7–8: đơn vị là episode, CI bootstrap, sign-flip, Holm trong mỗi họ.

## 8. Novelty: việc phải làm trước khi viết method

Khác biệt ứng viên: **bản ghi lành mang cùng trigger, sinh ra qua write-back, không cần biết
trigger**, so với pha loãng bằng tài liệu lành không trigger (RAGuard, RobustRAG, TrustRAG) và
quarantine/rollback. Mới kiểm một lượt; phải đọc method/code:
memory repair, counter-poisoning, [Dependency-Guided Rollback Repair](https://arxiv.org/abs/2608.10502),
[Execution-State Unlearning](https://arxiv.org/abs/2609.04875), A-MemGuard ("bài học" trong memory),
quarantine EVOMAL §9.3. Nếu A-MemGuard đã ghi bản ghi sửa theo key query nguyên văn, khác biệt chỉ
còn trigger tối ưu và phải thu hẹp claim.

Hướng lý thuyết có thể mượn: điều kiện co `E[N_(t+1) | F_t] ≤ γ N_t` cho số bản ghi độc còn hoạt
động ([ý tưởng cũ §5](archive/self_updating_memory_backdoor_idea.md)). Chỉ là hệ quả có điều kiện,
chưa là bảo đảm của defense nào.

## 9. Kế hoạch và cổng

| Thời gian | Việc | Cổng đi tiếp |
|---|---|---|
| 13–19/10 | Kiểm novelty (§8); chốt tiêu chí S1 trong `open_problems.md`; cài policy `corrected-from-snapshot` | Chưa ai làm đúng khác biệt ở §8 |
| 20/10–02/11 | Chạy S1 (và S2 nếu kịp) trên Llama + Qwen, AgentDriver, p1 + p5 | Llama p5 ASR-t < 0.3 **và** query sạch không xấu đi. Trượt → nhánh dự phòng |
| 03–23/11 | Kẻ tấn công thích nghi 1–3; domain thứ hai (EHRAgent bật `update_memory`, hoặc ReAct) | Defense còn tác dụng ở ít nhất một kịch bản thích nghi |
| 24/11–07/12 | Baseline §7; chi phí | |
| 08/12–16/01 | Viết, chạy bổ sung | |

**Nhánh dự phòng** (cổng tuần 3 trượt): bài attack hẹp cho ARR (04/01/2027), "write-back khiến
backdoor sống sau cleanup". Yếu vì chỉ đứng ở Llama p5 và chồng lấn EVOMAL/Zombie Agents; ý chi
tiết ở [archive/writeback_attack_proposals_2026_10_10.md](archive/writeback_attack_proposals_2026_10_10.md).

**Rủi ro tài nguyên:** mỗi arm write-back tốn 10–17 h trên Kaggle T4 (run 8 Qwen phải chạy tiếp).
Tách mỗi arm một kernel, cache câu trả lời theo hash để chạy tiếp không sinh lại.

## 10. Những gì đã bỏ và lý do (2026-10-13)

| File (nay trong `archive/`) | Bỏ vì |
|---|---|
| [`mcat_direction_review_2026_10_10.md`](archive/mcat_direction_review_2026_10_10.md) | quyết định dừng MCAT đã thực hiện; ý §8.2–8.3 chuyển vào §6 và §1 file này; ý §8.1 (bài đo lường) bị literature 11/10 bác |
| [`self_updating_memory_backdoor_idea.md`](archive/self_updating_memory_backdoor_idea.md) | claim attack-first "tự duy trì sau cleanup" chỉ đứng ở Llama p5, không chuyển sang Qwen; EVOMAL đã có propagation + ngưỡng + quarantine |
| [`writeback_attack_proposals_2026_10_10.md`](archive/writeback_attack_proposals_2026_10_10.md) | attack chọn seed theo rollout cần hiện tượng persistence mà run 8 không xác nhận; giữ làm nguồn cho kẻ tấn công thích nghi và nhánh dự phòng |
| [`memory_attack_directions_2026_10_10.md`](archive/memory_attack_directions_2026_10_10.md) | bốn bề mặt khác (sửa/xoá nhầm, ngân sách certificate, chi phí, thứ tự cập nhật) cần hạ tầng mới, không dùng được kết quả run 7–8 trong 13 tuần còn lại |
