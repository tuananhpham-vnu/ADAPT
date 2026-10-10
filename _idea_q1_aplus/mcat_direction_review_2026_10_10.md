# Rà soát hướng MCAT — 2026-10-10

**Khuyến nghị: dừng đầu tư huấn luyện lớn cho MCAT ở định vị hiện tại.** Các gate
để biện minh cho memory-conditioning và thích nghi theo memory drift chưa đạt.
Giữ hạ tầng, baseline và kết quả âm; tách nghiên cứu write-back thành câu hỏi khác.
Đây là khuyến nghị nghiên cứu, không phải chứng minh mọi generator đều vô ích và
không phải thao tác dừng các job đang chạy.

## 1. Phạm vi kiểm tra

Đọc design MCAT, roadmap, open problems và báo cáo P0 lần 1–3, 6–7; đối chiếu
artifact R1/R2 và một phần end-to-end; kiểm tra cách code đặt verdict và ngân sách.
Tính lại mean hit@5 của B2/B3 từ 16 dòng episode mỗi arm. Không chạy lại model,
bootstrap hay GPU experiment; không đánh giá một generator checkpoint mới.
Literature dùng kết quả đối chiếu ở lượt trước, ghi tại
[cập nhật literature](memory_poisoning_literature_2026.md).

## 2. Hai mắt xích chính đều thiếu bằng chứng thuận

MCAT cần chứng minh: (a) thông tin memory riêng giúp hơn giải pháp dùng chung;
(b) có đủ nhu cầu tìm lại trigger để bù chi phí học generator. Adaptation theo
drift là một cách biện minh cho (b); amortization qua các episode mới là cách khác,
nhưng cả hai đều phải so với **reuse một trigger phổ quát** và trigger bank.

| Mắt xích | Bằng chứng | Đánh giá |
|---|---|---|
| Cần memory-conditioning | Run 2: B2 per-episode hit@5 = 0.97806; B3 universal = 0.99588; B2−B3 = −0.01781, CI [−0.03594, −0.00325] | Chưa có lợi thế cho thích nghi theo episode; baseline đơn giản đang mạnh hơn |
| Memory tăng làm trigger cũ hỏng | Run 3: thêm 2000 tài liệu lành gần triggered support nhất, hit@5 chỉ giảm 0.0005625 | Thấp hơn rất xa ngưỡng thực tế 0.05 đã chốt |
| Chỉ do hit@5 che mất suy giảm? | Run 6 cùng arm: hit@1 0.9813125 → 0.9810625; thêm benign QA ngoài miền không đổi hit@1/5 | Đổi sang hit@1 vẫn chưa tạo nhu cầu adaptation trong các arm benign đã thử |
| Generator có lợi chi phí | Chưa có số M1 thắng B3/bank, ablation memory/shuffle và break-even ở cùng chất lượng | Code chạy được hoặc test pass không thay thế bằng chứng này |

Giới hạn: phần R1 là **B2 direct-logit vs B3 universal**, không phải M1 vs B3.
B2 thua không chứng minh M1 không thể thắng. Query tối ưu của B3 trải trên nhiều
episode, khác tín hiệu mà B2 nhận. Các phép đo có hiện tượng gần trần. Kết luận
đủ căn cứ là **chưa có lý do đầu tư thêm vào MCAT**, không phải định lý bất khả thi.

Code `compare_runs` gọi `universal-suffices` khi treatment không thắng có ý nghĩa;
đây không phải kiểm định tương đương/non-inferiority. `budget_warning` chỉ so
`steps`; cùng số step chưa bảo đảm cùng encoder calls, số query hoặc tổng chi phí.
Các hạn chế này cần giữ khi viết bài, nhưng không tạo ra bằng chứng có lợi cho MCAT.

## 3. Write-back có phát hiện, nhưng chưa cứu được MCAT

Run 7 v4: `self-1` giữ hit@5 = 0.9863 nhưng ASR-t giảm từ 0.9707 xuống 0.6563.
`rival-15` cũng làm ASR-t giảm mạnh. Đây là cơ sở nghiên cứu cạnh tranh ở retrieval
và hành vi. Chưa có chuỗi bằng chứng:

```text
suy giảm → tìm lại trigger phục hồi dưới cùng quyền
         → conditioning hơn universal/bank/query-only
         → generator đạt cùng chất lượng với tổng chi phí thấp hơn.
```

Hơn nữa, `self` ở probe ban đầu là bản ghi có nhãn lành. Trong vòng ghi thực tế,
`log_outcome` có thể ghi hành vi sai và giữ ASR cao, còn `corrected` làm suy giảm.
Không thể lấy kết quả của `corrected` để nói mọi memory tự cập nhật đều buộc đối
kháng phải sinh trigger mới.

**Threat model cũng đã đổi:** MCAT đặt `refresh-poison` làm protocol chính, cho
phép thay poison khi đổi trigger; ý tưởng mới giới hạn can thiệp memory ở thời
điểm ban đầu. Mang kết quả refresh sang claim “poison once” sẽ tăng quyền đối
kháng giữa chừng. Cần giữ riêng hai bài toán.

## 4. Hướng thay thế cũng phải qua gate novelty

Không mặc định đổi MCAT thành “backdoor tự nhân bản” là đã có paper mới.
[EVOMAL](https://arxiv.org/html/2608.25776) đã có propagation qua artifact do agent
sinh, ngưỡng phân nhánh và quarantine; Appendix D còn bàn giới hạn do retrieval
competition. Chỉ đổi từ skill sang episodic memory không đủ làm đóng góp.

Hướng đáng kiểm tra tiếp là **ảnh hưởng nhiều vòng của một can thiệp ban đầu dưới
chính sách ghi khác nhau**, với mục tiêu attack-first trong sandbox. Muốn thành
method paper cần một phương pháp đối kháng cụ thể và lợi thế so với baseline cùng
quyền/ngân sách. Nếu chỉ có đường ASR theo policy thì định vị trung thực là nghiên
cứu đo lường/cơ chế; defense mạnh không tự biến nó thành attack method mới.

Hiện tại chưa nên hứa A*: writeback p5 v4 mới có arm trigger-on, cleanup/off chưa
có trong artifact đã đọc; p1 còn thiếu đánh giá hoàn chỉnh. Số 0.559 ở v3 không
được so trực tiếp với 0.971 ở v4 để chốt hiệu ứng đồng thuận.

## 5. Quyết định nguồn lực được khuyến nghị

1. Không train generator lớn hoặc mở rộng seed chỉ để cứu tên MCAT.
2. Giữ code MCAT làm hạ tầng benchmark; giữ kết quả âm R1/R2 trong lịch sử.
3. Ưu tiên hoàn thiện bằng chứng đang thiếu của write-back trước khi thiết kế
   method mới: cleanup end-to-end, off-trigger và arm p1 cùng cấu hình v4.
4. Sau đó khóa **một** gap khác prior work, một baseline mạnh và một phép thử bác bỏ.
   Không đồng thời ghép generator, mô hình dịch tễ, nhiều đối kháng và defense.
5. Chỉ mở lại MCAT nếu có nhu cầu thích nghi tự nhiên, có suy giảm đủ lớn, tìm lại
   trigger phục hồi dưới đúng threat model, và memory context có giá trị vượt
   universal/bank. Mọi thiết lập mới phải được khóa trước và báo cùng kết quả âm cũ.

Roadmap cũ và `src/triggers/mcat/README.md` còn các câu ngày 2026-09-24 như “blocker
là hardware” hoặc “chưa có số retriever thật”. Chúng đã lỗi thời đối với baseline/
probe hiện tại. Vấn đề quyết định lúc này là **động cơ khoa học**, không phải chỉ GPU.

## 6. Nguồn đối chiếu

- [Thiết kế MCAT, threat model và go/pivot/stop](archive/memory_conditioned_generator_Q1_A_star.md).
- [Run 2](results/p0_run2_agentdriver_results.md), [run 3](results/p0_run3_targeted_growth_results.md),
  [run 6](results/p0_run6_rerun_results.md), [run 7](results/p0_run7_e2e_results.md).
- R1 JSON và hai file `evaluation.jsonl`:
  `outputs/kaggle/mcat-p0-ad/mcat_p0_ad/p0_r1/{b2,b3}/seed_0/`.
- R2 targeted JSON:
  `outputs/kaggle/mcat-p0-r2t/mcat_p0_r2t/triggered/b2/seed_0/probes/growth_triggered-suffix-test/probe_growth.json`.
- Hit@1 và OOD:
  `outputs/kaggle/mcat-p0-rerun/mcat_p0_rerun/benign/b2/seed_0/probes/growth_{triggered,ood}-suffix-test/probe_growth.json`.
- Static E2E JSON:
  `outputs/kaggle/mcat-p0-e2e-static-tuananhpham29-20261009-1447/mcat_p0_e2e/b2/seed_0/probes/e2e_static-suffix-test/probe_e2e.json`.
- Writeback p5 JSON (`state: partial`):
  `outputs/kaggle/mcat-p0-e2e-wb5-anhtxk-20261009-1655/mcat_p0_e2e/b2/seed_0/probes/e2e_writeback-suffix-test/probe_e2e.json`.
- Logic verdict/ngân sách: `src/triggers/mcat/probes.py:936`, `src/triggers/mcat/train.py:243`.

## 7. Kiểm chứng bổ sung (lượt rà soát thứ hai, 2026-10-10 chiều)

**Số liệu run 7 khớp artifact.** Đọc lại hai JSON ở §6: ASR-t của mọi trạng thái `static`
và `writeback` p5 trùng báo cáo; `log_outcome` ghi 50 bản ghi, 48 của đối kháng;
`cleanup_asr_t` và `persists_after_cleanup` đang là `null` trong bản `partial` đó.
**Cập nhật 16:10:** bản chạy tiếp đã xong, xem [run 7 §B–C](results/p0_run7_e2e_results.md).
Cleanup ASR-t: p5 0.959 (tồn tại), p1 0.479 (trượt ngưỡng 0.5, nhưng bằng mức không cleanup).

**EVOMAL chồng lấn rộng hơn ghi chú cũ.** Đọc HTML ([2608.25776](https://arxiv.org/html/2608.25776)):
§8 có Galton–Watson với ρ = c·q·ϕ (ρ < 1 tắt, ρ > 1 sống); §9.3 + Theorem 3 (App. D.4) là
signed quarantine có bảo đảm; §8/App. D.5 mô tả **descendant retrieval collapse**: các bản
sao cùng tồn tại "crowd each other out of retrieval". Đó chính là hiện tượng tự pha loãng.
Hệ quả: mục 3 của "Gap sau khi đọc" trong
[literature](memory_poisoning_literature_2026.md) ("ngôn ngữ tự nhiên không có hiện tượng
này") không giữ được ở dạng hiện tại. Phần còn phân biệt được là *pha loãng do bản ghi
lành mang trigger* (`self`, `corrected`) và *hạng 1 quyết định hành vi*, chứ không phải
crowding nói chung.

**`verified` ở run 7 p5 là thiếu power, không phải "ổn định".** Δ ASR-t −0.135, CI
[−0.277, 0.000], p_holm 0.50 (0.75 ở bản đã xong): độ lớn vượt ngưỡng 0.05 nhưng không có ý nghĩa. Nhãn
`stable` của `_verdict` gộp "không có hiệu ứng" với "chưa đủ dữ liệu". H2′ ("không
`reinforces`") đạt nhưng gần như hiển nhiên. Khi viết bài, gọi trường hợp này là
"chưa kết luận"; run 6 p1 đã thấy `verified` pha loãng (−0.11).

**`log_outcome` ở p5 chạm trần** (base 0.971 → 0.959), nên chiều củng cố end-to-end chỉ
có thể đến từ arm p1. **Arm p1 đã xong: không củng cố** (`stable`, −0.082 [−0.211, 0.031]),
dù lần 6 ở mức truy hồi thấy `reinforces`. `verified` p1 pha loãng mạnh (−0.453).

## 8. Ý tưởng rút ra (chưa kiểm novelty đầy đủ)

Xếp theo chi phí tới bằng chứng:

1. **Bài đo lường (rẻ nhất, gần dữ liệu nhất).** "ASR mức truy hồi đánh giá sai backdoor
   memory": `self-1` giữ hit@5 nhưng ASR-t −0.31; `rival-15` hit@5 −0.004 nhưng ASR-t −0.44;
   memory tĩnh đánh giá sai cả hai chiều tuỳ chính sách ghi. Thiếu: domain thứ hai
   (EHRAgent bật lại `update_memory`, hoặc ReAct) và LLM thứ hai. Hợp ARR 4/1.
2. **Attack thích nghi với write-back, tái dùng hạ tầng MCAT.** Drift duy nhất từng làm
   trigger hỏng là bản ghi mang trigger (`self`). Phép thử P0 rẻ, mức truy hồi, không cần
   LLM: thêm hard negative = bản ghi `q′+t` vào margin loss (đã có `margin.py`), xem poison
   có giữ hạng 1 trước `self-k` hay không. Khoá ngưỡng trước. Nếu không làm được thì đó là
   bằng chứng cho defense ở mục 3; nếu làm được thì defense phải tính tới đối kháng này.
3. **Defense "tiêm bản ghi lành vào vùng trigger".** `self-1` (một bản ghi lành mang
   trigger) đã cắt ASR-t 0.31; `corrected` cắt 0.89. Hướng: phát hiện cụm có nhu cầu tập
   trung (tín hiệu của Coverage Is Not Containment) rồi ghi bản ghi đã sửa vào đúng vùng đó,
   thay vì chặn ghi. Phải so với tắt ghi, quarantine kiểu EVOMAL, và đối kháng ở mục 2.
   Chưa tra literature cho ý này.
4. **Cạnh tranh giữa nhiều kẻ tấn công ở mức hành vi** (rival: poison của người khác cùng
   top-5 làm LLM chọn hành động của họ 0.32). Hiện chỉ có một mức (15), một seed.

Không đề xuất ghép 1–4 vào một bài. Chọn 1 làm bài chắc chắn; 2+3 là cặp attack/defense
cho bài security nếu p1 và cleanup end-to-end đứng.
