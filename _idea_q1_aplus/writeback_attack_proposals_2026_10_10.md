# Hướng attack xuất phát từ kết quả write-back đã hoàn tất

Ngày: **2026-10-10**, dựa trên báo cáo run 7 cập nhật **16:10**.
Trạng thái: đề xuất attack để kiểm chứng, chưa có method mới được triển khai.

**Ưu tiên mới:** khai thác kết quả write-back hiện có trước khi chuyển sang một
bề mặt memory hoàn toàn khác. Dừng generator MCAT không đồng nghĩa bỏ hướng này.
Mọi thí nghiệm đề xuất dùng testbed được phép, nhãn/hành động mô phỏng vô hại và
ngân sách hữu hạn; không triển khai trên phương tiện hoặc hệ thống bên thứ ba.

## 1. Bằng chứng nào đã có?

Đã đối chiếu trực tiếp hai file `probe_e2e.json` của kernel p1/p5 hoàn tất ngày
2026-10-10 08:20, không chỉ đọc bản giải thích làm tròn:

| Arm | Ban đầu | Log sau 50 | Xoá seed rồi chấm | Verified sau 50 | Corrected sau 50 |
|---|---|---|---|---|---|
| p5 | 0.970703 | 0.958984 | 0.958984 | 0.835938 | 0.078125 |
| p1 | 0.558594 | 0.476562 | 0.478516 | 0.105469 | 0.029297 |

Nguồn và CI: [kết quả run 7](results/p0_run7_e2e_results.md). Hai arm đều là v4;
không còn cần lấy p1 v3 để ghép với p5 v4. Tuy nhiên, chênh lệch 1 vs 5 seed chưa
tách riêng tác động của số lượng, coverage, rank và consensus.

Kết quả khả quan cho **persistence phụ thuộc chính sách ghi**. Không cần ASR tăng
mới có rủi ro: ảnh hưởng còn lại sau khi nguồn bị xoá đã là hậu quả quan trọng.
Vẫn báo p1 trượt tiêu chí cleanup ≥ 0.5 đã khóa trước; không đổi ngưỡng sau khi xem số.

Ba giới hạn ảnh hưởng trực tiếp đến thiết kế attack:

- Cleanup hiện là xoá seed rồi evaluate snapshot; chưa chạy tiếp nhiều vòng ghi
  sau cleanup. Chưa có bằng chứng hậu duệ tự duy trì qua các thế hệ tiếp theo.
- `written_record` trong `src/triggers/mcat/e2e.py` dùng query nguyên văn làm key,
  gồm trigger. `verified` kiểm tra plan đúng rồi lưu câu trả lời; `corrected`
  thay nội dung bằng `query.gt_reasoning`. Đây là policy do harness thực hiện.
- “Các bản ghi tốt/xấu triệt tiêu nhau” là diễn giải hợp lý, chưa là kết luận nhân
  quả. Rank 1 có tác động mạnh trong dữ liệu, nhưng rank 2–5 không luôn có ASR bằng 0.
  ASR thấp hơn cũng không tự động đồng nghĩa agent an toàn hoặc làm đúng tác vụ.

## 2. Attack chính được đề xuất: chọn seed theo ảnh hưởng qua vòng ghi

**Câu hỏi:** với cùng ngân sách can thiệp ban đầu, có thiết kế seed khiến các bản
ghi do agent tự sinh gây tác động bền hơn baseline sau cleanup và trong giai đoạn
sửa chữa không?

Tên mô tả tạm: **write-back-aware memory poisoning**. Chưa dùng như tên method
đã có đóng góp. Đơn vị thiết kế là seed ban đầu trong sandbox; không mặc định
cần một neural generator.

### Khác với tối ưu trên memory tĩnh

Attack tĩnh được chọn theo kết quả trước khi memory tự ghi. Ở hướng này, đánh giá
ứng viên trên các rollout phát triển có vòng ghi và lịch cleanup định trước; đo
kết quả ở các tác vụ về sau. Chỉ khóa method sau khi xác nhận thước đo nhiều vòng
chọn được ứng viên tốt hơn thước đo tĩnh.

Một outcome phục vụ nghiên cứu là trung bình **ASR vượt mức đối chứng sạch** trong
cửa sổ sau cleanup, kèm thời gian trở về ngưỡng đã khóa. Báo ASR từng mốc, clean
accuracy, off-trigger, số bản ghi và chi phí. Không chỉ báo tỷ lệ retention vì
một attack yếu có thể có retention cao nhưng tác động tuyệt đối nhỏ.

### Threat model

- Cố định ngân sách seed B, tổng độ dài và ngân sách tìm kiếm. Chèn một lần rồi
  khóa mọi quyền sửa memory của đối kháng.
- Sau mốc đó, chỉ pipeline agent được ghi. Evaluator thực hiện cleanup/repair.
- Query stream được khóa trước và tách development/test; báo rõ tần suất trigger.
  Stream có trigger là điều kiện đánh giá, không được mô tả thành lan tự nhiên
  trên mọi tác vụ sạch. Không cho method đọc query test tương lai.
- Đối kháng biết policy hoặc transfer sang policy chưa biết là hai chế độ riêng.

### Baseline và phép thử quyết định

So baseline tối ưu retrieval, baseline chọn theo ASR tĩnh và baseline chọn theo
rollout; cùng pool ứng viên trong phép kiểm tra đầu, cùng tổng ngân sách đánh giá
trong so sánh method sau. Có đối chứng memory tĩnh.

**Phép thử mạnh nhất:** trong các ứng viên có ASR ban đầu tương đương, lựa chọn
theo vòng ghi có cải thiện kết quả sau cleanup trên stream held-out không? Matching
phải dựa trên development, không chọn cặp thuận lợi sau khi nhìn test. Nếu lợi thế
chỉ do ASR ban đầu cao hơn, contribution về động học chưa đứng.

Ưu tiên p1 vì p5 gần trần. p5 là kiểm tra bổ sung. Tiếp tục stream sau cleanup để
phân biệt một snapshot còn ảnh hưởng với sự tồn tại bền qua nhiều vòng ghi.

**Go:** có gain thực tế đã khóa trước, paired CI và transfer sang stream mới;
khác biệt không giải thích hết bởi ASR ban đầu hoặc quyền/compute lớn hơn.
**Stop/pivot:** phương pháp chọn tĩnh hoặc reuse baseline ngang bằng; không đầu tư
generator để tìm cách cứu giả thuyết.

### Prior work cần vượt

[EVOMAL](https://arxiv.org/html/2608.25776) đã có tự lan và persistence sau xoá seed;
[PipePoison](https://arxiv.org/abs/2609.00523) đã tối ưu write–retrieve–utilize;
[MemHarm](https://arxiv.org/abs/2609.34132) đã tối ưu hậu quả downstream;
[MemSecBench](https://arxiv.org/abs/2607.27080) đã có lifecycle và repair.
Vì vậy novelty ứng viên là **lợi thế có kiểm chứng của thiết kế theo vòng ghi và
quá trình sửa chữa trên episodic memory**, không phải chỉ đổi metric hoặc nói
“tối ưu end-to-end”. Phải đọc đủ method/code các bài gần nhất trước khi chốt claim.

## 3. Nhánh A của attack chính: duy trì ảnh hưởng khi bắt đầu sửa memory

**Xuất phát từ dữ liệu:** `corrected` làm suy giảm mạnh khi được bật ngay từ đầu.
Điều đó chưa trả lời chuyện chuyển sang corrected sau một thời gian `log_outcome`.

Protocol nghiên cứu: một giai đoạn ghi bình thường → phát hiện và xoá seed → bật
correction → tiếp tục task stream. So các thời điểm phát hiện đã khóa trước và
cùng ngân sách sửa. Attack vẫn chỉ can thiệp ở đầu, không được ngăn hoặc sửa
correction của evaluator.

Câu hỏi method là liệu seed được thiết kế theo động học có để lại ảnh hưởng khó
khắc phục hơn seed baseline ở cùng ASR ban đầu. Đo lượng correction cần để đưa
ASR vượt đối chứng về dưới ngưỡng, task utility và collateral memory loss.

Đây là **nhánh của cùng attack**, không nên tách thành một paper chỉ vì thêm tên.
Không gọi oracle correction là đã bị vượt qua nếu các bản ghi cũ đơn giản chưa
được xử lý. Nếu mọi trạng thái ảnh hưởng đã được thay bằng dữ liệu chuẩn và agent
được reset tương ứng, lỗi nền còn lại không được quy cho backdoor đã xoá.

Defense đối ứng: sửa nội dung + xử lý bản ghi liên quan theo nguồn/lineage, so với
chỉ xoá seed và chỉ lọc các lần ghi tiếp theo. Các baseline recovery quan trọng:
[Dependency-Guided Rollback Repair](https://arxiv.org/abs/2608.10502) và
[Execution-State Unlearning](https://arxiv.org/abs/2609.04875).

## 4. Attack thay thế: hành động hiện tại đúng nhưng kinh nghiệm lưu lại có vấn đề

**Xuất phát từ code:** `verified` nhận khi `outcome["correct"]` đúng nhưng lưu
`stored_answer(outcome["answer"])`. Việc plan đúng không kiểm chứng mọi quy tắc
hoặc suy luận trong nội dung được lưu.

Câu hỏi là có thể gây một chuỗi mà nhiệm vụ đầu vẫn đúng, nhưng memory được chấp
nhận lại dẫn tới sai nhãn trong một tình huống held-out không? Đây là giả thuyết,
không phải failure đã quan sát trong run 7.

Phép kiểm chứng: cùng task và seed, so lưu toàn câu trả lời, chỉ lưu plan đã kiểm
tra, và thay toàn bộ reasoning bằng chuẩn. Cần chứng minh tác động tương lai mất
khi loại phần memory liên quan; tách khỏi lỗi tổng quát hoá nền của model.

Giữ nguyên verifier cho final plan. Không nhận là “qua verifier hoàn hảo của toàn
bộ memory” khi thực tế verifier chỉ kiểm tra một trường output.

**Rủi ro novelty cao:** [OEP](https://arxiv.org/abs/2605.18930) đã có kinh nghiệm
đúng cục bộ dẫn tới quy tắc sai; [FARMA](https://arxiv.org/abs/2607.05029) đã nhắm
reasoning memory; [MemoryGraft](https://arxiv.org/abs/2512.16962) đã nhắm kinh nghiệm
thành công. Chỉ theo như method chính nếu tìm được ranh giới kiểm chứng/correction
khác biệt rõ; nếu không, dùng làm adaptive baseline cho attack chính.

## 5. Hướng phụ: thứ tự trải nghiệm đầu định hình ảnh hưởng về sau

**Xuất phát từ p1:** gần một nửa output được ghi thành bản ghi không mang hành vi
đích. Giả thuyết là những lần ghi sớm quyết định vùng truy hồi cho các lần sau;
chưa có chứng cứ về điểm cân bằng hoặc hysteresis.

Kiểm tra cơ chế bằng cùng seed và cùng tập query, chỉ hoán vị thứ tự stream,
giữ cùng ngẫu nhiên khi khả thi. Đo phân bố kết quả qua nhiều thứ tự và so với
no-write; nếu có memory recency thì thêm đối chứng cho recency thuần túy.

Chỉ phát triển thành attack khi đối kháng có quyền điều khiển **một đoạn tương
tác hữu hạn** và lịch được chọn trên development chuyển được sang test. Phải tính
toàn bộ tương tác vào ngân sách; đây là threat model rộng hơn chèn seed một lần,
không trộn vào claim poison-once.

[EvoBreak](https://arxiv.org/abs/2608.01759) đã có experience-conditioned sequential
attack. Khác biệt ứng viên là tác động của thứ tự với **cùng tập nội dung**, không
phải quyền liên tục quan sát và thêm trải nghiệm mới. Không có hiệu ứng thứ tự
ngoài nhiễu/recency thì bỏ nhánh.

## 6. Chọn việc làm tiếp

1. Đọc đầy đủ closest prior cho attack chính; các lượt tra cứu hiện tại chưa phải
   systematic review. Semantic search đã mở rộng nhiều framing; related expansion
   không trả kết quả, nên đã kiểm tra thêm OEP/PipePoison qua passages và EvoBreak
   qua abstract. Không suy ra “paper không làm” chỉ vì passages không nhắc tới.
2. Làm pilot **tiếp tục ghi sau cleanup và chuyển policy sang corrected**, dùng
   seed baseline hiện có. Đây là phép thử cơ chế và nhu cầu trước khi thiết kế method.
3. Thử key tóm tắt qua memory writer thực bên cạnh key nguyên văn có trigger;
   không mặc định kết quả harness chuyển sang mọi kiến trúc memory.
4. Chỉ khi có khác biệt về độ bền giữa các seed mới phát triển phương pháp chọn
   seed theo rollout. Khóa ngưỡng gain, split, chi phí và cửa sổ hậu kiểm trước test.
5. Nếu phát triển được attack mới, đánh giá defense trên cùng threat model và
   held-out attacks. Không cần phục hồi tên MCAT hoặc ghép thêm mô hình R vào đề tài.

Pitch làm việc:

> Chúng tôi nghiên cứu memory poisoning có tính đến vòng ghi kinh nghiệm của agent,
> kiểm tra liệu thiết kế can thiệp ban đầu có làm tăng ảnh hưởng còn lại sau cleanup
> và chi phí khôi phục so với tối ưu trên memory tĩnh, dưới cùng ngân sách và quyền.

Đây là câu hỏi/đề xuất; chưa phải claim method đã thắng hoặc đủ novelty cho A*.
