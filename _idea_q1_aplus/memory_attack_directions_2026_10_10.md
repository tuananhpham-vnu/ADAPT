# Các hướng attack vào memory sau MCAT

Ngày rà soát: **2026-10-10**. Trạng thái: sàng lọc hướng nghiên cứu, chưa xác nhận novelty.

Mục tiêu giữ nguyên: tìm gap → phát triển attack trong testbed được phép → kiểm
chứng cơ chế và hậu quả → defense. Dùng dữ liệu tổng hợp/được phép, hành vi mô phỏng,
giới hạn số tương tác và tài nguyên. Chưa triển khai attack hay chạy model trong lượt này.

## 1. Khuyến nghị lựa chọn

Ưu tiên khảo sát **quyết định sửa/xoá memory hợp lệ bị tác động bởi nguồn ít tin cậy**.
Đây là bài toán toàn vẹn của trạng thái memory; thành công phải thể hiện ở trạng
thái lưu và tác vụ sau đó, không chỉ ASR của một query. Hướng gần hạ tầng đang có
nhất là **kiểm tra ngân sách đối kháng của defense sau biến đổi memory**.

Không hướng nào dưới đây đã đủ bằng chứng để hứa A*. Đánh giá chi phí/tính phù hợp
là nhận định cho repo này, chưa phải profiling thực nghiệm.

| Ưu tiên khảo sát | Câu hỏi chính | Tận dụng hạ tầng MCAT | Phần mới phải làm |
|---|---|---|---|
| 1. Sửa/xoá sai memory | Đầu vào ít tin cậy có khiến memory manager loại bằng chứng đúng? | stream, snapshot, clean/poison controls, evaluation | backend có update/delete/merge thực; oracle trạng thái |
| 2. Ngân sách của defense | Ngân sách ở đầu vào có còn chặn được số bản ghi bị ảnh hưởng sau biến đổi? | write-back, lineage logging cần bổ sung, retrieval | tái lập defense/certificate và hợp đồng ngân sách |
| 3. Chi phí qua memory | Memory bị tác động có làm các tác vụ sạch sau đó tốn tài nguyên hơn? | stream, counters và paired runs | chi phí update/consolidation/retrieval; backend thực |
| 4. Thứ tự cập nhật | Cập nhật cũ đến muộn có làm sống lại trạng thái đã bị thu hồi? | snapshot và evaluate | lịch tác vụ bất đồng bộ, versioning, oracle quyền/thời gian |

## 2. Hướng 1 — gây sửa hoặc xoá nhầm memory hợp lệ

**Câu hỏi:** với quyền chỉ cung cấp một số đầu vào qua kênh bình thường, liệu đối
kháng có khiến memory manager tự supersede, merge hoặc xoá một thông tin đúng mà
nguồn đó không có quyền thay đổi? Không cấp quyền trực tiếp sửa database.

Ví dụ vô hại: workspace tổng hợp có hai dự án và hai hạn nộp khác nhau; kiểm tra
pipeline cập nhật có làm mất thông tin đã xác nhận của sai dự án hay không.

Prior work bắt buộc:

- [ForgetEval / Control-Plane Placement](https://arxiv.org/abs/2606.15903): đã có
  supersession, purge, amnesia và kiểm tra giữ fact của thực thể khác. Các đoạn đã
  đọc mô tả mutation calls và chấm top-k. Không được claim “đầu tiên nghiên cứu quên”.
- [Governance Decay](https://arxiv.org/abs/2606.22528): đã có attack khiến compaction
  bỏ ràng buộc. Mất ràng buộc do summary nói chung không phải gap mới.
- [ShadowMerge](https://arxiv.org/abs/2605.09033): đã có đầu vào thông thường đưa
  quan hệ mâu thuẫn vào graph memory. Attack vào merge/conflict nói chung đã có.

**Khác biệt cần kiểm chứng:** gây mất hoặc vô hiệu hoá bằng chứng đúng qua thao tác
quản lý memory; sau khi evaluator loại bản ghi đối kháng, lỗi vẫn còn vì thông tin
đúng đã mất. Đây là trạng thái hư hại khác với một poison đang thắng retrieval.
Chưa xác nhận các công trình trên hoàn toàn chưa đo hiện tượng này: cần đọc full
method, supplementary và code trước khi khóa claim.

Pilot nên có: backend mutable vs append-only, cùng stream; đầu vào đối kháng vs
đối chứng cùng độ dài; snapshot trước/sau; đo fact đúng còn ở store, còn active,
có được recall, và tác vụ có đúng. Tách rõ bốn tầng này. Đo collateral loss trên
fact không liên quan. Dùng oracle trong testbed để loại mọi bản ghi có nguồn đối
kháng, rồi thử phục hồi fact sạch: đây là can thiệp nhân quả, không phải defense
khả thi mặc định trong thực tế.

**Loại hướng** nếu chỉ là poison mới đẩy fact sạch khỏi top-k, hoặc phải cấp quyền
delete trực tiếp cho đối kháng. Không gọi đổi tên hiện tượng ShadowMerge là novelty.

Defense ứng viên sau khi có attack: ràng buộc quyền cho thao tác thay thế/xoá,
phiên bản có thể phục hồi và xác nhận độc lập đối với mutation có phạm vi lớn.
Phải đo chi phí từ chối cập nhật hợp lệ, không chỉ retention.

## 3. Hướng 2 — kiểm tra ngân sách đối kháng qua vòng biến đổi memory

**Câu hỏi:** nếu defense dùng chặn trên `t` bản ghi đối kháng để tính certificate,
quota `B` đầu vào được enforce thế nào khi memory được tách, gộp, tóm tắt hoặc
agent ghi lại? Đây là kiểm tra liên hệ giữa giao diện triển khai và giả định toán.

[SMSR](https://arxiv.org/abs/2606.12703) có certificate phụ thuộc ngân sách `t` và
nêu quota/dedup/audit để giới hạn nó; có cả đánh giá query-only write. Phải đối
chiếu định nghĩa `t` chính xác. [EVOMAL](https://arxiv.org/html/2608.25776) đã có
propagation, nên bản thân “một seed sinh nhiều hậu duệ” không phải novelty.
[TMA-NM](https://arxiv.org/abs/2606.24322) còn xét origin và corroboration chống Sybil.

**Gap ứng viên:** một thiết kế cụ thể có báo certificate với ngân sách không còn
đúng ở trạng thái hiện tại hay không; và có thể enforce/định nghĩa ngân sách theo
nguồn và phép biến đổi mà vẫn cho phép học hay không?

Pilot: tái lập certificate đúng threat model trước; theo dõi `B_input`, số bản
ghi bị ảnh hưởng, và số bản ghi đó trong candidate pool mà defense thực sự dùng.
Đối chiếu certificate với ngân sách thực và với ngân sách cấu hình. Không dùng
ASR vượt bound sau khi đã vi phạm giả định để tuyên bố theorem sai.

**Loại hướng** nếu deployment đã đếm toàn bộ bản ghi bị ảnh hưởng, hoặc chỉ nhận
được kết quả hiển nhiên “t lớn hơn thì bound yếu hơn” mà không chỉ ra lỗi enforce
có hậu quả. Khả năng tái dùng code cao, nhưng rủi ro trùng ý tưởng cũ cũng cao.

Defense ứng viên: budget accounting qua các phép biến đổi, giới hạn ảnh hưởng
theo nguồn, cập nhật/thu hồi certificate khi trạng thái vượt hợp đồng.

## 4. Hướng 3 — memory làm tăng chi phí của các tác vụ về sau

**Câu hỏi:** sau giai đoạn đầu vào đối kháng đã kết thúc, memory có khiến các tác
vụ sạch sau đó tăng chi phí retrieval, consolidation hoặc reasoning, dù kết quả
cuối vẫn đúng? Đối tượng là availability/chi phí, không chỉ hành vi sai.

[Beyond Max Tokens](https://arxiv.org/abs/2601.10955) đã có khuếch đại chi phí qua
tool loop dưới ràng buộc task success. Các đoạn đã đọc đặt quyền đối kháng ở MCP
server trong tương tác. Đây là baseline gần, không được claim resource attack mới
nói chung. Khác biệt **cần tìm bằng chứng** là chi phí lưu lại qua memory và xuất
hiện trên tác vụ sạch sau khi nguồn đối kháng không còn hoạt động.

Pilot dùng testbed nhỏ có ngân sách cứng: so memory sạch, memory từ đầu vào thử
nghiệm và memory từ đầu vào lành có cùng kích thước. Đo chi phí toàn vòng đời,
token/call, latency, task success; reset memory để kiểm tra trung gian nhân quả.
Counter chi phí trong repo hỗ trợ một phần, chưa bao phủ mọi backend.

**Loại hướng** nếu chỉ do context dài hơn tuyến tính, phải tiếp tục dùng tool độc,
hoặc lỗi biến mất hoàn toàn dưới giới hạn tài nguyên thông thường với utility giữ
nguyên. Chưa thấy một nguồn trùng đầy đủ qua đợt tìm này không phải bằng chứng mới.

Defense ứng viên: giới hạn công việc cho mỗi operation và mỗi nguồn, cache/reuse
kết quả consolidation, đánh giá utility dưới cùng ngân sách. Cần chứng minh hạn
mức không làm hỏng tác vụ dài hợp lệ.

## 5. Hướng 4 — thứ tự cập nhật và hiệu lực của memory

**Câu hỏi:** khi nhiều tác vụ cập nhật bất đồng bộ, đối kháng có thể dùng quyền
đầu vào hạn chế để khiến một bản cũ ghi đè hoặc phục hồi trạng thái đã bị thu hồi?
Cần định nghĩa đối kháng thực sự kiểm soát nội dung, thời điểm hay cả hai.

[MemTxn](https://arxiv.org/abs/2607.27834) có admission, temporal resolution và
recovery, nhưng giới hạn được nêu gồm concurrency/repeated faults và source truth.
[TOKI](https://arxiv.org/abs/2606.06240) đã nghiên cứu write-time concurrency và
isolation; phần đã đọc phân biệt hợp đồng well-formed writes với adversarial injection.
[ChronoMem](https://arxiv.org/abs/2607.27773) và
[TEPA](https://arxiv.org/abs/2608.07429) cũng phải đối chiếu về rollback/revocation.

**Gap ứng viên:** một attack qua giao diện thông thường gây sai hiệu lực/thẩm quyền
trên stack thực, với hậu quả agent end-to-end. Không chỉ tái hiện lost update hoặc
race cổ điển trong database, không nhận lỗi ngoài giả định là bác bỏ theorem.

Pilot: cùng dữ liệu, lịch tuần tự vs bất đồng bộ; oracle version và quyền; chấm
trạng thái cuối lẫn quyết định ở giữa các lần ghi. Thử baseline version check và
serialization trước khi phát triển defense phức tạp.

**Loại hướng** nếu chỉ còn bug lập trình thông thường hoặc một version check sẵn
có giải quyết hết. Hướng này đòi hạ tầng mới nhiều nhất trong bốn lựa chọn.

## 6. Các hướng đã đông — không lấy ý chung làm novelty

| Ý tưởng tổng quát | Nguồn gần đã tìm được |
|---|---|
| Các memory riêng lẻ vô hại nhưng kết hợp gây sai | [MemCollusion/Salami Attack](https://arxiv.org/abs/2608.01637), [MemPoison 2607.14651](https://arxiv.org/abs/2607.14651) |
| Kinh nghiệm đúng cục bộ gây khái quát sai | [OEP](https://arxiv.org/abs/2605.18930) |
| Giả reasoning, làm sai consensus | [FARMA](https://arxiv.org/abs/2607.05029), [TMA-NM](https://arxiv.org/abs/2606.24322) |
| Tối ưu toàn chuỗi write–retrieve–use | [PipePoison](https://arxiv.org/abs/2609.00523), [BMA](https://arxiv.org/abs/2609.32186) |
| Tối ưu hậu quả thay vì ASR | [MemHarm](https://arxiv.org/abs/2609.34132) |
| Mất provenance qua consolidation | [PPMF](https://arxiv.org/abs/2607.29167), [Misattribution Gap](https://arxiv.org/abs/2605.22842) |
| Memory đa phương thức | [MemVenom](https://arxiv.org/abs/2606.10742), [Lucid](https://arxiv.org/abs/2607.15657) |
| Membership inference hoặc trích xuất memory | [MRMMIA](https://arxiv.org/abs/2605.27825), [SPORE](https://arxiv.org/abs/2607.23444) |
| Xoá memory nhưng ảnh hưởng vẫn còn trong execution state | [Execution-State Unlearning](https://arxiv.org/abs/2609.04875), [Dependency-Guided Rollback Repair](https://arxiv.org/abs/2608.10502) |

Privacy vẫn là một nhánh attack khác nếu muốn đổi mục tiêu, nhưng cần một câu hỏi
hẹp về quyền truy cập hoặc vòng đời thông tin; “MIA lên agent memory” đã có bài.

## 7. Mức độ kiểm chứng nguồn và bước tiếp theo

Đã dùng nhiều framing semantic search và hai lượt related-paper expansion; hai
lượt expansion không trả về ứng viên nên tiếp tục tìm theo truy vấn riêng.
Đã đọc passages thân bài của ForgetEval, MemTxn, SMSR, ShadowMerge, TOKI và Beyond
Max Tokens; các nguồn còn lại trong lượt này chủ yếu ở mức abstract. Passages
không trả lời một vấn đề **không chứng minh bài không làm vấn đề đó**.

Trước khi viết method, ưu tiên hướng 1 và làm ba việc: (1) đọc đầy đủ/code của ba
prior gần nhất; (2) chọn backend thực có mutation và khóa quyền đối kháng;
(3) dựng thí nghiệm đối chứng nhỏ để xác định có mất trạng thái sạch hay chỉ đổi
retrieval. Nếu không phân biệt được cơ chế mới, loại hướng sớm. Hướng 2 là lựa chọn
thay thế gần hạ tầng hiện có nhất; chưa cần train generator cho cả hai.
