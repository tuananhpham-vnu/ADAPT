# Stage 2 — Kiểm thử tích hợp giữa các agent (khoảng 4 tuần)

Trả lời câu hỏi "các agent khi ghép lại thì thế nào", và sửa đúng giới hạn paper gọi là
*unrealistic test cases*.

Code: `src/adapt/integration/`

## Vấn đề

ARTEMIS cô lập từng agent. Test case của agent B được sinh từ system prompt của B, rồi đưa thẳng
user query vào B. Nhưng trong hệ thật, B không nhận query từ người dùng — B nhận **output của A**.

Hệ quả: có những test case của B mô tả tình huống mà A không bao giờ tạo ra được. Điểm Q của B khi
đó bị trừ vì những tình huống không tồn tại trong thực tế. Ngược lại, có những tình huống A thường
xuyên tạo ra mà bộ test case của B lại không phủ.

## Cách làm

1. Lấy các cạnh `A -> B` từ đồ thị. Dùng lại `extractors/graph_assembler.py` và
   `extractors/router_extractor.py` của upstream, không tự parse lại code MAS.
2. Chạy A trên bộ test case của chính A, thu **output thật**. Đây là phân phối đầu vào thực tế của B.
3. Sinh test case cho B với ràng buộc: user query phải là thứ A có thể tạo ra, cả về nội dung lẫn
   định dạng.
4. Nếu không output nào của A đặt được B vào test state `z` thì đánh dấu `z` là **unreachable**.

## Metric

- **Tỉ lệ khả đạt** — bao nhiêu phần trăm test case của phase 1 là không thể xảy ra trong luồng thật.
  Con số này tự nó đã là một kết quả công bố được: nó định lượng chính xác mức lạc quan hoặc bi quan
  của cách đo cô lập. Nếu tỉ lệ unreachable cao thì mọi điểm Q trong Table 4 đều cần đọc lại có điều
  kiện.
- **Tỉ lệ vi phạm hợp đồng** — bao nhiêu phần trăm output của A vi phạm điều kiện đầu vào của B.

Hai metric này bổ sung cho nhau: cái đầu nói bộ test case sai ở đâu, cái sau nói hệ thống sai ở đâu.

## Nghiệm thu

Chạy trên hệ `langchain-multi-agents` — đúng hệ trong ví dụ tạo động lực của paper, có 4 agent theo
vòng writer-reviewer. Yêu cầu: tái lập được lỗi `ScenarioWriter` phát `FINAL ANSWER` sớm **bằng
metric vi phạm hợp đồng**, chứ không phải bằng cách chạy 200 lần rồi ngồi đọc log như paper đã làm.
Nếu metric này không bắt được lỗi mà cả paper lẫn người đọc đều thấy rõ thì định nghĩa hợp đồng đang
sai, phải sửa trước khi chạy rộng.

## Ghi chú

Stage này là tiền đề kỹ thuật cho stage 14: muốn đo lan truyền lỗi qua cả luồng thì trước hết phải
biết đầu ra của agent này có hợp lệ làm đầu vào cho agent kia hay không.
