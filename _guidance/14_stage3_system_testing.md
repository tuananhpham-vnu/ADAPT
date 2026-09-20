# Stage 3 — Kiểm thử toàn luồng (khoảng 4 tuần)

Phần anh Hiệp muốn xem kỹ hơn. Không phải làm từ đầu: upstream đã có `nodes/phase3_nodes.py` chạy
end-to-end, nhưng đang dùng **mock agent**. Việc của stage này là thay mock bằng agent thật và định
lượng ba hiện tượng chỉ xuất hiện ở mức luồng.

Code: `src/adapt/system/`

## Ba metric

**Độ phủ đường đi.** Bao nhiêu nhánh trong đồ thị thực sự được đi qua so với số nhánh có thể đi. Một
hệ mà mọi lần chạy đều đi đúng một đường thì các agent ở nhánh còn lại chưa từng được kiểm chứng
trong ngữ cảnh thật, dù điểm Q của chúng có đẹp đến đâu.

**Tỉ lệ lan truyền lỗi.** Xác suất output cuối cùng hỏng khi agent thứ `i` hỏng. Đây là cầu nối giữa
điểm Q ở mức agent và chất lượng đầu ra mà người dùng thực sự nhận được. Paper có dẫn nghiên cứu về
hallucination cascade nhưng không đo đại lượng này. Nếu tỉ lệ lan truyền thấp thì một agent điểm kém
nằm giữa luồng có thể không đáng lo; nếu cao thì thứ tự ưu tiên sửa prompt phải theo vị trí trong
luồng chứ không theo điểm Q.

**Tỉ lệ dừng sớm và tỉ lệ không dừng.** Biến ví dụ tạo động lực của paper thành thứ đo tự động được.
Dừng sớm là pipeline kết thúc trước khi đi hết các agent cần thiết. Không dừng là vòng
writer-reviewer lặp mãi tới khi chạm giới hạn lượt.

## Nghiệm thu

- Tỉ lệ lan truyền lỗi phải tương quan dương với `1 - Q` của agent đứng đầu chuỗi. Không tương quan
  thì hoặc cách xác định "output cuối hỏng" đang sai, hoặc hệ thật sự có khả năng tự sửa lỗi ở các
  agent phía sau — cả hai đều đáng viết ra.
- Hệ `langchain-multi-agents` phải cho tỉ lệ dừng sớm cao trước khi vá, và giảm rõ sau khi áp bản vá
  của stage 12.

Điểm thứ hai là chỗ ba stage đầu khớp lại thành một câu chuyện hoàn chỉnh: đo được lỗi ở mức agent
(stage 0), vá được prompt (stage 12), và chứng minh bản vá cải thiện hành vi ở mức toàn hệ (stage
này). Nếu vá prompt làm Q tăng mà tỉ lệ dừng sớm không giảm thì bản vá chỉ đang chiều lòng judge chứ
không sửa được hệ thống.
