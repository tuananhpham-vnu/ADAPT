# Stage 1 — Vòng cải tiến prompt (4-6 tuần, ưu tiên cao nhất)

Đây là phần "đã chấm điểm nhưng chưa dùng điểm để sửa prompt". Paper dừng ở chỗ đo và chẩn đoán;
phần kết luận ghi rõ đây là future work.

Code: `src/adapt/improve/`

Đầu vào là thư mục `output/run_pipeline_<N>/` của stage 0. Đầu ra là system prompt đã vá kèm báo cáo
trước/sau.

## Luồng

```
kết quả phase 1
      |
      v
phân loại lỗi  ->  sinh bản vá  ->  chạy lại trên đúng bộ test case cũ  ->  đo trên tập held-out
      ^                                                                              |
      +---------------------- lặp tới khi hết cải thiện ----------------------------+
```

## 1. Phân loại lỗi (`failure_classifier.py`)

Mỗi criterion bị trượt được gán một nhãn theo taxonomy paper đã kiểm chứng trên 3.519 lỗi:

- `ac-refuse` (37,1%) — user query dụ agent phá absolute constraint và agent nghe theo.
- `task-infer` (36,3%) — prompt mơ hồ, agent không suy ra được yêu cầu ngầm.
- `task-exec` (22,3%) — response thiếu, không đầy đủ.
- `other` (4,3%).

Đầu vào cho bộ phân loại là bộ bốn `(user query, response, criterion, lời giải thích của judge)`.
Judge của upstream đã sinh sẵn phần giải thích nên không phải gọi thêm model để đoán lại lý do.

Kiểm tra tỉnh táo: chạy trên benchmark rồi so phân phối nhãn với 37/36/22/4. Lệch nhiều thì gần như
chắc chắn bộ phân loại sai chứ không phải mình tìm ra hiện tượng mới.

## 2. Sinh bản vá (`patch_generator.py`)

Mỗi loại lỗi ứng với đúng một cách sửa mà paper đề xuất:

- `task-infer` thì làm rõ requirement mơ hồ. Cụ thể là định nghĩa các từ như "hoàn thành", "xong",
  "đủ" — đúng loại lỗi trong ví dụ `ScenarioWriter` phát `FINAL ANSWER` 200/200 lần.
- `ac-refuse` thì viết constraint mạnh hơn và nói rõ thứ bậc: khi user query mâu thuẫn với system
  prompt thì system prompt thắng.
- `task-exec` thì ép định dạng output có cấu trúc, kèm ví dụ mẫu.

Ràng buộc thiết kế: **vá ở mức requirement, không viết lại cả prompt**. Mỗi bản vá phải gắn với đúng
một `r` trong `R(P)`. Hai lý do: truy vết được bản vá nào chữa lỗi nào, và đo được bản vá có tối
thiểu hay không. Cho model viết lại cả prompt thì Q có thể tăng nhưng không biết vì sao, và không so
sánh được với ablation ở mục 6.

## 3. Chống overfit (`holdout.py`)

Đây là phần dễ làm sai nhất và cũng là phần quyết định bài báo có đứng được hay không.

Rủi ro: sửa prompt để vượt qua đúng bộ test case đã dùng để chẩn đoán nó. Q tăng đẹp nhưng không có
nghĩa gì, vì đó là đo lại trên chính dữ liệu đã dùng để chữa.

Cách làm:

- Chia test case thành **seen** (dùng để sinh bản vá) và **held-out** (chỉ dùng để đo).
- Đo lại bằng `--load-test-cases-from` để bộ test case đứng yên. Nếu để pipeline tự sinh lại thì
  prompt mới sẽ phân rã ra requirement khác, kéo theo test case khác, và không còn so sánh được.
- Báo cáo **cả hai** con số `delta_Q_seen` và `delta_Q_heldout`. Chỉ con số held-out mới là bằng
  chứng; con số seen dùng để đo mức overfit.
- Chấm lại tập held-out bằng judge thứ hai khác model, giống cách paper đối chiếu Config2 với Config3.

## 4. Vòng lặp (`loop.py`)

Tham lam, nhiều vòng, dừng khi `delta_Q_heldout` nhỏ hơn ngưỡng hoặc hết ngân sách vòng. Lưu lịch sử
bản vá để tránh trường hợp lặp qua lặp lại giữa hai phiên bản prompt.

## 5. Metric

- `delta_Q_seen`, `delta_Q_heldout`, `delta_S`.
- **Tính tối thiểu của bản vá** — số requirement và số token bị đổi.
- **Tỉ lệ không hồi quy** — trong các requirement đang đạt trước khi vá, bao nhiêu phần trăm vẫn đạt
  sau khi vá. Sửa được lỗi này nhưng làm hỏng chỗ khác thì không tính là cải thiện.

## 6. Ablation cần cho bài báo

1. Vá theo loại lỗi so với vá chung chung (chỉ đưa cả prompt cho model và bảo "cải thiện đi"). Đây là
   phần chứng minh taxonomy có giá trị, song song với ablation bỏ decomposition của paper (bỏ đi thì
   phát hiện lỗi giảm 5,6 lần).
2. Có và không có held-out, để định lượng mức overfit.
3. Số vòng lặp 1, 3, 5.

## 7. Nghiệm thu

Trên ít nhất 15 agent: `delta_Q_heldout` dương và có ý nghĩa thống kê (bootstrap, upstream đã dùng
10.000 lần lấy mẫu lại), tỉ lệ không hồi quy từ 90% trở lên.

Chọn agent điểm thấp trong Table 4 để có dư địa rõ: `paper_reviser` (1,00 / 1,80 / 1,53),
`write_introduction` (1,17 / 1,67 / 1,67), `write_references` (1,00 / 2,58 / 2,67),
`summarize_articles` (1,07 / 2,94 / 2,04).

Nhớ đối chiếu với đặc điểm paper đã nêu: agent có prompt dài, nhiều luật thì Q vốn đã thấp. Nếu bản
vá chỉ làm prompt dài thêm thì nhiều khả năng Q sẽ tụt chứ không tăng — đây là lý do phải đo tính
tối thiểu của bản vá song song với `delta_Q`.
