# Roadmap tổng

Bản đồ các stage mở rộng ARTEMIS. Chi tiết từng stage ở `11` đến `15`, nền lý thuyết ở
`10_artemis_overview.md`.

## Thứ tự và phụ thuộc

```
Stage 0 (nền)
   |
   +--> Stage 1 (cải tiến prompt)  <-- ưu tiên cao nhất
   |         |
   +--> Stage 2 (tích hợp)         |
             |                     |
             v                     |
        Stage 3 (toàn luồng) <-----+  cần bản vá của Stage 1 để chứng minh cải thiện
             |
             v
        Stage 4 (RAG + tooling) ---+  khép vòng ngược về Stage 1
             |
             v
        Stage 5 (đánh giá và viết bài)
```

| Stage | Nội dung | Thời gian | Trạng thái paper |
|---|---|---|---|
| 0 | Vendor code, cấu hình model, chạy lại baseline | 1 tuần | đã có |
| 1 | Vòng cải tiến prompt từ điểm số | 4-6 tuần | future work |
| 2 | Hợp đồng giữa cặp agent, tỉ lệ khả đạt | 4 tuần | limitation |
| 3 | Toàn luồng: độ phủ, lan truyền lỗi, dừng sớm | 4 tuần | có khung phase 3 |
| 4 | Taxonomy TC/GC, RAG bị đầu độc, AutoGen | 6 tuần | limitation |
| 5 | Chạy đủ benchmark, viết bài | 4 tuần | — |

Stage 1 và 2 độc lập với nhau nên chạy song song được nếu có hai người.

## Đóng góp mới so với paper

- Vòng cải tiến prompt có kiểm soát overfit bằng tập held-out.
- Tỉ lệ khả đạt: định lượng bao nhiêu phần trăm test case cô lập là không thể xảy ra thật.
- Tỉ lệ lan truyền lỗi: nối điểm ở mức agent với chất lượng đầu ra cuối cùng.
- Khoảng cách bền vững dưới tấn công đầu độc RAG.
- Hỗ trợ AutoGen, không chỉ LangGraph.

## Câu hỏi nghiên cứu đề xuất cho Stage 5

1. Vòng cải tiến prompt nâng được Q bao nhiêu trên tập held-out?
2. Bao nhiêu phần trăm test case cô lập là không khả đạt trong luồng thật?
3. Lỗi ở một agent lan tới đầu ra cuối cùng với xác suất bao nhiêu?
4. Khoảng cách bền vững dưới tấn công đầu độc RAG là bao nhiêu, và prompt nào chịu được?
5. Chi phí bao nhiêu?

## Ngân sách token

Paper báo hệ 15 và 16 tốn khoảng 25,8 triệu token mỗi hệ, tổng cả 16 hệ vượt 80 triệu. Bước judge
chiếm phần lớn, khoảng 1,005 triệu token mỗi agent.

Hai cách giảm, cả hai đều dựa trên kết quả của chính paper:

- Đặt `n_judge = 1`. RQ4 cho thấy tăng lên 5 vòng chấm chỉ đổi accuracy 0,01. Riêng cái này cắt phần
  judge còn khoảng một phần ba.
- Dùng model rẻ cho vai judge. Nhưng vẫn phải giữ judge khác test model — `check_role_separation()`
  trong `src/config.py` cảnh báo khi hai vai trùng nhau.

Đừng giảm `n_run`. Instability trung bình đo được là 0,042 với 3 lần chạy và 0,076 với 10 lần; chạy
ít thì metric instability mất ý nghĩa.

Với Stage 5, chốt danh sách hệ cần chạy trước khi bấm nút. Chạy đủ 16 hệ nhiều lần cho các cấu hình
ablation là khoản tốn nhất của cả dự án.

## Rủi ro

**Overfit prompt.** Rủi ro lớn nhất, nằm ở Stage 1. Đã có tập held-out và judge chéo, nhưng phải báo
cáo trung thực cả `delta_Q_seen` lẫn `delta_Q_heldout`. Chỉ đưa con số đẹp thì phản biện sẽ hỏi ngay.

**Judge bias.** Judge chấm cao hơn cho response do chính model đó sinh. Luôn tách hai vai.

**Trôi theo upstream.** Giữ `src/artemis/` nguyên vẹn và chỉ sửa `model_factory.py`, để
`git subtree pull` còn merge được khi anh Hiệp cập nhật code.

**Phòng thủ quá tay ở Stage 4.** Prompt vá theo hướng nghi ngờ mọi context sẽ thu hẹp khoảng cách
bền vững nhưng làm tụt luôn `Q_lành`. Đo cả hai.
