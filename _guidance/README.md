# `_guidance/` — Hướng dẫn chạy theo kịch bản

Thư mục này là tài liệu vận hành, viết bằng tiếng Việt, đọc theo thứ tự số. Trong khi
README của từng thư mục code trả lời câu hỏi "cái này là gì", `_guidance/` trả lời câu
hỏi "tôi phải gõ gì, theo thứ tự nào".

## Đọc theo thứ tự

| File | Nội dung |
|---|---|
| `00_setup.md` | Cài môi trường, dữ liệu nào có sẵn / phải tải, key nào cần. |
| `01_demo_fast.md` | Chạy trọn vòng (tối ưu → tiêm → đo) trong ~20 phút để kiểm tra pipeline còn sống. Số liệu không dùng để báo cáo. |
| `02_evaluation.md` | Bốn chỉ số ACC, ASR-r, ASR-a, ASR-t nghĩa là gì và đọc thế nào. |
| `03_attack.md` | Giải thích thuật toán tối ưu trigger và các tham số quan trọng. |
| `04_end_to_end.md` | Chạy một thí nghiệm đầy đủ để lấy số báo cáo, gồm cả agent `ad`. |
| `run.md` | Ghi chú chạy demo Corba. |

## Nguyên tắc chung khi chạy

1. Luôn chạy lệnh từ **thư mục gốc repo** (nhiều đường dẫn trong code là đường dẫn tương đối).
2. Sau khi tối ưu trigger, phải **dán trigger vào `trigger_token_list`** trong script
   inference trước khi chạy nhánh `adv` — không có bước tự động.
3. Script inference **ghi nối tiếp** vào file kết quả; chạy lại phải `make clean-out`.
4. Đổi embedder thì `make clean-cache`, nếu không sẽ dùng nhầm cache embedding cũ.
