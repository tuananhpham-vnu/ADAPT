# Hạ tầng dùng chung

- `tools.py`: schema và thực thi hai tool ngân hàng giả lập.
- `encoding.py`: encode văn bản bằng retriever đã nạp.
- `tracing.py`: tracing Langfuse.

Không đặt dữ liệu tấn công, gate hay logic sửa prompt ở đây. Các provider LLM
tiếp tục nằm trong `src/providers/`; cấu hình dùng chung ở `src/config.py`.
