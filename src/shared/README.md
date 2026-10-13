# `src/shared/` — hạ tầng dùng chung

- `tools.py`: schema và thực thi hai tool ngân hàng giả lập, dùng cho demo provenance gate
  (`src/artemis/run_gate_demo.py`).
- `encoding.py`: encode văn bản bằng retriever đã nạp (`src/artemis/retrieval.py`).
- `tracing.py`: tracing Langfuse (`src/artemis/engine.py`, `run_gate_demo.py`).

Không đặt dữ liệu tấn công, gate hay logic sửa prompt ở đây, và không import từ
`src/attack/`, `src/defense/`, `src/artemis/`. Provider LLM nằm trong `src/providers/`;
cấu hình dùng chung ở `src/config.py`.
