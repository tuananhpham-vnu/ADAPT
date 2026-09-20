# ADAPT: các cải tiến của dự án

Đây là phần triển khai của dự án, không phải mã gốc ARTEMIS.

| Module | Vai trò |
|---|---|
| `spec.py`, `benchmark.py` | Requirement, case có nhãn, chia train/validation/heldout |
| `engine.py`, `retrieval.py` | Chạy có ngân sách, resume, truy hồi |
| `oracle.py`, `metrics.py` | Chấm hành vi, coverage, thống kê |
| `mutation.py`, `analysis_tools.py` | Sinh mutant và phân tích lỗi |
| `repair.py`, `experiment.py` | Vá prompt, so sánh baseline và đánh giá |
| `provenance_gate.py`, `run_gate_demo.py` | Policy kiểm tra tool trước thực thi và pilot gate |
| `build_explainer.py`, `explainer.template.html` | Tạo trang giải thích từ log gate |
| `extensions/prompt_improve/` | API phân loại lỗi, sinh/áp patch và vòng cải tiến qua callback |
| `extensions/agent_contract/`, `extensions/system_testing/` | Hợp đồng agent và đồ thị kiểm thử |
| `extensions/rag_tooling/` | Entry point tương thích tới pipeline ADAPT |

```bash
python -m src.adapt --plan
python -m src.adapt --backend fixture --groups-per-split 1 --repeats 1 --mutants 2 --repair-rounds 1 --bootstrap 20
python -m src.adapt --backend live --provider deepseek
python -m src.adapt.run_gate_demo --help
```

Fixture là dữ liệu kiểm thử tổng hợp, không phải bằng chứng hiệu quả trên LLM.
Kết quả pipeline nằm trong `results/adapt/`; pilot gate dùng `results/gate_demo/`.
Các extension có code nhưng chưa có kết nối chạy baseline ARTEMIS upstream.
Vòng `extensions/prompt_improve/loop.py` cần caller cung cấp evaluator, proposer
và split; pipeline chạy được qua CLI là `src.adapt`.

`src/integration/` và các module gate cũ trong `src/toolpoison/` là alias tương thích.
