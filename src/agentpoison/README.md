# AgentPoison: tấn công và thực nghiệm

**Chạy với corpus thật:** xem [STRATEGYQA.md](STRATEGYQA.md).
`make agentpoison` hoặc `./make.ps1 agentpoison` chạy toàn bộ corpus StrategyQA,
DPR thật và provider thật; `make agentpoison-demo` chạy demo ngân hàng bên dưới.

Entry point chính dùng corpus thật, hỗ trợ `check`, `index`, `run`, `report`:

```bash
python -m src.agentpoison check
python -m src.agentpoison --num-queries 10 --provider deepseek
```

Pipeline artifact theo phase và ablation có các lệnh `optimize`, `prepare`, `retrieve`,
`infer`, `evaluate`, `all`, `ablate`; xem [STRATEGYQA.md](STRATEGYQA.md).

Nếu không ghi subcommand thì mặc định là `run`. Có thể dùng
`python -m src.main agentpoison ...` với cùng tham số.

Demo tool-calling ngân hàng được gọi riêng từ gốc repo:

```bash
python -m src.agentpoison.run_agentpoison_demo --num-queries 10 --provider deepseek
python -m src.agentpoison.attack_report --run results/agentpoison/20260907_152107
```

`run_agentpoison_demo.py` chạy thiết kế 2×2: memory sạch/độc × query không/có
trigger. Kết quả vẫn ghi vào `results/agentpoison/<timestamp>/`, gồm manifest,
records, summary, final và REPORT.md. Lệnh thực nghiệm gọi model thật và nạp DPR;
lệnh report chỉ đọc kết quả. Hai tool ngân hàng được giả lập.

`agent.py`, `memory.py`, `eval.py` chứa agent, dữ liệu độc và phép đo;
`run_demo.py`, `experiment.py` giữ các pilot trước đây.

**Đây là demo với seed trigger có sẵn, không tối ưu trigger trong lần chạy.**
Tái lập pipeline AgentPoison đầy đủ dùng `algo/` để tối ưu, rồi `ReAct/`,
`EhrAgent/`, `agentdriver/` để inference và đánh giá, cùng `embedder/` và
`scripts/`. Các lệnh `make opt-*`, `make run-*`, `make eval-*` vẫn giữ nguyên.

Phòng thủ và sửa prompt nằm trong `src/adapt/`. Hạ tầng dùng chung nằm trong
`src/shared/` và `src/providers/`. `src/toolpoison/` chỉ còn alias tương thích.
