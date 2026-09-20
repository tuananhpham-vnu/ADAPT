# `scripts/` — Script chạy sẵn và tiện ích

Thư mục gom các lệnh dài thành file để khỏi gõ tay. Có hai nhóm: **script shell theo
agent** (bản gốc của upstream) và **tiện ích Python** phục vụ Makefile.

Ghi chú: cách chạy được khuyến nghị trong repo này là `make <target>` (xem `Makefile`);
các script shell dưới đây giữ lại vì chúng ghi rõ tham số gốc của paper.

## 1. Script theo agent

| Thư mục | Script | Làm gì |
|---|---|---|
| `react_strategyqa/` | `run_optimization.sh` | Tối ưu trigger cho agent `qa`. |
| | `run_inference_gpt.sh`, `run_inference_llama3.sh` | Chạy inference StrategyQA theo backbone. |
| `ehragent/` | `run_optimization.sh` | Tối ưu trigger cho agent `ehr`. |
| | `run_inference_gpt.sh`, `run_inference_llama3.sh` | Chạy inference EhrAgent. |
| `agent_driver/` | `run_optimization.sh` | Tối ưu trigger cho agent `ad`. |
| | `run_inference.sh` | Chạy inference Agent-Driver. |
| | `run_finetune.sh` | Fine-tune planner (GPT-3.5) cho Agent-Driver. |
| | `run_evaluation.sh` | Chấm L2 error và tỉ lệ va chạm. |

Mỗi script chỉ là một lời gọi `python ...` với bộ tham số cố định — mở ra đọc là thấy
ngay tham số nào đang dùng.

## 2. Tiện ích

| File | Làm gì |
|---|---|
| `make_help.py` | Đọc `Makefile`, in danh sách target kèm mô tả (chính là output của `make`). |
| `show_trigger.py` | Tìm lần tối ưu gần nhất trong `results/<agent>/<algo>/` và in trigger cuối cùng. Dùng qua `make trigger`. |
| `run_ablation_sweep.sh` | Chạy một lưới thí nghiệm ablation (cần bash + nvidia-smi). Dùng qua `make sweep`. |

## 3. Ví dụ

```bash
make trigger AGENT=qa ALGO=ap        # chạy scripts/show_trigger.py
make sweep NUM_ITER=200              # chạy scripts/run_ablation_sweep.sh
bash scripts/ehragent/run_inference_gpt.sh
```
