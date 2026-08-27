# 2. Evaluation — đo và đọc kết quả

> Đọc [00_setup.md](00_setup.md) trước. Mọi lệnh chạy từ repo root.

## 2.1 Bốn chỉ số chính

| Chỉ số | Nghĩa | Tính ở đâu |
|---|---|---|
| **ACC** | Accuracy của agent trên tác vụ gốc. Trigger tốt phải giữ ACC ở nhánh benign gần như không đổi. | [ReAct/eval.py](ReAct/eval.py) |
| **ASR-r** (retrieval) | Tỉ lệ item độc được kéo lên trong top-k khi query có trigger. Đo riêng tầng RAG. | như trên |
| **ASR-a** (action) | Tỉ lệ agent thực sự sinh ra hành động/đáp án mục tiêu **khi** retrieval đã thành công. | như trên |
| **ASR-t** (target/end-to-end) | Tỉ lệ tấn công thành công tính trên toàn bộ test set. Trong ReAct hiện được tính là `1 - ACC`. | như trên |

Cần cả hai nhánh mới đọc được kết quả:
`--task_type benign` (không tiêm trigger) và `--task_type adv` (có trigger).

## 2.2 ReAct / StrategyQA

Sinh kết quả (tạo thư mục output trước — script không tự `mkdir`):

```powershell
mkdir result\ReAct -Force | Out-Null
# nhánh sạch
python ReAct/run_strategyqa_gpt3.5.py --model dpr --algo ap --task_type benign --save_dir ./result/ReAct
# nhánh tấn công (nhớ đã dán trigger vào file, xem 03_attack.md)
python ReAct/run_strategyqa_gpt3.5.py --model dpr --algo ap --task_type adv    --save_dir ./result/ReAct
```

File output: `result/ReAct/<embedder>-<algo>-<task_type>.jsonl`, ví dụ
`result/ReAct/dpr-ap-adv.jsonl`. Mỗi dòng có `answer`, `gt_answer`, `traj`,
`retrieval_success`, `overall_retrieval`.

Chấm điểm:

```powershell
python ReAct/eval.py -p ./result/ReAct/dpr-ap-benign.jsonl
python ReAct/eval.py -p ./result/ReAct/dpr-ap-adv.jsonl
```

⚠️ Script inference **append**, không ghi đè. Xoá `.jsonl` cũ trước mỗi lần chạy
lại, nếu không kết quả hai lần chạy bị trộn.

Backbone LLaMA-3 qua API dùng `ReAct/run_strategyqa_llama3_api.py` (thêm `--skip N`
để chạy tiếp từ câu N sau khi bị ngắt). Kết quả tham chiếu có sẵn trong
[ReAct/ablation/](ReAct/ablation/) để đối chiếu.

## 2.3 EhrAgent

```powershell
python EhrAgent/ehragent/main.py --backbone gpt --model dpr --algo ap --attack --num_questions 50
python EhrAgent/ehragent/main.py --backbone gpt --model dpr --algo ap          --num_questions 50   # benign
```

Output: `result/Ehragent/<backbone>/<algo>_<trigger|benign>_<embedder>.json`,
ví dụ `result/Ehragent/gpt/ap_trigger_dpr.json`.

> Thư mục output **không được tạo tự động** — `mkdir result\Ehragent\gpt` trước khi chạy.
> `--num_questions -1` để chạy toàn bộ dataset; `--num_questions 10` cho demo.

Chấm điểm:

```powershell
python EhrAgent/ehragent/eval.py -p ./result/Ehragent/gpt/ap_trigger_dpr.json
```

`eval.py` của EhrAgent nạp lại long-term memory từ
`EhrAgent/database/ehr_logs/logs_final` và **cần GPU** (`device = "cuda"` để load BERT).
ASR-r ở đây được tính theo việc knowledge độc (`Delete`/`delete`) xuất hiện ≥3 lần
trong phần retrieval.

## 2.4 Agent-Driver

Cần dữ liệu nuScenes đã tiền xử lý dưới `agentdriver/data/` (repo chỉ có
`split.json`) — xem [04_end_to_end.md](04_end_to_end.md) §4.4.

```powershell
$env:PYTHONPATH = $PWD
bash scripts/agent_driver/run_inference.sh
bash scripts/agent_driver/run_evaluation.sh uniad <path-to-result.pkl>
```

`--metric` nhận `uniad` hoặc `stp3`; ground truth đọc từ `--gt_folder`
(mặc định `data/metrics`). Chỉ số là L2 error và collision rate của quỹ đạo.

## 2.5 Đánh giá riêng embedder (không cần chạy agent)

Nhanh và rẻ hơn nhiều khi chỉ muốn so sánh chất lượng retriever:

```powershell
python embedder/eval_embed_contrastive.py
python embedder/eval_embed_classification.py
```

## 2.6 Bảng so sánh nên dựng

Với mỗi (agent × embedder) báo cáo 4 dòng:

| Setting | ACC | ASR-r | ASR-a | ASR-t |
|---|---|---|---|---|
| benign, no trigger | baseline | – | – | – |
| adv, golden trigger (chưa tối ưu) | | | | |
| adv, trigger tối ưu (`--ppl_filter`) | | | | |
| adv, trigger tối ưu + `--target_gradient_guidance` | | | | |

Kết luận cần: **ASR cao mà ACC ở nhánh benign không tụt** — trigger vừa hiệu quả
vừa tàng hình.
