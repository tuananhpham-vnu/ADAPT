# Evaluation

Bốn chỉ số, đều do `ReAct/eval.py` in ra:

- **ACC** — accuracy trên tác vụ gốc. Trigger tốt thì ACC ở nhánh benign không tụt.
- **ASR-r** — tỉ lệ item độc lọt top-k khi query có trigger, đo riêng tầng retrieval.
- **ASR-a** — tỉ lệ agent thực sự sinh hành động mục tiêu, tính trên các case đã
  retrieval thành công.
- **ASR-t** — end-to-end, trong ReAct đang tính là `1 - ACC`.

Luôn cần cả hai nhánh benign và adv mới đọc được kết quả.

## ReAct / StrategyQA

```powershell
mkdir result\ReAct -Force | Out-Null
python ReAct/run_strategyqa_gpt3.5.py --model dpr --algo ap --task_type benign
python ReAct/run_strategyqa_gpt3.5.py --model dpr --algo ap --task_type adv

python ReAct/eval.py -p ./result/ReAct/dpr-ap-benign.jsonl
python ReAct/eval.py -p ./result/ReAct/dpr-ap-adv.jsonl
```

Output là `result/ReAct/<embedder>-<algo>-<task_type>.jsonl`, mỗi dòng có `answer`,
`gt_answer`, `traj`, `retrieval_success`, `overall_retrieval`.

Script append chứ không ghi đè, và không tự tạo thư mục output — nhớ `mkdir` và
xoá `.jsonl` cũ trước mỗi lần chạy lại.

Backbone LLaMA-3 dùng `ReAct/run_strategyqa_llama3_api.py`, có thêm `--skip N` để
chạy tiếp sau khi bị ngắt. Kết quả tham chiếu để đối chiếu nằm ở [ReAct/ablation/](ReAct/ablation/).

## EhrAgent

```powershell
mkdir result\Ehragent\gpt -Force | Out-Null
python EhrAgent/ehragent/main.py --backbone gpt --model dpr --algo ap --num_questions 50
python EhrAgent/ehragent/main.py --backbone gpt --model dpr --algo ap --num_questions 50 --attack

python EhrAgent/ehragent/eval.py -p ./result/Ehragent/gpt/ap_benign_dpr.json
python EhrAgent/ehragent/eval.py -p ./result/Ehragent/gpt/ap_trigger_dpr.json
```

Tên file output cố định theo `<algo>_<trigger|benign>_<embedder>.json`.
`--num_questions -1` là chạy hết dataset. Dataset bị shuffle theo `--seed` (mặc
định 42), giữ nguyên seed thì hai nhánh mới so sánh được.

`eval.py` của EhrAgent load lại long-term memory và cần GPU (`device = "cuda"`).
ASR-r ở đây tính theo việc knowledge độc (`delete`) xuất hiện ≥3 lần trong retrieval.

## Agent-Driver

Cần dữ liệu nuScenes tiền xử lý dưới `agentdriver/data/`, xem [04_end_to_end.md](04_end_to_end.md).

```powershell
$env:PYTHONPATH = $PWD
bash scripts/agent_driver/run_inference.sh
bash scripts/agent_driver/run_evaluation.sh uniad <result.pkl>
```

`--metric` nhận `uniad` hoặc `stp3`, ground truth đọc từ `--gt_folder` (mặc định
`data/metrics`). Chỉ số là L2 error và collision rate.

## Chỉ đo embedder

Rẻ và nhanh hơn nhiều khi chỉ muốn so retriever:

```powershell
python embedder/eval_embed_contrastive.py
python embedder/eval_embed_classification.py
```

## Bảng cần dựng

Mỗi cặp (agent, embedder) báo 4 dòng: benign không trigger; adv với golden trigger
chưa tối ưu; adv với trigger tối ưu; adv với trigger tối ưu + target gradient
guidance. Kết luận cần rút ra là ASR cao trong khi ACC nhánh benign giữ nguyên.
