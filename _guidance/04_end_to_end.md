# 4. End-to-end — chạy trọn một thí nghiệm

> Đọc [00_setup.md](00_setup.md) trước. Guide này ghép 3 guide còn lại thành
> quy trình đầy đủ cho từng agent, kèm ước lượng thời gian và checkpoint kiểm tra.

Sơ đồ chung:

```
(0) môi trường + .env
        ↓
(1) [tuỳ chọn] train embedder RAG        → checkpoint dưới $EMBEDDER_CKPT_DIR
        ↓
(2) build/cache DB embeddings            → data/memory/*.pkl  (tự động lần chạy đầu)
        ↓
(3) tối ưu trigger  (GPU)                → results/<agent>/<algo>/<ts>/stdout.txt
        ↓
(4) dán trigger vào script inference
        ↓
(5) inference: nhánh benign + nhánh adv  → result/...
        ↓
(6) evaluation: ACC / ASR-r / ASR-a / ASR-t
```

---

## 4.1 ReAct-StrategyQA (khuyến nghị chạy đầu tiên — dữ liệu có sẵn)

| Bước | Lệnh | Thời gian |
|---|---|---|
| 1. Tối ưu trigger | `bash scripts/react_strategyqa/run_optimization.sh` | vài giờ trên 1 GPU (1000 iter); lần đầu +30–60' build cache embedding |
| 2. Lấy trigger | `Select-String -Path ./results/qa/ap/*/stdout.txt -Pattern "Current adv_passage" \| Select-Object -Last 1` | — |
| 3. Dán vào [ReAct/run_strategyqa_gpt3.5.py:115](ReAct/run_strategyqa_gpt3.5.py#L115) | sửa `trigger_token_list` | — |
| 4. Benign | `python ReAct/run_strategyqa_gpt3.5.py --model dpr --algo ap --task_type benign` | 1–2h, tốn API |
| 5. Adv | `python ReAct/run_strategyqa_gpt3.5.py --model dpr --algo ap --task_type adv` | 1–2h, tốn API |
| 6. Eval | `python ReAct/eval.py -p ./result/ReAct/dpr-ap-benign.jsonl` và `... dpr-ap-adv.jsonl` | vài giây |

`mkdir result\ReAct -Force` trước bước 4; xoá `.jsonl` cũ vì file được **append**.

Checkpoint kiểm tra:
- Sau bước 1: `stdout.txt` có nhiều dòng `Iteration:` và trigger thay đổi theo thời gian.
- Sau bước 5: `dpr-ap-adv.jsonl` có số dòng ≈ số câu trong dev split.
- Sau bước 6: ACC benign ≈ baseline không tấn công, ASR-r ở nhánh adv cao rõ rệt.

## 4.2 EhrAgent (eICU)

```powershell
# 1. tối ưu trigger
bash scripts/ehragent/run_optimization.sh          # --agent ehr
# 2. dán trigger vào EhrAgent/ehragent/main.py:91
# 3. inference
mkdir result\Ehragent\gpt -Force | Out-Null
python EhrAgent/ehragent/main.py --backbone gpt --model dpr --algo ap --num_questions -1            # benign
python EhrAgent/ehragent/main.py --backbone gpt --model dpr --algo ap --num_questions -1 --attack   # adv
# 4. eval (cần GPU)
python EhrAgent/ehragent/eval.py -p ./result/Ehragent/gpt/ap_benign_dpr.json
python EhrAgent/ehragent/eval.py -p ./result/Ehragent/gpt/ap_trigger_dpr.json
```

Backbone LLaMA-3 qua Replicate: đổi `--backbone llama3`
(`bash scripts/ehragent/run_inference_llama3.sh`), cần `REPLICATE_API_TOKEN`.

Lưu ý:
- EhrAgent thực thi code sinh ra bởi LLM trong `coding/` (AutoGen `code_execution_config`).
  Chạy trong môi trường cách ly / container khi thí nghiệm với trigger.
- Cache embedding nằm ở `EhrAgent/database/embedding/`.
- Dataset shuffle với `--seed` (mặc định 42) — giữ nguyên seed để hai nhánh so sánh được.

## 4.3 Agent-Driver (nuScenes) — cần chuẩn bị dữ liệu

Repo chỉ có `agentdriver/data/split.json`. Trước khi chạy phải có:

- `agentdriver/data/finetune/data_samples_train.json` (DB memory cho `load_db_ad`)
- dữ liệu nuScenes đã tiền xử lý + `data/metrics` (ground truth cho evaluation)
- `FINETUNE_PLANNER_NAME` trong `.env` — id model GPT-3.5 đã fine-tune
- `NUSCENES_DATAROOT` nếu muốn chạy visualization

```powershell
$env:PYTHONPATH = $PWD
bash scripts/agent_driver/run_finetune.sh                    # fine-tune motion planner (một lần)
bash scripts/agent_driver/run_optimization.sh                # tối ưu trigger, --agent ad
# dán trigger vào script inference tương ứng
bash scripts/agent_driver/run_inference.sh                   # sinh quỹ đạo
bash scripts/agent_driver/run_evaluation.sh uniad <result.pkl>
bash scripts/agent_driver/run_evaluation.sh stp3  <result.pkl>
```

Chỉ số: L2 error + collision rate. Tấn công thành công = collision rate tăng /
quỹ đạo lệch về hành vi mục tiêu, trong khi nhánh benign giữ nguyên.

## 4.4 (Tuỳ chọn) Train embedder RAG riêng

Chỉ cần khi dùng mã embedder `contrastive_user-*` / `classification_user-*`:

```powershell
python embedder/dataset_contrastive_preprocess.py
python embedder/train_contrastive_retriever.py
python embedder/eval_embed_contrastive.py

python embedder/dataset_classification_preprocess.py
python embedder/train_classification_retriever.py
python embedder/eval_embed_classification.py

python embedder/get_ada_v2_embedding.py     # nếu dùng --model ada, cần OPENAI_API_KEY
```

Checkpoint được ghi vào `$EMBEDDER_CKPT_DIR` (mặc định `RAG/embedder`), khớp với
mapping trong [algo/config.py](algo/config.py). Sau khi train xong, truyền mã
embedder tương ứng vào `--model` của bước tối ưu.

## 4.5 Thí nghiệm đầy đủ cho báo cáo

1. Chạy [scripts/run_ablation_sweep.sh](scripts/run_ablation_sweep.sh) với
   `NUM_ITER=1000 NUM_CAND=100` để có trigger cho mọi tổ hợp (agent × embedder × setting).
2. Với mỗi trigger: chạy inference **cả hai nhánh** benign/adv trên cả hai backbone (GPT, LLaMA-3).
3. Chấm điểm và dựng bảng 4 dòng như [02_evaluation.md](02_evaluation.md) §2.6.
4. Ablation nên báo cáo: có/không `--ppl_filter`, có/không `--target_gradient_guidance`,
   `--coh_sample` với các mức `--coh_temperature`, độ dài trigger `-t ∈ {5, 10, 15}`,
   và `--knn ∈ {1,3,5,7,9}` cho ReAct.

## 4.6 Chi phí & lưu ý vận hành

| Hạng mục | Ước lượng |
|---|---|
| Tối ưu trigger, 1000 iter, DPR | ~2–6h / 1 GPU 24GB |
| Build cache embedding lần đầu (20k passage) | 30–60 phút |
| ReAct inference full dev split (GPT-3.5) | 1–2h + chi phí API |
| EhrAgent full dataset (GPT-3.5) | 2–4h + chi phí API |

- Mọi lệnh chạy **từ repo root**.
- Không commit `.env`, `results/`, `result/`, các file `.pkl` cache.
- Các script inference đều dùng `open(..., "a")` hoặc ghi đè theo tên cố định →
  quản lý `--save_dir` theo từng lần chạy để không trộn kết quả.
