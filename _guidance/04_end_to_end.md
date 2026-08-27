# Chạy trọn một thí nghiệm

Thứ tự: tối ưu trigger (GPU) → dán trigger vào script inference → chạy cả hai
nhánh benign và adv → chấm điểm. Bước train embedder riêng chỉ cần khi dùng
checkpoint tự train.

## ReAct / StrategyQA

Nên làm agent này trước vì dữ liệu có sẵn.

```powershell
make opt-qa
make trigger
# dán trigger vào ReAct/run_strategyqa_gpt3.5.py:115
make run-qa-benign
make run-qa-adv
make eval-qa
```

Tối ưu 1000 iter mất vài giờ trên một GPU, lần đầu cộng thêm 30–60 phút build cache
embedding. Mỗi nhánh inference 1–2 giờ và tốn API.

Kiểm tra giữa chừng: `stdout.txt` phải có nhiều dòng `Iteration:` với trigger đổi
dần; `dpr-ap-adv.jsonl` phải có số dòng xấp xỉ số câu trong dev split.

## EhrAgent

```powershell
make opt-ehr
make trigger AGENT=ehr
# dán trigger vào EhrAgent/ehragent/main.py:91
make run-ehr-benign
make run-ehr-adv
make eval-ehr
```

`make ... BACKBONE=llama3` để chạy qua Replicate, cần `REPLICATE_API_TOKEN`.

Lưu ý: EhrAgent thực thi code do LLM sinh ra trong thư mục `coding/` (AutoGen
`code_execution_config`) — chạy thí nghiệm trigger trong môi trường cách ly.

## Agent-Driver

Chưa chạy được ngay. Cần chuẩn bị trước:
`agentdriver/data/finetune/data_samples_train.json` (memory DB cho `load_db_ad`),
dữ liệu nuScenes tiền xử lý, `data/metrics` cho evaluation, và `FINETUNE_PLANNER_NAME`
trong `.env`.

```powershell
make install-ad
$env:PYTHONPATH = $PWD
bash scripts/agent_driver/run_finetune.sh        # một lần
make opt-ad
# dán trigger, rồi:
make run-ad
bash scripts/agent_driver/run_evaluation.sh uniad <result.pkl>
```

Đọc kết quả qua L2 error và collision rate: tấn công thành công là collision rate
tăng ở nhánh adv trong khi benign giữ nguyên.

## Train embedder riêng

Chỉ cần khi `--model` là `contrastive_user-*` hoặc `classification_user-*`:

```powershell
python embedder/dataset_contrastive_preprocess.py
python embedder/train_contrastive_retriever.py
python embedder/dataset_classification_preprocess.py
python embedder/train_classification_retriever.py
```

Checkpoint ghi vào `$EMBEDDER_CKPT_DIR` (mặc định `RAG/embedder`), khớp mapping
trong [algo/config.py](algo/config.py). `embedder/get_ada_v2_embedding.py` dùng khi
chạy `--model ada`.

## Chạy đủ cho báo cáo

Dùng `make sweep NUM_ITER=1000 NUM_CAND=100` để có trigger
cho mọi tổ hợp, sau đó với mỗi trigger chạy inference cả hai nhánh trên cả hai
backbone. Ablation nên báo: có/không `--ppl_filter`, có/không
`--target_gradient_guidance`, các mức `--coh_temperature`, độ dài trigger
`-t ∈ {5,10,15}`, và `--knn ∈ {1,3,5,7,9}` cho ReAct.

Đừng commit `.env`, `results/`, `result/`, và các file cache `.pkl` — `.gitignore`
đã cover sẵn.
