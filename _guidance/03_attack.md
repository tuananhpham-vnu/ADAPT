# Attack

`algo/trigger_optimization.py` tìm chuỗi ~10 token sao cho query có trigger bị đẩy
về một vùng riêng trong không gian embedding của retriever (tối ưu MMD + khoảng
cách tới cluster centroid, giảm variance) để item độc luôn lọt top-k, trong khi
query không trigger giữ nguyên hành vi và bản thân chuỗi vẫn đủ tự nhiên để không
bị PPL detector bắt.

Bật `--golden_trigger` thì khởi tạo từ chuỗi mồi theo agent — `ad`: "Be careful and
drive safe.", `qa`: "Make efficient calls.", `ehr`: "Be careful"
([trigger_optimization.py:428](algo/trigger_optimization.py#L428)). Không bật thì
khởi tạo bằng `[MASK]`.

## Chạy

```powershell
make opt-qa      # hoặc opt-ehr / opt-ad
make trigger     # in trigger cuối của run gần nhất
```

Cấu hình chuẩn của upstream nằm ở `scripts/<agent>/run_optimization.sh`, tương đương:

```powershell
python algo/trigger_optimization.py `
  --agent qa --algo ap --model dpr-ctx_encoder-single-nq-base `
  --save_dir ./results --ppl_filter --target_gradient_guidance `
  --asr_threshold 0.5 --num_adv_passage_tokens 10 --golden_trigger -w -p
```

Kết quả vào `results/<agent>/<algo>/<timestamp>/stdout.txt`. Stdout bị redirect nên
terminal im lặng — `make trigger` đọc file đó và in ra số iteration đã chạy cùng
trigger cuối cùng.

## Các cờ hay dùng

`-n/--num_iter` (1000), `-c/--num_cand` (100), `-g/--num_grad_iter` (30),
`-b/--per_gpu_eval_batch_size` (64), `-t/--num_adv_passage_tokens` (10) là nhóm
điều khiển chi phí/chất lượng.

`--ppl_filter` lọc ứng viên bằng perplexity GPT-2. `--coh_sample` +
`--coh_temperature` lấy mẫu ứng viên theo `softmax(-log ppl / T)` thay vì top-k
tất định (Eq.10 trong paper). `--coh_select_weight` cộng thêm trọng số coherence
lúc chọn. `--exclude_special` cấm chọn `[CLS]/[SEP]/[MASK]/[unused*]`.

`--target_gradient_guidance` dùng loss của target LLM để hướng cập nhật, mặc định
là Llama-2-7b local; thêm `--use_gpt` để dùng GPT-3.5 thay thế. `--asr_threshold`
là ngưỡng đi kèm.

`-w` log lên W&B (`WANDB_PROJECT`, `WANDB_ENTITY`), `-p` vẽ PCA quá trình dịch
chuyển embedding. `--algo badchain` chạy baseline BadChain trên cùng đường ống.

Mã embedder hợp lệ liệt kê trong [algo/config.py](algo/config.py):
`dpr-ctx_encoder-single-nq-base`, `ance-dpr-question-multi`, `bge-large-en`,
`realm-cc-news-pretrained-embedder`, `realm-orqa-nq-openqa`, `ada`, cộng các
checkpoint tự train (cần file dưới `$EMBEDDER_CKPT_DIR`).

## Tiêm trigger

Trigger phải dán tay vào script inference, chỗ có comment
`##### Put your trigger tokens here #####`:

- [ReAct/run_strategyqa_gpt3.5.py:115](ReAct/run_strategyqa_gpt3.5.py#L115) và
  file `..._llama3_api.py` tương ứng
- [EhrAgent/ehragent/main.py:91](EhrAgent/ehragent/main.py#L91)

Cách chèn khác nhau: ReAct nối trigger vào `current_context` ở bước suy luận thứ 2
của vòng ReAct ([run_strategyqa_gpt3.5.py:199](ReAct/run_strategyqa_gpt3.5.py#L199)),
chỉ có tác dụng khi `--task_type adv`. EhrAgent nối vào cuối câu hỏi
([main.py:155](EhrAgent/ehragent/main.py#L155)), chỉ có tác dụng khi có `--attack`.

Chạy agent dưới tấn công:

```powershell
make run-qa-adv
make run-ehr-adv
```

## Sweep

[scripts/run_ablation_sweep.sh](scripts/run_ablation_sweep.sh) chạy lưới
agent × embedder × setting, tự xếp job lên GPU còn trống:

```bash
make sweep NUM_ITER=1000 NUM_CAND=100
```

Sửa mảng `GRID` trong script để chọn tổ hợp. Mặc định `NUM_ITER=30`, `NUM_CAND=50`
cho nhanh; paper-scale là 1000/100. Output vào `results/ppl_ablation/sweep/`, cuối
script in bảng số iteration + số lỗi + trigger cuối. Các dòng `ad_*` chỉ chạy khi
đã có dữ liệu Agent-Driver. Script cần bash và `nvidia-smi`.

## Chỉnh khi kết quả xấu

ASR-r thấp thì tăng `--num_iter`, `--num_cand`, hoặc trigger dài hơn (`-t`).
Retrieval trúng nhưng agent không hành động sai (ASR-a thấp) thì bật
`--target_gradient_guidance`. Trigger nhìn như rác, dễ bị lọc thì bật `--ppl_filter`
kèm `--coh_sample --coh_temperature 0.5`. Có token `[unused0]` thì bật
`--exclude_special`. ACC nhánh benign tụt là trigger quá xâm lấn, giảm `-t`.
OOM thì hạ `BATCH` và `NUM_CAND`.
