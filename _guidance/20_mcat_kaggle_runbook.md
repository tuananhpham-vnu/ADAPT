# 20 — Chạy MCAT trên Kaggle (M0–M2)

> Cập nhật: 2026-09-20. Đi kèm `src/mcat/README.md` và `_idea/memory_conditioned_generator_Q1_A_star.md`.
> File này chỉ nói về cách chạy. Ý nghĩa nghiên cứu và tiêu chí go/no-go nằm ở `_idea/`.

## 0. Đường ngắn nhất: `scripts/run_mcat_kaggle.sh`

Toàn bộ runbook này đã được đóng gói thành script. Chạy từ **gốc repo**:

```bash
bash scripts/run_mcat_kaggle.sh preflight   # kiểm tra môi trường + test + smoke CPU, không đụng GPU
bash scripts/run_mcat_kaggle.sh             # preflight + arm chính (m1)
bash scripts/run_mcat_kaggle.sh all         # cả 7 arm: m1 b2 b3 b4 b5 b6 margin
bash scripts/run_mcat_kaggle.sh m1 b4 b5 b6 # chọn arm
RESUME=1 bash scripts/run_mcat_kaggle.sh all   # tiếp tục sau khi Kaggle cắt 12h
```

Script tự lo ba thứ dễ sai khi gõ tay:

- **Cờ `train` và `evaluate` khớp nhau.** `evaluate` dựng lại `TrainConfig` từ CLI rồi so
  hash với checkpoint; lệch một cờ là bị từ chối. Script dựng `$TRAIN_FLAGS` một lần rồi
  truyền cho cả hai.
- **Cache dùng chung.** Mọi arm trỏ vào một `--cache-dir`, nên corpus chỉ encode một lần
  chứ không phải 7 lần.
- **Preflight chặn trước.** Thiếu scikit-learn, test đỏ, hoặc smoke không kích hoạt được
  resume guard → dừng, chưa đụng tới GPU.

Biến môi trường hay dùng: `STEPS`, `SEED`, `DOMAINS`, `DEVICE`, `RUN_ROOT`, `CACHE_DIR`,
`PER_SPLIT`, `DOCUMENTS`, `OPTIMIZATION`, `CORPUS_LIMIT`, `EVAL_SPLIT`, `PYTHON`.

```bash
STEPS=50 DOMAINS="qa" bash scripts/run_mcat_kaggle.sh m1     # shakedown nhanh
SEED=1 RUN_ROOT=outputs/mcat/pilot bash scripts/run_mcat_kaggle.sh all   # seed thứ hai
```

`FIXTURE=1` chạy khô mọi arm bằng encoder fixture trên CPU — không tải model, không cần
GPU, không cần scikit-learn. Dùng để kiểm tra đường ống; **số liệu của nó không phải kết
quả**, vì reference centers khi đó là `memory[:5]` chứ không phải GMM.

Cuối run script in một bảng tóm tắt các arm (hit@K, occupancy, margin, false activation,
tỉ lệ round-trip hợp lệ, và control shuffled-context).

Phần dưới là các lệnh thủ công tương đương, để hiểu script đang làm gì và để debug từng stage.

## 1. Khác biệt so với `_guidance/19` (AgentPoison margin)

| | AgentPoison margin | MCAT M0–M2 |
|---|---|---|
| GPU | **Bắt buộc 2** (DPR+GPT-2 `cuda:0`, Llama `cuda:1`) | **1 là đủ** |
| Model tải về | DPR + GPT-2 + Llama-3-8B (gated) | **chỉ DPR** |
| HF token | cần cho Llama | không cần |
| Coherence / target scorer | có, làm sampler + gate | **chưa có ở M0–M2** |

M0–M2 chỉ tối ưu hình học retrieval. `L_coh` và `L_tar` là việc của M4 (behavioral ASR).
Đừng khai báo 2 GPU rồi để một cái ngồi không — chọn T4×1 hoặc P100 cho rẻ.

## 2. Chuẩn bị môi trường

Kaggle đã có `torch`, `transformers`, `numpy`, `scikit-learn`. Chỉ cần kiểm tra, không cần cài thêm.

```python
!python -c "import torch, transformers, sklearn, numpy; \
print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), \
'| gpus', torch.cuda.device_count()); \
print('transformers', transformers.__version__, '| sklearn', sklearn.__version__)"
```

**scikit-learn là bắt buộc** cho run thật: `algo.clustering.fit_centers` (GMM 5 component,
`covariance_type='full'`, `random_state=0`) là nguồn benign reference centers. Không có nó
thì stage `train` sẽ dừng ngay — đúng như thiết kế, vì bản fixture dùng `memory[:5]` và
**không phải** hình học AgentPoison.

Tải trước DPR rồi bật offline để job dài không phụ thuộc mạng (theo mẹo ở `_guidance/18`):

```python
from transformers import AutoTokenizer, DPRContextEncoder
REV = "bb21a3c2b1656d60c6a8e920283bc40dabddadb8"
AutoTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base", revision=REV)
DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base", revision=REV)
import os; os.environ["HF_HUB_OFFLINE"] = "1"
```

Revision `bb21a3c2...` lấy từ `ReAct/database/embeddings/agentpoison_dpr/manifest.json`,
tức đúng checkpoint đã dựng index của repo. Luôn pin nó để số liệu so sánh được.

## 3. Smoke CPU trước (bắt buộc, ~30 giây)

Không bao giờ đốt GPU trước khi contract artifact/resume đã xanh:

```bash
cd /kaggle/working/<repo>
python -m unittest tests.test_mcat tests.test_mcat_pipeline
python -m src.mcat smoke --output-dir outputs/mcat/smoke --fixture \
  --domain qa --corpus-limit 400 --per-split 2 \
  --documents 24 --support 4 --optimization 6 --evaluation 8 \
  --poison-count 2 --trigger-tokens 4 --top-k 3 --steps 3 --max-length 32
```

Phải thấy dòng `resume guard: a changed contract is refused` ở cuối, và toàn bộ test xanh.

## 4. Pilot thật — arm chính (M1)

```bash
export RUN=outputs/mcat/pilot/seed_0
export REV=bb21a3c2b1656d60c6a8e920283bc40dabddadb8
export COMMON="--output-dir $RUN --domain qa --domain ehr --seed 0 \
  --retriever-device cuda:0 --retriever-revision $REV --max-length 256 \
  --per-split 4 --documents 512 --support 32 --optimization 64 \
  --evaluation 128 --poison-count 5 --trigger-tokens 10 --top-k 5"

python -m src.mcat prepare-episodes $COMMON
python -m src.mcat index            $COMMON --index-batch-size 64
python -m src.mcat train            $COMMON --mode generator --variant memory+query \
    --steps 400 --learning-rate 1e-3 --tau-start 2.0 --tau-end 0.5 --lambda-ret 0.0
python -m src.mcat evaluate         $COMMON --mode generator --variant memory+query \
    --steps 400 --learning-rate 1e-3 --tau-start 2.0 --tau-end 0.5 --lambda-ret 0.0 \
    --split test
python -m src.mcat report           $COMMON
```

Lưu ý về cờ: `evaluate` dựng lại `TrainConfig` từ CLI rồi **so contract với checkpoint**.
Mọi cờ training phải lặp lại y hệt ở `evaluate`, nếu không sẽ bị từ chối với
`evaluation refused: the checkpoint was trained under another contract`. Đó là chủ ý.

`--max-length 256` thay vì 512: query StrategyQA/eICU đều ngắn, và
`encode_with_trigger_embeddings` pad theo câu dài nhất trong batch chứ không pad tới
`max_length`, nên 256 chỉ là trần an toàn và tiết kiệm bộ nhớ.

### Chi phí ước tính

Mỗi step encode **toàn bộ** train episode (không minibatch theo episode):
8 episode × (64 optimization + 5 poison) ≈ 550 chuỗi ngắn qua DPR, cả forward lẫn backward.
Trên T4 vào khoảng 1–2 giây/step, tức 400 step ≈ 10–15 phút. Stage `index` thêm 1–2 phút.
Nếu bị bó thời gian, giảm `--optimization` trước rồi mới giảm `--steps` — compactness cần
nhiều query trong cùng episode mới có nghĩa (batch một query thì `L_cpt` bằng 0).

### Điều sẽ thấy ở stage `index`

```
  qa: prebuilt index re-encoded (prebuilt index is normalized)
```

Đúng, không phải lỗi. `ReAct/database/embeddings/agentpoison_dpr` được dựng với
`normalization: "l2"`, trong khi MCAT xếp hạng bằng dot product thô và fit GMM trên vector
thô. Tái dùng index đã normalize sẽ đổi **hình học**, không chỉ đổi scorer. Việc re-encode
9 251 đoạn trên GPU mất khoảng 1–2 phút, cache lại vào `$RUN/cache/` và resume được.

(Index đó còn lưu `paragraph_ids` theo thứ tự khác thứ tự domain phát ra. Code đã hoán vị
hàng về đúng thứ tự trước khi dùng; đây là bug đã bị bắt và có test
`test_rows_are_permuted_into_the_requested_order`.)

## 5. Các arm baseline

Chạy cùng `$COMMON`, cùng seed, cùng số step. Mỗi arm một `--output-dir` riêng.

```bash
# B3 — một trigger universal cho mọi episode
python -m src.mcat train $COMMON_B3 --mode universal-logit --steps 400
# B4 — generator vô điều kiện, cùng số tham số
python -m src.mcat train $COMMON_B4 --mode generator --variant none   --steps 400
# B5 — chỉ query
python -m src.mcat train $COMMON_B5 --mode generator --variant query  --steps 400
# B6 — chỉ memory
python -m src.mcat train $COMMON_B6 --mode generator --variant memory --steps 400
# B2 — logits trực tiếp, tối ưu lại trên từng episode
python -m src.mcat train $COMMON_B2 --mode direct-logit --steps 400
```

B2 không transfer được. Ở `evaluate`, nó tự tối ưu lại trên split đích và ghi vào
`$RUN/adapt-test/`. Chi phí online đó phải được tính khi so sánh — đó là toàn bộ ý nghĩa
của trục amortization ở `_idea` §11.2.

Bốn variant generator có **số tham số bằng nhau** (nhánh bị bỏ đọc pseudo-set hằng số),
nên chênh lệch giữa chúng là chênh lệch thông tin, không phải capacity.

## 6. Ablate margin

`--lambda-ret 0` là mặc định: đó là objective AgentPoison thuần
(`L_uni + 0.1·L_cpt`), dùng để xác nhận tương thích trước. Sau khi arm đó chạy xong mới bật:

```bash
python -m src.mcat train $COMMON_MARGIN --mode generator --variant memory+query \
    --steps 400 --lambda-ret 1.0 --margin 0.1
```

`compute_hit_at_k_margin_loss` là sự kiện *ít nhất một poison vào top-K*. **Không** phải
`algo.trigger_losses.compute_retrieval_margin_loss` (chiếm trọn top-K). Đừng báo cáo lẫn tên.

## 7. Đọc kết quả

`$RUN/REPORT.md` có bảng trigger-on / trigger-off và hai control. Ba con số quyết định:

1. **`shuffled memory context (within domain) changed the trigger in X/N`** — nếu tỉ lệ
   này thấp, generator đang bỏ qua memory và claim conditioning **không** được chứng minh.
   Đây là kết quả âm hợp lệ, không phải lỗi cần sửa. Báo cáo tách theo domain; việc tráo
   chỉ diễn ra trong cùng domain, vì đưa snapshot EhrAgent cho một episode StrategyQA là
   một distribution shift mà generator nhận ra được **mà không** cần học gì về memory state,
   sẽ thổi phồng control. Domain có dưới 2 episode bị bỏ qua và được liệt kê.
2. **`round-trip valid triggers`** trong `training.json` — phải gần 100%. Thấp nghĩa là
   trigger vỡ khi decode, mọi gain trên loss đều vô nghĩa.
3. **`false_activation`** — hit@K khi memory đã nhiễm nhưng query **không** có trigger.
   Cao nghĩa là poison được lấy ra kể cả không kích hoạt, tức trigger không có tác dụng riêng.

Cảnh báo cấu hình: `false_activation` chỉ có nghĩa khi snapshot lớn hơn hẳn K. Với
`--documents 24 --top-k 5`, chỉ có 24 clean key nên 5 poison lọt top-5 một cách tầm thường
và `false_activation` sẽ ra 1.0. Giữ `--documents 512 --top-k 5` cho run thật.

So chéo với baseline cũ: `l_uni`/`l_cpt` của arm `--lambda-ret 0` phải cùng bậc với
`metrics.jsonl` trong `outputs/agentpoison_margin/` trên cùng corpus. Tham chiếu đã đo trên
CPU với DPR thật, 24 doc: `l_uni ≈ -12.5`, `l_cpt ≈ 4.5`. Lệch xa nghĩa là đường encode sai.

## 8. Cắt job và resume

Mọi stage đều atomic và hash-guarded. Khi Kaggle hết 12 giờ:

```bash
python -m src.mcat train $COMMON --mode generator --variant memory+query \
    --steps 400 ... --resume
```

`--resume` chỉ tiếp tục khi `config_hash` + `split_hash` + retriever fingerprint không đổi.
Đổi bất kỳ cờ nào → bị từ chối thay vì trộn hai run. Đổi `--per-split` → stage
`prepare-episodes` báo `resume refused: the episode split changed`.

Nên `!cp -r outputs/mcat /kaggle/working/` cuối mỗi session để artifact sống sót.

## 9. Chưa làm được

- **`--domain ad` sẽ raise.** `agentdriver/data/finetune/data_samples_train.json` không có
  trong repo (`agentdriver/data/` chỉ có `split.json`). Chưa được mô tả kết quả nào là
  "3 agent domain".
- **`ehr` rất mỏng**: 193 doc, 199 query. Với `--per-split 4`, split của nó chỉ còn
  121/21/51 doc và 116/42/41 query, nên episode bị co xuống (train: support 16,
  optimization 32, evaluation 64, poison 2). Mức co ghi ở
  `manifest.json → domains.ehr.scaling` — đọc nó trước khi trích số. Vì snapshot `ehr`
  chỉ 21–121 clean key so với K=5, `false_activation` trên domain này gần như vô nghĩa;
  đọc chỉ số đó theo từng domain trong `evaluation.jsonl`, đừng đọc con số trung bình.
- **B0 và B1 chưa nối vào CLI.** B1 (HotFlip per-episode) vẫn phải chạy qua
  `algo/agentpoison_margin.py` riêng, và phải pin commit nếu có patch.
- **Chưa có behavioral ASR.** Retrieval hit không đồng nghĩa agent đổi hành vi. Đó là M4.
