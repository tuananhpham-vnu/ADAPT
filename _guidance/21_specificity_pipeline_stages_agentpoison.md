# Pipeline specificity trên StrategyQA: ý nghĩa từng stage

Tài liệu đi kèm [`scripts/run_specificity_ap.sh`](../scripts/run_specificity_ap.sh).
Mục tiêu của lượt chạy này: **so sánh ba mức chuyên biệt hóa trigger — một trigger
cho toàn tập, một trigger cho mỗi nhóm, một trigger cho mỗi câu — về mặt ngôn ngữ**.

Lượt chạy này **không đo ASR**. Số retrieval xuất hiện ở stage `bank` là sản phẩm phụ
của quá trình tối ưu trigger, không phải kết quả tấn công downstream.

## 0. Chạy trên Kaggle

```bash
# Notebook settings: Accelerator = GPU, Internet = ON (cần cho stage models)
!git clone <repo> /kaggle/working/adapt && cd /kaggle/working/adapt
!pip -q install "transformers==4.39.1"       # torch/numpy/sklearn đã có sẵn
!bash scripts/run_specificity_ap.sh all
```

Chạy lại từng phần khi hết 12h:

```bash
!bash scripts/run_specificity_ap.sh bank language summary
```

Biến môi trường hay dùng: `TRAIN_SIZE`, `VALIDATION_SIZE`, `TEST_SIZE`,
`FRESH_TEST_SIZE`, `GROUP_COUNT`, `BUDGET`, `SEED`, `DEVICE`, `RUN_ROOT`.

Lưu ý: **không đặt tên biến là `GROUPS`**; đó là mảng readonly có sẵn của bash và
sẽ không truyền được sang Python. Script dùng `GROUP_COUNT`.

## 0.1 Log và checkpoint khi bị ngắt

Khi session bị kill hoặc hết 12h, **chạy lại đúng lệnh cũ**. Có ba lớp phục hồi:

| Lớp | Cơ chế | Độ mịn |
|---|---|---|
| Stage | Marker `$RUN_ROOT/state/<stage>.done`, chỉ ghi khi stage **exit 0** | cả stage |
| `bank` | `compare()` lưu `prefix/<arm>.json`; arm đã xong thì đọc lại | mỗi arm |
| `language` | `fit_cache.json`, mỗi `Selector.fit` lưu ngay sau khi tính xong | mỗi query |

Log của mọi stage nằm ở `$RUN_ROOT/logs/<stage>.log`, ghi nối tiếp qua các lần chạy,
mỗi lần có header ghi thời gian UTC và toàn bộ tham số. Stage thất bại được ghi
`FAILED exit N` và **không** được đánh dấu done.

```bash
bash scripts/run_specificity_ap.sh all              # bỏ qua stage đã xong
FORCE=bank bash scripts/run_specificity_ap.sh bank  # chạy lại riêng một stage
FORCE=1 bash scripts/run_specificity_ap.sh all      # chạy lại tất cả
```

`preflight` và `summary` luôn chạy lại vì rẻ.

Cache của `language` nằm trong chính thư mục output, mà output đã bị contract check
ghim vào một cấu hình, nên không thể trộn kết quả của hai cấu hình khác nhau. Đã kiểm
chứng bằng cách cắt cache còn 40/72 fit rồi chạy lại: **0 khác biệt phi-float**, mọi
trigger và lựa chọn giống hệt. Sai khác còn lại chỉ ở mức 1e-5 tương đối và xuất hiện
cả ở nhánh `legacy_raw` vốn không dùng cache, tức là nhiễu float có sẵn khi encode
theo batch, không phải do checkpoint.

## 0.2 Ba nguồn dữ liệu và vì sao chia như vậy

StrategyQA có ba file câu hỏi rời nhau. Đã kiểm chứng: **không câu nào trùng nhau
giữa ba file** (so theo nội dung câu hỏi đã chuẩn hóa).

| File | Số câu unique | Có nhãn | Dùng làm |
|---|---:|---|---|
| `strategyqa_train_filtered.json` | 2.803 | có | train 24 + poison carrier 120 |
| `strategyqa_test.json` | 489 | **không** | validation, dùng hết |
| `strategyqa_dev.json` | 229 | có | test: 80 cho `bank` + 149 cho `language` |

`strategyqa_test.json` không có trường `answer` vì StrategyQA là benchmark có
leaderboard: nhãn của test set gốc bị giữ kín, người dùng nộp dự đoán lên hệ thống
chấm. File chỉ còn `qid` và `question`.

Nguyên tắc phân bổ: **dữ liệu không nhãn đi vào chỗ không bao giờ cần nhãn.**
Validation ở pipeline này chỉ để chốt variant, chấm bằng relevance cosine, PPL ratio
và preservation proxy — không chỉ số nào đọc `answer`. Ngược lại test phải giữ nhãn,
vì bước tiếp theo của lộ trình là đo downstream ASR và clean accuracy, và phải đo
trên **đúng tập test này** thì mới so sánh được với lượt hiện tại.

Hệ quả cần biết trước: nếu sau này muốn tune ngưỡng dựa trên clean accuracy thì
validation cũng sẽ cần nhãn. Khi đó đổi một biến là xong:

```bash
VALIDATION_FILE="" bash scripts/run_specificity_ap.sh all   # validation quay lại lấy từ train pool
```

Vì sao `TRAIN_SIZE` không tăng theo: nhánh `per_query` sinh **một trigger cho mỗi câu
train**, và `POISON_COUNT = TOP_K × TRAIN_SIZE`. Với 24 câu đã là 120 key độc
(1,30% corpus). Lấy cả 2.803 câu sẽ cần 14.015 key độc, nhiều hơn cả corpus 9.251 đoạn.
Đây là tập để **fit attack**, không phải tập huấn luyện model.

## 1. `preflight` — chặn lỗi trước khi tốn GPU

Kiểm tra bốn thứ thường gãy trên session mới: thiếu package (đặc biệt
**scikit-learn**, bắt buộc vì `src/triggers/clustering.py` fit GMM để lấy benign
reference centers), thiếu file dữ liệu, không có GPU, chạy sai thư mục gốc.

Ngoài ra kiểm tra trước ba ràng buộc mà CLI sẽ từ chối ở giữa chừng:

| Ràng buộc | Lý do |
|---|---|
| `TRAIN_SIZE + POISON_COUNT` ≤ 2.803 | train pool `strategyqa_train_filtered.json` |
| `VALIDATION_SIZE` ≤ 489 | validation pool `strategyqa_test.json` |
| `TEST_SIZE + FRESH_TEST_SIZE` ≤ 229 | cả hai lấy từ `strategyqa_dev.json`, **không được trùng nhau** |
| test pool phải **có nhãn** | để sau này còn đo được downstream trên đúng tập test này |
| `POISON_COUNT ≥ TRAIN_SIZE` | nhánh per_query cần một poison source cho mỗi câu train |

Stage này không đụng GPU và không cần mạng.

## 2. `models` — tải model vào đúng hai cache

Mọi loader trong `src/triggers` chạy với `local_files_only=True`, nên **không có
gì tự tải về ở các stage sau**. Có hai cache và chúng không thay thế cho nhau:

| Cache | Model | Vai trò |
|---|---|---|
| HF cache mặc định | `facebook/dpr-ctx_encoder-single-nq-base` | retriever bị tấn công |
| HF cache mặc định | `all-MiniLM-L6-v2` | chấm độ liên quan khi **lựa chọn**, và meaning guard |
| `outputs/model-cache` | `distilgpt2` | chấm độ trôi chảy khi **lựa chọn** |
| `outputs/model-cache` | `paraphrase-MiniLM-L3-v2` | **audit** độ liên quan, weights khác |
| `outputs/model-cache` | `gpt2` | **audit** độ trôi chảy, weights khác |
| `outputs/model-cache` | `bert-english-uncased-finetuned-pos` | lọc cụm danh từ |

Script tải mỗi tên vào **cả hai** cache. Tốn vài trăm MB nhưng loại bỏ hoàn toàn
lớp lỗi "chạy được ở máy mình, `local_files_only` fail trên Kaggle".

Tách model lựa chọn và model audit là có chủ ý: điểm của hai model audit **không**
tham gia chọn ứng viên hay chốt variant. Tuy vậy chúng cùng họ với model lựa chọn,
nên đây không phải đánh giá độc lập hoàn toàn, càng không phải human review.

## 3. `index` — encode corpus một lần

Encode 9.251 đoạn StrategyQA bằng DPR, L2-normalize, ghi `vectors.npy` kèm
`manifest.json` ghim ba thứ: hash corpus, tên encoder, `max_length`.

Stage `bank` sẽ **từ chối chạy** nếu một trong ba thứ đó lệch. Đây là cơ chế ngăn
cache cũ âm thầm làm đổi số retrieval. Nếu index đã khớp thì stage này dùng lại,
không encode lại.

Encoder ở đây bị ghim cứng trong `src/agentpoison/strategyqa.py` (hằng `ENCODER`).
Đổi `DPR_MODEL` mà không sửa hằng đó sẽ làm stage `bank` báo lệch cache.

## 4. `bank` — tối ưu trigger ở ba mức chuyên biệt

Đây là stage sinh ra các trigger. Ba nhánh, đều ở vị trí **prefix**, đều dưới cùng
một `--budget` tính bằng số logical retriever request:

| Nhánh | Số trigger | Cách định tuyến câu chưa thấy |
|---|---:|---|
| `universal` | 1 | không cần, mọi câu dùng chung |
| `semantic` | `GROUP_COUNT` | nearest clean center; nhóm fit bằng GMM trên embedding DPR của câu train |
| `per_query` | `TRAIN_SIZE` | nearest neighbor trong câu train |

Chỉ chạy prefix vì stage `language` chỉ đọc `$BANK_DIR/prefix/*.json`. Trả tiền cho
suffix và infix ở đây là lãng phí.

Ngân sách được chia đều cho các nhóm trong cùng một nhánh. Nghĩa là `per_query`
chia `BUDGET` cho `TRAIN_SIZE` trigger, mỗi trigger được rất ít request. Đây là
thiết kế "cùng tổng ngân sách", không phải "cùng ngân sách mỗi trigger". Khi đọc
kết quả phải nói rõ điều này.

**Điểm yếu đã biết:** `fit_centers` dùng GMM trên embedding câu train, thường cho
cụm rất mất cân bằng (lượt 12 câu/3 nhóm trước đây ra 1–7–4). UniC-RAG chỉ ra rằng
cụm mất cân bằng làm hỏng hiệu quả vì trigger của cụm lớn phải phủ quá nhiều câu.
Đây là ứng viên sửa đầu tiên nếu nhánh `semantic` tiếp tục không hơn `universal`.

Output: `REPORT.md`, `dataset.json`, `prefix/{universal,semantic,per_query}.json`,
`reference_clusters.json`.

## 5. `language` — phần đánh giá chính

Lấy `FRESH_TEST_SIZE` câu dev **không xuất hiện trong bất kỳ split nào** của stage
`bank`, định tuyến chúng bằng center đã đóng băng, rồi chấm sáu variant cho mỗi
scope cộng nhánh query-adaptive:

| Variant | Thay đổi so với gốc |
|---|---|
| `legacy_raw` | trigger từ stage `bank`, renderer cũ |
| `legacy_heading` | cùng trigger, chỉ sửa viết hoa và dấu hai chấm |
| `generic_language` | chọn lại seed chung theo objective ngôn ngữ |
| `grounded_language` | ứng viên lấy từ nội dung câu train, bank cố định khi test |
| `topic_heading` | chỉ cụm chủ đề, bỏ About/Regarding — **đối chứng lặp từ** |
| `per_query/query_adaptive` | chọn prefix cho từng câu đang đến, lúc inference |

Thứ tự bắt buộc: **chốt variant trên toàn bộ 489 câu validation trước, rồi mới chấm fresh
test**. Điều này được cài trong `__main__.py`, không phải kỷ luật thủ công.

Ba metric chính:

- **Relevance** — cosine giữa trigger và query gốc. Cao hơn là gần nhau hơn trong
  không gian embedding, **không phải** xác suất "đúng chủ đề".
- **PPL ratio** — trung bình nhân PPL toàn câu sau/trước khi chèn, bằng GPT-2.
  `1.000` là mức câu gốc; thấp hơn là trôi chảy hơn theo model, không phải theo người.
- **Preservation proxy** — similarity ≥ 0,85 và giữ nguyên số/phủ định. Đây là
  proxy, không phải entailment và không chứng nhận đáp án không đổi.

## 6. `summary` — bảng so sánh và các cảnh báo

In bảng của split fresh test theo điểm audit, kèm lựa chọn đã chốt trên validation
và dòng retrieval từ stage `bank`.

Bốn điều phải đọc trước khi trích dẫn bất kỳ con số nào:

1. `query_adaptive` đạt điểm relevance cao **chủ yếu nhờ lặp lại một từ chủ đề của
   chính câu hỏi**. Luôn so nó với `topic_heading` trước khi gọi đó là cải tiến.
2. `query_adaptive` tốn thêm chi phí tìm kiếm ở inference; bank cố định thì không.
   Không so hai thứ này như cùng một ngân sách inference.
3. Preservation là proxy. Chưa có entailment hai chiều, chưa có human review.
4. Không có gì ở đây là ASR. Một trigger trôi chảy mà không bao giờ được retrieve
   thì vẫn vô dụng.

## 7. Sau lượt này thì làm gì

Theo thứ tự ưu tiên:

1. Thay GMM bằng clustering cân bằng theo similarity (Algorithm 1 của UniC-RAG) cho
   nhánh `semantic`, rồi chạy lại đúng cấu hình này để so trực tiếp.
2. ~~Sửa false activation ở stage `bank`.~~ **Đã thêm công cụ đo, chưa chạy lại.**
   `evaluate_bank` nay báo thêm `false_activation_baseline` (mã hóa đúng các source đó
   nhưng **không** chèn trigger) và `false_activation_attributable` (hiệu của hai cột).
   Nguyên nhân cấu trúc: poison key là *câu hỏi + trigger* còn `objective.clean` là
   *đoạn Wikipedia*, nên câu sạch có thể vượt top-k của chính nó chỉ vì key có dạng câu
   hỏi. Con số 8–9/16 của lượt DPR trước được đo trước khi có baseline, nên chưa tách
   được hai nguyên nhân; phải chạy lại stage `bank` mới diễn giải được.
3. Thêm trục detector: đo tỷ lệ trigger lọt qua bộ lọc perplexity. Đây là trục làm
   cho claim "tồn tại mức chuyên biệt tối ưu" có chỗ đứng, vì trên trục ASR × chi phí
   thì UniC-RAG đã cho thấy càng nhiều nhóm càng tốt.

Related work phải đối chiếu: UniC-RAG (arXiv 2508.18652), BadRAG (2406.00083),
LOTUS (2403.17188). Xem [ghi chú ý tưởng](../_idea/group_conditioned_triggers.md).
