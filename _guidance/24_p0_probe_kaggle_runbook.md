# 24 — Chạy P0/P1 probe trên Kaggle

> Cập nhật: 2026-10-01. Đi kèm `_idea_q1_aplus/open_problems.md` (ý nghĩa nghiên cứu)
> và `scripts/run_p0_probes.sh` (code). File này chỉ nói **cách chạy** và **cách đọc số**.

## 0. Vì sao chạy cái này trước

Ba probe trả lời ba câu hỏi, và hai câu đầu có thể **kết thúc đề tài**:

| | Câu hỏi | Nếu trả lời sai thì |
|---|---|---|
| **R2** | Memory phình thì trigger cũ có suy giảm không? | Không suy giảm → `reuse` thắng mọi method → không còn gì để amortize |
| **R1** | Một trigger phổ quát đã đủ chưa? | Đã đủ → conditioning trên memory mất lý do tồn tại |
| **P1** | Trigger nên nằm ở đâu trong query? | Đây là **phát hiện**, không phải cổng. Chỉ có nghĩa nếu R1 và R2 đều trả lời đúng |

Chi phí cả ba cộng lại nhỏ hơn một lần train generator. Chạy xong mới quyết định
có đầu tư tiếp hay không.

## 1. Một lệnh

```bash
bash scripts/run_p0_probes.sh preflight   # kiểm tra môi trường + test + smoke CPU, không đụng GPU
bash scripts/run_p0_probes.sh r2          # cổng rẻ nhất, chạy đầu tiên
bash scripts/run_p0_probes.sh r2 r1       # cả hai cổng P0
bash scripts/run_p0_probes.sh all         # r2 + r1 + position
RESUME=1 bash scripts/run_p0_probes.sh all   # chạy tiếp sau khi Kaggle cắt 12h
```

`FIXTURE=1` chạy khô toàn bộ bằng encoder fixture trên CPU — không tải model, không
cần GPU. **Số của nó không phải kết quả**: reference centers khi đó là `memory[:5]`
chứ không phải GMM, và encoder là BERT khởi tạo ngẫu nhiên.

Biến hay dùng: `STEPS`, `SEED`, `DOMAINS`, `DEVICE`, `RUN_ROOT`, `CACHE_DIR`,
`PROBE_SPLIT`, `GROWTH`, `DRIFT_SEEDS`, `MIN_BASE_HIT`, `POSITIONS`, `POSITION_MODE`,
`BOOTSTRAP`, `CORPUS_LIMIT`, `PYTHON`.

## 2. Môi trường Kaggle

Giống `_guidance/20`: **một GPU là đủ** (T4×1 hoặc P100). Không cần Llama, không cần
HF token. `scikit-learn` bắt buộc cho run thật.

```python
!python -c "import torch, transformers, sklearn; \
print('torch', torch.__version__, '| cuda', torch.cuda.is_available()); \
print('transformers', transformers.__version__)"
```

`transformers>=5.3` là bắt buộc: bản 5.0–5.2 nạp `facebook/dpr-*` mà không lowercase,
mọi từ viết hoa thành `[UNK]`, và hình học sai hoàn toàn. `load_dpr` tự kiểm tra và
dừng ngay — xem `src/triggers/mcat/retrievers.py`.

Cell Kaggle điển hình:

```python
%cd /kaggle/working/<repo>
!bash scripts/run_p0_probes.sh preflight
!bash scripts/run_p0_probes.sh r2
```

## 3. Thí nghiệm R2 làm chính xác những gì

Đây là protocol trong `open_problems.md` §P0-R2, đã được ép trong code:

1. Lấy trigger của `s0` (với `--mode direct-logit` thì đó là một lần tìm kiếm
   per-episode thật sự, tính tiền như mọi method khác).
2. **Đóng băng poison ngay tại `s0`** qua `freeze_poison`. Đây là chỗ giữ cho **số
   lượng** poison cố định; memory phình lên thì **tỷ lệ** poison giảm — đúng thứ đang
   cần đo. Mỗi dòng mang theo `poison_records` và `poison_fraction` để kiểm tra lại
   từ artifact chứ không phải tin lời.
3. Phình memory 25/50/100% bằng tài liệu lành **cùng domain, cùng split**
   (`_check_pool` chặn lấy tài liệu split khác).
4. Lặp 5 seed (`DRIFT_SEEDS="0 1 2 3 4"`), mỗi seed bốc một tập tài liệu khác.
5. Đo `trigger_on.hit@K` (trigger cũ còn chạy không) và `trigger_off.hit@K` (poison có
   bị kích hoạt khi **không** có trigger không).

Giữ nguyên: retriever, top-k, `Q_eval`, trigger, bản ghi poison. Biến duy nhất là
lượng dữ liệu lành.

### Đọc `probe_growth.json`

```
verdict: decays | stable | inconclusive | no-data
```

`decays` chỉ được trả về khi **cả ba** điều sau đúng:

1. khoảng tin cậy ghép cặp của mức drop ở level lớn nhất **không chứa 0**;
2. `on_hit` **đơn điệu không tăng** theo mức phình;
3. hiệu ứng **lớn hơn `seed_spread`** — độ lệch chuẩn giữa các drift seed.

Điều kiện (3) là thứ chặn "một lần chia dữ liệu may mắn" thành phát hiện. Thiếu nó
thì verdict là `inconclusive` kèm lý do nêu rõ con số nào nhỏ hơn con số nào.

| verdict | Nghĩa | Làm gì tiếp |
|---|---|---|
| `stable` | **Chưa có bằng chứng** cần sinh trigger mới, trong thiết lập này | R2 có thể ĐÚNG. Dừng trước khi train generator. Chỉ áp cho growth cùng miền — chưa nói gì về `mixture`/`distractor` |
| `inconclusive` | Thường là drop nhỏ hơn spread giữa seed | Thêm seed hoặc thêm episode. **Không** báo cáo là suy giảm |
| `decays` | R2 bị bác bỏ | Câu hỏi kế: tối ưu lại trên memory mới có phục hồi không (`adapt` với `scratch`) |
| `no-data` | Không có dòng base nào | Kiểm tra `--mode` và checkpoint |

Nếu `stable` mà muốn chắc chắn hơn trước khi kết luận, thử theo thứ tự rẻ dần:
tăng `DRIFT_SEEDS`, tăng `PER_SPLIT`, rồi mới chuyển sang `mixture`/`distractor`
bằng stage `prepare-drift` + `adapt` của M3.

## 4. R1: trigger phổ quát đã đủ chưa

`probe-universal` ghép cặp **theo episode** giữa hai run đã `evaluate`:

- treatment: `b2` (`direct-logit`, per-episode)
- control: `b3` (`universal-logit`, một ma trận logits cho mọi episode)

Hàm từ chối thay vì đoán khi hai run chấm trên tập episode khác nhau hoặc ở hai vị
trí trigger khác nhau — ghép cặp hai thứ khác nhau không phải phép ghép cặp.

| verdict | Nghĩa |
|---|---|
| `universal-suffices` | Không có bằng chứng conditioning thắng, **ở ngân sách này** |
| `conditioning-helps` | R1 bị bác bỏ |
| `confounded` | Hai arm nhận **số step khác nhau**. Con số chưa có nghĩa — chỉnh `STEPS` cho bằng rồi chạy lại |

`confounded` là cảnh báo tự động cho đúng cái bẫy mà `group_conditioned_triggers.md:175`
đã nêu: cấp full budget cho từng episode rồi đem so với một universal duy nhất thì
arm per-episode thắng nhờ compute, không nhờ conditioning.

## 5. P1: vị trí trigger

```bash
POSITION_MODE=transfer   bash scripts/run_p0_probes.sh position   # rẻ
POSITION_MODE=reoptimize bash scripts/run_p0_probes.sh position   # đắt, nhưng là ablation thật
```

Hai chế độ trả lời hai câu khác nhau, **không được gộp thành một số**:

- `transfer`: một trigger (tối ưu ở `--trigger-position`) đem chấm ở mọi vị trí. Đo
  độ nhạy theo vị trí khi trigger cố định.
- `reoptimize`: tối ưu lại một trigger cho mỗi vị trí. Đây mới là câu "vị trí nào
  thực sự tốt hơn".

Vị trí hỗ trợ: `suffix` (mặc định cũ), `prefix`, `middle`, `both`, `both-repeat`.

- `both` **chia đôi** trigger (`T1 query T2`) → tổng token **không đổi** → so sánh
  công bằng với các vị trí khác.
- `both-repeat` viết **nguyên trigger hai lần** → tốn gấp đôi token. Dòng của nó mang
  `length_matched: false`, và `best_length_matched` trong summary **loại nó ra**. Nếu
  đặt nó cạnh các vị trí khác như thể cùng ngân sách thì "thắng vì vị trí" thực ra là
  "thắng vì dài hơn".

`ATTENTION=1` (mặc định) đo thêm phần attention mà `[CLS]` phát ra rơi vào token
trigger ở layer cuối. DPR pool chính hàng `[CLS]`, nên đây là kênh duy nhất trigger
tác động được tới vector — và là bằng chứng *mechanistic* cho nhận định "chèn vào
giữa thì trọng số không ảnh hưởng lắm". Encoder nào không trả về attention thì trường
này là `supported: false`, probe vẫn ra đủ số retrieval.

## 6. Thay đổi phá vỡ artifact cũ

Hai thay đổi dưới đây làm **mọi checkpoint và trajectory cũ không resume được**. Đây
là cố ý, nhưng cần biết trước khi chạy lại trên artifact đang có:

1. `TrainConfig` có thêm trường `trigger_position`. Nó nằm trong `config_hash`, nên
   checkpoint train trước đây sẽ bị `evaluate`/`probe-*` từ chối. Phải train lại.
   Lý do giữ nó trong contract: trigger export ra mà bị chấm ở vị trí khác lúc tối ưu
   là lỗi âm thầm, không crash — đúng cảnh báo ở `open_problems.md` §P1.
2. `Snapshot.note` có thêm `drift_seed`, nên `trajectory_hash` đổi. Chạy lại
   `prepare-drift`.

Đường suffix thì **không đổi một bit nào** — `tests/test_mcat_probes.py::PlacementTests::test_suffix_is_unchanged_by_the_position_argument` giữ điều đó, nên mọi số đã đo
trước đây vẫn so sánh được với số mới ở `--trigger-position suffix`.

## 7. Bug đã sửa khi dựng probe

Ghi lại vì cả hai đều thuộc loại **sai âm thầm**, không báo lỗi:

- `drift.build_trajectory` đặt `snapshot_id = f"{episode_id}-{kind}"`, không có seed.
  `drift_eval.row_key` băm chính id đó, nên chạy 5 drift seed sẽ sinh **khóa trùng
  nhau**: lúc resume sẽ nhận lại dòng của seed khác mà không báo gì — "lặp 5 lần" trở
  thành "1 lần nhân bản 5". Đã thêm `tag_seed`, mặc định tắt để không làm hỏng
  trajectory dựng trước đây; mọi caller đổi seed **bắt buộc** bật.
- Contract của probe mang tuple (growth levels, drift seeds, positions). JSON chỉ có
  array, nên so contract mới với contract đã lưu là so tuple với list → **mọi lần
  resume đều bị từ chối**. Đã round-trip cả hai phía qua JSON trước khi so.

## 8. Artifact sinh ra

```
$RUN_ROOT/
  b2/seed_0/
    episodes.jsonl  manifest.json  checkpoint.pt (chỉ khi r1 chạy)
    evaluation.jsonl  evaluation.json           # R1 treatment
    probe_universal.json                        # R1 verdict
    probes/
      growth-suffix-test/
        probe_growth.jsonl    # 1 dòng / (episode, snapshot, seed)
        probe_growth.json     # verdict R2
        probe_config.json     # contract, khóa thư mục vào một cấu hình
        poison_s0.pt          # key poison đóng băng
        costs.json
      position-transfer-suffix-test/
        probe_position.jsonl  probe_position.json  probe_config.json
  b3/seed_0/                                    # R1 control
  r2.console  r1.console  position.console      # log đầy đủ từng probe
```

Mỗi `(--trigger-position, --split)` có thư mục riêng: hai vị trí là hai thí nghiệm,
trộn dòng vào một file thì trung bình sẽ gộp hai thứ mà paper báo cáo tách nhau.

## 9. Ghi kết quả

Dù verdict là gì, chép vào `_idea_q1_aplus/p0_falsification_results.md`: ngày, lệnh
đã chạy, `RUN_ROOT`, và ba verdict. Kết quả xấu ở đây tiết kiệm hàng tuần GPU, nên nó
đáng được ghi như một kết quả chứ không phải một lần chạy hỏng.
