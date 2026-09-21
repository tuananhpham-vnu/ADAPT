# 22 — MCAT M3: memory drift, few-step adaptation và cost accounting

> Viết ngày 2026-09-20, dựa trên code thực tế của `src/triggers/mcat/` (M0–M2 đã xong).
> Đọc kèm `_idea/memory_conditioned_generator_Q1_A_star.md` §10, §11, §13 (định nghĩa
> nghiên cứu) và `_guidance/20_mcat_kaggle_runbook.md` (cách chạy M0–M2).
> **Cập nhật 2026-09-20: bước 1–7 đã implement xong** (xem §10); phần còn lại là bước 8,
> quét ngân sách few-step trên validation, cần GPU. Mọi con số hiện có đều là fixture CPU
> và **không phải kết quả**.

## 0. M3 trả lời câu hỏi nào

M2 đã cho biết generator có phân biệt được hai snapshot khác nhau hay không
(`shuffled_context_control`). M2 **không** cho biết điều mà toàn bộ luận điểm
amortization dựa vào:

> Khi bộ nhớ của agent thay đổi, sinh lại trigger bằng một forward pass có rẻ hơn
> và có tốt bằng việc tối ưu lại từ đầu không?

M3 là thí nghiệm duy nhất trả lời được câu đó. Nó cũng là mốc có thể **giết** đề tài:
nếu `warm-start` rẻ gần bằng generator mà chất lượng ngang, §14 của `_idea/` bắt pivot
sang "lookup + refinement", không được claim generator là cần thiết.

Ngoài phạm vi M3 (để ở M4): behavioral ASR trên agent thật, utility clean task,
transfer T2/T3/T4. M3 chỉ đo **retrieval** và **chi phí**. Không câu nào trong report
của M3 được phép dùng chữ "attack success".

## 1. Ba trục biến thiên

M3 là tích của ba trục, mọi ô đều phải báo cáo:

| Trục | Giá trị | Ghi chú |
|---|---|---|
| Drift | `s0` (gốc), `growth-25`, `growth-50`, `growth-100`, `mixture`, `deletion`, `distractor` | Khóa trước khi xem test |
| Method | `reuse`, `generate`, `warm-start`, `scratch` | Cùng snapshot, cùng quyền ghi poison |
| Poison policy | `refresh`, `fixed` | `PoisonPolicy` đã có trong `objectives.py` |

Baseline nào cũng được quyền như nhau: cùng snapshot, cùng `Q_sup`, cùng budget ghi
poison `B = episode.budget_poison`, cùng scorer. Nếu một method được refresh poison thì
mọi method đều được, và số lần ghi index phải bằng nhau — nếu không, so sánh chi phí là
vô nghĩa.

## 2. Snapshot và trajectory — module mới `src/triggers/mcat/drift.py`

`Episode` hiện đã có `snapshot_id = f"{domain}-{split}-{index:03d}-s0"` nhưng chưa có gì
sinh ra snapshot thứ hai. M3 thêm:

```python
@dataclass(frozen=True)
class Snapshot:
    snapshot_id: str          # "<episode_id>-growth-50"
    episode_id: str
    kind: str                 # base | growth | mixture | deletion | distractor
    growth: float             # 0.0 cho s0 và cho deletion
    doc_ids: list[str]        # danh sách tuyệt đối, không phải delta
    parent_snapshot_id: str | None
    note: dict[str, Any]      # số doc thêm/bớt, nguồn lấy doc, seed

def build_trajectory(
    episode: Episode,
    pool: list[dict[str, Any]],          # documents CÙNG split của cùng domain
    *,
    growth: tuple[float, ...] = (0.25, 0.50, 1.00),
    ablations: tuple[str, ...] = ("mixture", "deletion", "distractor"),
    seed: int,
) -> list[Snapshot]
```

Ràng buộc bắt buộc, phải có test:

1. **Không rò rỉ split.** `pool` chỉ chứa documents mà `assign_families` gán về đúng
   `episode.split`. Tái dùng `family_key` / `assert_no_family_leak` trong `episodes.py`,
   mở rộng cho `doc_ids` của mọi snapshot chứ không chỉ của episode.
2. **Không chọn drift theo kết quả.** Documents thêm vào lấy bằng
   `random.Random(stable_hash([episode.episode_id, kind, seed])[:16]).sample(...)`,
   không bao giờ lọc theo "doc này làm method X hỏng". Ghi rõ nguyên tắc này trong
   docstring — đây là chỗ dễ vô tình p-hack nhất của M3.
3. **Snapshot là danh sách tuyệt đối.** Lưu `doc_ids` đầy đủ chứ không lưu delta, để
   `Workspace` không phải biết lịch sử và để hash ổn định.
4. **`distractor` phải định nghĩa trước.** Hard benign distractor = document cùng split
   có dot-product cao nhất với trung bình `support_vectors`, **chọn trên `Q_sup` chứ
   không phải `Q_eval`**. Lấy top-M với `M = ceil(0.25 * len(doc_ids))`. Nếu cho phép
   nhìn `Q_eval`, control này trở thành oracle và mất giá trị.
5. **`mixture`** thay 30% documents bằng documents của domain khác cùng split (chỉ khả
   thi khi run có ≥ 2 domain; nếu không, ghi `{"applicable": false, "reason": ...}` chứ
   không im lặng bỏ qua).
6. **`deletion`** xóa 25% documents, kể cả khi document đó đang nằm trong top-K của một
   số query — nhưng chọn ngẫu nhiên, không cố ý.

Artifact: `trajectories.jsonl` (một `Snapshot` mỗi dòng) + `manifest["trajectory_hash"] =
stable_hash([s.to_json() for s in snapshots])`. Hash này phải đi vào `build_contract`
(§8) để resume không trộn hai trajectory khác nhau.

## 3. Thay đổi cần làm trong `runtime.py`

Ba chỗ hiện đang giả định "một episode = một snapshot":

```python
# hiện tại
def reference_centers(self, episode: Episode, memory: torch.Tensor) -> torch.Tensor:
    if episode.snapshot_id not in self._centers: ...

def context_for(self, episode: Episode) -> EpisodeContext:
    memory = select_rows(self.document_vectors(episode.domain), doc_order, episode.doc_ids)
```

Sửa thành:

```python
def reference_centers(self, snapshot_id: str, memory: torch.Tensor) -> torch.Tensor
def context_for(self, episode: Episode, snapshot: Snapshot | None = None) -> EpisodeContext
```

Khi `snapshot` khác `None`: `memory` lấy theo `snapshot.doc_ids`, centers cache theo
`snapshot.snapshot_id`. `support_vectors`, `optimization_texts`, `poison_texts` và
`eval_texts` **không đổi** — drift là drift của bộ nhớ, không phải của phân phối query.
Trộn hai thứ đó lại thì không còn biết gain đến từ đâu.

`EpisodeContext` thêm field `snapshot_id: str` (mặc định `episode.snapshot_id`) để mọi
dòng metrics nói được nó thuộc snapshot nào. Đây là thay đổi phá chữ ký, nên
`evaluate.generate_trigger` (chỗ dựng lại `EpisodeContext` cho `memory_override`) phải
được cập nhật cùng lúc — nếu không, control shuffled-context sẽ lặng lẽ mất field.

Cache vector: `encode_corpus` cache theo (corpus, retriever) nên snapshot **không** làm
encode lại; snapshot chỉ là một tập chỉ số khác trên cùng ma trận. Giữ nguyên tính chất
này, đừng thêm cache theo snapshot — nếu không mỗi trajectory tốn 7× dung lượng.

## 4. Fixed poison dưới drift — chi tiết dễ sai nhất

`PoisonPolicy("fixed")` hiện chỉ `detach()` gradient của poison trong **cùng** một bước
train. Dưới drift, "fixed" có nghĩa mạnh hơn: poison đã được ghi vào index ở `s0` bằng
trigger của `s0`, và **không được encode lại** ở snapshot sau.

Nghĩa là `evaluate._poison_keys(context, trigger_ids, retriever)` — hiện luôn encode
poison bằng trigger đang xét — là sai cho nhánh fixed. Cần:

```python
def _poison_keys(
    context, trigger_ids, retriever, *, frozen: torch.Tensor | None = None
) -> torch.Tensor:
    if frozen is not None:
        return frozen          # đã ghi ở s0, không encode lại
    ...
```

cộng với artifact `poison_s0.pt` (`{episode_id: tensor}`) sinh một lần ở snapshot gốc.
Test bắt buộc: hai snapshot khác nhau với `--poison-mode fixed` phải cho **đúng cùng**
poison vectors (bit-identical), còn `refresh` thì phải khác. Không có test này thì nhánh
fixed sẽ âm thầm mượn lợi thế của refresh và mọi số dưới đó đều vô giá trị.

## 5. Bốn method — module mới `src/triggers/mcat/adapt.py`

```python
ADAPT_METHODS = ("reuse", "generate", "warm-start", "scratch")

def adapt_trigger(
    method: str,
    module: nn.Module,
    config: TrainConfig,
    context: EpisodeContext,
    retriever: Retriever,
    *,
    steps: int,                    # ngân sách few-step, chỉ dùng cho warm-start
    ledger: CostLedger,
) -> tuple[dict[str, Any], nn.Module]
```

| Method | Làm gì | Vai trò |
|---|---|---|
| `reuse` | Dùng lại đúng trigger text đã export ở `s0`, không tính toán gì | Cận dưới. Nếu `reuse` không tụt, drift không đủ mạnh |
| `generate` | Một forward `_logits_for` trên snapshot mới rồi `export_hard_trigger` | Claim chính của MCAT |
| `warm-start` | Copy `module.state_dict()` rồi chạy `steps` bước optimizer trên snapshot mới | Đối thủ thật sự |
| `scratch` | `_logits_source(...)` mới hoàn toàn, chạy đủ `config.steps` | Cận trên về chất lượng, đắt nhất |

Ba điều bắt buộc:

- `warm-start` và `scratch` chỉ được thấy `Q_sup` và `Q_opt`. `EpisodeContext` đã cố ý
  không mang `Q_eval` — giữ nguyên tính chất đó, đừng truyền `workspace.eval_texts` vào
  `adapt.py` dưới bất kỳ hình thức nào.
- `steps` (ngân sách few-step) **chọn trên validation**, khóa trước khi chạy test. Quét
  `steps ∈ {1, 5, 10, 25, 50}` trên validation, lấy điểm mà `warm-start` đạt xấp xỉ chất
  lượng của `scratch`, rồi dùng đúng con số đó cho test.
- Mọi method export qua cùng `export_hard_trigger` + `round_trip_report`. Số của M3 là số
  trên text đã decode và re-tokenize, y như M2.

Với `config.mode == "direct-logit"`, `generate` không áp dụng được (B2 không có gì để
transfer) — trả `{"applicable": false, "reason": "direct-logit has nothing to transfer"}`,
đúng tinh thần mà `evaluate_stage` đang xử lý bằng thư mục `adapt-<split>/`.

## 6. Cost accounting — module mới `src/triggers/mcat/costs.py`

Đây là nửa còn lại của M3 và hiện repo **chưa đếm gì cả**. `costs.json` phải có:

```python
@dataclass
class CostLedger:
    encoder_forward: int = 0      # số lần gọi encode_* (nhân batch size)
    encoder_backward: int = 0     # số lần .backward() đi qua retriever
    optimizer_steps: int = 0
    index_writes: int = 0         # = B mỗi lần refresh poison, 0 khi fixed
    gmm_refits: int = 0           # fit_centers cho snapshot mới
    scorer_calls: int = 0         # GPT-2/Llama nếu bật; M3 mặc định 0
    wall_seconds: float = 0.0
    peak_vram_bytes: int = 0      # torch.cuda.max_memory_allocated, reset mỗi phase
    online_latency_seconds: list[float] = field(default_factory=list)
```

Cách đếm mà không rải counter khắp nơi: một `ContextVar` giữ ledger đang hoạt động, và
`costs.record(name, amount)` tính tiền vào đó — không có ledger thì nó là no-op, nên
instrumentation không đổi thứ mà run tính ra. Mỗi counter chỉ có đúng một điểm đo:
`encoder_forward` ở `encoding.encode_plain` / `encode_with_trigger_embeddings`,
`gmm_refits` ở nhánh cache-miss của `runtime.reference_centers`, `encoder_backward` và
`optimizer_steps` trong vòng lặp `train.train`, `index_writes` ở nhánh refresh của
`evaluate._poison_keys` và ở `poison.freeze_poison`.

`index_writes` **không** đếm mỗi bước optimizer: bước train có encode lại poison nhưng
không ghi vào index đã triển khai. Đếm ở đó sẽ thổi phồng chi phí của nhánh refresh lên
hàng trăm lần và làm hỏng đúng phép so sánh mà M3 cần.

Phân tách bắt buộc — §11.2 của `_idea/` yêu cầu cả hai:

- **deployment-only**: chi phí cho một snapshot mới, không tính train.
- **lifecycle**: `C_train + C_prep + N * C_online`.

Break-even:

```python
def break_even(train_cost: float, online_search: float, online_gen: float) -> float | None:
    denominator = online_search - online_gen
    return train_cost / denominator if denominator > 0 else None
```

Trả `None` khi mẫu số ≤ 0, và report in `null` — **không** in một số âm rồi diễn giải nó
là "hòa vốn ngay lập tức". Chỉ được so N* giữa các method có chất lượng tương đương;
generator nhanh nhưng hit@K thấp hơn nhiều không phải speedup. Report phải in N* kèm
`quality_gap` (hiệu hit@K so với `scratch`) ngay cạnh nó.

Warmup: trước khi đo latency, chạy 3 lần bỏ đi và `torch.cuda.synchronize()` trước/sau
mỗi phép đo. Báo p50 và p95, không báo mean.

## 7. Thống kê — module mới `src/triggers/mcat/stats.py`

§11.2 yêu cầu paired differences và bootstrap theo episode family/trajectory, không phải
theo query (query trong cùng episode không độc lập).

```python
def paired_bootstrap(
    left: list[float], right: list[float], groups: list[str],
    *, iterations: int = 10_000, seed: int = 0, alpha: float = 0.05,
) -> dict[str, float]     # {"mean_difference", "ci_low", "ci_high", "groups"}
```

Resample **theo group** (group = family của episode, hoặc trajectory id), không resample
từng hàng. Báo ba mức tổng hợp: macro theo episode, micro theo query, và nhóm kém nhất
(worst-episode). Pilot tối thiểu 3 seed, hướng tới 5.

## 8. Artifact và CLI

Stage mới, giữ đúng kiểu resume theo hash của `cli.py` hiện tại:

```bash
python -m src.triggers.mcat prepare-drift   --output-dir outputs/mcat/pilot --growth 0.25 0.5 1.0
python -m src.triggers.mcat adapt           --output-dir outputs/mcat/pilot --split test \
    --method reuse --method generate --method warm-start --method scratch \
    --adapt-steps 10 --poison-mode refresh
python -m src.triggers.mcat evaluate-drift  --output-dir outputs/mcat/pilot --split test
python -m src.triggers.mcat report          --output-dir outputs/mcat/pilot   # mở rộng
```

`make.ps1` thêm `mcat-drift`, `mcat-adapt`, `mcat-evaluate-drift` cạnh các target hiện có.

Artifact mới trong thư mục run:

```text
trajectories.jsonl        # Snapshot, một dòng một cái
poison_s0.pt              # khóa poison đã "ghi vào index" ở s0, cho nhánh fixed
adapt/<snapshot>/<method>/checkpoint.pt, triggers.jsonl, metrics.jsonl
drift_evaluation.jsonl    # một dòng cho mỗi (episode, snapshot, method, poison_mode)
drift_evaluation.json     # tổng hợp + CI
costs.json                # ledger từng phase + break-even
REPORT.md                 # thêm mục "Drift và few-step"
```

`build_contract` thêm `trajectory_hash` và `adapt_config_hash` (danh sách method + steps +
poison_mode). Đổi mức drift, đổi `--adapt-steps`, đổi danh sách method → contract khác →
resume bị từ chối. Đây là tính chất đã có của M0–M2, giữ nguyên, không nới lỏng.

Mỗi dòng `drift_evaluation.jsonl`:

```json
{"episode_id": "...", "snapshot_id": "...-growth-50", "kind": "growth", "growth": 0.5,
 "method": "warm-start", "poison_mode": "fixed", "trigger": "...",
 "round_trip_valid": true, "trigger_on": {}, "trigger_off": {},
 "false_activation": 0.0, "cost": {}}
```

`trigger_on`/`trigger_off` giữ nguyên schema của `retrieval_metrics` hiện tại (hit@K,
occupancy@K, mean/p10/worst margin, mean rank, MRR) — không đổi tên metric, vì bẫy số 1
trong README của MCAT chính là nhầm hai định nghĩa margin.

## 9. Test phải viết — `tests/test_mcat_drift.py`

| Test | Bắt lỗi gì |
|---|---|
| `test_trajectory_no_family_leak` | Snapshot lấy document từ split khác |
| `test_trajectory_deterministic` | Cùng seed cho cùng `doc_ids`; đổi seed thì khác |
| `test_growth_monotonic` | `len(doc_ids)` đúng tỉ lệ 1.25/1.5/2.0 và chứa toàn bộ `s0` |
| `test_deletion_subset` | `deletion` là tập con thật sự của `s0` |
| `test_distractor_uses_support_only` | Distractor chọn bằng `Q_sup`; mock `eval_texts` để nó raise nếu bị gọi |
| `test_fixed_poison_immutable` | Hai snapshot, `--poison-mode fixed` → poison tensor bit-identical |
| `test_refresh_poison_changes` | Cùng setup, `refresh` → poison khác |
| `test_adapt_budget_equal` | `index_writes` bằng nhau giữa 4 method ở cùng poison policy |
| `test_warm_start_with_no_steps_is_exactly_generate` | `warm-start` với `steps=0` phải trùng **`generate`**, không phải `reuse` — `reuse` là trigger sinh trên context `s0`, còn warm-start 0 bước sinh trên snapshot mới từ cùng tham số. Bản kế hoạch ban đầu ghi sai chỗ này |
| `test_break_even_null` | Mẫu số ≤ 0 → `None`, không phải số âm |
| `test_paired_bootstrap_groups` | Resample theo group; CI của hai dãy giống hệt nhau chứa 0 |
| `test_contract_rejects_trajectory_change` | Đổi `--growth` → resume bị từ chối |

Chạy trên fixture retriever, CPU, không tải model — giống `tests/test_mcat_pipeline.py`.

## 10. Thứ tự làm

1. ~~`drift.py` + test trajectory (không đụng GPU, không đụng train).~~ **xong** —
   `src/triggers/mcat/drift.py`, `tests/test_mcat_drift.py`.
2. ~~Sửa `runtime.py` cho snapshot + cập nhật `evaluate.generate_trigger`.~~ **xong** —
   `context_for(episode, snapshot=None)`, `reference_centers(snapshot_id, memory)`,
   `EpisodeContext.snapshot_id`, `Workspace.assignment(...)`.
3. ~~`poison_s0.pt` + nhánh fixed + test immutability.~~ **xong** —
   `src/triggers/mcat/poison.py`, `evaluate._poison_keys(..., frozen=...)`.
4. ~~`costs.py` + điểm đếm trong `encoding.py`.~~ **xong** — `src/triggers/mcat/costs.py`,
   `tests/test_mcat_costs.py`. Dùng `record()` gọi thẳng thay vì decorator: vẫn đúng một
   điểm đo mỗi hàm và không đổi chữ ký, nhưng đọc ra ngay ở chỗ encode.
5. ~~`adapt.py` + 4 method.~~ **xong** — `src/triggers/mcat/adapt.py`,
   `tests/test_mcat_adapt.py`.
6. ~~`stats.py` + mở rộng evaluate/report.~~ **xong** — `stats.py` (paired bootstrap,
   macro/micro/worst) và `drift_eval.py`. Đặt vòng chạy drift ở module riêng thay vì
   nhét vào `evaluate.py` để tránh vòng import `evaluate -> adapt -> evaluate`.
7. ~~CLI stage + `make.ps1` + smoke.~~ **xong** — `prepare-drift`, `adapt`,
   `evaluate-drift`; `smoke` chạy hết cả ba; target `mcat-drift`, `mcat-adapt`,
   `mcat-evaluate-drift`.
8. **Chưa chạy** — cần GPU. Script đã sẵn sàng:
   `DRIFT=1 SWEEP_STEPS="1 5 10 25 50" bash scripts/run_mcat.sh m1` quét budget trên
   **validation**, mỗi giá trị ghi vào `drift/refresh-steps<N>-validation/` riêng. Đọc
   `drift_evaluation.json` của từng thư mục, khóa `ADAPT_STEPS`, rồi mới chạy test.

Bước 1–7 chạy được hoàn toàn trên CPU bằng fixture. Chỉ bước 8 cần GPU, và nó dùng lại
`scripts/run_mcat_kaggle.sh` (thêm biến `DRIFT=1`) chứ không viết script mới.

## 11. Cái gì khiến M3 thất bại — ghi trước khi chạy

Theo §14 của `_idea/memory_conditioned_generator_Q1_A_star.md`, sau khi có số phải đối
chiếu ngay:

- `warm-start` với `steps` nhỏ đạt chất lượng của `generate` và tổng chi phí không lớn hơn
  đáng kể → **pivot**: MCAT là nghiên cứu optimizer, không phải bằng chứng generator cần
  thiết.
- `reuse` không tụt qua mọi mức drift → drift quá yếu, kết quả không nói lên điều gì; tăng
  mức drift **trước khi** nhìn test, và ghi lại rằng đã tăng.
- Gain của `generate` biến mất sau round-trip → không có gain, dừng claim.
- Không tồn tại N* dương trong số snapshot thực tế của một triển khai → bỏ claim
  amortization, giữ lại phần conditioning như kết quả cơ chế.

Không sửa ngưỡng sau khi thấy test. Nếu pilot thiếu power, kết luận là "chưa kết luận" và
tăng **số episode độc lập**, không tăng số query trong cùng episode.
