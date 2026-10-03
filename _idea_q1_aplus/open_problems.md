# Open problems (các vấn đề còn tồn tại)

Ghi lại ngày 2026-10-01. File này chỉ ghi **vấn đề mở** + hướng xử lý ứng viên,
không phải kết quả đã verify.

**Thứ tự ưu tiên: P0 → P3 → (P1, P2).** P0 là hai test bác bỏ, rẻ, và quyết định
đề tài có đứng được không. Chưa chạy xong P0 thì **không** đầu tư huấn luyện lớn.
P3 (cập nhật 2026-10-02) là các yêu cầu thực nghiệm rút ra từ related work —
không phải ý tưởng mới, mà là **những so sánh reviewer sẽ đòi**; thiếu cái nào
thì claim chính không đứng được dù số đẹp.

---

## P0. HAI RỦI RO GIẾT ĐỀ TÀI — phải đánh giá TRƯỚC khi huấn luyện lớn

**Ưu tiên tuyệt đối. P1/P2 dưới đây đều vô nghĩa nếu P0 cho kết quả xấu.**

> **Kết quả lần chạy đầu (2026-10-03) → [`p0_falsification_results.md`](p0_falsification_results.md).**
> R2 `stable`, R1 `universal-suffices` — cả hai rủi ro **đúng** trong thiết lập đã
> chạy, nhưng `hit@5` chạm trần 1.0 ở mọi arm nên phép đo không phân biệt được. Xem
> bảng margin và nguồn từng con số trong file kết quả.

### Lần chạy 2 — AgentDriver, 6 token (chốt TRƯỚC khi chạy, 2026-10-03)

Đổi thiết lập **sau** khi thấy lần 1 chạm trần → phải ghi trước, và lần 1 vẫn được
báo cáo nguyên vẹn. Quyết định của người dùng: tập trung AgentDriver, trigger 6 token,
đánh giá đủ lớn.

| | Lần 1 (qa) | **Lần 2 (ad)** |
|---|---|---|
| Domain | StrategyQA | **AgentDriver** (key = `ego + perception`, max_length 512) |
| Trigger | 10 token | **6 token** |
| Memory / episode | 512 | **2000** (split test có 4520 → +100% vẫn trong split) |
| Query eval / episode | 128 (509 query khác nhau) | **1000** (trên 1379 query test) |
| Episode test | 8 | **16** |
| Không đổi | top-5, 5 poison, 400 step, 5 drift seed, 25/50/100%, `refresh`, opt 16 query | |

**Tiêu chí đọc (chốt trước):**
1. Verdict của code (`probe_growth`, `probe_universal`) được báo nguyên văn.
2. **Cờ trần:** nếu `hit@5` ở base ≥ 0.95 cho **cả** B2 và B3 thì verdict R1/R2 được
   ghi là *chạm trần — không phân biệt được*, không phải bằng chứng R1/R2 đúng hay sai.
   Khi đó báo thêm mean / worst margin (paired theo episode), nhưng **không** đổi
   verdict dựa trên margin.
3. R2 chỉ được coi là bị bác bỏ khi verdict là `decays`; R1 chỉ khi `conditioning-helps`.
4. Báo `off_hit` (false activation) cạnh mọi con số ASR.

Toàn bộ luận điểm của MCAT (memory-conditioned trigger + amortization) chỉ tồn
tại nếu **cả hai** mệnh đề sau đều SAI:

| Rủi ro | Mệnh đề | Nếu đúng thì sao |
|---|---|---|
| **R1** | Một trigger **phổ quát** (một trigger cho mọi episode) đã đủ tốt | Không cần conditioning trên memory → generator mất lý do tồn tại → mất đóng góp chính |
| **R2** | Memory **trôi** nhưng trigger cũ **không suy giảm** đáng kể | Không cần re-generate → `reuse` thắng mọi method → mất luôn luận điểm amortization (tiết kiệm chi phí so với cái gì?) |

Đây là hai **giả thuyết cần bác bỏ**, không phải hai thứ cần tối ưu. Chi phí
kiểm tra gần như bằng 0 so với chi phí train generator, nên không có lý do nào
để train trước khi biết câu trả lời.

### Tin tốt: cả hai test đã có sẵn trong code, không cần viết method mới

**R1 → đã có baseline B3 `universal-logit`.**
`src/triggers/mcat/train.py:42` — `MODES = ("direct-logit", "universal-logit", "generator")`.
Docstring nói đúng mục đích này:
- `train.py:7` — *"`universal-logit` B3 — one logits matrix fitted across all train episodes"*
- `train.py:11` — *"B3 whether one universal trigger already suffices, and only the gap above both ..."*

Tức là ý định đã được ghi vào code từ đầu; việc còn lại là **chạy và đọc số**.

**R2 → đã có method `reuse` + hạ tầng drift.**
- `src/triggers/mcat/adapt.py:57` — `ADAPT_METHODS = ("reuse", "generate", "warm-start", "scratch")`
- `adapt.py:160-167` — `reuse` = lấy đúng trigger của `s0` áp lên snapshot đã trôi
- `src/triggers/mcat/drift.py` — sinh trajectory snapshot: phình 25/50/100%, và
  `mixture` / `deletion` / `distractor`
- `src/triggers/mcat/drift_eval.py` — chấm từng ô `(episode, snapshot, method)`,
  giữ riêng dòng `base` làm mốc chưa trôi

Vậy R2 chính là câu hỏi: **`reuse` tụt bao nhiêu khi đi từ `base` sang các snapshot trôi?**

### Thí nghiệm P0-R2 (protocol chốt) — memory phình, trigger giữ nguyên

Thí nghiệm nhỏ nhất đủ trả lời R2. Chạy cái này trước cả R1 vì nó rẻ hơn (không
cần train lại gì cả — chỉ encode và đo).

**Thiết kế**
1. Chọn `M_0` và một trigger **đang hoạt động tốt** ở `base` (không chọn trigger
   yếu rồi kết luận nó suy giảm — chọn theo ASR-r ở `base`, trước khi drift).
2. **Đóng băng cả ba thứ**: trigger, các bản ghi poison, và tập query đánh giá.
   Chỉ một biến duy nhất được thay đổi: lượng dữ liệu lành trong memory.
3. Thêm dữ liệu lành để memory tăng **25% / 50% / 100%**. Giữ nguyên retriever,
   giữ nguyên **top-k**.
4. Đo ở từng mức 2 chỉ số:

| Chỉ số | Ý nghĩa | Kỳ vọng nếu R2 SAI (tốt cho ta) |
|---|---|---|
| Tỷ lệ truy xuất poison **khi có trigger** | Trigger cũ còn hoạt động không? | giảm rõ và đơn điệu theo mức phình |
| Tỷ lệ truy xuất poison **khi không có trigger** | Poison có bị kích hoạt ngoài ý muốn không? | ở mức thấp và ổn định |

**Hai luật không được vi phạm**
- **Giữ cố định SỐ LƯỢNG poison, không giữ cố định TỶ LỆ poison.** Nếu tăng số
  poison theo memory thì hai nguyên nhân (memory to hơn / poison nhiều hơn) trộn
  vào nhau và không còn cô lập được tác động của việc memory phình. Đây là lỗi dễ
  mắc nhất ở thí nghiệm này.
- **Lặp 3–5 lần lấy mẫu** với dữ liệu thêm là **ngẫu nhiên cùng miền**. Một lần
  chia dữ liệu không đủ để kết luận: chênh lệch giữa các seed có thể lớn hơn
  chênh lệch giữa các mức phình, và nếu vậy thì "suy giảm" chỉ là nhiễu.

**Kết luận đọc theo đúng hai nhánh này**
- **Hiệu quả KHÔNG giảm** → *chưa có bằng chứng* cần sinh trigger mới **trong
  thiết lập này**. Lưu ý diễn đạt: đây là "chưa có bằng chứng", không phải "đã
  chứng minh là không cần" — kết luận chỉ áp cho growth cùng miền, chưa nói gì về
  `mixture` / `distractor`.
- **Hiệu quả giảm rõ và nhất quán** (nhất quán qua các seed, đơn điệu theo mức
  phình) → bước tiếp theo là kiểm tra **tối ưu lại trên memory mới có phục hồi
  được không**. Phục hồi được thì mới có luận điểm amortization; không phục hồi
  được thì vấn đề nằm ở chỗ khác (poison bị chôn vì top-k, không phải vì trigger lạc).

**Map vào code — gần như không cần viết mới**
- `drift.py:57` — `KINDS = ("base", "growth", "mixture", "deletion", "distractor")`;
  vòng này chỉ dùng `base` + `growth`.
- `drift.py:59` — `DEFAULT_GROWTH = (0.25, 0.50, 1.00)`, **khớp đúng** 25/50/100%.
- `drift.py:228` — tài liệu thêm lấy bằng `_rng(...).sample(available, taken)`,
  và `_check_pool` ép `available` chỉ trong split của chính snapshot → đúng yêu cầu
  "ngẫu nhiên cùng miền", không rò split.
- `poison.py:74` `FrozenPoison` — giữ nguyên key poison đã ghi ở `s0`, nên số
  lượng poison tự động cố định, không bị re-encode. Đây chính là luật 1 ở trên,
  đã được ép trong code.
- `evaluate.py:149-151,161-162` — `trigger_on` / `trigger_off` chính là 2 chỉ số
  trong bảng trên; không cần metric mới.

**Bug đã sửa (2026-10-01)**
`drift.py` từng đặt `snapshot_id = f"{episode_id}-{kind}"` — **không có seed trong
id**. Mà `drift_eval.row_key` băm chính id đó, nên chạy 5 seed sẽ sinh **row_key
trùng nhau**, và lúc resume sẽ nhận lại dòng của seed khác mà không báo lỗi — "lặp
3–5 lần" trở thành 1 lần nhân bản 5. Đã thêm `tag_seed` vào `build_trajectory`
(mặc định tắt, để trajectory dựng trước đây vẫn resume được; mọi caller đổi seed
bắt buộc bật). `tests/test_mcat_probes.py::GrowthProtocolTests` giữ điều này.

### Tiêu chí quyết định (chốt TRƯỚC khi nhìn số)

Phải ghi ngưỡng ra đây trước, nếu không sẽ tự lừa mình bằng cách đọc số rồi mới
chọn ngưỡng.

- **R1 bị bác bỏ** nếu per-episode / memory-conditioned trigger hơn B3
  `universal-logit` một khoảng **có ý nghĩa thống kê** (paired bootstrap đã có ở
  `src/triggers/mcat/stats.py`) trên ASR-r, **ở cùng ngân sách** (xem cảnh báo
  ngân sách ở `group_conditioned_triggers.md:175` — đừng cấp full budget cho từng
  episode rồi so với một universal duy nhất).
- **R2 bị bác bỏ** nếu `reuse` **suy giảm rõ** theo mức drift, và khoảng cách
  `reuse` vs `scratch` **nới ra** khi drift tăng. Nếu đường `reuse` gần như phẳng
  thì R2 đúng.
- Nếu **R1 hoặc R2 đúng** → dừng, không train generator lớn. Đổi hướng: khi đó
  câu chuyện paper phải chuyển sang P1/P2 (position sensitivity, trace dilution)
  hoặc sang trục khác, chứ không phải conditioning.

### Code đã có (2026-10-01)

Protocol ở trên đã được cài thành stage chạy được, không phải pseudo-code:

- `src/triggers/mcat/probes.py` — `growth_probe` / `summarize_growth` (R2),
  `compare_runs` (R1), `position_probe` / `summarize_position` / `attention_mass` (P1).
- CLI: `probe-growth`, `probe-position`, `probe-universal`.
- `scripts/run_p0_probes.sh` — `preflight` / `r2` / `r1` / `position` / `all`.
- `_guidance/24_p0_probe_kaggle_runbook.md` — cách chạy trên Kaggle và cách đọc verdict.
- `tests/test_mcat_probes.py` — 41 test, chạy CPU bằng fixture encoder.

Ba tiêu chí của R2 ở trên được **ép trong code**, không phải nhắc nhở: `summarize_growth`
chỉ trả `decays` khi khoảng tin cậy ghép cặp không chứa 0, **và** đơn điệu, **và**
hiệu ứng lớn hơn `seed_spread`. Ngân sách lệch ở R1 bị `compare_runs` gắn cờ
`confounded` thay vì trả ra một con số trông có vẻ có nghĩa.

### Cần làm (theo đúng thứ tự này)
- [x] `bash scripts/run_p0_probes.sh r2` → `stable` (2026-10-03), nhưng hit@5 chạm trần — xem `p0_falsification_results.md`.
- [x] `bash scripts/run_p0_probes.sh r1` → `universal-suffices` (2026-10-03); B3 còn hơn B2 trên margin. Trả lời R1 (B3 `universal-logit` ghép cặp
  với B2 `direct-logit`, cùng `--steps`).
- [ ] Chỉ khi **cả hai bị bác bỏ** mới train generator / chạy experiment lớn.
- [x] `bash scripts/run_p0_probes.sh position` → P1 chạy cả transfer + reoptimize; on_hit trần, chỉ off_hit khác (middle tệ nhất).
- [x] Ghi kết quả P0 vào một file riêng (`p0_falsification_results.md`), kể cả
  khi kết quả xấu — kết quả xấu ở đây tiết kiệm hàng tuần GPU.

---

## P1. Vị trí chèn trigger: đầu / cuối / cả hai

### Trạng thái hiện tại
Pipeline MCAT chỉ hỗ trợ **một** vị trí duy nhất: trigger được nối vào **cuối**
query, ngay trước `[SEP]`.

- `src/triggers/mcat/encoding.py:6` — assembly cố định là `[CLS] prefix trigger [SEP]`
- `src/triggers/mcat/encoding.py:55` — `encode_with_trigger_embeddings` docstring:
  *"appended before `[SEP]`"*
- `src/triggers/mcat/encoding.py:76-80` — `prefix = tokenizer(text, ...)` rồi
  `cat([CLS]+prefix, trigger_embeds, [SEP])`

Tức là cả lúc **tối ưu** (soft/relaxation) và lúc **eval** (hard trigger) đều
dùng chung một giả định vị trí. Không có biến điều khiển vị trí nào.

### Vấn đề
1. Vị trí là một **siêu tham số chưa được khảo sát**. Kết quả ASR hiện tại chỉ
   nói về trigger-suffix, không nói gì về trigger-prefix hay trigger hai đầu.
2. Nhận định (từ quan sát, **chưa đo chặt**): chèn vào **giữa** query thì ảnh
   hưởng của trigger lên embedding yếu — trọng số (attention mass) mà trigger
   giành được bị các token query hai bên nhấn chìm. Đầu / cuối thì mạnh hơn, vì:
   - đầu: gần `[CLS]`, mà `[CLS]` chính là vector pooling của DPR;
   - cuối: gần `[SEP]`, hưởng vị trí biên, ít bị "trung bình hoá".
3. Nếu điều này đúng thì nó là một **phát hiện có giá trị cho paper** (position
   sensitivity của retriever-level trigger), nhưng hiện ta chưa có số để claim.

### Đã làm (2026-10-01)
- [x] `trigger_position` xuyên suốt `encoding.py` → `train.py` → `evaluate.py` →
  `poison.py` → `drift_eval.py`, và nằm trong `TrainConfig` nên vào thẳng
  `config_hash`: checkpoint tối ưu ở một vị trí **không** eval được ở vị trí khác.
  Đường `suffix` không đổi một bit, có test giữ.
- [x] Quyết định cho `both`: **chia đôi trigger** (`T1 query T2`), tổng token không
  đổi → so sánh công bằng. Bản lặp nguyên trigger hai lần là một vị trí riêng tên
  `both-repeat`, mang `length_matched: false`, và bị loại khỏi `best_length_matched`.
  Hai thứ này trả lời hai câu khác nhau nên không được gộp.
- [x] `attention_mass`: phần attention `[CLS]` phát ra rơi vào token trigger ở layer
  cuối. `_trigger_mask` phải trùng `_assemble` từng ô — có test giữ, vì hai hàm nằm
  ở hai module và không có gì khác chặn việc đo nhầm sang token query.

### Còn phải chạy
- [ ] `POSITION_MODE=transfer` rồi `POSITION_MODE=reoptimize` trên DPR thật. Hai chế
  độ trả lời hai câu khác nhau: "cùng một trigger thì vị trí quan trọng cỡ nào" và
  "vị trí nào thực sự tốt hơn". Không gộp thành một số.
- [ ] Quét thêm theo **độ dài trigger** (`--trigger-tokens`) để tách "vị trí" khỏi
  "độ dài" trong bảng ablation cuối.

---

## P2. Hạn chế của StrategyQA: trigger bị pha loãng trong trace ReAct

### Vấn đề
Trong StrategyQA (ReAct), **key truy vấn memory tại bước k** không phải là query
gốc, mà là:

```
context_k = query + trigger + Σ_{i<k} (thought_i, action_i, observation_i)
```

Nhưng ta đang **chỉ tối ưu trigger trên `query + trigger`** (k = 0). Hệ quả:

1. **Pha loãng theo độ dài.** Khi k tăng, số token của trace tăng, tỷ lệ token
   trigger trong input giảm → đóng góp của trigger vào vector `[CLS]` giảm →
   ASR tụt dần theo bước. Trigger "thắng" ở bước 1 nhưng có thể "thua" ở bước 3-5,
   đúng lúc attack cần ảnh hưởng tới quyết định cuối.
2. **Distribution shift.** Trigger được tối ưu trên phân phối input (query ngắn)
   khác hẳn phân phối input lúc inference (query + trace dài, có cả văn bản
   observation lấy từ retriever). Train/test mismatch kinh điển.
3. **Vòng phản hồi không được mô hình hoá.** Observation là output của retriever
   — tức là chính thứ attack đang can thiệp. Trace ở bước k phụ thuộc vào việc
   bước k-1 có bị poison hay không. Tối ưu một-bước bỏ qua hoàn toàn động lực này.
4. **Đo lường.** Metric hiện tại (ASR-r một lần) không phản ánh "attack có giữ
   được ảnh hưởng qua cả episode hay không".

### Hướng xử lý ứng viên

**(a) Multi-step / trace-aware optimization — ưu tiên cao nhất**
Lấy mẫu `k ~ P(step)` từ các episode thật, dựng `context_k` thật (prefix query +
trace đã ghi lại), rồi tối ưu trigger trên **kỳ vọng qua các bước** thay vì chỉ
k = 0. Cần một cache trace: rollout clean episode một lần, lưu
`(episode_id, k, context_k)` rồi tái dùng để tối ưu — tránh rollout lồng trong
vòng lặp optimizer.

**(b) Worst-case thay vì average (min-max)**
Objective = `min_k ASR(context_k)` chứ không phải mean. Trigger nào đủ mạnh ở
bước dài nhất thì mặc nhiên mạnh ở bước ngắn. Robust, và dễ bán trong paper
("certified across the trajectory").

**(c) Trigger tái sinh qua observation (self-propagation)**
Thiết kế nội dung của poisoned memory sao cho **observation trả về cũng chứa
trigger**. Khi đó mỗi bước poison thành công tự tái chèn trigger vào context của
bước sau → chống pha loãng bằng cơ chế, không bằng tối ưu. Đây là điểm mới mạnh
nhất về mặt ý tưởng; cần kiểm tra giả định rằng observation được đưa nguyên văn
vào context.

**(d) Chuẩn hoá theo độ dài**
Cho độ dài trigger tăng theo độ dài context (`len(trigger) ∝ len(context)^α`),
hoặc chuẩn hoá lại trước khi pooling. Rẻ, nhưng phá stealth khi trace dài.

**(e) Trigger điều kiện theo pha (step-conditioned)**
Một trigger cho pha đầu (query ngắn), một cho pha sau (trace dài). Nối trực tiếp
với dòng `group_conditioned_triggers.md` / `trigger_hierarchy_pilot_results.md`.

**(f) Tấn công ở vị trí khác trong trace**
Nếu trigger ở query bị pha loãng là tất yếu, thì chèn vào **thought/action** (qua
poisoned memory ở bước trước) hiệu quả hơn. Về bản chất đây là (c) nhìn từ phía
khác.

### Cần làm
- [ ] Đo đường cong **ASR theo bước k** trên StrategyQA trước mọi thứ khác. Đây
  là bằng chứng cho thấy vấn đề có thật và định lượng được nó.
- [ ] Log độ dài context theo bước, để tách "ASR tụt vì dài" khỏi "ASR tụt vì nội
  dung trace".
- [ ] Dựng trace cache, rồi thử (a) + (b). (c) làm sau khi đã có baseline.


---

## P3. Yêu cầu thực nghiệm rút ra từ related work

Ghi lại 2026-10-02 từ việc đọc Zhong et al. (EMNLP 2023) và PoisonedRAG. Đây
**không** phải ý tưởng mới mà là danh sách **so sánh reviewer sẽ đòi**. Mỗi mục
ghi rõ: bài nào gợi ra, code đã có gì, còn thiếu gì.

### Câu hỏi nghiên cứu (chốt)

> **Conditioning trên memory hiện tại có sinh ra trigger hiệu quả hơn — trên
> memory *chưa thấy* — so với một generator chỉ thấy support queries, ở cùng
> ngân sách, mà không phải lặp lại tối ưu tốn kém cho từng memory không?**

Đóng góp phải chứng minh là **học được cách sinh trigger cho cấu hình memory chưa
thấy**. Sinh poison text thuyết phục hơn **không** chứng minh được điều này.

Thứ tự: **chứng minh memory-conditioning có ích trên held-out memory dưới ngân
sách khớp trước**, rồi mới tính tới generation-aware training. Thêm loss thứ hai
ngay từ đầu sẽ khiến không còn tách được *vì sao* MCAT hoạt động.

### Related work và bài học

**Zhong et al., *Poisoning Retrieval Corpora by Injecting Adversarial Passages*,
EMNLP 2023.**
- Ý chính: tạo một passage được retriever chấm điểm cao cho **nhiều câu hỏi khác
  nhau** dù không trả lời câu nào. Dùng HotFlip trực tiếp trên text passage.
- Mở rộng: **cluster các query theo embedding**, tối ưu một passage cho mỗi cụm
  → nhiều passage phủ nhiều vùng. Gần với hướng `group_conditioned_triggers.md`.
- Khác MCAT: Zhong **tìm kiếm trực tiếp** văn bản tấn công cho từng corpus;
  MCAT **học một quá trình sinh** tái dùng được. Clustering ở Zhong là chia query
  để tạo nhiều passage, **không** phải generator học có điều kiện theo memory.
- Khác threat model: Zhong chỉ sửa corpus, **không** sửa query; MCAT gắn trigger
  vào query. Phải nói rõ sự khác biệt này, không so số trực tiếp.

**PoisonedRAG.**
- **Đánh giá cả hai tầng**: query có trigger có lấy được poison không (retrieval),
  *rồi* agent có thực sự làm hành vi mục tiêu không (end-to-end).
- **Ghi rõ giả định quyền truy cập**: MCAT thấy toàn bộ `D`, một mẫu của `D`,
  hay chỉ embedding? Có gradient của retriever không? Có được sửa query không?
- **Kiểm tra giá trị của conditioning**: so với generator không điều kiện trên
  held-out memory, và shuffle memory input để xem generator có thật sự dùng nó.

### Bảng so sánh bắt buộc

| So sánh | Chứng minh điều gì | Trong code |
|---|---|---|
| MCAT vs. tối ưu trigger per-memory | Đánh đổi chất lượng ↔ chi phí | B2 `direct-logit` ✅; B1 `--mode hotflip` ✅ (2026-10-03, chưa có số thật) |
| MCAT vs. generator chỉ thấy query | Giá trị cộng thêm của memory input | B5 `--variant query` ✅ (`generator.py:14`) |
| MCAT với memory đúng vs. memory bị shuffle | Generator có thực sự dùng memory | `shuffled_context_control` ✅ đo ΔASR-r (G1 đã sửa 2026-10-03) |
| MCAT vs. một universal trigger cố định | Có cần thích nghi không | B3 `universal-logit` ✅ — chính là P0-R1 |
| MCAT vs. generator vô điều kiện | Network có chỉ ghi nhớ một lời giải | B4 `--variant none` ✅ |
| MCAT vs. trigger bank nearest-context | Lookup có đủ thay learning không | B7 trong design doc, **chưa cài** |

### Bốn điều thí nghiệm phải chứng minh

1. **Conditioning có tác dụng** — bảng trên, ở cùng ngân sách.
2. **Tổng quát hoá thật** — test trên memory **thực sự held-out**. Hiện split theo
   *family* (`episodes.py` `assign_families`), và drift lấy tài liệu chỉ trong split
   của chính snapshot → không rò. ✅ Cần nêu rõ đơn vị held-out trong paper.
3. **Hiệu quả chi phí tổng thể** — báo training cost, generation latency, và **số
   lần tấn công cần để hoà vốn** chi phí train. Đã có `costs.py`
   (`break_even`, p50/p95 latency) và `drift_eval.py` gắn quality vào break-even. ✅
4. **Tác động lên agent** — báo harmful-action success và false activation cạnh
   retrieval success. **Pipeline `mcat` hiện chỉ đo retrieval** — xem gap G2.

Báo cáo **tách riêng** ba nhóm số, không gộp:
- **Retrieval success** — query có trigger có lấy được poison? (`trigger_on`)
- **End-to-end success** — agent có ra answer/action mục tiêu?
- **Benign performance / false activation** — chuyện gì xảy ra khi *không* có
  trigger? (`trigger_off` cho retrieval; còn thiếu ở tầng agent)

### Gap phát hiện khi đối chiếu với code

**G1. Shuffled-context control đo "trigger có đổi không", không đo "ASR có tụt
không".** `evaluate.py:198-249` chỉ so `token_ids` trước/sau khi swap
(`changed`/`change_rate`). Docstring module (`evaluate.py:9-11`) và design doc
(§9 ablation 1: *"metric có xấu đi không?"*) đều nói phải đo **metric**. Trigger
đổi token nhưng ASR-r giữ nguyên vẫn là "generator bỏ qua memory" về mặt hiệu
quả — control hiện tại sẽ báo sai là *có* dùng memory. Cần: chấm ASR-r của trigger
sinh từ memory bị swap trên memory **gốc** của episode, và báo chênh lệch có
paired bootstrap (`stats.py`).

**G2. Không có đánh giá end-to-end trong `mcat`.** `evaluate.py` chỉ có
`trigger_on`/`trigger_off` ở tầng retrieval. Cần một bước nối trigger tốt nhất
vào agent (StrategyQA ReAct / EhrAgent) để đo ASR-a/ASR-t và false activation ở
tầng hành vi. Liên quan trực tiếp P2 (pha loãng trong trace).

**G3. Thiếu B1 (HotFlip per-episode) trong pipeline `mcat`.** Có HotFlip ở
`src/triggers/hotflip_margin.py` cho QA, nhưng không chạy được như một `--mode`
cùng episode/ngân sách. Không có B1 thì so sánh "MCAT vs. AgentPoison cùng quyền"
— baseline reviewer hỏi đầu tiên — không đứng được.

**G4. Threat model phải viết thành một bảng trong paper**, trả lời đúng ba câu
của PoisonedRAG. Design doc §4.1 đã có nội dung (white-box retriever, đọc benign
snapshot + support queries, sửa được query, B poison records, L token), nhưng cần
thêm: generator thấy **toàn bộ** snapshot hay chỉ summary/mẫu, và baseline được
cấp **cùng** quyền đó. Đối chiếu riêng với Zhong (chỉ sửa corpus).

### Cần làm
- [x] Sửa G1 (2026-10-03): `shuffled_context_control` chấm cả trigger gốc lẫn
  trigger sinh từ memory bị swap trên memory + Q_eval **gốc** của episode, báo
  `metric_effect` (drop `on_hit` / `mean_margin`, paired bootstrap theo episode) và
  verdict `memory-used | memory-inert | swap-helps | inconclusive`. Phép chấm chạy
  trong `costs.unmetered()` để không lẫn vào ledger break-even. Chỉ áp cho
  `--mode generator`; chưa có số thật.
- [x] G3: thêm B1 HotFlip như một mode của `mcat` (`--mode hotflip`, arm `b1` trong
  `scripts/run_mcat.sh`). Còn phải chạy trên DPR thật và so ngân sách qua
  `costs-train.json`, không qua `--steps`.
- [ ] G2: bước eval end-to-end trên agent — làm **sau** P0.
- [ ] G4: bảng threat model + đoạn so sánh với Zhong / PoisonedRAG.
- [ ] Đọc thêm các paper related work còn lại theo cùng khuôn: *bài học → map
  vào code → gap*.

---

## Liên quan
- `_idea_q1_aplus/trigger_hierarchy_pilot_results.md`
- `_idea_q1_aplus/group_conditioned_triggers.md`
- `_idea_q1_aplus/memory_conditioned_generator_Q1_A_star.md` (§4 threat model, §9 baselines)
- `_guidance/22_mcat_m3_drift_plan.md`
