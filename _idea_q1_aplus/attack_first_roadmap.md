# Lộ trình attack-first: MCAT trước, AQuA sau

> Lập ngày 2026-09-24. Quyết định: **làm nhánh attack (MCAT) trước, nhánh defense (AQuA) sau.**
> Liên quan: [kế hoạch MCAT](memory_conditioned_generator_Q1_A_star.md), [đề xuất AQuA](paper_Q1_A_plus.md),
> [runbook Kaggle](../_guidance/20_mcat_kaggle_runbook.md), [code MCAT](../src/triggers/mcat/README.md).
>
> File này chỉ nói **thứ tự làm và ngưỡng phải khóa trước**. Ý nghĩa nghiên cứu,
> threat model và tiêu chí go/pivot/stop đầy đủ nằm ở hai file `_idea/` kia.

## 0. Vì sao attack trước

Không phải vì attack quan trọng hơn. Vì nó **gần đích hơn nhiều**:

| | MCAT (attack) | AQuA (defense) |
|---|---|---|
| Code | M0–M3 hoàn chỉnh, 156/156 test pass | scaffold chạy được ở mức smoke |
| Thiếu gì | **chỉ thiếu số thật trên retriever thật** | thiếu benchmark AuthShift (~60% công sức cả bài) |
| Chi phí tới kết quả đầu tiên | vài giờ GPU | vài tháng |

Nhánh defense vẫn chạy song song hai việc **không cần GPU** (mục 6), nên "sau" không
có nghĩa là "đứng yên".

## 1. Trạng thái xuất phát (đã kiểm ngày 2026-09-24)

- `python -m unittest tests.test_mcat tests.test_mcat_pipeline tests.test_mcat_drift tests.test_mcat_costs tests.test_mcat_adapt` → **156/156 OK**.
- Máy local **không chạy được arm thật**: `torch 2.14.0+cpu`, `cuda False`; `.venv-adapt`
  thiếu `scikit-learn`; `.venv` không có `torch`. → mọi arm thật chạy trên Kaggle.
- DPR index có sẵn: `ReAct/database/embeddings/agentpoison_dpr/vectors.npy` (28 MB) + manifest.
- Dữ liệu: `qa` đủ dày (9 251 paragraph), `ehr` mỏng (199 record), **`ad` không có dữ liệu**.

### Điều phải khóa ngay vào manifest

Phạm vi thật là **một domain đủ dày (`qa`) + một domain mỏng (`ehr`)**. Không được
viết "three agent domains" ở bất kỳ đâu cho tới khi `ad` có dữ liệu thật.

## 2. Bước 1 — Sửa đường chặn (nửa ngày, không GPU)

- [x] **Lệch tên script.** Runbook, README gốc, `src/triggers/README.md` và header của
      chính script đều gọi `run_mcat_kaggle.sh`, nhưng file thật là `scripts/run_mcat.sh`.
      Đã sửa 21 tham chiếu trong 5 file ngày 2026-09-24. Copy-paste từ runbook giờ chạy được.
- [x] **Bảng Status của MCAT** đã ghi rõ blocker là hardware, không phải code.
- [ ] **Khóa ngưỡng go/no-go vào `config.json` của run**, không để trong đầu. Lấy từ
      mục 14 của [kế hoạch MCAT](memory_conditioned_generator_Q1_A_star.md). Ghi trước
      khi chạy, không sửa sau khi xem test.
- [ ] **Khóa domain claim** vào manifest như mục 1.

## 3. Bước 2 — Preflight trên máy local (15 phút, không GPU)

```bash
FIXTURE=1 bash scripts/run_mcat.sh preflight
```

Chạy được trên CPU, không tải model. Kiểm dependency, test suite, và smoke qua toàn
bộ pipeline. Quan trọng nhất là nó kiểm **resume guard** và **contract cờ giữa
`train` và `evaluate`** — `evaluate` dựng lại `TrainConfig` từ CLI rồi so hash với
checkpoint, lệch một cờ là bị từ chối. Đây là hai chỗ dễ làm mất một session Kaggle.

- [ ] Preflight xanh.

## 4. Bước 3 — Shakedown trên DPR thật (1–2 giờ GPU)

```bash
STEPS=50 DOMAINS="qa" bash scripts/run_mcat.sh m1
```

Một arm, một domain, 50 step. **Không phải để lấy kết quả.** Để xác nhận lần đầu
tiên code gặp retriever thật thì: DPR tải được, cache encode đúng, gradient chảy tới
generator và set encoders, export ra text thật được.

Nhắc: bỏ `--fixture` thì reference centers đi qua `src.triggers.clustering.fit_centers`
(GMM 5 component full-covariance, `random_state=0`) — cùng geometry với baseline
AgentPoison. Fixture lấy `memory[:5]`, **số của fixture không phải kết quả**.

- [ ] Shakedown hoàn tất, không lỗi.

## 5. Bước 4 — GATE: đọc round-trip gap (10 phút đọc file)

**Đây là việc quan trọng nhất của cả tuần và nó gần như miễn phí.**

Generator tối ưu trên phân phối token liên tục, nhưng trigger thật phải là **chữ**:
`argmax` → decode → tokenize lại theo đúng tokenizer runtime. Nếu lợi thế biến mất
trong bước đó thì mọi thứ phía sau vô nghĩa. Đây là RQ4 trong kế hoạch MCAT.

Code đã có sẵn `round_trip_report` trong `src/triggers/mcat/relaxation.py`; artifact
lưu cả optimized IDs, decoded text và re-encoded IDs.

Ngưỡng phải khóa **trước khi xem**:

- [ ] **Gap nhỏ** → đi tiếp bước 5.
- [ ] **Gap lớn** → **dừng**, sửa discretization trước khi tiêu thêm một giờ GPU.
      Không được "chạy thêm vài arm xem sao".

Lý do xếp gate này trước tất cả: rẻ nhất, và có khả năng giết cả hướng.

## 6. Bước 5 — Nếu gate qua: full 7 arm (1 seed trước)

```bash
bash scripts/run_mcat.sh all
```

**Chưa chạy 3 seed.** Một seed đủ để biết cơ chế có tồn tại hay không.

Đọc kết quả theo thứ tự này, **không** theo thứ tự trong file báo cáo:

1. **m1 so với b3** (universal logit) **và b7** (nearest-context trigger bank).
   Nếu m1 không thắng được hai cái này trên held-out episode → conditioning vô giá trị,
   bài tụt thành optimizer study. Đây là **điều kiện sống chết của claim**, không phải
   ablation phụ.
2. **Shuffled-context control.** Tráo memory summary giữa các episode cùng kích thước,
   giữ nguyên query. Nếu metric không xấu đi → generator chỉ đang nhớ một lời giải chung.
3. Đối xứng: giữ memory, đổi query distribution (b5 vs b6).

- [ ] m1 > b3 và b7, có paired CI.
- [ ] Shuffled-context làm metric xấu đi rõ rệt.
- [ ] Chỉ khi hai ô trên xanh: chạy seed 1 và 2.

## 7. Bước 6 — M4: từ retrieval sang hành vi (2–3 tuần)

Retrieval hit@K **không phải** security outcome. Đây là chỗ bài sống hoặc chết với
reviewer security.

Hai chỗ phải sửa theo [Influence Is Not Authority](https://arxiv.org/abs/2608.29942):

- [ ] **Đo ở kết cục, không đo ở cửa.** Họ chứng minh agent bị chặn thì thử đường khác:
      chặn 13 lệnh trái phép → 12/13 agent sau đó vẫn hoàn thành đúng việc. Nếu MCAT
      báo "trigger làm agent gọi tool X" rồi dừng, con số đó không phải ASR.
- [ ] **Đúng tên tool không phải thành công.** 22/23 ca sai trong bài đó vẫn gọi đúng
      tên tool, chỉ đổi tham số, và benchmark vẫn tính pass. Behavioral ASR phải check
      **tham số và hậu quả**, không phải tên tool.

Cộng các mục đã có trong kế hoạch MCAT:

- [ ] Đủ bốn điều kiện: memory sạch/độc × trigger tắt/bật, cùng base query và seed.
      Đây là cách duy nhất chứng minh trigger không tự gây behavior khi không có poison.
- [ ] False activation trên query không trigger; clean task success.
- [ ] Human audit mẫu stratified cho meaning preservation. PPL không thay được.
- [ ] Gộp kết quả specificity đã có (cosine 0,6264; PPL 0,598×; 48/48 qua proxy) vào đây
      làm baseline **"query-conditioned selection"** mà generator phải vượt.
      Xem [increase_specificity_results.md](increase_specificity_results.md).

## 8. Bước 7 — Break-even và viết bài (2–3 tuần)

Trục bán bài không phải "generator sinh trigger tốt hơn" mà **amortization economics**:

```
N* = C_train / (C_online_search − C_online_gen)
```

Sau bao nhiêu episode thì generator rẻ hơn tối ưu lại từ đầu. `costs.py` đã đo đủ counter.

- [ ] So ở **cùng mức chất lượng**. Generator nhanh nhưng ASR thấp hơn nhiều không được
      báo là speedup.
- [ ] Báo cả deployment-only và lifecycle cost; hai chế độ fairness (cùng online budget,
      cùng tổng lifetime budget).

## 9. Nhánh defense chạy song song (không cần GPU)

Hai việc này làm được ngay, lúc chờ Kaggle, không tranh tài nguyên:

- [ ] **Đọc AttriGuard** ([2603.10749](https://arxiv.org/abs/2603.10749)). Baseline bắt buộc,
      không còn tùy chọn: Influence Is Not Authority đã audit nó và công bố chỗ nó thất bại
      (block 47,1% hành động hợp lệ vs 36,3% trái phép, CI chứa 0). Lấy đúng số đó làm
      đường cơ sở. **Pin commit** — họ đã responsible disclosure một audit-coverage bug nên
      phiên bản đã test khác phiên bản hiện tại.
- [ ] Rồi AuthGraph, PACT, AgentLens, PRISMS (Tier 1, mục 16 của file AQuA).
- [ ] Sau khi đọc xong Tier 1 mới viết lại mục 4 và 13 của [paper_Q1_A_plus.md](paper_Q1_A_plus.md).

### Thí nghiệm 24-base: chạy kèm khi đã có session Kaggle

Tốn thêm khoảng **72 forward pass** — vài phút, không đáng kể so với một arm MCAT.
Nhưng nó định hình cả nhánh defense, nên đừng bỏ.

Dựng lại 24 tình huống của Influence Is Not Authority. Mỗi tình huống ba phiên bản:
(1) user đưa dữ liệu, (2) tool đưa cùng dữ liệu đó, (3) đổi recipient sang kẻ tấn công.
Đo khoảng cách **thô trong activation space, chưa học gì cả**, qua nhiều layer:

| Kết quả | Nghĩa là gì |
|---|---|
| `d(1,3) > d(1,2)` | Bên trong model đã có tín hiệu phân quyền tương đối sạch. AQuA có nền vững |
| `d(1,2) > d(1,3)`, hướng nhất quán | Khả năng cao nhất. Bài vẫn làm được nhưng **trọng tâm dịch**: contribution chính thành "gỡ được lớp nhiễu provenance", không phải "tìm ra subspace" |
| `d(1,2) > d(1,3)`, lộn xộn | Không chiếu bỏ được → chuyển sang audit paper (đường lùi ở cuối mục 10 file AQuA) |

Lý do phải làm **trước** khi xây AuthShift: ba kết cục dẫn tới ba bài khác nhau, và
AuthShift là ~60% công sức. Bài 1 đã đo được ở mức hành vi rằng source relocation dịch
tín hiệu **xa hơn** authorization flip — tức tiêu chí thất bại số 2 của AQuA là **tình
trạng mặc định**, không phải rủi ro xa. Đây là bằng chứng gián tiếp mạnh, không phải
chứng minh về activation, nhưng đủ để đổi thứ tự thí nghiệm.

Tin tốt: hiệu ứng đó **rất hệ thống** (24/24 cùng chiều, specificity check quy 99,7%
chuyển động về đúng một giá trị đã đổi nguồn). Nhiễu hệ thống thì một phép chiếu
low-rank có thể trừ đi được; nhiễu lộn xộn thì không.

## 10. Việc KHÔNG làm

- **Đừng khởi động AuthShift** cho tới khi có kết quả 24-base.
- **Đừng đọc Tier 2 / Tier 3** (mục 16 file AQuA). Chưa tới lúc, sẽ phải đọc lại.
- **Đừng chạy 3 seed** trước khi gate round-trip và gate cơ chế đều xanh.
- **Đừng mở rộng sang `ad` domain** khi chưa có dữ liệu thật.
- **Đừng ghép co-evolution ADAPT** ở phiên bản này. Đó là v2 hoặc journal extension, và
  khi làm thì defender phải được test bằng generator checkpoint **không tham gia train**.

## 11. Ghi chú trung thực

Mọi ô `[ ]` ở trên chưa làm. Hai ô `[x]` ở mục 2 đã làm ngày 2026-09-24 và chỉ là sửa
tài liệu, không phải kết quả nghiên cứu.

Chưa có một con số MCAT nào trên retriever thật. Không được mô tả bất cứ điều gì trong
file này là kết quả cho tới khi bước 4 và bước 5 hoàn tất.
