# 25 — Helper tổng: chạy cái gì, ra cái gì, nằm ở đâu

File này là **bản đồ** của repo cho người không muốn đọc code. Nó trả lời các câu:
thư mục nào chứa gì; pipeline nào có những stage nào; mỗi stage cần gì và sinh ra file gì;
config ở đâu; gặp lỗi thì làm gì.

Trọng tâm là **MCAT + các probe P0**, tức phần đang chạy hằng ngày (lần chạy 1–7 trong
[`open_problems.md`](../_idea_q1_aplus/open_problems.md)). Các nhánh khác chỉ được liệt kê ở
§10, kèm link tới tài liệu riêng.

Cập nhật: 2026-10-07 (commit `b01f502a`). Mọi lệnh, tên file và giá trị mặc định dưới đây
được đối chiếu với code ở commit đó. Nếu code đổi, nguồn sự thật là
`src/triggers/mcat/cli.py` (cờ CLI) và `scripts/run_p0_probes.sh` (biến môi trường).

---

## 0. Tôi muốn… → đọc mục nào

| Tôi muốn | Mục |
|---|---|
| Biết thư mục nào chứa gì, cái nào được đụng vào | §1 |
| Hiểu các từ "episode", "s0", "Q_eval", "hit@k", "ASR-t"… | §2 |
| Cài môi trường, chạy test hoặc thử nhanh trên máy mình | §3 |
| Đổi một tham số (số tài liệu, top-k, số token trigger, LLM…) | §4 |
| Chạy pipeline MCAT chính (train/evaluate trigger) | §5 |
| Chạy một probe P0 (growth, contamination, write-back, end-to-end) | §6 |
| Đưa một lần chạy lên Kaggle, lấy kết quả về | §7 |
| Biết một file output nằm ở đâu và đọc key nào | §8 |
| Biết quy trình "chốt tiêu chí → chạy → viết kết quả" | §9 |
| Chạy nhánh khác (margin, AQuA, hierarchy…) | §10 |
| Gặp lỗi | §11 |

### Bức tranh một trang

![Bức tranh một trang: ba tầng gọi nhau, pipeline MCAT, các probe P0, output và file kết quả](assets/25_tong_quan.svg)

Nguồn sửa được: [`assets/25_tong_quan.drawio`](assets/25_tong_quan.drawio) (mở bằng
[app.diagrams.net](https://app.diagrams.net) hoặc extension *Draw.io Integration* của VS Code;
sửa xong thì *File → Export as → SVG* đè lên `25_tong_quan.svg`).

Cách đọc hình: cột trái là **ai gọi ai**. Ba tầng gọi nhau: **Kaggle kernel** (`run.py`) gọi **script shell**
(`scripts/run_p0_probes.sh`), script gọi **CLI Python** (`python -m src.triggers.mcat <stage>`).
Bạn có thể vào ở bất kỳ tầng nào; tầng dưới không biết tầng trên tồn tại. Khung xanh dương là
các stage của pipeline MCAT, mỗi ô ghi file nó sinh ra. Khung xanh lá là các probe: chúng dùng lại
episode + index của MCAT; tên nhỏ bên dưới (`r2`, `contam`…) là tên gọi trong `run_p0_probes.sh`.
Hàng dưới cùng là nơi kết quả đi tới.

---

## 1. Bản đồ thư mục

### 1.1 Thư mục gốc

| Thư mục / file | Chứa gì | Có cần đụng không |
|---|---|---|
| `src/triggers/mcat/` | **Code chính đang dùng**: MCAT, các probe P0, end-to-end | có: đây là nơi sửa code |
| `src/triggers/` (phần còn lại) | các nhánh tấn công khác: `margin.py`, `hierarchy/`, `specificity/`, `hotflip_margin.py`; cùng các helper dùng chung `artifacts.py`, `clustering.py`, `losses.py` | chỉ khi làm nhánh đó |
| `src/aqua/`, `src/adapt/` | phía phòng thủ (chạy độc lập, chưa nối với MCAT) | không, với P0 |
| `src/agentpoison/`, `src/toolpoison/`, `src/integration/` | demo AgentPoison và các alias cũ | không |
| `algo/` | code AgentPoison upstream (HotFlip, nạp model và DB) | không sửa; `src/` import nó |
| `agentdriver/` | agent lái xe AgentPoison (agent `ad`) + **dữ liệu** `data/finetune/data_samples_{train,val}.json` | chỉ để lấy dữ liệu |
| `EhrAgent/` | agent y tế AgentPoison (agent `ehr`), dữ liệu `database/ehr_logs/` | đọc khi làm write-back thật cho EHRAgent |
| `ReAct/` | agent hỏi đáp StrategyQA (agent `qa`) + dữ liệu `database/` (kèm index DPR dựng sẵn ở `database/embeddings/agentpoison_dpr`) | chỉ để lấy dữ liệu |
| `scripts/` | script shell chạy sẵn. Quan trọng nhất: `run_p0_probes.sh` (probe P0) và `run_mcat.sh` (pilot MCAT) | có: chạy, đôi khi thêm bước |
| `tests/` | unit test (`test_mcat_*.py` là của MCAT) | chạy trước mỗi lần đẩy Kaggle |
| `.kaggle/` | mỗi kernel Kaggle là một thư mục `run.py` + `kernel-metadata.json`; `access_token`. **Cả thư mục bị gitignore**: chỉ có trên máy này | có: đóng gói, đẩy kernel |
| `outputs/` | mọi artifact (local và tải từ Kaggle về). Gitignore | đọc kết quả |
| `_idea_q1_aplus/` | **sổ nghiên cứu**: tiêu chí chốt trước, file kết quả từng lần chạy, literature | có: ghi chép |
| `_guidance/` | hướng dẫn vận hành (file này) | đọc |
| `configs/` | chỉ có `trigger_seed.json` (một trigger khởi tạo cũ cho `qa`); không file code nào đọc nó. MCAT **không** đọc file config nào: config của nó là cờ CLI (§4) | không |
| `.env` / `.env.example` | API key cho các nhánh dùng LLM qua API (OpenAI, DeepSeek, Langfuse…). MCAT chỉ cần `HF_TOKEN` (tuỳ chọn, cho `probe-e2e`) | khi dùng nhánh API |
| `survey/` | paper tham khảo | — |
| `make.ps1` | bí danh `python -m ...` trên Windows (`.\make.ps1 help`). **Đang hỏng**: trỏ tới `.venv-adapt/` đã bị xoá. Gọi thẳng `py -3.11 -m src.triggers.mcat ...` thay thế | không |
| `requirements*.txt`, `environment.yml` | phụ thuộc | khi cài máy mới |
| `adapt.ipynb`, `adapt_tracing.py`, `sanity_check.py`, `trick.txt`, `git` (file rỗng) | đồ cũ / nháp | không |

### 1.2 Bên trong `src/triggers/mcat/`: file nào làm gì

Đọc theo **luồng dữ liệu**, từ trên xuống:

| Tầng | File | Vai trò một câu |
|---|---|---|
| Lối vào | `__main__.py`, `cli.py` | `python -m src.triggers.mcat <stage>`; mỗi stage là một hàm trong `cli.py` |
| Dữ liệu | `domains.py` | đọc 3 domain `qa` / `ehr` / `ad` thành `documents` (memory) + `queries`, kèm `family` để chia split |
| | `episodes.py` | cắt thành **episode** (memory + các tập query), chia train/validation/test theo family |
| | `cache.py` | encode toàn corpus một lần, lưu vector vào `cache/` |
| | `retrievers.py` | nạp DPR (hoặc encoder fixture chạy CPU) |
| | `runtime.py` | `Workspace`: gom domain + cache + retriever; `context_for(episode)` trả tensor cho một episode |
| Tấn công | `encoding.py` | ghép trigger vào query ở vị trí `suffix`/`prefix`/`middle`/`both`/`both-repeat` rồi encode |
| | `objectives.py`, `relaxation.py` | loss hit@K-margin và Gumbel-softmax để tối ưu trigger liên tục, rồi export về token thật |
| | `generator.py` | generator sinh trigger theo trạng thái memory (ý tưởng gốc của MCAT) |
| | `hotflip.py` | baseline HotFlip của AgentPoison |
| | `train.py` | vòng tối ưu, 4 mode: `direct-logit` (B2, mỗi episode một trigger), `universal-logit` (B3, một trigger chung), `generator` (M1, B4–B6), `hotflip` (B1) |
| | `poison.py` | ghi poison **một lần** ở `s0` (`poison_s0.pt`), sau đó đóng băng |
| Đo | `evaluate.py` | `retrieval_metrics`: hit@1/2/3/5, `on_hit`/`off_hit`, false activation; hai control cơ chế (shuffled context, permutation) |
| | `stats.py` | paired bootstrap theo episode, sign-flip p-value, Holm |
| | `costs.py` | đếm chi phí (thời gian, số forward) → `costs*.json` |
| Drift (M3) | `drift.py`, `drift_eval.py`, `adapt.py` | dựng snapshot memory phình/trộn/xoá; 4 cách thích nghi `reuse`/`generate`/`warm-start`/`scratch` |
| Probe P0 | `probes.py` | `probe-growth` (R2), `probe-position` (P1), `probe-universal` (R1) |
| | `contamination.py` | `probe-contamination`: bản ghi mang trigger lọt vào memory (`self` / `rival`) |
| | `writeback.py` | `probe-writeback`: agent tự ghi lại tương tác theo chính sách ghi (vòng kín, chỉ đo truy hồi) |
| End-to-end | `e2e.py` | `probe-e2e`: agent **thật** đọc top-k, LLM ra hành động, chấm ASR-t |
| | `agent_ad.py` | prompt, parse `Driving Plan`, chấm đáp án của agent AgentDriver |
| | `llm.py` | backend LLM (`hf` / `vllm` / `fixture`) + cache câu trả lời `llm_cache.jsonl` |

### 1.3 `scripts/`

| Script | Làm gì | Mục |
|---|---|---|
| `run_p0_probes.sh` | chạy các probe P0: `preflight`, `r2`, `r1`, `position`, `contam`, `writeback`, `e2e`, `all` | §6 |
| `run_mcat.sh` | pilot MCAT M0–M3: các arm `m1 b1 … b6 margin`, tuỳ chọn drift | §5.3 |
| `run_specificity_ap.sh`, `run_ablation_sweep.sh` | nhánh specificity / ablation AgentPoison | §10 |
| `agent_driver/`, `ehragent/`, `react_strategyqa/` | script gốc của AgentPoison cho từng agent | §10 |
| `show_trigger.py`, `show_bank_trigger.py`, `evaluate_trigger_full.py`, … | tiện ích lẻ cho nhánh AgentPoison | — |
| `gpu_clean.sh` | dọn tiến trình GPU treo | — |

### 1.4 `_idea_q1_aplus/` (sổ nghiên cứu)

| File | Vai trò |
|---|---|
| `open_problems.md` | **file trung tâm**: các rủi ro P0–P4, tiêu chí chốt TRƯỚC mỗi lần chạy, link tới kết quả |
| `p0_falsification_results.md` | kết quả lần 1 (qa) |
| `p0_run2_agentdriver_results.md` | lần 2 (ad, 6 token) |
| `p0_run3_targeted_growth_results.md` | lần 3 (growth có chủ đích) |
| `p0_run4_contamination_results.md` | lần 4 (contamination) |
| `p0_run6_rerun_results.md` | lần 6 (gồm cả lần 5) |
| `memory_conditioned_generator_Q1_A_star.md`, `attack_first_roadmap.md`, `paper_Q1_A_plus.md` | kế hoạch phương pháp và paper |
| `memory_poisoning_literature_2026.md` | literature |

---

## 2. Từ vựng (đọc một lần là đủ)

| Từ | Nghĩa |
|---|---|
| **domain / agent** | `qa` = ReAct-StrategyQA, `ehr` = EHRAgent, `ad` = AgentDriver. Từ lần chạy 2 trở đi mọi thứ chạy trên `ad` |
| **family** | nhóm các bản ghi cùng gốc (paraphrase của một task). Chia split theo family, không theo dòng, để không rò rỉ |
| **episode** | một bài toán tấn công: `E = (D, Q_sup, Q_opt, Q_eval, P, B, L)` — memory `D` (`--documents` tài liệu), 3 tập query rời nhau, nguồn poison, ngân sách poison `B`, độ dài trigger `L` |
| `Q_sup` | query dùng làm *ngữ cảnh* (generator nhìn vào; growth có chủ đích xếp hạng theo nó) |
| `Q_opt` | query dùng để *tối ưu* trigger (loss) |
| `Q_eval` | query **khoá** cho tới khi trigger đóng băng; mọi con số báo cáo đo trên tập này |
| **split** | `train` / `validation` / `test`. Probe chạy trên `test` (`PROBE_SPLIT`) |
| **s0** | trạng thái memory gốc, lúc kẻ tấn công tìm trigger và ghi poison. Sau s0 trigger + poison **đóng băng** |
| **base-search** | kết quả tìm trigger ở s0 của từng episode (`base-search/<episode>-s0/`). Dùng lại được giữa các probe và giữa các kernel (`BASE_SEARCH_DIR`) |
| **arm** | một cấu hình tấn công: `b2` = direct-logit mỗi episode, `b3` = universal, `m1` = generator… Thư mục `RUN_ROOT/<arm>/seed_<SEED>/` |
| **hit@k** | tỉ lệ query có ít nhất một poison trong top-k (ASR-r ở k). Ghi `hit_1`, `hit_2`, `hit_3`, `hit_5` |
| `on_hit` / `off_hit` | hit@K khi query **có** / **không** trigger. `off_hit` cao = kích hoạt nhầm |
| **ASR-r / ASR-a / ASR-t** | truy hồi trúng / hành động độc khi đã truy hồi trúng / hành động độc vô điều kiện (end-to-end). Chi tiết: `02_evaluation.md` |
| **false activation** | hành động độc (hoặc truy hồi poison) khi query **không** có trigger |
| **drift seed** | một lần bốc lại tài liệu lành được thêm vào. Nhiều seed = đo độ dao động (`seed_spread`) |
| **write policy** | agent ghi gì vào memory sau mỗi tương tác: `none`, `log_outcome`, `verified`, `corrected` (§6.5) |
| **contract / hash** | mỗi run lưu hash của config + split + retriever. Đổi một cờ mà chạy tiếp vào thư mục cũ → bị **từ chối** (cố ý) |
| **fixture** | encoder BERT ngẫu nhiên chạy CPU, để thử ống nước. **Số của nó không phải kết quả** |
| **verdict** | kết luận tự động của mỗi probe, theo luật đã chốt trước (§8.2) |

---

## 3. General: môi trường và chạy trên máy mình

### 3.1 Python nào

| Môi trường | Có gì | Dùng cho |
|---|---|---|
| `.venv/` | **không** có torch / transformers; có `kaggle.exe` | gọi Kaggle CLI |
| Python hệ thống 3.11 (`py -3.11`) | torch 2.14 **CPU** | chạy test, smoke fixture |
| Kaggle (T4×2) | torch 2.10+cu128, transformers ≥ 5.3 (tự cài), GPU | **mọi con số thật** |

Máy này không có CUDA, nên số thật chỉ ra từ Kaggle. Trên máy chỉ kiểm tra code có chạy.

Phụ thuộc bắt buộc cho run thật: `torch`, `transformers>=5.3` (5.0–5.2 không lowercase DPR,
`load_dpr` sẽ dừng ngay), `numpy`, `scikit-learn` (GMM reference centers). Fixture không cần sklearn.

### 3.2 Luôn chạy từ thư mục gốc repo

Code dùng đường dẫn tương đối; `run_p0_probes.sh` tự kiểm tra và dừng nếu không thấy
`src/triggers/mcat/cli.py`.

### 3.3 Chạy test (≈1 phút)

```bash
py -3.11 -m pytest tests/test_mcat*.py          # toàn bộ test MCAT
py -3.11 -m pytest tests/test_mcat_e2e.py       # chỉ một file
```

| Test | Kiểm tra |
|---|---|
| `test_mcat.py`, `test_mcat_pipeline.py` | episode, encoding, train/evaluate, contract resume |
| `test_mcat_probes.py`, `test_mcat_hit_curve.py` | probe growth/position/universal, hit@1/2/3/5 |
| `test_mcat_contamination.py`, `test_mcat_writeback.py` | probe contamination, write-back |
| `test_mcat_e2e.py` | probe end-to-end với LLM fixture |
| `test_mcat_drift.py`, `test_mcat_adapt.py`, `test_mcat_costs.py` | M3 drift, thích nghi, chi phí |

### 3.4 Thử nhanh cả ống nước (CPU, không tải gì)

```bash
# mọi stage của pipeline MCAT trên fixture, kèm kiểm tra resume guard (~6 phút trên máy này;
# xong khi in "smoke completed")
py -3.11 -m src.triggers.mcat smoke --output-dir outputs/mcat/smoke

# preflight của probe: kiểm tra thư viện + unit test + smoke 3 probe trên fixture
tr -d '\r' < scripts/run_p0_probes.sh > /tmp/p0.sh       # file .sh trên Windows là CRLF
PYTHON="py -3.11" FIXTURE=1 DEVICE=cpu bash /tmp/p0.sh preflight
```

Lưu ý `PYTHON` phải không chứa dấu cách (script không đặt nó trong ngoặc kép), nên dùng
`py -3.11` chứ không dùng đường dẫn `C:/Users/TUAN ANH/...`.

---

## 4. Config: tham số nằm ở đâu, đổi ở đâu

MCAT **không có file config**. Tham số đi qua ba tầng, tầng trên ghi đè tầng dưới:

```
.kaggle/<kernel>/run.py: COMMON_ENV + env của từng step     (tầng 3: Kaggle)
        │ đặt biến môi trường
        ▼
scripts/run_p0_probes.sh: VAR="${VAR:-mặc định}"             (tầng 2: script)
        │ dịch thành cờ
        ▼
src/triggers/mcat/cli.py: add_common(): --flag default=…      (tầng 1: CLI)
```

Muốn đổi một tham số cho một lần chạy: đặt biến môi trường trước lệnh script, ví dụ
`TOP_K=3 bash scripts/run_p0_probes.sh r2`. Muốn đổi cho kernel: sửa `COMMON_ENV` hoặc env
của step trong `run_template.py` / `build.py` rồi build lại `run.py`.

### 4.1 Bảng tham số chính

Cột "Lần 2–7" là giá trị các kernel trên `ad` đã dùng (giống nhau ở lần 2, 6, 7; lần 1 chạy `qa`
với `PER_SPLIT=8`, còn lại theo mặc định script).

| Biến (script) | Cờ CLI | Mặc định script | Mặc định CLI | Lần 2–7 | Ý nghĩa |
|---|---|---|---|---|---|
| `RUN_ROOT` | (gốc của `--output-dir`) | `outputs/mcat/p0` | `outputs/mcat/run` | `/kaggle/working/mcat_p0_*` | thư mục gốc của lần chạy |
| `CACHE_DIR` | `--cache-dir` | `$RUN_ROOT/cache` | `<output-dir>/cache` | `/tmp/mcat_cache` | cache vector corpus |
| `SEED` | `--seed` | 0 | 0 | 0 | seed chia split / episode |
| `DOMAINS` | `--domain` (lặp) | `qa` | `qa` | `ad` | agent |
| `DEVICE` | `--retriever-device` | `cuda:0` | `cuda:0` | `cuda:0` | GPU cho DPR |
| `MODEL`, `REV` | `--retriever-model`, `--retriever-revision` | DPR ctx-encoder NQ, revision ghim `bb21a3c2…` | DPR, không ghim | như script | retriever |
| `PER_SPLIT` | `--per-split` | 4 | 4 | 16 | số episode mỗi split |
| `DOCUMENTS` | `--documents` | 512 | 512 | 2000 | kích thước memory mỗi episode |
| `SUPPORT` / `OPTIMIZATION` / `EVALUATION` | `--support` / `--optimization` / `--evaluation` | 32 / 64 / 128 | như script | 32 / 16 / 1000 | kích thước `Q_sup` / `Q_opt` / `Q_eval` |
| `POISON_COUNT` | `--poison-count` | 5 | 5 | 5 | số bản ghi poison ở s0 |
| `TRIGGER_TOKENS` | `--trigger-tokens` | 10 | 10 | 6 | độ dài trigger |
| `TOP_K` | `--top-k` | 5 | 5 | 5 | agent đọc bao nhiêu bản ghi |
| `MAX_LENGTH` | `--max-length` | 256 | 512 | 512 | độ dài tối đa khi encode (key `ad` dài ~580 token ở p90) |
| `INDEX_BATCH` | `--index-batch-size` | 64 | 32 | 64 | batch khi encode corpus |
| `CORPUS_LIMIT` | `--corpus-limit` | rỗng | không giới hạn | rỗng | cắt corpus (ghi vào manifest) |
| `STEPS` | `--steps` | 400 | 200 | 400 | số bước tối ưu trigger |
| `LR`, `TAU_START`, `TAU_END` | `--learning-rate`, `--tau-start`, `--tau-end` | 1e-3, 2.0, 0.5 | như script | như script | tối ưu Gumbel |
| `LAMBDA_CPT`, `MARGIN` | `--lambda-cpt`, `--margin` | 0.1, 0.1 | như script | như script | trọng số loss |
| (cố định) | `--lambda-ret` | 0.0 | 0.0 | 0.0 | hinge truy hồi, tắt |
| `POISON_MODE` | `--poison-mode` | `refresh` | `refresh` | `refresh` | lúc **train**: poison có được viết lại không |
| `PROBE_SPLIT` | `--split` | `test` | `test` | `test` | split đo |
| `BOOTSTRAP` | `--bootstrap-iterations` | 10000 | 10000 | 10000 | số lần bootstrap |
| `BASE_SEARCH_DIR` | `--base-search-dir` | rỗng | không | `/tmp/base-search` | dùng chung tìm kiếm s0 |
| `RESUME` | `--resume` | 0 | tắt | — | chạy tiếp vào thư mục cũ |
| `FIXTURE` | `--fixture` | 0 | tắt | 1 chỉ ở preflight | encoder CPU giả |
| `SKIP_PREFLIGHT` | — | 0 | — | 1 (kernel tự chạy preflight riêng) | bỏ preflight |

Biến riêng của từng probe nằm ở §6.

### 4.2 Luật "contract" — vì sao đổi tham số lại bị từ chối

Mỗi thư mục run và mỗi thư mục probe lưu hash của config (`manifest.json`,
`checkpoint.pt["contract"]`, `probes/<…>/probe_config.json`). Khi chạy lại vào cùng thư mục với
**bất kỳ** cờ episode/train/probe nào khác, code dừng với `resume refused` / `probe refused`.
Đây là cố ý: không trộn hàng của hai cấu hình vào một file.

Hệ quả thực tế:

- Đổi tham số → đổi `RUN_ROOT` (hoặc xoá thư mục cũ nếu nó vô giá trị).
- Cờ train phải giống hệt giữa `train`, `evaluate` và các probe đọc cùng checkpoint. Script đã
  gom chúng vào `TRAIN_FLAGS` một chỗ; khi gọi CLI tay, chép đủ bộ cờ.
- Tên thư mục probe đã chứa sẵn các biến hay đổi (`selection`, `scenario`, `seed_poison`,
  `trigger-position`, `split`), nên các biến đó **không** cần đổi `RUN_ROOT`.

---

## 5. Pipeline A — MCAT chính (train / evaluate trigger)

Dùng khi muốn **huấn luyện** một arm (generator hay baseline) và đo trên split giữ lại.
Các probe P0 (§6) dùng lại hai stage đầu của pipeline này.

### 5.1 Các stage

Mọi stage đều là `python -m src.triggers.mcat <stage> --output-dir <DIR> [cờ chung]`.
Mỗi stage khi xong ghi `<DIR>/stages/<stage>.json` với `"state": "completed"`.

| # | Stage | Cần có trước | Làm gì | Sinh ra (trong `<DIR>/`) |
|---|---|---|---|---|
| 1 | `prepare-episodes` | dữ liệu domain | đọc domain, chia family → split, cắt episode | `manifest.json` (hash split, môi trường, git), `episodes.jsonl` |
| 2 | `index` | 1 | encode toàn corpus + query một lần (với `qa` dùng lại index DPR dựng sẵn nếu khớp) | vector trong `--cache-dir`; `stages/index.json` |
| 3 | `train` | 1, 2 | tối ưu trigger theo `--mode` | `checkpoint.pt`, `training.json`, `metrics.jsonl`, `triggers.jsonl`, `costs-train.json` |
| 4 | `evaluate` | 3 | đo trên `--split`; mode mỗi-episode (`direct-logit`, `hotflip`) tối ưu lại trên split đó, ghi vào `adapt-<split>/` | `evaluation.json` (tổng), `evaluation.jsonl` (từng episode) |
| 5 | `report` | 3, 4 | tóm tắt thành markdown | `REPORT.md` |
| — | `smoke` | không | chạy 1→5 + drift trên fixture, kiểm tra resume guard | thư mục smoke |

Stage M3 (drift), chỉ khi làm memory drift: `prepare-drift` → `adapt` → `evaluate-drift`,
artifact trong `<DIR>/trajectories.jsonl` và `<DIR>/drift/<write-policy>-steps<N>-<split>/`
(`drift_evaluation.jsonl`, `drift_evaluation.json`, `costs.json`). Kế hoạch: `22_mcat_m3_drift_plan.md`.

### 5.2 Các mode / arm

| Arm (`run_mcat.sh`) | Cờ | Câu hỏi |
|---|---|---|
| `b1` | `--mode hotflip` | HotFlip của AgentPoison, mỗi episode một lần tìm |
| `b2` | `--mode direct-logit` | tối ưu gradient thuần, mỗi episode |
| `b3` | `--mode universal-logit` | một trigger chung đã đủ chưa |
| `b4` / `b5` / `b6` | `--mode generator --variant none / query / memory` | generator bỏ bớt nguồn thông tin |
| `m1` | `--mode generator --variant memory+query` | phương pháp |
| `margin` | `m1` + `--lambda-ret` > 0 | ablation hinge truy hồi |

### 5.3 Chạy

```bash
# gói sẵn: preflight + các arm, mỗi arm ở $RUN_ROOT/<arm>/seed_$SEED/
bash scripts/run_mcat.sh                     # preflight + m1
bash scripts/run_mcat.sh m1 b3               # chọn arm
RESUME=1 bash scripts/run_mcat.sh all        # chạy tiếp sau khi Kaggle cắt 12h
DOMAINS=ad MAX_LENGTH=512 OPTIMIZATION=16 bash scripts/run_mcat.sh m1

# hoặc tay, từng stage
py -3.11 -m src.triggers.mcat prepare-episodes --output-dir outputs/mcat/pilot --domain qa
py -3.11 -m src.triggers.mcat index    --output-dir outputs/mcat/pilot --domain qa
py -3.11 -m src.triggers.mcat train    --output-dir outputs/mcat/pilot --domain qa --mode generator
py -3.11 -m src.triggers.mcat evaluate --output-dir outputs/mcat/pilot --domain qa --mode generator --split test
py -3.11 -m src.triggers.mcat report   --output-dir outputs/mcat/pilot
```

Mặc định của `run_mcat.sh`: `RUN_ROOT=outputs/mcat/pilot`, `DOMAINS="qa ehr"`. Chi tiết chạy trên
Kaggle: `20_mcat_kaggle_runbook.md`. README code: `src/triggers/mcat/README.md`.

**Trạng thái**: pipeline này mới chạy pilot; các kết luận hiện tại của đề tài đều đến từ probe P0 (§6).

---

## 6. Pipeline B — các probe P0 (`scripts/run_p0_probes.sh`)

Probe là phép đo **rẻ, để bác bỏ**: trigger tìm ở s0 rồi đóng băng, sau đó thay đổi **một**
thứ (memory phình, bản ghi mang trigger, agent tự ghi lại…) và xem tấn công còn sống không.

### 6.1 Cú pháp

```bash
bash scripts/run_p0_probes.sh preflight          # chỉ kiểm tra (CPU), không đụng GPU
bash scripts/run_p0_probes.sh r2                 # một probe (mặc định nếu không truyền gì)
bash scripts/run_p0_probes.sh r2 contam e2e      # nhiều probe, chạy lần lượt
bash scripts/run_p0_probes.sh all                # = r2 r1 position
RESUME=1 bash scripts/run_p0_probes.sh r2        # chạy tiếp
```

Tên hợp lệ: `r2`, `r1`, `position`, `contam`, `writeback`, `e2e`, `all`, `preflight`.

### 6.2 Mỗi lệnh làm gì bên trong

```
run_p0_probes.sh <probe…>
 ├─ preflight (trừ khi SKIP_PREFLIGHT=1)
 │    ├─ kiểm tra torch / transformers / numpy / sklearn, CUDA
 │    ├─ unit test probes+pipeline   → $RUN_ROOT/_preflight-tests.log
 │    └─ smoke fixture growth+position → $RUN_ROOT/_smoke/, _preflight-smoke.log
 ├─ với từng probe:
 │    ├─ prepare_arm b2:  prepare-episodes → index   (→ train chỉ khi probe cần checkpoint)
 │    ├─ python -m src.triggers.mcat probe-<…>       (log → $RUN_ROOT/<tên>.console)
 │    └─ in "done -> <đường dẫn summary>"
 └─ summarize: in verdict của r2 / contam / r1 / position đã có trong $RUN_ROOT
```

Mọi probe (trừ `r1`) dùng chung một thư mục arm `$RUN_ROOT/b2/seed_$SEED/`: episode và index
chỉ tính một lần. Probe ghi vào thư mục con riêng `b2/seed_$SEED/probes/<tên>-suffix-<split>/`.

Mỗi thư mục probe có cùng bộ file:

| File | Nội dung |
|---|---|
| `probe_config.json` | contract của probe (khoá cấu hình, §4.2) |
| `probe_<x>.jsonl` | **hàng thô**: một dòng mỗi (episode, trạng thái memory, seed) |
| `probe_<x>.json` | **summary**: trung bình, khoảng tin cậy, verdict — file để đọc |
| `costs.json` | thời gian, số forward |
| `poison_s0.pt` | poison đóng băng ở s0 (với probe có poison) |

Bảng tổng các probe:

| Probe | Câu hỏi | Lần chạy | Stage CLI | Thư mục kết quả (dưới `b2/seed_0/probes/`) |
|---|---|---|---|---|
| `r2` | memory lành phình ra, trigger cũ có suy giảm không? | 1, 2, 3, 6 | `probe-growth` | `growth-suffix-test/` hoặc `growth_<selection>-suffix-test/` |
| `r1` | một trigger chung có đủ không? | 1, 2 | `evaluate` ×2 + `probe-universal` | `b2/seed_0/probe_universal.json` |
| `position` | trigger nên đặt ở đâu trong query? | 1, 2 | `probe-position` | `position-<mode>-suffix-test/` |
| `contam` | bản ghi mang trigger lọt vào memory thì sao? | 4, 6 | `probe-contamination` | `contam_<self\|rival>-suffix-test/` |
| `writeback` | agent tự ghi lại theo chính sách ghi (chỉ đo truy hồi) | 5, 6 | `probe-writeback` | `writeback[_p<N>]-suffix-test/` |
| `e2e` | agent thật đọc top-k và **hành động**: ASR-t | 7 | `probe-e2e` | `e2e_<static\|writeback>[_p<N>]-suffix-test/` |

### 6.3 `r2` — memory phình (P0-R2)

- **Làm gì**: tìm trigger ở s0, đóng băng trigger + 5 poison, thêm tài liệu lành vào memory
  theo các mức `GROWTH`, đo lại `on_hit` trên `Q_eval`.
- **Biến riêng**:

  | Biến | Mặc định | Ý nghĩa |
  |---|---|---|
  | `GROWTH` | `0.25 0.5 1.0` | mức phình (tỉ lệ so với memory gốc) |
  | `DRIFT_SEEDS` | `0 1 2 3 4` | số lần bốc lại tài liệu thêm vào |
  | `GROWTH_SELECTION` | `random` | `random` (ngẫu nhiên cùng domain), `support` (gần `Q_sup` sạch), `triggered` (gần `Q_sup` có trigger, cận trên), `ood` (tài liệu domain `qa`). `support`/`triggered` tất định → dùng `DRIFT_SEEDS=0` |
  | `MIN_BASE_HIT` | 0.0 | bỏ episode có trigger chưa hoạt động ở s0 |

- **Ra**: `growth[_<selection>]-suffix-test/probe_growth.json`, log `$RUN_ROOT/r2-<selection>.console`.
- **Verdict**: `decays` / `stable` / `inconclusive` / `no-data` (§8.2).

### 6.4 `contam` — bản ghi mang trigger (lần 4)

- **Làm gì**: thêm vào memory các bản ghi chứa trigger. `self` = agent ghi lại tương tác có
  trigger của chính nó với nhãn lành; `rival` = poison của những kẻ tấn công khác.
- **Biến**: `CONTAM_SCENARIO` (`self` | `rival`), `CONTAM_LEVELS` (rỗng → `self`: 1 2 5 10 25 bản
  ghi; `rival`: 1 3 7 15 kẻ tấn công), `DRIFT_SEEDS`.
- **Ra**: `contam_<scenario>-suffix-test/probe_contamination.json`, log `contam-<scenario>.console`.
- **Verdict**: như `r2`.

### 6.5 `writeback` — agent tự ghi lại, vòng kín, chỉ đo truy hồi (lần 5)

- **Làm gì**: stream các tương tác có trigger; mỗi tương tác truy hồi trên memory hiện tại;
  "nổ" = poison vào top-k; chính sách ghi quyết định ghi gì:

  | Policy | Ghi gì |
  |---|---|
  | `none` | không ghi (memory tĩnh như AgentPoison) |
  | `log_outcome` | ghi mọi tương tác kèm kết quả: nổ → bản ghi độc, không nổ → lành |
  | `verified` | chỉ ghi khi không nổ (verifier hoàn hảo) |
  | `corrected` | luôn ghi với nhãn đúng (người sửa) |

  Ngoài ra đo `cleanup`: xoá poison gốc, xem riêng bản ghi đã log có còn mang backdoor không.
- **Biến**: `SEED_POISON` (rỗng = cả 5 poison; `1` = chỉ poison đầu, để hạ base khỏi trần),
  `WRITEBACK_LEVELS` (rỗng → 5 10 25 50 tương tác), `DRIFT_SEEDS`.
- **Ra**: `writeback[_p<N>]-suffix-test/probe_writeback.json`, log `writeback[_p<N>].console`.
- **Verdict** theo từng policy: `dilutes` / `reinforces` / `significant-but-small` / `stable` /
  `no-interval`, kèm `persists_after_cleanup`.

### 6.6 `e2e` — end-to-end với LLM (lần 7)

- **Làm gì**: như trên nhưng agent AgentDriver **thật**: top-k (cảnh + reasoning) vào prompt,
  LLM ra `Driving Plan`, chấm hành động. Tấn công "nổ" khi plan là `SUDDEN STOP`.
  Chỉ domain `ad` có agent thật; domain khác chạy nội dung giả để thử ống nước.
- **Hai kịch bản** (`E2E_SCENARIO`):

  | Kịch bản | Các trạng thái memory được chấm (tên `family` trong summary) |
  |---|---|
  | `static` | `base` (sạch + poison, query có trigger), `base_off` (query không trigger), `control` (không poison), `self-<n>` (n bản ghi mang trigger), `rival-<n>` (n kẻ tấn công khác) |
  | `writeback` | `base`, `base_off`, `control`, rồi theo policy: `<policy>-<n>`, `<policy>-<n>-off`, `log_outcome-<n>-cleanup` |

- **Biến**:

  | Biến | Mặc định | Ý nghĩa |
  |---|---|---|
  | `E2E_SCENARIO` | `static` | `static` hoặc `writeback` |
  | `E2E_QUERIES` | 32 | số query đầu của `Q_eval` mỗi episode được hỏi LLM |
  | `E2E_SELF_LEVELS` | `1 5 25` | `static`: số bản ghi self (đặt rỗng để bỏ) |
  | `E2E_RIVAL_LEVELS` | `15` | `static`: số kẻ tấn công rival (đặt rỗng để bỏ) |
  | `E2E_STREAM` | 50 | `writeback`: số tương tác stream |
  | `WRITEBACK_LEVELS` | (rỗng → `E2E_STREAM`) | `writeback`: chấm memory sau bao nhiêu tương tác |
  | `SEED_POISON` | rỗng | như §6.5 |
  | `E2E_SEEDS` | `0` | lần bốc self/rival/stream (seed s = cùng lần bốc với lần 4/5) |
  | `LLM_BACKEND` | `hf` | `hf`, `vllm`, `fixture` (giả, cho test) |
  | `LLM_MODEL` | `NousResearch/Meta-Llama-3-8B-Instruct` | bản mirror không gated của Llama-3-8B-Instruct |
  | `LLM_BATCH`, `LLM_MAX_NEW` | 8, 320 | batch, số token sinh tối đa |
  | `LLM_CACHE` | `<output-dir>/llm_cache.jsonl` | cache câu trả lời: **chạy lại = resume** |
  | `DEADLINE_MINUTES` | rỗng | dừng nhận trạng thái mới sau N phút, tóm tắt phần đã xong (`state: partial`) |

- **Ra**: `e2e_<scenario>[_p<N>]-suffix-test/probe_e2e.json` (+ `probe_e2e.jsonl`,
  `probe_e2e_stream.jsonl` cho writeback), log `$RUN_ROOT/e2e_<scenario>[_p<N>].console`.
  `e2e` không nhận `RESUME`: cache LLM làm việc đó.
- **Đọc**: `gates` trước (số có ý nghĩa không), rồi `families.<state>.change_vs_base.asr_t`
  và `verdict`, rồi `writeback.claim` (§8.2).

### 6.7 `r1` — trigger chung vs mỗi episode (P0-R1)

- **Làm gì**: train arm `b2` (direct-logit) và `b3` (universal-logit) cùng số bước, `evaluate`
  cả hai, rồi `probe-universal` ghép cặp theo episode.
- **Ra**: `$RUN_ROOT/b2/seed_0/probe_universal.json`; `b3` ở `$RUN_ROOT/b3/seed_0/`. Log `r1.console`.
- **Verdict**: `universal-suffices` / `conditioning-helps` / `confounded` (hai arm không cùng ngân sách).

### 6.8 `position` — vị trí trigger (P1)

- **Biến**: `POSITIONS` (`suffix prefix both middle`), `POSITION_MODE` (`transfer` = một
  trigger chấm ở mọi vị trí; `reoptimize` = tối ưu riêng cho từng vị trí — hai câu hỏi khác nhau),
  `POSITION_BASELINE` (`suffix`), `ATTENTION` (1 = đo thêm phần attention `[CLS]` đổ vào trigger).
- **Ra**: `position-<mode>-suffix-test/probe_position.json`; đọc `best_length_matched`.

---

## 7. Pipeline C — Kaggle

Mọi con số thật chạy trên Kaggle T4×2 (tài khoản `dainn98s`). Kernel không chứa code tấn công:
nó clone repo, chồng file chưa commit lên, rồi gọi `run_p0_probes.sh`.

### 7.1 Một kernel gồm gì

```
.kaggle/<tên>/
  kernel-metadata.json   id, GPU (machine_shape NvidiaTeslaT4 = T4×2), internet,
                         kernel_sources = output của kernel trước được mount vào /kaggle/input
  run_template.py        mẫu, có chỗ trống __OVERLAY__ (và __STEPS__, __TITLE__ ở lần 7)
  run.py                 bản đã build: template + overlay base64 → file được đẩy lên
  build.py               (chỉ mcat-p0-e2e/) sinh run.py + metadata cho 2 kernel lần 7
```

`run.py` khi chạy trên Kaggle:

| Bước | Làm gì |
|---|---|
| Stage 0 | `nvidia-smi`; cài `transformers>=5.3,<6`, `gdown`; `git clone --branch tanh` repo GitHub; kiểm tra marker (code đủ mới); ghi **overlay** (file chưa commit) đè lên; `touch tests/__init__.py`; tải dữ liệu `ad` bằng gdown; copy `base-search` / `llm_cache.jsonl` từ kernel trước nếu có mount; (lần 7) tải trọng số LLM; chạy `run_p0_probes.sh preflight` trên fixture CPU |
| Stage 1+ | gọi `run_p0_probes.sh <probe>` với env của từng step; lần 2–6 chạy hai "lane" song song trên `cuda:0` và `cuda:1`, mỗi lane `RUN_ROOT` riêng |
| Cuối | gom các `probe_*.json` vào `/kaggle/working/<tên>_summary.json`, in verdict |

Ngân sách: Kaggle cắt ở 12 h; template lần 7 dừng ở 11.4 h và trừ 20 phút để kịp tóm tắt.

### 7.2 Quy trình đẩy một kernel

```bash
# 0. code + test pass ở máy (§3.3)
# 1. code đã commit phải được push lên nhánh tanh (người dùng tự push);
#    file chưa commit thì cho vào overlay
# 2. build run.py (ví dụ lần 7)
py -3.11 .kaggle/mcat-p0-e2e/build.py           # → .kaggle/mcat-p0-e2e-{wb,static}/
# 3. đẩy
export KAGGLE_API_TOKEN="$(tr -d '\r\n ' < .kaggle/access_token)"
.venv/Scripts/kaggle.exe kernels push -p .kaggle/mcat-p0-e2e-static
# 4. theo dõi (logs trống khi đang chạy; chỉ có output sau khi xong)
.venv/Scripts/kaggle.exe kernels status dainn98s/adapt-mcat-p0-e2e-static
# 5. tải output về
.venv/Scripts/kaggle.exe kernels output dainn98s/adapt-mcat-p0-e2e-static -p outputs/kaggle/mcat-p0-e2e-static
```

Overlay phải đổi CRLF → LF (`build.py` đã làm); một `.sh` CRLF làm preflight chết ngay trong 0 s.

### 7.3 Kernel ↔ lần chạy ↔ kết quả

| Lần | Thư mục `.kaggle/` | Kernel id (`dainn98s/…`) | Chạy gì | Output tải về | Kết quả |
|---|---|---|---|---|---|
| 1 | `mcat-p0` | `adapt-mcat-p0-probes` | `qa`: r2, r1, position | `outputs/kaggle/mcat-p0/` | `p0_falsification_results.md` |
| 2 | `mcat-p0-ad` | `adapt-mcat-p0-ad` | `ad`, 6 token: r2, r1, position | `outputs/kaggle/mcat-p0-ad/` | `p0_run2_agentdriver_results.md` |
| 3 | `mcat-p0-r2t` | `adapt-mcat-p0-r2-targeted` | r2 với `support` / `triggered` | `outputs/kaggle/mcat-p0-r2t/` | `p0_run3_targeted_growth_results.md` |
| 4 | `mcat-p0-contam` | `adapt-mcat-p0-contamination` | contam `self` / `rival` | `outputs/kaggle/mcat-p0-contam/` | `p0_run4_contamination_results.md` |
| 5 | `mcat-p0-writeback` | `adapt-mcat-p0-writeback` | writeback p5 / p1 — v1 lỗi preflight (CRLF), gộp vào lần 6 | `outputs/kaggle/mcat-p0-writeback/` | trong `p0_run6_rerun_results.md` |
| 6 | `mcat-p0-rerun` | `adapt-mcat-p0-rerun` | lần 2–5 lại kèm hit@1/2/3/5 + arm `ood`; lane `benign` (cuda:0) và `adversarial` (cuda:1) | `outputs/kaggle/mcat-p0-rerun/` | `p0_run6_rerun_results.md` |
| 7 | `mcat-p0-e2e-wb`, `mcat-p0-e2e-static` (build từ `mcat-p0-e2e/`) | `adapt-mcat-p0-e2e-writeback`, `adapt-mcat-p0-e2e-static` | e2e writeback p1 + p5; e2e static. Mount output lần 6 để dùng lại trigger | `outputs/kaggle/mcat-p0-e2e-{writeback,static}/` | **đang chạy** (đẩy 2026-10-07); viết vào `p0_run7_e2e_results.md` |
| — | `mcat-ad` | `adapt-mcat-agentdriver` | pilot MCAT (`run_mcat.sh`) trên `ad`: shakedown, m1 vs b3 | — | — |
| — | `hotflip-margin`, `-demo`, `-sweep` | `adapt-hotflip-margin-qa`, `-demo`, `-sweep` | nhánh hotflip-margin (§10) | `outputs/kaggle_demo/`, `outputs/kaggle_sweep/` | — |

Lần 7 bị cắt bởi deadline (`state: partial`)? Đẩy một kernel tiếp theo mount output của kernel
vừa xong (`kernel_sources`): `llm_cache.jsonl` được copy vào tự động, câu đã trả lời không tính lại.

---

## 8. Output: file nằm ở đâu, đọc key nào

### 8.1 Cây thư mục của một `RUN_ROOT`

```
<RUN_ROOT>/                                 local: outputs/mcat/p0 ; Kaggle: /kaggle/working/mcat_p0_*
├── _preflight-tests.log, _preflight-smoke.log, _smoke/      preflight
├── cache/                                  vector corpus (nếu CACHE_DIR mặc định)
├── base-search/<episode>-s0/               trigger s0 dùng chung (nếu BASE_SEARCH_DIR trỏ vào đây)
│     triggers.jsonl  checkpoint.pt  training.json  metrics.jsonl
├── r2-random.console, contam-self.console, writeback.console, e2e_static.console, r1.console, position.console
├── llm_cache.jsonl                         (e2e) cache câu trả lời LLM
├── b2/seed_0/                              arm per-episode, mọi probe dùng chung
│     manifest.json  episodes.jsonl  stages/*.json
│     checkpoint.pt  evaluation.json  adapt-test/     (chỉ khi r1 đã train/evaluate)
│     probe_universal.json                            (r1)
│     probes/
│        growth-suffix-test/                probe_growth.json  probe_growth.jsonl  probe_config.json  costs.json  poison_s0.pt
│        growth_triggered-suffix-test/      …
│        contam_self-suffix-test/           probe_contamination.json …
│        writeback_p1-suffix-test/          probe_writeback.json …
│        e2e_static-suffix-test/            probe_e2e.json  probe_e2e.jsonl
│        e2e_writeback_p1-suffix-test/      probe_e2e.json  probe_e2e.jsonl  probe_e2e_stream.jsonl
│        position-transfer-suffix-test/     probe_position.json …
└── b3/seed_0/                              (r1) arm universal
```

Output Kaggle tải về có thêm một tầng: `outputs/kaggle/<kernel>/` chứa `<log kernel>.log`,
`<tên>_summary.json` (gom mọi summary) và thư mục `<RUN_ROOT>`; lần 6 có hai `RUN_ROOT` con
`adversarial/` và `benign/`.

### 8.2 Đọc summary: key quan trọng và verdict

| File | Key đọc trước | Verdict nghĩa là gì |
|---|---|---|
| `probe_growth.json` | `verdict`, `reason`, `base.on_hit`, `levels.<mức>.on_hit`, `levels.<mức>.drop_vs_base.{mean_difference, ci_low, ci_high}`, `seed_spread`, `hit_curve` | `decays`: giảm đơn điệu, CI loại 0, vượt dao động seed → R2 bị bác bỏ. `stable`: CI ở mức lớn nhất chứa 0 → không có bằng chứng suy giảm (đây là **kết quả**, không phải lỗi). `inconclusive`: hiệu ứng nhỏ hơn dao động seed hoặc không đơn điệu → thêm seed/episode, **không** báo là suy giảm |
| `probe_contamination.json` | như trên + `scenario`, `level_unit` | như trên |
| `probe_writeback.json` | `policies.<policy>.verdict.{direction, change, ci, persists_after_cleanup, cleanup_hit}` | `dilutes`: CI < 0 và giảm ≥ 0.05. `reinforces`: CI > 0 và tăng ≥ 0.05. `significant-but-small`: CI loại 0 nhưng < 0.05. `stable`: CI chứa 0 |
| `probe_e2e.json` | `state` (`completed`/`partial`), `gates`, `families.<state>.{asr_t, asr_a, hit_1…hit_5, correct}`, `families.<state>.change_vs_base.asr_t.{mean_difference, ci_low, ci_high, p_value, p_holm}`, `families.<state>.verdict`, `writeback.policies`, `writeback.claim` | `gates`: `parse_ok` (≥ 95% câu trả lời parse được), `false_activation_ok` (≤ 0.05), `control_ok`, `floor` (= true nghĩa là ASR-t base < 0.10: tấn công không chạy end-to-end, **không** đọc hướng được). Verdict như writeback, nhưng "có ý nghĩa" cần **cả** CI loại 0 **và** `p_holm` < 0.05. `claim.holds`: có một policy pha loãng và một policy không, CI tách rời |
| `probe_universal.json` | `verdict`, `treatment.mean`, `control.mean`, `paired.mean_difference` | `universal-suffices`: không khác biệt → một trigger chung đã đủ. `conditioning-helps`: khác biệt có ý nghĩa. `confounded`: hai arm không cùng ngân sách, con số vô nghĩa |
| `probe_position.json` | `best_length_matched`, `positions.<vị trí>.{on_hit, off_hit, length_matched}`, `attention` | vị trí tốt nhất trong các vị trí cùng độ dài |
| `evaluation.json` | `trigger_on`, `trigger_off`, `false_activation`, `controls` | — |
| `stages/*.json` | `state` | stage đó đã xong chưa |

Khoảng tin cậy là paired bootstrap **theo episode** (episode là đơn vị độc lập; nhiều seed của
một episode được lấy trung bình trước).

---

## 9. Quy trình một lần chạy (từ ý tưởng đến kết quả)

1. **Chốt trước** trong `open_problems.md`: mục "Lần chạy N (chốt TRƯỚC khi chạy, <ngày>)" —
   câu hỏi, thiết lập, tiêu chí đọc kết quả. Không sửa tiêu chí sau khi thấy số.
2. **Code + test** ở máy: `py -3.11 -m pytest tests/test_mcat*.py`; smoke fixture (§3.4).
3. **Push** code lên nhánh `tanh` (người dùng tự làm), hoặc đưa file chưa commit vào overlay.
4. **Build + đẩy kernel** (§7.2). Nếu cần dùng lại trigger của lần trước, thêm kernel đó vào
   `kernel_sources` và đặt `BASE_SEARCH_DIR`.
5. **Tải output** về `outputs/kaggle/<kernel>/`.
6. **Viết kết quả** vào `_idea_q1_aplus/p0_runN_<tên>_results.md`: mỗi con số kèm file nguồn;
   tính tổng hợp bằng script, không tính tay. Thêm dòng `> **Kết quả lần chạy N → [link]**`
   dưới mục chốt trước trong `open_problems.md`.

---

## 10. Các nhánh khác

| Nhánh | Code | Tài liệu |
|---|---|---|
| Cài đặt chung, demo AgentPoison, chỉ số ACC/ASR | `src/agentpoison/`, `algo/` | `00_setup.md` → `04_end_to_end.md` |
| Tool-calling poison | `src/toolpoison/` | `05_toolpoison_demo.md` |
| ARTEMIS (kiểm thử prompt MAS) | `src/adapt/extensions/`, `src/integration/` | `10_…` → `17_…` |
| AgentPoison theo phase, retrieval-margin | `src/agentpoison/phases.py`, `src/triggers/margin.py` | `18_…`, `19_…` |
| MCAT pilot trên Kaggle | `scripts/run_mcat.sh` | `20_mcat_kaggle_runbook.md` |
| Specificity | `src/triggers/specificity/`, `scripts/run_specificity_ap.sh` | `21_…` |
| MCAT M3 drift (kế hoạch) | `drift.py`, `adapt.py` | `22_mcat_m3_drift_plan.md` |
| AQuA (phòng thủ) | `src/aqua/` | `23_aqua_agent_generation_plan.md` |
| Probe P0 (bản cũ, lần 1) | `scripts/run_p0_probes.sh` | `24_p0_probe_kaggle_runbook.md` |

---

## 11. Gặp lỗi

| Thông báo / triệu chứng | Nguyên nhân | Cách xử lý |
|---|---|---|
| `probe refused: … holds rows from another probe configuration` | chạy probe vào thư mục đã có hàng của cấu hình khác | đổi `RUN_ROOT`, hoặc trả cờ về như cũ (§4.2) |
| `probe/evaluation refused: the checkpoint was trained under another contract` | cờ train khác lúc train | dùng đúng `TRAIN_FLAGS`, hoặc train lại |
| `resume refused: the episode split changed` | đổi cờ episode (`DOCUMENTS`, `PER_SPLIT`, `SEED`…) trong thư mục cũ | `RUN_ROOT` mới |
| `stage prepare-episodes is required` / `stage train is required` | gọi stage khi chưa có stage trước | chạy stage còn thiếu (§5.1) |
| `!! run this from the repository root` | chạy script từ thư mục khác | `cd` về gốc repo |
| `scikit-learn is required` ở preflight | thiếu sklearn trong run thật | cài sklearn, hoặc `FIXTURE=1` nếu chỉ thử |
| `load_dpr` dừng vì tokenizer không lowercase | transformers 5.0–5.2 | `pip install "transformers>=5.3,<6"` |
| `AgentDriver memory is not vendored…` | thiếu `agentdriver/data/finetune/*.json` | `gdown 1YBZpsACVr7iK55WG3_igzPUL8_nfVvKq` (train), `gdown 1UOJWu2sR80QYJW6dRuhtgygfeIuWWqPd` (val) vào thư mục đó |
| Kaggle: preflight thất bại sau 0 s | `.sh` trong overlay là CRLF | build overlay với CRLF → LF (`build.py` đã làm) |
| Kaggle: `import tests…` lấy nhầm package | image Kaggle có package `tests` trong site-packages | `touch tests/__init__.py` (template đã làm) |
| Kaggle: `kernels logs` trống | bình thường khi đang chạy | chờ, dùng `kernels status`; output chỉ có khi xong |
| `probe_e2e.json` có `state: partial` | hết deadline giữa chừng | kernel tiếp theo mount output này; `llm_cache.jsonl` resume miễn phí |
| Script hỏng khi `PYTHON` là đường dẫn có dấu cách | `$PYTHON` không được đặt trong ngoặc kép | `PYTHON="py -3.11"` |
| `.\make.ps1 …` báo không thấy python | `make.ps1` trỏ tới `.venv-adapt/` đã xoá | gọi thẳng `py -3.11 -m src.triggers.mcat …` |
| Local: `ModuleNotFoundError: torch` | dùng `.venv` (không có torch) | dùng `py -3.11` |
