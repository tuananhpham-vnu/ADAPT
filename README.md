# ADAPT — Adversarial Dual-Agent Protection Training

Repo này nghiên cứu **tấn công đầu độc bộ nhớ RAG của LLM agent** (phát triển từ
AgentPoison) và là nền để xây phần phòng thủ. Ý tưởng một câu: nhét vài mẫu độc vào bộ
nhớ dài hạn của agent, rồi tối ưu một chuỗi "trigger" ngắn sao cho **chỉ khi** câu hỏi
chứa trigger thì agent mới lôi đúng mẫu độc ra và làm theo; câu hỏi bình thường vẫn chạy
đúng như cũ nên rất khó phát hiện.

> Vòng co-training attacker ↔ defender đúng như tên đề tài **chưa được triển khai**.
> Hiện có phía tấn công (`src/triggers/`) và phía phòng thủ (`src/aqua/`, `src/adapt/`)
> chạy độc lập. Xem `_idea/memory_conditioned_generator_Q1_A_star.md` §15 cho điều kiện
> nối hai phía lại.

## 1. Bức tranh chung

Mọi agent trong repo đều theo cùng một khuôn:

```
câu hỏi ──► [retriever] ──► lấy k mẫu giống nhất trong bộ nhớ ──► [LLM] ──► hành động
                  ▲
                  └── bộ nhớ dài hạn (đã bị tiêm vài mẫu độc)
```

Tấn công đánh vào mũi tên đầu tiên: sửa **embedding của câu hỏi** (bằng cách thêm trigger)
chứ không sửa LLM. Pipeline vì thế gồm 3 giai đoạn:

| Giai đoạn | Làm gì | Code |
|---|---|---|
| 1. Tối ưu trigger | Tìm chuỗi ~10 token đẩy query vào một vùng riêng trong không gian embedding | `algo/` (upstream), `src/triggers/` (của dự án) |
| 2. Inference | Dán trigger vào câu hỏi, chạy agent, ghi lại đáp án | `ReAct/`, `EhrAgent/`, `agentdriver/` |
| 3. Đánh giá | Tính ACC, ASR-r, ASR-a, ASR-t | `*/eval.py` |

## 2. Các nhánh nghiên cứu và trạng thái

### Tấn công — `src/triggers/`

| Nhánh | Nội dung | Trạng thái |
|---|---|---|
| [`margin.py`](src/triggers/README.md) | AgentPoison + retrieval margin, 6 stage, resume được | Pipeline đủ, cần 2 GPU. Xem [`_guidance/19`](_guidance/19_agentpoison_margin_implementation.md) |
| [`mcat/`](src/triggers/mcat/README.md) | Generator sinh trigger theo trạng thái long-term memory | M0–M2 xong, test xanh, **chưa có số thật**. Xem [`_guidance/20`](_guidance/20_mcat_kaggle_runbook.md) |
| [`hierarchy/`](src/triggers/hierarchy/README.md) | Trigger universal / theo nhóm / từng câu, gộp dần từ dưới lên | Có pilot DPR thật (**kết quả âm**), import được, test xanh |
| [`specificity/`](src/triggers/specificity/README.md) | Độ bám query và chất lượng ngôn ngữ của trigger | Có kết quả, chưa đo ASR |

`assign` trong [`src/triggers/clustering.py`](src/triggers/clustering.py) đã được khôi phục,
nên hai nhánh trên import bình thường trở lại.

> **Cảnh báo về kiểm thử:** `.gitignore` đang bỏ qua cả thư mục `tests/`, nên chỉ 3/14 file
> test nằm trong repo. CI vì thế chạy 13/254 test rồi báo xanh. Muốn CI có ý nghĩa thì phải
> gỡ `tests/` khỏi `.gitignore` và commit 11 file còn lại. `_idea/` cũng đang bị bỏ qua, nên
> toàn bộ đề xuất nghiên cứu và kết quả pilot mà README này trỏ tới chỉ tồn tại trên máy local.

### Phòng thủ — `src/aqua/`, `src/adapt/`

| Nhánh | Nội dung | Trạng thái |
|---|---|---|
| [`aqua/`](src/aqua/README.md) | Authorization-Quotient Activations: matched quadruples → activations → low-rank projection → causal scrubbing | Pipeline pilot có; evaluation hiện là **candidate replay**, chưa phải agent tự sinh tool call |
| [`adapt/`](src/adapt/README.md) | Provenance gate chặn tool call trước khi thực thi, + mutation/oracle/repair | Demo, có explainer HTML |

### Kiểm thử agent — ARTEMIS

Nhánh tách biệt, không phải security: sinh test case từ cấu trúc system prompt bằng pairwise
covering array. Nền lý thuyết ở [`_guidance/10`](_guidance/10_artemis_overview.md), roadmap
Stage 0–5 ở [`_guidance/16`](_guidance/16_roadmap.md). Code mở rộng nằm trong `src/adapt/`;
mã gốc upstream chưa vendor (`src/artemis/` là chỗ để sẵn).

Các đề xuất nghiên cứu đầy đủ nằm ở [`_idea/`](_idea/).

## 3. Bản đồ thư mục

| Thư mục | Vai trò | README |
|---|---|---|
| `algo/` | Tái lập AgentPoison upstream: vòng tối ưu HotFlip gốc, nạp model/DB dùng chung | [algo/README.md](algo/README.md) |
| `src/triggers/` | Code tấn công do dự án viết: margin, mcat, hierarchy, specificity | [src/triggers/README.md](src/triggers/README.md) |
| `src/` | Nhánh thực nghiệm khác (aqua, adapt, agentpoison) và hạ tầng chung | [src/README.md](src/README.md) |
| `ReAct/` | Agent hỏi-đáp StrategyQA (agent `qa`) | [ReAct/README.md](ReAct/README.md) |
| `EhrAgent/` | Agent y tế sinh code truy vấn hồ sơ bệnh án eICU (agent `ehr`) | [EhrAgent/README.md](EhrAgent/README.md) |
| `agentdriver/` | Agent lái xe tự hành trên nuScenes (agent `ad`) | [agentdriver/README.md](agentdriver/README.md) |
| `embedder/` | Train / đánh giá retriever riêng (contrastive, classification) | [embedder/README.md](embedder/README.md) |
| `scripts/` | Script shell chạy sẵn cho từng agent và từng thí nghiệm | [scripts/README.md](scripts/README.md) |
| `_guidance/` | Hướng dẫn chạy theo từng kịch bản, tiếng Việt | [_guidance/README.md](_guidance/README.md) |
| `_idea/` | Đề xuất nghiên cứu và kết quả pilot | — |
| `survey/` | Kho paper tham khảo | [survey/README.md](survey/README.md) |

File lẻ ở gốc: `make.ps1` (entry point mọi lệnh), `adapt_tracing.py` (tracing),
`requirements.txt` (+ `requirements-agentdriver.txt`, `requirements-aqua.txt`),
`.env.example` (mẫu biến môi trường). `environment.yml` là env conda upstream đã cũ,
chỉ để đối chiếu version.

`src/toolpoison/` và `src/integration/` chỉ còn alias tương thích; viết code mới trong
package tương ứng. Ranh giới phụ thuộc: **`src/` import `algo/`, không bao giờ ngược lại.**

## 4. Cài đặt

Không có `Makefile`; trên mọi nền tảng dùng `make.ps1` hoặc gọi thẳng `python -m`.

```bash
uv venv --python 3.11 .venv-adapt
uv pip install --python .venv-adapt/Scripts/python.exe -r requirements.txt
uv pip install --python .venv-adapt/Scripts/python.exe \
  --index-url https://download.pytorch.org/whl/cu121 torch
cp .env.example .env        # điền OPENAI_API_KEY / DEEPSEEK_API_KEY
```

Chi tiết ở [`_guidance/00_setup.md`](_guidance/00_setup.md).

## 5. Chạy nhanh

`.\make.ps1 help` liệt kê toàn bộ target. Mọi target đều là bí danh của `python -m ...`,
nên trên Linux/macOS gọi thẳng module cũng được.

```powershell
.\make.ps1 mcat-smoke --output-dir outputs/mcat/smoke   # MCAT, CPU, không tải model
.\make.ps1 aqua-smoke                                    # AQuA, tensor fixture
.\make.ps1 agentpoison-check                             # kiểm tra split/checksum, không gọi model
.\make.ps1 agentpoison                                   # StrategyQA corpus thật + LLM thật
.\make.ps1 adapt-plan                                    # ước lượng số lượt gọi, không chạy model
.\make.ps1 gate-demo                                     # demo provenance gate
```

```bash
python -m src.triggers.margin smoke --output-dir outputs/agentpoison_margin/smoke
bash scripts/run_mcat.sh preflight    # kiểm tra môi trường + test + smoke, không đụng GPU
bash scripts/run_mcat.sh all          # 7 arm MCAT trên 1 GPU
```

Entry point gộp: `python -m src.main {agentpoison,agentpoison-demo,artemis,adapt,gate,aqua} ...`.

Quy trình AgentPoison theo từng phase và ablation ở
[`_guidance/18`](_guidance/18_agentpoison_phases.md); chạy corpus thật với LLM thật xem
[hướng dẫn StrategyQA](src/agentpoison/STRATEGYQA.md).

Hai lần chạy `results/agentpoison/20260907_144612` và `20260907_152107` thuộc **demo
AgentPoison 2×2**, chưa phải tái lập đầy đủ tối ưu trigger.

## 6. Tracing từng bước với Braintrust

Mọi bước của cả 3 agent đều được bọc trong một span có tên, xem `adapt_tracing.py`.
Bật bằng cách điền key vào `.env`:

```bash
BRAINTRUST_API_KEY=sk-...
BRAINTRUST_PROJECT=ADAPT-EhrAgent
```

Không có key thì tracing tự tắt, chỉ in log ra stdout dạng
`[start] medagent.retrieve_examples` / `[done] medagent.retrieve_examples (0.42s)`.

Tên span theo agent:

- `ehragent.*` / `medagent.*` — EhrAgent
- `react.*` — ReAct
- `agentdriver.*` — Agent-Driver
- `trigger_opt.iteration` — mỗi vòng tối ưu trigger
- `eval.*` — điểm cuối cùng của mỗi lần chấm

Span đáng chú ý nhất là bước truy hồi (`medagent.retrieve_examples`, `react.retrieve`,
`agentdriver.memory_retrieve`): nó ghi `poisoned_hit` — mẫu lấy ra có phải mẫu độc không.
Đây chính là ASR-r đo trực tiếp trên từng request.
