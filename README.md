# ADAPT — Adversarial Dual-Agent Protection Training

Repo này nghiên cứu **tấn công đầu độc bộ nhớ RAG của LLM agent** (phát triển từ
AgentPoison) và là nền để xây phần phòng thủ. Ý tưởng một câu: nhét vài mẫu độc vào bộ
nhớ dài hạn của agent, rồi tối ưu một chuỗi "trigger" ngắn sao cho **chỉ khi** câu hỏi
chứa trigger thì agent mới lôi đúng mẫu độc ra và làm theo; câu hỏi bình thường vẫn chạy
đúng như cũ nên rất khó phát hiện.

> Vòng co-training attacker ↔ defender đúng như tên đề tài **chưa được triển khai**.
> Hiện có phía tấn công (`src/attack/`) và phía phòng thủ (`src/defense/aqua/`, `src/artemis/`)
> chạy độc lập. Xem `_idea_q1_aplus/archive/memory_conditioned_generator_Q1_A_star.md` §15 cho điều kiện
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
| 1. Tối ưu trigger | Tìm chuỗi ~10 token đẩy query vào một vùng riêng trong không gian embedding | `algo/` (upstream), `src/attack/` (của dự án) |
| 2. Inference | Dán trigger vào câu hỏi, chạy agent, ghi lại đáp án | `ReAct/`, `EhrAgent/`, `agentdriver/` |
| 3. Đánh giá | Tính ACC, ASR-r, ASR-a, ASR-t | `*/eval.py` |

## 2. Các nhánh nghiên cứu và trạng thái

### Tấn công — `src/attack/`

| Nhánh | Nội dung | Trạng thái |
|---|---|---|
| [`dilution/`](src/attack/dilution/README.md) | **Hướng chính.** Trigger tối ưu bị pha loãng thế nào khi memory có thêm bản ghi query+trigger với action sạch: tĩnh (`self-n`, `rival-n`) và write-back (`log_outcome`, `verified`, `corrected`), đo cả retrieval lẫn end-to-end với LLM. Tên cũ `mcat/` | Run 1–8 xong (AgentDriver, Llama-3-8B và Qwen2.5-7B). Kết quả ở [`_idea_q1_aplus/results/`](_idea_q1_aplus/results/) |
| [`margin/`](src/attack/README.md) | loss_margin để cải thiện tối ưu trigger AgentPoison: `margin.py` (pipeline 6 stage), `hotflip_margin.py` (vòng HotFlip của `algo/` với loss uni/cpt/margin) | Pipeline đủ; `hotflip_margin` chỉ agent `qa`, đã chạy demo Kaggle. Xem [`_guidance/19`](_guidance/19_agentpoison_margin_implementation.md) |

### Phòng thủ — `src/defense/`

| Nhánh | Nội dung | Trạng thái |
|---|---|---|
| [`aqua/`](src/defense/aqua/README.md) | Authorization-Quotient Activations: matched quadruples → activations → low-rank projection → causal scrubbing | Tạm gác. Evaluation hiện là **candidate replay**, chưa phải agent tự sinh tool call |

### Kiểm thử agent — `src/artemis/`

Nhánh tách biệt, không phải memory poisoning: sinh test case từ cấu trúc system prompt bằng
pairwise covering array, kèm provenance gate chặn tool call trước khi thực thi. Nền lý thuyết ở
[`_guidance/10`](_guidance/10_artemis_overview.md), roadmap Stage 0–5 ở
[`_guidance/16`](_guidance/16_roadmap.md). Mã gốc ARTEMIS upstream chưa vendor (chỗ để sẵn:
`src/artemis/upstream/`).

Đề xuất nghiên cứu và trạng thái chọn đề tài ở [`_idea_q1_aplus/`](_idea_q1_aplus/README.md).

Đã xoá ngày 2026-10-13 (lấy lại từ git nếu cần): `src/attack/hierarchy/`, `src/attack/specificity/`,
`src/agentpoison/` (StrategyQA và demo tool-calling ngân hàng), alias `src/toolpoison/`, `src/integration/`.

## 3. Bản đồ thư mục

| Thư mục | Vai trò | README |
|---|---|---|
| `algo/` | Tái lập AgentPoison upstream: vòng tối ưu HotFlip gốc, nạp model/DB dùng chung | [algo/README.md](algo/README.md) |
| `ReAct/` | Agent hỏi-đáp StrategyQA (agent `qa`) | [ReAct/README.md](ReAct/README.md) |
| `EhrAgent/` | Agent y tế sinh code truy vấn hồ sơ bệnh án eICU (agent `ehr`) | [EhrAgent/README.md](EhrAgent/README.md) |
| `agentdriver/` | Agent lái xe tự hành trên nuScenes (agent `ad`); dữ liệu memory/query của `dilution/` | [agentdriver/README.md](agentdriver/README.md) |
| `embedder/` | Train / đánh giá retriever riêng (contrastive, classification) | [embedder/README.md](embedder/README.md) |
| `src/` | Code do dự án viết: `attack/`, `defense/`, `artemis/`, hạ tầng chung | [src/README.md](src/README.md) |
| `scripts/` | Script chạy sẵn: upstream theo agent, pipeline `dilution/` và margin trên Kaggle | [scripts/README.md](scripts/README.md) |
| `tests/` | Unit test (`python -m unittest discover -s tests`, cần `torch`) | — |
| `_guidance/` | Hướng dẫn chạy theo từng kịch bản, tiếng Việt | [_guidance/README.md](_guidance/README.md) |
| `_idea_q1_aplus/` | Đề xuất nghiên cứu và kết quả pilot | [_idea_q1_aplus/README.md](_idea_q1_aplus/README.md) |
| `survey/` | Kho paper tham khảo | [survey/README.md](survey/README.md) |

File lẻ ở gốc: `make.ps1` (bí danh cho các lệnh `python -m`), `adapt_tracing.py` (tracing),
`requirements.txt` (+ `requirements-agentdriver.txt`, `requirements-aqua.txt`),
`.env.example` (mẫu biến môi trường). `environment.yml` là env conda upstream đã cũ,
chỉ để đối chiếu version.

Ranh giới phụ thuộc: **`src/` import `algo/`, không bao giờ ngược lại.** Trong `src/`,
`attack/` không import `defense/` hay `artemis/` (kiểm bằng `tests/test_package_layout.py`).

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
.\make.ps1 dilution-smoke --output-dir outputs/mcat/smoke   # dilution, CPU, không tải model
.\make.ps1 aqua-smoke                                       # AQuA, tensor fixture
.\make.ps1 artemis-plan                                     # ARTEMIS: ước lượng số lượt gọi, không chạy model
.\make.ps1 gate-demo                                        # demo provenance gate
```

```bash
bash scripts/run_p0_probes.sh preflight    # dilution: kiểm tra môi trường + CPU smoke, không GPU
python -m src.attack.margin.margin smoke --output-dir outputs/agentpoison_margin/smoke
```

Entry point gộp: `python -m src.main {artemis,artemis-upstream,gate,aqua} ...`.
Chạy `dilution/` trên Kaggle: [`_guidance/24`](_guidance/24_p0_probe_kaggle_runbook.md) và
[`_guidance/25`](_guidance/25_helper_tong_hop.md).

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
