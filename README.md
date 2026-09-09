# ADAPT — Adversarial Dual-Agent Protection Training

Repo này nghiên cứu **tấn công đầu độc bộ nhớ RAG của LLM agent** (nhánh phát triển từ
AgentPoison) và là nền để xây phần phòng thủ. Ý tưởng một câu: nhét vài mẫu độc vào
bộ nhớ dài hạn của agent, rồi tối ưu một chuỗi "trigger" ngắn sao cho **chỉ khi** câu
hỏi chứa trigger thì agent mới lôi đúng mẫu độc ra và làm theo; câu hỏi bình thường
vẫn chạy đúng như cũ nên rất khó phát hiện.

## Phân chia code sau refactor

**Chạy dữ liệu thật:** `make agentpoison` hoặc `./make.ps1 agentpoison` trên
Windows chưa có GNU Make. Xem [hướng dẫn StrategyQA](src/agentpoison/STRATEGYQA.md)
cho bước kiểm tra, tạo index và chạy với LLM thật.
Quy trình theo từng phase và ablation có tại
[`_guidance/18_agentpoison_phases.md`](_guidance/18_agentpoison_phases.md).

| Nhánh | Code chính | Cách chạy / trạng thái |
|---|---|---|
| Tái lập AgentPoison | `algo/`, `ReAct/`, `EhrAgent/`, `agentdriver/`, `embedder/` | `make opt-*`, `make run-*`, `make eval-*` |
| AgentPoison trên corpus thật | [`src/agentpoison/strategyqa.py`](src/agentpoison/STRATEGYQA.md) | `make agentpoison`; DPR + toàn bộ StrategyQA + LLM thật |
| Demo AgentPoison tool-calling | [`src/agentpoison/`](src/agentpoison/README.md) | `make agentpoison-demo`; dùng seed trigger có sẵn |
| ARTEMIS gốc | Dự kiến `src/artemis/` | Chưa vendor; xem `_guidance/11_stage0_setup.md` |
| Cải tiến ADAPT | [`src/adapt/`](src/adapt/README.md) | `python -m src.adapt`; mutation, oracle, repair, gate và extension |
| Hạ tầng chung | `src/providers/`, [`src/shared/`](src/shared/README.md), `src/config.py` | Provider, tool giả lập, encoding, tracing |

Entry point chung: `python -m src.main {agentpoison,agentpoison-demo,artemis,adapt,gate} ...`.
`python -m src.agentpoison` và `src.main agentpoison` hiện chạy corpus StrategyQA thật.
`python -m src.main adapt --plan` chỉ ước lượng số lượt gọi, không chạy model.
Hai lần chạy `results/agentpoison/20260907_144612` và `20260907_152107` thuộc
**demo AgentPoison 2×2**, chưa phải tái lập đầy đủ tối ưu trigger.
Đường dẫn kết quả giữ nguyên. `src/toolpoison/` và `src/integration/` chỉ còn
alias tương thích; viết code mới trong package tương ứng.

## 1. Bức tranh chung

Mọi agent trong repo đều theo cùng một khuôn:

```
câu hỏi ──► [retriever] ──► lấy k mẫu giống nhất trong bộ nhớ ──► [LLM] ──► hành động
                  ▲
                  └── bộ nhớ dài hạn (đã bị tiêm vài mẫu độc)
```

Tấn công đánh vào mũi tên đầu tiên: sửa **embedding của câu hỏi** (bằng cách thêm
trigger) chứ không sửa LLM. Vì thế toàn bộ pipeline gồm 3 giai đoạn:

| Giai đoạn | Làm gì | Code |
|---|---|---|
| 1. Tối ưu trigger | Tìm chuỗi ~10 token đẩy query vào một vùng riêng trong không gian embedding | `algo/` |
| 2. Inference | Dán trigger vào câu hỏi, chạy agent, ghi lại đáp án | `ReAct/`, `EhrAgent/`, `agentdriver/` |
| 3. Đánh giá | Tính ACC, ASR-r, ASR-a, ASR-t | `*/eval.py` |

## 2. Bản đồ thư mục

| Thư mục | Vai trò | README |
|---|---|---|
| `algo/` | Thuật toán tối ưu trigger + tiện ích nạp DB/embedding dùng chung | [algo/README.md](algo/README.md) |
| `ReAct/` | Agent hỏi-đáp StrategyQA (agent `qa`) | [ReAct/README.md](ReAct/README.md) |
| `EhrAgent/` | Agent y tế sinh code truy vấn hồ sơ bệnh án eICU (agent `ehr`) | [EhrAgent/README.md](EhrAgent/README.md) |
| `agentdriver/` | Agent lái xe tự hành trên nuScenes (agent `ad`) | [agentdriver/README.md](agentdriver/README.md) |
| `embedder/` | Train / đánh giá retriever riêng (contrastive, classification) | [embedder/README.md](embedder/README.md) |
| `scripts/` | Script shell chạy sẵn cho từng agent + tiện ích Makefile | [scripts/README.md](scripts/README.md) |
| `src/` | Demo AgentPoison, cải tiến ADAPT và hạ tầng dùng chung | [src/README.md](src/README.md) |
| `_guidance/` | Hướng dẫn chạy theo từng kịch bản, tiếng Việt | [_guidance/README.md](_guidance/README.md) |
| `survey/` | Kho paper tham khảo | [survey/README.md](survey/README.md) |

File lẻ ở gốc: `Makefile` (mọi lệnh chạy), `adapt_tracing.py` (tracing), `requirements.txt`,
`.env.example` (mẫu biến môi trường).

## 3. Chạy nhanh

```bash
make venv && make install      # tạo .venv-adapt bằng uv, cài deps
cp .env.example .env           # điền OPENAI_API_KEY / DEEPSEEK_API_KEY
make opt-fast AGENT=qa         # tối ưu trigger bản rút gọn (~10 phút)
make trigger AGENT=qa          # in trigger vừa tìm được
# dán trigger vào trigger_token_list trong script inference
make run-qa-benign && make run-qa-adv
make eval-qa
```

Chi tiết từng bước xem `_guidance/`. Gõ `make` để xem toàn bộ target.

## 4. Tracing từng bước với Braintrust

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
