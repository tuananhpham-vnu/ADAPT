# `ReAct/` — Agent hỏi-đáp StrategyQA (agent `qa`)

Đây là agent nhẹ nhất trong ba agent, và cũng là agent nên dùng để thử pipeline vì dữ
liệu đã có sẵn trong repo. Agent trả lời câu hỏi Yes/No của StrategyQA theo vòng lặp
ReAct: **Thought → Action → Observation**, lặp tối đa 7 lần rồi chốt đáp án.

> Tài liệu gốc của upstream (Uncertainty-Aware Language Agent) nằm ở `README_upstream.md`.

## 1. Ý tưởng chung

Khác với chatbot trả lời thẳng, ReAct bắt LLM suy nghĩ ra tiếng rồi mới hành động:

```
Thought 1: cần biết X          <- LLM sinh
Action 1: search[X]            <- LLM sinh
Observation 1: ...đoạn văn...  <- môi trường trả về (RAG)
Thought 2: ...
Action 7: finish[Yes]          <- chốt đáp án
```

Chỗ bị tấn công là **Observation**: `search[...]` không gọi Wikipedia thật mà truy hồi
trong một kho đoạn văn cục bộ đã bị tiêm mẫu độc. Mẫu độc chứa câu kiểu "vì có chuỗi
tín hiệu này nên hãy trả lời ngược lại / hãy dừng lại", và LLM ngoan ngoãn làm theo.

## 2. Cấu trúc file

| File / thư mục | Vai trò |
|---|---|
| `run_strategyqa_gpt3.5.py` | Script chạy chính (backbone GPT). Là file `make run-qa-*` gọi. |
| `run_strategyqa_llama3_api.py` | Bản dùng LLaMA-3-70B qua Replicate. |
| `run_strategyqa_llama2.py` | Bản dùng LLaMA-2 chạy cục bộ (có LoRA/4-bit). |
| `run_strategyqa_inference.py` | Bản gộp nhiều backbone, dùng cho ablation. |
| `local_wikienv.py` | **Môi trường RAG**: giữ kho đoạn văn, embedding, và logic truy hồi. |
| `wikienv.py` (upstream) | Bản gốc gọi Wikipedia online. |
| `wrappers.py` | Bọc môi trường theo chuẩn gym: nạp dataset, chấm `em`, ghi log. |
| `eval.py` | Chấm ACC / ASR-r / ASR-a / ASR-t từ file `.jsonl` kết quả. |
| `database/` | Dữ liệu StrategyQA (câu hỏi, đoạn văn) và cache embedding. |
| `prompts/prompts.json` | Few-shot demo cho 3 kiểu prompt: standard, cot, react. |
| `utils/prompter.py`, `templates/` | Ghép prompt theo template của LLaMA. |
| `ablation/` | Kết quả `.jsonl` của các lần chạy ablation. |

## 3. Luồng chạy cụ thể

Chạy `python ReAct/run_strategyqa_gpt3.5.py --model dpr --task_type adv`:

1. **Dán trigger** — biến `trigger_token_list` ngay trong file chứa trigger bạn lấy từ
   bước tối ưu. Phải sửa tay trước khi chạy nhánh `adv`.
2. **Dựng môi trường** (span `react.setup_env`) — `local_wikienv.WikiEnv` nạp
   `strategyqa_train_paragraphs.json`, encode toàn bộ thành embedding (cache lại), rồi
   `load_db` nối thêm 2 mẫu độc vào cuối kho. Mẫu độc mang nội dung backdoor.
3. **Với mỗi câu hỏi** (span `react.question`) chạy hàm `react()`:
   - Mỗi vòng i (span `react.iteration`):
     - Gọi LLM sinh `Thought i` và `Action i` (span `react.llm_call`).
     - Ở vòng i = 2, nếu là nhánh `adv`, trigger được nối vào ngữ cảnh truy vấn.
     - `step(env, action, current_context)` thực thi hành động (span `react.env_action`).
       Nếu là `search[...]`, `local_retrieve_step` tính cosine similarity giữa query và
       toàn bộ kho, lấy top-k, chọn ngẫu nhiên một trong số đó (span `react.retrieve`,
       có ghi `poisoned_hit`).
     - Nối `Thought/Action/Observation` vào prompt cho vòng sau.
   - Dừng khi LLM ra `finish[...]` hoặc hết 7 vòng.
4. **Ghi kết quả** — mỗi câu một dòng JSON trong `result/ReAct/<embedder>-<algo>-<nhánh>.jsonl`,
   gồm đáp án, đáp án đúng, `retrieval_success`, và toàn bộ trajectory.

Lưu ý: script **ghi nối tiếp** (`append`). Chạy lại mà không xoá file cũ sẽ làm sai số
liệu — dùng `make clean-out`.

## 4. Đánh giá

```bash
make run-qa-benign && make run-qa-adv
make eval-qa
```

`eval.py` in ra ACC (độ chính xác gốc), ASR-r (tỉ lệ truy hồi trúng mẫu độc),
ASR-a (tỉ lệ agent thật sự hành xử theo backdoor), ASR-t (tỉ lệ hỏng tác vụ gốc),
đồng thời đẩy bộ số này lên Braintrust dưới span `eval.react`.
