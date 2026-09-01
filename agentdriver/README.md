# `agentdriver/` — Agent lái xe tự hành (agent `ad`)

Agent nhận trạng thái xe và kết quả nhận thức (detection, prediction, occupancy, map) của
một khung cảnh nuScenes, truy hồi kinh nghiệm lái tương tự trong bộ nhớ, suy luận ra kế
hoạch lái, rồi sinh **quỹ đạo 6 điểm** cho 3 giây tiếp theo. Đây là agent có nhiều tầng
nhất, và mẫu độc nhắm vào tầng bộ nhớ để ép agent phanh gấp ("SUDDEN STOP").

## 1. Kiến trúc chung

```
dữ liệu khung cảnh
   ├─► PerceptionAgent   -> mô tả vật thể, occupancy, bản đồ thành text
   ├─► MemoryAgent       -> luật giao thông (common sense) + kinh nghiệm lái tương tự (RAG)
   ├─► ReasoningAgent    -> chuỗi suy luận + "Driving Plan"
   └─► PlanningAgent     -> quỹ đạo 6 điểm, có kiểm tra va chạm
```

Bốn agent con này chạy nối tiếp, đầu ra của cái trước là prompt của cái sau. `MemoryAgent`
là chỗ bị đầu độc: kinh nghiệm lái độc được gán nhãn `ADV_INJECTION`, khi nó lọt vào
prompt thì `ReasoningAgent` đổi kế hoạch thành dừng đột ngột.

## 2. Cấu trúc thư mục

| Thư mục / file | Vai trò |
|---|---|
| `execution/inference.py` | Điểm vào. Dựng `LanguageAgent` rồi chạy toàn bộ tập val. |
| `main/language_agent.py` | Điều phối 4 agent con; `inference_single` (1 khung cảnh) và `inference_all` (cả tập). |
| `perception/` | `perception_agent.py` gọi các functional tool, biến số liệu thành mô tả text. |
| `functional_tools/` | Tool thật: `detection.py`, `prediction.py`, `occupancy.py`, `map.py`, `ego_state.py`. |
| `memory/` | `memory_agent.py` (điều phối), `common_sense_memory.py` (luật), `experience_memory.py` (RAG kinh nghiệm lái + chỗ tiêm mẫu độc). |
| `reasoning/` | `reasoning_agent.py`, `chain_of_thoughts.py`, `collision_check.py`, `collision_optimization.py`. |
| `planning/` | `motion_planning.py` (**vòng lặp inference thật sự**), `planning_agent.py`, `planning_target.py`. |
| `llm_core/` | `chat.py`, `chat_utils.py` (client OpenAI dùng chung, có gắn tracing), `api_keys.py`, `timeout.py`. |
| `evaluation/` | Tính L2 error và tỉ lệ va chạm theo chuẩn UniAD / ST-P3. |
| `motion_planner/` | Code fine-tune planner (SFT, TRL) — chỉ cần khi tự train planner. |
| `utils/`, `visualization/` | Hình học, chuyển đổi toạ độ, vẽ kết quả. |
| `data/` | `split.json` có sẵn; phần dữ liệu nuScenes phải tải riêng (xem `_guidance/04`). |

## 3. Luồng chạy cụ thể

Chạy `make run-ad` (tức `python agentdriver/execution/inference.py`):

1. `inference.py` dựng `LanguageAgent`, đọc `data_samples_val.json`, mở span
   `agentdriver.run` rồi gọi `inference_all`.
2. `inference_all` → `PlanningAgent.run_batch` → `planning_batch_inference`
   (`planning/motion_planning.py`) — đây là nơi mọi thứ thật sự xảy ra:
   - Dán trigger: `trigger_token_list` ở đầu hàm; `trigger_insertion` ghép trigger vào bộ
     ví dụ chain-of-thought để tạo `CoT_prefix` độc.
   - Dựng `MemoryAgent` với embedder tương ứng và số mẫu độc `num_of_injection`.
   - **Với mỗi khung cảnh** (span `agentdriver.scenario`):
     - Dựng `working_memory` từ ego state + mô tả nhận thức; nếu đang tấn công thì nối
       `"Notice: <trigger>"` vào phần perception.
     - `memory_agent.run(working_memory)` (span `agentdriver.memory_retrieve`) truy hồi luật
       giao thông + kinh nghiệm lái. Span ghi `poisoned_hit` khi lấy trúng `ADV_INJECTION`.
     - Nếu truy hồi trúng mẫu độc thì dùng `CoT_prefix` độc làm system message, ngược lại
       dùng prompt mặc định. `reasoning_agent.run` sinh suy luận (span `agentdriver.reasoning`).
     - `planning_single_inference` sinh quỹ đạo (span `agentdriver.planner`), có kiểm tra
       va chạm và tối ưu lại nếu cần.
   - Cuối cùng in ACC, retrieval success rate, backdoor success rate.
3. Quỹ đạo dự đoán được lưu vào `result/<thời gian>/pred_trajs_dict.pkl` để `evaluation/`
   chấm L2 và tỉ lệ va chạm.

## 4. Điều kiện chạy

Agent này **cần dữ liệu nuScenes** và một planner đã fine-tune (`FINETUNE_PLANNER_NAME`
trong `.env`). Nếu chỉ muốn thử pipeline tấn công, dùng agent `qa` (ReAct) thay vì `ad`.
Xem `_guidance/04_end_to_end.md`.
