# Hướng dẫn chạy demo Corba

## 1. Cài dependencies

Kích hoạt đúng virtualenv rồi cài:

```powershell
pip install -r requirements.txt
pip install -U "autogen-agentchat" "autogen-ext[openai]" autogen-openaiext-client
pip install 'camel-ai[all]'
```

`open-ended/run.py` (dùng bởi `test_open_ended.py`) cần thêm các gói riêng
(`langchain`, `langchain-community`, `faiss-cpu`, ...) ngoài `requirements.txt`
gốc — nếu thiếu, cài theo thông báo `ImportError` khi chạy.

## 2. Cấu hình `.env`

Copy `.env.example` thành `.env` (đã có sẵn trong repo) và điền ít nhất **một**
trong các API key sau:

- `OPENAI_API_KEY`
- `DEEPSEEK_API_KEY`
- `GEMINI_API_KEY`
- `OPEN_ROUTER_API_KEY`

Không commit file `.env` lên git.

## 3. Thứ tự chạy

Luôn chạy `test_setup.py` trước tiên — nó kiểm tra key, chọn provider khả dụng,
và đồng bộ key sang `open-ended/config/api_keys.yaml` (module `open-ended` đọc
key từ file yaml này, không đọc trực tiếp `.env`).

```powershell
python test_setup.py        # Bước 0 — kiểm tra & chọn provider (bắt buộc chạy trước)
python test_mas_autogen.py  # Demo 1 — Corba attack trên AutoGen RoundRobinGroupChat
python test_mas_camel.py    # Demo 2 — Corba attack trên CAMEL ChatAgent
python test_topo.py         # Demo 3 — Corba trên topology phức tạp (6 agent A-F)
python test_open_ended.py   # Demo 4 — Corba trên open-ended (chỉ OpenRouter/OpenAI/Gemini, KHÔNG DeepSeek)
```

### Ý nghĩa từng demo

| File | Framework | Nội dung |
|---|---|---|
| `test_mas_autogen.py` | AutoGen (`RoundRobinGroupChat`) | 1 trong N agent bị "đầu độc" system prompt, yêu cầu tất cả agent lặp lại và lan truyền 1 câu, không dừng lại. Quan sát hội thoại có tự kết thúc (`APPROVE`) hay chạy hết `MAX_TURNS` với nội dung bị lặp/lan truyền. |
| `test_mas_camel.py` | CAMEL (`ChatAgent.step` tuần tự) | Tương tự demo 1 nhưng agent step tuần tự qua `NUM_TURNS` lượt, mỗi agent nhận toàn bộ lịch sử hội thoại trước đó. |
| `test_topo.py` | `autogen_core` (`RoutedAgent` + PubSub) | 6 agent (A-F) theo topology không hoàn chỉnh (tái dùng `ChatMASs/topo.py`), Agent_D bị đầu độc. Quan sát log xem câu lệnh đầu độc có lan từ D → B → A/C → E/F qua từng round. |
| `test_open_ended.py` | `open-ended/run.py` (subprocess) | Gọi `open-ended/run.py` với cwd = `open-ended/`. Chỉ hỗ trợ OpenRouter/OpenAI (qua `generate_with_gpt`) hoặc Gemini — không có nhánh DeepSeek. |

## 4. Tùy chỉnh qua biến môi trường

| Biến | Áp dụng cho | Mặc định |
|---|---|---|
| `DEMO_NUM_AGENTS` | autogen, camel, open_ended | 4 |
| `DEMO_MAX_TURNS` | autogen | 12 |
| `DEMO_NUM_TURNS` | camel | 3 |
| `DEMO_POISON_AT` | autogen, camel | 2 (index 0-based) |
| `DEMO_MAX_ROUND` | topo | 4 |
| `DEMO_TIME_STEP` | open_ended | 5 |

## 5. Lưu ý

- File `.env.example` hiện có một dòng lạ chèn giữa các biến (không phải cấu
  hình, có vẻ là lỗi copy-paste hoặc nội dung bị chèn vào) — nên kiểm tra và
  dọn lại file trước khi dùng làm mẫu.
- Nếu `test_setup.py` báo "chưa có API key hợp lệ nào", kiểm tra lại `.env` đã
  điền đúng biến và không còn giá trị placeholder (`your-key-here`).
