# `src/` — Code do dự án viết

```
src/
  attack/
    dilution/     hướng chính: trigger bị pha loãng khi memory có thêm query+trigger với action sạch
                  (tĩnh và write-back, retrieval và end-to-end); tên cũ mcat/
    margin/       loss_margin cải thiện tối ưu trigger AgentPoison; kèm helper dilution/ dùng lại
                  (artifacts, clustering, losses)
  defense/
    aqua/         pilot phòng thủ AQuA: authorization probe và causal scrubbing (tạm gác)
  artemis/        cải tiến ARTEMIS: spec, mutation, oracle, repair, provenance gate, extensions/
                  (mã gốc upstream, khi vendor, đặt ở artemis/upstream/)
  providers/      adapter LLM chung (OpenAI, Anthropic, Gemini, DeepSeek, OpenRouter)
  shared/         tracing, encode bằng retriever, tool giả lập cho demo gate
  config.py       ba vai model của ARTEMIS và kiểm tra tách vai
  main.py         CLI gộp: artemis, artemis-upstream, gate, aqua
```

README từng phần: [`attack/`](attack/README.md), [`attack/dilution/`](attack/dilution/README.md),
[`defense/aqua/`](defense/aqua/README.md), [`artemis/`](artemis/README.md), [`shared/`](shared/README.md).

`dilution/` có CLI riêng (`python -m src.attack.dilution`, script `scripts/run_p0_probes.sh`) và
**không** dùng `providers/`: nó gọi LLM local qua `dilution/llm.py`.

Ranh giới phụ thuộc:
- `src/` import `algo/`, không bao giờ ngược lại.
- `attack/` không import `defense/` hay `artemis/`; `shared/` không import nhánh nào
  (`tests/test_package_layout.py`).

## Lớp provider


Phần code gốc của AgentPoison gọi thẳng SDK `openai` ở khắp nơi, nên đổi nhà cung cấp
model là phải sửa nhiều chỗ. Thư mục `src/` là lớp trung gian: **mọi nhà cung cấp đều
được quy về cùng một giao diện**, phần gọi model ở trên không cần biết bên dưới là
OpenAI, Anthropic, Gemini, DeepSeek hay OpenRouter.

### Giao diện chung

`providers/base.py` định nghĩa ba thứ:

- `ToolCall(name, args)` — một lời gọi tool đã được chuẩn hoá.
- `ModelResponse(text, tool_calls, raw)` — kết quả trả về, luôn cùng hình dạng.
- `Provider` (Protocol) — mọi adapter phải có phương thức:

```python
complete(messages, tools=None, *, model=None, temperature=0.0, tool_choice=None) -> ModelResponse
```

Nhờ `raw` giữ lại phản hồi gốc nên khi cần chi tiết riêng của từng vendor vẫn lấy được.

### Cấu trúc file

| File | Vai trò |
|---|---|
| `providers/base.py` | Định nghĩa giao diện và kiểu dữ liệu chung. |
| `providers/openai_provider.py` | Adapter OpenAI (và mọi endpoint tương thích OpenAI). |
| `providers/anthropic_provider.py` | Adapter Anthropic (Claude). |
| `providers/gemini_provider.py` | Adapter Google Gemini. |
| `providers/deepseek_provider.py` | Adapter DeepSeek (kế thừa endpoint kiểu OpenAI). |
| `providers/openrouter_provider.py` | Adapter OpenRouter. |
| `providers/__init__.py` | `make_provider(name, api_key=None)` — factory chọn adapter theo tên. |
| `main.py` | CLI chọn nhánh `artemis`, `artemis-upstream`, `gate`, `aqua`. |

### Cách dùng

```python
from src.providers import make_provider

provider = make_provider("deepseek")            # key đọc từ biến môi trường
provider = make_provider("openai", api_key=k)   # hoặc truyền key tường minh

resp = provider.complete(
    messages=[{"role": "user", "content": "Xin chào"}],
    model="deepseek-chat",
)
print(resp.text, resp.tool_calls)
```

Lưu ý về `api_key`: truyền tường minh khi key đến từ request của người dùng. `os.environ`
là biến toàn process, hai request chạy song song sẽ đè key lên nhau.
