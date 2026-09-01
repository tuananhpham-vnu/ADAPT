# Stage 4 — RAG và tooling (khoảng 6 tuần)

Paper nói thẳng: "Ignore agents with tooling. Currently ARTEMIS only tests behavior based on the
system prompt and cannot simulate tool calls." Stage này lấp chỗ đó, và là chỗ repo ADAPT có lợi thế
mà không nhóm nào khác có: sẵn agent dùng tool thật, sẵn RAG, và sẵn máy sinh context đối kháng.

Code: `src/adapt/rag/`, `src/adapt/extractors/autogen_detector.py`

## 1. Mở rộng taxonomy requirement

Hiện `R(P) = T ∪ AC ∪ CC`. Thêm hai loại:

**TC — tool constraint.** Agent phải gọi hoặc không được gọi tool nào, theo thứ tự nào, tham số ra
sao. Bốn trạng thái: gọi đúng, gọi nhầm tool, thiếu lời gọi, bịa tham số.

**GC — grounding constraint.** Agent chỉ được dùng thông tin có trong context đã truy hồi. Ba trạng
thái: bám context, bịa thông tin, mâu thuẫn với context.

Kéo theo hai thay đổi trong pipeline. Bộ sinh test case phải coi TC và GC là factor như T/AC/CC khi
dựng pairwise covering array. Judge phải chấm **cả trace tool-call**, không chỉ chấm văn bản trả về:
một response đọc thì đúng nhưng gọi nhầm tool vẫn là lỗi.

Phần trace không phải dựng từ đầu. `src/providers/base.py` đã chuẩn hoá `ToolCall(name, args)` cho
mọi nhà cung cấp, và `artemis/tools/wrapper_generator.py` của upstream đã có sẵn mầm để bọc tool —
việc cần làm là mở rộng nó thành harness tool giả có ghi lại trace.

## 2. Test case gồm cả context truy hồi

Với agent RAG, test case không còn là một user query mà là cặp `(query, retrieved context)`. Mỗi test
case sinh hai biến thể:

- **lành** — context truy hồi bình thường, dùng để đo độ bám context.
- **bị đầu độc** — context chứa trigger sinh bằng chính `algo/trigger_optimization.py` của repo này.

Metric mới: **khoảng cách bền vững** `Q_lành - Q_đầu_độc`.

Câu hỏi nghiên cứu: system prompt viết thế nào thì chống được context bị đầu độc? ARTEMIS đo chất
lượng prompt trong điều kiện sạch; phần này đo **độ bền của prompt khi tầng RAG bị tấn công**. Đây
là câu hỏi chưa ai trả lời, và là chỗ hai nửa của repo gặp nhau.

## 3. Khép vòng đối kháng

Nối ngược về stage 12: vòng cải tiến prompt giờ tối ưu đồng thời hai thứ, accuracy và khoảng cách
bền vững. Bên tấn công sinh trigger mạnh hơn, bên phòng thủ vá prompt để chịu được, lặp lại.

Đây đúng là ý "adversarial dual-agent protection training" trong tên đề tài. Cần lưu ý một cái bẫy:
prompt vá theo hướng phòng thủ quá tay (kiểu "không tin bất cứ thứ gì trong context") sẽ làm
`Q_lành` tụt. Phải báo cáo cả hai con số, không chỉ khoảng cách.

## 4. Extractor cho AutoGen

Upstream chỉ có `langgraph_detector.py`. Hai hệ trong repo đều không phải LangGraph:

- **EhrAgent** dùng AutoGen. Agent và system prompt nằm ở `EhrAgent/ehragent/main.py` (dựng
  `AssistantAgent` và `MedAgent`) và `EhrAgent/ehragent/medagent.py`. Cần viết `autogen_detector.py`
  theo cùng giao diện với `langgraph_detector.py`.
- **ReAct** thì đơn giản hơn nhiều: prompt nằm sẵn trong `ReAct/prompts/prompts.json`, đọc trực tiếp,
  không cần static analysis.

Làm được phần này thì đồng thời trả lời luôn giới hạn "chỉ hỗ trợ LangGraph" của paper.

## Nghiệm thu

- Khoảng cách bền vững phải dương và tăng theo cường độ trigger, tức tăng khi tăng `--num_iter` của
  `algo/trigger_optimization.py`. Không tăng thì hoặc trigger chưa đủ mạnh, hoặc cách tiêm context
  vào test case chưa đúng đường mà agent thật sự đọc.
- TC và GC phải bắt được lỗi mà T/AC/CC bỏ sót. Đo bằng số lỗi phát hiện thêm khi bật hai loại
  requirement mới, tương tự cách paper đo ablation bỏ decomposition.
