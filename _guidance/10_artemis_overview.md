# ARTEMIS — phương pháp và hiện trạng

Tài liệu nền cho các stage 11-16. Nguồn: `survey/2026.nda_recommend_JSSOFTWARE-S-26-02813-4.pdf`
(anh Hiệp là tác giả chính) và repo gốc `github.com/hype1524/MultiAgentTesting`.

## Ý tưởng

Các công cụ đánh giá agent hiện có (DeepEval, RAGAS, promptfoo) coi agent là hộp đen: người dùng tự
viết user query và đáp án tham chiếu, chạy một lần, chấm ngữ nghĩa. ARTEMIS đi vào **cấu trúc bên
trong system prompt**: mỗi phần của prompt phải được ít nhất một test case chạm tới.

Với mỗi agent:

1. **Phân rã** system prompt thành tập requirement `R(P) = T ∪ AC ∪ CC`
   - `T` (task) — việc agent phải làm. 2 trạng thái: có yêu cầu / không yêu cầu.
   - `AC` (absolute constraint) — luật không được phá trong mọi trường hợp. 2 trạng thái: không
     thách thức / thách thức.
   - `CC` (conditional constraint) — luật chỉ áp dụng khi điều kiện xảy ra. 4 trạng thái: tổ hợp
     (điều kiện có/không) x (thách thức có/không).
2. **Sinh test case**. Mỗi requirement là một factor. Liệt kê hết tổ hợp thì tốn `2^|T| · 2^|AC| · 4^|CC|`
   test case, nên dùng **pairwise covering array**: mọi cặp trạng thái giữa hai requirement bất kỳ
   phải xuất hiện trong ít nhất một test case. Thực tế còn 5-20 test case mỗi agent.
   Mỗi test case gồm user query và bộ evaluation criteria, đều do LLM sinh.
3. **Chạy** mỗi test case `n_run` lần (mặc định 5), vì cùng một query LLM vẫn trả lời khác nhau.
4. **Chấm** bằng LLM-as-a-judge, thang 1-5 cho từng criterion: 1 là vi phạm hoàn toàn, 3 là đạt một
   phần, 5 là đạt đủ.

## Hai metric

- **Accuracy Q** — trung bình composite score trên mọi cặp (test case, run). Composite score của một
  lần chạy là trung bình điểm các criterion.
- **Instability S** — trung bình của `sigma_i / mu_i` trên các test case, tức hệ số biến thiên của
  điểm qua các lần chạy. Thấp là agent trả lời nhất quán.

Hai metric này **không tương quan** (Pearson |r| <= 0.067 trên 71 agent), nên phải đọc cùng nhau.
Paper chia bốn trường hợp chẩn đoán: S thấp + Q thấp là lỗi hệ thống trong prompt, sửa prompt;
S thấp + Q cao là tốt; S cao + Q cao thì đừng tin điểm trung bình, phải soi từng run; S cao + Q thấp
là khó nhất, vừa phải làm rõ prompt vừa phải xem lại model.

Lưu ý khi đọc điểm: user query được cố ý sinh ra để thách thức constraint, nên Q tập trung quanh
2.5-3.5 là bình thường. Q **không** phản ánh agent chạy tốt thế nào trong tình huống thường. Agent
có prompt càng dài, càng nhiều luật thì Q càng thấp, đơn giản vì có nhiều requirement để vi phạm hơn.

## Ba phát hiện đáng dùng lại

- **Ba loại lỗi phổ biến** trên 3.519 lần chạy trượt: `ac-refuse` 37,1% (user query dụ phá absolute
  constraint và agent nghe theo), `task-infer` 36,3% (prompt mơ hồ, agent không suy ra được yêu cầu
  ngầm), `task-exec` 22,3% (response thiếu). Đây là taxonomy cho stage 12.
- **Decomposition là thành phần quan trọng nhất**. Bỏ bước phân rã T/AC/CC và sinh test case thẳng từ
  cả prompt thì số lỗi phát hiện được **giảm 5,6 lần** — LLM chỉ sinh test case xoay quanh Task và
  gần như bỏ qua AC/CC.
- **n_judge = 1 là đủ**. Tăng số vòng chấm từ 1 lên 5 chỉ đổi accuracy tối đa 0,01 khi judge chạy ở
  temperature 0, trong khi bước judge ngốn khoảng 1,005 triệu token mỗi agent. Ngược lại `n_run` thì
  phải giữ lớn: instability trung bình tăng 0,042 (3 run) lên 0,076 (10 run), chạy ít thì không lộ ra.

## Ví dụ tạo động lực

Hệ `langchain-multi-agents` có 4 agent theo vòng writer-reviewer. Chạy 200 lần cùng một query,
`ScenarioWriter` phát `FINAL ANSWER` ngay sau khi viết xong kịch bản trong **200/200 lần**, làm
pipeline dừng trước khi tới reviewer. Nguyên nhân là prompt viết "khi công việc hoàn thành thì phát
FINAL ANSWER" mà không nói rõ "hoàn thành" là xong phần của mình hay xong cả pipeline. Đây là ví dụ
mẫu cho cả stage 12 (vá prompt) lẫn stage 14 (đo dừng sớm).

## Cấu trúc repo gốc

Package `artemis/`, chạy qua `run_pipeline.py`, kết quả vào `output/run_pipeline_<N>/`.

- `nodes/phase1_nodes.py` — test từng agent, ra Q và S. Đây là phần đã có kết quả.
- `nodes/phase2_nodes.py`, `nodes/phase2_router_tester.py` — test router.
- `nodes/phase3_nodes.py` — end-to-end, hiện chạy bằng **mock agent**.
- `agents/task_decomposer.py`, `agents/state_combiner.py`, `agents/criteria_writer.py` — phân rã,
  pairwise, sinh criteria.
- `extractors/langgraph_detector.py`, `graph_assembler.py`, `template_resolver.py` — static analysis,
  **chỉ LangGraph**.
- `tools/wrapper_generator.py` — mầm cho phần tooling, chưa dùng tới.
- `model_factory.py` — chỗ chọn model. Dựng LLM bằng LangChain (`ChatOpenAI`), lấy thông số từ
  `INTERNAL_AGENT_CONFIGS` và `TEST_MODEL_CONFIGS` trong `config.py`. Thêm provider mới thì thêm
  entry vào hai dict đó, không viết wrapper.
- `benchmarks/test_system/` — đủ 16 hệ LangGraph của paper, đánh số 1 đến 16.
- `docs/` — `ARTEMIS_MODULE_GUIDE.md`, `ARTEMIS_WORKFLOW.md`, `EXPERIMENT_CONFIG.md`.
- `runs/ablation/`, `analysis/scripts/` — kết quả cũ và script dựng bảng, biểu đồ.

Cờ CLI đáng nhớ nhất là `--load-test-cases-from <path>`: chạy lại đúng bộ test case cũ. Không có nó
thì stage 12 không so sánh trước/sau được, vì prompt đổi sẽ kéo theo requirement đổi và test case đổi.

## Bốn giới hạn paper tự nêu

Đây chính là bốn stage tiếp theo:

- Test case **không thực tế**: test case của agent B sinh từ prompt của B, nhưng đầu vào thật của B
  là output của A. Có thể A không bao giờ tạo ra được tình huống đó. Xem `13_stage2_integration.md`.
- **Bỏ qua tooling**: chỉ chấm hành vi sinh từ system prompt, không mô phỏng tool call.
  Xem `15_stage4_rag_tooling.md`.
- **Chỉ LangGraph**: chưa thử trên AutoGen, CrewAI.
- **Một user query cho mỗi test state** và chưa có vòng cải tiến prompt. Xem
  `12_stage1_prompt_improvement.md`.
