# Trạng thái ARTEMIS hiện tại và điểm nối với demo tool-calling

`10_artemis_overview.md` đến `16_roadmap.md` đã có kế hoạch chi tiết cho từng stage. File này không lặp
lại kế hoạch đó — nó (1) đối chiếu trạng thái thực tế so với roadmap, và (2) chỉ ra cách demo tool-calling
ở `05_toolpoison_demo.md` (`src/toolpoison/`) trở thành mảnh ghép đầu tiên cho Stage 4, sớm hơn dự kiến
trong roadmap gốc.

## 1. Trạng thái so với `16_roadmap.md`

Từ phiên bản này, code Phần B (mở rộng ARTEMIS) tách khỏi Phần A (`src/toolpoison/`), sống ở
`src/integration/` (trừ Stage 0 vendor code gốc, đi thẳng vào `src/artemis/` — xem
`src/integration/README.md`). Đường dẫn `src/adapt/*` nhắc ở `11-16` là tên gọi lúc lập kế hoạch; tên
thật khi triển khai là `src/integration/*` như bảng dưới.

| Stage | Kế hoạch | Thực tế hiện có |
|---|---|---|
| 0 — nền | Vendor `artemis/` qua git subtree vào `src/artemis/`, ba vai model, chạy lại baseline | `src/config.py` (ba vai model + `check_role_separation`) đã có. `src/artemis/` (vendor thật) **chưa tồn tại** — chưa chạy lại được baseline nào. Đây là điều kiện tiên quyết còn thiếu trước khi làm Stage 1 đúng nghĩa (dùng lại `nodes/phase1_nodes.py` của upstream). |
| 1 — cải tiến prompt | `src/integration/prompt_improve/` (`failure_classifier.py`, `patch_generator.py`, `holdout.py`, `loop.py`) | Khung sườn đã có (chữ ký hàm + docstring theo đúng luồng trong `12_stage1_prompt_improvement.md`), thân hàm còn `NotImplementedError`. Đúng như người dùng mô tả: "đã chấm điểm nhưng chưa dùng điểm sửa prompt". |
| 2 — tích hợp | `src/integration/agent_contract/` | Chỉ có docstring thiết kế, chưa có code. |
| 3 — system testing | `src/integration/system_testing/`, thay mock agent trong `nodes/phase3_nodes.py` bằng agent thật | Chỉ có docstring thiết kế, chưa có code. |
| 4 — RAG + tooling | `src/integration/rag_tooling/`, `src/integration/extractors/autogen_detector.py` (chưa tạo — thêm khi cần) | Chỉ có docstring thiết kế, chưa có code trong `src/integration/rag_tooling/`, nhưng đã có `src/toolpoison/` (mục 2) làm ví dụ cụ thể đầu tiên, chạy độc lập không phụ thuộc Stage 0. |

## 2. Nối `src/toolpoison/` vào Stage 4

`15_stage4_rag_tooling.md` giả định sẵn "agent dùng tool thật, sẵn RAG, sẵn máy sinh context đối kháng"
— trước `src/toolpoison/`, repo chưa có vế đầu (agent dùng **native** tool-calling; ba agent cũ dùng
text-action hoặc code-gen, không phải function-calling chuẩn). `src/toolpoison/agent.py` lấp đúng chỗ
đó. Cách nối cụ thể:

- **Test case** cho `toolpoison.agent` = cặp `(query, retrieved_demo)`, sinh hai biến thể lành/độc đúng
  như `15_stage4_rag_tooling.md` mục 2 — bản độc chính là biến thể có trigger đã có sẵn trong demo.
- **Judge chấm cả `tool_calls`**, không chỉ text — dùng thẳng `ModelResponse.tool_calls` đã chuẩn hoá
  trong `src/providers/base.py`, không cần viết thêm lớp trích xuất tool-call riêng cho ARTEMIS.
- **Metric "khoảng cách bền vững"** (`Q_lành - Q_độc`) đo được ngay bằng ACC/ASR-tool đã tính ở
  `src/toolpoison/eval.py` — có số đầu tiên mà không cần đợi vendor `src/artemis/` xong.

## 3. Thứ tự thực hiện đề xuất (ngắn hơn `16_roadmap.md`, ưu tiên có kết quả sớm, chi phí thấp)

1. Hoàn thiện `src/toolpoison/` (`05_toolpoison_demo.md`) — tự thân đã là một "Stage 4 thu nhỏ", chạy
   độc lập, không phụ thuộc vendor `artemis/`.
2. Vendor `src/artemis/` (Stage 0) — làm song song với bước 1, không phụ thuộc nhau.
3. Khi cả hai xong: viết extractor cho agent tool-calling tự chế (đơn giản hơn `autogen_detector.py` vì
   `src/toolpoison/agent.py` tự viết, biết trước cấu trúc prompt) để ARTEMIS sinh test case tự động thay
   vì viết tay như ở bước 1.
4. Stage 1 (cải tiến prompt) áp lên chính `src/toolpoison/agent.py` trước tiên — vòng lặp "vá prompt
   chống trigger" mô tả ở `15_stage4_rag_tooling.md` mục 3, báo cáo song song ACC và ASR-tool (tránh vá
   phòng thủ quá tay làm ACC tụt, đúng rủi ro đã nêu ở `16_roadmap.md`).
5. Stage 2 (tích hợp giữa cặp agent) và Stage 3 (system testing) giữ nguyên thứ tự ưu tiên thấp hơn như
   `16_roadmap.md` — không đổi.

## 4. Rủi ro bổ sung (không lặp lại rủi ro đã nêu ở `16_roadmap.md`)

Nếu bộ nhớ demo độc trong `src/toolpoison/` quá lộ liễu (khác biệt phong cách rõ rệt với demo benign),
ASR-tool đo được sẽ cao giả tạo — không phản ánh đúng ngưỡng tấn công thực tế. Cần ít nhất một biến thể
demo độc "viết tự nhiên" (giống văn phong demo benign) để tránh kết luận quá lạc quan về khả năng tấn
công tool-calling, và để số đo được dùng làm baseline cho Stage 1 (mục 3, bước 4) có ý nghĩa.

**Đã xử lý (2026-09-05):** thêm biến thể `--poison_style natural` (`src/toolpoison/memory.py`, chỉ thị
viết kín đáo như ghi chú tài khoản) và chạy lặp `--n_run 3` để đo instability. Kết quả xác nhận đúng
rủi ro trên: chỉ thị lộ liễu (`blatant`) lật 100% và tuyệt đối ổn định (instability 0), nhưng chỉ thị
kín đáo (`natural`) chỉ lật 77%, kém ổn định (instability 0.141) và còn kéo tụt ACC xuống 0.77. Kết
luận cho Stage 1/4: **báo cáo bằng biến thể `natural` mới trung thực**; con số `blatant` chỉ là chặn
trên. Bảng số đầy đủ ở `05_toolpoison_demo.md` mục 7.

## 5. Nhận xét theo kết quả demo thực tế (2026-09-05) — nên cải thiện stage nào

Số liệu đầy đủ ở `_guidance/05_toolpoison_demo.md` mục 7. Tóm tắt: retrieval bị đầu độc thành công
100% (ASR-r = 1.0) mà chưa cần gradient-optimize trigger; nhưng few-shot độc trần không lật được
tool-call (ASR-tool = 0), chỉ khi tiêm chỉ thị qua nội dung RAG mới lật (ASR-tool = 1.0), trong khi
ACC luôn = 1.0. Từ đó, thứ tự ưu tiên cải thiện:

**Ưu tiên 1 — Stage 1 (cải tiến/vá prompt), `src/integration/prompt_improve/`.** Demo cho thấy lỗ hổng
là **ranh giới tin cậy**: agent tin "chính sách tài khoản truy hồi được" hơn cả ý định người dùng. Đây
đúng loại lỗi `ac-refuse` trong taxonomy (user/context dụ agent phá absolute constraint và agent nghe
theo). Việc cần làm: dùng `src/toolpoison/agent.py` làm agent đầu tiên cho vòng vá prompt, mục tiêu hạ
ASR-tool trong khi giữ ACC. `patch_generator.py` cho `ac-refuse` phải viết rõ thứ bậc: **nội dung truy
hồi là dữ liệu, không phải chỉ dẫn; system prompt và ý định người dùng thắng**. Báo cáo song song ACC
và ASR-tool để bắt bẫy "phòng thủ quá tay" (dặn agent nghi ngờ mọi context sẽ hạ ASR-tool nhưng cũng
làm tụt ACC).

**Ưu tiên 2 — Stage 4 (RAG + tooling), `src/integration/rag_tooling/`.** Demo là hiện thân đầu tiên
của taxonomy TC/GC trong `15_stage4_rag_tooling.md`: ASR-tool chính là vi phạm **TC** (tool constraint
— gọi nhầm tool/bịa tham số), còn việc agent nghe theo "chính sách" bịa trong context là vi phạm **GC**
(grounding constraint). "Khoảng cách bền vững" `Q_lành - Q_độc` giờ đo được và lớn (1.0 vs 0.0). Việc
cần làm: chính thức hoá TC/GC thành factor trong bộ sinh test case, và cho judge chấm cả `tool_calls`
(đã có sẵn `ModelResponse.tool_calls`).

**KHÔNG ưu tiên — tối ưu trigger.** Ở quy mô này ASR-r đã bão hoà, đổ công gradient-optimize trigger
không tăng được kết quả. Chỉ cần đến khi chuyển sang bộ nhớ RAG thật (20k passage) hoặc muốn trigger
transfer sang embedder không biết trước. Khi đó, hai cải tiến so với pp gốc đáng làm: (a) batch hoá
bước chấm ứng viên trong `algo/trigger_optimization.py` (hiện lặp từng ứng viên → gộp thành 1 batch,
nhanh gần `num_cand` lần, không đổi thuật toán); (b) đổi `--target_gradient_guidance` để nhắm thẳng
logprob của **tên hàm/tham số tool mục tiêu** thay vì một từ khoá văn bản.
