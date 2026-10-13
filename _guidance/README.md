# `_guidance/` — Hướng dẫn chạy theo kịch bản

Thư mục này là tài liệu vận hành, viết bằng tiếng Việt, đọc theo thứ tự số. Trong khi
README của từng thư mục code trả lời câu hỏi "cái này là gì", `_guidance/` trả lời câu
hỏi "tôi phải gõ gì, theo thứ tự nào".

## Đọc theo thứ tự

| File | Nội dung |
|---|---|
| `00_setup.md` | Cài môi trường, dữ liệu nào có sẵn / phải tải, key nào cần. |
| `01_demo_fast.md` | Chạy trọn vòng (tối ưu → tiêm → đo) trong ~20 phút để kiểm tra pipeline còn sống. Số liệu không dùng để báo cáo. |
| `02_evaluation.md` | Bốn chỉ số ACC, ASR-r, ASR-a, ASR-t nghĩa là gì và đọc thế nào. |
| `03_attack.md` | Giải thích thuật toán tối ưu trigger và các tham số quan trọng. |
| `04_end_to_end.md` | Chạy một thí nghiệm đầy đủ để lấy số báo cáo, gồm cả agent `ad`. |
| `19_agentpoison_margin_implementation.md` | Hợp đồng code và vận hành so sánh AgentPoison gốc với retrieval-margin loss trên StrategyQA. |
| `20_mcat_kaggle_runbook.md` | Chạy MCAT M0–M2 trên Kaggle (`scripts/run_mcat.sh`). Hướng generator đã dừng; giữ để tái lập. |
| `22_mcat_m3_drift_plan.md` | Kế hoạch M3 của MCAT (memory drift, adaptation, break-even). Lịch sử; hướng đã dừng. |
| `23_aqua_agent_generation_plan.md` | **Kế hoạch** bước kế tiếp của AQuA (đang tạm gác): agent tự sinh tool call thay cho candidate replay, và tập clean benign. Chưa có code. |
| `24_p0_probe_kaggle_runbook.md` | Chạy `src/attack/dilution/` trên Kaggle qua `scripts/run_p0_probes.sh` (probe P0/P1, write-back, e2e). |
| `25_helper_tong_hop.md` | **Đọc đầu tiên nếu không muốn đọc code**: bản đồ thư mục, từ vựng, config, từng stage của pipeline `dilution/` (probe P0, write-back, e2e) và Kaggle, output nằm đâu, lỗi thường gặp. |
| `run.md` | Ghi chú chạy demo Corba. |

## Hướng mở rộng ARTEMIS (kiểm thử prompt của MAS)

Nhánh nghiên cứu thứ hai của repo, dựa trên paper ARTEMIS trong `survey/`. Khác với nhóm
`00`-`04` là hướng dẫn vận hành, nhóm này là kế hoạch triển khai.

| File | Nội dung |
|---|---|
| `10_artemis_overview.md` | Phương pháp ARTEMIS, hai metric, ba phát hiện dùng lại được, cấu trúc repo gốc. Đọc trước tiên. |
| `11_stage0_setup.md` | Vendor code gốc, cấu hình ba vai model, chạy lại baseline. |
| `12_stage1_prompt_improvement.md` | Vòng cải tiến prompt từ điểm số. Ưu tiên cao nhất. |
| `13_stage2_integration.md` | Hợp đồng giữa cặp agent, tỉ lệ khả đạt. |
| `14_stage3_system_testing.md` | Kiểm thử toàn luồng: độ phủ, lan truyền lỗi, dừng sớm. |
| `15_stage4_rag_tooling.md` | Taxonomy TC/GC, RAG bị đầu độc, hỗ trợ AutoGen. |
| `16_roadmap.md` | Bảng tổng, phụ thuộc giữa stage, ngân sách token, rủi ro. |
| `17_trang_thai_va_ket_noi_toolpoison.md` | Trạng thái thực tế so với roadmap, cách nối demo tool-calling vào Stage 4 (demo đã xoá 2026-10-13; giữ làm lịch sử). |

Đã xoá ngày 2026-10-13 cùng phần code tương ứng: `05_toolpoison_demo.md`,
`18_agentpoison_phases.md` (`src/agentpoison/`), `21_specificity_pipeline_stages_agentpoison.md`
(`src/attack/specificity/`). Lấy lại từ git nếu cần.

## Nguyên tắc chung khi chạy

1. Luôn chạy lệnh từ **thư mục gốc repo** (nhiều đường dẫn trong code là đường dẫn tương đối).
2. Với pipeline retrieval-margin, truyền `trigger.json` và `split.json` cho inference;
   không dán WordPiece vào source code.
3. Pipeline mới ghi checkpoint và tự bỏ qua iteration/episode đã hoàn thành; giữ nguyên
   cấu hình và dùng `--resume` khi chạy lại.
4. Cache embedding của `algo/` đặt tên theo `model_code`; thay checkpoint mà giữ nguyên tên thì xoá
   cache cũ (xem `algo/README.md` §4).
