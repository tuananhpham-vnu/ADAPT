# Ý tưởng nghiên cứu bảo mật agent

Cập nhật định hướng: **2026-10-13**.

Đây là tài liệu nghiên cứu bảo mật có kiểm soát, nhằm phát hiện, đo lường và khắc phục
rủi ro của model/agent có memory tĩnh hoặc tự cập nhật. Thí nghiệm chỉ thực hiện trên testbed
do nhóm quản lý hoặc được cho phép, dùng dữ liệu được phép và hành động mô phỏng.
Không sử dụng kết quả để gây hại, xâm nhập hệ thống bên thứ ba hay phát tán nội dung
đầu độc ra môi trường thật. Nếu xác nhận vấn đề có ảnh hưởng thực tế, mục tiêu là
cung cấp bằng chứng tái lập và biện pháp giảm thiểu cho bên phát triển model/agent.

**Thứ tự nghiên cứu giữ nguyên: literature gap → attack trong sandbox → phân tích
cơ chế → defense → kiểm chứng lại cả hiệu quả và utility.** Run 1–8 cung cấp bằng chứng
để chọn hướng; chưa xác nhận một đóng góp attack mới hoặc chốt chuyển sang defense.

Đọc theo thứ tự:

0. [**Các hướng attack-first trên memory tĩnh**](static_memory_attack_options_2026_10_13.md):
   bốn phương án, peers, limitation, pilot và điều kiện dừng; chưa chốt method.
   [Rà soát AgentPoison trước đó](agentpoison_attack_first_review_2026_10_13.md)
   giữ làm tham khảo code và baseline; không phải hướng người dùng đã chọn.
1. [**Trạng thái và cổng chọn hướng memory attack**](memory_attack_research_status_2026_10_13.md):
   ý tưởng attack giữ để sàng lọc, giới hạn run 8, tiêu chí tiếp tục/dừng và vị trí của defense.
2. [Nhánh ứng viên: sửa memory qua write-back](writeback_repair_defense_idea.md).
   Giữ thiết kế và đối chứng; chưa xác nhận novelty hoặc chọn làm hướng chính.
3. [Literature](memory_poisoning_literature_2026.md) và
   [audit gap sửa memory ngày 12/10](../reports/Kiểm%20chứng%20gap%20sửa%20memory.md):
   claim rộng về persistence/correction không đủ mới.
4. [Open problems](open_problems.md): nhật ký tiêu chí chốt trước của từng lần chạy.
5. Kết quả gần nhất: [run 7 end-to-end (Llama)](results/p0_run7_e2e_results.md),
   [run 8 (Qwen, key tóm tắt)](results/p0_run8_results.md).

## Cấu trúc thư mục (dọn lại ngày 2026-10-13)

| Vị trí | Nội dung |
|---|---|
| gốc | trạng thái chọn đề tài, nhánh defense ứng viên, literature, `open_problems.md` |
| [`results/`](results/) | báo cáo P0 lần 1–8, mỗi số truy được về artifact trong `outputs/kaggle/` |
| [`archive/`](archive/README.md) | ý tưởng và kế hoạch không còn hiện hành; phân biệt dừng theo bằng chứng, đã thay thế và tạm gác |

Trong `archive/`:

| File | Trạng thái |
|---|---|
| [rà soát MCAT 2026-10-10](archive/mcat_direction_review_2026_10_10.md) | hồ sơ quyết định dừng đầu tư MCAT ở định vị cũ |
| [backdoor qua vòng ghi memory](archive/self_updating_memory_backdoor_idea.md) | claim rộng đã chồng lấn literature; độ bền cao ở Llama p5 không chuyển sang Qwen |
| [attack từ kết quả write-back](archive/writeback_attack_proposals_2026_10_10.md) | bản đề xuất trước run 8/audit 12/10; giữ giả thuyết có điều kiện, chưa chứng minh method |
| [các hướng attack memory khác](archive/memory_attack_directions_2026_10_10.md) | không chọn (cần hạ tầng mới); bảng "các hướng đã đông" vẫn dùng khi kiểm novelty |
| [MCAT](archive/memory_conditioned_generator_Q1_A_star.md), [roadmap](archive/attack_first_roadmap.md) | đã dừng theo kết quả P0 |
| [AQuA](archive/paper_Q1_A_plus.md) | đề xuất defense phân quyền tool, tạm gác, chưa bị bác bỏ |
| [trigger theo nhóm](archive/group_conditioned_triggers.md), [phân cấp](archive/trigger_hierarchy_pilot_results.md), [specificity](archive/increase_specificity_results.md), [ngôn ngữ trigger](archive/trigger_language_audit.md), [ba mức ngôn ngữ](archive/trigger_three_levels_language.md) | pilot StrategyQA trước P0 |

Các tên Q1/A* biểu thị mục tiêu công bố, không phải đánh giá đã được xác nhận về
novelty hay chất lượng. Phân biệt rõ kết quả đã đo, giả thuyết, mô hình toán và kế hoạch.
