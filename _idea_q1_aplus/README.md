# Ý tưởng nghiên cứu bảo mật agent

Cập nhật định hướng: **2026-10-10**.

Đây là tài liệu nghiên cứu bảo mật có kiểm soát, nhằm phát hiện, đo lường và khắc phục
rủi ro của model/agent có memory tự cập nhật. Thí nghiệm chỉ thực hiện trên testbed
do nhóm quản lý hoặc được cho phép, dùng dữ liệu được phép và hành động mô phỏng.
Không sử dụng kết quả để gây hại, xâm nhập hệ thống bên thứ ba hay phát tán nội dung
đầu độc ra môi trường thật. Nếu xác nhận vấn đề có ảnh hưởng thực tế, mục tiêu là
cung cấp bằng chứng tái lập và biện pháp giảm thiểu cho bên phát triển model/agent.

**Thứ tự nghiên cứu giữ nguyên: literature gap → attack trong sandbox → phân tích
cơ chế → defense → kiểm chứng lại cả hiệu quả và utility.**

Đọc theo thứ tự:

Các hướng thay thế đang sàng lọc: [memory attack directions, 2026-10-10](memory_attack_directions_2026_10_10.md).

Sau khi p1/p5 và cleanup hoàn tất: [các attack rút trực tiếp từ kết quả write-back](writeback_attack_proposals_2026_10_10.md).
Ưu tiên kiểm tra ảnh hưởng sau cleanup và trong giai đoạn sửa memory trước khi mở hướng xa hơn.

1. [Rà soát MCAT ngày 2026-10-10](mcat_direction_review_2026_10_10.md): khuyến nghị dừng
   đầu tư generator ở định vị hiện tại; bằng chứng, giới hạn và điều kiện mở lại.
2. [Ý tưởng hiện hành: backdoor qua vòng ghi memory](self_updating_memory_backdoor_idea.md).
   Có bản mô tả dùng khi trao đổi với trợ lý nghiên cứu, threat model, gap ứng viên,
   sửa công thức ngưỡng và tiêu chí bác bỏ.
3. [Literature và cập nhật ngày 2026-10-10](memory_poisoning_literature_2026.md).
   Các ghi chú cũ được giữ để truy vết; cập nhật mới thu hẹp claim novelty.
4. [Open problems và lịch sử quyết định](open_problems.md),
   [kết quả retrieval lần 6](results/p0_run6_rerun_results.md),
   [kết quả end-to-end lần 7](results/p0_run7_e2e_results.md).

## Cấu trúc thư mục (dọn ngày 2026-10-10)

| Vị trí | Nội dung |
|---|---|
| gốc | hướng hiện hành, rà soát MCAT, literature, `open_problems.md` (nhật ký tiêu chí chốt trước) |
| [`results/`](results/) | báo cáo P0 lần 1–7, mỗi số truy được về artifact trong `outputs/kaggle/` |
| [`archive/`](archive/) | lịch sử, không còn là kế hoạch hiện hành: thiết kế MCAT, roadmap 2026-09-24, đề xuất AQuA, các pilot trigger ngày 2026-09-15 |

Trong `archive/`: [MCAT](archive/memory_conditioned_generator_Q1_A_star.md) và
[roadmap](archive/attack_first_roadmap.md) đã dừng theo kết quả P0;
[AQuA](archive/paper_Q1_A_plus.md) là đề xuất defense riêng đang tạm gác, chưa bị bác bỏ;
[trigger theo nhóm](archive/group_conditioned_triggers.md),
[phân cấp](archive/trigger_hierarchy_pilot_results.md),
[specificity](archive/increase_specificity_results.md),
[ngôn ngữ trigger](archive/trigger_language_audit.md),
[ba mức ngôn ngữ](archive/trigger_three_levels_language.md) là pilot StrategyQA trước P0.

Các tên Q1/A* biểu thị mục tiêu công bố, không phải đánh giá đã được xác nhận về
novelty hay chất lượng. Phân biệt rõ kết quả đã đo, giả thuyết, mô hình toán và kế hoạch.
