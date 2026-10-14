# Ý tưởng lưu trữ

Cập nhật **2026-10-14**. Lưu trữ gồm tài liệu nguồn chi tiết và kế hoạch cũ; không mặc định đã bị bác bỏ.
Nội dung cũ được giữ để truy vết; các mốc “hiện hành” và kế hoạch trong thân bài là lịch sử.
**Bản tổng hợp hiện hành để phản biện:** [REVIEW_FOR_CLAUDE.md](../REVIEW_FOR_CLAUDE.md).

## Tài liệu nguồn gom lại ngày 14/10/2026

Mười file trước đây ở thư mục gốc được chuyển vào [research_2026_10_14/](research_2026_10_14/).
Giữ nội dung và lịch sử phản biện, cập nhật đường dẫn khi di chuyển. Các kết luận “hiện hành”
trong từng bản là trạng thái tại thời điểm viết; không cần gửi cả thư mục này cho Claude.

| Nhóm tài liệu | File nguồn |
|---|---|
| Ba hướng N1/N2/N3 | [Đề xuất đầy đủ](research_2026_10_14/static_memory_attack_new_directions_2026_10_14.md) |
| A/B và phản biện | [Cơ chế, review và phản hồi](research_2026_10_14/static_memory_mechanism_gap_analysis_2026_10_14.md) |
| Peers và chi phí | [Ledger limitations](research_2026_10_14/static_memory_peer_limitations_2026_10_14.md), [trigger/algorithm review](research_2026_10_14/static_trigger_peers_review_2026_10_14.md) |
| Rà soát ngày 13/10 | [AgentPoison](research_2026_10_14/agentpoison_attack_first_review_2026_10_13.md), [các lựa chọn attack](research_2026_10_14/static_memory_attack_options_2026_10_13.md), [trạng thái sau run 8](research_2026_10_14/memory_attack_research_status_2026_10_13.md) |
| Literature và nhánh cũ | [Literature](research_2026_10_14/memory_poisoning_literature_2026.md), [write-back defense](research_2026_10_14/writeback_repair_defense_idea.md) |
| Nhật ký tiêu chí thực nghiệm | [Open problems / các cổng đã đăng ký](research_2026_10_14/open_problems.md) |

## Ý tưởng đã lưu trước đó

| Ghi chú | Lý do / trạng thái |
|---|---|
| [Rà soát MCAT](mcat_direction_review_2026_10_10.md) | Hồ sơ quyết định dừng đầu tư generator ở định vị đã thử; không phải chứng minh mọi generator vô ích |
| [Thiết kế MCAT](memory_conditioned_generator_Q1_A_star.md) | Gate lợi ích conditioning và drift chưa ủng hộ hướng cũ |
| [Roadmap attack cũ](attack_first_roadmap.md) | Kế hoạch trước các kết quả P0; thứ tự đầu tư đã thay đổi |
| [Backdoor qua vòng ghi](self_updating_memory_backdoor_idea.md) | Claim rộng propagation/ngưỡng/provenance chồng lấn literature; bằng chứng thực nghiệm có giới hạn theo model và writer |
| [Attack từ write-back](writeback_attack_proposals_2026_10_10.md) | Bản đề xuất trước run 8 và audit 12/10; các giả thuyết chưa bị bác bỏ hết, chỉ được tiếp tục qua pilot phân biệt cơ chế |
| [Các hướng memory khác](memory_attack_directions_2026_10_10.md) | Danh sách dự phòng cần hạ tầng/pilot riêng, chưa xác nhận novelty |
| [AQuA](paper_Q1_A_plus.md) | Nhánh defense riêng tạm gác, chưa bị bác bỏ |
| [Trigger theo nhóm](group_conditioned_triggers.md) | Pilot StrategyQA trước P0 |
| [Phân cấp trigger](trigger_hierarchy_pilot_results.md) | Kết quả pilot lịch sử |
| [Specificity](increase_specificity_results.md) | Kết quả pilot lịch sử |
| [Audit ngôn ngữ](trigger_language_audit.md) | Ghi chú pilot lịch sử |
| [Ba mức ngôn ngữ](trigger_three_levels_language.md) | Ghi chú pilot lịch sử |

Báo cáo P0 giữ riêng trong [results](../results/). Trạng thái novelty mới phải đọc cùng
[literature](research_2026_10_14/memory_poisoning_literature_2026.md) và
[audit sửa memory 12/10](../../reports/Kiểm%20chứng%20gap%20sửa%20memory.md).
