# Hướng memory attack: trạng thái và cổng chọn đề tài

> **Cập nhật 14/10/2026:** trọng tâm hiện tại là attack-first với trigger trên memory tĩnh,
> tìm hướng khác sau phản biện A/B. Đọc [đề xuất mới N1/N2/N3](static_memory_attack_new_directions_2026_10_14.md):
> ưu tiên kiểm cơ chế poison qua embedding nén; nhiều trigger–mục tiêu là phương án dễ pilot hơn;
> ANN traversal poisoning giữ dự phòng rủi ro cao. Chưa xác nhận novelty hoặc chất lượng A/A*.
> Đối chiếu thêm Eyes-on-Me, UniC-RAG, KEPo, DRNOISE; xem
> [bảng peers và limitations hiện hành](static_memory_peer_limitations_2026_10_14.md).
> [Phân tích cơ chế A/B trước đó](static_memory_mechanism_gap_analysis_2026_10_14.md) giữ làm hồ sơ
> kiểm chứng và phản biện; không còn là thứ tự ưu tiên hiện hành. Chưa có attack result mới.
> [Phản hồi review 14/10](static_memory_mechanism_gap_analysis_2026_10_14.md#review-response)
> giữ A ở mức pilot có điều kiện, hạ B xuống diagnostic tùy chọn; sửa các suy luận về
> selectivity/feasibility và bổ sung cổng kiểm chứng, hạn mức cùng yêu cầu profile chi phí.
> Các pilot trong review trước chưa được chọn; kỹ thuật sẽ xét sau khi xác minh gap.
> Không mặc định AgentPoison làm gốc. Chưa chốt method; các ứng viên memory động
> bên dưới giữ làm tham khảo lịch sử.
> Kết quả và ngưỡng run 1–8 không thay đổi.

Ngày **2026-10-13**. Đây là bản tổng hợp định hướng hiện tại từ ghi chú và kết quả trong repo;
không phải vòng tra cứu literature mới, không phải method đã được xác nhận.

**Giữ hướng attack-first: tìm gap → attack trong testbed được phép → xác định cơ chế → defense.**
Run 7–8 là bằng chứng để chọn câu hỏi tiếp theo, chưa chứng minh đã hoàn tất một đóng góp attack mới.
Mục đích là phát hiện và khắc phục rủi ro cho model/agent; dùng dữ liệu được phép, hành động mô phỏng
và ngân sách hữu hạn. Không triển khai lên hệ thống bên thứ ba hoặc phương tiện thật.

## 1. Những gì kết quả hiện tại cho phép nói

| Bằng chứng | Diễn giải được giữ | Giới hạn |
|---|---|---|
| Llama p5: log 0.959; xoá seed 0.959 | Xoá seed không đủ ở thiết lập này | Chưa có nhiều thế hệ ghi tiếp sau cleanup |
| Llama p1: log 0.477; cleanup 0.479 | Ảnh hưởng hầu như còn nguyên sau cleanup | Trượt ngưỡng cleanup ≥ 0.5 đã đăng ký; không sửa ngưỡng hậu nghiệm |
| Qwen p5: base 0.396 → log 0.293 → cleanup 0.270 | Mức làm theo và động học phụ thuộc model | Không chuyển giao độ bền cao của Llama p5; không chứng minh mọi attack qua write-back đều thất bại |
| Qwen p1: base 0.189, mức nền sạch 0.168 | Hiệu ứng ban đầu nhỏ trên mức nền | ASR thô không đủ chứng minh persistence do attack |
| `corrected` giảm ASR ở hai model | Bản ghi đúng là đối chứng sửa chữa có tác dụng | Dùng ground truth; chưa có defense thực tế không cần oracle |
| Key tóm tắt v1: bản ghi tự ghi không vào top-5 | Writer suy biến, phải sửa thiết kế trước khi kết luận | Không dùng kết quả này để khẳng định trigger mất qua tóm tắt là nguyên nhân chữa backdoor |

Nguồn: [run 7](../../results/p0_run7_e2e_results.md), [run 8](../../results/p0_run8_results.md).
Mức nền sạch và `control` có trigger nhưng không poison là hai đối chứng khác nhau;
thí nghiệm mới cần đối chứng ghép cặp có cùng stream và chính sách ghi.

ASR thấp không đồng nghĩa làm đúng tác vụ. Luôn báo ACC, kích hoạt nhầm, số bản ghi và chi phí.
`verified` hiện dùng kiểm tra plan theo ground truth trong harness; không phải verifier thực tế
đã được chứng minh tin cậy. Không dùng kết quả này để nói mọi verifier đều vô dụng.

## 2. Các ý attack giữ lại để sàng lọc

Các hàng dưới là giả thuyết, chưa phải gap mới. Không train generator trước khi qua cổng.

| Ưu tiên | Câu hỏi attack | Pilot cần phân biệt | Khi nào dừng |
|---|---|---|---|
| 1 | Chọn seed theo ảnh hưởng sau vòng ghi có hơn chọn theo ASR tĩnh không? | Cùng ngân sách seed/token/search; ghép mức ASR ban đầu trên dev; so ảnh hưởng vượt đối chứng trên stream test sau cleanup | Không có lợi ích ngoài hiệu ứng ban đầu, hoặc chỉ tái hiện persistence đã biết |
| 2 | Attack cố định ban đầu có làm recovery thất bại vì một giả định cụ thể không? | Phân biệt correction không truy hồi, không được dùng, hậu duệ chưa thu hồi, và correction ngoài phạm vi áp dụng | Chỉ tái hiện stale retrieval, thiếu revocation closure hoặc correction sai scope |
| 3 | Bản ghi có action đúng nhưng kinh nghiệm sai có gây lỗi tác vụ sau không? | Tách kiểm tra action hiện tại khỏi tính đúng/khả năng áp dụng của nội dung lưu | Trùng cơ chế kinh nghiệm sai/khái quát sai của prior art, hoặc không có tác động về sau |

Chi tiết thiết kế lịch sử: [đề xuất attack write-back](../writeback_attack_proposals_2026_10_10.md).
Ưu tiên 1 chỉ là phép thử chọn hướng; ưu tiên 2 chịu rủi ro novelty cao theo
[audit gap sửa memory ngày 12/10](../../../reports/Kiểm%20chứng%20gap%20sửa%20memory.md).
Theo audit này, persistence sau thu hồi nguồn, chống textual correction và transfer sau repair
đều đã có nghiên cứu gần. Ghép các yêu cầu đó hoặc đổi objective không tự tạo đóng góp.

## 3. Pilot trước khi đầu tư method

1. Khóa quyền attacker: chỉ can thiệp một lần; tách rõ quyền chèn seed và quyền đưa trigger vào query.
   Không cho chỉnh poison, thêm seed hay thích nghi online sau thời điểm khóa ở arm chính.
2. Dùng cùng seed budget, stream và split cho baseline; khóa checkpoint cleanup, repair và ngưỡng
   trên dev. Đánh giá trên held-out stream, có nhánh không ghi và nhánh không poison.
3. Tiếp tục vận hành sau cleanup, ghi lineage để phân biệt bản ghi còn sót với hậu duệ mới.
   Không gọi một lần evaluate snapshot là tự nhân bản nhiều thế hệ.
4. Nếu nghiên cứu recovery, đặt cạnh textual correction, A-MemGuard, MemAudit và selective
   revocation/replay trong phạm vi tái lập được. Có oracle removal của seed + descendants và
   đối chứng đưa correction hợp lệ trực tiếp vào context. Báo giới hạn quyền/log của từng baseline.
5. Chỉ xây attack method nếu pilot chỉ ra được cơ chế nhân quả khác biệt và giả định baseline
   bị phá trong threat model đã khóa. Nếu không, giữ kết quả âm và chuyển ứng viên; chưa chốt paper.

Các prior-art/baseline ở đây lấy từ audit nội bộ đã liên kết; cần xác minh phương pháp và artifact
trước khi đưa vào protocol chính thức. Chốt tiêu chí mới trong [open problems](open_problems.md),
không sửa lại tiêu chí run 1–8.

## 4. Vị trí của defense

Giữ [đề xuất nguồn sửa qua write-back](writeback_repair_defense_idea.md) như **nhánh ứng viên**.
Các lựa chọn snapshot tin cậy, bỏ từng bản ghi, luật domain và phản hồi người hữu ích cho baseline
hoặc ablation. Chưa gọi “ghi corrective memory không cần biết trigger” là novelty đã xác nhận.
Audit 12/10 đã chỉ ra overlap với corrective lessons và recovery; phải đưa vào đánh giá trước
khi chốt method. Đổi từ attack sang defense tự nó không giải quyết vấn đề novelty.

Sau khi có failure mechanism cụ thể, defense phải xử lý đúng cơ chế đó, báo cả ASR, utility,
chi phí và đánh giá adaptive attacker với quyền/ngân sách công bố rõ. Không hứa bảo đảm mọi trigger.

## 5. Quyết định lưu trữ

- MCAT generator: dừng đầu tư ở định vị cũ vì gate chưa ủng hộ lợi ích conditioning/adaptation.
- “Tự nhân bản + R + provenance”: bỏ làm claim mới độc lập; prior art chồng lấn, bằng chứng nhiều thế hệ còn thiếu.
- Đề xuất write-back ngày 10/10: giữ lịch sử, thay cách ưu tiên bằng các cổng ở trên; Qwen không phải
  phép bác bỏ mọi phương pháp chọn seed theo rollout chưa được thử.
- Bốn hướng memory khác: dự phòng do chi phí và thiếu pilot, **chưa bị bác bỏ khoa học**.
- AQuA và pilot trigger cũ: giữ lịch sử; AQuA tạm gác, không suy ra đã thất bại.

Mục lục và lý do từng file: [archive](../README.md). Không xoá kết quả để làm câu chuyện đẹp hơn.
