# Defense sửa memory và khái quát sau sửa chữa

## Đã có defense đề xuất cơ chế và chứng minh điều gì?

### Takeaway
Phạm vi rộng “lesson giúp tránh lỗi tương tự” đã có A-MemGuard. Kết quả đọc trong nhánh này chưa thấy protocol hẹp đầy đủ; tuy nhiên nhánh nghiên cứu song song đã tìm được Revoked but Still Authoritative, nên không được giữ verdict gap hẹp còn trắng. Đây là kết luận rà soát có giới hạn, không phải bảo đảm novelty.

### Cited Findings
- **A-MemGuard**: §4.2 lưu reasoning path bất thường thành negative lesson, truy hồi theo plan trước hành động. Appendix G truy hồi hai bước theo query rồi action, nhằm dùng lesson từ câu hỏi khác nhưng thao tác tương tự. §5 đánh giá AgentPoison, MINJA, utility, ablation và top-k. Bảng 5: bỏ lessons đổi ASR-t 36.17 → 40.63; không phải bài kiểm tra split correction/held-out. Không nên diễn giải “giảm >95%” thành ASR-t mọi cấu hình. ArXiv v1 ngày 29/09/2025; venue xác nhận **ICML 2026**, kỳ hội nghị 06–11/07/2026. [Method/evaluation](https://arxiv.org/html/2510.02373#S4.SS2), [Appendix G](https://arxiv.org/html/2510.02373#A7), [arXiv lịch sử](https://arxiv.org/abs/2510.02373), [proceedings chính thức](https://proceedings.mlr.press/v306/wei26ac.html).
- **Memory Poisoning Attack and Defense on Memory Based LLM-Agents**: §§6.2–8 thêm moderation, trust score, decay và filtering ở append/retrieval; không tạo corrective lesson. Một cấu hình từ chối toàn bộ 23 candidate entries, nên memory rỗng; không chứng minh repair khái quát. ArXiv v1 09/01/2026, v2 12/01/2026; trang metadata chưa nêu venue. [§6.2 và đánh giá](https://arxiv.org/html/2601.05504#S6.SS2), [metadata](https://arxiv.org/abs/2601.05504).
- **TMA-NM**: §II đặt bất biến authority; đối kháng không sửa được origin/scope do monitor gán. §VI-B có multi-turn và Mem0 consolidation. §IX giới hạn: bảo vệ consequential action, không loại bỏ answer bias; cần origin và corroboration đúng. Đây là phòng ngừa authority laundering, không phải sửa lesson sau incident. ArXiv 23/06/2026; chưa xác minh venue. [Threat model](https://arxiv.org/html/2606.24322#S2), [multi-turn](https://arxiv.org/html/2606.24322#S6.SS2), [scope](https://arxiv.org/html/2606.24322#S9), [metadata](https://arxiv.org/abs/2606.24322).

### Inferences
- A-MemGuard là baseline bắt buộc: method đã nhắm tới chuyển lesson qua câu hỏi, nên “thêm corrective memory” hoặc “lesson generalization” riêng lẻ rất dễ trùng. [Cơ chế](https://arxiv.org/html/2510.02373#S4.SS2).
- TMA-NM có thể triệt tiêu một phần attack scope nếu kết quả đích là hành động cần authority. Cần xác định rõ đề tài về factual/decision bias hay authorization, không dùng việc attack bị gate làm bằng chứng correction thành công. [Giới hạn guarantee](https://arxiv.org/html/2606.24322#S9).

### Gaps
- Đã đọc toàn bộ các mục methods/evaluation liên quan trong HTML arXiv A-MemGuard; chưa đọc được PDF proceedings cuối: raw GitHub trả lỗi, OpenReview yêu cầu xác minh. Do vậy kết luận “chưa thấy protocol” chỉ chắc cho arXiv v1, không bảo đảm phiên bản proceedings không bổ sung thí nghiệm. [PDF chính thức được trỏ từ PMLR](https://raw.githubusercontent.com/mlresearch/v306/main/assets/wei26ac/wei26ac.pdf), [OpenReview](https://openreview.net/forum?id=udqe7UZUZ6).
- Các nguồn trên không mô tả rõ correction-set/test-set cùng phạm vi với attacker bị khóa sau seed. Không được biến thiếu mô tả thành khẳng định defense thất bại.

## Attack vượt defense và transfer sau sửa chữa có trùng gap không?

### Takeaway
“Vượt A-MemGuard” đã có FARMA; “accepted repair chưa chắc chuyển được sang task liên quan” đã có After the Fix. Hai kết quả làm mất tính mới của claim rộng, nhưng chưa ghép thành thí nghiệm security một lần đầu độc rồi sửa thật và đo tái phát.

### Cited Findings
- **FARMA/SENTINEL**, arXiv 06/07/2026: §§III-B/C đối kháng trực tiếp ghi seed và các bản amplification vào store; toàn bộ có thể dựng offline trước victim query. §V-D so sánh A-MemGuard heuristic/LLM, mặc định 10 amplification cycles. Không mô tả giai đoạn correction hợp lệ rồi reuse held-out. §VI thừa nhận adaptive paraphrase làm Reasoning Guard không bảo vệ đáng kể và simulation không bao quát multi-turn/write diversity. SENTINEL lọc trước commit, không repair sau incident. Chưa xác minh venue. [Attack](https://arxiv.org/html/2607.05029#S3), [evaluation](https://arxiv.org/html/2607.05029#S5), [limitations](https://arxiv.org/html/2607.05029#S6), [metadata](https://arxiv.org/abs/2607.05029).
- **After the Fix: Transfer of Corrected Agent Experience**: title hiện tại khác lead ban đầu. V1 28/09/2026, v2 29/09/2026. §§3–5 chuyển cùng source trước/sau accepted repair sang fixed related target, có baseline chạy mới. ThinkingBox tăng; APEX không có lợi ích aggregate tương đương. §8 để mở applicability conditions trên held-out và successive repairs trong continuing memory. Không có attacker/poison protocol: lỗi source tự nhiên, handoff một lần. Chưa xác minh venue. [Design/results](https://arxiv.org/html/2609.34603#S3), [open questions](https://arxiv.org/html/2609.34603#S8), [metadata](https://arxiv.org/abs/2609.34603).

### Inferences
- Không claim đầu tiên phát hiện corrected memory không chuyển tốt: After the Fix nghiên cứu đúng hiện tượng rộng này. Câu hỏi security cần yêu cầu correction vẫn đúng và áp dụng được cho target; source-local correction chuyển sai ngoài phạm vi chưa chứng minh poisoning sống sót. [§5.4/§8](https://arxiv.org/html/2609.34603#S5.SS4).
- FARMA làm forged records chiếm consensus; đó là attack vào detection với quyền ghi amplification. Pilot poison cố định sau insertion cần tách nó khỏi repeated attacker writes, và dùng cùng ngân sách khi so sánh. [§III-C](https://arxiv.org/html/2607.05029#S3.SS3).

### Gaps
- Chưa thấy security experiment tương ứng toàn bộ chuỗi: fixed poison → successful legitimate correction → unseen same-scope tasks → delayed recurrence. Kết luận “chưa thấy” cần ghép với researcher phụ trách MemSecBench/PipePoison/EvoMal.
- After the Fix nêu hypothesis applicability, chưa xây dựng/evaluate defense chống adaptive poisoning; đóng góp mới không thể chỉ chuyển tên correction scope vào security.

## Có prototype post-hoc repair và chuẩn quyết định nào?

### Takeaway
Post-hoc selective repair cũng không phải vùng trắng. Có prototype và benchmark liên quan; phải phân biệt demonstration, repository claim, và peer-reviewed bằng chứng.

### Cited Findings
- **Antivenom** repository mô tả root attribution bằng counterfactual ablation, lineage invalidation và giữ descendant còn clean support. README phần “learning claim” đánh giá held-out attack classes sau surgery để giảm trust theo delivery channel; đó không phải cùng poison fixed và later same-scope tasks. README tự nhận project cho MongoDB Persistent Context Sprint Hackathon 2026, không phải conference paper; live Atlas chưa được chạy ở mục “What's next”. [README/claims](https://github.com/AK20202007/Antivenom).
- **Utility Under Attack**, arXiv 21/08/2026: §§5–7 kiểm tra false facts, content screening và provenance ranking; không adaptive attacker. Raising trust weight loại cả benign untrusted evidence; proposed occupancy remedy chưa được xây dựng. Đây là evidence cho utility tradeoff, không corrective-memory solution. Chưa xác minh venue. [Protocol](https://arxiv.org/html/2608.21230#S5), [limitations](https://arxiv.org/html/2608.21230#S7), [metadata](https://arxiv.org/abs/2608.21230).

### Inferences
- **Verdict cập nhật theo nhánh song song:** không giữ framing hiện tại làm novelty. Researcher khác đã xác minh **Revoked but Still Authoritative**, Appendix C.1: root bị revoke và thay bằng record hợp lệ; khi journal writeback tạo descendant qua Graphiti, task liên quan ở hops 1–3 vẫn có conditional unsafe, dù root validity filter chặn được nhánh không writeback. Đây là overlap trực tiếp với tái phát sau repair. Ghi chú này dựa trên xác minh của researcher song song; xem notes nhánh đó để có bằng chứng đầy đủ. [Paper](https://arxiv.org/html/2609.08258).
- **Verdict rộng:** lesson, transfer failure, phá consensus và post-hoc repair đều có prior art. Novelty phải nằm ngoài việc chỉ phát hiện lại cơ chế hậu duệ còn hoạt động sau revoke; cần đối chiếu method/assumptions của Revoked but Still Authoritative trước khi đề xuất attack/defense.
- Trước method mới, pilot cần giữ cùng poison/budget, xác nhận correction thành công trên observed examples; chọn held-out targets có cùng rule áp dụng; kiểm tra clean baseline và oracle poison/descendant removal; tách retrieval thất bại khỏi correction tuân thủ thất bại. Đây là thiết kế đề xuất của researcher, không kết quả literature.

### Gaps
- Không kết luận Antivenom đã “solve” bằng số trên README khi chưa tái lập benchmark và xem source attribution giả định.
- Ngày tra cứu: **12/10/2026, Asia/Saigon**. Publication/arXiv dates phía trên không dùng crawl date. Chỉ A-MemGuard xác minh venue từ proceedings chính thức; các nguồn khác được đánh dấu đúng mức.
