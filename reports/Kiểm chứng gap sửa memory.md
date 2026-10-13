# Literature bác bỏ gap sửa memory

**Không nên giữ đề xuất hiện tại như một gap mới để hướng tới A/A*.** Hiện tượng poison sống qua correction, hậu duệ tiếp tục gây lỗi sau thu hồi nguồn, và kinh nghiệm đã sửa không chuyển tốt sang tác vụ liên quan đều có nghiên cứu trực tiếp. Đồng thời, đã có phương pháp recovery với kết quả thực nghiệm trong những giả định cụ thể. Điều này bác tính mới của cách đặt vấn đề rộng; nó không chứng minh toàn bộ bài toán đã được giải quyết. Câu trả lời trước đã đề xuất phạm vi quá rộng. Quyết định hợp lý là dừng đầu tư method theo framing đó, rồi chỉ tiếp tục nếu pilot xác định được cơ chế khác biệt mà các baseline recovery hiện có không xử lý.

## Bằng chứng trực tiếp làm mất tính mới

**Revoked but Still Authoritative là overlap quyết định.** Appendix C.1 xét policy đã bị thu hồi và có bản thay thế: quyết định sai được ghi thành journal, rồi agent mới dùng journal ở tác vụ liên quan, không reinjection. Khi chỉ lọc nguồn, natural Graphiti ingestion còn unsafe **7,5% / 7,8% / 12,8%** ở hops 1–3; direct insertion là **71,6% / 79,5% / 83,1%**. Đây là tỷ lệ **có điều kiện trên hop 0 đã unsafe**, không phải ASR toàn tập. Trong single-read, validity filter đạt **0/1.620 unsafe actions**. Vì vậy, thành công khi chặn nguồn không bảo đảm an toàn sau write-back. ([Appendix C.1](https://arxiv.org/html/2609.08258#A3.SS1))

**After the Fix đã đo transfer sau accepted repair.** §3 chuyển cùng nguồn trước/sau lần sửa được chấp nhận đầu tiên sang target khác cố định, với 3.300 runs trên ThinkingBox/APEX. Lợi ích khác nhau giữa hai benchmark; §8 đặt applicability conditions, held-out targets và successive repairs trong continuing memory thành câu hỏi tiếp theo. Paper không có poisoning hay attacker. Nó bác claim rộng “lần đầu phát hiện sửa đúng ở source không bảo đảm làm đúng ở target”, nhưng không tự giải bài toán an ninh. ([§3](https://arxiv.org/html/2609.34603#S3), [§8](https://arxiv.org/html/2609.34603#S8))

MEMFLOW cũng đã có one-shot write, persistence sau khi attacker dừng, và Resistance với textual correction trong khi malicious memory vẫn retrievable. Table 2 báo Resistance ASR 100% trên ba model và hai framework trong Online regime. Đây là correction không thắng ảnh hưởng đang tồn tại, **chưa chứng minh repair thành công rồi mới tái phát**. ([MEMFLOW](https://arxiv.org/html/2603.15125))

## Đã có giải pháp, nhưng mỗi giải pháp có phạm vi

Ma trận dưới đây phân biệt bằng chứng đã có với điều chưa được chứng minh. Một tên method hoặc số ASR tốt không đủ để gọi là giải pháp phổ quát; ngược lại, thiếu một protocol cụ thể trong paper cũng không chứng minh novelty.

| Công trình | Đã nghiên cứu hoặc xử lý | Giới hạn liên quan |
|---|---|---|
| [Revoked but Still Authoritative](https://arxiv.org/html/2609.08258#A3.SS1) | Thu hồi nguồn; write-back giữ hậu duệ gây lỗi trên tác vụ liên quan. | Source filter; chưa comprehensive semantic repair. |
| [After the Fix](https://arxiv.org/html/2609.34603#S3) | Transfer của kinh nghiệm trước/sau accepted repair. | Lỗi tự nhiên, không adaptive poisoning. |
| [MEMFLOW](https://arxiv.org/html/2603.15125) | Persistence và chống textual correction. | Poison còn retrievable; không certified repair checkpoint. |
| [A-MemGuard](https://arxiv.org/html/2510.02373#S4.SS2) | Negative lessons; truy hồi theo query và action để tránh lỗi tương tự. | Chưa xác minh correction/held-out split ở proceedings cuối. |
| [MemAudit](https://arxiv.org/html/2605.23723) | Counterfactual attribution và consistency graph để chọn/remove memory gây lỗi. | Batch memory cố định; phụ thuộc failure signal, replay/logs. |
| [Selective Revocation and Replay](https://github.com/aliuyar1234/selective-revocation-replay) | Thu hồi descendants, replay dirty writes, fallback rollback; giữ retention. | Root gần oracle, logs đầy đủ; hidden state ngoài scope. |
| [MemSecBench](https://arxiv.org/html/2607.27080) | Đánh giá malicious semantics bị neutralize và benign semantics còn trong backend. | Execute/Forget độc lập từ cùng snapshot; chưa Forget → future Execute. |
| [EvoMal](https://arxiv.org/html/2608.25776) | Quarantine/signatures ngăn propagation khi attacker đã rời đi. | Admission do curator kiểm soát; hạn chế self-evolution. |

A-MemGuard đã được xác nhận ở **ICML 2026** qua proceedings chính thức. Appendix G nhắm chuyển lesson giữa câu hỏi có thao tác tương tự. Vì thế “defense thêm corrective memory để khái quát” riêng lẻ đã trùng hướng method. FARMA đã đánh giá vượt A-MemGuard bằng amplification entries do attacker ghi; đó không phải defender tự tạo descendants sau attacker exit. ([Proceedings](https://proceedings.mlr.press/v306/wei26ac.html), [A-MemGuard Appendix G](https://arxiv.org/html/2510.02373#A7), [FARMA](https://arxiv.org/html/2607.05029#S3))

MemAudit có recovery theo hành vi: RAP post-repair ASR=0 trên ba backbone trong main setting, nhưng QA còn residual 10% trên DeepSeek. Không nên gộp hai setting hoặc suy ra bảo vệ mọi memory evolution. ([MemAudit](https://arxiv.org/html/2605.23723))

SRR có matrix deterministic **8 chains × 2 architectures × 2 attack variants**. Live Qwen2.5:14b gồm hai explicit retrieval chains, mỗi chain ba repeats: **sáu no-recovery runs có ASR=1; sáu selective-replay runs có ASR=0**, correctness/retention=1. Đây là bằng chứng live hẹp, không sáu dataset hay bảo đảm chung. Thư mục `usenix_security26` không chứng minh acceptance. ([Artifact](https://github.com/aliuyar1234/selective-revocation-replay), [paper source](https://github.com/aliuyar1234/selective-revocation-replay/blob/main/paper/usenix_security26/main.tex))

## Phòng ngừa, correction và recovery không đồng nghĩa

Một defense có thể chặn quyền của memory trước hành động mà không sửa nội dung sai. TMA-NM bảo vệ consequential action dưới origin labeling và trusted monitor; paper không bảo đảm loại mọi answer bias. Đây là prevention theo authority assumptions, không chứng minh semantic correction được học đúng. ([TMA-NM](https://arxiv.org/html/2606.24322#S9))

Tương tự, thêm một câu correction không tương đương thu hồi nguồn và các biến đổi phát sinh từ nguồn. Với proposal hiện tại, cần tách bốn khả năng: correction không được truy hồi; correction có trong context nhưng không được áp dụng; hậu duệ còn hoạt động; correction vốn không áp dụng cho target. Một ASR sau sửa khác 0 không phân biệt được chúng. Thành công trên observed examples cũng chưa xác nhận trạng thái memory đã được repair toàn diện.

**Phần chưa được các nguồn đọc chứng minh đầy đủ** là một phương pháp bảo đảm an toàn hậu repair trước adaptive attacker, trên target mới cùng phạm vi correction, trong evolving memory, đồng thời giữ utility. Nhưng đây là tập yêu cầu rất rộng. Ghép các yêu cầu chưa đồng thời xuất hiện trong một paper không tự tạo đóng góp. Một paper mới phải chỉ ra assumption cụ thể bị phá và cơ chế có thể kiểm chứng; đổi vector backend, trigger, objective hoặc tăng số model chưa đủ.

## Chỉ tiếp tục khi pilot phân biệt được cơ chế

Không nên train generator theo mục tiêu “ASR sau correction vẫn cao” ngay. Trước hết, cố định một lần can thiệp và khóa quyền attacker; xác nhận correction hợp lệ, có cùng quy tắc áp dụng cho target; đo observed và held-out tasks cùng với stream benign. Sau đó đặt cạnh textual correction, validity filter, A-MemGuard, MemAudit và SRR trong scope triển khai được. Cần oracle removal của seed và descendants, cùng can thiệp đưa correction trực tiếp vào context để phân biệt retrieval với hành vi sử dụng.

**Chỉ go nếu còn hiệu ứng nhân quả khác biệt sau những đối chứng này**, và chỉ ra được baseline thất bại vì assumption nào. Stop hoặc thu hẹp nếu kết quả chỉ tái hiện stale retrieval, descendants ngoài revocation closure, hay correction ngoài scope. Protocol ấy là tiêu chí chọn đề tài, chưa là ý tưởng method mới hoặc gap đã xác nhận. Không nên dùng “chưa thấy ai đánh giá đúng tổ hợp này” làm lập luận novelty chính cho A/A*.

Audit chốt **12/10/2026, Asia/Saigon**, dựa trên ba ghi chú đã đọc; không tái lập code hoặc mở thêm vòng nghiên cứu. Firecrawl bị chặn trong quá trình tra cứu; các xác minh dùng primary pages/artifacts đọc được. PDF proceedings cuối của A-MemGuard chưa inspect được, nên nhận định protocol dựa trên arXiv đã đọc. Ngoài A-MemGuard, các nguồn chính trên chưa được xác minh venue; repository claims không được coi là peer-reviewed results. Không có cơ sở kết luận universal solve, cũng không có cơ sở xác nhận một gap thay thế mới.

**Verdict:** bỏ claim rộng hiện tại. Nếu tiếp tục, đầu tư một pilot có causal controls và recovery baselines trước; chỉ xây attack–defense sau khi tìm được failure mechanism khác prior art. Đây là điều kiện cần cho định vị A/A*, không phải bảo đảm acceptance.
