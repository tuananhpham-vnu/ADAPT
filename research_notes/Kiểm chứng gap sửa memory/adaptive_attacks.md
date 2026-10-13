# Attack sống sót sau correction hoặc repair: kiểm chứng novelty đến 12/10/2026

## Gap rộng «memory poisoning tồn tại/tái phát sau correction» đã được nghiên cứu chưa?

### Takeaway
Đã được nghiên cứu trực tiếp. Không nên claim «phát hiện correction không chữa được memory poisoning» là gap mới. Việc đã nghiên cứu hiện tượng không có nghĩa tồn tại defense hoàn chỉnh.

### Cited Findings
- **MEMFLOW — From Storage to Steering** (2603.15125v3, 05/06/2026), §3.2, §4.2 RQ4, Appendix B.5–B.6: có one-shot memory write; Persistence cho attacker dừng rồi thêm benign turns; Resistance đưa textual correction sau deviation nhưng giữ malicious memory retrievable. Table 2 báo Resistance ASR 100% trên ba model, hai framework. Resistance dùng Online regime; chưa chứng minh trạng thái đã sửa thành công rồi hỏng lại. RBMS defense ở §4.4 chỉ đánh giá Override, không phải comprehensive recovery; ASR còn khác 0. — [Full body](https://arxiv.org/html/2603.15125)
- **Zombie Agents** (2602.15654): nghiên cứu infection qua poisoned web content, normal memory update, persistence across sessions; instruction defenses vẫn còn ASR >60%. Body không trình bày accepted authoritative correction rồi kiểm chứng generalization của correction. Đây là persistence, không phải cùng protocol repair. — [Full body](https://arxiv.org/html/2602.15654)
- **PipePoison** (2609.00523), §RQ4: optimize trên undefended shadow systems, giữ nguyên payload khi đánh giá defended victims, không feedback từ defense. Có conflict resolution và 100 subsequent sessions với related/unrelated new memories. Related new memories không được mô tả là authoritative corrections. Đây không phải adaptive repair attack, nhưng đã bao phủ benign competition và post-write memory management. — [Full body](https://arxiv.org/html/2609.00523)

### Inferences
- Chỉ thêm nhiều benign, đo ASR theo thời gian hoặc cho user nói correction rồi quan sát relapse đều không đủ novelty: những nhánh tương ứng đã có trong MEMFLOW/PipePoison.
- Không nên gọi «ASR sau sửa khác 0» là recurrence sau successful repair nếu chưa xác nhận repair đã vô hiệu hóa nguồn sai và khôi phục hành vi trước rollout mới.

### Gaps
- MEMFLOW mô tả correction bằng text và giữ poison trong store; chưa thấy phép đo repair checkpoint xác nhận semantic removal, preservation of benign memory, rồi post-checkpoint held-out evaluation.
- Chưa có bằng chứng từ các paper trên rằng adaptive optimization nhằm vào một correction mechanism cụ thể sẽ thắng trong cùng initial attacker budget.

## Gap hẹp «nguồn đã bị bác bỏ nhưng bản ghi hậu duệ gây tái phát trên tác vụ liên quan» còn mới không?

### Takeaway
Không mới ở mức hiện tượng; **Revoked but Still Authoritative** là overlap quyết định.

### Cited Findings
- **Revoked but Still Authoritative** (2609.08258v1), §3–4: seed policy của defender đã revoked + replacement. Verified mark ở Graphiti/mem0-exp; validity filter đạt 0/1,620 unsafe khi single-read. mem0-exp **tắt** default expiry filter. — [Full body](https://arxiv.org/html/2609.08258)
- Appendix C.1: unsafe hop 0 → defender journal qua **natural ingestion** hoặc direct write → new agent/related task ở hops 1–3; không reinjection. Root filter bỏ sót active journal. Natural Graphiti: 43/358 journals stored, 22 active; unsafe 7.5%, 7.8%, 12.8%; direct-write 71.6%, 79.5%, 83.1%, đều conditional hop-0 unsafe. Đây là source filtering, chưa comprehensive semantic repair. — [Appendix C.1](https://arxiv.org/html/2609.08258)
- Author artifact **Selective Revocation and Replay** nghiên cứu recovery sau persistent indirect injection: revoke persisted descendants, replay dirty writes hoặc coarse rollback. Frozen deterministic matrix gồm 8 chains × 2 architectures × 2 attack variants; focused live Qwen2.5:14b có sáu selective-replay reruns hồi phục. Đã có phương pháp chạm trực tiếp descendant recovery, nhưng bằng chứng live hẹp và phần lớn matrix deterministic. — [Author repository](https://github.com/aliuyar1234/selective-revocation-replay)

### Inferences
- Thay trusted obsolete policy bằng trigger poisoning, thay Graphiti bằng episodic vector memory, hoặc mở rộng số benchmark **riêng lẻ** chưa tạo gap đủ mạnh. Cần một failure mechanism mới hoặc security guarantee mà prior methods không đáp ứng, không chỉ đổi surface.
- «Đã giải hoàn toàn» cũng là kết luận quá mạnh: source filter bị write-back phá; selective replay yêu cầu lineage/replay soundness và tác giả giới hạn explicit persisted state.

### Gaps
- Chưa thấy ở paper revocation phép kiểm chứng correction hợp lệ trên source, chứng minh state repair toàn diện, rồi adversarial residual effect trên held-out tasks thuộc **cùng quy tắc correction**; họ kiểm tra stale validity và journal lineage.
- Tuy nhiên đây chỉ là phần chưa thấy trong phạm vi đọc; không được nâng thành claim «first» mà bỏ qua recovery/provenance literature do các researcher khác đang kiểm chứng.

## Correction generalization và defense thích nghi có prior nào làm yếu ý tưởng đề xuất?

### Takeaway
Có prior rất gần cả hai phía. Generic correction-transfer gap đã được đánh giá; beating A-MemGuard cũng đã được đánh giá. Security-specific adaptive post-repair causation có thể còn một phiên bản nghiên cứu được, nhưng hiện chưa đủ cơ sở chốt novelty A/A*.

### Cited Findings
- **After the Fix: Transfer of Corrected Agent Experience** (2609.34603v2, 29/09/2026), §3: cùng failed source trước/sau **first accepted repair**, chuyển sang fixed distinct target; 3,300 runs trên ThinkingBox/APEX. §5.4 cho ví dụ accepted convention áp sai yêu cầu mới. §8 đề xuất explicit source-derived applicability conditions, target-blind constructors, held-out targets và utility controls. Body không có poisoning/adversarial setup. — [Full body](https://arxiv.org/html/2609.34603)
- **FARMA** (2607.05029v1), §III-C và §V: attacker **tự append** amplification entries; có thể làm offline trước victim query. Evaluation amplification cycle=10, 50 trials; vượt heuristic/LLM A-MemGuard. Không phải defender tự sản sinh hậu duệ sau attacker exit, cũng chưa phải post-authoritative-correction evaluation. SENTINEL đạt 0% EHR, nhưng ReAct-QA còn 6%; không được claim mọi domain đều 0%. — [Full body](https://arxiv.org/html/2607.05029v1)
- **MemPoison** (2605.29960), §3/5.4: selective extraction bypass qua dialogue + trigger optimization; defenses là PPL và paraphrasing. Không thấy correction/repair protocol trong body. — [Full body](https://arxiv.org/html/2605.29960)
- **MemPoison benchmark** (2607.14651), §5.4/B.3: L1 single-record, L2 composition, L3 dormant activation; stage-restricted defenses và residual failures. Đây là paper khác cùng tên ngắn, không nên gộp với 2605.29960; không thấy accepted-correction protocol. — [Full body](https://arxiv.org/html/2607.14651)

### Inferences
- Reject gap rộng được đề xuất trước: «correction có ích trên seen nhưng không đảm bảo future tasks», «persistence despite benign additions», «descendant recurrence after source exclusion», «vượt A-MemGuard» đều có overlap.
- Chưa có basis để nói toàn bộ bài toán đã solve; bài nghiên cứu tốt phải phân biệt một guarantee thiếu cụ thể với «đã có attack/defense mang tên tương tự».
- Pilot chỉ đáng đầu tư khi nó phân biệt được correction applicability error, stale-root retrieval, descendant write-back và lỗi nền; sau đó đánh giá recovery baselines thực, attacker budget cố định, checkpoint repair được kiểm chứng và clean utility. Đổi trigger/objective thôi chưa phải đóng góp A/A*.

### Gaps
- Không paper được đọc nào tự nó chứng minh một defense bảo đảm post-repair safety dưới mọi adaptive attack và vẫn học mới tốt.
- Chưa xác nhận novelty cho một cơ chế attack/defense cụ thể vì chưa có proposed mechanism ngoài «tối ưu ASR sau correction». Cần kiểm chứng remaining gap sau đối chiếu toàn bộ recovery literature.
