# Kiểm chứng gap: sửa chữa, thu hồi và tái phát poisoning trong memory agent

## 1. Hiện tượng tái phát sau sửa chữa/thu hồi và selective recovery đã có chưa?

### Takeaway

Có prior art trực tiếp cho cả hiện tượng và giải pháp trong phạm vi giới hạn. Không nên định vị “xóa seed vẫn còn descendants”, “memory sai đã được thu hồi vẫn ảnh hưởng tác vụ sau”, hoặc “repair có giữ utility” như phát hiện mới. Chưa có bằng chứng các giải pháp dưới đây giải quyết phổ quát semantic correction transfer và adaptive attack trên mọi backend.

### Cited Findings

- **Revoked but Still Authoritative** — arXiv v1, 08/09/2026; metadata không ghi venue. §3–5 thử năm hệ memory, chín scenario và chín model. Retrieval filter đạt 0/1.620 unsafe actions trong phép đo single-read. **Appendix C.1** trực tiếp kiểm tra write-back: revoked policy có replacement, quyết định sai ở hop 0 được ghi thành journal hiện hành; agent mới làm tác vụ liên quan ở hops 1–3. Filter chỉ loại record gốc, bỏ sót journal. Conditional theo poisoned hop 0, natural Graphiti ingestion cho unsafe rates **7,5/7,8/12,8%** sau filter; direct insertion **71,6/79,5/83,1%**. Hai chế độ không được gộp. Đây là descendants còn sống sau thu hồi, không phải original record đã hard-delete tự xuất hiện lại. — [Full text, Appendix C.1](https://arxiv.org/html/2609.08258#A3.SS1); [metadata](https://arxiv.org/abs/2609.08258).

- **Selective Revocation and Replay (SRR), Ali Uyar, 2026** — official repository có paper source và DOI tự trích dẫn 10.5281/zenodo.18967061, ghi “Independent research”. Thư mục `usenix_security26` không chứng minh acceptance. Cơ chế: revoke descendants của suspicious root, replay dirty writer events khi sound, fallback rollback. Artifact deterministic: 8 chains × 2 architectures × 2 attacks, five methods + one ablation. Bốn session: S1 seeding, S2 benign accumulation, recovery, S3 targeted follow-up, S4 retention. Live Qwen2.5:14b: hai explicit retrieval chains, mỗi chain ba repeats; sáu no-recovery ASR=1, sáu selective-replay ASR=0, correctness/retention=1. **Limitations:** root effectively oracle; complete object logs; hidden model state/chat buffers ngoài scope; summaries fallback rollback; stealth weak; không có external side effects. Chưa thấy held-out correction-scope split hoặc adaptive post-repair optimization. — [README](https://github.com/aliuyar1234/selective-revocation-replay); [paper source: Evaluation, Results, Limitations](https://github.com/aliuyar1234/selective-revocation-replay/blob/main/paper/usenix_security26/main.tex).

- **MemAudit** — arXiv v1 22/05/2026, metadata không ghi venue. §4–6 dùng counterfactual causal attribution + consistency graph để rank/remove memories, không cần oracle poison labels. Audit chạy batch trên memory cố định rồi remove. MINJA QA/RAP: RAP post-repair ASR=0 trên ba backbones ở main setting; QA có residual 10% trên DeepSeek. §8 nêu phụ thuộc failure signal/replay/logs; dense mutually reinforcing poison làm repair yếu. Tác giả chưa đánh giá adaptive adversaries rộng và heterogeneous memory environments. Là behavioral recovery thực nghiệm, vượt xa chỉ kiểm tra row deletion; chưa là chứng minh recovery bền trong memory evolution. — [Full text §§4–8](https://arxiv.org/html/2605.23723); [metadata](https://arxiv.org/abs/2605.23723).

- **MemSecBench** — arXiv:2607.27080v1, 29/07/2026; 310 cases, 24 configurations. Methodology cho Execute và Forget chạy **độc lập từ cùng post-Write snapshot**. Appendix E.4 chấm malicious semantics bị remove/neutralize và mọi benign semantics còn trong backend. Chính paper nói phép chấm không establish irrecoverability; benign preservation không guarantee later retrieval/task use; inaccessible caches/latent state ngoài audit. Vì vậy benchmark này chưa chấm một chuỗi **Forget → held-out Execute → memory evolution**. — [Methodology và Appendix E.4](https://arxiv.org/html/2607.27080).

### Inferences

- Claim broad “repair thất bại vì descendants” bị overlap rất mạnh bởi Revoked but Still Authoritative; “selective descendants revocation + replay giữ utility” đã là core contribution SRR.
- MemAudit là baseline recovery cần thử nếu nghiên cứu adaptive post-repair attack. Chỉ so benign append hoặc original seed deletion sẽ thiếu baseline trực tiếp.
- Chuỗi đánh giá semantic correction theo phạm vi áp dụng, held-out tasks, evolving memory cần so với prior art khác do nhóm nghiên cứu correction transfer phụ trách; chưa thể suy từ thiếu trong bốn nguồn này rằng gap mới.

### Gaps

- Chưa kiểm chứng venue chính thức cho các preprints trên; không nên tự gọi A/A* publication.
- SRR DOI resolver không tải được; paper source/official README đã đọc. Không có kiểm chứng độc lập hay reproduction local trong audit này.

## 2. Có software/benchmark nào đã xử lý hoặc đo re-emergence và causal recovery?

### Takeaway

Có cả SDK recovery và benchmark re-emergence công khai. Chất lượng bằng chứng khác với paper có thí nghiệm rộng, nhưng vẫn là prior art phải đối chiếu.

### Cited Findings

- **MnemoIR / MIRAGE Bench**, official GitHub: given compromised memory ID/source, provenance graph + blast-radius analysis, quarantine/invalidate/supersede/recompute/rollback/replay/compensate/escalate, counterfactual replay và recovery report. README báo một poisoning scenario end-to-end, Groq demo, mem0 adapter, 41 offline tests; **1/5 scenario families implemented, còn lại labelled fixtures**. Causal attribution có distractor retrieved nhưng không gây decision. Không có bằng chứng conference paper hoặc large multi-model held-out evaluation trên trang đọc. “Proof” ở đây là machine-verifiable report cho tracked state/replayed decision, không guarantee mọi future behavior. — [official project README, MIRAGE Bench, Status](https://github.com/DheerajPranav/mnemo-ir).

- **PurgeBench**, official README: RC = HarmonicMean(ER, UR) × RR; ER original triggers + paraphrases không fire, UR benign memory/performance, RR không re-emerge qua summarization/compaction/backflow. README đồng thời ghi “concept/protocol v0.1; reference implementation forthcoming” nhưng có v0 real-memorywire results: 30 poisons, five classes, 20 benign checks, seven procedures. Vì status mâu thuẫn mức hoàn thiện, không gọi benchmark mature. Các file RESULTS/concept-note/code tree không tải được qua web trong lần kiểm chứng này, nên chưa xác minh RR thực sự chạy LLM consolidation hay scorer proxy, chưa xác minh train/test split/adaptive budget. — [official README](https://github.com/mthamil107/purgebench).

- **memorywire**, arXiv:2606.01138v4, 12/08/2026. §4.6 recovery dùng provenance forget, heuristic quarantine trusted directive và expire. §5.4 self-built PurgeBench trên sqlite-vec: provenance purge RC=0,64; entangled trusted memories là hard case, automatic eradication xung đột utility. §5.5 nói synthetic corpora, authors-as-evaluators, microbench và limited scale. Không có adaptive attack tối ưu qua repair được mô tả trong mục recovery. §8 nêu span-level revocation là future direction. — [full text §§4.6, 5.4–5.5, 8](https://arxiv.org/html/2606.01138).

- **Provenance loss in agent memory**, independent GitHub research artifact: reimplement MemLineage và dùng published SRR runtime; tác giả thừa nhận no MemLineage author code và small N. Consolidation session tạo lại malicious rule không tool calls trong 11/16 histories; writer thiếu logged retrieval inputs khiến rule nằm ngoài revocation closure. Add parents giảm residual ASR 43,8%→18,8%, N=16, p=0,125. Đây là preliminary source về incomplete lineage; không gán kết quả reimplementation thành flaw đã xác nhận của original authors. — [official artifact README, Main findings](https://github.com/dan0nchik/memlineage-provenance-loss).

### Inferences

- Một paper mới chỉ thêm recovery-completeness metric hoặc counterfactual replay sẽ không đủ novelty, vì PurgeBench và MnemoIR đã mô tả trực tiếp.
- Nếu dùng incomplete provenance như cơ chế mới phải đối chiếu reimplementation artifact và formal authority work; việc “ghi mọi input thành parent” dễ overtaint clean utility.
- Bằng chứng software không universal solve; vẫn phải cite nếu ý tưởng tương tự. Chưa reproduce nên giữ các numbers là author-reported.

### Gaps

- Chưa đọc được PurgeBench actual scorer; RR implementation, query novelty, adaptive budget vẫn chưa xác minh.
- Không tìm thấy peer-reviewed venue từ primary pages đã đọc cho MnemoIR, PurgeBench hoặc independent provenance-loss artifact.

## 3. Các defense có guarantee đã đóng gap đến đâu, và verdict novelty?

### Takeaway

Guarantee có tồn tại với origin/authority hoặc quarantine assumptions mạnh. Chúng ngăn ảnh hưởng hoặc propagation chứ không chứng minh agent đã khái quát đúng một correction semantic. **Verdict cho hướng ban đầu: reject positioning như fresh A/A* gap; có thể còn một câu hỏi hẹp hơn, nhưng phải đối chiếu thêm correction-transfer work.**

### Cited Findings

- **TMA-NM**, arXiv:2606.24322, §§II–IV: origin bound do trusted monitor; origin labeling oracle A1, authenticated channels, authority không nâng qua summarization/trusted echo/fake corroboration. Consequential action cần independent trusted corroboration hoặc fresh action-bound user authorization. Paper thừa nhận non-consequential answer bias mitigated, không eliminated. Đây là information-flow enforcement, không recovery từ nguồn đã trusted nhưng về sau corrected. — [Threat Model và Non-Malleable Construction](https://arxiv.org/html/2606.24322).

- **SMSR**, arXiv:2606.12703, §§III–VII: signed provenance + smoothed retrieval; bound theo số poisoned signed memories và random sampling assumptions. Enterprise 15 scenarios; end-to-end query-only write test giảm ASR 65,3%→5,3%. Không thấy repair/rollback stage trong full text; đây là robustness lúc retrieval, không semantic correction learning. — [full text](https://arxiv.org/html/2606.12703).

- **EvoMal**, arXiv:2608.25776, §9.3: curator signs retrievable entries, agent-authored entries ở unretrievable quarantine. Unforgeable signatures + curator-controlled admission log cho propagation extinction trong attacker-removed condition; persistent attacker vẫn có initial seed rate. §9.2 counterprompt giảm create-path copying với utility đo completion; §10 nói structural quarantine hạn chế self-evolution. Không phải phương pháp cho agent tự học correction trên unrestricted writable memory. — [§§9.2–9.3 và Conclusion](https://arxiv.org/html/2608.25776).

- **MemEvoBench**, arXiv:2604.15774v2, 21/05/2026; §§3.3/4.3: ba related non-identical rounds, agent responses append; biased feedback reinforces risky behavior. +ModTool có web verification/correct_memory thay misleading memory trước final response. Thí nghiệm đã kết hợp correction tools và memory evolution; không nên claim “lần đầu correction trong evolving memory”. Đó không phải fixed verified correction rồi evaluate held-out scope dưới benign-only stream: mitigation online, biased feedback và chỉ ba rounds. — [§§3.3, 4.3](https://arxiv.org/html/2604.15774).

### Inferences

- Không claim “chưa ai solve”; đúng hơn: các **subproblems đã có solutions được đánh giá**, nhưng không thấy universal solution. Novelty không xuất phát từ đòi guarantee rộng hơn một cách chung chung.
- Cần phân biệt (a) same stale root retrieval, (b) contaminated descendants sống sau revoke, (c) semantic correction không transfer đến task cùng scope, (d) adaptive attack khiến một defense đã repair thành công tái phát. Nguồn mạnh trực tiếp đã bao phủ (a), (b), phần recovery; nhóm khác đang kiểm chứng (c), (d).
- Nếu tiếp tục, threat model phải chốt attacker một lần và quyền cố định; correction từ authority hợp lệ; loại original + tracked descendants oracle làm causal baseline; so SRR/MemAudit/validity guard; held-out scope và long-run benign stream. Đây là yêu cầu để phân biệt đóng góp, chưa là method đề xuất hoặc gap xác nhận.

### Gaps

- Full texts MEMFLOW/MemLeak đã mở nhưng chưa phân tích sâu vì các nguồn trực tiếp hơn đủ bác positioning rộng. Không dùng chúng làm bằng chứng “không có”.
- Literature audit chốt ngày 12/10/2026 theo client context; việc không thấy một setting trong sources đã đọc không chứng minh không tồn tại công trình nào khác.
