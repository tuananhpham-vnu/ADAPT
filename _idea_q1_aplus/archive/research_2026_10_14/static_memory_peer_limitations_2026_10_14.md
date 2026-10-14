# Static-memory attacks: từng peer đã xử lý đến đâu và còn thiếu gì?

Mốc rà soát: **14/10/2026**. Đây là tổng hợp nguồn sơ cấp, chưa tái lập các kết quả được trích. Mục tiêu hiện tại là xác định vấn đề còn mở; **chưa chọn method, chưa chọn AgentPoison làm gốc, chưa xếp hạng ý tưởng**. Các pilot trong [review trước](static_trigger_peers_review_2026_10_14.md) là đề xuất chưa được chọn.

**Bổ sung sau vòng đọc sâu:** [phân tích cơ chế và gap](static_memory_mechanism_gap_analysis_2026_10_14.md) đối chiếu thêm WARP, KidnapRAG, FORGE, RAG-Narok, CompressionAttack, Selection Integrity và context-selection diagnostics. Bảng 23 công trình bên dưới không phải danh mục đầy đủ; các claim về trajectory, context-aware poisoning, compression và selection cần đọc cùng phần bổ sung này. Đã đề xuất thứ tự kiểm chứng, chưa chọn hoặc chạy method.

## 1. Cách đọc và phạm vi

Memory tĩnh ở đây: attacker chuẩn bị poison, sau đó khóa nội dung kho trong giai đoạn đánh giá. Agent vẫn có thể truy hồi nhiều lần. Corpus QA, kho demonstrations và memory graph khác nhau; kết quả trên một loại không tự chuyển sang loại khác.

Phân biệt bốn loại nhận định:

- **[TG]**: giới hạn hoặc điều kiện tác giả nêu rõ.
- **[ĐO]**: điểm yếu thể hiện trong thí nghiệm của bài, chỉ có hiệu lực trong thiết lập đó.
- **[PV]**: giới hạn phạm vi mà review xác định từ threat model/thí nghiệm; chưa có bằng chứng method thất bại ngoài phạm vi.
- **[BC]**: thiếu bằng chứng/đối chứng hoặc khó tái lập; không tự động tạo ra gap thuật toán.

“Đã xử lý” nghĩa là có phương pháp và thực nghiệm cho một thiết lập, không có nghĩa giải quyết mọi trường hợp. “Chưa xác lập” nghĩa là các nguồn rà soát chưa đủ bằng chứng, không phải khẳng định chưa ai làm.

Không so ASR trực tiếp giữa các paper khi khác quyền attacker, target, corpus, ngân sách hoặc cách chấm. Đặc biệt:

| Cụm từ | Phải hỏi thêm |
|---|---|
| Một document | Một document mỗi query, mỗi trigger hay cả tập query? |
| Black-box | Thấy rank, score, embedding, retrieved context, hay chỉ final answer? |
| Trigger | Chuỗi attacker đưa vào query, từ khóa tự nhiên, hay suffix trong document? |
| Transfer | Đổi retriever, generator, domain, query hay toàn bộ pipeline? |
| Hiệu quả cuối pipeline | Truy hồi đúng poison, trả lời chứa target, hay thực hiện đúng hành động? |
| Ít chi phí | Đã tính khởi tạo, surrogate, lần tìm thất bại và chi phí dùng lại chưa? |

## 2. Nhóm backdoor và trigger — sát hướng nghiên cứu nhất

### AgentPoison — 2024

**Đã làm:** tối ưu trigger để query có trigger tập trung và tách khỏi query sạch trong embedding space; poison memory/knowledge để dẫn tới hành vi mục tiêu. Có đánh giá agent và transfer, không chỉ retrieval.

**Còn giới hạn:** [TG] cần white-box embedder khi tối ưu. [PV] cần đường đưa trigger vào đầu vào; không đồng nhất với chỉ chèn corpus và chờ query tự nhiên. [TG] cấu hình tối ưu tới 1.000 iterations; số poison/độ dài trigger khác nhau theo task.

**Đối chiếu:** Phantom/BadRAG đã giảm yêu cầu kiểm soát query; không nên lấy “natural trigger” làm novelty mới từ AgentPoison.

Nguồn: [paper, Limitations và Appendix A.1](https://arxiv.org/html/2407.12784).

### Phantom — 2024; bản mở rộng 2025

**Đã làm:** backdoor bằng một document với trigger tự nhiên; tối ưu retrieval và generation. Đã nghiên cứu trigger viability, không chỉ chọn trigger tùy ý.

**Còn giới hạn:** [ĐO] transfer poison từ Contriever sang DPR/ANCE đạt 0% top-5 trong thí nghiệm §6.2.5. [ĐO] §6.2.2: tăng phần retrieval trong document từ 32 lên 128 tokens giảm failure 82,3% xuống 4%; đây không phải độ dài query trigger. [ĐO] generator black-box mạnh hơn vẫn làm hiệu quả giảm.

**Đối chiếu:** đã xử lý query tự nhiên và khả năng dùng chung; còn câu hỏi về độ bền khi đổi hệ thống và ngân sách.

Nguồn: [paper, §6.1.10, §6.2–6.3](https://arxiv.org/html/2405.20485).

### BadRAG — 2024; bản cập nhật 2026

**Đã làm:** poison theo từ khóa tự nhiên mà không sửa query; contrastive optimization phân biệt query mục tiêu/sạch, có gom nhiều trigger và transfer.

**Còn giới hạn:** [TG] dùng white-box shadow retriever; phần thí nghiệm chính tập trung shadow/victim cùng retriever. [PV] bằng chứng cho từ khóa/nhóm từ khóa không xác lập chọn lọc cho mọi điều kiện ngữ nghĩa. Appendix K có mở rộng GCG cho generator khó hơn, nhưng thêm quyền white-box generator.

**Đối chiếu:** positive/negative contrastive loss và multi-trigger đã là prior art; không thể đề xuất lại như đóng góp chính.

Nguồn: [paper, §3 và Appendix K](https://arxiv.org/html/2406.00083).

### PR-Attack — SIGIR 2025

**Đã làm:** phối hợp poison và prompt trigger bằng bilevel optimization, thay phiên tối ưu hai thành phần.

**Còn giới hạn:** [TG] attacker cung cấp prompt cho người dùng, đồng thời chèn poison cho target question. [PV] quyền prompt-provider mạnh hơn corpus-only; kết quả không chứng minh backdoor chờ query nguyên bản của người dùng.

**Đối chiếu:** “joint prompt–poison” hoặc “bilevel attack” đã có; giảm quyền truy cập cần so với cả Phantom/BadRAG, không riêng PR-Attack.

Nguồn: [paper, §3.1–3.3](https://arxiv.org/html/2504.07717).

## 3. Nhóm tối ưu poison — thuật toán, fluency, chi phí

| Peer | Đã xử lý | Giới hạn có căn cứ |
|---|---|---|
| [Joint-GCG — AAAI 2026](https://arxiv.org/html/2506.06151) | Phối hợp gradient retrieval–generation; xử lý lệch vocabulary/tokenizer, điều chỉnh trọng số; có transfer. | [TG] thiết lập chính white-box cả hai model. Chuẩn bị vocabulary projection khoảng 2 giờ/A6000 cho mỗi cặp, có thể dùng lại. Batch-query chỉ chạy một round vì compute; mở rộng domain/pipeline còn ngoài phạm vi (§2, Appendix F). |
| [GASLITE — CCS 2025](https://arxiv.org/html/2412.20953) | Tối ưu suffix document mạnh; gradient averaging, nhiều tọa độ, retokenization; query sets, clustering/set cover và so sánh giới hạn thời gian. | [TG] §10 thừa nhận chi phí tối ưu lớn. [PV] trọng tâm là khả năng xuất hiện trong dense retrieval; có RAG case study nhưng chưa xác lập action control phổ quát. Không nên gọi “amortize nhiều query” hoặc “equal-time comparison” là mới. |
| [AdvDec — Findings EACL 2026](https://aclanthology.org/2026.findings-eacl.108.pdf) | Sinh adversarial text dễ đọc qua decoding có ràng buộc; có đánh giá downstream RAG. | [TG] Limitations: cần logits của scoring LM hoặc quyền truy cập embedding tùy objective. Scoring LM có thể là model cục bộ, không được diễn giải thành bắt buộc logits victim. Hệ hoàn toàn đóng có thể không cung cấp tín hiệu cần thiết. |
| [CPA-RAG — 2025](https://arxiv.org/html/2505.19864) | Poison dễ đọc, refine bằng nhiều LLM/surrogate và evaluator; có nghiên cứu transfer. | [TG/ĐO] Appendix E: clean evidence ở k lớn có thể lấn át poison; model giữ đáp án thật; generator tạo poison đôi khi vô tình giữ thông tin đúng. Giải thích do pretraining là giả thuyết tác giả, không phải quan hệ nhân quả đã xác lập. |
| [CtrlRAG — 2025/2026](https://arxiv.org/html/2503.06950) | Tối ưu document bằng feedback black-box; cố giữ mục tiêu độc hại khi sửa để tăng retrieval. | [TG] §3.2 cần response và retrieved context; thiết lập chính còn có rank. Biến thể extreme vẫn thấy context. §4.2 cần poison được truy hồi để có feedback. [PV] không chứng minh optimizer hoạt động với final-answer-only từ cold start. |
| [SilentRetrieval — preprint 2026](https://arxiv.org/html/2605.28074) | Phối hợp beam search và trigger phù hợp context để cân bằng fluency, retrieval, generation; thử adaptive reranking. | [TG] §8: gradient retriever, seed đúng chủ đề; adaptive reranker mới khớp MiniLM-L6-v2. [BC] baseline tái triển khai theo protocol khác bản gốc; đối chứng Joint-GCG cùng mức fluency chưa có. Không suy ra ưu thế thuật toán tuyệt đối từ bảng ASR. |

Các hạn chế này khác nhau: cần gradient là ràng buộc quyền truy cập; chi phí lớn là ràng buộc tài nguyên; thiếu đối chứng cùng fluency là giới hạn bằng chứng. Chúng không tự thay thế cho nhau.

## 4. Nhóm single-document và điều khiển đáp án

| Peer | Đã xử lý | Giới hạn có căn cứ |
|---|---|---|
| [PoisonedRAG — 2024](https://arxiv.org/html/2402.07867) | Chèn ít passages để trả lời target answer, có white-box/black-box trên corpus lớn. | [PV] target question/answer xác định trước; ngân sách gắn từng query. Không tương đương một poison/trigger dùng chung cho họ query chưa thấy. Giảm nhiều documents xuống một đã có peers kế tiếp. |
| [HijackRAG — 2024](https://arxiv.org/html/2410.22832) | Tách retrieval text, hijack text, instruction text; có transfer retriever. | [TG/PV] bài kiểm định định lượng chủ yếu bằng yêu cầu xuất một marker, trên 100 closed-form questions/dataset (§Experiments). Marker xuất hiện không chứng minh agent làm hành động mục tiêu; target queries biết trước. |
| [CorruptRAG — 2025](https://arxiv.org/html/2504.03957) | Targeted attack chỉ một document mỗi query, với construction tĩnh và rewrite. | [TG] §7: chủ yếu closed-ended queries; untargeted/open-ended và multimodal ngoài phạm vi. “Single-document poisoning” không còn là gap riêng. |
| [AuthChain — Findings EMNLP 2025; cập nhật 2026](https://arxiv.org/html/2505.11548) | Một document với chuỗi bằng chứng/authority; đã thử multi-hop và multi-turn. | [TG] Limitations: chủ yếu factual knowledge và KB công khai; reasoning/open-ended rộng hơn, private KB hoặc kiểm chứng nghiêm ngặt chưa được xác lập. Không được mô tả bài này là chỉ single-turn. |
| [DenialRAG — preprint 2026](https://arxiv.org/html/2608.02678) | Dùng một document phủ nhận đáp án từ parametric knowledge để thúc đẩy target answer. | [PV] cần target và reference answer. [ĐO] bảng theo generator cho thấy không có một chiến lược tốt nhất trên mọi model; đôi chỗ direct injection hoặc CorruptRAG-AK tốt hơn. Không mặc định cùng narrative chuyển tốt giữa generators. |
| [ToxicRAG — preprint 2026](https://arxiv.org/html/2609.11082) | Kết hợp denial, cập nhật giả và authority trong một document, kèm self-validation. | [BC] §5.1: main corpus được lấy mẫu, không phải toàn corpus triệu docs; số calls/độ dài chưa cân bằng; target subsets giữa methods không đảm bảo giống nhau; retrieval metric tính any-poison, không riêng đúng poison của query. Cần tái lập trước khi dùng thứ hạng làm kết luận. |

“Thêm authority”, “phủ nhận knowledge cũ”, “một document” đều đã có prior art. Việc một paper mới báo ASR cao hơn không tự chứng minh một cơ chế mới, hoặc superiority với ngân sách bằng nhau.

## 5. Nhóm pipeline, reranker và phối hợp nhiều bằng chứng

### P³A — ICML 2026

**Đã làm:** rule-based prompt kết hợp perturbation cấp ký tự để tăng thứ hạng qua reranker; khoảng 1% thay đổi và single-document evaluation.

**Giới hạn kiểm chứng của review:** [BC] đã xác nhận proceedings/abstract chính thức; PDF final chưa tải đọc được trong phiên này. Chưa gán điều kiện white-box từ bản submission cũ cho bản final. Vì vậy chưa chốt scientific limitation chi tiết của P³A. Đây là thiếu bằng chứng của review, không phải chứng minh bài không làm được.

**Đối chiếu:** riêng claim “attack vượt reranker với thay đổi nhỏ” đã bị phủ.

Nguồn: [PMLR chính thức](https://proceedings.mlr.press/v306/yang26ap.html).

### CRCP — preprint 2026

**Đã làm:** attack xét việc poison bị chia chunk và lọc qua reranker, thay vì dừng ở dense retrieval.

**Còn giới hạn:** [TG] thiết lập chính dùng surrogate cùng họ kiến trúc; biến thể finite-difference cần reranker score. [PV] không chứng minh tương đương final-answer-only trên pipeline không biết. Phạm vi pipeline hữu hạn không có nghĩa method chắc chắn thất bại ngoài đó.

**Đối chiếu:** “chunking-aware + reranker-aware” đã có peer trực tiếp; muốn claim robustness mới phải chỉ ra loại biến đổi và quyền truy cập còn khác.

Nguồn: [paper, §III và thiết lập thực nghiệm](https://arxiv.org/html/2606.11265).

### InceptionRAG — 2026, tác giả ghi nhận accepted CCS 2026

**Đã làm:** nhiều document riêng lẻ có vẻ vô hại phối hợp dẫn reasoning tới target; tối ưu zero-order và dùng lại qua targets.

**Còn giới hạn:** [TG] §4/Appendix B: tìm kiếm có bước đưa query+payload trực tiếp vào API để mô phỏng context; một phần cost được amortize. [PV] reward ở context dựng sẵn có thể khác deployed retrieval; hiệu quả cần các phần poison được đưa vào context phù hợp. Đây là điều kiện, không phải khẳng định bài không đánh giá end-to-end.

**Đối chiếu:** “compositional/dormant poison” đã có prior art.

Nguồn: [paper](https://arxiv.org/html/2609.16818).

### PoisonArena — benchmark 2025/2026

**Đã làm:** xét nhiều attacker cạnh tranh, cho thấy thứ hạng method phụ thuộc môi trường cạnh tranh.

**Giới hạn:** [PV] benchmark RAG không tự đại diện agent-memory/action. [BC] cách tính success cho output nhắc nhiều target không tương đương quyền điều khiển độc quyền một target.

**Đối chiếu:** kết quả một attacker không đủ kết luận dominance khi có poison khác; nhưng benchmark này cũng không tự giải quyết trigger selectivity.

Nguồn: [paper và appendices](https://arxiv.org/html/2505.12574).

## 6. Multimodal và memory graph: phải đối chiếu, nhưng không gộp threat model

| Peer | Đã xử lý | Giới hạn/phân loại |
|---|---|---|
| [MemVenom — preprint 2026](https://arxiv.org/html/2606.10742) | Triggered retrieval của graph memory đa phương thức và bước hướng agent tới action; có transfer retriever. | [TG] Limitations: sandbox/web frameworks chọn lọc; tập trung screenshot-based graph memory. Chưa xác lập tool-specific/profile memories và pipeline thực tế khác; guards được thử còn nhẹ. Không thể gọi “multimodal agent-memory poisoning” là hướng chưa có. |
| [ViTeGate — preprint 2026](https://arxiv.org/html/2609.14685) | Visual trigger cho retrieval, textual trigger cho generation; có cluster-shared trigger và điều kiện kích hoạt kết hợp. | [TG] Threat Model: white-box source retriever, surrogate LVLM; kiểm soát query image và cung cấp response-side instruction qua prompt/template service. [PV] không phải corpus-only với query nguyên bản. |
| [ShadowMerge — preprint 2026](https://arxiv.org/html/2605.09033) | Poison qua interaction đi qua extraction–merge–retrieval của graph memory, gây xung đột relation cùng anchor. | [TG] Discussion: cần suy ra target query/anchor; tập trung một poisoned relation quanh một anchor. **Peer kề bên:** tấn công đường ghi memory, khác giả định chèn trực tiếp vào kho đã cố định. Không lấy hiệu quả của nó làm baseline static-store ngang quyền. |

Memory động không phải bước tiếp theo bắt buộc. Nó thêm bài toán poison có được ghi, chuẩn hóa, giữ lại và cập nhật hay không. Với scope memory tĩnh, có thể giữ nguyên kho sau poison và nghiên cứu sâu selection/retrieval/conditioning; vẫn phải đối chiếu prior art graph/multimodal khi claim cơ chế rộng.

## 7. Limitation cũ nào đã có peer xử lý?

Bảng này là đối chiếu giữa các bài, không phải tuyên bố paper sau giải quyết toàn bộ paper trước.

| Limitation/claim dễ nghĩ là mới | Peer đã xử lý ít nhất một phần | Điều kiện vẫn phải giữ khi diễn giải |
|---|---|---|
| Phải chủ động nhét trigger vào query | Phantom, BadRAG | Corpus-only natural condition khác arbitrary semantic condition |
| Cần nhiều poison documents | CorruptRAG, AuthChain; các bài single-document khác | Một doc/query khác một doc/họ query |
| Poison không đọc được | AdvDec, CPA-RAG, SilentRetrieval | Fluency/PPL không đồng nghĩa hợp lệ về nguồn gốc và nội dung |
| Chỉ tối ưu retrieval, quên generation | Joint-GCG; PR-Attack với quyền khác | End-to-end optimization đã có; quyền truy cập và transfer vẫn khác |
| Poison bị reranker loại | P³A, CRCP, SilentRetrieval | Không có bảo đảm cho mọi reranker/pipeline |
| Poison bị chunking phá vỡ | CRCP | Phạm vi transformations và tín hiệu tối ưu cần đối chiếu |
| Cần gradients của victim | Nhiều black-box/surrogate peers | Không mặc định black-box = final-answer-only |
| Một poison không phủ nhiều queries | Phantom/BadRAG; GASLITE ở retrieval | Coverage chưa đồng nghĩa selective target behavior |
| Chưa biết trigger nào khả thi | Phantom §6.3 | Dự báo end-to-end cost/success là câu hỏi khác, chưa được chứng minh mới ở đây |
| Mỗi tài liệu phải mang payload hoàn chỉnh | InceptionRAG | Hiệu quả phụ thuộc phối hợp context |
| Chỉ memory văn bản phẳng | MemVenom; ShadowMerge ở write path | Đơn vị memory và quyền attacker khác nhau |
| Chỉ cần chạy optimizer nhanh hơn | GASLITE đã có equal-time comparison; SilentRetrieval có cost comparison | Phải vượt baseline phù hợp và tính chi phí đầy đủ; giảm iterations đơn thuần chưa đủ |

Nguồn cho từng hàng nằm tại entry tương ứng phía trên. Vì vậy, không nên chọn một limitation của AgentPoison rồi mặc định nó còn trống năm 2026.

## 8. Sau đối chiếu, phần nào còn đáng xác minh?

Đây là **tổng hợp của review**, chưa phải danh sách novelty đã xác nhận. Những điểm đáng quan tâm nằm ở điều kiện mà các phương pháp khó đáp ứng đồng thời:

| Vấn đề còn cần xác minh | Bằng chứng khiến nó đáng xét | Chưa được phép kết luận |
|---|---|---|
| Dùng chung rộng nhưng kích hoạt đúng phạm vi | Natural-trigger peers có coverage; targeted peers thường thiết kế theo query; retrieval success khác target behavior | “Chưa ai làm shared/selective trigger” |
| Giữ hiệu quả khi đổi retriever và pipeline với ít thông tin | Phantom có transfer failure; Joint-GCG cần access mạnh; CtrlRAG/CRCP cần feedback cụ thể | “Black-box/transfer là mới” |
| Poison đã tới context nhưng không thắng bằng chứng sạch/knowledge sẵn có | Failure analysis của CPA-RAG; DenialRAG và các authority methods mới xử lý một phần | “Chưa ai giải quyết knowledge conflict” |
| Tối ưu trên feedback quan sát được mà phản ánh deployment | CtrlRAG cần context; InceptionRAG có context simulation; surrogate methods lệch victim | “Chỉ final answer chưa ai thử” — cần rà soát riêng sâu hơn |
| Đổi chi phí lấy độ bền/chọn lọc một cách có cơ chế | Phantom có length/time trade-off; Joint-GCG có setup cost; GASLITE thừa nhận compute lớn | “Nhanh hơn là mặc nhiên đủ mới” |

Việc gộp nhiều ràng buộc vào một benchmark chưa đủ thành đóng góp. Cần tìm được **một nguyên nhân thất bại chung**, chỉ ra vì sao giải pháp hiện có chưa xử lý nó, rồi chứng minh một thay đổi giải quyết được nguyên nhân đó.

## 9. Chưa chọn kỹ thuật; yêu cầu bằng chứng trước khi chọn ý tưởng

Theo yêu cầu hiện tại, không khóa vào cải tiến AgentPoison, gradient search hay efficiency. Kỹ thuật có thể thay đổi sau khi biết failure mechanism.

Để một limitation đủ sức làm nền cho đề tài, cần lần lượt xác nhận:

1. Nó xuất hiện ở peer mạnh, với implementation/budget/threat model đúng; không phải baseline bị làm yếu.
2. Nó không chỉ là artifact của một dataset, model hoặc cách chấm.
3. Peer khác chưa giải quyết được dưới **cùng quyền attacker và cùng ràng buộc**.
4. Có giải thích cơ chế và hướng khắc phục kiểm chứng được; không chỉ thêm một tập thí nghiệm.

Không có yêu cầu rằng đóng góp phải là một optimizer hoàn toàn mới. Một attack mechanism hoặc formulation mới, được triển khai bằng kỹ thuật có sẵn nhưng chứng minh tốt, có thể đáng nghiên cứu hơn một biến thể optimizer. Đây là đánh giá phương pháp nghiên cứu, không phải bảo đảm venue A*.

## 10. Những phần còn thiếu trong chính review

- P³A: xác nhận được bản proceedings, chưa đọc được final PDF; giữ limitation chi tiết ở trạng thái chưa kết luận.
- Các preprint mới năm 2026 được dùng để kiểm tra novelty overlap, không mặc định chất lượng bằng peer đã qua phản biện.
- Chưa tái lập claim ASR/runtime của bất kỳ paper nào trong phiên rà soát này.
- Danh mục tập trung static corpus/trigger và peers trực tiếp. Không phải systematic review bao phủ mọi prompt injection, training backdoor hay query evasion.
- DeRAG là query evasion; MINJA và các memory-write attacks cần bảng quyền riêng. Không trộn vào thứ hạng corpus poisoning.

Trạng thái công việc: **đã lập bản đồ peers và limitations có dẫn nguồn; chưa xác nhận research gap độc lập hoặc chọn ý tưởng**.

## 11. Bổ sung sau yêu cầu tìm hướng khác — 14/10/2026

Vòng tra cứu tiếp theo tìm được các peers quan trọng chưa có trong bảng chính. Vì vậy ledger này **không đủ để bảo vệ claim “đầu tiên”** về universal/reusable attack, multi-trigger, multi-target, coordinated poisoning hoặc dừng kiểm chứng. Đối chiếu method, giới hạn bằng chứng và hệ quả chọn hướng tại [bản đề xuất mới, §2](static_memory_attack_new_directions_2026_10_14.md#2-peers-mới-khiến-phải-bỏ-hoặc-thu-hẹp-nhiều-ý-tưởng).

| Peer bổ sung | Phạm vi cần đưa vào kiểm novelty |
|---|---|
| [Eyes-on-Me](https://arxiv.org/html/2510.00586) | Reusable attention attractors, thay payload và transfer; không đề xuất “optimize một lần” như claim mới |
| [UniC-RAG](https://arxiv.org/html/2508.18652) | Query clustering + shared payload cho universal objective; baseline cần thích nghi nếu nghiên cứu nhiều mục tiêu khác nhau |
| [KEPo](https://arxiv.org/html/2603.11501) | Multi-target cross-subgraph coordination; có đo suy giảm khi mở rộng liên kết |
| [Coverage Is Not Containment](https://arxiv.org/html/2608.16044) | Coordinated top-k capture bằng văn bản, có HNSW end-to-end; khác giả thuyết ANN-specific traversal failure |
| [Worst-case Performance of Popular ANN Implementations](https://arxiv.org/html/2310.19126) | Tập vector khó cho graph ANN; phải phân biệt với ít poison texts chèn vào corpus có sẵn |
| [DRNOISE](https://arxiv.org/html/2607.17291) | False direct-answer document, verification inertia, stopping/evidence reconciliation |
| [ICLAttack](https://aclanthology.org/2024.emnlp-main.642.pdf), [When Correct Demonstrations Hurt](https://arxiv.org/abs/2605.26350) | Correct-label/task-preserving demonstrations không phải bề mặt attack hoàn toàn mới; chưa audit toàn bộ hai methods |
| [Quantization Backdoors](https://arxiv.org/abs/2108.09187) | Backdoor kích hoạt sau lượng tử hóa model đã có; phân biệt với quyền chỉ chèn corpus và nén embedding của N1 |

**Ưu tiên khảo sát mới:** N1 (embedding nén), N2 (multi-target interference), N3 dự phòng (ANN traversal). Đây là quyết định phân bổ công sức, không phải xác nhận peers thất bại hoặc gap đã chứng minh. Không thay thế đánh giá hạn chế bằng chứng ở các mục trước.

PoisonArena ở §5 cũng là đối chứng bắt buộc về interference cho N2; khác biệt đề xuất nằm ở phối hợp conditional mappings dưới budget chung, không phải phát hiện poison cạnh tranh lần đầu.
