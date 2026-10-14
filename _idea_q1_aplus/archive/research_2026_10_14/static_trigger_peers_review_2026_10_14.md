# Static-memory trigger attacks: peers, limitations và kế hoạch chọn hướng

**Mốc rà soát: 14/10/2026.** Trọng tâm: attack-first trên memory/knowledge corpus cố định; cải tiến thuật toán tìm trigger, pipeline và chi phí. Đây là literature review và thiết kế thí nghiệm, **chưa phải kết quả chạy method mới**.

**Cập nhật theo yêu cầu tiếp theo:** ưu tiên hiện tại là [đối chiếu từng peer và limitations](static_memory_peer_limitations_2026_10_14.md). Ba hướng/pilot dưới đây là đề xuất trước đó, chưa được chọn; chưa dùng chúng để định hướng kết luận về gap.

**Đánh giá khả thi 14/10 — nhận xét ban đầu, được phản hồi ngay dưới:** pilot 2 (dự đoán trigger nào đáng tìm) về bản chất trùng hướng B trong [phân tích cơ chế](static_memory_mechanism_gap_analysis_2026_10_14.md#review-feasibility). Review dự đoán relaxation gần như luôn feasible trong embedding ~768 chiều nên ít giá trị dự báo; chỉ nên kiểm ~2 ngày và giữ làm công cụ chẩn đoán, không làm hướng chính. Pilot 1 và 3 chưa được đánh giá lại.

**Phản hồi review 14/10 — kết luận hiện hành:** đồng ý hạ ưu tiên pilot 2 vì overlap với Phantom/WARP và chưa chứng minh ích lợi cho cách xây attack. Tuy nhiên, số constraints nhỏ hơn số chiều **không bảo đảm đạt margin yêu cầu** dưới norm bound và ngưỡng corpus. Bài toán maximize gamma cho phép gamma âm vốn luôn feasible; phải so gamma* với required positive margin, không dùng solver status. Diagnostic này chỉ nói về retrieval độc lập, không chứng nhận attack qua MMR/set-dependent selection có hay không tồn tại. Nếu kiểm, giới hạn 0.5–2 ngày với embeddings đúng protocol; không lấy nhãn success của query-side trigger hoặc write-back làm nhãn corpus-only attack. Xem [phản hồi chi tiết, phản ví dụ và cổng kiểm chứng](static_memory_mechanism_gap_analysis_2026_10_14.md#review-response). Chưa có số đo mới để kết luận predictor hữu ích hay vô ích.

## 1. Kết luận để chọn việc làm tiếp

**Không cần lấy AgentPoison làm gốc.** Nên lấy một câu hỏi nghiên cứu làm gốc, rồi chọn baseline theo đúng quyền attacker. AgentPoison vẫn hữu ích cho agent-memory và code hiện có, nhưng không đủ đại diện cho tiến bộ về tối ưu corpus poisoning.

Nhóm nên đọc và đối chiếu trước: **Phantom, BadRAG, Joint-GCG, GASLITE, PR-Attack, P³A, CRCP, SilentRetrieval**. Khi bàn hiệu năng, thêm **Faster-GCG, Probe Sampling và TRIPLE**. Các mục dưới ghi nguồn, thiết lập và giới hạn của từng bài.

Ba hướng đáng làm pilot, theo đánh giá của bản review này:

| Ưu tiên | Câu hỏi cụ thể | Vì sao hợp mục tiêu | Điều kiện để thành đóng góp |
|---|---|---|---|
| 1 | Có thể tìm **trigger chọn lọc** đạt hiệu quả cuối pipeline với ít phép đánh giá đắt hơn, trong cùng ngân sách? | Đi thẳng vào thuật toán và tốc độ; tái sử dụng testbed văn bản | Vượt cả optimizer mạnh và phân bổ ngân sách tổng quát; không chỉ giảm batch/iterations |
| 2 | Có thể dự đoán **trigger nào đáng đầu tư tìm kiếm**, với độ dài, quyền truy cập và ngân sách đã cố định? | Giảm những lượt tối ưu vô ích; có câu hỏi cơ chế rõ | Dự đoán được kết quả tìm kiếm ngoài tập train tốt hơn các chỉ báo hình học và early stopping đơn giản |
| 3 | Trigger ngắn có giữ **đúng điều kiện kích hoạt** khi cả query và document đi qua preprocessing/pipeline chưa thấy? | Đi sâu vào trigger và giới hạn deploy | Hơn augmentation/ensemble thông thường, giữ false activation thấp, tính đủ chi phí |

**Chưa có hướng nào được xác nhận là gap chưa ai làm.** Đây là các giả thuyết còn đáng phân biệt bằng pilot sau khi loại những claim rộng đã có prior art. Ưu tiên 1 là đề xuất của người rà soát, chưa phải lựa chọn người dùng đã chốt.

## 2. Khóa nghĩa của “memory tĩnh” và “trigger”

Memory tĩnh trong review này là **nội dung lưu trữ không bị agent ghi thêm/sửa trong lượt đánh giá**. Attacker tạo poison trước, khóa artifact, rồi evaluate. Agent vẫn có thể tìm kiếm nhiều lần, tạo query mới hoặc suy luận nhiều bước. “Agent động” không tự đồng nghĩa “memory động”.

Knowledge corpus RAG, kho demonstration và memory graph đều là external memory, nhưng khác đơn vị truy hồi và cách dùng bằng chứng. Chuyển kết quả từ QA sang agent tool-use cần kiểm chứng riêng.

| Loại | Attacker điều khiển gì? | Trigger nằm đâu? | Ví dụ/nguồn |
|---|---|---|---|
| Query-trigger + memory poison | Chèn memory và có đường đưa chuỗi vào đầu vào | Query/observation | [AgentPoison](https://arxiv.org/abs/2407.12784) |
| Natural-trigger corpus backdoor | Chèn document; chờ query tự chứa điều kiện | Điều kiện ngôn ngữ trong query; chuỗi tối ưu nằm trong document | [Phantom](https://arxiv.org/html/2405.20485), [BadRAG](https://arxiv.org/abs/2406.00083) |
| Document-side adversarial suffix | Sửa/chèn passage để thay thứ hạng | Document; không nhất thiết có activation key phía user | [GASLITE](https://arxiv.org/html/2412.20953) |
| Prompt-provider + corpus poison | Cung cấp prompt và poison | Prompt được attacker cung cấp | [PR-Attack](https://arxiv.org/html/2504.07717), §3.1 |
| Query evasion | Sửa query để kéo một passage có sẵn lên | Query; không cần poison memory | [DeRAG](https://arxiv.org/html/2507.15042), §4.4 |
| Multimodal conditional poisoning | Thêm memory ảnh–text/graph hoặc poison image–text pair | Điều kiện ảnh/text | [MemVenom](https://arxiv.org/html/2606.10742), [ViTeGate](https://arxiv.org/html/2609.14685) |

Không gộp sáu loại vào một bảng ASR để tuyên bố method mạnh nhất. Đặc biệt: **một poison cho mỗi query** khác **một poison dùng chung cho một họ query**; “black-box” không cho biết attacker có score, embedding, retrieved text, logprob hay chỉ final answer.

## 3. Peers chính và limitation có căn cứ

Quy ước: **[P]** là nội dung/kết quả paper báo; **[I]** là suy luận của review; **[H]** là giả thuyết cần đo. Có code được công bố chưa đồng nghĩa đã tái lập được kết quả.

### 3.1. Trigger và tối ưu retrieval–generation

| Peer / phiên bản kiểm tra | Đã giải quyết gì? | Limitation liên quan và ý nghĩa cho mình |
|---|---|---|
| **AgentPoison**, 2024 — [paper](https://arxiv.org/html/2407.12784) | [P] Tối ưu trigger bằng uniqueness/compactness trong embedding space, kèm ràng buộc coherence và mục tiêu; thử trên agent-memory/KB. | [I] Hình học embedding là surrogate, không bảo đảm hành vi cuối. Cần quyền đưa trigger vào input. Bài đã xét perturbation/rephrasing: không thể claim đầu tiên về trigger semantic robustness. Giữ làm baseline agent-native, không làm trần thuật toán. |
| **Phantom**, v3 01/10/2025 — [§4, §6.2–6.3](https://arxiv.org/html/2405.20485) | [P] Một document, natural trigger, tối ưu retrieval và generation theo hai stage; MCG; có **trigger viability**. | [P] Thí nghiệm transfer Contriever→Contriever-MS: top-5 16%; sang DPR/ANCE: 0% trong thiết lập đó. [I] Không suy rộng thành mọi retriever đều không transfer. Viability hình học đã có; một predictor mới phải vượt chỉ báo khoảng cách hai tâm của họ. |
| **BadRAG**, arXiv v3 29/09/2026 — [metadata](https://arxiv.org/abs/2406.00083), [method](https://arxiv.org/html/2406.00083) | [P] Conditional corpus poisoning với trigger words tự xuất hiện trong query; contrastive alignment và tổng hợp passage. Không cần thay model weights. | [I] “Natural trigger + positive/negative queries” không phải gap mới. Nhóm keyword có thể khác điều kiện quan hệ/phủ định; cần kiểm chứng hard negatives trước khi kết luận thiếu selectivity. Không bỏ bài chỉ vì arXiv ID bắt đầu bằng 2406. |
| **PR-Attack**, SIGIR 2025, v3 20/06/2025 — [§3](https://arxiv.org/html/2504.07717) | [P] Bilevel formulation, alternating optimization poison và soft prompt; một poison/target question; hành vi khác nhau khi trigger bật/tắt. | [P] Attacker có quyền cung cấp prompt cho user. [I] Quyền này mạnh hơn corpus-only; không đem ASR so trực tiếp. “Joint prompt–poison optimization” và “bilevel” đã có; bảo đảm cho formulation không tự là bảo đảm tìm tối ưu toàn cục trong không gian text rời rạc. |
| **Joint-GCG**, AAAI 2026; arXiv v2 12/11/2025 — [§2–4, Appendix C/F](https://arxiv.org/html/2506.06151) | [P] Joint gradients; Cross-Vocabulary Projection, Gradient Tokenization Alignment, Adaptive Weighted Fusion; có transfer, synthetic corpus và batch poisoning. | [P] Tối ưu chính cần white-box retriever + generator. Batch-query experiment chỉ chạy một round vì tốn thời gian (§C.5). [I] Ràng buộc selectivity và ngân sách end-to-end là nơi cần kiểm tiếp; không claim “đầu tiên joint optimization/khác tokenizer”. |
| **GASLITE**, CCS 2025, arXiv v2 — [§4, Appendix F/G.8](https://arxiv.org/html/2412.20953) | [P] Multi-coordinate search, gradient averaging, retokenization; query centroid; concept/unseen-query và multi-passage budget. | [I] Visibility trong retrieval không phải target action. Centroid hiệu quả khi query cố định; không thể bê nguyên vào query-trigger khiến mọi query embedding đổi mỗi step. Đây là baseline optimizer quan trọng, không chỉ là một cách tạo payload. |
| **Adversarial Decoding / AdvDec**, Findings EACL 2026 — [paper](https://aclanthology.org/2026.findings-eacl.108/) | [P] Sinh text bằng decoding có scorer cho adversarial objective và readability; có embedding/retrieval applications. | [I] Readability không còn là claim riêng. Beam/scorer tăng chi phí theo lượng ứng viên; cần đo với cùng quyền encoder/scorer. Là baseline sinh passage tự nhiên, không mặc nhiên tương đương conditional agent-memory backdoor. |

### 3.2. Pipeline hiện đại và các bài mới gần mục tiêu

| Peer | Kết quả/phạm vi đã có | Limitation cần phân biệt |
|---|---|---|
| **P³A**, ICML 2026 — [proceedings](https://proceedings.mlr.press/v306/yang26ap.html) | [P] Prompt engineering kết hợp perturbation cấp ký tự khoảng 1%, hướng vào reranker; có setting một poison và transfer. | [I] “Reranker-aware”, ít edit, một poison không còn mới. Muốn so thuật toán phải kiểm quyền score và số lần scoring. Chưa tái lập; tình trạng code ở §5. |
| **CRCP**, 06/2026 — [When Poison Fails After Retrieval, §III–IV](https://arxiv.org/html/2606.11265) | [P] Chunking + retrieval + reranking; surrogate cùng họ là setting chính, score-only finite differences là fallback; không biết chính xác chunk config. | [I] “Survive unknown chunk boundaries” đã bị chiếm. Score-only cũng mạnh hơn final-answer-only. Cần phân biệt robustness của poison theo target query với selectivity của trigger dùng chung. |
| **SilentRetrieval**, 05/2026 — [§4–6, Appendix G/H](https://arxiv.org/html/2605.28074) | [P] Coordinated Beam Search + Context-Adaptive Trigger Generation, fluency constraint; có adaptive reranker/hybrid. | [P] Setting chính: một poison riêng cho mỗi query. Appendix H: HR@10 còn 76,3% với hai query, 58,4% với năm query/document. [I] Hướng universal selective có thể đáng thử; không được gọi họ chỉ đánh retriever. |
| **InceptionRAG**, 15/09/2026; tác giả ghi accepted CCS ’26 — [§4, Appendix B.2](https://arxiv.org/html/2609.16818), [status](https://arxiv.org/abs/2609.16818) | [P] Chia tác động qua các passage phối hợp; ZOSO dùng feedback victim để chọn suffix. | [P] Reward search đưa query + payload qua API để mô phỏng payload đã được thấy, rồi mới chèn artifact cuối. [I] Reward đó chưa đồng nhất với poisoning qua ingestion/retrieval thật. “Compositional poison” và “black-box suffix optimization” đã có. |
| **MemVenom**, 06/2026 — [§3.1–3.2](https://arxiv.org/html/2606.10742) | [P] Poison memory graph ảnh–text trước deploy/indexing; tách recall, goal-bearing và prioritization component, mục tiêu web action. | [I] Rất gần static agent-memory, phải kiểm novelty nếu chuyển multimodal. Hạ tầng graph + VLM + web task đắt hơn text pilot. Cần kiểm riêng đường tạo trigger-bearing observation; “memory-level access” không tự giải thích quyền sửa observation. |
| **ViTeGate**, 09/2026 — [§1, method](https://arxiv.org/html/2609.14685) | [P] Visual trigger điều khiển retrieval, textual trigger điều khiển generation; kích hoạt phối hợp, có cluster-shared visual trigger. | [I] Hai trigger/semantic cluster/multimodal selectivity không còn trắng. Quyền áp trigger lên query image và text phải khai báo; không lấy làm corpus-only baseline trực tiếp. |

### 3.3. Peers mở rộng: cần để tránh bỏ sót novelty, chưa cần tái lập tất cả

Các mục ngắn dưới đây chủ yếu dùng để phân loại phạm vi; không suy đoán cost hoặc thiếu ablation từ abstract.

| Peer / nguồn gốc | Liên quan và giới hạn so sánh |
|---|---|
| [**Poisoning Retrieval Corpora**, EMNLP 2023](https://arxiv.org/abs/2310.19156) | Tiền thân gradient-based corpus poisoning. Retrieval baseline; không thay thế đánh giá hành vi agent. |
| [**PoisonedRAG**, arXiv 2402.07867](https://arxiv.org/abs/2402.07867) | Target-query knowledge corruption, white/black-box settings. Baseline đơn giản cần có; không đồng nghĩa trigger dùng chung ngoài tập query. |
| [**LIAR — Glue pizza and eat rocks**, 2024](https://arxiv.org/abs/2406.19417) | Bilevel RAG attack khi không biết user queries/LLM parameters, nhưng có retriever access. Không gọi mọi thành phần black-box. |
| [**CorruptRAG**, 04/2025](https://arxiv.org/abs/2504.03957) | Đã đánh targeted poisoning với một adversarial text. “Giảm xuống một document” tự nó không là đóng góp. |
| [**AuthChain**, 05/2025](https://arxiv.org/abs/2505.11548) | Multi-hop poisoning bằng chuỗi mang tính thẩm quyền. Cần đối chiếu trước khi đề xuất reasoning-chain payload. |
| [**The RAG Paradox**, Findings EMNLP 2025](https://aclanthology.org/2025.findings-emnlp.1291/) | Khai thác thông tin nguồn được hệ thống công khai; phù hợp black-box source poisoning. Source-discovery có quyền quan sát khác corpus snapshot local. |
| [**CtrlRAG**, arXiv 2503.06950](https://arxiv.org/abs/2503.06950) | Black-box/MLM-based document attacks. Dùng arXiv identifier để tránh lẫn tên/metadata giữa các phiên bản; cần khóa version trước tái lập. |
| [**CPA-RAG**, arXiv 2505.19864](https://arxiv.org/abs/2505.19864) | Covert document poisoning với surrogate/shadow evaluation. Không còn có thể claim black-box + fluent text là gap. |
| [**PoisonArena**, 05/2025](https://arxiv.org/abs/2505.12574) | Nhiều attacker cạnh tranh trong corpus. Đối chứng cho poison competition; không nhất thiết là cấu hình đầu tiên cần triển khai. |
| [**“Someone Hid It!”**, 02/2026](https://arxiv.org/html/2602.00364), §2–3 | Query-agnostic transfer, min–max với learned query samples; formulation chính làm document **khó được truy hồi**. Không đồng nhất demotion với đưa poison vào context. |
| [**Semantic Chameleon**, 03/2026](https://arxiv.org/abs/2603.18034) | Dual-document poisoning, dense/hybrid retrieval. Cần kiểm trước khi claim sleeper + trigger document hoặc hybrid objective mới. |
| [**DeRAG**, arXiv 2507.15042](https://arxiv.org/html/2507.15042) | Differential evolution, query-side suffix ngắn, feedback ranking. Có ích làm optimizer neighbor; không phải corpus poisoning cùng quyền. |
| [**DenialRAG**, 08/2026](https://arxiv.org/abs/2608.02678), [**ToxicRAG**, 09/2026](https://arxiv.org/abs/2609.11082) | Thay đổi cách phủ định/định khung bằng chứng để đổi output. Kiểm trước khi gọi narrative payload là mới; không tự giải bài toán search efficiency. |
| [**TellTail**, 02/10/2026](https://arxiv.org/abs/2610.04026) | Fingerprinting retriever từ tương tác. Là reconnaissance adjacent; giảm bất định về pipeline có thể đã có lời giải từng phần, không phải poisoning baseline. |

Memory động chỉ giữ để phân định biên: [MINJA](https://arxiv.org/abs/2503.03704), [MemoryGraft](https://arxiv.org/abs/2512.16962), [MemPoison — selective memory](https://arxiv.org/abs/2605.29960), [InjecMEM](https://arxiv.org/abs/2608.23471) nghiên cứu các đường memory ingestion/write hoặc sử dụng kinh nghiệm. Không đem persistence/write-back làm claim cho một thí nghiệm đã khóa memory. Cũng có một bài khác tên [MemPoison — structural blind spots](https://arxiv.org/abs/2607.14651); cần trích đúng arXiv ID.

## 4. Những claim rộng nên loại ngay

| Claim dự định | Prior art khiến claim chưa đủ mới |
|---|---|
| Trigger tự nhiên, chỉ bật khi xuất hiện keyword | Phantom, BadRAG |
| Thêm negative queries / tăng specificity | Phantom, BadRAG; cần điều kiện chọn lọc khó hơn và kết quả ghép cặp |
| Poison dùng chung nhiều query hoặc một poison duy nhất | Phantom, GASLITE, CorruptRAG; phải phân biệt “mỗi query” với “toàn họ query” |
| Joint retrieval + generator hoặc xử lý hai tokenizer | Joint-GCG; PR-Attack ở threat model prompt-provider |
| Thêm reranker/chunk-aware loss | P³A, CRCP, adaptive SilentRetrieval |
| Black-box optimization, không biết user query | RAG Paradox, CPA-RAG; GASLITE và Someone Hid It cho các kiểu query knowledge khác nhau |
| Poison chia nhỏ, các mảnh phối hợp | InceptionRAG, Semantic Chameleon; multimodal có ViTeGate |
| Bớt candidate / cache / early stop / multi-token update | GASLITE, Faster-GCG, MAGIC, Probe Sampling; cần hơn một phép ghép kỹ thuật |
| Bandit chia ngân sách đánh giá | TRIPLE đã làm fixed-budget prompt selection |

Các tên trên liên kết tới nguồn ở §3 và §5. “Có overlap” không có nghĩa hết cơ hội: cần thay claim từ tên kỹ thuật sang **một failure mechanism + một cải tiến đo được trong cùng ràng buộc**.

## 5. Hiệu năng: bằng chứng thực tế và khả năng tái lập

### 5.1. Các số có thể dùng, nhưng không so trực tiếp giữa paper

| Nguồn | Chi phí paper công bố | Cách đọc đúng |
|---|---|---|
| [GASLITE, Appendix G.8/F.1](https://arxiv.org/html/2412.20953) | Cap **4.000 giây/passage**, GPU 3090 24 GB theo paper; batch điều chỉnh theo attack | Là giới hạn thực nghiệm, không phải thời gian tối thiểu hay mọi case đều chạy đúng 4.000 giây. |
| [Joint-GCG, Appendix F.1/C.5](https://arxiv.org/html/2506.06151) | CVP khoảng **2 giờ trên một A6000**, trả một lần cho mỗi cặp retriever–generator; máy thí nghiệm 256 GB RAM | Phải tính preprocessing và điểm hòa vốn; đây không phải latency tạo một poison. |
| [SilentRetrieval, Table 5/Appendix G](https://arxiv.org/html/2605.28074) | **15 phút/document** chuẩn; **42 phút** adaptive reranker; **51 phút** adaptive hybrid + reranker | Số tác giả báo trong implementation của họ; chưa xác minh đủ cấu hình phần cứng để chuyển thành dự toán máy mình. |
| [InceptionRAG, Appendix B.2](https://arxiv.org/html/2609.16818) | 100 suffix candidates × 40 golden queries = **4.000 victim calls**; amortize 100 targets thành 40 calls/target, cộng trung bình 2,1 calls khác | “100 candidates” không phải 100 calls. Chi phí trung bình công bố khoảng 42,1 calls/target phụ thuộc reuse. |

Không điền runtime cho paper chưa tìm được phép đo đáng tin. Số iteration, số candidate, số query và số API call là bốn đơn vị khác nhau.

### 5.2. Peers thuật toán cần có khi claim “nhanh hơn”

| Peer | Cơ chế / kết quả tác giả báo | Vai trò trong thí nghiệm của mình |
|---|---|---|
| [**Faster-GCG**, v2 19/05/2026](https://arxiv.org/abs/2410.15362) | Regularization theo khoảng cách, sampling theo temperature, đánh dấu suffix đã xét; 256K→32K evaluations, tối đa 7× wall-clock speedup trong benchmark của bài | Baseline chống lặp và proposal tốt hơn. Không lấy speedup jailbreak này làm dự báo cho retriever. |
| [**Probe Sampling**, 2024](https://arxiv.org/abs/2403.01251) | Draft model sàng candidate, điều chỉnh theo mức đồng thuận với target; tối đa 5,6× trên Llama2-7B-chat | “Cheap model lọc trước expensive model” đã có. Phải so với bản adapted cùng quyền và ngân sách. |
| [**MAGIC**, COLING 2025](https://arxiv.org/abs/2412.08615) | Chọn update nhờ index-gradient; tác giả báo tối đa 1,5× | Đối chứng nếu claim tốt hơn ở coordinate selection. |
| [**TRIPLE**, NeurIPS 2024](https://arxiv.org/abs/2402.09723) | Fixed-budget best-arm identification cho chọn prompt | Baseline phân bổ evaluation budget; bandit/successive elimination không tự là novelty. |
| [**BEAST**, 2024](https://arxiv.org/abs/2402.15570) | Gradient-free beam search | Đối chứng nếu chọn hướng không backprop; chi phí sinh ứng viên và quality phải cùng thang đo. |
| [**AdvPrompter**, ICML 2025](https://arxiv.org/abs/2404.16873) | Train generator bằng alternating optimization, sinh suffix nhanh ở inference | Chỉ phù hợp nếu workload đủ lớn để hoàn vốn train. Chưa có lý do quay lại train generator chỉ vì inference rẻ. |

### 5.3. Artifact audit tại thời điểm tra cứu

| Artifact | Đã xác minh | Chưa xác minh / hệ quả |
|---|---|---|
| [GASLITE official](https://github.com/matanbt/gaslite) | Có source/config/scripts/demo; README ghi dependency/submodule và cache retrieval similarities | Chưa cài/chạy. Môi trường cũ nên isolate; tính cả cost dựng cache nếu chưa có. |
| [Joint-GCG official](https://github.com/NicerWang/Joint-GCG) | Có implementation, experiment scripts, projection training; đã đọc [core.py](https://raw.githubusercontent.com/NicerWang/Joint-GCG/main/core.py) và [utils.py](https://raw.githubusercontent.com/NicerWang/Joint-GCG/main/utils.py) | `step_llm` tính loss candidate ở cả decoder và encoder; đây là chỗ hợp lý để profile. Chưa benchmark, chưa pin commit upstream. |
| [AdvDec official](https://github.com/collinzrj/adversarial_decoding) | Có beam/scorer framework và retrieval experiment configuration | Chưa chạy; README có phần legacy. Cần khóa commit đúng phiên bản dùng làm baseline. |
| [P³A official](https://github.com/YyyxKun/P3A) | Chỉ thấy README: code coming soon | **Không đánh dấu runnable.** Nếu tự dựng phải ghi reimplementation, kiểm protocol paper trước. |
| [CPA-RAG official](https://github.com/lcy5201314/CPA-RAG) | Chỉ thấy README, thông báo sẽ phát hành | **Không đánh dấu runnable.** Bibliography repo có năm 2027; ở mốc review này dùng arXiv, không coi issue 2027 đã xảy ra. |
| CRCP / SilentRetrieval / MemVenom / ViTeGate | Đọc nguồn paper ở các phần nêu trên | Chưa xác minh được artifact đủ để tái lập; không kết luận “không có code”. |

## 6. Gap ứng viên 1 — Tối ưu trigger chọn lọc trong ngân sách hữu hạn

**Câu hỏi:** với memory đã khóa, có thể giảm số phép đánh giá retriever/reranker/LLM cần thiết để tìm được trigger đạt cùng ASR cuối pipeline và cùng giới hạn kích hoạt nhầm không?

**[H] Cơ chế cần kiểm tra:** surrogate dễ tính xếp hạng candidate sai ở những query sát ranh giới top-k hoặc khi retriever và downstream model bất đồng. Chạy tất cả candidate qua toàn bộ pipeline lãng phí; lọc quá mạnh theo retrieval có thể bỏ candidate có hiệu quả cuối tốt.

Đây là giả thuyết của review. Prior art gần nhất là Joint-GCG/AWF, GASLITE, Probe Sampling và TRIPLE. Claim mới phải nằm ở **cách quyết định cần thêm bằng chứng ở query/stage nào để chọn candidate dưới ràng buộc selectivity**, không phải ở việc có một cascade.

**Kế hoạch pilot:**

1. Khóa hai arm quyền truy cập riêng: corpus-only natural trigger; query-trigger + poison kiểu AgentPoison. Khởi đầu với một arm để giảm khối lượng; khuyến nghị corpus-only cho câu hỏi selective trigger. Không so ASR chéo arm.
2. Giữ candidate generator và payload family giống nhau để đo riêng selection. Thu bảng candidate × query × stage trên tập pilot nhỏ; đây là overhead nghiên cứu, không được bí mật cấp cho method khi evaluate cost.
3. Đo tương quan và đảo thứ tự giữa surrogate score, retrieval hit, retained-context hit và target output. Kiểm riêng near-negative queries, không chỉ query ngoài topic rất xa.
4. So exhaustive scoring, random screening, fixed cascade, successive halving/TRIPLE adaptation và Probe-style screening. Cùng runtime hoặc forward/token budget, cùng số poison và độ dài.
5. Chỉ phát triển selector mới nếu dữ liệu chỉ ra sai lệch có cấu trúc. Selector cần biết lúc nào cheap score không đáng tin; dự trữ một phần budget để audit candidate bị loại, tránh tự tạo bằng chứng thiên lệch.

**Tiêu chí go/no-go đề xuất, chưa chạy:** giảm ít nhất 30% search wall-time ở mức chất lượng tương đương (biên không kém hơn 2 điểm phần trăm), hoặc tăng ít nhất 5 điểm phần trăm ΔASR ở cùng cost; không tăng false activation quá 1 điểm phần trăm. Các ngưỡng cần đăng ký trên dev trước khi mở test; đây là ngưỡng chọn hướng, không phải power analysis.

**Dừng claim algorithm** nếu lợi ích biến mất khi baseline dùng cùng batching/cache, hoặc chỉ có trên một retriever/query family. Khi đó giữ cải tiến như engineering artifact.

**Độ khó:** vừa; gần code hiện có nhất. **Rủi ro novelty:** vừa–cao, vì multi-fidelity/BAI đã có nền tảng mạnh.

## 7. Gap ứng viên 2 — Dự báo khả năng tìm được trigger dưới ràng buộc rời rạc

**Câu hỏi:** trước khi trả toàn bộ search cost, dự đoán được một điều kiện trigger có tạo ra poison hiệu quả hay không, với giới hạn token, corpus/pipeline và compute đã cố định?

Phantom đã có trigger viability; do đó không đặt claim “đầu tiên kiểm khả thi hình học”. Phần có thể phát triển là **khả năng tìm được text hợp lệ trong ngân sách thực tế**, khác với tồn tại một vector embedding phân tách được hai nhóm query.

**[H] Cơ chế:** một hướng trong embedding space có vẻ tốt nhưng không thể được optimizer hiện tại hiện thực bằng text ngắn, retokenization ổn định và ít false activation. Những early search traces có thể giúp biết khi nào nên tiếp tục, restart hoặc chuyển trigger family.

**Kế hoạch:**

1. Tập trigger có các họ điều kiện khác nhau; chia train/dev/test theo họ và entity, không chỉ random query. Giữ đủ near-negatives cùng topic.
2. Với mỗi trigger, chạy cùng một search budget và nhiều seed. Nhãn là “đạt criterion trong budget B”, **không phải “bất khả thi về toán”**.
3. Baseline predictor: khoảng cách tâm kiểu Phantom; score/rank margin ban đầu; slope sau một budget ngắn; random allocation; equal-budget multistart.
4. Chỉ dùng feature mà threat model cho phép. Corpus threshold thật hoặc victim score không được dùng trong zero-query transfer arm.
5. Đo số family thành công trên tổng GPU-hours và calibration ngoài retriever, ngoài token-length train. Tính cả chi phí thu dữ liệu predictor và cold-start; không chỉ inference predictor.

**Tiếp tục** nếu cùng tổng cost giải được thêm ≥20% số family so với baseline mạnh nhất, và không chỉ học tên entity/model. **Dừng** nếu margin/slope đơn giản giải thích hết lợi ích hoặc việc train predictor chưa hoàn vốn trong workload thực tế.

Một continuous relaxation infeasible chỉ cho kết luận theo đúng miền/ràng buộc của relaxation; relaxation feasible hoặc search thất bại không chứng minh text khả thi/bất khả thi. Cần đặc biệt tránh claim certificate từ một classifier heuristic.

**Độ khó:** vừa về implementation, cao về thiết kế nhãn và đóng góp. **Rủi ro novelty:** cao nếu chỉ thêm predictor lên §6.3 Phantom.

## 8. Gap ứng viên 3 — Selectivity dưới biến đổi đồng thời của query và document

**Câu hỏi:** một poison cố định và trigger ngắn dùng chung có giữ activation đúng điều kiện khi query được rewrite, document bị chunk/normalize, rồi rerank bởi pipeline chưa dùng trong search?

CRCP/P³A/SilentRetrieval đã làm nhiều phần của pipeline; AgentPoison cũng xét biến đổi trigger. Vì vậy phạm vi cần thu hẹp vào **selectivity cùng lúc ở positive và hard-negative queries**, dưới cả hai nhánh preprocessing. Đây là giả thuyết giao nhau, chưa có bằng chứng “giao nhau = novelty”.

**Pilot phân biệt mechanism:**

1. Giữ poison artifact cố định; dùng điều kiện hư cấu có thành phần, ví dụ entity + quan hệ + thời điểm. Negatives gồm thiếu một điều kiện, đảo quan hệ, phủ định, cùng keyword nhưng khác ý nghĩa.
2. Khóa bảng 2×2: query nguyên bản/biến đổi × document nguyên bản/biến đổi. Sau đó mới thêm reranker. Đừng thay tất cả cùng lúc rồi không biết vì sao thất bại.
3. Log việc trigger còn tồn tại sau rewrite, tokenization/truncation, poison ở candidate pool, poison vào context cuối và output. Query rewriting chỉ chạy online trên câu hỏi; không sửa memory.
4. Baselines: natural-trigger attack gốc, augmentation đơn giản, surrogate ensemble, CRCP-style pipeline adaptation theo quyền tương ứng. So cả short và long trigger để tránh baseline bị handicap.
5. Hold out loại preprocessing/reranker; không chọn method bằng kết quả pipeline test. Báo worst-group hoặc lower quantile ASR và FAR, không chỉ trung bình.

**Tiếp tục** nếu có failure pattern mới lặp lại và cải tiến hơn simple augmentation ≥5 điểm phần trăm worst-group ΔASR tại cùng FAR/cost. **Dừng** nếu gains chỉ do tăng token, thêm poison, thêm model hoặc biết trước pipeline test.

**Độ khó:** vừa–cao, chủ yếu do eval matrix. **Rủi ro novelty:** cao; không nên là pilot đầu tiên nếu tài nguyên ít.

## 9. Engineering có thể cải thiện ngay, nhưng đừng nhầm với novelty

Rà source local tại HEAD `9f698d8`. Repo hiện đã chuyển các module cũ sang `src/attack/`; các đường `src/triggers/...` trong ghi chú cũ không còn là vị trí hiện tại. Các phát hiện dưới đây là đọc code, **chưa đo runtime/ASR**.

| Phát hiện | Bằng chứng local | Hệ quả và phép kiểm cần làm |
|---|---|---|
| Hai margin loss tối ưu **hai event khác nhau** | [margin/losses.py](../../../src/attack/margin/losses.py): K-th poison > best clean; [dilution/objectives.py](../../../src/attack/dilution/objectives.py): best poison > K-th clean | Full top-k takeover khác hit≥1. Khóa metric và objective trước khi so optimizer. |
| Candidate scoring lặp candidate × query batch | [hotflip_margin.py](../../../src/attack/margin/hotflip_margin.py), vùng 382–400 | Profile candidate forward và tokenization; thử batching/cached encodings hợp lệ. `no_grad` và poison-key reuse theo candidate **đã có**, không tính lại là cải tiến mới. |
| Current loss và candidate loss có thể dùng hai tập batch khác nhau | Cùng file: `DataLoader(...shuffle=True)` ở 282; `iter(loader)` ở 339 và 393 | Acceptance có thể nhiễu vì data khác nhau. Cần so incumbent/candidate trên cùng sample hoặc re-score incumbent. Chưa chứng minh ảnh hưởng đến kết quả cũ. |
| Gradient hook cộng dồn; consumer lại cộng giá trị trả về mỗi batch | [trigger_optimization.py](../../../algo/trigger_optimization.py), `GradientStorage`; caller tại hotflip_margin vùng 343–360 | Với nhiều batch, có nguy cơ trọng số không đều; flag reset mỗi iteration chưa tự giải reset mỗi batch. Cần kiểm gradient bằng bài toán nhỏ trước khi gắn speedup vào đổi thuật toán. |
| Margin proposal bỏ đạo hàm phía poison key | `embed_poison_keys` ở hotflip_margin chạy `no_grad`, score mỗi candidate lại encode key mới | [H] Gradient proposal là xấp xỉ một phía. Có thể thử ablation full/two-sided gradient, nhưng đây không tự là đóng góp vì joint-gradient literature đã có. |
| Micro-batched backward **đã được triển khai** | [trigger_optimization.py](../../../algo/trigger_optimization.py), `micro_batched_backward` | Tính loss chung trên detached embeddings rồi replay từng microbatch; tốn thêm forward để tiết kiệm activation memory. Không claim phát minh mới hoặc hứa vừa ít VRAM vừa nhanh hơn. |
| `free_memory` gọi GC và empty-cache giữa stage | Cùng file, `free_memory` | Chỉ thay sau profile. Đổi tần suất có thể giảm overhead nhưng tăng peak VRAM; ghi allocated và reserved riêng. |
| Có cost ledger tái sử dụng được | [dilution/costs.py](../../../src/attack/dilution/costs.py) | Có forward/backward/scorer counters, phase timing, VRAM và amortization; audit coverage cho reranker/API tokens thay vì dựng ledger mới. |
| E2E hiện chấm plan text | [dilution/e2e.py](../../../src/attack/dilution/e2e.py), metric definitions | Không gọi ASR của plan text là tỷ lệ hoàn thành hành động vật lý/tool execution. |

Đối với bidirectional encoder, cache embedding corpus sạch là hợp lệ khi corpus/model không đổi; cache representation của query chứa trigger cũ thì không. Không bê prefix KV caching của causal LM vào encoder hai chiều mà mặc định giữ nguyên phép tính.

MVP hệ thống nên ưu tiên **correctness + profiler + equal-budget harness**. Pure batching/caching có thể rất có ích cho tốc độ thực nghiệm, nhưng cần gọi đúng là engineering contribution nếu chưa có thuật toán mới.

## 10. Protocol để kết luận đáng tin

### 10.1. Chỉ số và đối chứng

Ký hiệu R là poison có trong retrieval pool, C là poison thực sự đi vào context sau rerank/truncation, A là target output/action. Đo R, C, A riêng; nếu C suy ra R thì có thể báo từng conditional rate, nhưng không giả định mọi target output đều do poison.

Tối thiểu bốn arm ghép cặp:

| Memory | Query | Mục đích |
|---|---|---|
| Sạch | Không trigger | Task utility nền |
| Sạch | Có trigger | Trigger tự nó gây lỗi/target response đến đâu |
| Có poison | Không trigger + hard negatives | False activation và suy giảm utility |
| Có poison | Có trigger | Hiệu quả attack đầy đủ |

Với từng nhóm query, báo **ΔASR = ASR(poison) − ASR(clean memory)** khi giữ nguyên query và generation setting. Đây là effect estimate trong thiết kế ghép cặp; kiểm soát thêm stochasticity và confound trước khi diễn giải nhân quả. Nếu target vốn là đáp án đúng của một số query thì tách chúng khỏi target-error metric.

Chi phí tối thiểu: wall-time search, time-to-target, encoder forward theo số sequence/token, backward count, reranker pair count, LLM input/output tokens/calls, peak VRAM, peak RAM, cache/index build, retries, projection/generator/predictor training. Không chỉ báo iterations hoặc ASR cuối.

### 10.2. Thiết kế vừa sức để bắt đầu

**Đề xuất**, chưa chạy và chưa xác nhận hardware sẵn có:

- Một text corpus, một dense retriever, một generator local; thêm một reranker sau khi dense arm đúng. Dùng hai nhóm surrogate/victim khác họ ở giai đoạn xác nhận transfer.
- Pilot khoảng 20 trigger families × 20 positive + 20 near-negative queries, chia theo family/entity; 3 search seeds. Quy mô này dùng phát hiện failure mechanism, chưa đủ cho claim phổ quát.
- Poison budget 1 và 5; trigger length 4/8/16 trong nhánh query-trigger. Document-suffix budget riêng, không ép mọi baseline về độ dài ngoài thiết lập tự nhiên của nó rồi kết luận yếu.
- Fixed-budget checkpoints, chẳng hạn 1/5/15 phút search; phải giữ cả run timeout/fail/OOM. So cùng hardware, dtype, batch policy, truncation và inference backend.
- Báo cold-start và warm-cache riêng. So cả equal wall-clock và equal expensive-evaluation budget để tách code throughput khỏi sample efficiency.
- Confidence intervals bằng resampling theo trigger family/search seed; không coi các paraphrase cùng family là thí nghiệm độc lập. Điều chỉnh số families sau pilot variance.
- Test set chỉ mở khi đã khóa artifact/selector. Báo one-shot success ở final frozen poison; best-of-N hay retry success là metric phụ, kèm N và toàn bộ cost.

Nếu có model/predictor train trước, dùng:

`C_lifecycle(N) = C_prepare + C_train + N × C_search_or_generate`.

Chỉ có break-even hữu hạn khi chi phí mỗi artifact của method thấp hơn baseline; tính N hòa vốn với cùng quality. Không giấu các lượt search tạo training data. Không có dự toán GPU-hours đáng tin trước khi profile pilot trên máy thực tế.

## 11. Kế hoạch công việc và điểm quyết định

| Giai đoạn | Đầu ra cần review được | Điều kiện đi tiếp |
|---|---|---|
| 0 — Khóa bài toán | Một trang threat model: query quyền gì, memory unit gì, observable feedback gì, poison/token/compute budget | Chọn một arm chính; không trộn quyền baseline |
| 1 — Audit harness | Gradient/sample consistency checks, serializer/retokenization checks, cost ledger có coverage | Các metric đo đúng event; không cần sửa kết quả lịch sử để hợp câu chuyện |
| 2 — Reproduce tối thiểu | AgentPoison cho arm phù hợp; Phantom/GASLITE; Joint-GCG nếu đủ budget; baseline đơn giản PoisonedRAG | Có curve ASR/FAR/cost và failure-stage breakdown trên cùng testbed |
| 3 — Pilot gap 1 | So selectors cùng candidate pool, audit candidate bị loại, kiểm near-negatives | Qua gate ở §6; nếu không thì giữ kết quả âm và thử gap 2 |
| 4 — Xác nhận | Held-out retriever/generator/pipeline, nhiều seed/family, đầy đủ cost | Gain còn sau stronger baselines và fair-budget controls |
| 5 — Defense-later | Defense nhắm failure mechanism đã xác nhận; adaptive re-test và utility | Chỉ làm sau khi có attack/mechanism đủ rõ |

Không cần bắt đầu bằng training một generator hay dựng memory write-back. Với mục tiêu hiện tại, **text static-memory** là lựa chọn thực dụng nhất để cô lập tác dụng của optimizer. Multimodal/graph là nhánh mở rộng sau khi có bằng chứng, không phải cách né prior art của text attack.

## 12. Phạm vi tra cứu, độ tin cậy và việc chưa được chứng minh

- Tìm theo các cụm static memory, triggered memory poisoning, corpus poisoning, query-agnostic attack, retrieval/reranking/chunking, efficient discrete optimization; lần theo references của các paper gần nhất, rồi đọc nguồn gốc arXiv/ACL/PMLR và repo tác giả.
- Đọc sâu method/limitations/cost của Phantom, GASLITE, Joint-GCG, PR-Attack, CRCP, SilentRetrieval, InceptionRAG; rà method/threat model multimodal và metadata cập nhật. Bảng peers mở rộng có mức đọc nông hơn đã ghi rõ.
- Không coi một preprint tự nhận “first” hoặc số ASR tự báo là chứng minh độc lập. Các bài 2026 không có venue được xác nhận ở bảng này được sử dụng như preprint, không mặc định đã peer-review.
- BadRAG có revision tháng 09/2026; Faster-GCG có revision tháng 05/2026. Năm xuất hiện đầu tiên không đủ quyết định một peer “đã cũ”.
- Chưa pin/download toàn bộ upstream artifacts, chưa chạy reproduction, chưa đo hardware local. Vì vậy các nhận xét triển khai là mức đọc source/README, không là lời hứa chạy được ngay.
- Không tìm thấy một hướng nào đủ bằng chứng để ghi “gap chắc chắn chưa có ai làm”. Kết luận mạnh nhất hiện tại là **các claim rộng đã đông, còn ba câu hỏi có thể kiểm chứng với chi phí hữu hạn**.

Tài liệu này cập nhật trọng tâm từ [bốn hướng ngày 13/10](static_memory_attack_options_2026_10_13.md) sang trigger/algorithm/pipeline/cost theo yêu cầu mới. Các run cũ và các ngưỡng của chúng giữ nguyên; không dùng literature review để hợp thức hóa một method chưa đo.
