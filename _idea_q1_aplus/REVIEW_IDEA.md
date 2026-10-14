# Phản biện chọn hướng nghiên cứu: attack vào memory tĩnh

**Ngày tổng hợp: 14/10/2026.** Đây là bản duy nhất cần gửi để phản biện; các nguồn sơ cấp được dẫn ngay trong file. Nội dung tổng hợp từ các lần rà soát trước, không phải một vòng xác minh literature mới. **N1/N2/N3 đều chưa có kết quả thực nghiệm.**

> **Trạng thái sau phản biện (14/10/2026):** đã có phản hồi ở §8. Thứ tự đổi so với đề xuất ban đầu ở §7:
> **N3 (HNSW) là ứng viên chính** dưới định vị "suppression without displacement", **A (MMR) dự phòng**,
> **N1 hạ thành phép thử ½ ngày** (bị giới hạn bởi slot bound §8.2), **N2 bỏ khỏi kế hoạch A/A\***.
> Bước tiếp theo: đọc 4 peer sát nhất (§8.7) rồi chạy pilot §8.5. Chưa có hướng nào được xác nhận đủ A/A\*.

**Ý tưởng mới (N4) — Suppression without displacement:** xem [§9](#9-ý-tưởng-mới-suppression-without-displacement-n4).

**Yêu cầu với Claude:** phản biện độc lập, không mặc định bảo vệ đề xuất hoặc thứ tự ưu tiên dưới đây. Kiểm novelty, cơ chế, quyền attacker, baseline và khả năng triển khai. Có thể bác bỏ cả ba hướng. Nếu tìm thêm nguồn, dẫn paper và phần hỗ trợ nhận định; phân biệt nguồn đã đọc với suy luận. Ghi phản hồi vào **§8**; §3–5 giữ đề xuất gốc kèm dòng verdict để đối chiếu.

## 1. Mục tiêu và ràng buộc

- **Attack-first, defense-later**, nhắm chất lượng đóng góp phù hợp venue A/A*. Không mặc định lấy AgentPoison làm gốc.
- **Memory tĩnh:** corpus/memory đóng băng sau khi chèn poison; agent có thể truy hồi nhiều lần nhưng không tự ghi memory trong evaluation.
- Quan tâm trigger, thuật toán, pipeline và chi phí; muốn pilot nhỏ trước khi đầu tư GPU hoặc hạ tầng lớn.
- Threat model đề xuất: attacker chỉ chèn ít văn bản, không sửa model, vector, edges hoặc query của người dùng; trigger là điều kiện tự nhiên trong query. Đây là lựa chọn thiết kế để kiểm quyền yếu, có thể bị reviewer phản biện.
- Đánh giá trên testbed được phép, mục tiêu vô hại có checker rõ. Retrieval hit, nhắc target trong câu trả lời và thực hiện đúng hành vi mục tiêu là các kết quả khác nhau.

**Trạng thái bằng chứng:** repo có thử nghiệm cũ về memory động, kết quả phụ thuộc model và writer; chúng không xác nhận N1/N2/N3. Code retrieval hiện có dùng DPR, nối trigger vào query và poisoned keys; tái sử dụng batching/cache được, nhưng không dùng nguyên protocol để báo corpus-only attack. Chưa có profiling chi phí cho ba hướng mới.

## 2. Peers quan trọng và giới hạn thực sự

`[ĐO]`: failure/trade-off đã đo trong thiết lập của bài. `[PV]`: khác phạm vi, chưa chứng minh method thất bại. `[BC]`: thiếu bằng chứng của lần rà soát. Đây là bảng chọn lọc để phản biện, không phải systematic review; paper không có trong bảng không đồng nghĩa gap còn trống.

| Peer / nguồn | Đã xử lý đến đâu | Hệ quả cho đề xuất |
|---|---|---|
| [AgentPoison](https://arxiv.org/html/2407.12784) | Tối ưu trigger bằng geometry embedding, có agent evaluation | `[PV]` Quyền đưa trigger vào đầu vào khác corpus-only. White-box embedder và tối ưu nhiều bước không phải limitation còn trống mặc nhiên. |
| [Phantom](https://arxiv.org/html/2405.20485) | Một document, natural trigger, retrieval + generation, trigger viability | `[ĐO]` Có transfer failure trong một protocol Contriever→DPR/ANCE và trade-off độ dài document. Natural trigger/viability đã có prior art. |
| [BadRAG](https://arxiv.org/html/2406.00083) | Contrastive optimization với positive/negative queries; MCOP gộp triggers; transfer | `[PV]` Multi-trigger và negative loss đã có. N2 phải vượt MCOP có conditional payload, không chỉ bản gốc. |
| [Joint-GCG](https://arxiv.org/html/2506.06151), [GASLITE](https://arxiv.org/html/2412.20953), [AGGD](https://aclanthology.org/2025.findings-acl.222/) | Joint retrieval–generation trên context thực; query-set optimization; token search và đối chứng chi phí | Không định vị bằng “joint loss”, “tối ưu nhiều query” hoặc “HotFlip nhanh hơn” đơn thuần. Phải so methods thích nghi cùng quyền/budget. |
| [Eyes-on-Me](https://arxiv.org/html/2510.00586), §§3–5 | Reusable attention attractors, thay payload, optimize white-box rồi thử transfer | `[PV]` Optimize một lần/tái sử dụng/attention/transfer không mới riêng lẻ. Chưa xác lập capacity nhiều mapping đồng thời trong các thí nghiệm đã đọc. |
| [UniC-RAG](https://arxiv.org/html/2508.18652), §§3–4 | Cluster queries, retrieval prefixes, shared payload cho một universal objective | `[PV]` Khác nhiều nhóm→nhiều mục tiêu có kiểm soát nhầm nhóm; chưa chứng minh mở rộng baseline không giải được N2. |
| [KEPo](https://arxiv.org/html/2603.11501), §§3.3, 4.2.2 | Multi-target cross-subgraph coordination | `[ĐO]` Lợi ích chậm sau 5 corpora liên kết, giảm nhẹ ở 8. Multi-target và suy giảm khi phối hợp đã được quan sát. |
| [PoisonArena](https://arxiv.org/abs/2505.12574) | Poison có mục tiêu xung đột cạnh tranh trên cùng query; đánh giá hiệu quả tương đối | `[PV]` Interference không phải phát hiện mới. N2 cần năng lực phối hợp mapping có điều kiện, dưới một ngân sách chung. |
| [P³A](https://proceedings.mlr.press/v306/yang26ap.html), [CRCP](https://arxiv.org/html/2606.11265) | Poison vượt reranker; tính đến chunking/reranking | “Pipeline-aware” hoặc “qua được reranker” chưa đủ mới. `[BC]` P³A mới xác minh proceedings/abstract trong lượt rà soát trước, chưa audit final method đầy đủ. |
| [CompressionAttack](https://arxiv.org/html/2510.22963), [Quantization Backdoors](https://arxiv.org/abs/2108.09187) | Attack vào prompt compression; backdoor model dormant ở full precision rồi kích hoạt sau quantization | `[PV]` N1 khác ở nén **embedding truy hồi** và quyền chỉ chèn corpus. Đổi bề mặt chưa tự tạo đóng góp lớn. |
| [Coverage Is Not Containment](https://arxiv.org/html/2608.16044), §IV | Nhiều texts phối hợp chiếm top-k; có text realization và HNSW end-to-end | `[PV]` “Coordinated poisoning trên HNSW” đã có. N3 phải tách approximation failure khỏi rank capture vốn xảy ra cả khi exact. |
| [Worst-case ANN](https://arxiv.org/html/2310.19126) | Constructions tập vector khó cho HNSW/NSG/DiskANN | `[PV]` N3 chỉ khác đáng kể nếu ít texts chèn vào corpus đang có tạo ra failure; không được thiết kế toàn bộ tập vector. |
| [Selection Integrity](https://arxiv.org/html/2606.12290) | Graph structural writes làm lệch evidence selection | `[ĐO]` Text-injection-only không có trường hợp vừa stealthy vừa harmful trong 35 rows đã xét; không phải định lý text-only bất khả thi. Peer sát hướng A cũ. |
| [DRNOISE](https://arxiv.org/html/2607.17291) | False direct-answer document, verification inertia, stopping và sử dụng phản chứng | Không chọn “poison khiến agent ngừng kiểm chứng sớm” như novelty riêng. |
| [Black-Hole Attack](https://arxiv.org/abs/2604.05480) | Chèn vector gần tâm dữ liệu → hubness, poison vào top-k đa số query; HNSW/IVF tương đương Flat (theo tóm tắt) | `[BC]` Mới đọc abstract. Quyền chèn vector, mục tiêu được truy hồi. Tín hiệu bất lợi cho N3: ANN có thể không tạo khác biệt riêng. |
| [Jamming / blocker documents](https://arxiv.org/abs/2406.05870), USENIX Sec 2025 | Một blocker document làm RAG từ chối trả lời | `[PV]` Suppression đã có, nhưng blocker phải được truy hồi. N3 khác ở chỗ poison không vào context. |
| [Unreachable points](https://arxiv.org/abs/2407.07871), [Escape Hardness](https://arxiv.org/abs/2510.22316) | Update thường làm 2–4% điểm HNSW unreachable; có cách phát hiện/sửa vùng graph lỗi | `[PV]` Không có attacker. Ủng hộ cơ chế N3, đồng thời là defense sẵn có phải thử. |
| [Learned-index poisoning](https://arxiv.org/abs/2008.00297), SIGMOD 2022 | Chèn key làm hỏng learned index | `[PV]` Tiền lệ "index poisoning"; N3 phải trích và nêu khác biệt. |

Bảng đầy đủ hơn cho từng hướng ở §8.7. Không so ASR giữa papers khác corpus/quyền/budget/cách chấm. Các bài mới chưa xác minh proceedings được xem là preprint. **Khác phạm vi không đồng nghĩa peer không làm được.**

## 3. N1 — poison kích hoạt bởi truy hồi dùng embedding nén

> **Verdict §8.3: sửa → hạ thành phép thử ½ ngày.** Với scoring độc lập, m poison chỉ đẩy evidence lùi tối đa m bậc, nên chỉ query có evidence ở hạng `(L−m, L]` trong chế độ nén mới tấn công được (§8.2). Đo tỉ lệ này trên corpus sạch trước khi làm gì khác.

**Câu hỏi:** cùng model, query và corpus đã có poison, liệu chế độ full-vector trả lời đúng nhưng chế độ vector ngắn/quantized bị điều khiển, dù clean utility của chế độ nhanh vẫn tốt?

**Cơ chế giả thuyết:** biểu diễn nén chọn sai candidates và làm mất evidence sạch; rescoring/reranker không thể lấy lại tài liệu đã bị loại. Pipeline short-vector shortlist → full-vector processing có trong [Matryoshka](https://sbert.net/examples/sentence_transformer/training/matryoshka/README.html); [quantization docs](https://sbert.net/examples/sentence_transformer/applications/embedding-quantization/README.html) có binary/scalar retrieval + rescoring. Đây là bằng chứng tính năng, không phải số liệu mức phổ biến deployment.

**Điểm toán cần giữ đúng:** gọi `T_k` là full-vector exact top-k trên corpus **đã có poison**, `C_L` là compressed candidates, `L ≥ k`. Nếu `T_k ⊆ C_L`, rescoring bằng cùng full-vector scores sẽ khôi phục `T_k`, với tie-break nhất quán. Vì vậy poison tăng rank ở bước nén chưa đủ: phải làm mất evidence có ảnh hưởng. Đây là tính chất ranking sơ đẳng, không phải theorem mới và không áp nguyên xi cho MMR/listwise reranking.

**Thuật toán ứng viên:** token/paraphrase search tối ưu chênh lệch thứ hạng full–short, có near-miss negatives; đánh giá candidates qua rescoring thật, chỉ gọi generator cho shortlist. MRL trước, quantization sau; calibration quantization lấy từ tập sạch cố định. Không coi thêm paired loss là đóng góp đủ mạnh.

**Pilot và phép bác bỏ:**

- 10k passages, encoder thực sự hỗ trợ MRL, full và 2 mức chiều; exact search trong từng biểu diễn để chưa lẫn HNSW. `k=5`, `L=20/50/100`, poison budget 1/5/10 texts; train/dev/test tách entity/topic.
- Baselines: semantic/random decoys, attack full rồi transfer, cùng optimizer tối ưu trực tiếp short, GASLITE/AGGD thích nghi, payload-only. Cùng token/search budget.
- Đo gold-evidence recall, coverage của `T_k` trong candidates, poison sống qua rescoring, targeted ASR, false activation, clean ACC và tổng chi phí attacker/victim.
- Đo ghép cặp: `Δ = (ASR_short_poison − ASR_short_clean) − (ASR_full_poison − ASR_full_clean)`; đồng thời yêu cầu ASR full thấp và utility short tốt. Chỉ `Δ > 0` chưa đủ.
- Sàng lọc đề xuất: clean recall short ≥95% mức tham chiếu trên validation; hiệu ứng còn sau rescoring ở hơn một `L`. Ngưỡng 95% là lựa chọn dự án cần review, không phải chuẩn ngành.
- **Dừng/hạ hướng** nếu chỉ vector tự do hoặc nén quá mạnh mới hoạt động; tăng `L` nhỏ chữa được với cost thấp; hoặc baseline short thông thường đạt tương đương. Hạn mức khảo sát dự kiến 2–3 ngày công khi hạ tầng sẵn, chưa phải GPU-hours đo được.

**Điểm có thể đủ sâu:** attack bằng text dưới quyền yếu + nhân quả candidate loss + đánh đổi security–utility–latency/memory. Một biểu đồ “giảm chiều làm ASR tăng” là quá mỏng.

## 4. N2 — nhiều trigger→mục tiêu trong một poison bank cố định

> **Verdict §8.3: bỏ khỏi kế hoạch A/A\*.** Baseline "một tài liệu bảng luật" nhiều khả năng đã đủ; KEPo, PoisonArena, BadRAG-MCOP, UniC-RAG phủ sát. Chỉ quay lại nếu thấy interference ở tầng generator mà bảng luật không sửa được.

**Câu hỏi:** với budget token/document cố định, có bao nhiêu nhóm query `Q_i` có thể được định tuyến đến các mục tiêu khác nhau `t_i`, trong khi nhóm ngoài trigger vẫn hoạt động đúng? Không thay poison theo test query; không thêm trigger vào query.

**Failure giả thuyết:** (1) retrieval kéo poison nhầm nhóm; (2) poison đúng vào context nhưng bị mục tiêu khác lấn át; (3) generator áp dụng sai điều kiện. Nhóm query phải có thông tin đủ phân biệt, không tạo ambiguity không có nghiệm.

**Thuật toán ứng viên:** đo tác động thêm/bớt poison để dựng đồ thị nhiễu chéo, dùng nó quyết định nhóm nào chia sẻ text, nhóm nào tách payload và nhận thêm budget; ưu tiên nhóm yếu nhất. Phải tốt hơn semantic clustering, greedy allocation và robust loss đơn giản; tính cả chi phí đo đồ thị, không chỉ vòng tối ưu cuối.

**Pilot và phép bác bỏ:**

- Bắt đầu 4 nhóm, mỗi nhóm 20 test queries và tổng 80 near-miss/out-of-group queries; train/dev riêng. Đây là pilot, chưa đủ thống kê cho paper. Sau đó tăng 8/16 nhóm dưới cùng budget.
- Baselines bắt buộc: mỗi nhóm một poison; một tài liệu bảng luật “điều kiện→mục tiêu”; MCOP với conditional payload; UniC-RAG thích nghi nhiều mục tiêu; Eyes-on-Me reuse. Nếu mở sang graph-memory phải so KEPo đúng setting.
- Đo `C_ij = Pr(output thực hiện t_j | q ∈ Q_i)`, thêm hàng ngoài trigger; báo diagonal/worst-group success, confusion, false activation, clean accuracy và cost. “Ra bất kỳ target nào” không phải success đúng nhóm.
- Dùng fixed-context và payload-swap để tách retrieval khỏi generator; oracle context chỉ dùng chẩn đoán. Thí nghiệm poison riêng so với cùng tồn tại là ablation nhân quả, không phải so sánh equal-budget.
- **Tiếp tục** khi có interference ổn định mà bảng luật/phân bổ đơn giản chưa sửa được, và method phục hồi worst-group success dưới cùng budget. **Dừng** nếu lợi ích do thêm tokens/calls hoặc baseline đơn giản đã đủ. Hạn mức pilot composition 1–2 ngày công nếu runner sẵn.

**Điểm có thể đủ sâu:** tăng số mapping đáng tin cậy trên mỗi budget và giải thích rào cản mở rộng. Multi-trigger, multi-target, reuse và interference đều đã có prior art; chỉ thêm confusion matrix/negative loss không đủ. Đây là hướng dễ tận dụng repo nhất nhưng rủi ro novelty đáng kể.

## 5. N3 — chèn text làm lệch đường tìm kiếm ANN

> **Verdict §8.3: nâng thành ứng viên chính, qua pilot vector trước.** Cơ chế đề xuất: chèn node mới làm HNSW prune cạnh của node cũ, khiến evidence khó tới được dù exact rank không đổi và poison không vào top-k (§8.4). Rủi ro do peers chỉ ra: Black-Hole gợi ý ANN-specific effect có thể nhỏ; hiệu ứng phụ thuộc heuristic pruning; đã có defense sẵn (MN-RU, Escape Hardness). Câu "chưa đề xuất làm chính" ở cuối mục này là đánh giá cũ.

**Câu hỏi:** vài văn bản chèn qua ingestion bình thường có làm HNSW bỏ lỡ evidence vốn vẫn ở exact top-k, khi giữ nguyên full-precision embeddings và metric?

**Khác N1:** không nén vector; biến số là tìm kiếm exact hay approximate. Không sửa edges, entry point, model/vector hoặc toàn corpus. Index đóng băng trong evaluation.

**Pilot:** 10k–100k passages có embedding cache, exact reference và HNSW đạt clean recall đã chốt; chọn poison texts từ candidate pool trước khi xây optimizer phức tạp. So random/hard-negative insertion cùng budget; nhiều index seeds và insertion orders mà attacker không kiểm soát. Instrument traversal chỉ sau khi thấy hiệu ứng.

**Điều kiện kết luận:** báo toàn bộ mẫu, tách exact-safe cases để chẩn đoán; không loại các ca thất bại khỏi mẫu số ASR. Đo mất gold evidence do ANN, rồi mới kiểm tác động generator. Nếu poison đã thắng cả exact, chưa chứng minh cơ chế đề xuất.

**Dừng** nếu chỉ một graph seed/order thuận lợi, cần sửa vector/edges, hoặc loss không vượt random insertion. Hạn mức tìm tín hiệu 2–3 ngày công. Worst-case ANN và top-k capture đã có; đóng góp tiềm năng nằm ở low-budget text insertion tạo targeted approximation failure. **Rủi ro hiện thực cao, chưa đề xuất làm chính.**

## 6. Vì sao không tiếp tục các hướng cũ?

| Hướng | Kết luận sau phản biện trước |
|---|---|
| A: text-only selection poisoning qua MMR/dedup | Có toy vector, chưa có text attack end-to-end; dễ chỉ là crowding/paraphrase thiếu chi tiết. Selection Integrity rất sát. Không suy tính phổ biến từ việc framework có MMR; tắt MMR/tăng k cần đo cả utility/cost. |
| B: geometric trigger feasibility | Chỉ diagnostic retrieval; nghiệm vector không bảo đảm text realization/generation. Norm/margin/threshold khiến “constraints ít hơn số chiều → luôn feasible” không đúng; bound có thể không hữu ích. Phantom đã có viability. |
| AgentPoison nhanh hơn / thêm margin | Chưa có failure mới; optimizer/efficiency/query-set optimization đã có nhiều peers. |
| Universal/reusable trigger | Eyes-on-Me và UniC-RAG đã xử lý claim rộng. |
| Poison khiến agent dừng kiểm chứng | DRNOISE đã nghiên cứu trực tiếp; chưa có lý do ưu tiên thêm stopping reward. |

Không có hướng nào bị chứng minh bất khả thi tổng quát. Chúng được hạ ưu tiên vì bằng chứng/novelty hiện tại chưa đủ, không phải để mặc định N1/N2/N3 tốt hơn.

## 7. Quyết định đang đề xuất và câu hỏi cần phản biện

> **Đã thay bằng quyết định ở §8.3–8.5:** N3 chính, A dự phòng, N1 thành phép thử ½ ngày, bỏ N2. Sáu câu hỏi dưới đây được trả lời ở §8.6. Đoạn đề xuất gốc giữ để đối chiếu.

**Đề xuất ban đầu (trước phản biện):** kiểm N1 trước vì có đối chứng cơ chế rõ và liên hệ tài nguyên; N2 dự phòng vì dễ pilot hơn; N3 chỉ mở khi có tín hiệu. Không triển khai cả ba hoặc ghép chúng để bù novelty. Ước lượng ngày công là timebox sàng lọc, chưa bao gồm xây benchmark, reproduction đầy đủ hay paper experiments.

Claude cần trả lời cụ thể:

1. Hướng nào bị prior art phủ? Chỉ rõ paper/section, quyền attacker và phần trùng; đừng chỉ nói “incremental”.
2. N1 có thực tế sau oversampling/rescoring không? Ràng buộc full-safe/short-unsafe có làm bài toán quá hẹp hoặc khó hơn mức impact đáng nhận? Baseline nào dễ giết ý tưởng nhất?
3. N2 có failure đủ lớn ngoài việc nối nhiều poison? Vì sao bảng luật, clustering hoặc Eyes-on-Me thích nghi không đủ? Đo interference có đáng chi phí không?
4. N3 có cửa với ít texts và index randomness thật không, hay chỉ tái hiện worst-case vectors/rank capture? Phép thử rẻ nhất để phân biệt?
5. Chọn tối đa một hướng chính, một dự phòng hoặc bác cả ba. Nếu bác, đề xuất tối đa hai hướng khác với peer gần nhất, cơ chế mới, pilot và điều kiện dừng.
6. Phần nào đủ thành contribution về attack, phần nào chỉ là benchmark/ablation/engineering? Bằng chứng tối thiểu nào còn thiếu để nhắm A/A*?

Đầu ra mong muốn: bảng verdict **giữ / sửa / bỏ**, phản biện mạnh nhất mỗi hướng, nguồn hỗ trợ, rồi một pilot 48–72 giờ có kết quả giúp ra quyết định. Không gán xác suất được nhận paper khi chưa có dữ liệu.

## 8. Phản hồi của Claude

**Ngày: 14/10/2026.** Tổng hợp từ 10 file trong [`archive/research_2026_10_14/`](archive/research_2026_10_14/) và §1–7 ở trên. Ký hiệu: **[nguồn]** = dựa trên paper/tài liệu đã dẫn; **[suy luận]** = lập luận của review, chưa đo; **[đề xuất]** = thí nghiệm chưa chạy. Chưa có kết quả thực nghiệm mới; không gán xác suất được nhận paper.

**Đính chính review trước:** ở §9 của file gap-analysis, tôi viết "số ràng buộc < số chiều → relaxation gần như luôn feasible". Nhận định này sai; §10.4 của file đó đã sửa đúng (norm bound + ngưỡng corpus theo query có thể làm gamma* nhỏ dù chỉ có hai query). Lý do hạ B vẫn đứng, nhưng là do prior art và khoảng cách tới text realization, không phải do "luôn feasible".

### 8.1 Bức tranh tổng hợp: cái gì đã bị phủ, cái gì còn

**Gần như mọi trục "dễ nghĩ" đã có peer** [nguồn: §2, ledger peers, bảng §7 của ledger]:

| Trục | Peer đã phủ |
|---|---|
| Trigger tự nhiên, không sửa query; viability | Phantom, BadRAG |
| Một document / nhiều trigger / universal / reusable | CorruptRAG, AuthChain, BadRAG-MCOP, UniC-RAG, Eyes-on-Me |
| Joint retrieval–generation, context thật | Joint-GCG, PR-Attack |
| Fluency, black-box, reranker, chunking | AdvDec, CPA-RAG, CtrlRAG, SilentRetrieval, P³A, CRCP |
| Poison phối hợp, rank capture, cả trên HNSW | InceptionRAG, Coverage Is Not Containment |
| Đẩy evidence sạch ra, poison biết context | RAG-Narok, Financial Evidence Crowding |
| Selection manipulation | Selection Integrity (quyền ghi edges) |
| Compression (prompt), quantization (model) | CompressionAttack, Quantization Backdoors |
| Multi-target, interference | KEPo, PoisonArena |
| Dừng kiểm chứng sớm, trajectory | DRNOISE, KidnapRAG, FORGE |
| Memory động: persistence, correction | EVOMAL, Zombie Agents, A-MemGuard (audit 12/10) |

**Limitation thật còn lại** [suy luận từ ledger]: gần như mọi peer đánh giá attack bằng **poison có mặt trong context** (retrieval hit, rank capture, displacement bằng chiếm slot). Các defense context-level (RobustRAG, TrustRAG, detector trên retrieved set) cũng nhìn vào context. Khoảng trống đáng kiểm là **làm mất evidence sạch mà poison không cần vượt rank evidence, thậm chí không cần vào context**.

### 8.2 Một điều kiện cần quyết định số phận N1 — "slot bound"

[suy luận, chứng minh sơ cấp] Nếu bước tạo candidates chấm **từng document độc lập** (exact search trong không gian nén, đúng setup P0 của N1), thêm m poison chỉ có thể đẩy rank của evidence e xuống tối đa m bậc. Do đó e rời `C_L` chỉ khi rank nén của e trên corpus sạch nằm trong `(L−m, L]`. Kết hợp với tính chất rescoring ở §3:

`Tập query N1 tấn công được (full-safe) ⊆ { q : ∃ e ∈ T_k(q) có rank_nén(e) ∈ (L−m, L] trên corpus sạch }`.

Hệ quả:

- N1 chỉ tấn công được các query mà chế độ nén **vốn đã gần làm mất evidence**. Khi clean recall ≥95% và L = 4k (mức oversampling thường dùng), tập này nhiều khả năng rất nhỏ. Đây là dự đoán, cần đo.
- **Phép thử giết N1 rẻ nhất không cần poison, không cần GPU:** đo trên corpus sạch tỉ lệ query có thành viên `T_k` rơi vào dải `(L−m, L]`, với từng (dim, L, m). Mất vài giờ CPU khi đã có embeddings.
- Nếu tỉ lệ nhỏ, N1 không thành attack mạnh; mệnh đề này có thể giữ làm **kết quả âm/đối chứng** ("nén + rescoring an toàn trước text insertion có budget nhỏ").
- Lập luận này **không** áp dụng cho selector phụ thuộc tập (MMR) hoặc cho ANN dạng đồ thị, vì ở đó poison tác động mà không cần vượt rank e. Đây chính là chỗ đáng đào.

### 8.3 Verdict từng hướng

| Hướng | Verdict | Phản biện mạnh nhất | Điều kiện để giữ |
|---|---|---|---|
| **N1** embedding nén | **Sửa → hạ thành phép thử 1 ngày** | Slot bound §8.2: với scoring độc lập, attack bị giới hạn ở query mà chế độ nén vốn đã sát ngưỡng mất evidence; tăng L chữa với cost thấp. Quantization Backdoors là analog cấp model của framing "an toàn ở full, độc ở nén". | Tỉ lệ query trong dải `(L−m, L]` đủ lớn ở cấu hình có clean recall ≥95%. Nếu không, chỉ giữ làm mệnh đề/kết quả âm. |
| **N2** multi-mapping | **Bỏ khỏi hướng chính, giữ dự phòng NLP** | Baseline "một tài liệu bảng luật điều kiện→mục tiêu" nhiều khả năng đã đủ khi retrieval định tuyến đúng; phần lỗi còn lại chủ yếu là nhầm retrieval, vốn đã biết. KEPo, PoisonArena, BadRAG-MCOP, UniC-RAG đều sát. Code hiện có dùng protocol query-trigger nên cũng không tái dùng được nhiều. | Chỉ quay lại nếu pilot cho thấy interference ở tầng generator mà bảng luật không sửa được. |
| **N3** ANN/HNSW | **Nâng thành ứng viên chính, nhưng chỉ qua pilot vector** | Phải tách khỏi rank capture (Coverage Is Not Containment) và worst-case ANN (thiết kế toàn bộ tập vector). Black-Hole gợi ý ANN không khác exact với attack kiểu hút query. Hiệu ứng có thể phụ thuộc heuristic pruning của từng implementation. Text realization tới đúng vị trí vector có thể khó. Rebuild index, tăng `ef_search`, MN-RU hoặc Escape Hardness có thể chữa. | Xem §8.4–8.5, §8.7. |
| A cũ (MMR) | Giữ làm **case thứ hai** của cùng cơ chế, không làm riêng | Poison vẫn vào reader context (decoy được chọn), nên detector context-level vẫn có thể thấy; framework thường không bật MMR mặc định (§10.2 gap-analysis). | Dùng như ví dụ "selector phụ thuộc tập" bên cạnh HNSW. |
| B (feasibility) | Bỏ làm hướng; công cụ tùy chọn | Phantom §6.3 đã có viability; relaxation không nói về text realization. | — |
| Write-back repair defense | Tạm gác | Overlap A-MemGuard/corrective lessons; mỗi arm 10–17 h Kaggle. | Sau khi có attack mới. |
| AgentPoison chọn demonstrations | Engineering, không phải paper | Phải thắng best-of-N cùng budget; context-aware selection đã có. | — |

### 8.4 Hướng đề xuất triển khai: "Suppression without displacement"

**Câu hỏi:** chèn ít văn bản qua ingestion bình thường có làm hệ truy hồi **mất evidence sạch mà poison không vượt rank evidence trong exact search**, thậm chí không xuất hiện trong context của reader?

**Vì sao HNSW là điểm vào tốt nhất** [nguồn + suy luận]:

- **Cơ chế khác hẳn slot competition:** khi chèn node mới, hnswlib/FAISS nối node đó với các neighbor và có thể **prune lại danh sách láng giềng của node cũ** (heuristic giữ tối đa M cạnh). Poison đặt quanh e có thể làm các node lân cận bỏ cạnh trỏ tới e, khiến e khó tới được khi search, dù exact rank của e không đổi. [suy luận từ thuật toán HNSW; cần đọc lại code `mutuallyConnectNewElement` của hnswlib và `shrink_neighbor_list` của FAISS trước khi viết claim]
- **Hiện tượng nền đã được ghi nhận nhưng ngoài bối cảnh tấn công:** [Xiao et al., arXiv 2407.07871](https://arxiv.org/abs/2407.07871) báo "unreachable points" tăng lên khoảng 2–4% sau các vòng update/delete trên Gist và ImageNet, và đề xuất MN-RU để giảm. Đây là bằng chứng HNSW có thể mất node do cập nhật bình thường, **chưa phải** bằng chứng attacker nhắm được node cụ thể chỉ bằng chèn. Khoảng 10 lượt tìm kiếm (14/10, §8.7) chưa thấy paper attack nhắm làm một target unreachable qua insertion; "chưa thấy" không có nghĩa "chưa ai làm". Peer gần nhất là Black-Hole (vector, hubness) và Coverage Is Not Containment (rank capture trên HNSW), cả hai đều cần poison vào top-k.
- **Liên quan deployment cao hơn MMR:** HNSW là index mặc định hoặc phổ biến trong nhiều vector DB (Qdrant, Weaviate, pgvector, Milvus) [cần dẫn tài liệu chính thức từng hệ khi viết]. Poison được **append sau corpus có sẵn** là đúng kịch bản thực tế, nên insertion order bất lợi cho attacker ít đáng lo hơn N3 trong §5 lo ngại.
- **Claim bảo mật mạnh:** nếu poison không vào top-k, mọi defense kiểm tra retrieved context sẽ **không thấy gì**. Audit offline bằng exact search cũng không thấy. Đây là câu chuyện đúng kiểu venue bảo mật A* (CCS/USENIX/NDSS), hơn là thêm một optimizer.

**Đơn vị đóng góp tiềm năng, nếu pilot qua:**

1. Mệnh đề slot bound (§8.2) phân loại component nào cho phép suppression không cần displacement: scoring độc lập (exact, nén + rescoring) thì không; selector phụ thuộc tập (MMR) và index đồ thị (HNSW) thì có.
2. Attack text-only cho HNSW: chọn vị trí vector làm "che" e, rồi hiện thực bằng text (bắt đầu bằng chọn passage có sẵn gần vị trí đó, sau đó token search kiểu GASLITE).
3. Đánh giá trên index thật (hnswlib/FAISS, ít nhất một vector DB), nhiều seed, cùng defense: tăng `ef_search`, rebuild index, MN-RU-style repair, detector context-level; báo đánh đổi recall/latency.
4. MMR là case thứ hai cùng nguyên lý (đã có phản ví dụ §10.3 gap-analysis).

Không ghép thêm N1/N2 vào để bù novelty; N1 chỉ xuất hiện dưới dạng mệnh đề/đối chứng âm.

### 8.5 Pilot 48–72 giờ để ra quyết định

Toàn bộ trên CPU với embeddings cache; chưa gọi generator.

| Bước | Việc [đề xuất] | Thời gian | Dừng / chuyển khi |
|---|---|---|---|
| 0 | **Đọc peers sát nhất (§8.7):** full Black-Hole (index nào, đo ANN vs Flat thế nào), Coverage Is Not Containment §IV, cơ chế unreachable trong 2407.07871, Escape Hardness; đọc code pruning của hnswlib và FAISS | ½ ngày | Black-Hole hoặc peer khác đã đo rằng ANN không khác exact cho cả kịch bản làm mất document → bỏ qua bước 2–4, chuyển sang A |
| 1 | **Slot bound N1:** corpus sạch 10k–100k, encoder có MRL; với dim ∈ {full, 256, 128}, L ∈ {20, 50, 100}, m ∈ {1, 5, 10}: đo tỉ lệ query có thành viên `T_k` trong dải `(L−m, L]` | ½ ngày | Tỉ lệ nhỏ ở cấu hình có clean recall ≥95% → đóng N1 như attack; ghi kết quả âm |
| 2 | **HNSW, mức vector:** build index trên corpus sạch (M, `ef_construction`, `ef_search` chốt để clean recall@k đã định), append m vector đặt quanh e (giữa e và các neighbor của e), đo e có mất khỏi ANN top-k trong khi exact top-k không đổi và poison không vào top-k. Đối chứng: vector ngẫu nhiên, passage hard-negative thật, cùng m. ≥5 seed index; chạy cả hnswlib và FAISS vì heuristic pruning khác nhau | 1 ngày | Hiệu ứng không vượt đối chứng ngẫu nhiên, hoặc chỉ xuất hiện ở `ef_search` rất thấp làm clean recall kém → đóng N3 |
| 3 | **HNSW, mức text thô:** thay vector tự do bằng passage thật gần nhất với vị trí mong muốn (chưa tối ưu token) | ½ ngày | Bước 2 có tín hiệu nhưng bước 3 không có gì → còn khả năng nhưng rủi ro text realization cao; đánh giá lại |
| 4 | **Kiểm defense rẻ:** tăng `ef_search` ×2–4, rebuild với thứ tự ngẫu nhiên, MN-RU nếu có code; đo cost để xóa hiệu ứng | ½ ngày | Defense rẻ xóa hoàn toàn → impact thấp, cân nhắc chỉ viết như finding |

**Quyết định sau pilot:** bước 2 có tín hiệu vượt đối chứng → đầu tư N3 làm hướng chính, MMR làm case thứ hai. Bước 2 thất bại → quay về A (MMR) với baseline "paraphrase bỏ ngoại lệ" theo §10.6 gap-analysis. Cả hai thất bại → không còn hướng nào trong vòng này đủ cho A*; nên rà soát lại câu hỏi nghiên cứu thay vì mở N2.

### 8.6 Trả lời 6 câu hỏi của §7

1. **Prior art phủ:** N2 bị phủ nhiều nhất (KEPo §4.2.2 multi-target + suy giảm; PoisonArena §4.3 interference; BadRAG MCOP; UniC-RAG cluster + payload). N1 bị Quantization Backdoors phủ framing ở cấp model (quyền khác), và bị slot bound giới hạn cơ chế. N3 sát Coverage Is Not Containment (HNSW nhưng là rank capture) và worst-case ANN (quyền thiết kế toàn bộ tập).
2. **N1 sau oversampling/rescoring:** với scoring độc lập, slot bound cho thấy bài toán full-safe/short-unsafe bị giới hạn ở query sát ngưỡng; baseline giết nhanh nhất là **tăng L** và **chính phép đo tỉ lệ dải `(L−m, L]` trên corpus sạch**.
3. **N2:** chưa thấy failure nào ngoài nối nhiều poison mà bảng luật không giải; đo đồ thị interference khó hoàn vốn. Không đáng chi phí ở thời điểm này.
4. **N3:** có cửa vì cơ chế prune cạnh khi chèn khác bản chất với slot competition, và kịch bản append sau corpus là thực tế. Rủi ro lớn nhất là Black-Hole (ANN ≈ exact với attack hút query) và sự phụ thuộc vào heuristic pruning. Phép thử rẻ nhất: bước 0 (đọc Black-Hole) rồi bước 2 ở §8.5 (vector-level, so với vector ngẫu nhiên cùng m, nhiều seed, hai implementation).
5. **Chọn:** chính = N3 dưới định vị "suppression without displacement"; dự phòng = A (MMR) với cùng định vị; bỏ N2 khỏi kế hoạch A*; N1 giữ làm mệnh đề/đối chứng âm.
6. **Contribution vs engineering:** attack = cơ chế suppression mới + attack text-only có selectivity + chứng minh vô hình với defense context-level. Benchmark/ablation = sweep k, λ, `ef_search`, nhiều encoder. Engineering = cache, batching, tái dùng `qa_retrieval_eval.py`. **Bằng chứng tối thiểu còn thiếu để nhắm A/A*:** (a) hiệu ứng vượt đối chứng ngẫu nhiên trên index thật, nhiều seed; (b) hiện thực bằng text; (c) tác động tới câu trả lời cuối với target vô hại có checker; (d) ít nhất một defense hiện có thất bại và một mitigation có cost đo được; (e) rà soát prior art kỹ cho "graph ANN poisoning / unreachable target".

**Lịch so với deadline:** file defense ghi CCS 16/01/2027 và USENIX Security 26/01/2027 (cần xác minh lại trên trang chính thức). Từ nay còn khoảng 13 tuần. Nếu pilot §8.5 xong trước 21/10 và có tín hiệu, một bài attack tập trung vào HNSW vẫn kịp về mặt khối lượng; nếu phải lùi về A, nên nhắm chu kỳ sau thay vì ép deadline tháng 1.

### 8.7 Existing peers theo từng hướng (tra cứu bổ sung 14/10)

Khoảng 10 lượt tìm kiếm web; phần lớn mới đọc abstract hoặc trang tóm tắt. **Đây không phải systematic review.** "Chưa thấy" chỉ có nghĩa là các lượt tìm này không ra, không phải "chưa ai làm".

**N3 — HNSW/ANN suppression.** Peer sát nhất được liệt kê trước:

| Peer | Đã làm gì | Khác N3 ở đâu / hệ quả |
|---|---|---|
| [Coverage Is Not Containment, 2608.16044](https://arxiv.org/abs/2608.16044) | Nhiều text ít nổi bật cùng chiếm top-k trên HNSW thật (100k docs; 9.9/10 slots; generator lặp claim 88%) | Rank capture, poison **có trong context**. N3 phải chứng minh evidence mất mà poison không vào top-k. |
| [Black-Hole Attack, 2604.05480](https://arxiv.org/abs/2604.05480) | Chèn vector ở gần tâm dữ liệu → hubness, poison vào top-k của đa số query (tới 94.4%); theo tóm tắt kết quả search, HNSW/IVF cho hiệu quả tương đương Flat | Quyền chèn **vector**, mục tiêu là được truy hồi. Tín hiệu bất lợi: với attack kiểu hút query, ANN không khác exact; N3 phải đo hiệu ứng riêng của ANN. Chưa đọc full paper. |
| [Worst-case ANN, 2310.19126](https://arxiv.org/abs/2310.19126); [HENN, 2505.17368](https://arxiv.org/abs/2505.17368) | Tập dữ liệu đối kháng làm HNSW/NSG/DiskANN chậm tới mức tuyến tính | Attacker thiết kế **toàn bộ** dataset; mục tiêu là latency/chất lượng chung, không nhắm một document. |
| [Adversarially robust ANN, 2601.00272](https://arxiv.org/abs/2601.00272) | Adversary chọn cả dataset lẫn chuỗi query; động lực từ attack trên LSH | Lý thuyết, không phải graph index với ít insertion. |
| [Unreachable points, 2407.07871](https://arxiv.org/abs/2407.07871); [FreshDiskANN, 2105.09613](https://arxiv.org/abs/2105.09613); [LSM-VEC, 2505.17152](https://arxiv.org/abs/2505.17152) | Update/delete bình thường làm 2–4% điểm unreachable và giảm recall; đề xuất MN-RU, backup index | Không có attacker. Ủng hộ tính khả thi của cơ chế, đồng thời đã có **mitigation sẵn** mà N3 phải thử. |
| [Grafeo issue #391](https://github.com/GrafeoDB/grafeo/issues/391) | Cụm > m điểm gần nhau + backlink pruning chỉ giữ nearest → các insert sau bị tách khỏi graph; recall@10 ≈ 0.80 ở m=16 | Bug report, hướng ngược (node **mới** bị mất). Gợi ý cơ chế tồn tại nhưng **phụ thuộc heuristic pruning**: hnswlib dùng heuristic giữ láng giềng đa dạng có thể kháng tốt hơn. Phải thử nhiều implementation. |
| [Escape Hardness, 2510.22316](https://arxiv.org/abs/2510.22316) (SIGMOD 2026 theo tóm tắt) | Phát hiện và sửa vùng graph lỗi quanh query khó | Diagnostic/defense có sẵn; reviewer sẽ hỏi nó có chặn N3 không. |
| [Bounded Provisional Visibility, 2610.05826](https://arxiv.org/abs/2610.05826) | Defense cho kho ingest liên tục, test HNSW trên Milvus | Không phân tích cấu trúc graph; là defense cùng bối cảnh ingestion cần đặt cạnh. |
| [Ghost Vectors, 2606.18497](https://arxiv.org/abs/2606.18497) | Soft-delete trên HNSW để lộ embedding | Privacy, nhưng cho thấy nội bộ HNSW đã là chủ đề security. |
| [Kornaropoulos et al., SIGMOD 2022](https://arxiv.org/abs/2008.00297); [ALEX poisoning, 2604.24975](https://arxiv.org/abs/2604.24975) | Poisoning **learned index** (RMI, ALEX) bằng chèn key | Tiền lệ "index poisoning" ở cộng đồng DB. N3 sẽ bị nhìn như "index poisoning cho graph ANN"; phải trích và nói rõ khác biệt (graph ANN, semantic retrieval, text-only, mục tiêu ẩn một document). |
| [Jamming / blocker documents, USENIX Sec 2025](https://arxiv.org/abs/2406.05870) | Một blocker document làm RAG từ chối trả lời một query | Mục tiêu suppression gần N3 nhưng blocker **phải được truy hồi**. Threat model của họ ghi attacker không xoá được document, nên N3 có thể định vị là "xoá hiệu quả mà không cần quyền xoá". |

**Gap còn lại cho N3** (sau các lượt tìm này): nhắm **một document có sẵn** trong graph ANN, bằng **ít insertion qua text**, sao cho document đó mất khỏi kết quả mà **poison không vào top-k**. Chưa thấy paper làm đúng điều này. Ba rủi ro do peers chỉ ra: (1) Black-Hole gợi ý ANN-specific effect có thể nhỏ; (2) hiệu ứng có thể phụ thuộc mạnh vào heuristic pruning của từng implementation; (3) MN-RU/Escape Hardness có thể là defense rẻ.

**A — MMR/dedup/diversity selection.**

| Peer | Hệ quả |
|---|---|
| Selection Integrity (§2) | Selection manipulation bằng graph writes; text-only 0/35 trong setting của họ. |
| [CPA-RAG](https://arxiv.org/abs/2505.19864), [Polymorphic Sybil benchmark, 2607.03739](https://arxiv.org/abs/2607.03739), [Divide and Doubt, 2609.27090](https://arxiv.org/abs/2609.27090) | Poison đa dạng để **né** dedup/diversity filter, có crowding-out. Nhắm đưa poison vào, không phải lợi dụng dedup để loại evidence sạch. |
| [RAGSieve, 2608.13010](https://arxiv.org/abs/2608.13010) | Near-duplicate injection làm detector mật độ gần như mất tác dụng (theo snippet). Sát với decoy "paraphrase bỏ ngoại lệ" của A. |
| [GRADA, 2505.07546](https://arxiv.org/abs/2505.07546) | Graph-based reranking làm defense; baseline cho selector-level defense. |
| RAG-Narok, Financial Evidence Crowding (§2 gap-analysis) | Displacement/crowding đã có. |

Chưa thấy paper tấn công trực tiếp MMR để loại evidence sạch. Gap hẹp và poison vẫn vào context, nên giữ A ở vị trí case thứ hai.

**N1 — embedding nén.** Quantization Backdoors (model), CompressionAttack (prompt), [continuous-space corpus poisoning, 2504.17884](https://arxiv.org/abs/2504.17884), [robustness của LLM-based retrievers, 2604.16576](https://arxiv.org/abs/2604.16576). Chưa thấy paper tấn công MRL/binary + rescoring. Gap còn trống, nhưng slot bound §8.2 giới hạn cơ chế; vì vậy "trống" ở đây có thể vì giá trị thấp.

**Việc cần làm trước khi chốt N3:** xem bước 0 ở §8.5; thêm code pruning của một vector DB (Qdrant hoặc Milvus) nếu bước 2 có tín hiệu.

## 9. Ý tưởng mới: Suppression without displacement (N4)

> Ý tưởng này phát triển từ phản hồi §8.4. Mục này chỉ ghi **ý tưởng, novelty claim và peers**; kỹ thuật và pilot ở §8.5. Chưa có kết quả thực nghiệm; novelty mới kiểm qua khoảng 10 lượt tìm kiếm, chưa phải systematic review.

**Câu hỏi:** chèn một ít văn bản qua ingestion bình thường có làm hệ truy hồi **mất bằng chứng sạch đang có sẵn**, trong khi poison **không cần thắng thứ hạng của bằng chứng đó**, và có thể **không bao giờ xuất hiện trong context** của mô hình?

**Vì sao đây là một câu hỏi khác.** Hầu hết attack đã công bố cần poison lọt vào context: được truy hồi, chiếm slot top-k, hoặc thuyết phục generator. Hầu hết defense cũng giả định như vậy nên chỉ kiểm tra tập đã truy hồi. Ý tưởng này đổi **mục tiêu** của attack từ "poison được thấy" sang "bằng chứng đúng biến mất".

### 9.1 Novelty claim

1. **Mục tiêu attack mới.** Làm mất một document có sẵn, không cần quyền xoá/sửa, không cần poison được truy hồi. Jamming cũng nhắm suppression nhưng blocker phải được truy hồi; Coverage Is Not Containment và Black-Hole đều cần poison vào top-k.
2. **Nguyên lý phân loại thành phần truy hồi.** Thành phần **chấm điểm từng tài liệu độc lập** (exact search; nén vector rồi rescoring) không cho phép lớp attack này: thêm m tài liệu chỉ đẩy bằng chứng lùi tối đa m bậc, nên bằng chứng chỉ mất nếu vốn đã nằm sát ngưỡng. Thành phần **có tương tác giữa các tài liệu** (index đồ thị như HNSW; selector theo tập như MMR/dedup) thì có thể cho phép, vì kết quả của một tài liệu phụ thuộc tài liệu khác. Đây là đóng góp khái niệm, không gắn với kỹ thuật cụ thể. [suy luận, chứng minh sơ cấp cho nhánh "độc lập"; nhánh "tương tác" cần kiểm chứng]
3. **Điểm mù của defense.** Nếu ý tưởng đứng được, defense kiểm tra retrieved context và audit offline bằng exact search đều không thấy dấu vết. Đây là kết luận có giá trị cho venue bảo mật (CCS/USENIX Security/NDSS).

### 9.2 Hai trường hợp minh họa cùng nguyên lý

| Trường hợp | Vì sao thuộc nhóm "tương tác" | Ghi chú novelty |
|---|---|---|
| **Index đồ thị ANN (HNSW)** — trường hợp chính | Cấu trúc láng giềng thay đổi khi có node mới; khả năng tìm thấy một node phụ thuộc node khác | HNSW phổ biến trong vector DB; poison có thể không vào top-k nên đúng câu chuyện "vô hình" |
| **Selector theo tập (MMR/dedup)** — trường hợp phụ | Điểm của một tài liệu phụ thuộc tài liệu đã chọn | Yếu khi đứng riêng vì poison vẫn vào context; có giá trị như ví dụ thứ hai |
| **Nén embedding + rescoring** — kết quả âm | Chấm điểm độc lập | Minh họa ranh giới của nguyên lý: thành phần này an toàn trước lớp attack trên với budget nhỏ |

### 9.3 Rủi ro novelty

- **"Hiện tượng đã biết + attacker".** HNSW mất node khi cập nhật bình thường đã được ghi nhận (unreachable points), và index poisoning có tiền lệ ở cộng đồng DB (learned index). Để vượt qua, đóng góp phải là **nguyên lý (9.1-2) và điểm mù defense (9.1-3)**, không chỉ một attack trên HNSW.
- **Black-Hole** gợi ý ANN và exact search cho hiệu quả tương đương với attack kiểu hút query. Nếu điều đó cũng đúng cho mục tiêu làm mất bằng chứng, trụ chính của ý tưởng sụp.
- **Phụ thuộc implementation.** Heuristic chọn láng giềng khác nhau giữa các thư viện; claim chỉ đứng trên một implementation sẽ bị coi là bug, không phải lớp lỗ hổng.
- **Defense có sẵn.** MN-RU, Escape Hardness, rebuild index có thể chặn rẻ; khi đó tác động giảm.

### 9.4 Peers sát nhất với N4

| Peer | Đã làm | Khác ý tưởng ở đâu |
|---|---|---|
| [Coverage Is Not Containment, 2608.16044](https://arxiv.org/abs/2608.16044) | Nhiều text phối hợp chiếm top-k trên HNSW thật | Rank capture, poison **có trong context** |
| [Black-Hole Attack, 2604.05480](https://arxiv.org/abs/2604.05480) | Vector gần tâm dữ liệu thành hub cho đa số query; ANN ≈ Flat (theo tóm tắt) | Quyền chèn vector, mục tiêu được truy hồi. **Tín hiệu bất lợi** cho ý tưởng |
| [Jamming / blocker documents, USENIX Sec 2025](https://arxiv.org/abs/2406.05870) | Một document làm RAG từ chối trả lời | Cùng mục tiêu suppression, nhưng blocker **phải được truy hồi** |
| [Unreachable points, 2407.07871](https://arxiv.org/abs/2407.07871); [FreshDiskANN, 2105.09613](https://arxiv.org/abs/2105.09613) | Update/delete bình thường làm một phần node không tìm thấy được; có cách sửa | Không có attacker, không nhắm document cụ thể |
| [Worst-case ANN, 2310.19126](https://arxiv.org/abs/2310.19126); [HENN, 2505.17368](https://arxiv.org/abs/2505.17368) | Dataset đối kháng làm graph ANN suy giảm | Thiết kế **toàn bộ** dataset; nhắm latency/chất lượng chung |
| [Learned index poisoning, SIGMOD 2022](https://arxiv.org/abs/2008.00297); [ALEX poisoning, 2604.24975](https://arxiv.org/abs/2604.24975) | Poisoning index bằng chèn key | Learned index, không phải semantic retrieval/graph ANN; tiền lệ bắt buộc trích |
| [Escape Hardness, 2510.22316](https://arxiv.org/abs/2510.22316) | Phát hiện và sửa vùng graph lỗi | Defense tiềm năng cần đặt cạnh |
| [Selection Integrity, 2606.12290](https://arxiv.org/html/2606.12290) | Ghi cấu trúc graph làm lệch evidence selection; text-only 0/35 trường hợp vừa kín vừa hại | Quyền ghi edges; sát trường hợp MMR |
| [RAG-Narok, 2609.25469](https://arxiv.org/html/2609.25469); [Financial Evidence Crowding, 2609.35782](https://arxiv.org/html/2609.35782) | Đẩy evidence sạch ra bằng chiếm slot | Displacement, poison có trong context |
| [CPA-RAG](https://arxiv.org/abs/2505.19864); [Polymorphic Sybil, 2607.03739](https://arxiv.org/abs/2607.03739); [RAGSieve, 2608.13010](https://arxiv.org/abs/2608.13010) | Poison đa dạng/near-duplicate để né dedup và detector | Né dedup để vào context, không lợi dụng dedup để loại evidence |

**Gap còn lại sau các lượt tìm:** nhắm một document có sẵn, bằng ít văn bản chèn qua ingestion, để document đó mất khỏi kết quả mà poison không vào top-k. Chưa thấy paper làm đúng điều này.

**Cần đọc kỹ trước khi chốt novelty:** full Black-Hole (ANN so với exact đo thế nào), Coverage Is Not Containment §IV, 2407.07871 (cơ chế unreachable), Escape Hardness.

### 9.5 Điều cần có để claim novelty đứng được

- Hiệu ứng làm mất bằng chứng **riêng của thành phần có tương tác**, vượt đối chứng ngẫu nhiên cùng budget, trên hơn một implementation.
- Poison **không xuất hiện** trong kết quả trả về ở phần lớn trường hợp thành công.
- Ít nhất một defense context-level hiện có không phát hiện được, và một mitigation có chi phí đo được.
- Rà soát prior art có hệ thống cho "graph ANN poisoning / targeted unreachability" trước khi viết claim "đầu tiên".
