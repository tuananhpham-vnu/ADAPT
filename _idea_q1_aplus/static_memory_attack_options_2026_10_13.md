# Các hướng attack-first trên memory tĩnh: peers và kế hoạch pilot

Ngày: **2026-10-13**. Yêu cầu: đề xuất nhiều hướng độc lập từ hạn chế AgentPoison và peers,
không coi đề xuất trước là lựa chọn của người dùng. **Chưa chốt phương pháp nào.**

Memory tĩnh: trong mỗi lượt victim evaluation, corpus/episodic store đã bị can thiệp một
lần rồi giữ nguyên. Agent có thể search nhiều bước; không có write-back, consolidation
hay tự sinh memory mới. Mọi phương án dưới đây giữ nguyên trọng số victim.

Đây là rà soát có mục tiêu tới ngày trên, không phải systematic review. Phân biệt:
(a) điều paper đã nêu; (b) suy luận đề xuất của mình; (c) novelty chưa xác nhận.
Không lấy “không thấy trong abstract” làm bằng chứng chưa có người nghiên cứu.
Không chạy thí nghiệm mới trong lần lập kế hoạch này.

## 1. AgentPoison cho mình nền nào, không còn gap nào miễn phí?

[AgentPoison, §3–4](https://arxiv.org/pdf/2407.12784) cho baseline backdoor memory với
trigger được tối ưu, quyền chèn một số key–value và surrogate embedder truy cập white-box.
Bài đã có mục tiêu retrieval, ràng buộc target generation/coherence, đánh giá transfer
và perturbation. Vì vậy “thêm action objective”, “trigger mượt”, “transfer embedder”
không tự tạo novelty. Một hạn chế hữu ích để mở hướng là sự phụ thuộc của thiết lập vào
query mang trigger đã chọn và hình học dense retrieval.

Các trục cải tiến có thể lựa chọn độc lập:

- **Quyền:** chèn record, sửa record, biết encoder, thấy rank/score hay chỉ final answer.
- **Phạm vi kích hoạt:** token cố định, một topic, hay một điều kiện ngữ nghĩa hẹp.
- **Đích:** final answer, lựa chọn action, khái quát từ experience, hoặc next-search query.
- **Pipeline:** dense, hybrid, rerank, chunking; đánh giá trên cấu hình chưa thấy.
- **Ngân sách:** record, token, số query victim, chi phí tạo ứng viên và compute surrogate.

Chỉ đổi một trục cho khó hơn chưa đủ thành paper: phải có method giải được và bằng chứng
cơ chế/hiệu quả tốt hơn baseline phù hợp trong cùng quyền/ngân sách.

## 2. Bản đồ existing peers

| Nhóm / nguồn chính | Đã có gì | Phần không được claim lại |
|---|---|---|
| [Phantom](https://arxiv.org/abs/2405.20485), [PoisonedRAG](https://arxiv.org/abs/2402.07867) | Tấn công cả retrieval và generation, trigger/targeted query | Chỉ “joint retrieval + output” |
| [BadRAG](https://arxiv.org/html/2406.00083), §4 | Trigger tự nhiên, nhóm trigger và contrastive positive/negative query | Natural/semantic trigger hoặc thêm negative loss nói chung |
| [GASLITE](https://arxiv.org/html/2412.20953), §3–4, §10 | Passage phủ phân phối query; tối ưu tốn compute, có bàn early stopping | Universal/topic coverage hoặc claim đầu tiên giảm chi phí |
| [AdvDec](https://aclanthology.org/2026.findings-eacl.108/) | Văn bản dễ đọc; retrieval objective dùng black-box embedding access | Fluency hoặc black-box nói chung |
| [CorruptRAG](https://arxiv.org/abs/2504.03957), [AuthChain](https://arxiv.org/abs/2505.11548) | Một document; AuthChain xét cả câu hỏi multi-hop | “Chỉ cần một poison”, “thắng nhiều tài liệu sạch” |
| [P³A](https://proceedings.mlr.press/v306/yang26ap.html) | Poison vẫn hiệu quả qua reranker, perturbation nhỏ | Thêm reranker-aware score |
| [CRCP](https://arxiv.org/html/2606.11265), §III–IV | Chunk/rerank robustness; surrogate white-box và score-only fallback | Chunk-aware hoặc black-box reranker chung chung |
| [CtrlRAG](https://arxiv.org/abs/2503.06950), [CPA-RAG](https://arxiv.org/abs/2505.19864) | Black-box/context-aware refinement, generation và retriever scoring | LLM tự sửa candidate hoặc feedback loop nói chung |
| [TellTail](https://arxiv.org/abs/2610.04026) | Suy đoán retriever từ ranking, nguồn hoặc final answer | Fingerprint retriever trước khi attack |
| [InjecMEM](https://arxiv.org/html/2608.23471) | Tối ưu command qua context/vị trí/backbone | Chỉ đổi context khi chọn payload |
| [PoisonArena](https://arxiv.org/abs/2505.12574) | Poison của nhiều attacker cạnh tranh | Rival poisoning như một gap mới |
| [MemoryGraft](https://arxiv.org/abs/2512.16962), [FARMA](https://arxiv.org/html/2607.05029) | Bắt chước kinh nghiệm; giả reasoning history | Poison procedure hoặc reasoning memory nói chung |
| [Fact2Fiction](https://arxiv.org/html/2508.06059) | Tấn công các sub-claim của agent fact-checking | Nhắm nhiều bước reasoning/decomposition nói chung |
| [InceptionRAG](https://arxiv.org/abs/2609.16818) | Các passage riêng lẻ kết hợp gây suy luận sai | Tách payload thành nhiều mảnh |
| [Salience Induction](https://arxiv.org/html/2607.17535), §2–4 | Chỉnh cách trình bày fact đúng, làm lệch binding và search về sau | “Fact đúng vẫn gây sai”, “đổi hướng next search” nói chung |

Mức đọc: AgentPoison đã đọc PDF; BadRAG, GASLITE, CRCP đọc các mục liên quan;
Salience Induction đọc intro/cơ chế; P³A và AdvDec đối chiếu nguồn proceedings.
Các hàng còn lại chủ yếu abstract/tổng quan từ paper hoặc repo; cần đọc đầy đủ và kiểm
artifact của peer sát nhất trước khi đăng ký claim. Nhiều nguồn 2026 là preprint;
không gọi chung chúng là công trình đã peer-review.

CPA-RAG có trang nhà xuất bản ghi số tập năm 2027; rà soát này dùng bản arXiv có từ 2025,
không đưa thông tin xuất bản tương lai vào bằng chứng hiện tại.

## 3. Hướng A — Backdoor theo điều kiện ngữ nghĩa có tính chọn lọc

**Câu hỏi:** có thể làm poison chỉ tác động một điều kiện hẹp trong cùng topic, đồng thời
giữ các câu hỏi gần như giống hệt nhưng khác một điều kiện hoạt động bình thường không?

Ví dụ sandbox: cùng sản phẩm, cùng từ vựng; query “phiên bản cũ + chuyển dữ liệu” thuộc
scope, còn “phiên bản mới + chuyển dữ liệu” và “phiên bản cũ + sao lưu” ngoài scope.
Không thêm chuỗi token nhân tạo vào query người dùng. Target chỉ là nhãn/đáp án sai trong
benchmark; corpus đóng băng.

**Từ limitation tới đề xuất:** AgentPoison dùng trigger có mặt trong query. BadRAG đã
thay bằng từ/nhóm từ tự nhiên và contrastive optimization, nên đề xuất hẹp hơn là kiểm
soát điều kiện quan hệ/AND/NOT với hard negatives cùng từ vựng. Đặc biệt phải generalize
qua cách diễn đạt và tổ hợp chưa thấy. Đây là giả thuyết research, không phải claim rằng
BadRAG chưa bao giờ dùng hard negatives.

**Method ứng viên:** tối ưu nội dung record trên paired queries; ưu tiên target behavior
trong scope, phạt excess target behavior ngoài scope. Hai term retrieval và generation
được ablate riêng. Giữ record tự nhiên, cùng dạng với clean store. Không sửa query test.

**Pilot:**

1. Dựng 30–50 intent groups dev, mỗi group có target, paraphrase và 2–3 counterfactual
   hard negatives; test dùng group/entity/template khác. Tách thành phần điều kiện để
   kiểm tra shortcut từ khóa. Đây là cỡ sàng lọc, không đủ kết luận FPR cực thấp.
2. So AgentPoison ở setting trigger riêng; với setting query tự nhiên, so BadRAG,
   GASLITE/AdvDec và phiên bản không negative objective. AgentPoison được adapt phải ghi rõ.
3. Dùng 1 và 3 records, cùng cap token; một dense retriever trước, model thứ hai sau.
4. Đo target ASR, excess off-target action, clean ACC và đường Pareto ASR–off-target.
   Phân biệt poison bị truy hồi ngoài scope với agent thực sự làm sai ngoài scope.

**Điều kiện tiếp tục:** có lợi ích trên hard negatives và tổ hợp held-out ở cùng budget;
ví dụ tăng ≥5 điểm ASR tại cap excess off-target đã chốt trên dev. Nếu chỉ phân biệt topic
xa nhau hoặc hard-negative BadRAG đơn giản đã ngang bằng thì dừng claim method.

**Chi phí/tái sử dụng:** thấp–trung bình. Dùng encoding, optimizer, metrics sẵn có; thêm
dataset counterfactual và objective. `specificity/selection.py` đã có contrastive scoring,
nên “thêm specificity score” không phải phần mới. Không dùng số pilot cũ bị confound
question-key so với paragraph-corpus để chứng minh hiệu quả hướng này.

**Rủi ro novelty:** vừa–cao. Cần audit cả compositional backdoor ngoài RAG; hard-negative
mining là kỹ thuật đã biết. Đóng góp phải vượt một ablation thêm loss thông thường.

## 4. Hướng B — Ít phản hồi trên pipeline chưa biết

**Câu hỏi:** khi attacker chỉ thấy final answer, một ngân sách nhỏ có đủ chọn poison tốt
trên pipeline dense/hybrid/rerank chưa biết không, và phân bổ budget thế nào tốt hơn search
đều hoặc transfer mù?

**Quyền phải khóa:** trong pha preparation, sandbox cho phép nộp candidate store và xem
final answer; không lộ score/rank/context. Test store nhận đúng bộ poison cuối cùng rồi
đóng băng. Nếu không cho thử candidate trên victim, chuyển thành arm zero-feedback riêng;
không gọi hai quyền này cùng là “black-box”. Quyền thử ghi candidate nhiều lần không miễn phí.

**Peer gần nhất:** CRCP đã có score-only và pipeline transfer; CtrlRAG/CPA-RAG đã có
black-box refinement; AdvDec dùng embedding API; TellTail đã fingerprint từ final answer.
Phần thử mới là **phân bổ hữu hạn final-answer feedback giữa chẩn đoán và chọn candidate**,
đánh giá trên pipeline held-out. Không claim black-box hoặc ensemble tự nó mới.

**Method ứng viên:** portfolio nhỏ các candidate được tạo offline; adaptive allocation
chọn nơi cần thêm đánh giá, dừng sớm candidate yếu, cân bằng hiệu quả với bất định.
Surrogate ensemble giúp tạo candidate, nhưng không được đọc internals victim. Đây là
ứng viên; bandit/successive halving cơ bản phải là baseline, không đổi tên thành method mới.

**Pilot:**

1. Tạo ba victim pipelines cố định: dense; hybrid; dense+rerank. Giữ sạch đủ utility,
   không cố tình chọn cấu hình hỏng. Giữ nguyên query tự nhiên và 1/3 poison records.
2. Cap preparation feedback 0/25/50/100 calls; tính cả fingerprint, kiểm candidate,
   retries, repeats; báo token và offline compute riêng.
3. So transfer AgentPoison/Phantom, P³A/CRCP theo đúng access, best-of-N, uniform search,
   successive halving và fingerprint-then-attack. Không cho baseline xem thêm thông tin.
4. Dev trên hai pipelines, test một pipeline chưa thấy; đảo vòng. Sau pilot, giữ nguyên
   method rồi thêm retriever/reranker khác kiến trúc. Report ASR theo budget và worst-pipeline.

**Điều kiện tiếp tục:** thắng baseline mạnh ở budget thấp trên pipeline chưa thấy, không
chỉ giảm số iteration; lợi ích tồn tại khi tính chi phí preparation đầy đủ. Nếu advantage
chỉ do biết đúng retriever hoặc dư calls, dừng claim.

**Chi phí/tái sử dụng:** trung bình. Có CostLedger/cache/optimizer, thiếu adapter hybrid,
reranker và wrapper che observability. Không cần train generator. Claim nghiên cứu phụ
thuộc một thuật toán phân bổ có lợi ích rõ, không chỉ benchmark thêm cấu hình.

**Rủi ro novelty:** cao; cần đọc sâu CtrlRAG/CPA-RAG và literature black-box ranking.

## 5. Hướng C — Demonstration đúng nhưng gây áp dụng sai điều kiện

**Câu hỏi:** một bộ experience có input–output đúng theo oracle có thể làm agent học nhầm
quy tắc áp dụng ở query mới không, khi attacker chỉ chọn/biên tập các record hợp lệ?

Ví dụ: các ví dụ routing đúng trong một phiên bản workflow, nhưng agent áp dụng chúng
cho phiên bản khác dù query đã nêu phiên bản rõ. Không thay đáp án record bằng đáp án sai.
Giữ nguyên key/scope metadata trong arm nghiêm ngặt để tránh lợi ích chỉ do đổi key.

**Từ limitation tới đề xuất:** AgentPoison sử dụng demonstration có target action độc.
Ở đây ràng buộc record phải pass correctness/scope validation. MemoryGraft đã khai thác
bắt chước kinh nghiệm; FARMA đã giả reasoning; Salience Induction đã dùng fact đúng gây
sai. Phần hẹp cần chứng minh là **suy ra sai quy tắc từ tập demonstration hợp lệ**, không
chỉ đọc nhầm entity hoặc bị một rationale giả thuyết phục.

**Method ứng viên:** chọn tập experience từ pool đã được oracle xác nhận để tối đa hóa
target mistake trên dev queries ngoài điều kiện áp dụng. Arm selection-only giữ nguyên
toàn bộ record; arm chỉnh cách trình bày phải được kiểm chứng lại tất cả phát biểu. Tối ưu
cấp tập, có regularizer diversity và matched random/relevance-only subset controls.

**Pilot:**

1. Tạo task hữu hạn như ticket routing hoặc workflow versioning với luật xác định và
   oracle chấm đúng mọi record. Tách query dev/test theo tổ hợp điều kiện.
2. Giữ fixed keys/retrieval trong ablation để kiểm ảnh hưởng demonstration; phép đo chính
   vẫn retrieval tự nhiên. Không chỉ ép các record vào context.
3. So clean random examples, relevance-matched examples, greedy subset search, AgentPoison
   đổi nhãn (upper-bound khác quyền) và presentation attack phù hợp.
4. Đo target error, clean ACC, validity của từng record, và mức lỗi sau khi thêm một
   counterexample làm rõ điều kiện áp dụng. Đổi thứ tự memory để kiểm positional shortcut.

**Điều kiện tiếp tục:** tất cả record pass oracle và scope check; causal controls cho thấy
sai ở generalization, effect chuyển sang task thứ hai. Nếu bỏ record không hợp lệ làm
attack biến mất, hoặc chỉ cần một entity-salience edit để tái hiện, không giữ claim mới.

**Chi phí/tái sử dụng:** trung bình. Không nhất thiết cần gradient/GPU; cần xây oracle,
định nghĩa validity và task có ý nghĩa. Dễ làm toy demo, khó chứng minh external validity.

**Rủi ro novelty:** cao nhưng câu hỏi agent-specific rõ. Phải audit adversarial in-context
learning, demonstration selection và clean-label poisoning trước khi định vị paper.

## 6. Hướng D — Chọn điểm can thiệp trên chuỗi search, store vẫn tĩnh

**Câu hỏi:** với 1–3 poisoned records cố định, có thể chọn vị trí can thiệp để agent tự
đổi next-search query và đi đến kết quả sai, trên nhiều task chưa thấy không?

Store không đổi từ đầu đến cuối. Thứ thay đổi là working context và search query của
agent. Đây vẫn là static-memory attack, không phải write-back attack.

**Peers:** Fact2Fiction đã nhắm sub-claim; Salience Induction đã mô tả sai binding dẫn
đến next search sai; InceptionRAG đã kết hợp nhiều passage. Do đó không giữ claim rộng
“poison làm lệch reasoning path”. Phần đáng thử là phân bổ một ngân sách poison nhỏ trên
các điểm search dùng chung cho nhiều tasks, có bằng chứng can thiệp nhân quả.

**Method ứng viên:** từ trajectories sạch ở train, chọn các retrieval bottlenecks;
phân bổ fixed poison set theo coverage/influence của từng điểm. Không dùng test trajectory
để đặt poison, không inject trực tiếp tool output, không thay tool list.

**Pilot:**

1. Dùng ReAct QA 2–3 hop, trace chỉ search query, retrieved IDs, tool calls và final answer.
2. So random-hop, first-hop, last-hop, relevance-only coverage và baseline gần Fact2Fiction.
3. Các ablation: khôi phục next query sạch sau encounter; bỏ riêng record gây encounter;
   giữ final retrieved set cố định. Kiểm xem sai thực sự do lệch đường search hay final
   answer đã bị payload trực tiếp quyết định.
4. Target metric: excess final-error trên held-out families, cùng poison/call budget;
   path divergence chỉ là metric cơ chế, không thay ASR.

**Điều kiện tiếp tục:** có lợi ích trên task families chưa thấy và mediation ablation
ủng hộ đúng cơ chế. Nếu chỉ tái hiện Salience Induction/Fact2Fiction, dừng novelty claim.

**Chi phí/tái sử dụng:** trung bình–cao. ReAct có sẵn nhưng cần tracing/replay và oracle
cho hop. Dễ thành project lớn trước khi thấy tín hiệu; không chọn làm pilot đầu.

## 7. Xếp hạng và cách quyết định

Đây là đánh giá kỹ thuật của mình, không phải novelty score đã được chứng minh.

| Hướng | Chi phí pilot | Giá trị nếu thành công | Rủi ro chính | Ưu tiên |
|---|---|---|---|---|
| A: điều kiện ngữ nghĩa hẹp | Thấp–trung bình | Selective backdoor với quyền query yếu hơn AgentPoison | Trùng BadRAG + hard negatives; lexical shortcut | **1: pilot dễ phân biệt nhất** |
| C: demonstrations hợp lệ, khái quát sai | Trung bình | Cơ chế agent-memory rõ, khác payload đổi nhãn | Validity khó; overlap ICL/salience | **2: nếu ưu tiên cơ chế** |
| B: feedback hạn chế/pipeline chưa biết | Trung bình | Đường ASR–chi phí có tính triển khai | Prior art black-box dày; thêm setting chưa đủ | **3: nếu ưu tiên optimization** |
| D: phân bổ poison trên search path | Trung bình–cao | Tấn công process với corpus đóng băng | Peer rất gần, tracing tốn công | **4: dự phòng** |

Hướng “chọn demonstrations theo end-to-end context” ở đề xuất trước giữ làm **baseline
engineering/công cụ dùng chung** cho A–C; chưa chọn làm main paper vì overlap với các
phương pháp context-aware và best-of-N.

Những mở rộng không ưu tiên làm đóng góp độc lập: margin loss, thêm fluency, 1 poison,
thêm reranker/chunking, semantic trigger nói chung, nhiều poison phối hợp, chỉ báo retrieval
khác action. Mỗi ý đều có peer hoặc code hiện hữu.

## 8. Lịch sàng lọc đề xuất, không phải cam kết thời gian

Ước lượng giả định một người, data/model chạy sẵn; GPU/API chờ tải và tái lập peer có thể
làm tăng thời gian. Không chạy tất cả full experiment trước khi chọn.

1. **2–3 ngày:** đọc sâu BadRAG, chốt scope A và đối chứng; dựng paired-query dataset.
2. **3–5 ngày:** pilot A trên một retriever + một LLM; kiểm shortcut và matched budget.
3. **2–3 ngày nếu qua gate:** model thứ hai và held-out composition; chuẩn bị bảng overlap
   từng claim với peer. Nếu fail, giữ negative result rồi chuyển C hoặc B.
4. **Trước full run:** khóa primary metric, split, thresholds, seeds, quyền và chi phí.
   CI bootstrap theo intent/task family; không coi nhiều paraphrase là các mẫu độc lập.

Một ngưỡng pilot ví dụ là gain ≥5 điểm phần trăm và CI không chạm 0, utility loss ≤1 điểm,
nhưng phải đặt trên dev và theo task. Để claim off-target ≤1% cần test đủ lớn; pilot nhỏ
không chứng nhận FPR thấp. Không chỉnh ngưỡng hậu nghiệm hoặc ghi đè tiêu chí P0 cũ.

Thành công của một pilot chỉ cho quyền đầu tư method tiếp; không chứng minh gap chưa ai
làm hoặc đảm bảo công bố. Các source gần nhất phải được so bằng cùng threat model trước.
