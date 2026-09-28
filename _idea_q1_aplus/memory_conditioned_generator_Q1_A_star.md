# MCAT: Memory-Conditioned Amortized Trigger Generation

> Trạng thái: kế hoạch nghiên cứu, chưa triển khai hoặc có kết quả cho method này.  
> Ngày: 2026-09-18. Nhánh đề xuất hướng tới Q1 / A*.  
> Tên MCAT là working name, cần kiểm tra trùng tên trước khi dùng trong paper.  
> Liên quan: [đề xuất AQuA](paper_Q1_A_plus.md), [trigger theo nhóm](group_conditioned_triggers.md).
>
> Thứ tự chạy các run thật và ngưỡng phải khóa trước: [attack_first_roadmap.md](attack_first_roadmap.md).
> Code M0–M3 đã hoàn chỉnh (156/156 test pass ngày 2026-09-24); blocker là máy local
> không có GPU, nên arm thật chạy trên Kaggle. Số đầu tiên phải đọc là round-trip gap,
> không phải hit@K.

## 1. Quyết định nghiên cứu

Học một neural generator nhận trạng thái benign memory và phân phối query sạch,
sinh trigger dùng chung cho một episode. Generator được huấn luyện qua nhiều
episode để giảm nhu cầu tối ưu lại từ đầu khi memory/query distribution thay đổi.

Không lấy việc thay HotFlip bằng `backward()` và Adam làm đóng góp chính. Học một
ma trận logits bằng continuous relaxation là baseline bắt buộc. Đóng góp dự kiến
là **khả năng amortize việc tìm trigger theo trạng thái memory và query**, được
kiểm chứng trên episode chưa thấy, memory drift và tổng chi phí thực tế.

Pitch dự kiến, chỉ dùng như claim sau khi có bằng chứng:

> A memory- and query-conditioned generator amortizes discrete trigger search
> across retrieval-agent environments and adapts to memory changes under explicit
> poisoning budgets.

MVP dùng một retriever/tokenizer cố định. Cross-retriever generation, cluster-free
loss và adversarial training cho defender là các nhánh mở rộng có điều kiện;
không ghép tất cả vào phiên bản đầu.

## 2. Research questions và giả thuyết có thể bác bỏ

| RQ | Câu hỏi | Phép kiểm chứng quyết định |
|---|---|---|
| RQ1: conditioning | Memory/query context có giúp hơn một generator vô điều kiện? | Bỏ từng input, tráo memory context giữa các episode, so với trigger bank |
| RQ2: amortization | Chất lượng trigger trên episode mới có đủ tốt để bù chi phí train? | Đường chất lượng–chi phí, số episode đạt hòa vốn |
| RQ3: dynamics | Sinh lại trigger theo snapshot mới có tốt hơn giữ trigger cũ? | Chuỗi memory growth/drift, tách fixed-poison và refresh-poison |
| RQ4: discretization | Gain trong loss có tồn tại sau decode và tokenize lại? | Retrieval và agent behavior trên text thực, đo round-trip gap |
| RQ5: security relevance | Retrieval thành công có dẫn đến thay đổi hành vi target? | Thí nghiệm memory sạch/poison × trigger tắt/bật trong sandbox |

Giả thuyết trung tâm: conditioning giúp chọn trigger phù hợp với hình học của
episode mới, thay vì chỉ sinh một trigger phổ quát hoặc truy xuất lời giải đã nhớ.
Trigger khác nhau về mặt chữ chưa đủ để xác nhận giả thuyết này.

## 3. Novelty boundary và nguồn neo

Đã đối chiếu landing page/abstract của các nguồn dưới đây ngày 2026-09-18.
Đây là khảo sát ban đầu, chưa phải systematic review hay bằng chứng rằng không có
công trình trùng. Cần đọc full paper và đối chiếu code trước khi chốt claim.

| Nguồn | Phần đã có / cần đối chiếu | Hệ quả cho đề xuất |
|---|---|---|
| [AgentPoison](https://arxiv.org/abs/2407.12784) | Poisoning memory/knowledge base của retrieval agents bằng trigger | Là baseline cùng bài toán; giữ rõ quyền sửa query và memory |
| [GBDA, EMNLP 2021](https://aclanthology.org/2021.emnlp-main.464/) | Tối ưu phân phối adversarial text bằng ma trận liên tục | Học logits bằng gradient không phải novelty |
| [Gumbel-Softmax](https://arxiv.org/abs/1611.01144) | Relaxation cho biến categorical | Là công cụ tối ưu kế thừa |
| [AdvPrompter](https://arxiv.org/abs/2404.16873) | Học model sinh suffix theo instruction | Có generator thích ứng đã là prior work; phải chứng minh giá trị của memory context |
| [AdvBDGen](https://arxiv.org/abs/2410.11283) | Prompt-specific fuzzy backdoor generator | Cần phân biệt poisoning weights/alignment với poisoning retrieval memory |
| [Neural Exec](https://arxiv.org/abs/2403.03792) | Execution triggers cho prompt injection | Không nhận novelty chỉ từ việc học trigger hoặc dùng trong RAG |
| [Natural Triggers](https://arxiv.org/abs/2005.00174) | Universal adversarial triggers tự nhiên cho classification | Naturalness là một trục đánh giá, không phải claim mới tự thân |
| [Calibrated Gumbel-Softmax suffixes](https://arxiv.org/abs/2512.08123) | Universal suffix qua relaxation | Cần so direct-logit baseline và discretization |
| [Optimizing Against Safety Representations](https://arxiv.org/abs/2607.08883) | Có phương pháp Soft-GCG | Soft-GCG là tên method trong paper này; không chuyển speedup được báo cáo sang bài toán của mình |
| [TROPT](https://arxiv.org/abs/2606.23496) | Framework discrete text optimization | Đối chiếu optimizer/loss/task coverage trước khi khẳng định khoảng trống |

Không claim “first neural trigger generator”, “first backpropagation over tokens”,
hoặc “end-to-end differentiable agent attack”. Claim hẹp cần hướng tới là khả năng
thích nghi theo **memory distribution**, có kiểm soát query context, chi phí và
quyền cập nhật poison records.

## 4. Threat model và đơn vị thích nghi

### 4.1 Quyền và thông tin trong MVP

- Nghiên cứu white-box retriever: biết checkpoint, tokenizer và scoring function.
- Có quyền đọc benign memory snapshot và một tập query support sạch đã được cho phép.
- Có ngân sách B poison records và tối đa L trigger tokens; quy định vị trí chèn.
- Có thể gắn trigger vào query trong môi trường đánh giá có kiểm soát. Không suy
  diễn thành attacker chỉ được sửa memory nhưng vẫn điều khiển được mọi query.
- Target LLM được giữ cố định. PPL và target behavior là kiểm tra rời rạc bên ngoài
  ở MVP, không coi API/ASR là một đường gradient.
- Không sửa model weights, tokenizer hoặc logic retrieval của hệ thống đích.

MVP sinh **một trigger mỗi episode/snapshot**, dùng lại trên nhiều query chưa thấy.
Không cho generator đọc từng test query rồi gọi kết quả đó là universal trigger.

### 4.2 Hai protocol phải báo cáo riêng

| Protocol | Khi memory thay đổi | Chi phí và giới hạn |
|---|---|---|
| Refresh-poison, protocol chính | Sinh trigger mới và thay tối đa B poison records theo quyền đã định nghĩa | Tính lại poison embeddings, index update và số writes; cùng quyền cho baseline |
| Fixed-poison, stress test | Poison text/keys được khóa từ snapshot đầu; chỉ có thể đổi trigger phía query | Không encode lại poison bằng trigger mới; không giả định chất lượng được giữ |

Ở fixed-poison, generator có thể nhận thêm summary của poison keys đã cố định,
nhưng phải đánh dấu đây là biến thể input khác và cấp cùng thông tin cho baseline.
Nếu không thêm input đó, đo trực tiếp mức suy giảm của generator chính.

Giới hạn B là tổng số records đang tồn tại, không phải B mới ở mỗi vòng rồi cộng
dồn vô hạn. Báo cáo thêm tổng write budget xuyên suốt chuỗi snapshot. Mọi payload
và template behavior giữ cố định giữa các phương pháp trong so sánh chính.

## 5. Episode và chia dữ liệu

Một episode:

\[
\mathcal E_e=(\mathcal D_e,Q_e^{sup},Q_e^{opt},Q_e^{eval},R_e,\mathcal P_e,B,L).
\]

- D: benign memory snapshot; P: poison templates hoặc fixed poison records.
- Q_sup: query sạch dùng tạo conditioning; không có nhãn hành vi test.
- Q_opt: query dùng loss khi train hoặc few-step adaptation nếu protocol cho phép.
- Q_eval: query chỉ dùng chấm điểm sau khi đã khóa trigger.
- R: encoder query/key, tokenizer, normalization và similarity đúng của runtime.

Train generator bằng support → context, rồi tính loss trên Q_opt khác support.
Validation có split riêng để chọn checkpoint, temperature, weights và ngưỡng PPL.
Ở test zero-shot, chỉ đọc D và Q_sup; không backward và không chọn candidate bằng
Q_eval. Ở test few-step, được dùng Q_opt với ngân sách công bố, vẫn khóa Q_eval.

Quy tắc split:

1. Chia theo nguồn document/base-task/domain trước khi tạo episode hoặc augmentation.
2. Giữ mọi paraphrase, near-duplicate và snapshot descendants của cùng base family
   trong cùng outer split. Dedupe bằng ID/text và kiểm tra near-duplicate có thể có.
3. Memory trong test là input hợp lệ lúc triển khai; đọc nó không có nghĩa là được
   đưa nó vào train loss hoặc chọn hyperparameter. Ghi rõ mức độ transductive này.
4. Các timestamp trong một test trajectory có thể dùng chung tài liệu; coi trajectory
   là đơn vị thống kê, không coi từng snapshot là mẫu độc lập.
5. Resample subset của một corpus chỉ chứng minh within-corpus generalization.
   Claim unseen domain/corpus cần nguồn ngoài độc lập thực sự.

Pilot đề xuất, điều chỉnh sau kiểm tra dung lượng dữ liệu: 60/20/20 episode
train/validation/test, 3 training seeds, mỗi episode 32 support và 64 optimization
queries; tối đa 128 evaluation queries nếu đủ dữ liệu độc lập. Bắt đầu L=10,
B=5, K_retrieval=5; đây là cấu hình thử nghiệm, không phải thông số đã tối ưu.
Không nhân bản query để đạt các con số này; giảm quy mô và báo cáo overlap nếu thiếu.

## 6. Kiến trúc generator

### 6.1 Memory encoder và query encoder

Encode benign memory bằng key encoder của retriever và support query bằng query
encoder đúng runtime. Cache vectors sạch; không tính lại chúng mỗi gradient step.

Hai cách memory summary cần so sánh:

- Mốc đơn giản: 5 GMM centers, mixture weights và dispersion mỗi component.
- Nhánh chính: lấy một tập key embeddings có seed rồi dùng DeepSets mean-pooling
  hoặc attention pooling không có positional encoding theo thứ tự document.

Query summary dùng một set encoder riêng trên support embeddings. GMM centers là
**benign reference**, không phải query router hoặc năm trigger bắt buộc. Với GMM,
pool các component như một tập có weights để tránh phụ thuộc thứ tự nhãn cụm.

\[
c_m=f_\phi(\{E_k(d):d\in D\}),\quad
c_q=f_\psi(\{E_q(q):q\in Q^{sup}\}).
\]

MVP dùng pooled context c=[c_m;c_q], learned position vectors p_l và một MLP:

\[
h_l=\operatorname{MLP}_\theta([c;p_l]),\quad
A_{l,:}=h_lW_{out}^{\top}+b.
\]

H có shape L×h; W_out có shape V×h; A có shape L×V. Sau MLP baseline mới thử
decoder attention để các vị trí phối hợp tốt hơn. Tất cả weights của set encoders
và generator được học; retriever đóng băng. MVP deterministic, chưa cần noise z.
Sampling nhiều z là ablation sau này và phải tính mọi candidate vào ngân sách.

### 6.2 Từ logits đến input embeddings

\[
P=\operatorname{softmax}((A+G)/\tau),\quad
H_{onehot}=\operatorname{onehot}(\arg\max P),
\]
\[
\widetilde P=P+\operatorname{stopgrad}(H_{onehot}-P),\qquad
T_q=\widetilde PW_q.
\]

Forward lấy embedding token thật; backward dùng gradient của relaxation. Đây là
straight-through estimator có bias, không phải đạo hàm chính xác của argmax.
Mask special/unused/padding tokens trước softmax. Đặt temperature schedule trên
validation; tránh khởi tạo logits quá nhọn làm mất gradient.

Nếu query/key encoders dùng cùng vocabulary và token-ID mapping, poison side có
thể dùng T_k=P_tilde W_k với cùng lựa chọn token. Nếu khác tokenizer, không được
nhân cùng P với embedding matrix khác rồi gọi đó là cùng text; MVP phải từ chối
cấu hình này hoặc dùng protocol rời rạc riêng đã công bố.

Truyền embedding qua `inputs_embeds`, với attention mask, vị trí, truncation và
special tokens giống runtime. Đóng băng parameters bằng `requires_grad_(False)`
không đồng nghĩa bọc forward của triggered inputs trong `no_grad()`.

### 6.3 Xuất trigger và chọn checkpoint

Xuất `argmax(A)` không có Gumbel noise, decode thành text rồi tokenize lại cả
query+trigger và poison record. Chấm lại theo runtime thật, kể cả thay đổi độ dài
WordPiece/BPE, khoảng trắng hoặc truncation. Reject candidate vi phạm token budget.
Lưu cả optimized IDs, text và re-encoded IDs để đo round-trip mismatch.

Checkpoint/candidate selection chỉ dùng validation hoặc adaptation support được
cho phép. Báo cáo riêng one-shot deterministic và best-of-S; best-of-S không được
trình bày như một lần forward duy nhất.

## 7. Objective và ranh giới gradient

### 7.1 AgentPoison-compatible geometry

Với u_i là embedding query gắn trigger, mu_j là benign reference centers:

\[
\mathcal L_{uni}=-\frac{1}{NK}\sum_{i,j}\|u_i-\mu_j\|_2,
\qquad
\mathcal L_{cpt}=\frac1N\sum_i\|u_i-\bar u\|_2.
\]

Baseline minimize L_AP=L_uni+0.1 L_cpt. Tính compactness trên nhiều query cùng
episode; không trộn các episode thành một centroid chung. Loss batch một query
có compactness bằng 0 nên không dùng để chứng minh cơ chế này.

Giữ raw embedding geometry cho bản AP-compatible. Loss khoảng cách có thể cải
thiện nhờ thay đổi norm mà không tăng retrieval; log norm và actual ranking để
phát hiện. Không đồng nhất loss AP giảm với agent attack thành công.

### 7.2 Retrieval margin bám đúng sự kiện đo

Với mục tiêu **ít nhất một poison xuất hiện trong top-K**, đặt s theo đúng scorer
runtime, a_i là điểm clean cao thứ K và b_i là điểm poison cao nhất:

\[
\mathcal L_{ret}=\frac1N\sum_i[\delta+a_i-b_i]_+.
\]

Điều kiện b_i>a_i đưa ít nhất một poison vào top-K khi xét xếp hạng chính xác,
đủ K clean records và không có tie. ANN index phải được kiểm tra thực nghiệm;
ranking surrogate không bảo đảm ANN sẽ trả đúng kết quả.

Nếu muốn K poison chiếm toàn bộ top-K, cần loss khác: điểm poison cao thứ K phải
vượt clean tốt nhất, và B>=K. Hàm `compute_retrieval_margin_loss` hiện tại dùng
định nghĩa mạnh hơn này, normalize cosine nội bộ; **không tái sử dụng như thể đó
là loss cho ít nhất một poison hoặc mặc định giống scorer triển khai**.

\[
\mathcal L_{train}=\mathcal L_{uni}+\lambda_c\mathcal L_{cpt}
                  +\lambda_r\mathcal L_{ret}.
\]

Bắt đầu lambda_r=0 để kiểm tra tương thích AP, sau đó ablate margin. Khi refresh
poison, gradient có thể qua cả query và poison embeddings; fixed-poison phải giữ
poison embeddings cố định. Việc chọn hard negatives từ index là bước rời rạc;
được phép dùng gradient qua scores của tập đã chọn và refresh theo lịch ghi log.

### 7.3 Coherence, behavior và optional surrogates

GPT-2 và Llama scorers hiện tại nhận text và dùng `no_grad()`. Không cộng các giá
trị đó vào loss rồi tuyên bố generator nhận gradient từ chúng. MVP dùng chúng để
lọc/chấm candidate trên validation, đo PPL cả query+trigger với cùng định nghĩa
cho mọi baseline; target probability chỉ là proxy, ASR phải chạy agent thật.

Nếu MVP bị hạn chế bởi fluency: thử distill một coherence surrogate từ PPL labels
của train-only candidates, rồi kiểm tra sai số trên held-out text. Nếu thêm term
L_gap, phải định nghĩa rõ so sánh representation trước/sau round-trip và xác định
nhánh teacher đã detach; không giả định decode/re-tokenize có gradient. Hai nhánh
này là ablation riêng, chưa thuộc objective chính và chưa bảo đảm cải thiện.

## 8. Training và deployment protocol

```text
PREPARE
  Lock outer splits, retriever/tokenizer/scorer versions, poison templates, budgets.
  Cache clean vectors and reference centers per snapshot.

TRAIN
  For each episode from train:
    Build context from benign memory + clean support queries.
    Generate A; ST-Gumbel gives differentiable trigger embeddings.
    Encode Q_opt with trigger; encode poison templates if refresh is allowed.
    Compute geometry + optional retrieval margin.
    Backprop only into set encoders and generator; optimizer.step().
  Periodically export hard text and evaluate on validation episodes.
  Save checkpoint selected by a predeclared discrete validation metric.

TEST ZERO-SHOT
  Read new snapshot + support; freeze generator and all models.
  Generate one hard trigger; round-trip through the actual tokenizer.
  Materialize poison records once if protocol allows; freeze them before Q_eval.
  Run retrieval and agent evaluation; record total online costs.

TEST FEW-STEP (separate result)
  Initialize episode-local logits from generator output.
  Adapt only those logits for a fixed budget on Q_opt; do not carry test updates
  into the next independent episode. Select using allowed support only.
```

Do not generate poison records from the current evaluation query. Poison source
templates được chọn trước từ train/support theo manifest và giữ cùng giữa methods.
Không cache triggered embeddings qua các optimizer steps vì sẽ mất graph hoặc
dùng embedding của trigger cũ. Cache key sạch theo corpus/model/scorer hashes.

## 9. Baselines và thí nghiệm phân biệt cơ chế

| ID | Method | Vai trò |
|---|---|---|
| B0 | Fixed/random allowed trigger, clean/no-trigger controls | Mốc behavior và false activation |
| B1 | AgentPoison HotFlip per episode | Baseline discrete optimization cùng quyền |
| B2 | Direct logits + ST-Gumbel/Adam per episode | Tách đóng góp optimizer khỏi conditioning |
| B3 | Một universal logits matrix train qua mọi train episode | Kiểm tra trigger phổ quát đã đủ chưa |
| B4 | Generator vô điều kiện, capacity tương đương | Kiểm tra network chỉ ghi nhớ một lời giải |
| B5 | Query-only generator | Tách thông tin memory khỏi query adaptation |
| B6 | Memory-only generator | Tách đóng góp query distribution |
| B7 | Nearest-context trigger bank từ train | Kiểm tra lookup có đủ thay learning không |
| M1 | Memory+query generator, zero-shot | Method chính |
| M2 | M1 + ngân sách adaptation cố định | Chất lượng sau vài bước, tính đủ chi phí |

Tách nhãn “reproduction” và “adaptation”: B2 áp dụng relaxation vào retrieval loss,
không phải reproduction nguyên kết quả GBDA/Soft-GCG. Nếu B1 phải sửa lỗi để dùng
được, pin patch/commit, ghi khác biệt và chạy lại mọi đối chứng bị ảnh hưởng.

Ablations ưu tiên:

1. Full context vs shuffled memory context: giữ query không đổi, trao đổi memory
   summaries giữa episode cùng kích thước; metric có xấu đi không?
2. Giữ memory không đổi, đổi query distribution; làm phép đối xứng để phát hiện
   generator chỉ dùng một nhánh conditioning.
3. Permute document order/GMM component order: output hoặc chất lượng phải ổn định
   trong sai số số học khi pooling thực sự permutation-invariant.
4. GMM summary vs sampled-key set pooling, cùng ngân sách encoder/summary.
5. AP-only vs AP+margin vs margin-only. Set pooling không tự làm objective
   cluster-free; margin-only là nhánh bỏ centers thực sự trong bản này.
6. Số train episodes, support size, trigger length, poison budget; luôn tính tổng.
7. Hard-forward ST vs soft-forward relaxation, deterministic vs best-of-S.

## 10. Memory drift và transfer

Mỗi test trajectory bắt đầu D_0 rồi tạo snapshot với growth 25%, 50%, 100%; bổ
sung ablation thay domain mixture, xóa records, hoặc thêm hard benign distractors.
Các mức là thiết kế đề xuất; khóa trước khi xem kết quả test. Không chọn documents
drift dựa trên việc chúng làm một method cụ thể thất bại.

So sánh trigger cũ, generate mới, warm-start optimizer và optimize lại từ đầu.
Mọi method nhận cùng snapshot/support và quyền refresh poison. Tính cả vector
encoding, summary update, GMM refit nếu có, write/index update và PPL/target calls.

Transfer chia cấp:

- T1: query và snapshot mới, cùng retriever — claim chính của MVP.
- T2: corpus/domain khác, cùng retriever — cần dữ liệu nguồn độc lập.
- T3: chuyển text trigger sang retriever khác mà không thích nghi — transfer attack.
- T4: generator thực sự conditioned trên retriever chưa thấy — nhánh nghiên cứu sau.

T4 không giải quyết bằng nối thêm retriever-ID. Different encoders có thể khác
chiều, vocabulary và hệ tọa độ dù cùng 768D. Cần thiết kế alignment bằng shared
anchor texts/relational features và vocabulary head phù hợp, khai báo calibration
access; chưa có thiết kế đó thì không claim zero-shot cross-retriever generator.

## 11. Metrics, chi phí và thống kê

### 11.1 Đánh giá bảo mật và utility

- Retrieval hit@K: tỷ lệ query có ít nhất một poison trong top-K; báo thêm poison
  occupancy@K và rank để không nhầm với full top-K takeover.
- Behavioral ASR: target behavior được quan sát trên toàn bộ triggered eval queries.
  Báo cả ASR có điều kiện trên retrieval hit; mẫu số luôn rõ.
- Clean task success, utility under attack, false activation trên query không trigger.
- Meaning preservation, relevance và fluency: evaluator độc lập và human audit
  một mẫu stratified. PPL không chứng minh nghĩa gốc được giữ.
- Round-trip mismatch, loss/ranking gap, invalid/empty trigger rate, norm và entropy.

Chạy đủ bốn điều kiện: clean-memory/trigger-off, clean-memory/trigger-on,
poison-memory/trigger-off, poison-memory/trigger-on. Dùng cùng base queries và
seeds để kiểm tra trigger có tự gây behavior mà không cần poison hay không.
Với sandbox QA có thể dùng target behavior sai lệch đã định nghĩa trước; kết quả
đó không thay thế bằng chứng external-effect nếu bài định vị tool-agent security.

### 11.2 Hiệu quả tính toán

Đo training GPU-hours, peak VRAM, preprocessing CPU time, số encoder
forward/backward, target/PPL calls, index writes, online p50/p95 latency và chi phí
của mọi candidate bị loại. Đồng bộ GPU/timing warmup theo cùng giao thức.

\[
C_{gen}(N)=C_{train}+C_{prep}^{gen}(N)+N C_{online}^{gen},
\]
\[
C_{search}(N)=C_{prep}^{search}(N)+N C_{online}^{search}.
\]

Nếu preprocessing chung đã triệt tiêu và online costs ổn định, điểm hòa vốn xấp xỉ
N*=C_train/(C_online_search-C_online_gen), chỉ có nghĩa khi mẫu số dương.
So sánh tại mức chất lượng tương đương; generator nhanh nhưng ASR thấp hơn nhiều
không được báo như speedup tương đương. Báo cả deployment-only và lifecycle cost.

Chạy hai chế độ fairness: cùng online compute budget và cùng tổng lifetime budget.
Giữ L, B, payload, support access, scorer và candidate-selection budget nhất quán.
Ít nhất 3 seeds ở pilot, hướng tới 5 khi full evaluation nếu đủ tài nguyên. Báo
paired differences, macro theo episode, micro theo query và nhóm kém nhất;
bootstrap theo episode family/trajectory để có 95% CI, tránh pseudo-replication.

## 12. Mapping sang code hiện tại và deliverables

Đọc local code ngày 2026-09-18; những mục dưới đây là kế hoạch, chưa tạo module.

| Thành phần | Tận dụng / thay đổi dự kiến |
|---|---|
| `algo/trigger_optimization.py` | Giữ baseline HotFlip; main fit GMM 5, `evaluate_property` dùng KMeans và NumPy nên không lấy nguyên hàm làm train loss |
| `src/triggers/losses.py` | Tái sử dụng uniqueness/compactness; thêm margin contract riêng theo runtime nếu cần |
| `src/triggers/clustering.py` | `fit_centers` hiện là full-covariance GMM trên raw vectors; cache theo snapshot và version |
| `src/triggers/scorers.py` | Giữ scorer text bên ngoài graph; không tuyên bố đã có differentiable PPL/target loss |
| `src/triggers/margin.py` | Tham khảo `_encode_triggered` qua `inputs_embeds`; tách helper nhận embeddings trainable thay vì chỉ IDs |
| `src/triggers/hierarchy/` | Tận dụng ý tưởng đánh giá group/universal nếu contract khớp; không coi hierarchy pilot là kết quả MCAT |
| `algo/memory_trigger/episodes.py` — mới | Manifest, splits, trajectory và cache metadata |
| `algo/memory_trigger/generator.py` — mới | Set encoders, position-conditioned generator, vocabulary mask |
| `algo/memory_trigger/relaxation.py` — mới | ST-Gumbel, export hard text, round-trip validation |
| `algo/memory_trigger/objectives.py` — mới | Loss contracts, scorer policy, fixed/refresh poison handling |
| `scripts/train_memory_trigger.py` — mới | Train/validation, checkpoint và resume RNG |
| `scripts/evaluate_memory_trigger.py` — mới | One-shot/few-step, drift, metrics và accounting |

Một run cần có `config.json`, `episodes.jsonl`, split/corpus/model/tokenizer hashes,
`checkpoint.pt`, `metrics.jsonl`, `triggers.jsonl`, `costs.json`, `report.md`.
Trigger artifact chứa episode/snapshot ID, decoded text, raw/re-encoded IDs,
poison record IDs, generation mode, seed và validation selection rule.

Kiểm chứng trước khi chạy GPU dài:

- Hard forward bằng embedding lookup tương ứng; gradient tới generator/set encoders
  hữu hạn và khác 0; retriever không bị cập nhật.
- ID-input và embedding-input cho cùng text có output khớp trong tolerance;
  test padding, truncation, special-token placement và batch nhiều độ dài.
- Margin trên fixture khớp exact top-K với ties được xử lý theo policy; cố ý có case
  one-poison-hit khác full-takeover để bắt nhầm định nghĩa.
- Permutation invariance, no split leakage, fixed-poison immutability, budget/cost
  accounting và checkpoint resume có thể tái lập.
- Chạy tiny local episode qua export → decode → retrieval → report trước full train.

## 13. Lộ trình theo milestone

Thời gian dưới đây là ước lượng cho lập kế hoạch, không phải lịch cam kết. Chốt
GPU/VRAM và số giờ sau profiling M0; baseline và generator dùng cùng phần cứng.

| Mốc | Khối lượng dự kiến | Deliverable | Điều kiện qua mốc |
|---|---|---|---|
| M0 | 2–3 ngày: audit runtime/splits/scorer và profile | Experiment manifest + baseline timing | Xác định được exact text path, normalization, quyền poison và dữ liệu độc lập |
| M1 | 3–5 ngày: direct-logit ST baseline | B1/B2 discrete pilot | Gradient đúng, hard-text pipeline đúng; số liệu có seed và chi phí |
| M2 | Khoảng 1 tuần: episode generator | B3–B7/M1 trên held-out episodes | Chạy conditioning/shuffle controls, không chỉ train-loss curves |
| M3 | Khoảng 1 tuần: drift + few-step | Fixed/refresh reports và cost curves | So sánh cùng write/compute budgets, không dùng test để chọn trigger |
| M4 | 1–2 tuần: agent behavior + domain transfer | Behavioral ASR, utility, CI và failure cases | Retrieval gain còn tồn tại ở agent level và text thực |
| M5 | Sau go decision: full evaluation và paper | Ablations, artifacts, related-work audit | Claims khớp evidence; tái lập được từ manifests |

Ưu tiên chạy QA sandbox hiện có trước. Không mở rộng đồng thời nhiều agent/retriever
khi chưa chứng minh được conditioning ở một thiết lập có kiểm soát.

## 14. Go / pivot / stop criteria

Các ngưỡng sau là **mục tiêu pilot đề xuất**, cần khóa trên validation trước test:

- Go về chất lượng: zero-shot M1 nằm trong 5 điểm phần trăm behavioral ASR của
  per-episode baseline mạnh nhất và giảm ít nhất 3× online cost tại cấu hình đó.
  Nếu chỉ đạt retrieval, chỉ được go sang bước đo behavior, chưa claim security.
- Go về cơ chế: full context tốt hơn query-only/unconditional và shuffled-context
  trên paired held-out episodes; CI và effect size phải đủ để bác bỏ gain do seed.
- Go về utility: clean task success suy giảm không quá 2 điểm phần trăm so với
  đối chứng phù hợp; báo cả false activation, không chỉ average accuracy.
- Go về amortization: tồn tại điểm hòa vốn trong số episode dự kiến sử dụng và
  lợi ích vẫn còn khi tính train, summary/index update và selection calls.
- Pivot nếu trigger bank đạt tương đương: nghiên cứu cấu trúc episode/trigger
  families hoặc lookup+refinement thay vì claim generator cần thiết.
- Pivot nếu direct logits tốt nhưng conditioning không có ích: giữ kết quả như
  optimizer study; không nâng claim thành adaptive memory generator.
- Stop claim nếu gain mất sau round-trip, chỉ có trên episode gần trùng train,
  chỉ tăng norm/loss, hoặc cần poison-write budget lớn hơn baseline.

Không đổi ngưỡng sau khi thấy test để tạo kết luận thành công. Nếu pilot thiếu
power, báo chưa kết luận và tăng số episode độc lập trước khi tăng số query lặp.

## 15. Gói đóng góp để hướng tới Q1 / A*

1. Method: generator conditioned trên memory/query, cơ chế gradient và protocol
   discretization rõ; chứng minh conditioning có giá trị ngoài optimizer/bank.
2. Evaluation protocol: episodic memory drift với giới hạn poison writes và
   support access, tách zero-shot/few-step, fixed/refresh và train/test families.
3. Evidence: behavioral effect, utility, uncertainty, lifecycle cost và failure
   analysis; không chỉ hình PCA hoặc retrieval loss.

Với hướng ML, trọng tâm là amortized optimization và generalization có ablation
chặt. Với hướng security, cần chứng minh threat model thực tế, hành vi agent và
kiểm tra dưới các defense phù hợp. Liên kết ADAPT bằng generator làm đối thủ
cho adversarial training là follow-up; test defender phải có attack families và
generator checkpoints không dùng khi train để tránh co-adaptation.

Không cam kết tier hoặc acceptance từ ý tưởng. Venue/ranking Q1/A* cần xác minh
theo năm submission và chọn sau khi biết contribution nào có bằng chứng mạnh.
Phiên bản đầu chỉ được gọi là nghiên cứu khả thi khi hoàn thành M0–M2; tài liệu
này chưa phải kết quả thực nghiệm hay xác nhận novelty.
