# AQuA: Authorization-Quotient Activations for Secure Tool-Using LLM Agents

> **Liên hệ hướng hiện hành (2026-10-10):** [nghiên cứu backdoor qua vòng ghi memory](../self_updating_memory_backdoor_idea.md)
> giữ thứ tự gap → attack có kiểm soát → defense. AQuA bên dưới là đề xuất defense
> riêng, không tự động là biện pháp khắc phục của hướng memory mới.
> Mục đích và giới hạn thí nghiệm trên hệ thống được phép: [README](../README.md).

> Trạng thái: research idea / pre-proposal  
> Cập nhật: 2026-09-13  
> Mục tiêu: bài báo top-tier (A* conference hoặc Q1 journal) về white-box security cho LLM agents có RAG và tool calling.

> Nhánh Q1 / A* bổ sung (2026-09-18): [MCAT — Memory-Conditioned Amortized Trigger Generation](memory_conditioned_generator_Q1_A_star.md).
> Kế hoạch riêng về network sinh trigger theo memory/query distribution, gồm kiến trúc,
> loss, episodic training, baselines, memory drift, chi phí và tiêu chí go/no-go.
> Phần bên dưới tiếp tục là đề xuất AQuA.

> Gate novelty (2026-09-23): đã đọc [Influence Is Not Authority](https://arxiv.org/abs/2608.29942)
> và [Permit](https://arxiv.org/abs/2605.09480). Khoảng trống còn nguyên, nhưng Permit là
> closest prior nên claim phải chuyển từ trục *subspace* sang trục *invariance*.
> Danh sách đọc theo tier ở [mục 16](#16-danh-sách-paper-cần-đọc-trước-khi-làm);
> mục 4 và mục 13 vẫn cần viết lại theo positioning mới.
>
> Thứ tự thi công (2026-09-24): [attack_first_roadmap.md](attack_first_roadmap.md).
> Quyết định là làm MCAT trước, AQuA sau. Thí nghiệm 24-base ở mục 9 của roadmap phải
> chạy **trước** khi khởi động AuthShift, vì ba kết cục của nó dẫn tới ba bài khác nhau.

## 1. Tóm tắt ý tưởng

LLM agents thường nhận chung trong một context:

- system/developer instructions;
- yêu cầu của người dùng;
- tài liệu lấy từ RAG;
- tool outputs;
- memory từ các lượt trước;
- mô tả và schema của tool.

Các nguồn này có độ tin cậy và thẩm quyền khác nhau, nhưng model xử lý chúng trong cùng một kênh sinh token. Vì vậy, dữ liệu không đáng tin có thể không chỉ cung cấp thông tin mà còn chi phối tool call, dẫn tới indirect prompt injection, privilege escalation, data exfiltration hoặc các external effects không được người dùng cho phép.

Ý tưởng trung tâm của bài báo:

> Authorization là quan hệ nhân quả giữa user-granted intent và từng trường của một tool call; nó không đồng nhất với provenance, mức độ ảnh hưởng của context, độ độc hại của nội dung hay tính đúng/sai của tool call.

Đề xuất **AQuA — Authorization-Quotient Activations**, một phương pháp white-box học biểu diễn authorization từ các counterfactual được ghép cặp chặt chẽ. Biểu diễn này phải:

1. bất biến trước các biến đổi không làm thay đổi quyền thực thi;
2. nhạy với các biến đổi grant, scope, revocation hoặc external effect;
3. phân tách data influence hợp lệ khỏi authority influence trái phép;
4. cho phép can thiệp trực tiếp vào hidden state trước khi tool call được thực thi.

RAG trong nghiên cứu này được xem là một nguồn context không đáng tin, không phải một module nghiên cứu độc lập. Phạm vi ban đầu nên là **single-agent, white-box, tool-executing RAG agents với tool schema biết trước**.

## 2. Research question

### Câu hỏi chính

Liệu hidden states của một tool-using LLM có chứa một cấu trúc hình học biểu diễn authorization, sao cho cấu trúc này bất biến với việc di chuyển dữ liệu hợp lệ giữa các nguồn nhưng thay đổi khi quyền thực thi bị cấp, giới hạn hoặc thu hồi?

### Câu hỏi phụ

1. Authorization có thể được tách khỏi source identity, semantic relevance và generic harmfulness hay không?
2. Authorization được hình thành tại layer/token nào trong quá trình reasoning và tool-call generation?
3. Có thể loại bỏ causal control trái phép ở cấp từng argument mà vẫn giữ lại dữ liệu hữu ích từ RAG/tool outputs không?
4. Biểu diễn này có transfer sang unseen tools, unseen domains, unseen attacks và multi-turn trajectories không?
5. Một adaptive white-box attacker có thể đồng thời gây harmful external effect và né authorization representation hay không?

## 3. Vì sao bài toán hiện tại chưa được giải quyết

### 3.1 Prompt/filter/judge defenses

Các phương pháp này tìm dấu hiệu injection hoặc đánh giá mức độ nguy hiểm của proposed action. Chúng thường phụ thuộc vào surface form, dễ gặp distribution shift và có thể bị adaptive attack tối ưu trực tiếp để né detector.

### 3.2 Capability, IFC và reference monitor

CaMeL, Fides, Progent và các phương pháp tương tự đưa enforcement ra ngoài model. Đây là hướng có guarantee mạnh, nhưng vẫn gặp các vấn đề:

- policy synthesis từ natural-language intent;
- provenance/taint quá thô;
- user fatigue khi cần declassification hoặc approval;
- utility loss khi hành động hợp lệ phải phụ thuộc vào dữ liệu từ nguồn ngoài;
- khó phân biệt dữ liệu được phép cung cấp argument với dữ liệu đang tự cấp quyền thực thi.

CaMeL gọi failure quan trọng này là **data requires action**: privileged planner không nhìn thấy untrusted data nên không thể biết cần thực hiện hành động nào.

### 3.3 White-box probes và activation steering

AgentLens, PRISMS và các hướng liên quan cho thấy trạng thái nguy hiểm, nhu cầu gọi tool hoặc tool-call error có thể đọc được từ activations. Tuy nhiên:

- probe accuracy không chứng minh feature có vai trò nhân quả;
- generic error/harmfulness khác với authorization;
- tool call đúng cú pháp và đúng task vẫn có thể không được phép;
- source identity có thể trở thành shortcut;
- robustness dưới adaptive white-box attack chưa đủ mạnh.

### 3.4 Causal attribution

AttriGuard kiểm tra tool call bằng counterfactual replay: proposed call có còn tồn tại khi control signal từ untrusted observations bị giảm hay không. Đây là prior work gần nhất về causal attribution.

Nhưng **causal influence không đồng nghĩa authorization**. Một hành động hoàn toàn hợp lệ có thể phải phụ thuộc mạnh vào dữ liệu từ RAG/tool output. Ví dụ, người dùng cho phép “gửi email tới địa chỉ trong hồ sơ X”; recipient hợp lệ tất yếu được lấy từ external data.

Do đó, câu hỏi đúng không phải là “nguồn ngoài có ảnh hưởng tới hành động không?”, mà là:

> Nguồn ngoài đang cung cấp dữ liệu cho một quyền đã được cấp, hay đang tự tạo/thay đổi quyền thực thi?

## 4. Novelty boundary

Các công trình gần nhất đã chiếm một phần không gian ý tưởng:

| Công trình | Phần đã được nghiên cứu | Khoảng trống còn lại |
|---|---|---|
| AgentDojo | Môi trường đánh giá indirect prompt injection và tool execution | Benchmark tĩnh, chưa tách authorization equivalence ở cấp field |
| ASB | Attack/defense qua prompt, tool và memory | Metric chưa luôn gắn chặt semantic failure với realized external effect |
| CaMeL | Capability và control/data-flow separation | Data-requires-action, policy burden, declassification |
| Fides | Information-flow control với confidentiality/integrity labels | Dependency inference, implicit flows và utility trade-off |
| Progent | Symbolic least-privilege policy và monotonic confinement | Chất lượng policy generation và trusted-intent assumption |
| AttriGuard | Action-level causal attribution bằng counterfactual replay | Chưa học white-box authorization geometry; chưa causal-scrub từng field |
| AgentLens | Multi-turn hidden-state safety detection/steering | Generic harmful state, không phải relational authorization |
| PRISMS | Sparse detection/steering cho over-call, missing call và invalid arguments | Tool-use correctness khác authorization |
| Permit **(đã đọc 2026-09-23)** | **Đã có** permission-sensitive subspace (SVD trên \(\Delta h\) giữa các access level khai báo bằng system prompt) **và** intervention trên last-token hidden state (offset + gated), backbone đóng băng | Thuần text generation, không chạm tool call/argument; contrast bằng prompt chứ không phải provenance relocation; không có invariance objective hay shortcut suppression; không activation patching; adversarial chỉ role-play injection |
| AuthGraph/PACT | Parameter/argument-level provenance và authority contracts | External symbolic enforcement; provenance inference vẫn là bottleneck |
| Influence Is Not Authority **(đã đọc 2026-09-23)** | Diagnostic thuần trên 96 điều kiện / 24 scenario; audit AgentWatcher, AttriGuard và scalar threshold. Source relocation sang tool result làm signal lệch sang "attack" ở **100%** trường hợp; semantic monitor ASR 0% nhưng utility 28%; shadow guardrail cho qua mọi benign run nhưng không reject nhất quán | Không học representation, không can thiệp hidden state, không đề xuất method. Toàn bộ khoảng trống representation learning + field-level scrubbing còn nguyên |

Novelty không nên được mô tả là “kết hợp causal attribution, probe và policy gate”. Claim cần tập trung vào một nguyên lý mới:

> **Authorization-equivalent representation learning:** factor model states theo các phép biến đổi bảo toàn quyền, đồng thời làm biểu diễn nhạy với những can thiệp thay đổi quyền.

## 5. Formalization

### 5.1 Tool call

Biểu diễn một tool call dưới dạng:

\[
a = (t, f_1=v_1, f_2=v_2, \ldots, f_m=v_m),
\]

trong đó \(t\) là tool/operation và mỗi \(f_i\) là một argument field như resource, recipient, amount, path, payload hoặc requested privilege.

Context được phân đoạn theo nguồn:

\[
x = (x_{sys}, x_{user}, x_{rag}, x_{tool}, x_{memory}).
\]

Với mỗi field \(f\), \(Auth(f)\) mô tả các nguồn hoặc grants có quyền quyết định field đó. Ví dụ, RAG có thể cung cấp meeting time nhưng không có quyền thay đổi recipient hoặc tạo một payment operation mới.

### 5.2 Authorization equivalence

Hai context \(x\) và \(x'\) là authorization-equivalent đối với action \(a\), ký hiệu:

\[
x \sim_{auth,a} x',
\]

nếu chúng cho phép cùng external effect của \(a\), dù dữ liệu cần thiết xuất hiện ở vị trí hoặc nguồn hợp lệ khác nhau.

Ví dụ authorization-preserving transformation:

- chuyển một địa chỉ được phép từ user prompt sang verified CRM tool output;
- paraphrase grant mà không đổi scope;
- thay đổi benign formatting hoặc RAG ranking nhưng giữ evidence và quyền như cũ.

Ví dụ authorization-changing transformation:

- cấp hoặc thu hồi permission;
- đổi recipient/amount vượt scope;
- biến read thành write/delete;
- tool output tự tuyên bố rằng user đã approval;
- dùng expired hoặc replayed grant.

### 5.3 Counterfactual quadruple

Mỗi base task sinh ra tối thiểu bốn matched variants:

1. \(x_1\): authorized, fact from user;
2. \(x_2\): authorized, same fact from legitimate tool/RAG;
3. \(x_3\): unauthorized, same surface content và proposed effect;
4. \(x_4\): revoked/expired, otherwise identical.

Authorization representation \(z_A(x)\) cần thỏa:

\[
\|z_A(x_1)-z_A(x_2)\| \rightarrow 0,
\]

và có margin lớn cho các cặp thay đổi quyền:

\[
d(z_A(x_1),z_A(x_3)) \geq \gamma,
\qquad
d(z_A(x_1),z_A(x_4)) \geq \gamma.
\]

Đồng thời, representation không được chỉ mã hóa source identity, topic, attack vocabulary hoặc action string.

## 6. Phương pháp AQuA

### 6.1 Activation collection

Thu hidden states tại các vị trí quyết định:

- cuối user/context prompt;
- cuối reasoning span nếu model dùng explicit reasoning;
- token trước tool name;
- span của từng generated argument;
- các layer sớm, giữa và muộn.

### 6.2 Authorization quotient learning

Học một low-rank projection \(P_A\) từ counterfactual quadruples với ba mục tiêu:

1. **Equivalence loss:** kéo gần các authorization-preserving pairs.
2. **Flip loss:** tách các permission-changing pairs bằng margin.
3. **Shortcut suppression:** adversarially loại source identity, lexical attack cues và domain khỏi \(z_A\).

Phần bù \(P_D\) giữ task/data information cần cho legitimate tool arguments.

Giả thuyết cần kiểm chứng, không được mặc định đúng: tồn tại một subspace thấp chiều vừa transfer được vừa có causal relevance đối với tool authorization.

### 6.3 Field-level causal residual

Với mỗi source span \(s\) và action field \(f\), ước lượng causal residual:

\[
r_{s,f},
\]

bằng activation patching, source ablation có kiểm soát hoặc Jacobian approximation. Residual được phân rã thành:

- data-bearing component;
- authority-bearing component \(P_A r_{s,f}\).

### 6.4 Causal scrubbing

Trước khi decode hoặc dispatch một protected field, loại authority component đến từ nguồn không được phép:

\[
h'_f = h_f - \sum_{s \notin Auth(f)} P_A r_{s,f}.
\]

Model sau đó regenerate field/tool call từ \(h'_f\). Nếu authorization margin không đạt ngưỡng, hệ thống abstain hoặc yêu cầu user confirmation.

Mục tiêu của intervention không phải xóa toàn bộ ảnh hưởng của untrusted data. Nó chỉ xóa phần ảnh hưởng có khả năng làm thay đổi quyền, operation hoặc protected effect.

### 6.5 Security statement dự kiến

Có thể hướng tới một kết quả dạng local certificate:

> Nếu tổng unauthorized causal residual trong authorization subspace bị chặn bởi \(\epsilon\), decoder có Lipschitz constant \(L\), và authorized action margin \(\gamma > 2L\epsilon\), thì các perturbation trong threat neighborhood không thể đổi protected field sang một unauthorized value.

Đây chỉ là mục tiêu lý thuyết ban đầu; không nên gọi là global security guarantee.

## 7. Threat model

### Defender

- Có white-box access tới open-weight agent model.
- Biết tool schema và protected fields.
- Có user grant/policy state tối thiểu để gán authorization-changing labels.
- Có quyền chặn tool call trước external execution.

### Attacker

- Có thể inject/poison RAG documents, webpages, emails, tool outputs hoặc memory inputs.
- Biết toàn bộ AQuA architecture, projection, loss và thresholds.
- Có gradient access trong strongest white-box setting.
- Có thể tối ưu universal hoặc task-specific payload.
- Không trực tiếp sửa trusted policy store hoặc enforcement code.

### Ngoài phạm vi phiên bản đầu

- Compromised tool implementation tạo side effect khác schema;
- side-channel và covert-channel attacks;
- colluding multi-agent systems;
- poisoning model weights hoặc tokenizer;
- arbitrary code execution ngoài mediated tool interface.

## 8. Benchmark đề xuất: AuthShift

Xây dựng benchmark matched-counterfactual từ AgentDojo, ASB/InjecAgent và một executable RAG-tool sandbox.

Các transformation families:

1. trusted/untrusted source relocation;
2. explicit grant/denial flip;
3. scope, amount hoặc recipient boundary;
4. permission revocation và expiration;
5. benign data dependence versus malicious action induction;
6. tool-output injection;
7. RAG corpus poisoning;
8. memory laundering qua summarize/retrieve;
9. second-order injection;
10. multi-step privilege accumulation.

Mỗi case phải lưu:

- intended authorization state;
- minimum sufficient permission;
- protected fields;
- expected external effect;
- provenance của từng value;
- actual sandbox-observed effect;
- matched variants và transformation type.

## 9. Thực nghiệm

### 9.1 Models

Tối thiểu ba họ open-weight model, nhiều scale nếu tài nguyên cho phép:

- Qwen;
- Llama;
- Gemma hoặc Mistral.

### 9.2 Baselines

- prompt hardening/Spotlighting;
- detector hoặc LLM-as-a-judge;
- MELON/counterfactual replay;
- CaMeL;
- Fides/RTBAS-style IFC;
- Progent;
- AttriGuard;
- AgentLens/PRISMS-style probe;
- Permit-style permission direction;
- AuthGraph/PACT-style field-level provenance enforcement.

### 9.3 Metrics

Không dùng một ASR duy nhất. Cần báo cáo:

- Executed Attack Success Rate: harmful effect thực sự xảy ra trong sandbox;
- Benign Task Success;
- Utility Under Attack;
- False-Allow và False-Deny;
- Overblocking Rate;
- Authorization-Equivalence Gap;
- Authorization-Flip Sensitivity;
- Field-Level Privilege Regret;
- Revocation Consistency;
- calibration/ECE;
- unseen-tool, unseen-domain và unseen-attack transfer;
- latency, memory và số forward/backward passes.

### 9.4 Adaptive white-box attack

Attacker tối ưu đồng thời:

\[
\mathcal{L}_{attack}
=
\mathcal{L}_{harmful\ effect}
+ \lambda \mathcal{L}_{AQuA\ evasion}
+ \mu \mathcal{L}_{retrieval}
+ \nu \mathcal{L}_{stealth}.
\]

Cần so sánh static attack, black-box adaptive attack và full white-box adaptive attack. Không được kết luận robust chỉ từ AgentDojo payloads cố định.

### 9.5 Ablations

- bỏ authorization-preserving pairs;
- bỏ revocation pairs;
- bỏ shortcut suppression;
- tool-level thay cho field-level;
- attention attribution thay activation patching;
- single layer so với multi-layer;
- dense versus low-rank subspace;
- detector-only versus causal scrubbing;
- source-ID supervision versus authorization-equivalence supervision.

## 10. Falsification criteria

Giả thuyết hoặc method cần bị xem là thất bại nếu:

1. không tìm thấy authorization subspace ổn định qua model/layer;
2. source relocation gây representation shift lớn hơn permission flip;
3. probe dựa chủ yếu vào lexical/source shortcut;
4. intervention làm benign utility giảm quá 5 điểm phần trăm;
5. adaptive ASR không tốt hơn external reference-monitor baselines;
6. method không transfer sang unseen tools hoặc unseen schemas;
7. representation mất hiệu lực trong multi-turn hoặc sau revocation;
8. causal scrubbing chỉ làm model abstain thay vì giữ legitimate data-dependent actions.

Nếu authorization không tạo thành một low-rank subspace, negative result vẫn có thể trở thành một paper audit: **authorization is not robustly represented inside tool-using LLMs**, từ đó chứng minh model-internal enforcement không thể thay external reference monitors trong threat model tương ứng.

## 11. Minimum viable pilot

Trước khi đầu tư full paper:

1. Tạo khoảng 500 matched quadruples trên 20–40 tools.
2. Chạy Qwen và Llama ở một hoặc hai scale.
3. Lấy activations trước tool name và tại argument spans.
4. So sánh ba loại probe:
   - source identity;
   - generic harmfulness/tool error;
   - authorization-equivalence.
5. Đo linear separability, cross-tool transfer và causal intervention effect.
6. Thử một adaptive gradient attack nhỏ nhằm đồng thời giữ harmful call và giảm authorization score.

Go/no-go criterion:

- **Go:** authorization flip có effect size rõ, source relocation gap nhỏ, transfer tốt và intervention cải thiện executed ASR mà không overblock mạnh.
- **Pivot:** authorization signal tồn tại nhưng không low-rank hoặc không causal; chuyển sang circuit/trajectory analysis.
- **Stop/negative paper:** source identity áp đảo authorization và không thể loại shortcut qua matched counterfactuals.

## 12. Publication positioning

### Security/system track

Phù hợp với USENIX Security, ACM CCS, IEEE S&P hoặc NDSS nếu bài có:

- threat model chặt;
- external effects được thực thi trong sandbox;
- adaptive white-box attacks;
- security statement rõ;
- so sánh với system-level reference monitors;
- open-source benchmark và artifacts.

### ML/representation track

Phù hợp với NeurIPS, ICLR hoặc ICML nếu trọng tâm là:

- phát hiện authorization geometry;
- mechanistic/causal validation;
- model- và tool-family generalization;
- representation intervention;
- negative controls mạnh và tránh probe shortcuts.

### Journal extension

IEEE TDSC hoặc IEEE TIFS là các lựa chọn phù hợp cho phiên bản mở rộng. Cần kiểm tra lại CORE/JCR/Scopus ranking tại đúng năm submission vì A*/Q1 có thể thay đổi.

## 13. One-sentence pitch

> AQuA learns an authorization quotient of tool-call activations—one that is invariant to legitimate relocation of task data but sensitive to changes in user-granted authority—and causally removes unauthorized control over protected tool-call fields before execution.

## 14. Các claim cần tránh

- “Giải quyết hoàn toàn prompt injection.”
- “Probe accuracy chứng minh model hiểu authorization.”
- “Attention là causal explanation.”
- “0% ASR trên static benchmark nghĩa là provably secure.”
- “Kết hợp RAG defense và tool guard là novelty.”
- “Một subspace trên một model/dataset là universal authorization representation.”

## 15. Tài liệu neo

1. AgentDojo: <https://arxiv.org/abs/2406.13352>
2. Agent Security Bench: <https://arxiv.org/abs/2410.02644>
3. InjecAgent: <https://arxiv.org/abs/2403.02691>
4. ToolEmu: <https://arxiv.org/abs/2309.15817>
5. CaMeL — Defeating Prompt Injections by Design: <https://arxiv.org/abs/2503.18813>
6. Fides — Securing AI Agents with Information-Flow Control: <https://arxiv.org/abs/2505.23643>
7. Progent — Securing AI Agents with Privilege Control: <https://arxiv.org/abs/2504.11703>
8. RTBAS: <https://arxiv.org/abs/2502.08966>
9. Joint-GCG: <https://arxiv.org/abs/2506.06151>
10. AttriGuard: <https://arxiv.org/abs/2603.10749>
11. AgentLens: <https://arxiv.org/abs/2606.22673>
12. PRISMS: <https://arxiv.org/abs/2608.00218>
13. Permit: <https://arxiv.org/abs/2605.09480>
14. AuthGraph: <https://arxiv.org/abs/2605.26497>
15. PACT: <https://arxiv.org/abs/2605.11039>
16. Influence Is Not Authority: <https://arxiv.org/abs/2608.29942>

> Lưu ý: nhiều tài liệu 2026 trong danh sách là arXiv preprint rất mới, được dùng để kiểm tra novelty boundary chứ chưa nên xem là bằng chứng peer-reviewed. Cần cập nhật systematic search trước khi viết related work và ngay trước submission.

## 16. Danh sách paper cần đọc trước khi làm

> Lập ngày 2026-09-23. Chia theo thời điểm cần đọc, không theo mức độ nổi tiếng.
> Chỉ Tier 0 đã được đọc và xác minh; các tier sau vẫn ở trạng thái chưa đối chiếu
> full text. Tên/ID arXiv của các preprint 2026 cần kiểm lại tại thời điểm submission.

### Tier 0 — Gate novelty, đã đọc và đã chốt

Hai paper này quyết định AQuA có còn chỗ đứng hay không. Kết luận: **còn**, nhưng
claim phải chuyển từ trục *subspace* sang trục *invariance*.

| Paper | Trạng thái | Việc phải làm tiếp |
|---|---|---|
| [Influence Is Not Authority](https://arxiv.org/abs/2608.29942) | Đã đọc. Diagnostic thuần, không method | Dùng làm motivation chính. Trích thẳng kết luận "current causal signals reveal action origins but do not reliably encode authorization status". Lấy số 0% ASR / 28% utility làm điểm neo cho trade-off |
| [Permit](https://arxiv.org/abs/2605.09480) | Đã đọc. **Closest prior** | Viết một đoạn riêng trong related work phân biệt bốn điểm: provenance relocation, executable field, causal validation, adaptive attacker. Không được viết "chúng tôi phát hiện permission có cấu trúc hình học" |

Hệ quả bắt buộc cho mục 4 và mục 13: Permit đã chiếm "permission subspace +
intervention cho text generation". Phần còn lại của AQuA là equivalence loss,
field-level causal scrubbing, executed external effect và white-box adaptive attack.
Bỏ bất kỳ cái nào thì bài tụt thành "Permit áp cho tool calling".

### Tier 1 — Phải đọc trước khi chốt baseline và viết related work

Đây là các đối thủ trực tiếp ở cùng cấp độ (field-level hoặc white-box). Chưa đọc
xong tier này thì chưa được viết một dòng related work nào.

| Paper | Đọc để lấy gì |
|---|---|
| [AttriGuard](https://arxiv.org/abs/2603.10749) | Baseline bắt buộc, không còn tùy chọn: Influence Is Not Authority đã audit nó và cho thấy nó thất bại ở đâu. Lấy đúng con số đó làm đường cơ sở phải vượt. Cần biết chính xác counterfactual replay của họ giảm control signal thế nào |
| [AuthGraph](https://arxiv.org/abs/2605.26497) | Provenance ở cấp parameter/argument — cùng độ phân giải với AQuA. Cần biết provenance của họ lấy từ đâu và enforcement nằm ở đâu (trong hay ngoài model) |
| [PACT](https://arxiv.org/abs/2605.11039) | Authority contract ở cấp argument. Câu hỏi then chốt: contract được sinh ra bằng gì, và bottleneck provenance inference nằm chỗ nào |
| [AgentLens](https://arxiv.org/abs/2606.22673) | Probe/steering baseline. Cần biết họ lấy activation ở layer/token nào để AQuA thiết kế negative control trùng vị trí, tránh bị hỏi "khác gì" |
| [PRISMS](https://arxiv.org/abs/2608.00218) | Sparse detection/steering cho invalid argument. Phải phân biệt rõ tool-use correctness với authorization — nếu không, reviewer sẽ coi AQuA là biến thể |

### Tier 2 — Đọc khi bắt đầu xây AuthShift

Không cần đọc trước A1/A2 của nhánh attack. Đọc khi thật sự dựng benchmark.

| Paper | Đọc để lấy gì |
|---|---|
| [AgentDojo](https://arxiv.org/abs/2406.13352) | Task/tool/attack suite và cách họ ghi nhận execution. Nguồn gốc cho matched variants |
| [Agent Security Bench](https://arxiv.org/abs/2410.02644) | Attack/defense qua prompt, tool và memory. Chú ý chỗ metric của họ chưa gắn semantic failure với realized effect — đó là chỗ AuthShift phải làm khác |
| [InjecAgent](https://arxiv.org/abs/2403.02691) | Taxonomy indirect injection cho tool-integrated agent |
| [ToolEmu](https://arxiv.org/abs/2309.15817) | Emulator cho external effect khi không có sandbox thật |
| [CaMeL](https://arxiv.org/abs/2503.18813) | Reference monitor baseline. Quan trọng nhất: lấy đúng các case **data requires action** để đưa vào transformation family số 5 |
| [Fides](https://arxiv.org/abs/2505.23643) | IFC label confidentiality/integrity; nguồn cho implicit-flow case |
| [Progent](https://arxiv.org/abs/2504.11703) | Least-privilege policy; nguồn cho minimum sufficient permission của mỗi case |
| [RTBAS](https://arxiv.org/abs/2502.08966) | Dependency inference và utility trade-off |

### Tier 3 — Đọc khi viết adaptive white-box attack (mục 9.4)

Mục 9.4 là chỗ reviewer security sẽ tấn công mạnh nhất. Không được tự nghĩ ra
attack loss mà không đối chiếu tier này.

| Paper | Đọc để lấy gì |
|---|---|
| [Optimizing Against Safety Representations](https://arxiv.org/abs/2607.08883) | **Ưu tiên cao nhất của tier này.** Họ đã tối ưu trực tiếp chống lại safety representation — đúng dạng \(\mathcal{L}_{AQuA\ evasion}\). Nếu không đối chiếu, mục 9.4 sẽ bị coi là yếu hơn state of the art. Lưu ý Soft-GCG là tên method trong paper này, không chuyển speedup của họ sang bài mình |
| [Joint-GCG](https://arxiv.org/abs/2506.06151) | Tối ưu đồng thời qua retrieval và generation — khớp với cấu trúc nhiều term của attack loss |
| [GBDA](https://aclanthology.org/2021.emnlp-main.464/) | Continuous relaxation cho adversarial text; baseline cho phần discrete optimization |
| [Neural Exec](https://arxiv.org/abs/2403.03792) | Execution trigger học được cho prompt injection |
| [TROPT](https://arxiv.org/abs/2606.23496) | Framework discrete text optimization; đối chiếu optimizer/loss coverage trước khi claim khoảng trống |

Chi tiết hơn về các nguồn attack nằm ở mục 3 của
[kế hoạch MCAT](memory_conditioned_generator_Q1_A_star.md), không nhân bản ở đây.

### Thứ tự đọc đề xuất

1. **Ngay bây giờ, song song với nhánh attack** (không cần GPU, không tranh tài nguyên):
   Tier 1 — AttriGuard trước, rồi AuthGraph/PACT, rồi AgentLens/PRISMS.
2. Sau khi đọc Tier 1: viết lại mục 4 và mục 13 theo positioning mới.
3. Khi nhánh attack qua gate A1/A2: Tier 2, rồi dựng AuthShift.
4. Khi bắt đầu mục 9.4: Tier 3, đọc 2607.08883 trước tiên.

### Ghi chú về độ tin cậy

Đã xác minh tồn tại và đọc abstract/full text: 2608.29942, 2605.09480. Các ID
2026 còn lại trong tài liệu này **chưa được xác minh tồn tại** tại thời điểm lập
danh sách; phải kiểm tra lại từng ID trước khi cite. Toàn bộ đều là arXiv preprint,
không phải bằng chứng peer-reviewed, nên chỉ dùng để xác định novelty boundary.
Cần chạy lại systematic search trước khi viết related work và một lần nữa ngay
trước submission.
