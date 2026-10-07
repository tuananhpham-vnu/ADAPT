# Literature: memory poisoning cho LLM agent (đọc 2026-10-05)

Đọc bản HTML trên arXiv qua công cụ tóm tắt. Mọi con số dưới đây lấy từ bài gốc, nhưng
**cần đối chiếu lại PDF trước khi trích dẫn trong bài**. Liên kết từ `open_problems.md` §P4.

## Tóm tắt từng bài

### MINJA — [arXiv 2503.03704](https://arxiv.org/abs/2503.03704) (2025)
- **Ý tưởng:** kẻ tấn công chỉ là một user bình thường. Họ gửi query kèm "indication prompt"
  để agent tự sinh ra một suy luận cầu nối (ví dụ "dữ liệu của bệnh nhân A nay lưu dưới B"),
  và agent **tự lưu** bản ghi đó. "Progressive shortening" bỏ dần prompt qua từng vòng, nên bản
  ghi cuối cùng trông như một query lành. Memory dùng chung cho mọi user.
- **Chính sách ghi:** EHRAgent và QA agent lưu **mọi** tương tác; RAP (Webshop) chỉ lưu
  tương tác thành công.
- **Số liệu:** ISR 98.2%, ASR 76.8% (EHRAgent/MIMIC, eICU, Webshop, MMLU; GPT-4/4o).
- **Chi tiết quan trọng cho mình:** ASR **giảm khi có nhiều bản ghi lành hơn**. Trên MIMIC,
  ASR 68.9% với 25 bản ghi lành, còn 31.1% với 100 bản ghi. Đây là hiện tượng "chen chỗ",
  nhưng do bản ghi lành không liên quan gây ra, không phải do chính các lần tấn công được ghi lại.
- **Phòng thủ thử:** lọc embedding thất bại (bản ghi độc và lành "đan xen" trong không gian
  embedding); LLM làm giám khảo hoặc không tổng quát được, hoặc nhiều dương tính giả.
- **Hạn chế:** không xét chính sách ghi như một biến; đo ASR **sau** khi tiêm xong, các query
  của nạn nhân lúc test không được ghi lại; chỉ dùng cosine retrieval.

### A-MemGuard — [arXiv 2510.02373](https://arxiv.org/abs/2510.02373) (ICML 2026)
- **Ý tưởng:** lấy K=4 bản ghi, sinh K đường suy luận song song, dùng LLM làm giám khảo để
  tìm đường lệch khỏi đồng thuận. Đường lệch được cất vào một "lesson memory", và trước khi
  hành động agent tra lesson memory để được cảnh báo.
- **Đánh giá:** AgentPoison (EHRAgent, ReAct-StrategyQA), MINJA (MMLU), multi-agent;
  GPT-4o-mini và LLaMA-3.1-8B; DPR và REALM. Đây cũng là **testbed của repo mình**.
- **Số liệu:** AgentPoison trên EHRAgent: ASR-r từ 100% xuống 2.13%. MINJA: ASR 0.667 → 0.256.
  Chi phí khoảng 7.8K token mỗi query.
- **Vòng lặp tự củng cố:** chỉ đo với MINJA, theo nghĩa ISR tăng dần qua các vòng tương tác.
  Với AgentPoison thì memory vẫn tĩnh.
- **Hạn chế:** không có adaptive attacker nhắm vào cơ chế đồng thuận; giả định bản ghi độc
  chiếm "phần nhỏ"; tốn nhiều lần gọi LLM; nhạy với top-k.

### MemSecBench — [arXiv 2607.27080](https://arxiv.org/abs/2607.27080) (07/2026)
- **Ý tưởng:** benchmark vòng đời Write → Execute → Forget. Gồm 2 harness (OpenClaw, Hermes),
  4 backend (native, Mem0, Mem0-Graph, A-MEM), 3 LLM, 310 case.
- **Số liệu:** nội dung độc tồn tại trong memory 84.2%; E2E-ASR 50.3%; sửa chọn lọc thành công
  56.1%. Có khoảng cách 30 điểm giữa "xoá được độc" và "không làm hỏng memory lành".
- **Hạn chế (tự nêu):** mỗi case chỉ **một chu kỳ**. Không nghiên cứu nhiều tương tác, memory
  phình, hay ảnh hưởng của chính sách ghi.

### MemPoison — [arXiv 2605.29960](https://arxiv.org/abs/2605.29960) (05/2026)
- **Ý tưởng:** AgentPoison hỏng trên memory có chọn lọc (Mem0, A-Mem, LangMem có bước
  extract/rewrite). MemPoison dùng ba thành phần:
  - một "cầu nối ngữ nghĩa" gắn trigger với payload;
  - ngụy trang trigger thành thực thể có tên để nó sống sót qua bước rewrite;
  - **loss tập trung**, kéo mọi văn bản chứa trigger về một cụm chặt; cộng loss tách khỏi các
    tâm cụm lành. Tối ưu theo kiểu HotFlip trên một embedder thay thế.
- **Số liệu:** ASR 0.83–0.95 (AgentPoison 0.02–0.72); ổn định khi memory lành tăng từ 1K lên
  7K, và sau 10–40 vòng hội thoại **lành** (consolidate/update/forget).
- **Hạn chế:** chuyển giao kém sang embedder isotropic; chưa xét hybrid/BM25. Quan trọng nhất
  với mình: **không đo khi chính trigger được dùng nhiều lần và các lần dùng đó được ghi lại**.
  Loss tập trung bảo đảm mọi văn bản chứa trigger rơi vào cùng một cụm, mà đó đúng là điều kiện
  để tự pha loãng xảy ra (run 4).

### MEMSAD — [arXiv 2605.03482](https://arxiv.org/abs/2605.03482) (05/2026)
- **Ý tưởng:** phát hiện **lúc ghi**. Mỗi bản ghi mới được chấm bằng max và mean cosine so với
  lịch sử query gần đây (FIFO 50), rồi hiệu chuẩn trên một tập lành. Có định lý "gradient coupling"
  (detector đơn điệu theo điểm truy hồi thì không bị né liên tục) và bán kính chứng nhận.
- **Phát hiện phụ:** nhiều bản tái lập AgentPoison test bằng query **không** có trigger, nên ASR-R
  đo được chỉ 0.25 thay vì 1.00.
- **Số liệu:** AgentPoison TPR 1.00, nhưng chỉ khi hiệu chuẩn bằng query có trigger. MINJA TPR 0.40,
  InjecMEM 0.20. Kẻ tấn công thay từ đồng nghĩa né được 80–100%.
- **Hạn chế:** dữ liệu tổng hợp (1000 bản ghi); xét từng bản ghi riêng lẻ, không xét cụm bản ghi
  tích tụ theo thời gian; giả định trôi chậm.

### Coverage Is Not Containment — [arXiv 2608.16044](https://arxiv.org/abs/2608.16044) (08/2026)
- **Ý tưởng:** phòng thủ lúc nạp tài liệu không thể phân biệt một "nón" poison phối hợp với việc
  nạp hợp lệ hàng loạt tài liệu ngách (AUC 0.46). Tín hiệu duy nhất tách được hai thứ là **nhu cầu
  truy vấn**: tài liệu thật được nhiều query tự nhiên truy hồi, còn nón poison chỉ được query mục
  tiêu truy hồi. Detector dùng độ mới + độ đa dạng query đạt recall 100% ở FPR 1%.
- **Hạn chế (tự nêu):** corpus động (nạp/xoá liên tục) chưa xét; kẻ tấn công thích ứng có thể
  **tự tạo nhu cầu** bằng query tổng hợp. Chưa xét agent memory có write-back.

### MPBench / Untrusted→Trusted Memory — [arXiv 2606.04329](https://arxiv.org/abs/2606.04329) (06/2026)
- **Ý tưởng:** 4 kênh ghi: (C1) lệnh "hãy nhớ", (C2) agent tự đánh giá nên giữ gì, (C3) nén
  lịch sử khi đầy context, (C4) **chuyển kinh nghiệm thành quy trình/skill**. Có 9 lỗ hổng cấu
  trúc; V-S5 "tự cải thiện thành khuếch đại": mỗi lần chạy không lỗi được coi là bằng chứng hợp lệ.
- **Số liệu:** ASR ghi 66.67% (Hermes), 34.25% (OpenClaw); phòng thủ prompt-injection chỉ bắt
  được 38–67%.
- **Hạn chế:** tự củng cố chỉ được mô tả **định tính**, không đo động học.

### Hidden in Memory — [arXiv 2605.15338](https://arxiv.org/abs/2605.15338) (05/2026)
- **Ý tưởng:** tài liệu hoặc trang web độc làm trợ lý (ChatGPT, Claude, Gemini, Mem0) lưu một ký
  ức giả về user. Ký ức đó nằm im, rồi tái xuất ở các cuộc hội thoại sau.
- **Số liệu:** tỉ lệ ghi tới 99.8%; tỉ lệ dùng theo ý kẻ tấn công 60–89%. Activation probe đạt
  AUROC > 0.95.
- **Hạn chế:** tách riêng ghi / truy hồi / dùng; không có động học nhiều lần dùng.

## Tổng hợp: lĩnh vực đang tách thành hai góc nhìn không nối với nhau

| Góc nhìn | Ai nói | Bằng chứng |
|---|---|---|
| Ghi lại **khuếch đại** tấn công | A-MemGuard, MPBench (V-S5) | MINJA trên MMLU; định tính |
| Memory phình **không hại** trigger tối ưu | MemPoison (benign rounds), run 2–3 của mình | đo |
| Bản ghi lành **chen chỗ** làm giảm ASR | MINJA (25 → 100 bản ghi lành) | đo, nhưng không phải do chính các lần tấn công |
| Chính các lần tấn công được ghi lại **pha loãng** tấn công | run 4 của mình | đo (giả định nhãn lành) |

**Chưa ai đo:** vòng phản hồi do **sử dụng** (usage feedback). Mỗi lần backdoor được kích hoạt,
agent ghi lại chính lần đó. Chưa ai đo bản ghi ấy làm backdoor ăn sâu hay phai đi, chiều nào
phụ thuộc vào gì (chính sách ghi, nhãn, extract/rewrite), sau bao nhiêu lần dùng thì đổi chiều,
và **bên phòng thủ quan sát được gì** từ vòng đó.

Hai tín hiệu bên phòng thủ có được mà kẻ tấn công khó kiểm soát:
1. Bản ghi do nạn nhân sinh ra khi trigger được dùng đều **tụ vào cùng một cụm**. Loss tập
   trung của MemPoison và AgentPoison còn làm cụm đó chặt hơn.
2. Cụm đó có **nhu cầu tập trung**: chỉ được truy hồi bởi các query mang trigger, đúng tín
   hiệu của *Coverage Is Not Containment*. Nhưng lần này có thêm chiều thời gian: cụm lớn lên
   theo số lần sử dụng.

Ba câu hỏi mà phòng thủ hiện có chưa trả lời: MEMSAD chấm từng bản ghi nên không thấy cụm lớn
dần; A-MemGuard tốn khoảng 7.8K token mỗi query; MemSecBench dừng ở một chu kỳ.

---

## Đọc full text + quét bài mới (2026-10-07)

Đọc bản HTML đầy đủ trên arXiv (qua công cụ tóm tắt; **vẫn phải đối chiếu PDF trước khi trích**).

### Ba bài cần chốt gap: xác nhận

| Bài | Trigger tối ưu? | Agent tự ghi nhiều vòng? | Chính sách ghi là biến? | Thấy pha loãng? | Thử xoá poison gốc sau khi agent tự ghi? |
|---|---|---|---|---|---|
| MemSecBench | không (tấn công gắn vào tác vụ, ngôn ngữ tự nhiên) | không (Write–Execute–Forget, mỗi giai đoạn một lượt) | không (mỗi backend dùng cơ chế ghi mặc định) | không đo | không (Forget chỉ so trạng thái backend, không có đường "tái nhiễm" qua output của agent) |
| A-MemGuard | AgentPoison có, nhưng memory **tĩnh** | chỉ với MINJA/MMLU (Fig. 3: ISR tăng theo vòng, **không có bảng số**) | không nêu chính sách ghi | không | không |
| MEMSAD | AgentPoison có | không mô hình hoá write-back | không | không | không |

Thêm từ MEMSAD: bản tái lập AgentPoison chấm bằng query **không trigger** cho ASR-R 0.25, còn có
trigger thì 1.00. Các lần chạy của mình luôn chấm `Q_eval` có trigger, và báo riêng `off_hit`.

### Bài mới phải trích, và chỗ chúng chạm vào gap

| Bài | Đã làm | Chạm gap ở đâu | Khác mình ở đâu |
|---|---|---|---|
| [Zombie Agents](https://arxiv.org/abs/2602.15654) (02/2026) | injection ngôn ngữ tự nhiên qua web. Agent tự ghi payload vào memory (sliding window / RAG), payload **tự nhân bản** qua các phiên. So 3 "hàm tiến hoá" memory: Raw History ≈ 77% ASR, Verbal Reflection ≈ 12%, Refined Experience ≈ 3–15% (Fig. 4) | **gần nhất**: cách agent ghi memory đổi được ASR | không có trigger tối ưu. Không có mốc "không ghi" nên không biết ghi có **làm yếu đi** tấn công so với memory tĩnh hay không. Không thử xoá nguồn độc. Chỉ có hình, không có bảng số theo vòng |
| [SkillJack](https://arxiv.org/abs/2608.03509) (08/2026) | trải nghiệm độc → pipeline **trích skill** (SkillX, A2S) → skill bền. **80% tấn công còn sống sau khi xoá bản ghi gốc** | trùng hiện tượng "dọn poison gốc không đủ" | cơ chế khác: skill tách khỏi kho trải nghiệm (lifecycle isolation). Không phải agent ghi lại chính tương tác bị kích hoạt. Không trigger tối ưu, không so chính sách, không đo theo số lần dùng. Một LLM (DeepSeek-v4-flash) |
| [MemoryGraft](https://arxiv.org/abs/2512.16962) (12/2025) | trải nghiệm độc "không trigger" trong RAG của MetaGPT DataInterpreter; PRP 0.479 | — | không write-back, không theo thời gian, 12 query |
| [OEP](https://arxiv.org/abs/2605.18930) (05/2026) | trải nghiệm "đúng cục bộ" bị agent tổng quát hoá quá mức khi reflection | agent tin quá mức vào reflection tự sinh | không trigger, không so chính sách ghi |
| [Revoked but Still Authoritative](https://arxiv.org/abs/2609.08258) (09/2026) | 5 hệ memory không thực thi thu hồi: fact đã thu hồi vẫn được truy hồi và dẫn tới hành động sai | "xoá/thu hồi không đủ" | về thu hồi fact, không về backdoor hay write-back |
| [Utility Under Attack](https://arxiv.org/abs/2608.21230) (08/2026) | 1.2% corpus độc làm accuracy 0.85 → 0.30. Content screening bắt 0/360. Provenance ranking với trọng số mặc định không khác không phòng thủ (p = 0.80) | phòng thủ lúc ghi thất bại | không trigger, không write-back |

### EHRAgent gốc ghi memory thế nào (P4, bước 2): **đã trả lời**

`wshi83/EhrAgent/ehragent/main.py`:
- trước mỗi câu hỏi gọi `user_proxy.update_memory(num_shots, long_term_memory)`;
- `if result:` (tức `judge()` khớp ground truth) thì append `{"question": question, "knowledge": ..., "code": ...}`.

Như vậy:
1. Chính sách gốc của EHRAgent chính là **`verified`** (chỉ ghi khi đáp án đúng).
2. Key = `question` **nguyên văn**. Fork AgentPoison nối trigger vào `question` *trước* `initiate_chat`
   (`main.py` trong repo), nên bản ghi mang trigger. Giả định "key = query + trigger" của lần 5/7
   **khớp** EHRAgent thật.
3. Một lần tấn công thành công (code chứa `DeleteDB`) cho đáp án sai nên **không** được ghi; một lần
   trigger không nổ cho đáp án đúng nên **được** ghi, kèm trigger. Theo lần 6 (`verified` pha loãng
   −0.11 ở arm 1 poison), vòng memory gốc của EHRAgent sẽ **tự làm yếu** AgentPoison theo thời gian.
   Đây là dự đoán kiểm được nếu bật lại dòng `update_memory` đang bị comment (`main.py:159`).
   Cần database eICU để `judge()` chạy được.

### Gap sau khi đọc (thu hẹp lại so với 2026-10-05)

"Cách ghi memory ảnh hưởng tấn công" **không còn mới hoàn toàn**: Zombie Agents đã cho thấy hàm tiến
hoá đổi ASR (77% → 3–15%); SkillJack cho thấy xoá bản ghi gốc không đủ (80%). Phần còn lại, theo những
gì đã đọc thì chưa ai làm:
1. **Backdoor trigger tối ưu** (họ AgentPoison) dưới write-back. Mọi bài trên đều dùng injection ngôn
   ngữ tự nhiên.
2. **Cả hai chiều trên cùng một tấn công, so với mốc memory tĩnh.** Cùng trigger, cùng stream, chỉ đổi
   chính sách ghi, thì backdoor **mạnh lên** (`log_outcome` +0.063) hoặc **yếu đi dưới mốc tĩnh**
   (`corrected` −0.83, `verified` −0.11). Zombie Agents không có mốc "không ghi", nên không nói được ghi
   có **làm yếu đi** tấn công hay không.
3. **Cơ chế pha loãng riêng của trigger tối ưu:** bản ghi lành mang trigger chiếm hạng 1 trước poison
   (`self`: 1 bản ghi làm hit@1 tụt 0.142). Ngôn ngữ tự nhiên không có hiện tượng này, vì không có vùng
   embedding hẹp do tối ưu tạo ra.
4. **Dọn poison gốc thất bại do chính agent tự ghi lại các lần bị kích hoạt** (`cleanup_hit` 1.00).
   Cơ chế khác SkillJack (trích skill), nhưng phải trích SkillJack như hiện tượng song song.
5. **Hệ quả cho agent thật:** chính sách gốc của EHRAgent là `verified`, nên số đo trên memory tĩnh
   **đánh giá sai** AgentPoison trên EHRAgent đang chạy.
6. **End-to-end** (lần 7): con số ở mức hành động, không chỉ truy hồi. Nếu lần 7 đứng thì đây là bằng
   chứng phân biệt với mọi bài trên (đa số chỉ có ASR ở một tầng).

Rủi ro reviewer: "Zombie Agents đã cho thấy evolution function quan trọng". Trả lời: (a) trigger tối ưu
khác hẳn về cơ chế, vì có vùng embedding hẹp và tự cạnh tranh; (b) mình có mốc tĩnh và có **cả hai
chiều**; (c) có thí nghiệm xoá poison gốc; (d) chính sách ghi được tách thành biến có kiểm soát (cùng
stream, cùng seed), không phải ba kiến trúc memory khác nhau.
