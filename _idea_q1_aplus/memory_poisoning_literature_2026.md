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
