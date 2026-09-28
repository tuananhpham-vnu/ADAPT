# Tăng specificity: từ bank trigger chung sang chọn theo query hiện tại

Ngày 2026-09-15. Phạm vi: **liên quan đến query và chất lượng ngôn ngữ**, không
đánh giá ASR. Code ở [src/triggers/specificity](../src/triggers/specificity/README.md).

## Kết quả chính

Đã tăng điểm bám query và độ trôi chảy theo model bằng cách **trích topic từ câu
đang đến rồi chọn prefix cho chính câu đó**. Phần lớn mức tăng điểm liên quan là
nhờ nhắc lại topic; không phải phát hiện nội dung mới hay chứng minh khả năng
suy luận tốt hơn. Không nên chuyển một trigger chứa tên riêng của câu train sang
câu mới chỉ vì hai câu gần nhau trong embedding.

Trên **48 câu mới của lượt xác nhận**, điểm từ model audit không tham gia lựa chọn:

| Phương án | Cosine trigger/query ↑ | PPL sau/trước ↓ | Qua proxy giữ nghĩa |
|---|---:|---:|---:|
| Specific cũ, chuyển qua nearest neighbor | 0,0213 | 2,762× | 45/48 |
| Chỉ sửa dấu câu/viết hoa của specific cũ | 0,0213 | 1,525× | 45/48 |
| Chọn seed chung bằng objective ngôn ngữ | 0,0676 | 1,209× | 47/48 |
| Trigger có topic nhưng bank train cố định | 0,1141 | 1,291× | 23/48 |
| **Chọn theo query hiện tại, có lọc POS** | **0,6264** | **0,598×** | **48/48** |

PPL là trung bình nhân tỷ lệ PPL của toàn câu sau/trước chèn bằng GPT-2; 1× là mức
câu gốc. Không diễn giải 0,598× thành phần trăm tự nhiên do con người cảm nhận.
Giữ nguyên chuỗi câu hỏi + similarity không chứng minh intent/đáp án hoàn toàn
không đổi; chưa có human review hoặc chứng nhận bằng entailment.

Paired bootstrap 2.000 lần trên 48 query cho chênh lệch cosine của phương án mới
so với specific cũ: +0,6052, khoảng 95% [0,5602; 0,6446]. Chênh lệch mean NLL:
−1,5302, khoảng [−1,6732; −1,3868]. Đây là mô tả pilot, không có điều chỉnh nhiều
phép so sánh hay đánh giá biến thiên qua nhiều training seeds.

## Khám phá đã dẫn đến thay đổi nào?

### Lượt khám phá v1

Giữ 12 câu train, 8 validation; lấy 48 query chưa xuất hiện trong test 48 câu
của lần so sánh trước. Thử 16 variants: 5 loại bank cho mỗi scope universal /
semantic / per_query, cộng query-adaptive. Model audit xác nhận query-adaptive
có cosine 0,6348, PPL ratio 0,755×, trong khi specific cũ đạt 0,0266 và 2,589×.

Tuy nhiên, trích n-gram đơn giản sinh ra cụm vụng như `chaff produced`, `ropes
required`. Một universal ứng viên thành `About far north`, và các trigger tên
riêng cố định làm giảm giữ nghĩa khi chuyển sang query khác. Đó là lỗi dù các
proxy riêng lẻ có thể tăng. Không chọn v1 làm bản mặc định cuối cùng.

### Lượt xác nhận v2

Sửa bằng quy tắc tổng quát trước khi chấm tập test tiếp theo:

1. POS model lọc cụm 1–2 từ có head danh từ/tên riêng; bỏ động từ, trạng từ,
   không nối qua dấu câu hoặc cắt chuỗi tên riêng theo nhãn POS.
2. Topic candidate của một nhóm phải có cosine >= 0,15 với mọi câu train trong
   nhóm. Nếu không có topic phủ được nhóm, giữ seed chung thay vì ép dùng một
   chủ đề chỉ đúng với một thành viên.
3. Chọn prefix theo query mới khi inference. Chỉ đọc văn bản query, không đọc
   đáp án/nhãn test; giới hạn 20 candidates được chấm đầy đủ.
4. Chốt variant trên validation trước khi chấm fresh test. Validation chọn
   generic-language cho universal/nhóm và query-adaptive cho từng câu.

Lấy **48 câu khác**, loại toàn bộ test của v1 và lượt so sánh trước. Kết quả chính
ở bảng đầu là v2. Hai lượt không phải hai seed huấn luyện độc lập: cùng train/
validation, nhưng quy tắc và test partition khác nhau. Không gộp để tuyên bố
96 mẫu test độc lập cho cùng một thuật toán cố định.

## Riêng universal và nhóm

Lựa chọn chốt từ validation có kết quả audit trên test v2:

| Scope | Bản cũ: cosine / PPL | Bản được chọn: cosine / PPL |
|---|---|---|
| Universal | 0,0094 / 2,751× | 0,0709 / 1,152× |
| Nhóm | 0,0207 / 2,175× | 0,0651 / 1,218× |
| Từng câu | 0,0213 / 2,762× | 0,6264 / 0,598× |

Universal/nhóm cải thiện bằng dấu câu và chọn lại seed chung, **chưa đạt specificity
theo domain**. Giữ nguyên nhóm DPR 1–7–4 từ source để tách thay đổi ngôn ngữ khỏi
thay đổi clustering. Nhóm lớn vẫn trộn chủ đề nên chưa có một topic ngắn bao phủ
tốt. Kết quả không chứng minh mọi cách chia nhóm đều kém query-adaptive.

## Tăng điểm có phải chỉ do lặp từ?

**Phần lớn mức tăng điểm liên quan là nhờ nhắc lại topic.** Trong query-adaptive,
100% content words của prefix thuộc query hiện tại; tối đa hai từ được trích.
Đây là cơ chế có chủ ý, được log để tránh nhầm là tăng thông tin.

Chạy thêm diagnostic sau thí nghiệm: giữ nguyên topic đã chọn, chỉ bỏ từ
About/Regarding, vẫn có dấu hai chấm. Không dùng diagnostic này để chọn lại
phương án hoặc sửa thuật toán sau khi xem test.

| Dạng prefix với cùng topic | Audit cosine | Audit PPL ratio |
|---|---:|---:|
| Chỉ topic | 0,6211 | 0,690× |
| About/Regarding + topic | 0,6264 | 0,598× |

Vì chỉ-topic đã đạt gần toàn bộ điểm liên quan, không quy mức tăng 0,02 → 0,63
cho template About/Regarding. Template chủ yếu giúp cách nối đọc trôi chảy hơn
theo LM trong tập này. Topic heading có thể vẫn thừa đối với người đọc; chưa
đánh giá redundancy/naturalness bằng người.

## Ví dụ và lỗi còn lại

Các câu đầu tiên của test v2, theo thứ tự artifact:

```text
About Superhero fiction: Was Superhero fiction invented in the digital format?
About dental bills: Can professional boxers expect to have low dental bills?
About photosynthesis: Are all the elements plants need for photosynthesis present in atmosphere of Mars?
Regarding Cyril Ramaphosa: Can Cyril Ramaphosa become Secretary General of NATO?
```

Phần query sau dấu hai chấm giữ nguyên. Không phải viết lại query thành câu mới.
POS vẫn có thể sai: ví dụ `About John George` trước câu hỏi về `John George Bice`
cho thấy bộ lọc chưa bảo đảm giữ nguyên mọi tên nhiều từ. Vẫn cần review ngữ pháp,
tính thừa và độ đầy đủ của topic; không mặc định mọi prefix được chọn đều tốt.

## Mô hình, ngân sách và kiểm tra

- Lựa chọn: all-MiniLM-L6-v2 và DistilGPT2.
- Audit: [paraphrase-MiniLM-L3-v2](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L3-v2)
  và [GPT-2](https://huggingface.co/openai-community/gpt2), không dùng để chọn
  candidate/variant. Chúng cùng họ hoặc có liên hệ huấn luyện với model lựa chọn,
  nên không gọi đây là đánh giá con người hoặc hoàn toàn độc lập.
- [POS model](https://huggingface.co/vblagoje/bert-english-uncased-finetuned-pos)
  gán nhãn từ loại. Cụm từ không chứa số/phủ định để tránh nhân đôi protected tokens.
- Mỗi lượt có 12 train + 8 validation + 48 test, 16 variants, 2 bộ scorer:
  2.176 records; chỉ 68 query khác nhau mỗi lượt, không phải 2.176 query độc lập.
- Query-adaptive v2 chấm 774 modified texts cho 48 câu test, trung bình 16,125
  candidates/câu; tổng thời gian trong các lần fit khoảng 20,47 giây ở CPU với
  model/cache đã ấm. Không bao gồm nạp model, routing hay audit, không suy ra
  tốc độ production hoặc so công bằng với bank chỉ cần một lần routing.
- CLI tạo prefix mới đã chạy: `Regarding bicycles: Can bicycles use this lane?`.
- Toàn bộ 56 unit tests đã qua; đã kiểm tra số records/CSV, query IDs ghép cặp,
  các test partition không trùng, bảo toàn chuỗi gốc/số/phủ định và candidate cap.

## Code và artifact

```powershell
.\make.ps1 trigger-suggest --query "Can bicycles use this lane?" --output outputs/specificity/example-bicycle.json
```

- [Hướng dẫn file, model cache và lệnh tái lập](../src/triggers/specificity/README.md).
- [Báo cáo xác nhận đầy đủ](../outputs/specificity/confirm-pos-v2/REPORT.md).
- [Điểm từng câu](../outputs/specificity/confirm-pos-v2/results.json).
- [Câu trước/sau và điểm CSV](../outputs/specificity/confirm-pos-v2/records.csv).
- [Lựa chọn chốt trên validation](../outputs/specificity/confirm-pos-v2/selection.json).
- [Diagnostic chỉ-topic](../outputs/specificity/confirm-pos-v2/topic_control.json).
- [Lượt khám phá ban đầu](../outputs/specificity/explore-v1/REPORT.md).

Hướng sử dụng hiện tại: query-adaptive cho query mới; không tự động lấy prefix
chứa tên riêng từ câu train khác. Mục tiêu tiếp theo nếu cần tự nhiên hơn theo
người đọc là giảm lặp/thừa và kiểm chứng giữ ý bằng review, chưa có bằng chứng
phần việc đó đã được giải quyết bởi PPL/cosine.
