# So sánh ngôn ngữ: từng câu, nhóm câu và universal

Ngày 2026-09-15. Đánh giá độ liên quan với query gốc và độ trôi chảy của toàn câu;
không dùng ASR để xếp hạng ba cách.

## Thiết lập

- 12 câu train dùng chung, 8 validation, 48 test; seed 42.
- Universal: một trigger trên toàn bộ 12 câu.
- Specific theo nhóm: K-means trên clean DPR embeddings, K=3; mỗi nhóm học một trigger.
- Specific từng câu: 12 vị trí lưu trigger, mỗi vị trí học từ một câu train.
- Test định tuyến bằng tâm nhóm sạch hoặc câu train gần nhất; không tối ưu trực
  tiếp trên test. Bảng train mới là đánh giá specific trên chính câu đã học.
- Cùng template ba từ, tối đa 12 retriever tokens; cùng cap 768 logical requests
  mỗi nhánh/vị trí, không bảo đảm mức dùng thực tế bằng nhau.
- Đo ba vị trí suffix/prefix/infix. Mỗi vị trí có trigger được tối ưu riêng.
- Giữ cách tìm trigger hiện có: objective không tối ưu trực tiếp độ liên quan
  trigger/query hoặc PPL. Đây là phép đo ngôn ngữ hậu kiểm của các trigger đó.

Nhóm train có kích thước **1, 7, 4**. Test định tuyến **0, 39, 9** câu vào các nhóm
tương ứng. Một nhóm là singleton và một nhóm chiếm đa số câu test, nên không
diễn giải thành ba domain cân bằng. Nhóm về người/tiểu sử có các câu về Tony
Blair, Christopher Hitchens, Jay-Z/Kanye West, John Kerry/John McCain; nhóm lớn
chứa nhiều chủ đề khác nhau. Clustering chưa bảo đảm mức liên quan domain mong muốn.

## Định nghĩa điểm

- **Liên quan:** cosine MiniLM giữa trigger riêng và query gốc; cao hơn nghĩa là
  gần nhau hơn trong embedding, không phải xác suất liên quan.
- **Trôi chảy:** trung bình nhân PPL sau chèn / PPL trước chèn bằng DistilGPT2;
  thấp hơn tốt hơn theo model, 1× là mức của câu gốc. Không phải thang đo tự nhiên
  do con người chấm. Bản thân câu gốc cũng không mặc định là chuẩn ngữ pháp.
- Specificity contrast đối chiếu từng trigger với tất cả query khác trong split;
  khoảng bootstrap ghép cặp theo query được lưu trong JSON.

PPL bị ảnh hưởng bởi độ dài, tên riêng và từ phổ biến. MiniLM cũng từng được dùng
làm meaning guard trong tìm kiếm. Chưa có người đánh giá độc lập. Xem
[giải thích phép đo](trigger_language_audit.md).

## Kết quả trên chính 12 câu dùng để học trigger

Cosine trigger/query, cao hơn là gần hơn trong embedding:

| Vị trí | Universal | Specific nhóm | Specific từng câu |
|---|---:|---:|---:|
| Cuối | 0,0166 | 0,0046 | 0,0133 |
| Đầu | 0,0307 | 0,0218 | 0,0180 |
| Giữa | 0,0441 | 0,0293 | 0,0385 |

PPL sau/trước, trung bình nhân, thấp hơn tốt hơn theo model:

| Vị trí | Universal | Specific nhóm | Specific từng câu |
|---|---:|---:|---:|
| Cuối | 3,210× | 3,868× | 3,484× |
| Đầu | 1,448× | 1,253× | 1,428× |
| Giữa | 3,804× | 4,694× | 4,770× |

Universal có cosine trung bình cao hơn trong cả ba vị trí, nhưng nhiều khoảng
bootstrap của chênh lệch còn chứa 0. Không coi mọi chênh lệch nhỏ là lợi thế đã
được xác nhận. Nhóm tốt nhất về PPL ở prefix; universal tốt nhất ở suffix/infix.
Không có thứ tự ổn định "universal < nhóm < từng câu" về chất lượng ngôn ngữ.

## Chuyển sang 48 câu test chưa dùng để tối ưu

| Vị trí | Cosine universal | Cosine nhóm | Cosine từng câu | PPL universal | PPL nhóm | PPL từng câu |
|---|---:|---:|---:|---:|---:|---:|
| Cuối | 0,0317 | 0,0261 | 0,0288 | 2,555× | 2,770× | 2,496× |
| Đầu | 0,0444 | 0,0249 | 0,0344 | 1,316× | 1,089× | 1,282× |
| Giữa | 0,0467 | 0,0286 | 0,0422 | 2,849× | 3,191× | 3,805× |

Nhóm tiếp tục có PPL tốt nhất ở prefix. Specific từng câu nhỉnh hơn universal
về PPL suffix, nhưng khoảng bootstrap chênh lệch NLL chứa 0, chưa cho thấy ưu
thế rõ. Điểm liên quan trung bình của nhóm thấp hơn universal ở cả ba vị trí;
chênh lệch của từng câu so với universal còn bất định. Cả nhóm lẫn từng câu có
specificity contrast gần 0 trên test và các khoảng bootstrap đều chứa 0.

## Trigger thực tế và giới hạn diễn giải

| Vị trí | Số cụm khác nhau trong bank universal | Nhóm (3 vị trí lưu) | Từng câu (12 vị trí lưu) |
|---|---:|---:|---:|
| Cuối | 1 | 3 | 7 |
| Đầu | 1 | 3 | 6 |
| Giữa | 1 | 3 | 8 |

Trên test, chỉ hai trong ba nhóm được chọn. Nhánh từng câu dùng 7, 6, 7 cụm khác
nhau tương ứng suffix/prefix/infix. Số cụm ngôn ngữ phân biệt thực tế thấp hơn
số vị trí lưu trigger vì search có thể hội tụ về cùng một cụm.

Query đầu tiên theo thứ tự train là:

> Can Tony Blair be imprisoned in the Tower of London?

Trigger suffix tương ứng:

- Universal: `use useful context`.
- Nhóm: `review reliable context`.
- Từng câu: `review practical information`.

Cả ba vẫn là chỉ dẫn chung; không có nội dung đặc trưng về câu hỏi. Infix từng
câu tạo ra `Can Tony Blair be imprisoned consider useful information in the
Tower of London?`, có một mệnh lệnh chen vào cấu trúc câu hỏi. Đây là nhận xét
định tính của trợ lý trên ví dụ đầu tiên, không phải đánh giá mù của con người.

Kết luận chỉ áp dụng cho bộ sinh template hiện tại, một seed, số câu và cách
phân cụm trên. Chưa có ràng buộc sinh từ ngữ dựa trên nội dung query/domain, nên
không suy ra specific có điều kiện theo nội dung không thể tốt hơn. Mọi PPL trung
bình sau chèn đều lớn hơn mức câu gốc; không tuyên bố câu đã tự nhiên chỉ vì một
nhánh có điểm tốt hơn nhánh khác.

Đã kiểm tra đủ 9 bank (3 mức × 3 vị trí), 540 language records
(12 train + 48 test) × 9 và CSV tương ứng. Mỗi bảng ghép trên cùng query ID;
không gộp các bản chèn của một query thành các câu độc lập.

## Lệnh tái lập

```powershell
.\make.ps1 trigger-hierarchy --backend hf --arms universal semantic per_query --groups 3 --train-size 12 --validation-size 8 --test-size 48 --poison-count 12 --budget 768 --positions suffix prefix infix --seed 42 --corpus-embeddings ReAct/database/embeddings/agentpoison_dpr/vectors.npy --output outputs/trigger_hierarchy/three-levels-12q-seed42
.\make.ps1 trigger-language-audit --input outputs/trigger_hierarchy/three-levels-12q-seed42 --output outputs/trigger_hierarchy/three-levels-language-seed42
```

Các model chạy cục bộ từ cache; không gọi API đánh giá bên ngoài.

## Artifact ngôn ngữ

- [Bảng train/test](../outputs/trigger_hierarchy/three-levels-language-seed42/REPORT.md).
- [Điểm từng câu và paired bootstrap](../outputs/trigger_hierarchy/three-levels-language-seed42/results.json).
- [CSV câu trước/sau](../outputs/trigger_hierarchy/three-levels-language-seed42/records.csv).
