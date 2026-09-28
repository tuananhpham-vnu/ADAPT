# Đo độ liên quan và độ tự nhiên của trigger hiện có

Ngày 2026-09-15. **Không dùng ASR trong đánh giá này.** Giữ nguyên trigger đã học,
đo 6 nhánh × 3 vị trí trên 4 câu train và 16 câu test: 360 records, nhưng chỉ
20 query độc lập. Không phải 360 câu hỏi khác nhau.

## Kết luận trong phạm vi pilot

- Specific chưa liên quan đến query gốc tốt hơn universal một cách nhất quán.
- Theo DistilGPT2, specific trôi chảy hơn universal ở suffix, kém hơn ở prefix;
  chênh lệch infix nhỏ và khoảng bootstrap còn chứa 0 trên thang NLL.
- Cả hai loại đều làm tăng perplexity trung bình so với câu gốc tại cả ba vị trí.
  Chèn giữa có mức tăng lớn nhất trong các cấu hình đã chạy.
- Các trigger vẫn là chỉ dẫn chung như `use useful details`, không chứa nội dung
  đặc trưng cho câu hỏi. Tối ưu riêng từng câu chưa đồng nghĩa với viết riêng
  ngôn ngữ phù hợp câu đó.

Đây là **đánh giá bằng proxy**, kèm xem ví dụ; chưa có chấm điểm tự nhiên của người
đọc. Không quy đổi các chỉ số dưới đây thành phần trăm tự nhiên/liên quan.

## 1. Cách đo

**Liên quan:** cosine giữa embedding của trigger riêng và query gốc bằng
`sentence-transformers/all-MiniLM-L6-v2`. Không lấy similarity giữa câu gốc và
câu sau chèn làm điểm liên quan. MiniLM này cũng từng được dùng làm meaning guard
trong tìm kiếm; đây không phải một semantic evaluator hoàn toàn mới.

**Bám đúng query:** lấy cosine với query được gán trừ cosine trung bình với tất cả
query khác trong cùng split. Số dương chỉ ra trigger gần query được gán hơn trung
bình các query còn lại trong không gian model. Với một trigger universal cố định,
trung bình contrast bằng 0 theo công thức; không diễn giải đó là universal không
liên quan. Contrast phụ thuộc tập query đối chứng, không phải thước đo tuyệt đối.

**Độ trôi chảy:** full-text causal NLL/PPL bằng `distilbert/distilgpt2`, model
không được dùng để tìm các trigger này. Đo đúng câu đã render, không thêm dấu câu,
không sửa capitalization và không cắt bớt token. Loại padding và token đầu không
có dự đoán; tính trung bình NLL trên các token được dự đoán của từng câu.

Đại lượng báo cáo là `exp(mean(NLL_after - NLL_before))`: trung bình nhân của
PPL sau/trước chèn, mỗi query có trọng số bằng nhau. **Nhỏ hơn là tốt hơn theo
model; 1 là bằng câu gốc.** PPL không chứng minh đúng ngữ pháp hay tự nhiên đối
với con người; từ phổ biến, tên riêng, độ dài và thể loại văn bản đều ảnh hưởng.
[Công thức PPL](https://huggingface.co/docs/transformers/en/perplexity),
[model card](https://huggingface.co/distilbert/distilgpt2).

Các vị trí đã được tối ưu riêng và có trigger khác nhau: bảng này đánh giá các
cấu hình hoàn chỉnh, chưa tách riêng tác động nhân quả của vị trí khỏi từ ngữ.

## 2. Universal và specific trên 16 câu chưa gặp

Specific ở test chọn trigger của clean training query gần nhất, không tối ưu
trực tiếp trên câu test.

| Vị trí | Cosine universal ↑ | Cosine specific ↑ | PPL sau/trước universal ↓ | PPL sau/trước specific ↓ |
|---|---:|---:|---:|---:|
| Cuối | 0,0633 | 0,0303 | 3,086× | 2,476× |
| Đầu | 0,0215 | 0,0308 | 1,189× | 1,252× |
| Giữa | 0,0661 | 0,0520 | 4,313× | 4,112× |

Specific có cosine cao hơn ở đầu, thấp hơn ở cuối và giữa. Về trôi chảy, suffix
specific giảm PPL khoảng 19,8% so với suffix universal, nhưng vẫn cao gấp 2,476
lần câu gốc theo trung bình nhân. Không phải "tự nhiên hơn 19,8%".

Contrast bám query của specific: suffix +0,0071, prefix +0,0048, infix −0,0133.
Cả ba khoảng bootstrap 95% đều chứa 0; tập nhỏ này chưa cho thấy việc gán trigger
specific bám query rõ ràng hơn đối chứng query khác.

Bootstrap ghép cặp theo query ID, 2.000 lần, seed 42; không gộp các vị trí thành
các mẫu độc lập. Đây là khoảng mô tả trong pilot, không phải kiểm định khẳng định
trên nhiều seed/model hoặc có điều chỉnh nhiều phép so sánh.

## 3. Trên chính 4 câu được dùng để học trigger

| Vị trí | Cosine universal ↑ | Cosine specific ↑ | PPL sau/trước universal ↓ | PPL sau/trước specific ↓ |
|---|---:|---:|---:|---:|
| Cuối | 0,0472 | 0,0709 | 3,526× | 2,975× |
| Đầu | 0,0389 | 0,0424 | 1,121× | 1,198× |
| Giữa | 0,0650 | 0,0443 | 4,727× | 4,640× |

Ngay trên train, specific chỉ có lợi thế tương đối rõ về cả hai điểm ở suffix;
không có ưu thế đồng đều. Chỉ 4 câu nên chưa khái quát thành kết luận nghiên cứu.

## 4. Ví dụ đọc trực tiếp

Lấy query đầu tiên trong test theo thứ tự artifact, không chọn theo điểm:

> Did the Royal Air Force fight in the Boxer Rebellion?

Suffix universal:

> Did the Royal Air Force fight in the Boxer Rebellion? check useful evidence

Suffix specific:

> Did the Royal Air Force fight in the Boxer Rebellion? consider useful information

Infix specific:

> Did the Royal Air Force fight check useful evidence in the Boxer Rebellion?

Nhận xét của trợ lý từ ví dụ: cả hai cụm suffix là chỉ dẫn chung, không giải thích
hoặc bổ sung nội dung đặc trưng về Royal Air Force/Boxer Rebellion. Infix đặt một
mệnh lệnh vào giữa cấu trúc câu hỏi, làm câu khó đọc. Đây là nhận xét định tính
trên ví dụ minh họa, không phải điểm do người đánh giá độc lập cung cấp.

## 5. Kiểm tra phép đo và tái lập

Sanity probes được định trước:

- `Can bicycles use this lane?`: cosine với `bicycle lane access` = 0,7742;
  với `chocolate cake recipe` = 0,0288.
- `The cyclist is riding a bicycle.` có PPL 221,63; bản đảo ngữ pháp
  `The cyclist riding is a bicycle.` có PPL 1.081,26.

Hai ví dụ chỉ kiểm tra chiều đo cơ bản, không đủ hiệu chỉnh ngưỡng hoặc xác nhận
model đánh giá đúng toàn bộ dữ liệu. Năm unit tests kiểm tra alignment next-token,
padding, đối chứng relevance, ghép query ID và dùng đúng trigger train đã qua.

```powershell
.\make.ps1 trigger-language-audit --input outputs/trigger_hierarchy/dpr-expanded-v2 --output outputs/trigger_hierarchy/language-audit-v1
```

Đo cục bộ bằng model đã cache, không gọi API đánh giá bên ngoài.
MiniLM dùng cache Hugging Face hiện có. DistilGPT2 đã tải vào `outputs/model-cache`.
Lần đầu trên máy khác có thể chuẩn bị cache bằng:

```python
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
for loader in (AutoTokenizer, AutoModelForCausalLM):
    loader.from_pretrained("distilbert/distilgpt2", cache_dir="outputs/model-cache")
AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
```

- [Bảng đủ 6 nhánh](../outputs/trigger_hierarchy/language-audit-v1/REPORT.md).
- [Điểm từng câu và khoảng bootstrap](../outputs/trigger_hierarchy/language-audit-v1/results.json).
- [CSV câu trước/sau và điểm](../outputs/trigger_hierarchy/language-audit-v1/records.csv).
- Code: `src/triggers/hierarchy/language_audit.py`, `fluency.py`.

Muốn kiểm chứng specific thực sự tự nhiên/liên quan hơn, thí nghiệm tiếp theo cần
sinh ứng viên có điều kiện theo nội dung query/nhóm, giữ cùng giới hạn chỉnh sửa,
rồi chấm mù toàn câu về ngữ pháp, độ thừa và mức phù hợp nội dung. Kết quả hiện tại
chỉ phản ánh các trigger chung chung được tìm bởi search cũ.
