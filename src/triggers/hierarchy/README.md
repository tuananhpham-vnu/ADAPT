# Trigger hierarchy: từ từng câu đến universal

Pilot kiểm tra giả thuyết: **trigger riêng có chứa một tác động chung có thể gộp
thành universal tốt hơn tối ưu universal trực tiếp hay không?**

[Kết quả DPR pilot và phân tích](../../_idea/trigger_hierarchy_pilot_results.md):
đã chạy đủ 6 nhánh × 3 vị trí; chưa thấy mix tăng joint success so với merge.

## Chạy ngay

```powershell
.\make.ps1 trigger-hierarchy-smoke --output outputs/trigger_hierarchy/smoke-v2
```

Tương đương trên Windows/Linux, trong môi trường có PyTorch và NumPy:

```bash
python -m src.triggers.hierarchy smoke --output outputs/trigger_hierarchy/smoke-v2
python -m src.triggers.hierarchy --help
python -m unittest discover -s tests -p test_trigger_hierarchy.py -v
```

Mặc định so sánh 6 nhánh × 3 vị trí, 8 câu train, 8 validation, 8 test, 8 poison
keys, cap 1.024 logical retriever-text requests cho mỗi nhánh/vị trí.
Fixture là hash-vector encoder tổng hợp, không phải kết quả của model thật.
Fixture dùng bag-of-words nên **không phản ánh ảnh hưởng thứ tự/vị trí**; backend
HF mới dùng positional representations.

## Các nhánh

| Arm | Training | Định tuyến câu chưa gặp |
|---|---|---|
| `universal` | Tìm một trigger trên toàn train, có seed restarts | Dùng cùng một trigger |
| `random` | Chia train ngẫu nhiên cân bằng; mỗi nhóm một trigger | Hash câu sạch, cố định theo seed |
| `semantic` | K-means trên clean query embeddings | Tâm sạch gần nhất |
| `per_query` | Mỗi câu train học một trigger | Trigger của câu train gần nhất |
| `hierarchical_merge` | Học từng câu, dựng cây, lấy trigger con làm khởi tạo rồi tối ưu ở cha | Dùng root universal |
| `hierarchical_mix` | Như trên, thêm crossover giữa các đoạn của trigger con | Dùng root universal |

`per_query` trên test là **chuyển giao bộ trigger từng câu train qua nearest neighbor**.
Không tối ưu trực tiếp trên test, cũng không chọn trigger thắng bằng test labels.
Nó không phải baseline per-instance oracle được phép truy cập test lúc tối ưu.

## Cây được xây như thế nào?

1. Chia 35% budget cap cho các leaf triggers.
2. Trên cùng tối đa hai anchor queries thuộc train, thu `E(q + t) - E(q)` cho từng trigger.
3. Nối các displacement vectors theo anchor thành một signature, chuẩn hóa L2.
4. Dùng average linkage theo cosine distance giữa signatures để gộp hai node gần nhất.
5. Ở mỗi cha, tối ưu trên hợp các câu train và poison sources của hai con.
6. Giữ tổng độ dài tối đa 3 từ và 12 retriever tokens; không lấy trung bình token IDs.

Nhánh mix ghép prefix của một trigger con với suffix của trigger còn lại, rồi
tiếp tục discrete search. Nhánh merge không thực hiện parent crossover.
Cả hai phải thỏa cùng giới hạn độ dài với universal trực tiếp.

Artifact lưu cả **lexical Jaccard**, **effect cosine**, tần suất từ xuất hiện ở các
leaf, trigger của từng node và đường gộp. Từ giống nhau không mặc định có tác động
retrieval giống nhau. Signature chỉ là mô tả dịch chuyển embedding, chưa phải
bằng chứng causal về downstream agent.

## Search và kiểm tra giữ nghĩa

Search hiện dùng **word-coordinate mutation**, seed restarts và crossover trong
một từ vựng nhỏ từ seed phrases. Mặc định `--candidate-style template` giữ ba vị trí
động từ / tính từ / danh từ của seed; `--candidate-style free` bỏ ràng buộc vị trí từ.
Cấu trúc cụm từ này chưa bảo đảm cả câu sau khi chèn tự nhiên. Đây là pilot tìm kiếm rời rạc có giới hạn, **chưa
phải HotFlip/GCG toàn bộ vocabulary** và không thay thế reproduction AgentPoison.
Muốn đổi search engine, giữ interface `Objective.score()` và `Budget` trong `search.py`.

Objective tái dùng `src/triggers/losses.py`: uniqueness + 0,1 compactness + top-k
retrieval margin. Reference centers được fit K-means trên clean corpus trong
chiều gốc, tối đa 4.096 vectors; không chọn năm vector đại diện làm tâm.

`MeaningGuard` dùng encoder độc lập so sánh original/altered và kiểm tra số,
phủ định không đổi. Trên HF, retriever mặc định là DPR, semantic encoder là MiniLM.
Một candidate chỉ được coi là feasible nếu mọi câu trong nhóm vượt ngưỡng.
Nếu chưa tìm được candidate feasible, artifact đánh dấu `constraint_feasible=false`;
vẫn giữ ứng viên tốt nhất để phân tích thất bại.

**Similarity không chứng minh giữ nguyên ý.** Chưa có entailment hai chiều,
kiểm tra đáp án gốc hoặc human review; chưa đo fluency/PPL. Báo cáo đồng thời:

- retrieval hit rate;
- meaning-proxy pass rate;
- joint success: cùng một câu vừa retrieval hit vừa vượt meaning guard;
- false activation trên câu không thêm trigger;
- micro, macro và nhóm kém nhất.

## Vị trí chèn

- `suffix`: cuối câu.
- `prefix`: đầu câu.
- `infix`: chèn tại khoảng trắng gần giữa câu nhất; câu không có khoảng trắng
  bên trong fallback về prefix. Đây là vị trí thứ ba mặc định.
- `sentence`: sau dấu kết thúc câu/mệnh đề đầu tiên nếu có phần văn bản phía sau.
  Nếu query chỉ có một câu, fallback về prefix; không gọi đó là chèn giữa.

Renderer giống nhau khi train, tính giữ nghĩa, tạo poison keys và evaluate.
Mỗi record lưu `actual_position`; metric `actual_infix_rate` cho biết tỷ lệ thực sự
chèn giữa. Không tính fallback prefix là bằng chứng về chèn giữa.
Backend HF từ chối truncation vượt `--max-length` để tránh mất câu hỏi/trigger.

## Công bằng về ngân sách

`--budget` giới hạn tổng **logical retriever text requests** cho training một nhánh,
không phải budget riêng cho mỗi trigger. Một candidate tiêu `n_queries + n_poison_sources`.
Hierarchy tính cả chi phí leaf training, effect signatures và mọi internal node.
Câu/corpus sạch và semantic representations gốc là phần chuẩn bị dùng chung.

Mỗi nhánh giữ nguyên **tổng** poison keys. Khi evaluate, truy hồi trên toàn bộ
poison keys của tất cả nhóm cùng clean corpus; không lọc sẵn nhóm “đúng”.
Tại các cut của cây, poison sources giữ owner theo subtree đã học.

Đây là cùng **cap**, không đảm bảo luôn dùng hết cap hoặc tốn cùng GPU time.
Report lưu request usage, thời gian và physical cache misses cho retriever/semantic
encoder. Cache được dùng chung, nên không suy ra speedup từ thời gian một lượt
theo thứ tự nhánh này. Cần benchmark độc lập nhiều seed khi đánh giá chi phí.

## Chạy trên StrategyQA và model thật

Mặc định chỉ đọc model đã cache tại máy. Dùng `--allow-downloads` nếu cần tải model.
`vectors.npy` phải có `manifest.json` cùng thư mục, khớp corpus/model/normalization/
max-length của cache StrategyQA hiện có.

```powershell
.\make.ps1 trigger-hierarchy --backend hf --train-size 4 --validation-size 4 --test-size 4 --poison-count 4 --budget 256 --positions suffix --arms universal random semantic per_query hierarchical_merge hierarchical_mix --corpus-embeddings ReAct/database/embeddings/agentpoison_dpr/vectors.npy --output outputs/trigger_hierarchy/dpr-small
```

Chạy đủ các vị trí:

```bash
python -m src.triggers.hierarchy compare --backend hf --positions suffix prefix infix --train-size 8 --validation-size 16 --test-size 32 --poison-count 8 --budget 2048 --corpus-embeddings ReAct/database/embeddings/agentpoison_dpr/vectors.npy --output outputs/trigger_hierarchy/dpr-positions
```

Có thể chọn `--device cuda` và `--batch-size` theo phần cứng. Không yêu cầu hai GPU.
Đổi seed/budget/dataset/vị trí cần output mới. Chạy lại cùng lệnh sẽ tái sử dụng các
arm đã hoàn tất; arm bị gián đoạn chạy lại từ seed, chưa resume từng candidate.

## File và artifact

| File | Trách nhiệm |
|---|---|
| `data.py` | Split, chống trùng query/ID, nguồn poison dùng chung |
| `backends.py` | Fixture/HF, cache và bộ đếm encode |
| `text.py` | Chèn trigger, seed phrases, mutation/crossover |
| `quality.py` | Kiểm tra giữ nghĩa dạng proxy |
| `search.py` | Budget và tối ưu trên một tập câu |
| `hierarchy.py` | Effect signatures và cây average linkage |
| `experiment.py` | Các baseline, gộp node và resume theo arm |
| `evaluation.py` | Routing sạch, full poison bank, metrics |
| `cli.py`, `io.py` | Lệnh chạy và artifact |
| `report.py` | Bảng so sánh, xu hướng theo cut và đóng góp crossover |
| `../clustering.py` | Reference K-means dùng chung |

Output: `config.json`, `dataset.json`, `reference_clusters.json`, `REPORT.md`,
`summary.json`, và `{position}/{arm}.json`. Mỗi arm chứa trigger bank, các record
validation/test, câu trước/sau sửa, poison keys; hierarchy có thêm cây và các cut
N, N/2, 2, 1 để xem xu hướng. Các cut được định trước, không chọn theo test.
History ghi nguồn ứng viên (`seed`, `parent_seed`, `mutation`, `restart`, `crossover`),
loss và việc cải thiện best feasible. Báo cáo chỉ đếm crossover đã được đánh giá
sau khi loại trùng; phân biệt cải thiện loss trên train với cải thiện kết quả test.
Contract phiên bản 2 bổ sung template search và infix; output phiên bản cũ cần
giữ riêng và chọn thư mục mới khi chạy lại.

## Cách đọc kết quả để quyết định hướng

### Đo riêng độ liên quan và độ trôi chảy

```powershell
.\make.ps1 trigger-language-audit --input outputs/trigger_hierarchy/dpr-expanded-v2 --output outputs/trigger_hierarchy/language-audit-v1
```

Lệnh này giữ nguyên trigger và không dùng retrieval metrics. `language_audit.py`
đo trigger/query cosine, đối chứng query khác, ghép cặp với universal; `fluency.py`
đo full-text NLL/PPL bằng DistilGPT2. Có cả train và test, CSV câu trước/sau và
khoảng bootstrap theo query. Model cần được cache trước; không tự tải khi audit.
[Cách đọc điểm, kết quả và chuẩn bị cache](../../_idea/trigger_language_audit.md).
[So sánh ba mức trên 12 câu train và 48 câu test](../../_idea/trigger_three_levels_language.md).
[Bản cải thiện specificity theo query hiện tại](../specificity/README.md).

### Đánh giá hiệu quả retrieval

Chỉ coi mix có ích nếu joint success/meaning tốt hơn universal trực tiếp trên
test chưa gặp, với cap và tổng poison count như nhau, qua nhiều seed. Nếu leaf
triggers tốt nhưng hiệu quả giảm khi lên root, đó là dấu hiệu các tác động khó
dùng chung hoặc search gộp chưa đủ tốt; chưa chứng minh không tồn tại universal.

Related work cần đối chiếu: [Adversarial Tuning](https://arxiv.org/abs/2406.06622)
đã có hierarchical meta-universal prompt learning. Điểm cần kiểm chứng ở đây là
**bottom-up grouping theo retrieval effect, crossover giữ giới hạn độ dài và
meaning-constrained transfer**, không phải tuyên bố mọi dạng trigger phân cấp đều mới.
