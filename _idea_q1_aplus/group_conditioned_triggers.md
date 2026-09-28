# Trigger theo nhóm câu hỏi: chốt vấn đề và kế hoạch kiểm chứng

Ngày: 2026-09-15. Trạng thái: đã có retrieval pilot; chưa có downstream action ASR.

## Cập nhật triển khai: trigger phân cấp

Đã có [pilot chạy được](../src/triggers/hierarchy/README.md) để so sánh sáu nhánh,
dựng cây từ tác động embedding của các trigger riêng và kiểm tra gộp/mix thành
universal. Seed search có giới hạn; chưa phải reproduction HotFlip toàn vocabulary.
Report phân biệt retrieval success, meaning-proxy pass và joint success.
Các kết luận tăng hiệu quả vẫn cần thí nghiệm đủ lớn và nhiều seed.
[Kết quả 6 nhánh × 3 vị trí trên DPR](trigger_hierarchy_pilot_results.md): mix
chưa tăng joint success so với merge; tín hiệu phân cấp phụ thuộc vị trí chèn.

Related work mới đối chiếu: [Adversarial Tuning, 2024](https://arxiv.org/abs/2406.06622)
đã nghiên cứu hierarchical meta-universal adversarial prompt learning. Vì vậy,
novelty phải cụ thể hơn “có hierarchy”; xem giao thức bottom-up trong README pilot.

## 1. Hướng chính

Nghiên cứu mức độ chuyên biệt hóa trigger: một trigger toàn tập, một trigger mỗi
nhóm, và một trigger mỗi câu. Mục tiêu là tìm điểm cân bằng giữa hiệu quả,
giữ nguyên ý câu hỏi, tính tự nhiên và tổng chi phí.

Giả thuyết có thể bác bỏ:

> Một số ít nhóm câu hỏi được chọn phù hợp có thể đạt hiệu quả gần tối ưu từng câu,
> với chi phí thấp hơn từng câu và ít làm biến dạng ngữ nghĩa hơn universal trigger,
> dưới cùng tổng ngân sách poisoning và tính toán.

Không mặc định trigger theo nhóm rẻ hơn universal. Không khẳng định novelty trước
khi khảo sát thêm công trình về conditional/universal attacks và trigger routing.

## 2. Đối chiếu nguồn gốc và code

- Paper AgentPoison, §3.3.1: dùng tâm benign-key embeddings cho uniqueness; nêu
  k-means như một cách lấy tâm. §3.3.2 và chú thích 3: vị trí chèn không bị giới
  hạn ở suffix. [Paper NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/eb113910e9c3f6242541c1652e30dfd6-Paper-Conference.pdf).
- Implementation upstream được xem ngày trên fit
  `GaussianMixture(n_components=5, covariance_type='full', random_state=0)` trên
  `db_embeddings`, lấy `means_` trước vòng tối ưu. Có KMeans trong hàm đánh giá phụ
  và PCA 2D để vẽ hình. Đây là snapshot hiện tại, không khẳng định là đúng commit
  dùng trong submission 2024. [Source upstream](https://github.com/AI-secure/AgentPoison/blob/master/algo/trigger_optimization.py).
- Local `algo/trigger_optimization.py:605` cũng fit GMM trước vòng tối ưu.
- Khi audit ban đầu, local `src/triggers/margin.py` lấy năm vector cách đều
  theo index, không phải tâm cụm được fit. Bản triển khai hiện tại đã sửa sang
  seeded K-means dùng chung trong `src/triggers/clustering.py`, fit một lần trên tối đa
  4.096 vectors. Đây vẫn không phải GMM baseline; không quy khác biệt này cho
  dynamic clustering. Loss contract đã đổi, cần output mới cho lượt margin mới.
- `src/triggers/losses.py` nhận centers từ bên ngoài; bản thân file loss không
  quyết định phương pháp clustering.

Ba khái niệm cần đặt tên riêng trong code:

| Khái niệm | Dữ liệu | Vai trò |
|---|---|---|
| Benign reference clusters | Keys của memory/corpus | Mốc cho uniqueness loss |
| Query routing groups | Câu hỏi sạch ở train | Chọn trigger chuyên biệt |
| Triggered embedding cluster | Câu hỏi sau biến đổi | Kết quả của compactness optimization |

Thay GMM bằng dynamic KMeans cho reference clusters không tự tạo ra trigger routing.
Reference centers cũng không cần refit nếu corpus và encoder không đổi.

## 3. Cách chia nhóm

“Ô tô / xe máy / xe đạp” là loại đối tượng/chủ đề. “Tìm đường / tránh vật cản /
đỗ xe” mới là ví dụ về tác vụ. Một câu có thể cùng chủ đề nhưng khác hành vi cần
thực hiện; một tác vụ có thể xuất hiện ở nhiều chủ đề.

Pilot nên so sánh bốn cách gán nhóm:

1. Nhóm ngẫu nhiên cân bằng — kiểm tra lợi ích chỉ đến từ việc có nhiều trigger.
2. Nhóm theo nhãn domain/task có sẵn — baseline dễ giải thích.
3. Nhóm theo embeddings câu hỏi sạch — routing học từ dữ liệu.
4. Nhóm theo task/semantic features kết hợp cấu trúc câu — giả thuyết mở rộng.

Fit router trên train, định tuyến validation/test bằng router đã cố định. Không
dùng câu đã gắn trigger để liên tục tái định nghĩa nhóm; trigger có thể làm mọi
embedding hội tụ và xóa mất cấu trúc câu hỏi gốc. Không fit trên test.

Trong mỗi nhóm, xem một số câu gần tâm và xa tâm để đặt tên sau clustering.
Không suy ra nhãn domain chỉ từ việc cụm trông tách biệt trên biểu đồ 2D.

## 4. Dynamic clustering nghĩa là gì?

### 4.1 Cập nhật tâm, số nhóm cố định

`MiniBatchKMeans.partial_fit` cập nhật từ từng batch mà không cần fit lại toàn bộ.
Nó vẫn có K cố định. [API chính thức](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html).

Thích hợp khi có thêm câu hỏi sạch. Cần version router cùng trigger bank; thay
tâm/routing có thể làm trigger cũ được gán cho một tập câu khác trước.

### 4.2 Thay đổi cả số nhóm

DP-means dùng objective gồm distortion cộng penalty cho số cụm, là baseline
thích hợp khi nghiên cứu adaptive K. Nó không tự biết nhóm nào có trigger tốt.
[Kulis & Jordan, Revisiting k-means](https://arxiv.org/abs/1111.0352).

Đề xuất nghiên cứu riêng: tách nhóm khi nhóm lớn có semantic drift cao và chất
lượng validation thấp; chỉ giữ phép tách nếu cải thiện utility/ASR/naturalness
sau khi tính cả chi phí tăng. Gộp nhóm nhỏ/tương đồng nếu trigger chung không
làm giảm chất lượng. Đây là heuristic đề xuất, chưa phải thuật toán đã kiểm chứng.

Dùng inner validation để chọn K/split/merge, outer test để báo cáo; có K_max,
min_group_size và ngân sách tối ưu cố định để tránh biến thành tối ưu mỗi câu.

Nên chạy static K trước. Dynamic clustering chỉ có lý do tồn tại nếu cải thiện
được so với static routing hoặc có dữ liệu drift được mô phỏng rõ ràng.

## 5. Tính tự nhiên phải tách khỏi giữ nguyên ý nghĩa

Đo riêng:

- Fluency: câu có đúng/nghe tự nhiên không; PPL trên toàn câu sau biến đổi chỉ là proxy.
- Relevance: phần thêm có liên quan đến tác vụ không.
- Meaning preservation: thực thể, phủ định, con số, thời gian, ràng buộc và câu trả
  lời đúng có giữ nguyên không. Cần kiểm tra thủ công một mẫu và/hoặc evaluator độc lập.

Không dùng duy nhất encoder đang bị tối ưu để chấm semantic preservation.
Một câu rất trôi chảy vẫn có thể hỏi sang việc khác. Nếu phần thêm đã yêu cầu trực
tiếp hành vi đích thì không thể coi đó là thành công của backdoor trên cùng tác vụ.

AdvPrompter đã nghiên cứu prompt thích ứng, dễ đọc và giữ nghĩa trong bối cảnh
jailbreak. Đây là related work cần đối chiếu, dù khác poisoning retrieval.
[AdvPrompter](https://arxiv.org/abs/2404.16873).

Tính tự nhiên không tự chứng minh khả năng triển khai thực tế. Phải nêu ai được
sửa query, ai được poison memory, router có nhìn thấy query không, hay trigger
cần xuất hiện tự nhiên ở người dùng. Đo cả false activation trên benign queries:
trigger càng phổ biến có thể càng làm hỏng benign utility.

## 6. Thử giảm chiều trên cache thật

Đã chạy `scripts/audit_embedding_clusters.py` với cache DPR L2-normalized:
`ReAct/database/embeddings/agentpoison_dpr/vectors.npy`, kích thước 9.251 × 768.
Lấy mẫu 3.000 documents bằng seed 42; fit PCA/clustering trên 2.400 vectors và
đánh giá 600 vectors còn lại. MiniBatchKMeans K=5. PCA xong normalize lại để
so cosine neighbors. Không gọi model, không train trigger.

| Số chiều | Variance giữ lại | Mean overlap top-10 neighbors so với 768D |
|---|---:|---:|
| 768 | 100% | 100% |
| 2 | 9,02% | 2,10% |
| 32 | 51,42% | 60,65% |
| 128 | 86,77% | 84,43% |

Diễn giải: trong thử nghiệm này, mỗi top-10 ở 2D chỉ giữ trung bình 0,21 trong
10 hàng xóm ở không gian gốc. 2D không phải surrogate retrieval đáng tin ở đây.
Đây là **document-neighbor geometry**, không phải query retrieval ASR; kết quả một
seed/một corpus subset không đủ để chọn số chiều tốt nhất cho mọi trường hợp.

Thử online K=5 qua 9 batches: 23% assignments của held-out documents thay đổi từ
sau batch đầu đến batch cuối; ARI là 0,528. Điều này minh họa router có thể đổi
phân vùng khi cập nhật tâm, không chứng minh dynamic tốt hơn static.

Chiếu embedding đã encode xuống 2D không bỏ được forward pass của encoder. Nó
chỉ giảm một phần chi phí xử lý vector sau đó. Muốn tiết kiệm, đo riêng encoder,
candidate scoring, clustering, target-model calls; dùng cache và xử lý batch.
32D/128D là ứng viên khảo sát, chưa được chấp nhận làm retrieval surrogate.

## 7. Thiết kế so sánh tối thiểu

| Experiment | Điều được kiểm tra |
|---|---|
| Universal, K=1 | Mốc gốc |
| Random groups, K=4/8/16 | Nhiều trigger có tự mang lợi ích không |
| Semantic groups, cùng K | Cấu trúc nhóm có đóng góp không |
| Task groups, cùng K | Task tốt hơn chỉ topic không |
| Per-query | Baseline chuyên biệt nhất, không mặc định là upper bound |
| Static vs online/split-merge | Giá trị riêng của dynamic routing |
| Suffix vs sentence-boundary insertion | Ablation về vị trí, không tuyên bố novelty riêng |

Giữ cùng tổng poison records, token budget, train queries và tổng model-call/compute
budget. Không cấp cùng ngân sách cho từng trigger rồi so với chỉ một universal
trigger. Báo cáo thêm mỗi nhóm được bao nhiêu budget.

Metrics: retrieval ASR, realized action ASR trong sandbox, clean task accuracy,
meaning preservation, full-query fluency, false activation, GPU/CPU time, encoder
calls và target-model calls. Report micro-average, macro-average và nhóm kém nhất.

Transfer: test queries chưa gặp, target retriever chưa dùng khi tối ưu, và nếu cần
target agent model khác. Dùng vector loss chiều gốc để xác nhận kết quả; projection
2D phục vụ biểu đồ. Không đánh đồng giảm chiều với transferability.

## 8. Lệnh tái lập geometry audit

Python cần numpy, scikit-learn (threadpoolctl được cài cùng scikit-learn).
Lần kiểm tra này dùng `python` hệ thống; `.venv-adapt` hiện chưa có scikit-learn.

```powershell
python scripts/audit_embedding_clusters.py --embeddings ReAct/database/embeddings/agentpoison_dpr/vectors.npy --output outputs/research/trigger_groups/geometry.json
```

Kết quả là geometry audit; chưa đo tăng ASR, tính tự nhiên hoặc giảm tổng chi phí
của trigger theo nhóm. Không thay đổi pipeline tối ưu trigger hiện tại trong bước này.
