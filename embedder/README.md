# `embedder/` — Train và đánh giá retriever

Retriever là mô hình biến câu hỏi thành vector; nó quyết định mẫu nào trong bộ nhớ được
lấy ra. Thư mục này để **tự train một retriever riêng** thay vì dùng DPR/ANCE/BGE có sẵn,
và để đo chất lượng retriever mà không tốn tiền API (không gọi LLM).

## 1. Hai cách train

| Cách | Ý tưởng | File train | File eval |
|---|---|---|---|
| Contrastive | Học bằng bộ ba (anchor, positive, negative): kéo câu hỏi lại gần mẫu đúng, đẩy xa mẫu sai. | `train_contrastive_retriever.py` | `eval_embed_contrastive.py` |
| Classification | Học phân loại câu hỏi vào nhóm hành động, dùng chính embedding của lớp phân loại. | `train_classification_retriever.py` | `eval_embed_classification.py` |

Cả hai đều dựng trên `bert-base-uncased` (`TripletNetwork` / `ClassificationNetwork`,
định nghĩa lại trong `algo/utils.py` để phần tấn công nạp được checkpoint).

## 2. Cấu trúc file

| File | Vai trò |
|---|---|
| `dataset_contrastive_preprocess.py` | Sinh dữ liệu bộ ba từ log tương tác của agent. |
| `dataset_classification_preprocess.py` | Sinh dữ liệu có nhãn cho hướng phân loại. |
| `train_contrastive_retriever.py` | Train theo triplet loss, lưu checkpoint. |
| `train_classification_retriever.py` | Train theo cross-entropy, lưu checkpoint. |
| `eval_embed_contrastive.py` | Đo chất lượng truy hồi của checkpoint contrastive. |
| `eval_embed_classification.py` | Đo chất lượng truy hồi của checkpoint classification. |
| `get_ada_v2_embedding.py` | Lấy embedding bằng API OpenAI (baseline `ada`). |

## 3. Nối vào phần còn lại của repo

1. Train xong, checkpoint nằm dưới thư mục trỏ bởi `EMBEDDER_CKPT_DIR` trong `.env`
   (mặc định `RAG/embedder`).
2. Khai báo tên gọi tắt của checkpoint trong `algo/config.py`
   (`model_code_to_embedder_name`), ví dụ `classification_user-ckpt-500`.
3. Dùng tên đó cho `python algo/trigger_optimization.py --model <tên>` và cho các script inference.

Cache embedding đặt tên theo `model_code`, nên đổi sang tên khác thì không lẫn cache. Thay checkpoint
mà **giữ nguyên tên** thì phải xoá `embeddings_<tên>.pkl` và `gmm_centers_*_<tên>.pt` cũ
(xem [`algo/README.md`](../algo/README.md) §4).

## 4. Chạy nhanh

```bash
python embedder/eval_embed_contrastive.py       # không tốn API
python embedder/eval_embed_classification.py
```
