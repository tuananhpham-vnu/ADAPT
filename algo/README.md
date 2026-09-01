# `algo/` — Bộ não của cuộc tấn công

Thư mục này chứa **thuật toán tối ưu trigger** và các hàm nạp dữ liệu / embedding mà cả
ba agent đều dùng chung. Nếu chỉ đọc một thư mục trong repo để hiểu AgentPoison làm gì,
hãy đọc thư mục này.

## 1. Nó giải bài toán gì

Cho một retriever (mô hình biến câu hỏi thành vector) và một bộ nhớ đã bị tiêm vài mẫu độc.
Cần tìm một chuỗi ngắn (khoảng 10 token) sao cho:

1. Câu hỏi **có** trigger thì vector của nó bị kéo về gần cụm mẫu độc, nên mẫu độc luôn lọt top-k.
2. Câu hỏi **không** trigger thì mọi thứ giữ nguyên (agent vẫn trả lời đúng).
3. Chuỗi trigger đọc vẫn tương đối tự nhiên, không bị bộ lọc perplexity loại.

Ba yêu cầu này thành ba thành phần của hàm mục tiêu: khoảng cách tới cụm, độ phân tán
(variance), và perplexity.

## 2. Cấu trúc file

| File | Vai trò |
|---|---|
| `trigger_optimization.py` | Vòng lặp tối ưu chính. Là file bạn chạy. |
| `utils.py` | Nạp model/tokenizer, nạp DB embedding cho từng agent, các Dataset. |
| `config.py` | Bảng ánh xạ `model_code` → tên model thật trên HuggingFace / đường dẫn checkpoint. |
| `linear_embedder_optimization.py` | Biến thể tối ưu trên embedder tuyến tính (thử nghiệm). |

## 3. `trigger_optimization.py` chạy như thế nào

Cụ thể hoá vòng lặp (mỗi vòng đổi đúng **một** token của trigger):

1. **Chuẩn bị** — nạp retriever (`load_models`), nạp embedding của bộ nhớ theo agent
   (`load_db_qa` / `load_db_ehr` / `load_db_ad`), rồi dùng Gaussian Mixture chia bộ nhớ
   thành 5 cụm để lấy các tâm cụm làm mốc.
2. **Tính gradient** — chạy vài batch câu hỏi thật, mỗi câu được nối thêm trigger hiện tại,
   tính loss (`compute_avg_cluster_distance` cho thuật toán `ap`, hoặc
   `compute_avg_embedding_similarity` cho `cpa`) rồi backward. Gradient tại vị trí token
   trigger được `GradientStorage` giữ lại qua backward hook.
3. **Sinh ứng viên** — `hotflip_attack` dùng gradient đó xếp hạng toàn bộ từ vựng, lấy ra
   `num_cand` token có khả năng cải thiện loss nhất cho **một** vị trí ngẫu nhiên.
4. **Lọc theo độ tự nhiên** — nếu bật `--ppl_filter`, `candidate_filter` dùng GPT-2 tính
   perplexity của trigger sau khi thay token và chỉ giữ những ứng viên đọc xuôi tai.
5. **Chấm điểm và chọn** — thay thử từng ứng viên, đo lại loss trên dữ liệu thật, chọn
   ứng viên tốt nhất. Nếu bật `--target_gradient_guidance` thì còn kiểm tra cả xác suất
   LLM đích thực sự nói ra từ khoá backdoor.
6. **Lặp lại** cho tới `--num_iter`, ghi log ra `results/<agent>/<algo>/<thời gian>/`.

Mỗi vòng lặp là một span Braintrust tên `trigger_opt.iteration`, kèm `loss`,
`best_candidate_score`, `token_to_flip` và trigger hiện tại — nhìn trace là biết trigger
tiến hoá ra sao.

## 4. `utils.py` cung cấp gì

- `load_models(model_code, device)` — trả về `(model, tokenizer, get_emb)` cho mọi loại
  retriever được hỗ trợ (DPR, ANCE, BGE, REALM, ada, checkpoint tự train).
- `load_db_qa/ehr/ad(...)` — đọc dữ liệu gốc, encode toàn bộ thành ma trận embedding,
  cache ra đĩa (chạy lần đầu rất lâu; đổi embedder thì phải `make clean-cache`).
- `bert_get_adv_emb`, `bert_get_cpa_emb` — ghép trigger vào batch câu hỏi rồi lấy embedding,
  giữ nguyên đồ thị gradient để bước 2 ở trên backward được.
- `target_asr`, `target_word_prob` — đo mức độ LLM đích thật sự bị dẫn dụ.
- `AgentDriverDataset`, `StrategyQADataset`, `EHRAgentDataset` — wrapper Dataset của PyTorch.

## 5. Cách chạy

```bash
make opt AGENT=qa           # hoặc ehr / ad
make opt-fast AGENT=qa      # bản rút gọn để smoke test
make trigger AGENT=qa       # in trigger cuối cùng
```

Tham số hay dùng: `NUM_ITER` (số vòng), `NUM_CAND` (số ứng viên mỗi vòng),
`MODEL` (retriever), `OPT_FLAGS` (`--ppl_filter --golden_trigger ...`).
