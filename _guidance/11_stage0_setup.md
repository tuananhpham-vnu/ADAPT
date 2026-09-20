# Stage 0 — Dựng nền (khoảng 1 tuần)

Mục tiêu: đưa code ARTEMIS vào repo, chạy lại được số liệu của paper trên vài hệ nhỏ. Chưa có gì mới
về mặt nghiên cứu, nhưng mọi stage sau đều dựa vào baseline này để so sánh.

Đọc `10_artemis_overview.md` trước.

## 1. Vendor code gốc

Đưa upstream vào `src/artemis/` bằng subtree để sau này merge lại được khi anh Hiệp cập nhật:

```powershell
git remote add artemis https://github.com/hype1524/MultiAgentTesting.git
git subtree add --prefix=src/artemis artemis HEAD --squash
```

Lệnh này tạo commit vào lịch sử repo, chạy khi đã sẵn sàng.

Đừng `git clone` thẳng vào trong repo: nó tạo một repo lồng có `.git` riêng, git ngoài chỉ thấy một
thư mục lạ không theo dõi được nội dung, và mọi thay đổi bên trong nằm ngoài lịch sử của repo này.

Upstream mang theo sẵn, không phải tải riêng:

- `benchmarks/test_system/` — **đủ 16 hệ LangGraph** của paper, đánh số `1.langgraph-email-automation`
  đến `16.systematic_review_of_scientific_articles`.
- `docs/ARTEMIS_MODULE_GUIDE.md`, `docs/ARTEMIS_WORKFLOW.md`, `docs/EXPERIMENT_CONFIG.md` — đọc ba
  file này trước khi sửa gì trong pipeline.
- `runs/ablation/` — kết quả ablation đã chạy.
- `analysis/scripts/` — script dựng bảng và biểu đồ từ output.

Quy ước quan trọng: **không sửa trong `src/artemis/`**, trừ đúng một file là `model_factory.py`.
Toàn bộ phần mở rộng của mình nằm ở `src/adapt/`. Sửa lung tung trong cây upstream thì lần merge sau
sẽ conflict khắp nơi.

Cập nhật về sau:

```powershell
git subtree pull --prefix=src/artemis artemis HEAD --squash
```

## 2. Cấu hình model

`src/config.py` đã có, đọc `.env` và cấp ba vai model riêng biệt:

| Vai | Biến env | Mặc định | Việc |
|---|---|---|---|
| internal | `ARTEMIS_INTERNAL_PROVIDER` / `_MODEL` | gemini | phân rã prompt, sinh test case và criteria |
| test | `ARTEMIS_TEST_PROVIDER` / `_MODEL` | deepseek | chạy agent đang bị kiểm thử |
| judge | `ARTEMIS_JUDGE_PROVIDER` / `_MODEL` | gemini | chấm response |

Tham số chạy: `ARTEMIS_N_RUN` (mặc định 5), `ARTEMIS_N_JUDGE` (mặc định 1), `ARTEMIS_TEMPERATURE`
(mặc định 0).

`n_judge = 1` là lấy theo RQ4 của paper: tăng lên 5 vòng chấm chỉ đổi accuracy 0,01 mà bước judge
chiếm phần lớn chi phí token. `n_run` thì giữ 5, vì chạy ít lần sẽ không lộ instability.

Kiểm tra nhanh:

```powershell
python -c "from src.config import get_settings, check_role_separation; s=get_settings(); print(s.role('test'), s.role('judge')); print(check_role_separation())"
```

`check_role_separation()` trả về chuỗi cảnh báo nếu judge và test đang dùng chung một model. Paper
cố tình tách hai vai vì judge cho điểm cao hơn với response do chính model đó sinh ra. Đừng bỏ qua
cảnh báo này rồi báo cáo số.

## 3. Thêm provider vào upstream

`model_factory.py` của upstream **dựng LLM bằng LangChain** (`ChatOpenAI` từ `langchain_openai`),
đọc thông số từ hai dict `INTERNAL_AGENT_CONFIGS` và `TEST_MODEL_CONFIGS` trong `artemis/config.py`.
Cả pipeline mong nhận về object LangChain, nên **không** thay thẳng bằng `src/providers` được — đó
là adapter gọi SDK trực tiếp, khác kiểu trả về.

Cách rẻ nhất: thêm một entry cho DeepSeek vào hai dict đó. DeepSeek tương thích endpoint OpenAI nên
vẫn dùng `ChatOpenAI`, chỉ cần `base_url` và `api_key_env` riêng — đúng cách upstream đã làm cho
Qwen. Không phải viết wrapper nào.

Vậy `src/providers` dùng vào đâu? Dùng cho code của mình ở stage 12 và 15, nơi cần `ToolCall` đã
chuẩn hoá để ghi trace tool-call. Không ép nó vào cây upstream.

`src/config.py` đứng độc lập với `artemis/config.py`: nó cấp ba vai model cho code của mình và cho
`check_role_separation()`. Khi hai bên cùng mô tả một cấu hình thí nghiệm thì lấy `artemis/config.py`
làm chuẩn cho phần pipeline chạy, `src/config.py` làm chuẩn cho phần mở rộng.

## 4. Chạy baseline

Benchmark đã nằm sẵn trong upstream, không phải clone. Bắt đầu bằng ba hệ nhỏ nhất: hệ 6
`simple_travel_planner_langgraph` (3 agent), hệ 8 `music_compositor_agent_langgraph` (4 agent),
hệ 14 `news_tldr_langgraph` (2 agent). Paper báo 30-113 phút mỗi hệ.

```powershell
python src/artemis/run_pipeline.py --folder src/artemis/benchmarks/test_system/6.simple_travel_planner_langgraph --phase1-only
```

Đường dẫn `--folder` trỏ vào thư mục chứa source của hệ cần test; xem `docs/EXPERIMENT_CONFIG.md`
của upstream để biết mỗi hệ trỏ vào thư mục con nào.

Giữ lại thư mục `output/run_pipeline_<N>/` — bộ test case trong đó là đầu vào bắt buộc của stage 12
(chạy lại qua `--load-test-cases-from`).

## 5. Nghiệm thu

So Q và S thu được với cột Config2 và Config3 trong Table 4 của paper. Ngưỡng chấp nhận là **lệch
không quá 1,0 điểm** — đây chính là ngưỡng paper dùng khi đo agreement giữa hai judge khác nhau
(80% test case lệch <= 1,0). Lệch nhiều hơn thì nhiều khả năng phần cắm provider ở bước 3 sai, chứ
không phải phát hiện mới.

Vài mốc để đối chiếu: `input_interests` 2,67/3,00 với instability 0; `melody_generator` 2,46/2,17;
`summarize_articles` 2,94/2,04.
