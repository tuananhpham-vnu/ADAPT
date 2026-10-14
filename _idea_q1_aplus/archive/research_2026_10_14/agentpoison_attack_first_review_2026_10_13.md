# AgentPoison làm nền chính: chọn hướng cải tiến dễ triển khai

> **Đã thay thế phần lựa chọn hướng:** người dùng yêu cầu xem nhiều phương án độc lập.
> Đọc [các hướng attack-first memory tĩnh](static_memory_attack_options_2026_10_13.md).
> Đề xuất chọn demonstrations bên dưới là khuyến nghị trước của assistant, không phải
> quyết định của người dùng. Giữ lại như ứng viên/baseline triển khai.

Ngày rà soát: **2026-10-13**. Theo yêu cầu hiện tại: **attack-first, cải thiện AgentPoison trước**.

**Khuyến nghị:** làm trên memory tĩnh trước; ưu tiên cách chọn bộ poison demonstrations
theo kết quả hành vi khi có các memory khác cùng được đưa vào context. Giữ memory động
làm phép thử chuyển giao sau khi phương pháp thắng baseline tĩnh. Đây là hướng pilot
khả thi với repo, **chưa phải novelty đã xác nhận**.

## 1. Phạm vi và độ chắc chắn

- Đã đọc PDF AgentPoison, nguồn gốc của các peers bên dưới, code local liên quan retrieval,
  target scoring, prompt assembly và write-back; đối chiếu báo cáo P0.
- Đây là rà soát có mục tiêu, không phải systematic review. Các bài chỉ đọc abstract được
  đánh dấu riêng; không suy ra chúng thiếu một thí nghiệm chỉ vì abstract không nhắc tới.
- Không chạy model/GPU/API trong lần rà soát này. `outputs/kaggle/` không có trong checkout,
  nên số P0 bên dưới lấy từ **báo cáo đã lưu**, chưa đối chiếu lại raw artifact.
- Code upstream hiện trên GitHub có thể khác checkout local. Không dùng README upstream
  để khẳng định local đã hỗ trợ một chức năng.

## 2. Memory tĩnh và động: phải tách ba thiết lập

| Thiết lập | Định nghĩa vận hành trong thí nghiệm | Quyết định |
|---|---|---|
| Tĩnh | Sau khi chèn poison, giữ nguyên store trong một lượt đánh giá | Làm chính trước: ít biến, dễ quy kết nguyên nhân |
| Corpus thay đổi bên ngoài | Thêm/xoá bản ghi theo lịch độc lập với output của agent | Stress test; không gọi đây là tự học qua memory |
| Memory có vòng phản hồi | Agent trả lời rồi ghi/extract/rewrite/merge; bản ghi mới ảnh hưởng lượt sau | Làm sau: phải kiểm soát writer, stream, policy và lineage |

RAG là cách truy hồi, không đồng nghĩa với memory tĩnh. Memory tĩnh ở đây cũng không có
nghĩa là trọng số LLM tĩnh. Một hệ thống có khả năng ghi vẫn có thể được đánh giá bằng
protocol đóng băng store; cần mô tả protocol thay vì gắn nhãn cả kiến trúc.

[AgentPoison](https://arxiv.org/pdf/2407.12784), §3, tối ưu trigger và chèn ít cặp key–value;
query có trigger truy hồi demonstration độc. Threat model cho quyền chèn vào DB và dùng
embedder white-box khi tối ưu. Bài đã xét target generation, coherence và transfer;
không được định vị cải tiến bằng claim “bài gốc chỉ tối ưu retrieval”. Evaluation chính
không nghiên cứu vòng feedback ghi lại output nhiều thế hệ. PDF v1 phân biệt khoảng
82% retrieval và 63% end-to-end ở phần giới thiệu; không đọc con số ≥80% trong abstract
thành ≥80% hành vi cuối cùng cho mọi agent.

## 3. Peers cần đặt cạnh đề tài

Các năm dưới là năm bản arXiv đầu, không mặc định là năm hội nghị.

| Peer / nguồn gốc | Phần đã được nghiên cứu | Hệ quả với hướng của mình | Mức đọc lần này |
|---|---|---|---|
| [Phantom, 2024](https://arxiv.org/abs/2405.20485) | Backdoor một document, hai bước retrieval và generation | “Một poison + trigger + hành vi” không tự mới | Abstract |
| [PoisonedRAG, 2024](https://arxiv.org/abs/2402.07867) | Poison tài liệu để đổi đáp án cho target question | Baseline QA hữu ích, nhưng khác universal-trigger threat model | Abstract; [repo](https://github.com/sleeepeer/PoisonedRAG) |
| [MINJA, 2025](https://arxiv.org/abs/2503.03704) | Ghi memory qua query-only interaction | Bỏ quyền sửa DB trực tiếp đã có prior art | Abstract |
| [PoisonArena, 2025](https://arxiv.org/abs/2505.12574) | Nhiều poison cạnh tranh mục tiêu khác nhau | Không claim mới chỉ vì thêm rival vào top-k | Abstract; [repo](https://github.com/yxf203/PoisonArena) |
| [MemoryGraft, 2025](https://arxiv.org/abs/2512.16962) | Trải nghiệm độc được lưu rồi được bắt chước về sau | “Poison kinh nghiệm” và persistence đều đã có | Abstract |
| [MemPoison, 2605.29960](https://arxiv.org/html/2605.29960) | Qua selective extraction/rewrite bằng semantic bridge, entity-like trigger và tối ưu embedding | Peer sát nhất nếu chuyển AgentPoison sang memory có writer; không lấy “trigger sống qua rewrite” làm claim mới | Full text trọng điểm: §3–6; [repo đã mở](https://github.com/ys-miracle/MemPoison) |
| [InjecMEM, 2026](https://arxiv.org/html/2608.23471) | Một tương tác; anchor và command tối ưu qua nhiều context/vị trí/backbone | Hướng context robustness mình đề xuất có overlap lớn; phải so trực tiếp trước khi claim | Abstract và phần phương pháp tổng quan |
| [EvoMal, 2026](https://arxiv.org/abs/2608.25776) | Skill độc sinh hậu duệ, tiếp tục sau xoá seed | Persistence sau cleanup không đủ mới; skill store khác episodic store nhưng đổi domain chưa đủ | Abstract |
| [MemSecBench, 2026](https://arxiv.org/abs/2607.27080) | Đánh giá persistence, consequence và repair | Vòng đời memory đã là đối tượng benchmark | Trang abstract |
| [MemPoison, 2607.14651](https://arxiv.org/abs/2607.14651) | Benchmark corruption đơn lẻ, kết hợp nhiều record và dormant trigger | Chỉ ghép nhiều memory thành attack chưa đủ | Abstract |

**Có hai bài cùng tên MemPoison**; luôn ghi ID. Với hướng chính mới, thứ tự đọc sâu/tái lập
là AgentPoison → Phantom/PoisonedRAG theo đúng task → InjecMEM → PoisonArena.
MemPoison 2605.29960 trở thành baseline bắt buộc khi thêm selective writer.

Hướng hybrid/BM25/reranker cũng chưa thể gọi là gap trống. MemPoison §6 để mở hybrid,
nhưng [Semantic Chameleon](https://arxiv.org/abs/2603.18034) đã xét tối ưu sparse+dense.
Đây là ví dụ một limitation của một bài không đồng nghĩa toàn lĩnh vực chưa giải quyết.
Các kết quả ASR giữa các paper không xếp hạng trực tiếp được: khác quyền attacker,
query, số poison, model, metric và số lần tương tác.

## 4. Repo hiện có gì và nút thắt nằm ở đâu

| Thành phần | Đã kiểm tra | Ý nghĩa |
|---|---|---|
| `algo/trigger_optimization.py` | Local chặn `target_gradient_guidance` ngoài agent `ad` | Chạy QA không bật target guidance chưa phải full target-aware baseline |
| `src/triggers/losses.py`, `hotflip_margin.py` | Đã có margin cho poison thứ K vượt clean tốt nhất | “Thêm retrieval margin” đã có code; dùng làm baseline, chưa phải method mới |
| `src/triggers/scorers.py` | GPT-2/target scorer dưới `no_grad()` | Đây là scoring/gate, không phải gradient loss xuyên qua LLM |
| `src/triggers/mcat/e2e.py` | Đọc top-k, chấm plan, đối chứng static, vòng ghi thật qua LLM | Tái sử dụng harness, không cần hồi sinh generator MCAT |
| `src/triggers/mcat/agent_ad.py` | Prompt chứa các experience theo hạng; `replace_plan` giữ reasoning cũ rồi đổi plan | Có thể kiểm tra sự nhất quán demonstration như một confound; chưa có bằng chứng đây là nguyên nhân chính |
| `src/agentpoison/strategyqa.py` | Chọn ngẫu nhiên một hit trong top-k | Không mặc định cùng cơ chế với harness AD đưa toàn bộ top-k vào prompt |
| `src/triggers/mcat/writeback.py` | Probe cũ dùng hit làm proxy; `e2e.py` mới chấm output thật | Không gộp số retrieval simulation với hành vi LLM |

Theo [run 8](../../results/p0_run8_results.md), Qwen p5: hit@5 **0.986**, target-plan rate
**0.396**, trong khi query có trigger nhưng không poison (`control`) **0.174**.
Vậy hiệu ứng so control khoảng **0.223**. Nút thắt trong thiết lập này nằm sau retrieval.
Không suy ra tất cả model hoặc task đều có cùng nút thắt.

Theo [run 6](../../results/p0_run6_rerun_results.md), tăng memory lành từ 2.000 lên 4.000 chỉ làm
hit@5 giảm khoảng 0.0004 ở arm random. Điều này không ủng hộ đầu tư generator thích nghi
chỉ để xử lý benign growth trong thiết lập đã thử. Theo run 7–8, cleanup p5 giữ target-plan
rate khoảng 0.959 ở Llama nhưng còn 0.270 ở Qwen; chưa có persistence mạnh xuyên model.

`asr_t` trong harness này chấm **Driving Plan do LLM sinh ra**, chưa phải hành động vật lý,
trajectory execution hay full-system AgentDriver ASR. Khi viết báo cáo phải nêu đơn vị đó.

Một số README đã cũ: ngày rà soát, `agentdriver/data/finetune/data_samples_train.json`
và `agentdriver/data/memory/database.pkl` có trên máy; không còn đúng khi nói toàn bộ AD
data vắng mặt. Có corpus StrategyQA; thiếu index ở đường dẫn ghi trong README và thiếu
file `ReAct/data/strategyqa/strategyqa_train.json`. Kiểm tra tồn tại chưa xác nhận data
đủ/đúng phiên bản. Không có raw artifact P0 để tái chấm ngay trong checkout này.

## 5. Cải tiến làm trước: lựa chọn poison demonstrations theo hành vi

**Câu hỏi pilot:** với cùng trigger, số bản ghi và tổng token, chọn bộ demonstration
bằng hiệu quả trên các context dev thực sự truy hồi được có tăng target-plan rate trên
query/context chưa thấy, so với chọn theo retrieval hoặc chọn ngẫu nhiên không?

Phương án tối thiểu chỉ thêm tầng chọn bộ demonstrations, dùng ngân hàng ứng viên hữu
hạn từ train. Đóng băng trigger trước để tách đóng góp của lựa chọn memory khỏi tối ưu
trigger. Context của mỗi ứng viên phải được dựng bằng retriever thật; không ép poison
vào prompt ở phép đo chính. Chỉ dùng hoán đổi rank/giữ retrieval cố định ở ablation cơ chế.
Chấm trên dev, khoá lựa chọn rồi mới test. Mọi lời gọi dùng để tạo/chọn ứng viên đều tính
vào search budget.

Nên bắt đầu bằng harness AD có sẵn để kiểm tra cơ chế; mục tiêu chỉ là output trong testbed.
QA làm domain thứ hai sau khi xử lý khác biệt random-one-hit so với all-top-k. Nếu đổi
retrieval assembly của QA, ghi đó là một victim variant riêng, không gọi là tái lập gốc.

| Arm | Mục đích |
|---|---|
| A: AgentPoison đầy đủ, có target-generation constraint | Baseline chính; tránh thắng bản tái lập thiếu thành phần |
| B: AgentPoison + margin hiện có | Tách lợi ích retrieval ranking |
| C: Chọn random/best-of-N với cùng tổng ngân sách | Tách lợi ích chỉ do thử nhiều candidate |
| D: Chọn demonstrations theo hành vi qua nhiều context dev | Ứng viên cải tiến |

Cùng số poison, cap token, split, seed khởi tạo và ngân sách tương đương. Arm best-of-N
là đối chứng quan trọng: nếu D chỉ hơn A vì được gọi LLM nhiều hơn thì chưa có đóng góp.
Không so D với A mà âm thầm đổi cả payload length, trigger length hoặc quyền attacker.

Thứ tự pilot:

1. Chốt threat model AgentPoison: quyền ghi một lần, quyền kích hoạt query công bố riêng;
   không tự chuyển thành query-only hoặc trigger-free. Chốt data/split và target hợp lệ
   (loại hoặc báo riêng các query mà target vốn là đáp án đúng).
2. Đo baseline và no-poison control trên dev. Phân loại lỗi retrieval, không làm theo,
   parse/format. Nếu retrieval đã bão hoà, không dành phần lớn budget để nâng hit@5.
3. So A–D trên memory tĩnh, số poison nhỏ và mixed context. Candidate selection chỉ dùng
   train/dev; giữ query families và poison source của test tách khỏi search.
4. Đo target-plan rate, ACC sạch, false activation, hit@1/@k, poison occupancy, conditional
   adoption và chi phí. Báo raw ASR cùng chênh so no-poison control; không gọi hiệu đó là
   một metric chuẩn của paper gốc.
5. Nếu có tín hiệu, kiểm trên model thứ hai rồi mới thêm write-back với attack đã đóng băng.
   Phép thử động phải có no-write và no-poison với cùng stream/policy; không chỉ dùng
   stream 100% query có trigger để đại diện vận hành tự nhiên.

**Cổng đề xuất cho pilot mới**, chốt trước khi chạy: tăng ≥5 điểm phần trăm target-plan
rate trên held-out, CI ghép cặp theo episode không chạm 0; clean ACC giảm ≤1 điểm so
baseline, không tăng excess false activation quá ngưỡng đã khoá trên dev. Báo cả hiệu
ứng tuyệt đối và chi phí; các ngưỡng này là lựa chọn thực nghiệm, không là chuẩn phổ quát.
Không sửa ngưỡng của run 1–8. Pilot nhỏ chỉ sàng lọc, không thay kiểm chứng nhiều seed/model.

**Dừng claim method** nếu lợi ích biến mất khi cân bằng budget, chỉ do sửa lỗi template
hoặc thiếu target gate, hoặc không thắng best-of-N. Giữ các sửa đó như cải tiến engineering.
“Tối ưu hành vi”, “robust context” và “one-shot poison” đều đã có prior art; cần phân biệt
cụ thể với InjecMEM/Phantom trước khi định vị công bố. Chưa hứa hướng này đủ Q1/A*.

## 6. Thứ tự đầu tư được đề xuất

1. **Ngay bây giờ:** AgentPoison + pilot lựa chọn demonstrations trên memory tĩnh.
2. **Sau khi thắng static baseline:** memory động như bài kiểm tra transfer, reuse writer hiện có.
3. **Chưa ưu tiên:** generator conditioning theo memory, persistence sau cleanup làm claim độc lập,
   memory graph/multimodal, adaptive recovery attack. Chúng tăng chi phí hoặc overlap lớn.
4. **Defense:** để sau khi xác nhận cơ chế; [A-MemGuard](https://arxiv.org/abs/2510.02373)
   là peer liên quan. Yêu cầu hiện tại không chốt chuyển sang defense.

Đây là quyết định chọn bước thử tiếp theo, không phải kết luận khoa học rằng memory tĩnh
mới hơn memory động, hay attack đề xuất đã hiệu quả.
