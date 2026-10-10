# Backdoor qua vòng ghi memory: nghiên cứu đối kháng có kiểm soát

Ngày: **2026-10-10**. Trạng thái: đề xuất nghiên cứu, gap đang kiểm chứng.

Tên tiếng Anh dự kiến: **Backdoor Persistence and Dilution in Self-Updating Agent Memory**.

## 1. Ý tưởng chính và mục đích

Trong agent tự ghi kinh nghiệm vào memory, một lần đầu độc ban đầu có thể để lại
ảnh hưởng qua các bản ghi do chính agent sinh ra. Nghiên cứu kiểm tra khi nào ảnh
hưởng đó tồn tại, tăng lên hoặc bị pha loãng sau nhiều tương tác, kể cả sau khi
xoá bản ghi nguồn; từ đó thiết kế và đánh giá biện pháp ngăn tái sử dụng bản ghi sai.

**Hướng chính vẫn là attack-first:** tìm khoảng trống trong threat model và cách
đánh giá hiện có, xây dựng phương pháp đối kháng trong sandbox để kiểm chứng khoảng
trống đó, phân tích nguyên nhân, rồi phát triển defense trên cùng thiết lập.
Đích đến là hỗ trợ các model và hệ thống agent nếu rủi ro được xác nhận.

Đây là nghiên cứu bảo mật với hệ thống do nhóm quản lý hoặc được cho phép; không
nhằm gây hại, truy cập trái phép, lấy dữ liệu, phá hoại hay phát tán backdoor. Hành
vi đích trong thí nghiệm là nhãn sai hoặc marker vô hại, tool có tác động bên ngoài
được mô phỏng. Với AgentDriver, kết quả là quyết định trên benchmark offline,
không điều khiển phương tiện thật. Mỗi run có memory riêng và giới hạn số vòng.

## 2. Bản mô tả dùng khi trao đổi với trợ lý nghiên cứu

> Tôi đang nghiên cứu bảo mật cho LLM agents có memory tự cập nhật, trên testbed
> do tôi quản lý hoặc được phép đánh giá. Tôi muốn đi theo hướng attack-first:
> xác định literature gap, thiết kế thí nghiệm đối kháng có kiểm soát với hành vi
> đích vô hại, đo ảnh hưởng qua vòng truy hồi–hành động–ghi memory, rồi xây dựng
> và đánh giá defense. Câu hỏi là liệu ảnh hưởng của một lần đầu độc ban đầu có
> được agent duy trì qua các bản ghi tự sinh, hay bị pha loãng, và điều gì xảy ra
> sau khi xoá nguồn ban đầu. Hãy phân biệt giả thuyết với bằng chứng, kiểm tra
> novelty và giả định toán học, đồng thời báo cáo clean utility. Mục đích chỉ là
> nghiên cứu và hỗ trợ cải thiện an toàn của model/agent nếu vấn đề thực sự tồn tại;
> không triển khai trên hệ thống bên thứ ba hay dùng để gây hại.

Đoạn này mô tả đúng phạm vi công việc; không phải yêu cầu bỏ qua cơ chế an toàn.
Không thể bảo đảm một model khác sẽ chấp nhận mọi yêu cầu chỉ nhờ phần tuyên bố mục đích.

## 3. Threat model cần khóa trước

- **Đối tượng:** agent có retrieval memory và chính sách ghi sau tương tác.
- **Quyền đối kháng:** chèn một ngân sách bản ghi thử nghiệm tại thời điểm ban đầu;
  sau đó không sửa, thêm hoặc xoá memory bằng quyền của đối kháng. Một lần can thiệp
  không đồng nghĩa chỉ một bản ghi: báo riêng arm 1 và 5 seed.
- **Trigger:** arm chính phát lại stream query đã khóa trước. Báo rõ tần suất có
  trigger và ai kiểm soát nó. Nếu đối kháng tiếp tục gửi query có chủ đích thì đó
  là arm có tương tác tiếp diễn, không gọi là hoàn toàn rút khỏi hệ thống.
- **Vòng ghi:** các bản ghi hậu duệ phải do pipeline agent sinh và lưu thực sự.
  Harness tự gán nhãn hoặc sao chép key là mô phỏng riêng, không thay thế bằng chứng này.
- **Cleanup:** người đánh giá xoá toàn bộ seed theo ID tại mốc định trước, kiểm tra
  index/cache, rồi tiếp tục trên các bản ghi còn lại. Không tối ưu lại sau cleanup.
- **Mục tiêu đo:** hành vi sai trong sandbox, bản ghi hậu duệ có tác dụng, thời gian
  tồn tại, và chất lượng tác vụ sạch. Phạm vi là một memory, chưa suy ra lây giữa agent.

Phân biệt ba hiện tượng: **tồn tại** (seed vẫn có tác dụng), **ghi sai** (có bản ghi
sai mới), và **tự duy trì qua hậu duệ** (bản ghi tự sinh được dùng lại và tạo ảnh
hưởng/vòng ghi tiếp khi seed đã bị loại). Chỉ hiện tượng cuối hỗ trợ claim tự nhân bản.

## 4. Literature gap: thu hẹp trước khi thiết kế attack

Không nhận là mới các ý chung “memory động”, “poison một lần rồi tồn tại”, hoặc
“xoá nguồn chưa đủ”. Xem [bảng đối chiếu cập nhật](memory_poisoning_literature_2026.md).

Đặc biệt, [EVOMAL](https://arxiv.org/html/2608.25776) đã có tự lan qua skill do agent
sinh, mô hình phân nhánh và cách ly write-back có bảo đảm. Vì vậy ba thành phần
này không tự tạo thành novelty cho đề xuất hiện tại.

**Gap ứng viên:** trong episodic retrieval memory với trigger tối ưu, liệu có thể
giải thích và dự đoán ảnh hưởng qua nhiều vòng khi thứ hạng truy hồi, nội dung
đồng thuận và chính sách ghi cùng thay đổi, tốt hơn baseline hệ số cố định?
Đây là câu hỏi cần kiểm chứng, chưa phải tuyên bố “chưa ai làm”.

Ba nhánh đáng kiểm tra:

1. **Cơ chế nhân quả:** giữ nguyên query, seed và nội dung, can thiệp riêng thứ hạng,
   tỷ lệ bản ghi đồng thuận, và chính sách ghi. Tách tác động truy hồi khỏi tác động
   lên hành vi. So với memory tĩnh ở cùng ngân sách và stream.
2. **Khả năng dự đoán:** ước lượng mô hình trên các run phát triển, dự đoán chiều và
   thời gian suy giảm/tồn tại ở run held-out. Đo trường hợp cạnh tranh khiến một
   hệ số cố định dự đoán sai; không chỉ đặt tên mới cho tỷ lệ tăng trưởng quan sát.
3. **Defense vẫn cho phép học:** kiểm tra cách ghi/cách ly/thu hồi theo lineage khi
   bộ kiểm tra có sai số và bản ghi được tóm tắt. So với tắt ghi hoàn toàn, tính cả
   clean task accuracy, learning gain, tỷ lệ từ chối ghi và chi phí.

Nếu khác biệt chỉ còn dataset hoặc đổi tên “skill” thành “memory”, cần thu hẹp
thành nghiên cứu tái lập/đo lường; không mặc định đủ đóng góp cho A*.

## 5. Sửa công thức ngưỡng

Công thức ban đầu là tích ba xác suất nên nằm trong [0,1]; không thể dùng nó để
lập luận có miền **R > 1**. Nó cũng chỉ xét hạng 1, bỏ qua tác dụng của các hạng khác.

Có thể giữ phân rã xác suất để phân tích một cơ hội ghi:

```text
p_new(h) = P(E | h) × P(A | E,h) × P(W | A,E,h)
```

Trong đó `h` gồm trạng thái memory, query và chính sách; `E` là truy hồi có liên
quan, `A` là hành vi đích vô hại trong sandbox, `W` là ghi bản ghi sai có khả năng
được dùng lại. Đây là xác suất đường E→A→W; cần đo riêng các đường ghi sai khác.
Phân rã theo xác suất có điều kiện không đòi ba sự kiện độc lập. Hạng 1 và các hạng
2–k là các nhóm phân tích, không giả định nhóm sau luôn bằng 0.

Một hệ số sinh sản phù hợp hơn là:

```text
R_eff = E[số hậu duệ trực tiếp còn có khả năng tác động,
          quy cho một bản ghi trong vòng đời truy hồi đã định nghĩa]
```

Đại lượng này có thể lớn hơn 1 vì một bản ghi có nhiều cơ hội được dùng lại. Khi
xấp xỉ bằng số cơ hội × xác suất sinh hậu duệ, phải nêu giả định đồng nhất/ổn định.
Mỗi hậu duệ chỉ được đếm một lần; retrieval nhiều parent cần quy tắc attribution
hoặc đồ thị lineage và thí nghiệm loại parent, không cộng trùng.

Ngưỡng 1 chỉ có diễn giải phân nhánh quen thuộc dưới các giả định của mô hình.
`R_eff > 1` không bảo đảm mọi run tồn tại; cạnh tranh top-k, đồng thuận, xoá/TTL
và memory hữu hạn có thể phá giả định. **Không sinh hậu duệ mới cũng không đồng
nghĩa seed cũ đã mất tác dụng** nếu nó vẫn được lưu và truy hồi mãi.

Một mục tiêu lý thuyết cho defense là chứng minh, trong lớp threat model đã khóa:

```text
E[N_(t+1) | F_t] ≤ γ N_t, với 0 ≤ γ < 1
⇒ E[N_t] ≤ γ^t E[N_0]
⇒ P(N_t ≥ 1) ≤ γ^t E[N_0].
```

`N_t` đếm toàn bộ bản ghi có khả năng gây hành vi đích còn hoạt động, gồm seed;
`F_t` là lịch sử. Cần không có nguồn đầu độc mới và trạng thái 0 phải hấp thụ.
Đây là hệ quả toán học có điều kiện, **chưa phải guarantee của một defense đã xây**.
Phần khó là chứng minh chính sách thực sự đạt bất đẳng thức, tính cả bản ghi cũ
sống sót, sai số verifier và mọi kênh ghi thuộc mô hình. Không suy ra bảo vệ mọi
trigger từ một tập trigger thử nghiệm, hoặc bảo vệ toàn hệ thống từ chặn một kênh.

## 6. Bằng chứng hiện có và giới hạn diễn giải

Nguồn nội bộ: [run 6](results/p0_run6_rerun_results.md) và [run 7](results/p0_run7_e2e_results.md).
Các số sau được đối chiếu với báo cáo trong repo, chưa chạy lại từ raw outputs
trong lần chỉnh tài liệu này.

| Quan sát trong báo cáo | Diễn giải phù hợp | Chưa được kết luận |
|---|---|---|
| Run 7 v4, `self-1`: hit@5 không đổi, ASR-t giảm 0.314 | hit@5 đơn lẻ bỏ sót thay đổi hành vi; cần hit@1 và ASR | hit@1 đủ cho mọi model hoặc top-k |
| Run 7 v4 hoàn tất: 1 poison ASR-t 0.559; 5 poison 0.971 | so sánh p1/p5 đã có ở cùng phiên bản v4; gợi ý hiệu ứng số lượng seed | chưa tách tác động số lượng, coverage, rank và đồng thuận bằng can thiệp riêng |
| Run 7 p5 `log_outcome`: khoảng 48/50 bản ghi mang hành vi đích | output sai có thể được ghi nhiều trong arm này | 48/50 là R, R > 1, hoặc hậu duệ có khả năng tự duy trì |
| Run 7 p5 `verified`: khoảng 2.9 lần ghi/episode; ASR-t 0.836, verdict `stable` | chọn lọc khi ghi có thể để ảnh hưởng cũ tồn tại trong cửa sổ đo | verifier luôn thất bại hoặc memory đóng băng hoàn toàn |
| Run 7 p5 `corrected`: ASR-t 0.078 so với base 0.971 | sửa bản ghi làm giảm mạnh tác động trong thiết lập này | R = 0 hoặc phòng thủ tuyệt đối |
| Run 7 `rival-15`: ASR-t giảm 0.443; hành vi rival 0.316 | có cạnh tranh ở mức hành vi ngoài thay đổi hit@1 | hệ sinh thái nhiều backdoor tự lan đã được chứng minh |
| Run 6 `cleanup_hit` cao dưới `log_outcome` | persistence ở tầng retrieval trong mô phỏng | persistence end-to-end sau cleanup (xem dòng dưới) |
| Run 7 `log_outcome-50-cleanup`: ASR-t p5 0.959, p1 0.479 (≈ không cleanup 0.477) | xoá seed không làm yếu ảnh hưởng đã lan vào bản ghi tự sinh, ở cả hai arm | tự duy trì qua thế hệ hậu duệ tiếp theo; chỉ đo một mốc, một stream |
| Run 7 p1 `log_outcome`: `stable` (−0.082), không `reinforces` như lần 6 ở mức truy hồi | ghi lại output sai không tự động làm backdoor mạnh lên ở mức hành vi | điểm cân bằng ≈ tỉ lệ LLM làm theo poison: mới là giả thuyết |

Câu thay cho “Verification without correction freezes the infection”:
**“Trong thiết lập đã đo, chỉ lọc bản ghi mới mà không sửa hoặc loại bản ghi cũ
có thể khiến ảnh hưởng đối kháng tồn tại lâu.”** Run 6 arm p1 còn cho thấy
`verified` pha loãng; phải báo cả kết quả này.

## 7. Lộ trình attack trước, defense sau

1. **Khóa gap:** cập nhật bảng công trình gần nhất; viết claim khác biệt và phép
   thử có thể bác bỏ nó trước khi chạy. Bắt đầu bằng baseline đã có.
2. **Xây dựng attack thử nghiệm:** đề xuất phương pháp đối kháng trên testbed với
   ngân sách seed và thời điểm rút quyền rõ ràng. Mục tiêu đánh giá là tác động
   nhiều vòng và sau cleanup; mọi hành vi đích đều vô hại. So cùng chi phí và quyền.
3. **Kiểm chứng cơ chế:** tách seed/hậu duệ; chạy no-write, log-outcome, verified,
   corrected; có clean/poison × trigger off/on, stream trộn sạch và trigger,
   key nguyên văn so với key tóm tắt. Theo dõi ít nhất hai thế hệ hậu duệ.
4. **Đánh giá dự đoán:** so baseline hệ số cố định với mô hình phụ thuộc trạng thái;
   dùng run held-out, CI theo episode, phân biệt tác động đối kháng với lỗi nền.
5. **Thiết kế defense:** đánh giá ghi có sửa sai, cách ly có kiểm tra độc lập và
   thu hồi theo lineage; báo giả định về provenance, kiểm tra và kênh ghi. Oracle
   correction là mốc chặn trên, tách khỏi verifier thực tế có sai số.
6. **Kiểm chứng lại:** dùng đối kháng thích nghi trong cùng sandbox và ngân sách,
   báo utility/cost, failure cases, khả năng chuyển model/retriever. Nếu có vấn
   đề thực tế, chuẩn bị báo cáo tái lập có giới hạn và đề xuất khắc phục cho bên liên quan.

## 8. Tiêu chí giữ hoặc đổi hướng

- Không có hậu duệ tiếp tục gây tác động sau cleanup: bỏ claim tự duy trì; có thể
  giữ nghiên cứu về dilution và sai lệch giữa đo retrieval với hành vi.
- Không vượt baseline ở cùng quyền/ngân sách: không nhận là attack mới mạnh hơn.
- Mô hình mới không dự đoán tốt hơn baseline trên held-out: bỏ claim ngưỡng mới.
- Defense chỉ có hiệu quả khi tắt mọi ghi: báo rõ chi phí mất khả năng học.
- Không phân biệt được đóng góp với prior work: đổi câu hỏi, không đổi ngôn từ để
  làm hiện tượng cũ có vẻ mới.
