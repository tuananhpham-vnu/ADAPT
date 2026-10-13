# `src/attack/` — nghiên cứu tấn công của dự án

`algo/` giữ phần tái lập AgentPoison upstream. Mọi thứ ở đây là code do dự án viết,
xây trên nó. **Chiều phụ thuộc: `src/` import `algo/`, không bao giờ ngược lại.**

Mỗi hướng tấn công một thư mục con. Phía phòng thủ ở `src/defense/` (hiện có `aqua/`);
cải tiến ARTEMIS ở `src/artemis/`.

## Bố cục

```
src/attack/
  margin/          loss_margin để cải thiện AgentPoison
    margin.py          pipeline AgentPoison + retrieval margin, 6 stage, resume được
    hotflip_margin.py  vòng hotflip của algo/ giữ nguyên, loss chuẩn hoá uni/cpt/margin (Σw = 1)
    qa_retrieval_eval.py  chấm retrieval của trigger QA
    losses.py          L_uni, L_cpt, retrieval margin (định nghĩa chiếm trọn top-K)
    scorers.py         GPT-2 coherence, Llama target probability, candidate gate
    clustering.py      fit_centers: GMM 5 component, full covariance, random_state=0
    artifacts.py       atomic write, sha256/stable hash, RNG save-restore, ROOT của repo
  dilution/        trigger bị pha loãng thế nào khi memory có thêm query+trigger với action sạch
                   (probe tĩnh self-n/rival-n, write-back log_outcome/verified/corrected, e2e với LLM);
                   tên cũ mcat/
```

**Phụ thuộc chéo:** `dilution/` import `margin/artifacts.py` (IO, hash, `ROOT`),
`margin/clustering.py` (`fit_centers`) và `margin/losses.py` (L_uni, L_cpt). Sửa ba file này
thì phải chạy lại test của cả hai hướng.

`hierarchy/` và `specificity/` đã xoá ngày 2026-10-13 (pilot kết thúc, kết quả ở
`_idea_q1_aplus/archive/`); lấy lại từ git nếu cần.

## Trạng thái từng hướng

| Hướng | Trạng thái | Tài liệu |
|---|---|---|
| `margin/margin.py` | Pipeline đủ 6 stage, cần 2 GPU (DPR+GPT-2 `cuda:0`, Llama `cuda:1`) | [`_guidance/19`](../../_guidance/19_agentpoison_margin_implementation.md) |
| `margin/hotflip_margin.py` | Chỉ agent `qa`. Vòng hotflip của `algo/trigger_optimization.py`, tối thiểu hoá tổ hợp lồi `w_uni·l_uni + w_cpt·l_cpt + w_margin·l_margin` (Σw = 1, mặc định 0.8/0.1/0.1). `l_uni`, `l_cpt` chia cho khoảng cách trung bình DB sạch → tâm GMM để cùng thang với margin (cosine). Tỉ lệ `1 : 0.1 : 0` chấp nhận đúng các flip như upstream. Trigger khởi tạo tokenize không kèm `[CLS]/[SEP]`, từ chối `[UNK]`. Log `Score`, `Loss`, `Loss/{uni,cpt,margin}` lên wandb. Đã chạy demo Kaggle end-to-end (3 iteration). Chạy: `bash scripts/react_strategyqa/run_optimization_margin.sh` | docstring của module |
| `dilution/` | Pipeline P0 run 1–8 (tĩnh, write-back, e2e Llama/Qwen). Phần generator MCAT (`generator.py`, `train.py`, `drift*.py`, `adapt.py`) vẫn còn trong thư mục nhưng đã dừng đầu tư | [README](dilution/README.md), [`results/`](../../_idea_q1_aplus/results/) |

## Chạy

```powershell
.\make.ps1 dilution-smoke            # src.attack.dilution smoke --fixture
```

```bash
python -m src.attack.margin.margin smoke --output-dir outputs/agentpoison_margin/smoke
bash scripts/run_p0_probes.sh preflight     # dilution: kiểm tra + CPU smoke
```

## Ba ranh giới không được lẫn

1. **Hai định nghĩa margin khác nhau.** `margin/losses.compute_retrieval_margin_loss` là *chiếm
   trọn top-K* (poison thứ K vượt clean tốt nhất). `dilution/objectives.compute_hit_at_k_margin_loss`
   là *ít nhất một poison vào top-K*. Không báo cáo cái này dưới tên cái kia.
2. **`scorers.py` không nằm trong đường gradient.** GPT-2 và Llama đều chạy dưới `no_grad()`;
   chúng là sampler và feasibility gate, không phải term của loss.
3. **Ba khái niệm cụm khác nhau**, theo [`_idea_q1_aplus/archive/group_conditioned_triggers.md`](../../_idea_q1_aplus/archive/group_conditioned_triggers.md):
   benign reference cluster (`fit_centers`), query routing group (`hierarchy/`, đã xoá), và
   triggered embedding cluster. Đừng dùng chung tên biến cho chúng.
