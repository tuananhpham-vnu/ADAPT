# `src/triggers/` — nghiên cứu tấn công của dự án

`algo/` giữ phần tái lập AgentPoison upstream. Mọi thứ ở đây là code do dự án viết,
xây trên nó. **Chiều phụ thuộc: `src/` import `algo/`, không bao giờ ngược lại.**

Mỗi hướng tấn công một thư mục con. Phía phòng thủ hiện ở `src/aqua/` và `src/adapt/`
(dự kiến gom về `src/defend/`).

## Bố cục

```
src/triggers/
  losses.py        L_uni, L_cpt, retrieval margin (định nghĩa chiếm trọn top-K)
  clustering.py    fit_centers: GMM 5 component, full covariance, random_state=0
  scorers.py       GPT-2 coherence, Llama target probability, candidate gate
  artifacts.py     atomic write, sha256/stable hash, RNG save-restore
  margin.py        pipeline AgentPoison + retrieval margin, 6 stage, resume được
  hierarchy/       trigger universal / theo nhóm / từng câu
  specificity/     độ bám query và chất lượng ngôn ngữ của trigger
  mcat/            generator sinh trigger theo trạng thái memory
```

Bốn module ở gốc là phần dùng chung: `hierarchy/`, `margin.py` và `mcat/` đều gọi tới
chúng. Đặt cạnh nhau để cả ba dùng đúng một định nghĩa loss và một contract artifact.

## Trạng thái từng hướng

| Hướng | Trạng thái | Tài liệu |
|---|---|---|
| `margin.py` | Pipeline đủ 6 stage, cần 2 GPU (DPR+GPT-2 `cuda:0`, Llama `cuda:1`) | [`_guidance/19`](../../_guidance/19_agentpoison_margin_implementation.md) |
| `mcat/` | M0–M2 xong, 63 test xanh, **chưa có số thật** | [README](mcat/README.md), [`_guidance/20`](../../_guidance/20_mcat_kaggle_runbook.md) |
| `hierarchy/` | Có kết quả pilot DPR thật (**kết quả âm**); import đã chạy lại sau khi khôi phục `assign` | [README](hierarchy/README.md), [`_idea`](../../_idea/trigger_hierarchy_pilot_results.md) |
| `specificity/` | Có kết quả, chưa đo ASR; import đã chạy lại, có checkpoint theo query | [README](specificity/README.md), [`_idea`](../../_idea/increase_specificity_results.md) |

### `assign` đã được khôi phục

Cả hai module gọi `assign` từ `clustering.py`. Hàm đó bị mất ở commit `6caa991a` và
chưa từng tồn tại trong lịch sử git, nên hai module không import được. Nay đã khôi
phục trong `clustering.py` theo đúng chữ ký suy ra từ hai chỗ gọi
(`hierarchy/evaluation.py:17` và `hierarchy/experiment.py:25`):

```python
def assign(vectors, centers) -> torch.Tensor:
    """Nearest-center index cho từng hàng: LongTensor [rows]."""
    return torch.cdist(x, c).argmin(dim=1).long()
```

Dùng khoảng cách Euclid vì centers là GMM means chưa chuẩn hóa. Với nhánh per_query,
centers là chính các query sạch đã L2-normalize, khi đó Euclid và cosine cho cùng
kết quả. Đây là **khôi phục theo suy luận**, không phải bản gốc của tác giả.

## Chạy

```powershell
.\make.ps1 trigger-hierarchy-smoke   # src.triggers.hierarchy smoke
.\make.ps1 trigger-specificity       # src.triggers.specificity
.\make.ps1 trigger-language-audit    # src.triggers.hierarchy.language_audit
.\make.ps1 mcat-smoke                # src.triggers.mcat smoke --fixture
```

```bash
python -m src.triggers.margin smoke --output-dir outputs/agentpoison_margin/smoke
bash scripts/run_mcat_kaggle.sh preflight
```

## Ba ranh giới không được lẫn

1. **Hai định nghĩa margin khác nhau.** `losses.compute_retrieval_margin_loss` là *chiếm
   trọn top-K* (poison thứ K vượt clean tốt nhất). `mcat/objectives.compute_hit_at_k_margin_loss`
   là *ít nhất một poison vào top-K*. Không báo cáo cái này dưới tên cái kia.
2. **`scorers.py` không nằm trong đường gradient.** GPT-2 và Llama đều chạy dưới `no_grad()`;
   chúng là sampler và feasibility gate, không phải term của loss.
3. **Ba khái niệm cụm khác nhau**, theo [`_idea/group_conditioned_triggers.md`](../../_idea/group_conditioned_triggers.md):
   benign reference cluster (`fit_centers`), query routing group (`hierarchy/`), và
   triggered embedding cluster. Đừng dùng chung tên biến cho chúng.
