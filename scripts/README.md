# `scripts/` — Script chạy sẵn và tiện ích

Chạy mọi script từ **gốc repo**. Repo không có `Makefile`; lệnh tắt nằm trong `make.ps1`,
còn các script dưới đây giữ bộ tham số cố định của từng thí nghiệm.

## 1. Hướng chính: `src/attack/dilution/`

| Script | Làm gì |
|---|---|
| `run_p0_probes.sh` | Các probe P0/P1 và write-back: `preflight` (kiểm tra + CPU smoke), `r2`, `r1`, `position`, `contam`, `writeback`, `e2e` (agent AgentDriver với LLM thật, run 7–8). `RESUME=1` chạy tiếp sau khi Kaggle cắt 12h; `FIXTURE=1` chỉ kiểm tra đường ống, không phải kết quả. Runbook: [`_guidance/24`](../_guidance/24_p0_probe_kaggle_runbook.md) |
| `run_mcat.sh` | Pilot MCAT M0–M2 (generator + 7 arm baseline) trên StrategyQA/EhrAgent. Hướng generator đã dừng; giữ để tái lập. Runbook: [`_guidance/20`](../_guidance/20_mcat_kaggle_runbook.md) |

## 2. loss_margin: `src/attack/margin/` (agent `qa`)

| Script | Làm gì |
|---|---|
| `react_strategyqa/run_optimization_margin.sh` | Tối ưu trigger bằng `src.attack.margin.hotflip_margin` (loss uni/cpt/margin). |
| `react_strategyqa/run_optimization_validation.sh` | Pipeline `src.attack.margin.margin` có tập validation. |
| `react_strategyqa/run_margin_comparison.sh` | Lưới so baseline với margin-k1/k5 (`pilot`: dò trọng số margin; `full`: 3 seed), gọi lại `run_optimization_validation.sh`. |
| `react_strategyqa/prepare_dev_split.py` | Tách dev split StrategyQA cho inference. |
| `optimize_trigger_topk.py` | Tối ưu black-box nhỏ cho margin top-K trực tiếp poison vs clean. |
| `evaluate_trigger_full.py` | So trigger baseline và trigger đã tối ưu trên StrategyQA dev. |

## 3. Upstream AgentPoison theo agent

| Thư mục | Script | Làm gì |
|---|---|---|
| `react_strategyqa/` | `run_optimization.sh` | Tối ưu trigger cho agent `qa` bằng `algo/trigger_optimization.py`. |
| | `run_inference_gpt.sh`, `run_inference_llama2.sh`, `run_inference_llama3.sh` | Inference StrategyQA theo backbone. |
| `ehragent/` | `run_optimization.sh` | Tối ưu trigger cho agent `ehr`. |
| | `run_inference_gpt.sh`, `run_inference_llama3.sh` | Inference EhrAgent. |
| `agent_driver/` | `run_optimization.sh` | Tối ưu trigger cho agent `ad`. |
| | `run_inference.sh` | Inference Agent-Driver. |
| | `run_finetune.sh` | Fine-tune planner (GPT-3.5) cho Agent-Driver. |
| | `run_evaluation.sh` | Chấm L2 error và tỉ lệ va chạm. |

Mỗi script chỉ là một lời gọi `python ...` với bộ tham số cố định; mở ra đọc là thấy tham số.

## 4. Tiện ích

| File | Làm gì |
|---|---|
| `show_trigger.py` | In trigger cuối của lần tối ưu gần nhất trong `results/<agent>/<algo>/`. |
| `run_ablation_sweep.sh` | Lưới ablation cho `algo/trigger_optimization.py` (agent × embedder × setting), xếp job theo GPU còn trống. |
| `audit_embedding_clusters.py` | Audit phân cụm/chiếu trên embedding đã cache; không gọi model. |
| `check_adv_emb_equiv.py` | Kiểm `bert_get_adv_emb` bản batch khớp bản vòng lặp gốc. |
| `gpu_clean.sh` | Báo (hoặc `--kill`) tiến trình GPU còn sót sau khi tối ưu trigger bị dừng. |
| `make_help.py` | Di sản từ thời còn `Makefile`; hiện không dùng (`.\make.ps1 help` thay thế). |

```bash
bash scripts/run_p0_probes.sh preflight
python scripts/show_trigger.py --agent qa --algo ap
bash scripts/ehragent/run_inference_gpt.sh
```
