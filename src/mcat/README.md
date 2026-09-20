# MCAT — Memory-Conditioned Amortized Trigger Generation

Implementation of milestones M0–M2 of `_idea/memory_conditioned_generator_Q1_A_star.md`.
Read that document first: it holds the threat model, the research questions and the
go/pivot/stop criteria. This README only covers the code.

The research claim under test is **not** "a neural network can generate triggers by
backpropagation" — GBDA, AdvPrompter, AdvBDGen and Neural Exec already did that. It is
whether conditioning a generator on the *state of an agent's long-term memory* buys
anything over per-episode search, a universal trigger, or a trigger bank. M2 exists to
answer that, and it can answer "no".

## Status

| Milestone | Scope | State |
|---|---|---|
| M0 | Episodes, family split, encoding parity, CPU fixture | done |
| M1 | ST-Gumbel relaxation, direct-logit baseline B2, round-trip export | done |
| M2 | Conditioned generator, B3–B6, shuffled-context and permutation controls | done |
| M3–M5 | Memory drift, few-step adaptation, behavioral ASR, transfer | not started |

No real-retriever numbers have been produced yet. Everything so far is verified against
the CPU fixture encoder.

## Stages

```powershell
.\make.ps1 mcat-smoke --output-dir outputs/mcat/smoke   # every stage on CPU, no download
.\make.ps1 mcat-prepare --output-dir outputs/mcat/pilot --domain qa --domain ehr
.\make.ps1 mcat-index    --output-dir outputs/mcat/pilot
.\make.ps1 mcat-train    --output-dir outputs/mcat/pilot --mode generator
.\make.ps1 mcat-evaluate --output-dir outputs/mcat/pilot --split test
.\make.ps1 mcat-report   --output-dir outputs/mcat/pilot
```

Stages are resumable and hash-guarded exactly like `algo/agentpoison_margin.py`: changing
the configuration, the episode split or the retriever makes a resume fail loudly rather
than blending two runs.

Artifacts per run: `manifest.json`, `episodes.jsonl`, `checkpoint.pt`, `metrics.jsonl`,
`triggers.jsonl`, `training.json`, `evaluation.json`, `evaluation.jsonl`, `REPORT.md`.

## Modes and what each one is for

| Mode / variant | Plan ID | Question it answers |
|---|---|---|
| `--mode direct-logit` | B2 | What does gradient optimization alone buy, per episode? |
| `--mode universal-logit` | B3 | Is one universal trigger already enough? |
| `--mode generator --variant none` | B4 | Is the network just memorizing one solution? |
| `--mode generator --variant query` | B5 | How much comes from the query distribution? |
| `--mode generator --variant memory` | B6 | How much comes from the memory state? |
| `--mode generator --variant memory+query` | M1 | The method. |

All four generator variants have **identical parameter counts** — a dropped branch reads a
learned constant pseudo-set rather than being deleted — so the comparison isolates
information, not capacity.

`direct-logit` cannot transfer: at evaluation it re-optimizes on the evaluated split and
writes that run under `adapt-<split>/`, so its extra online cost stays visible.

## Module map

| File | Contents |
|---|---|
| `domains.py` | Per-agent adapters (`qa`, `ehr`, `ad`) exposing documents, queries and `family` |
| `episodes.py` | `Episode`, family-level outer split, per-domain size scaling, manifest |
| `cache.py` | Resumable memory-mapped clean-vector cache; reuses the repo's DPR index |
| `retrievers.py` | Frozen DPR bundle, and the CPU fixture encoder used by tests and smoke |
| `encoding.py` | `encode_with_trigger_embeddings` — the gradient-carrying encode path |
| `relaxation.py` | ST-Gumbel, vocabulary mask, deterministic export, round-trip report |
| `objectives.py` | `compute_hit_at_k_margin_loss`, total loss, fixed/refresh poison policy |
| `generator.py` | DeepSets/attention set encoders and the conditioned generator |
| `runtime.py` | `Workspace` / `EpisodeContext`: the glue that holds `Q_eval` back |
| `train.py` | The three training modes and the checkpoint contract |
| `evaluate.py` | Retrieval metrics, false activation, and the two mechanism controls |

## Three traps this code deliberately avoids

1. **The margin definitions are not interchangeable.**
   `algo.trigger_losses.compute_retrieval_margin_loss` optimizes *full top-K takeover*
   (K-th poison beats the best clean key). MCAT reports *at least one poison in top-K*,
   which is `objectives.compute_hit_at_k_margin_loss`. Reporting one under the other's
   name would overstate the attack.

2. **Coherence and target probability are not loss terms.**
   The GPT-2 and Llama scorers in `algo/constraint_scorers.py` run under `no_grad()`.
   They are a candidate sampler and a feasibility gate. Adding their values to a sum and
   calling the generator "trained on fluency" would be false.

3. **A gain on the relaxed objective is not a result.**
   Straight-through Gumbel is a biased estimator. Every trigger is exported by argmax,
   decoded, re-tokenized and re-scored on the ids the runtime would really produce
   (`round_trip_report`). If the gain does not survive that, it does not exist.

## Data availability

| Domain | Corpus | State |
|---|---|---|
| `qa` | `ReAct/database/strategyqa_train_paragraphs.json` (9 251 paragraphs) | available, with a prebuilt DPR index |
| `ehr` | `EhrAgent/database/ehr_logs/logs_final` (199 records) + `eicu_ac.json` | available but thin; episodes are scaled down and the reduction is recorded in the manifest |
| `ad` | `agentdriver/data/finetune/data_samples_train.json` | **absent** — only `split.json` is vendored. The adapter raises with a fetch instruction |

Until `ad` runs on real data, no result may be described as covering three agent domains.

## Running the checks

```powershell
.\.venv-adapt\Scripts\python.exe -m unittest tests.test_mcat tests.test_mcat_pipeline
.\.venv-adapt\Scripts\python.exe -m unittest tests.test_agentpoison_margin tests.test_package_layout
```

The second command matters because `algo/agentpoison_margin.py` now takes its atomic-write
and hashing helpers from `algo/run_artifacts.py`, shared with this package.

## Kaggle notes

`--fixture` is CPU-only and is for plumbing, not for numbers: under it the reference
centers are the first rows of the snapshot rather than a fitted GMM, because the
lightweight venv has no scikit-learn. Real runs must omit `--fixture`, which routes
through `algo.clustering.fit_centers` (5 full-covariance components, `random_state=0`) —
the same geometry as the AgentPoison baseline.

Keep DPR on `cuda:0` and any scorer on `cuda:1`, matching `_guidance/19`.
