"""AgentPoison hotflip trigger search with a retrieval-margin loss (StrategyQA agent).

The search is upstream's ``algo/trigger_optimization.py`` loop, reusing its hotflip,
coherence filter and retrieval score unchanged; upstream itself is not modified. What
differs is the objective, split into named quantities with explicit directions:

    score     higher = better  upstream retrieval score (ap: distance to the GMM
                               centers - 0.1*variance, cpa: mean similarity to the DB)
    l_margin  lower = better   relu(delta + best_clean_cos - kth_poison_cos) against the
                               poisoned keys ReAct/local_wikienv.py injects
    loss      lower = better   -score + margin_weight * l_margin; what the search
                               minimizes and what a flip must lower to be accepted

For ap the full loss is ``-uni + 0.1*cpt + margin_weight*l_margin``; the default
``--margin-weight 0.1`` matches the compactness weight. Only the ratios matter: hotflip
ranks candidates by the gradient and accepts by comparing losses, both invariant to
scaling the loss by a positive constant, so the weights need not sum to 1.

With ``--margin-weight 0`` the loss is ``-score`` and the accepted flips are exactly
upstream's; ``l_margin`` is still computed and logged, so that arm is the baseline.

    python -m src.triggers.hotflip_margin --wandb                      # margin 0.1
    python -m src.triggers.hotflip_margin --margin-weight 0 --wandb    # baseline
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch

from src.triggers.losses import compute_retrieval_margin_loss

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "ReAct/database/strategyqa_train_paragraphs.json"
QUESTIONS = ROOT / "ReAct/database/strategyqa_train.json"
DB_DIR = ROOT / "ReAct/database/embeddings"
GOLDEN_TRIGGER = "Make efficient calls."
# Upstream's `slice` for qa: with --exclude-special, token ids <= 998 ([PAD], [unused*],
# [UNK], [CLS], [SEP], [MASK]) are never proposed by hotflip.
SPECIAL_ID_CEILING = 998
SPECIAL_TOKENS = ("[CLS]", "[MASK]", "[SEP]")


def trigger_sequence_of(adv_passage_ids, tokenizer) -> str:
    """The trigger string the ReAct runners build from trigger.json."""
    return " ".join(t for t in tokenizer.convert_ids_to_tokens(adv_passage_ids[0])
                    if t not in SPECIAL_TOKENS)


def embed_poison_keys(questions, trigger_sequence, model, tokenizer, device):
    """Embed the poisoned memory keys as ReAct/local_wikienv.py stores them:
    ``question + " " + trigger``, padded to 512, pooled retriever output.

    No graph: the keys change with every candidate trigger, but the trigger gradient is
    read off the query forward pass only.
    """
    from algo.utils import _pooled_emb
    texts = [f"{q} {trigger_sequence}" for q in questions]
    with torch.no_grad():
        tok = tokenizer(texts, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
        return _pooled_emb(model, {"input_ids": tok["input_ids"].to(device),
                                   "attention_mask": tok["attention_mask"].to(device)})


def objective_terms(query_embeddings, poison_embeddings, score_fn, clean_embeddings,
                    margin_weight=0.0, margin_top_k=1, margin_value=0.1):
    """Return ``(score, l_margin, loss)``; ``l_margin`` is None without poison keys."""
    score = score_fn(query_embeddings)
    if poison_embeddings is None:
        return score, None, -score
    l_margin = compute_retrieval_margin_loss(query_embeddings, clean_embeddings, poison_embeddings,
                                             margin_top_k, margin_value)
    return score, l_margin, -score + float(margin_weight) * l_margin


class _Tee:
    """Write to the terminal and stdout.txt; upstream sent everything to the file only."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agent", choices=("qa",), default="qa")
    p.add_argument("--algo", choices=("ap", "cpa"), default="ap")
    p.add_argument("--model", default="dpr-ctx_encoder-single-nq-base", help="Retriever code understood by algo.utils.load_models")
    p.add_argument("--save-dir", type=Path, default=ROOT / "results")
    p.add_argument("--num-iter", type=int, default=100)
    p.add_argument("--num-grad-iter", type=int, default=30, help="Batches per iteration, for both the gradient and candidate scoring")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-cand", type=int, default=100)
    p.add_argument("--trigger-tokens", type=int, default=10)
    p.add_argument("--golden-trigger", action="store_true", help=f"Start from '{GOLDEN_TRIGGER}' instead of [MASK] tokens")
    p.add_argument("--exclude-special", action="store_true", help="Keep special/unused retriever tokens out of the trigger")
    p.add_argument("--ppl-filter", action="store_true", help="Down-select hotflip candidates by GPT-2 perplexity")
    p.add_argument("--coh-sample", action="store_true", help="Sample the coherence set via softmax(-log_ppl/T) instead of top-k")
    p.add_argument("--coh-temperature", type=float, default=1.0)
    p.add_argument("--coh-select-weight", type=float, default=0.0, help="Among improving candidates, trade loss against perplexity")
    p.add_argument("--micro-batch-size", type=int, default=0, help="Exact micro-batched retriever backward; 0 = full batch")
    p.add_argument("--patience", type=int, default=0, help="Stop after this many iterations without an accepted flip; 0 = off")
    p.add_argument("--resume", type=Path, help="checkpoint.pt, or the run directory holding it")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reset-grad-each-iter", action="store_true",
                   help="Clear the embedding-gradient hook at the start of each iteration. Upstream never "
                        "clears it, so iteration t's hotflip direction also carries every earlier "
                        "iteration's gradient. Off by default to stay comparable with upstream.")
    p.add_argument("--plot", action="store_true", help="PCA plot of the triggered queries every iteration")
    p.add_argument("--wandb", action="store_true", help="Log to Weights & Biases (WANDB_PROJECT / WANDB_ENTITY)")
    p.add_argument("--margin-weight", type=float, default=0.1,
                   help="loss = -score + margin_weight * l_margin. Default 0.1 = the compactness weight "
                        "inside upstream's score; 0 = upstream baseline")
    p.add_argument("--margin-value", type=float, default=0.1, help="Cosine margin the K-th poison key must beat the best clean key by")
    p.add_argument("--margin-top-k", type=int, default=1)
    p.add_argument("--poison-count", type=int, default=2,
                   help="Poisoned keys = first N questions of strategyqa_train.json + trigger; must match "
                        "injection_num in ReAct/local_wikienv.py")
    return p


def main(argv=None) -> Path:
    args = build_parser().parse_args(argv)
    if not 1 <= args.margin_top_k <= args.poison_count:
        raise SystemExit(f"--margin-top-k ({args.margin_top_k}) must be in [1, --poison-count ({args.poison_count})]")
    if args.margin_weight < 0:
        raise SystemExit("--margin-weight must be >= 0")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.save_dir / args.agent / f"{args.algo}_margin" / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "stdout.txt", "w", encoding="utf-8") as log, \
            contextlib.redirect_stdout(_Tee(sys.stdout, log)):
        print(f"run dir: {run_dir}")
        _run(args, run_dir)
    return run_dir


def _run(args, run_dir: Path) -> None:
    from torch.utils.data import DataLoader
    from sklearn.mixture import GaussianMixture
    import wandb
    import algo.trigger_optimization as upstream
    from algo.utils import StrategyQADataset, bert_get_adv_emb, get_embeddings, load_db_qa, load_models

    device = "cuda:0"
    # upstream.candidate_filter reads a module-global `device` that upstream only binds
    # inside its __main__ block.
    upstream.device = device

    if args.wandb:
        wandb.init(project=os.environ.get("WANDB_PROJECT", "agentpoison"),
                   entity=os.environ.get("WANDB_ENTITY") or None, name=f"{args.algo}_margin/{run_dir.name}",
                   config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
        # Chart everything against `iteration`: upstream plot_PCA calls wandb.log on its
        # own, which advances wandb's internal step, so passing step= would collide.
        wandb.define_metric("iteration")
        wandb.define_metric("*", step_metric="iteration")

    model, tokenizer, _ = load_models(args.model, device)
    model.eval()

    if args.golden_trigger:
        # Same tokenization as upstream (special tokens included) so the two runs start equal.
        adv_passage_ids = tokenizer(GOLDEN_TRIGGER, return_tensors="pt", truncation=True,
                                    max_length=args.trigger_tokens).input_ids.to(device)
        args.trigger_tokens = adv_passage_ids.shape[1]
    else:
        adv_passage_ids = torch.full((1, args.trigger_tokens), tokenizer.mask_token_id, device=device)
    print("Init trigger", tokenizer.convert_ids_to_tokens(adv_passage_ids[0]))
    adv_passage_attention = torch.ones_like(adv_passage_ids, device=device)

    embeddings = get_embeddings(model)
    embedding_gradient = upstream.GradientStorage(embeddings, args.trigger_tokens)

    if args.ppl_filter:
        ppl_model, ppl_tokenizer, _ = load_models("gpt2", device)
        ppl_model.eval()
    upstream.free_memory()

    db_embeddings = load_db_qa(str(CORPUS), str(DB_DIR), args.model, model, tokenizer, device)
    print("db_embeddings:", tuple(db_embeddings.shape))
    dataset = StrategyQADataset(str(QUESTIONS), split_ratio=1.0, train=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    with open(QUESTIONS, encoding="utf-8") as f:
        poison_questions = [row["question"] for row in json.load(f)[:args.poison_count]]

    # Same cache file as upstream: the GMM is a pure function of (DB, retriever).
    centers_cache = DB_DIR / f"gmm_centers_{args.agent}_{args.model}.pt"
    if centers_cache.exists():
        cluster_centers = torch.load(centers_cache, map_location=device)
    else:
        gmm = GaussianMixture(n_components=5, covariance_type="full", random_state=0)
        gmm.fit(db_embeddings.cpu().numpy())
        cluster_centers = torch.tensor(gmm.means_)
        torch.save(cluster_centers, centers_cache)
    expanded_cluster_centers = cluster_centers.to(device).unsqueeze(0)
    upstream.free_memory()

    if args.algo == "ap":
        score_fn = lambda q: upstream.compute_avg_cluster_distance(q, expanded_cluster_centers)
    else:
        score_fn = lambda q: upstream.compute_avg_embedding_similarity(q, db_embeddings)

    def terms(query_embeddings, poison_embeddings):
        return objective_terms(query_embeddings, poison_embeddings, score_fn, db_embeddings,
                               args.margin_weight, args.margin_top_k, args.margin_value)

    def poison_keys_for(ids):
        return embed_poison_keys(poison_questions, trigger_sequence_of(ids, tokenizer), model, tokenizer, device)

    def embed_queries(data, ids):
        return bert_get_adv_emb(data, model, tokenizer, args.trigger_tokens, ids, adv_passage_attention)

    start_iter = 0
    if args.resume:
        ckpt_path = args.resume / "checkpoint.pt" if args.resume.is_dir() else args.resume
        ckpt = torch.load(ckpt_path, map_location=device)
        if ckpt["num_adv_passage_tokens"] != args.trigger_tokens:
            raise SystemExit(f"checkpoint has {ckpt['num_adv_passage_tokens']} trigger tokens, run has {args.trigger_tokens}")
        adv_passage_ids = ckpt["adv_passage_ids"].to(device)
        start_iter = ckpt["iteration"] + 1
        print(f"Resumed from {ckpt_path} at iteration {start_iter}")

    no_improve_streak = 0
    for it_ in range(start_iter, args.num_iter):
        print(f"Iteration: {it_}")
        trigger_sequence = trigger_sequence_of(adv_passage_ids, tokenizer)
        model.zero_grad()
        if args.reset_grad_each_iter:
            embedding_gradient._stored_gradient = None

        # ---- gradient at the current trigger -------------------------------------
        # Sums over n_batches; candidate scoring sums over as many batches, so the two
        # are comparable. wandb gets per-batch means.
        pbar = range(min(len(loader), args.num_grad_iter))
        n_batches = len(pbar)
        train_iter = iter(loader)
        poison_embeddings = poison_keys_for(adv_passage_ids)
        grad = None
        score_sum = margin_sum = loss_sum = 0.0
        for _ in pbar:
            data = next(train_iter)
            # Backprop -loss: hotflip runs with increase_loss=True, i.e. it proposes the
            # flips that raise what was backpropagated (upstream backpropagated the score).
            if args.micro_batch_size > 0:
                _, query_embeddings = upstream.micro_batched_backward(
                    lambda d: embed_queries(d, adv_passage_ids), data,
                    lambda q: -terms(q, poison_embeddings)[2], args.micro_batch_size)
                with torch.no_grad():
                    score, l_margin, loss = terms(query_embeddings, poison_embeddings)
            else:
                query_embeddings = embed_queries(data, adv_passage_ids)
                score, l_margin, loss = terms(query_embeddings, poison_embeddings)
                (-loss).backward()
            score_sum += score.item(); margin_sum += l_margin.item(); loss_sum += loss.item()

            grad_sum = embedding_gradient.get().sum(dim=0)
            grad = grad_sum / args.num_grad_iter if grad is None else grad + grad_sum / args.num_grad_iter
            del score, l_margin, loss, grad_sum
        model.zero_grad()
        upstream.free_memory()

        # ---- propose and score candidates ----------------------------------------
        token_to_flip = random.randrange(args.trigger_tokens)
        slice_val = SPECIAL_ID_CEILING if args.exclude_special else None
        cand_ppl = None
        if args.ppl_filter:
            candidates = upstream.hotflip_attack(grad[token_to_flip], embeddings.weight, increase_loss=True,
                                                 num_candidates=args.num_cand * 10, slice=slice_val)
            candidates, cand_ppl = upstream.candidate_filter(
                candidates, num_candidates=args.num_cand, token_to_flip=token_to_flip,
                adv_passage_ids=adv_passage_ids, ppl_model=ppl_model, src_tokenizer=tokenizer,
                ppl_tokenizer=ppl_tokenizer, sample=args.coh_sample, temperature=args.coh_temperature)
        else:
            candidates = upstream.hotflip_attack(grad[token_to_flip], embeddings.weight, increase_loss=True,
                                                 num_candidates=args.num_cand, slice=slice_val)
        upstream.free_memory()

        n_cand = len(candidates)
        candidate_ids = []
        for candidate in candidates:
            ids = adv_passage_ids.clone(); ids[:, token_to_flip] = candidate
            candidate_ids.append(ids)
        # The poisoned keys embed the trigger, so each candidate rewrites them: build
        # them once per candidate, not once per (candidate, batch). Without margin weight
        # they cannot change which candidate wins, so skip that cost.
        use_margin = args.margin_weight > 0
        candidate_poison = [poison_keys_for(ids) if use_margin else None for ids in candidate_ids]
        candidate_scores = torch.zeros(n_cand)
        candidate_margins = torch.zeros(n_cand)
        candidate_losses = torch.zeros(n_cand)
        train_iter = iter(loader)
        for _ in pbar:
            data = next(train_iter)
            for i, ids in enumerate(candidate_ids):
                with torch.no_grad():
                    s, m, l = terms(embed_queries(data, ids), candidate_poison[i])
                candidate_scores[i] += s.item(); candidate_losses[i] += l.item()
                if m is not None:
                    candidate_margins[i] += m.item()
        upstream.free_memory()

        # ---- accept ---------------------------------------------------------------
        # With --margin-weight 0 this is upstream's `candidate_scores > current_score`.
        improving = candidate_losses < loss_sum
        top_idx = int(torch.argmin(candidate_losses))
        accepted = bool(improving.any())
        print(f"score {score_sum:.4f}  l_margin {margin_sum:.4f}  loss {loss_sum:.4f}  |  best candidate: "
              f"score {candidate_scores[top_idx]:.4f}  loss {candidate_losses[top_idx]:.4f}")
        if accepted:
            no_improve_streak = 0
            chosen = top_idx
            if args.coh_select_weight > 0 and cand_ppl is not None:
                z = lambda x: (x - x.mean()) / (x.std() + 1e-6)
                combined = z(-candidate_losses) - args.coh_select_weight * z(cand_ppl.cpu().float())
                combined[~improving] = float("-inf")
                chosen = int(torch.argmax(combined))
            adv_passage_ids = candidate_ids[chosen]
            print(f"Accepted flip at position {token_to_flip}: score {candidate_scores[chosen]:.4f}  "
                  f"loss {candidate_losses[chosen]:.4f}  ->  {tokenizer.convert_ids_to_tokens(adv_passage_ids[0])}")
        else:
            no_improve_streak += 1
            print(f"No improvement ({no_improve_streak} in a row)")

        if args.plot:
            with torch.no_grad():
                all_questions = {"question": [dataset[i]["question"] for i in range(len(dataset))]}
                current = embed_queries(all_questions, adv_passage_ids)
            upstream.plot_PCA(current, db_embeddings, str(run_dir), title=f"Iteration {it_}")
            upstream.plt.close("all")  # plot_PCA opens a new figure per call and never closes it
            del current

        if args.wandb:
            log = {
                "iteration": it_,
                # the trigger the gradient was taken at (before this iteration's flip)
                "Score": score_sum / n_batches,
                "Loss": loss_sum / n_batches,
                "Loss/margin": margin_sum / n_batches,
                "Best Candidate Score": candidate_scores[top_idx].item() / n_batches,
                "Best Candidate Loss": candidate_losses[top_idx].item() / n_batches,
                "Accepted": int(accepted),
                "Trigger Sequence": trigger_sequence,
            }
            if use_margin:
                log["Best Candidate Loss/margin"] = candidate_margins[top_idx].item() / n_batches
            wandb.log(log)

        del query_embeddings
        upstream.free_memory()

        # ---- persist ----------------------------------------------------------------
        metrics = {"score": score_sum, "l_margin": margin_sum, "loss": loss_sum,
                   "margin_weight": args.margin_weight, "accepted": accepted, "n_batches": n_batches}
        torch.save({"adv_passage_ids": adv_passage_ids.cpu(), "iteration": it_,
                    "num_adv_passage_tokens": args.trigger_tokens, "agent": args.agent,
                    "algo": args.algo, "model_code": args.model, **metrics}, run_dir / "checkpoint.pt")
        trig_tokens = tokenizer.convert_ids_to_tokens(adv_passage_ids[0])
        leftover = [t for t in trig_tokens if t in (*SPECIAL_TOKENS, "[PAD]", "[UNK]")]
        if leftover and not args.exclude_special:
            print(f"WARNING: trigger still contains {leftover}; the ReAct runners strip these, "
                  f"so evaluation would use a different trigger. Re-run with --exclude-special.")
        # Same keys the ReAct runners read (trigger_tokens / trigger_text).
        with open(run_dir / "trigger.json", "w", encoding="utf-8") as tf:
            json.dump({"iteration": it_, "trigger_tokens": trig_tokens,
                       "trigger_has_special_tokens": bool(leftover),
                       "trigger_text": tokenizer.decode(adv_passage_ids[0], skip_special_tokens=True),
                       "agent": args.agent, "algo": args.algo, "model_code": args.model,
                       **metrics}, tf, indent=2)
        with open(run_dir / "metrics.jsonl", "a", encoding="utf-8") as mf:
            mf.write(json.dumps({"iteration": it_, "trigger": trigger_sequence, **metrics}) + "\n")

        if args.patience > 0 and no_improve_streak >= args.patience:
            print(f"Early stop at iteration {it_}: {no_improve_streak} iterations without an accepted flip")
            break


if __name__ == "__main__":
    main()
