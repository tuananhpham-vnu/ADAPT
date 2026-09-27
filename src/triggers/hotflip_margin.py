"""AgentPoison hotflip trigger search with a normalized, weighted loss (StrategyQA agent).

The search is upstream's ``algo/trigger_optimization.py`` loop, reusing its hotflip and
coherence filter; upstream itself is not modified. The objective is a convex combination
of three terms brought to a common scale:

    l_uni     lower = better   -mean dist(triggered query, GMM center) / s
    l_cpt     lower = better    mean dist(triggered query, batch centroid) / s
    l_margin  lower = better    relu(delta + best_clean_cos - kth_poison_cos) against the
                                poisoned keys ReAct/local_wikienv.py injects
    loss      lower = better    w_uni*l_uni + w_cpt*l_cpt + w_margin*l_margin, sum(w) = 1

``s`` is the mean distance from the clean DB embeddings to the GMM centers, fixed once
per run. It makes l_uni/l_cpt dimensionless (l_uni = -1: the query sits as far from the
centers as an average clean passage); l_margin is a cosine gap and already is. Without
it the Euclidean terms (~7) would drown the margin (~0.05) whatever the weights.

Weights are renormalized to sum to 1. Hotflip only ranks and compares losses, so only
their ratios matter: upstream AgentPoison, loss = -(uni - 0.1*cpt), is the ratio
1 : 0.1 : 0, and ``--w-uni 1 --w-cpt 0.1 --w-margin 0`` accepts exactly its flips.
``score`` (upstream's raw retrieval score, higher = better) is still logged for reference.

    python -m src.triggers.hotflip_margin --wandb                                     # 0.8 / 0.1 / 0.1
    python -m src.triggers.hotflip_margin --w-uni 1 --w-cpt 0.1 --w-margin 0 --wandb  # upstream ratio
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

from src.triggers.losses import (
    compute_compactness_loss, compute_retrieval_margin_loss, compute_uniqueness_loss,
)

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "ReAct/database/strategyqa_train_paragraphs.json"
QUESTIONS = ROOT / "ReAct/database/strategyqa_train.json"
DB_DIR = ROOT / "ReAct/database/embeddings"
GOLDEN_TRIGGER = "Make efficient calls."
# Upstream's `slice` for qa: with --exclude-special, token ids <= 998 ([PAD], [unused*],
# [UNK], [CLS], [SEP], [MASK]) are never proposed by hotflip.
SPECIAL_ID_CEILING = 998
SPECIAL_TOKENS = ("[CLS]", "[MASK]", "[SEP]")
TERM_NAMES = ("score", "l_uni", "l_cpt", "l_margin", "loss")


def fmt_terms(sums, n_batches):
    return "  ".join(f"{name} {sums[name] / n_batches:+.4f}" for name in TERM_NAMES)


def initial_trigger_ids(tokenizer, text):
    """Tokenize the golden trigger without [CLS]/[SEP] and refuse special or [UNK] ids.

    Upstream tokenized it with special tokens, so [CLS]/[SEP] became trigger positions
    that hotflip could flip and the ReAct runners strip, and on the Kaggle image
    "Make" came out as [UNK], which the runners paste into the query verbatim. The
    retriever is uncased, so lowercasing first loses nothing.
    """
    ids = tokenizer(text.lower(), add_special_tokens=False).input_ids
    bad = [t for t, i in zip(tokenizer.convert_ids_to_tokens(ids), ids)
           if i in set(tokenizer.all_special_ids)]
    if bad:
        raise SystemExit(f"initial trigger {text!r} tokenizes to special/unknown tokens {bad} "
                         f"with {type(tokenizer).__name__}; pick another --initial-trigger")
    return ids


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


def normalized_weights(w_uni, w_cpt, w_margin):
    """Scale the three weights to sum to 1 (only their ratios matter to hotflip)."""
    weights = (float(w_uni), float(w_cpt), float(w_margin))
    if min(weights) < 0 or sum(weights) <= 0:
        raise ValueError(f"weights must be >= 0 with a positive sum, got {weights}")
    total = sum(weights)
    return tuple(w / total for w in weights)


def reference_scale(clean_embeddings, centers):
    """Mean distance from the clean DB to the GMM centers: the unit of l_uni and l_cpt."""
    return float(torch.cdist(clean_embeddings.float(), centers.float()).mean())


def objective_terms(query_embeddings, poison_embeddings, centers, clean_embeddings, scale,
                    weights, margin_top_k=1, margin_value=0.1):
    """Named terms for one batch; ``l_margin`` is None without poison keys.

    ``weights`` must already be normalized. ``score`` is upstream's raw ap score.
    """
    w_uni, w_cpt, w_margin = weights
    l_uni_raw = compute_uniqueness_loss(query_embeddings, centers)   # -mean dist to centers
    cpt_raw = compute_compactness_loss(query_embeddings)
    terms = {"score": -l_uni_raw - 0.1 * cpt_raw,
             "l_uni": l_uni_raw / scale, "l_cpt": cpt_raw / scale, "l_margin": None}
    loss = w_uni * terms["l_uni"] + w_cpt * terms["l_cpt"]
    if poison_embeddings is not None:
        terms["l_margin"] = compute_retrieval_margin_loss(query_embeddings, clean_embeddings, poison_embeddings,
                                                          margin_top_k, margin_value)
        loss = loss + w_margin * terms["l_margin"]
    terms["loss"] = loss
    return terms


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

    def isatty(self):
        # transformers' loading report asks; a file in the mix means no ANSI colours.
        return False

    def __getattr__(self, name):
        # Anything else a library probes (encoding, fileno, ...) comes from the terminal.
        return getattr(self.streams[0], name)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agent", choices=("qa",), default="qa")
    p.add_argument("--algo", choices=("ap",), default="ap")
    p.add_argument("--model", default="dpr-ctx_encoder-single-nq-base", help="Retriever code understood by algo.utils.load_models")
    p.add_argument("--save-dir", type=Path, default=ROOT / "results")
    p.add_argument("--num-iter", type=int, default=100)
    p.add_argument("--num-grad-iter", type=int, default=30, help="Batches per iteration, for both the gradient and candidate scoring")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-cand", type=int, default=100)
    p.add_argument("--trigger-tokens", type=int, default=10)
    p.add_argument("--golden-trigger", action="store_true", help="Start from --initial-trigger instead of [MASK] tokens")
    p.add_argument("--initial-trigger", default=GOLDEN_TRIGGER,
                   help="Text for --golden-trigger; its token count sets the trigger length")
    p.add_argument("--exclude-special", action="store_true", help="Keep special/unused retriever tokens out of the trigger")
    p.add_argument("--ppl-filter", action="store_true", help="Down-select hotflip candidates by GPT-2 perplexity")
    p.add_argument("--coh-sample", action="store_true", help="Sample the coherence set via softmax(-log_ppl/T) instead of top-k")
    p.add_argument("--coh-temperature", type=float, default=1.0)
    p.add_argument("--coh-select-weight", type=float, default=0.0, help="Among improving candidates, trade loss against perplexity")
    p.add_argument("--micro-batch-size", type=int, default=0, help="Exact micro-batched retriever backward; 0 = full batch")
    p.add_argument("--patience", type=int, default=0, help="Stop after this many iterations without an accepted flip; 0 = off")
    p.add_argument("--stage", choices=("all", "prep"), default="all",
                   help="'prep' only builds the DB-embedding and GMM caches, then exits; run it once "
                        "before launching several arms in parallel so they do not race on the cache files")
    p.add_argument("--resume", type=Path, help="checkpoint.pt, or the run directory holding it")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reset-grad-each-iter", action="store_true",
                   help="Clear the embedding-gradient hook at the start of each iteration. Upstream never "
                        "clears it, so iteration t's hotflip direction also carries every earlier "
                        "iteration's gradient. Off by default to stay comparable with upstream.")
    p.add_argument("--plot", action="store_true", help="PCA plot of the triggered queries every iteration")
    p.add_argument("--wandb", action="store_true", help="Log to Weights & Biases (WANDB_PROJECT / WANDB_ENTITY)")
    p.add_argument("--w-uni", type=float, default=0.8, help="Weight of l_uni (renormalized with the others to sum to 1)")
    p.add_argument("--w-cpt", type=float, default=0.1, help="Weight of l_cpt")
    p.add_argument("--w-margin", type=float, default=0.1, help="Weight of l_margin; 0 skips building candidate poison keys")
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
    try:
        args.weights = normalized_weights(args.w_uni, args.w_cpt, args.w_margin)
    except ValueError as exc:
        raise SystemExit(str(exc))
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # The weights in the name keep arms launched in the same second apart.
    run_dir = (args.save_dir / args.agent / f"{args.algo}_margin"
               / f"{stamp}_u{args.w_uni:g}_c{args.w_cpt:g}_m{args.w_margin:g}")
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "stdout.txt", "w", encoding="utf-8") as log, \
            contextlib.redirect_stdout(_Tee(sys.stdout, log)):
        print(f"run dir: {run_dir}")
        print("weights (uni, cpt, margin), normalized:", tuple(round(w, 4) for w in args.weights))
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

    if args.wandb and args.stage == "all":
        wandb.init(project=os.environ.get("WANDB_PROJECT", "agentpoison"),
                   entity=os.environ.get("WANDB_ENTITY") or None, name=f"{args.algo}_margin/{run_dir.name}",
                   config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
        # Chart everything against `iteration`: upstream plot_PCA calls wandb.log on its
        # own, which advances wandb's internal step, so passing step= would collide.
        wandb.define_metric("iteration")
        wandb.define_metric("*", step_metric="iteration")

    model, tokenizer, _ = load_models(args.model, device)
    model.eval()

    import transformers
    print(f"transformers {transformers.__version__}, {type(tokenizer).__name__}, "
          f"do_lower_case={getattr(tokenizer, 'do_lower_case', None)}; "
          f"'Is Paris the capital?' -> {tokenizer.tokenize('Is Paris the capital?')}")
    if args.golden_trigger:
        adv_passage_ids = torch.tensor([initial_trigger_ids(tokenizer, args.initial_trigger)], device=device)
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
    cluster_centers = cluster_centers.to(device).float()
    upstream.free_memory()
    if args.stage == "prep":
        print(f"--stage prep: caches ready under {DB_DIR}")
        return

    scale = reference_scale(db_embeddings, cluster_centers)
    print(f"reference scale s (mean clean-DB distance to GMM centers): {scale:.4f}")
    if args.wandb:
        wandb.config.update({"weights_normalized": list(args.weights), "reference_scale": scale})

    def terms(query_embeddings, poison_embeddings):
        return objective_terms(query_embeddings, poison_embeddings, cluster_centers, db_embeddings, scale,
                               args.weights, args.margin_top_k, args.margin_value)

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
        sums = dict.fromkeys(TERM_NAMES, 0.0)
        for _ in pbar:
            data = next(train_iter)
            # Backprop -loss: hotflip runs with increase_loss=True, i.e. it proposes the
            # flips that raise what was backpropagated (upstream backpropagated the score).
            if args.micro_batch_size > 0:
                _, query_embeddings = upstream.micro_batched_backward(
                    lambda d: embed_queries(d, adv_passage_ids), data,
                    lambda q: -terms(q, poison_embeddings)["loss"], args.micro_batch_size)
                with torch.no_grad():
                    batch = terms(query_embeddings, poison_embeddings)
            else:
                query_embeddings = embed_queries(data, adv_passage_ids)
                batch = terms(query_embeddings, poison_embeddings)
                (-batch["loss"]).backward()
            for name in TERM_NAMES:
                sums[name] += batch[name].item()

            grad_sum = embedding_gradient.get().sum(dim=0)
            grad = grad_sum / args.num_grad_iter if grad is None else grad + grad_sum / args.num_grad_iter
            del batch, grad_sum
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
        use_margin = args.weights[2] > 0
        candidate_poison = [poison_keys_for(ids) if use_margin else None for ids in candidate_ids]
        cand = {name: torch.zeros(n_cand) for name in TERM_NAMES}
        train_iter = iter(loader)
        for _ in pbar:
            data = next(train_iter)
            for i, ids in enumerate(candidate_ids):
                with torch.no_grad():
                    batch = terms(embed_queries(data, ids), candidate_poison[i])
                for name in TERM_NAMES:
                    if batch[name] is not None:
                        cand[name][i] += batch[name].item()
        upstream.free_memory()

        # ---- accept ---------------------------------------------------------------
        # With weights 1 : 0.1 : 0 this is upstream's `candidate_scores > current_score`.
        candidate_losses = cand["loss"]
        improving = candidate_losses < sums["loss"]
        top_idx = int(torch.argmin(candidate_losses))
        accepted = bool(improving.any())
        print("  current         " + fmt_terms(sums, n_batches))
        print("  best candidate  " + fmt_terms({k: v[top_idx].item() for k, v in cand.items()}, n_batches))
        if accepted:
            no_improve_streak = 0
            chosen = top_idx
            if args.coh_select_weight > 0 and cand_ppl is not None:
                z = lambda x: (x - x.mean()) / (x.std() + 1e-6)
                combined = z(-candidate_losses) - args.coh_select_weight * z(cand_ppl.cpu().float())
                combined[~improving] = float("-inf")
                chosen = int(torch.argmax(combined))
            adv_passage_ids = candidate_ids[chosen]
            print(f"Accepted flip at position {token_to_flip}: loss {candidate_losses[chosen] / n_batches:+.4f}"
                  f"  ->  {tokenizer.convert_ids_to_tokens(adv_passage_ids[0])}")
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
            # Per-batch means at the trigger the gradient was taken at (before this
            # iteration's flip). Loss/* are the unweighted normalized terms.
            log = {"iteration": it_, "Score": sums["score"] / n_batches, "Loss": sums["loss"] / n_batches,
                   "Loss/uni": sums["l_uni"] / n_batches, "Loss/cpt": sums["l_cpt"] / n_batches,
                   "Loss/margin": sums["l_margin"] / n_batches,
                   "Best Candidate Score": cand["score"][top_idx].item() / n_batches,
                   "Best Candidate Loss": cand["loss"][top_idx].item() / n_batches,
                   "Accepted": int(accepted), "Trigger Sequence": trigger_sequence}
            if use_margin:
                log["Best Candidate Loss/margin"] = cand["l_margin"][top_idx].item() / n_batches
            wandb.log(log)

        del query_embeddings
        upstream.free_memory()

        # ---- persist ----------------------------------------------------------------
        # Sums over n_batches, as compared during acceptance.
        metrics = {**sums, "weights": list(args.weights), "reference_scale": scale,
                   "accepted": accepted, "n_batches": n_batches}
        torch.save({"adv_passage_ids": adv_passage_ids.cpu(), "iteration": it_,
                    "num_adv_passage_tokens": args.trigger_tokens, "agent": args.agent,
                    "algo": args.algo, "model_code": args.model, **metrics}, run_dir / "checkpoint.pt")
        trig_tokens = tokenizer.convert_ids_to_tokens(adv_passage_ids[0])
        leftover = [t for t in trig_tokens if t in (*SPECIAL_TOKENS, "[PAD]", "[UNK]")]
        if leftover:
            print(f"WARNING: trigger contains {leftover}; the ReAct runners strip [CLS]/[MASK]/[SEP] "
                  f"and paste [UNK]/[PAD] into the query verbatim, so evaluation would not use this "
                  f"trigger. Use --golden-trigger and --exclude-special.")
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
