"""Resumable, stage-oriented AgentPoison + retrieval-margin pipeline.

Long Kaggle jobs are deliberately split into ``prepare``, ``index``, ``optimize``
and ``retrieval-eval``.  ``smoke`` uses a deterministic tensor fixture so the
artifact/resume contract can be verified without downloading gated models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from typing import Any, Iterable

import numpy as np
import torch

from algo.clustering import fit_centers
from algo.run_artifacts import (
    atomic_json, atomic_torch, git_metadata, read_json, restore_rng,
    rng_state, sha256_file, stable_hash, versions,
)

from algo.constraint_scorers import (
    GPT2CoherenceScorer, LlamaTargetScorer, sample_coherence_candidates,
    select_candidate,
)
from algo.trigger_losses import (
    compute_compactness_loss, compute_retrieval_margin_loss,
    compute_total_loss, compute_uniqueness_loss,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs/agentpoison_margin"
DEFAULT_CORPUS = ROOT / "ReAct/database/strategyqa_train_paragraphs.json"
DEFAULT_TRAIN = ROOT / "ReAct/database/strategyqa_train_filtered.json"
DEFAULT_DEV = ROOT / "ReAct/database/strategyqa_dev.json"
LOSS_DEFINITIONS = {
    "reference_centers": "GaussianMixture(n_components=5, covariance_type='full', random_state=0), fitted on all raw clean DPR embeddings",
    "convention": "minimize",
    "l_uni": "-mean(cdist(triggered_query, clean_cluster_center))",
    "l_cpt": "mean(norm(triggered_query - triggered_query_centroid))",
    "l_margin": "mean(relu(delta + best_clean_similarity - kth_poison_similarity))",
    "l_total": "l_uni + lambda_cpt*l_cpt + margin_weight*l_margin",
    "constraints": ["l_coh is used for sampling", "l_tar=-target_probability is a gate"],
}


# The implementations live in algo/run_artifacts.py so algo/ and src/mcat/
# share one atomic-write and hashing contract.  The private names are kept as
# aliases because they are referenced throughout this module and its tests.
_json = read_json
_atomic_json = atomic_json
_atomic_torch = atomic_torch
_sha256 = sha256_file
_stable_hash = stable_hash
_git_metadata = git_metadata
_versions = versions
_rng_state = rng_state
_restore_rng = restore_rng


def _question_rows(path: Path) -> list[dict[str, Any]]:
    rows = _json(path)
    seen: set[str] = set()
    result = []
    for row in rows:
        normalized = " ".join(row["question"].casefold().split())
        if normalized not in seen:
            seen.add(normalized)
            result.append(row)
    return result


def _corpus_texts(path: Path) -> list[str]:
    value = _json(path)
    if isinstance(value, dict):
        return [row["content"] for row in value.values()]
    return [row.get("content", row.get("text", "")) for row in value]



def _config(args: argparse.Namespace) -> dict[str, Any]:
    excluded = {"command", "resume", "stage", "func"}
    cli = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items() if key not in excluded and key != "hf_token"
    }
    cli["hf_token_configured"] = bool(args.hf_token)
    config = {
        "schema_version": 1, "cli": cli, "losses": LOSS_DEFINITIONS,
        "inputs": {name: {"path": str(path.resolve()), "sha256": _sha256(path)} for name, path in (
            ("corpus", args.corpus), ("train_questions", args.train_questions),
            ("dev_questions", args.dev_questions),
        )},
        "environment": _versions(), "git": _git_metadata(),
    }
    config["config_hash"] = _stable_hash(config["cli"] | {"losses": config["losses"], "inputs": config["inputs"]})
    return config


def prepare(args: argparse.Namespace) -> Path:
    run = args.output_dir
    if run.exists() and not args.resume:
        raise FileExistsError(f"{run} exists; use --resume or choose another --output-dir")
    run.mkdir(parents=True, exist_ok=True)
    
    train = _question_rows(args.train_questions)
    dev = _question_rows(args.dev_questions)
    train_ids, dev_ids = {str(row["qid"]) for row in train}, {str(row["qid"]) for row in dev}
    train_text = {" ".join(row["question"].casefold().split()) for row in train}
    dev_text = {" ".join(row["question"].casefold().split()) for row in dev}
    
    if train_ids & dev_ids or train_text & dev_text:
        raise ValueError("train/dev overlap detected by QID or normalized question")
    
    rng = random.Random(args.seed)
    order = list(range(len(train)))
    rng.shuffle(order)
    source = [train[i] for i in order[:args.poison_count]]
    remaining = [train[i] for i in order[args.poison_count:]]
    validation_size = min(args.validation_size, max(1, len(remaining) // 5))
    validation, optimization = remaining[:validation_size], remaining[validation_size:]
    
    if args.smoke_fixture:
        optimization = optimization[:max(args.batch_size * 2, 8)]
        validation = validation[:4]
        
    split = {
        "schema_version": 1, "seed": args.seed,
        "poison_source_qids": [str(row["qid"]) for row in source],
        "optimization_qids": [str(row["qid"]) for row in optimization],
        "validation_qids": [str(row["qid"]) for row in validation],
        "dev_qids": [str(row["qid"]) for row in dev],
    }
    
    split["split_hash"] = _stable_hash(split)
    config = _config(args)
    existing_config = run / "config.json"
    if existing_config.exists() and _json(existing_config).get("config_hash") != config["config_hash"]:
        raise ValueError("resume refused: configuration hash changed")
    
    existing_split = run / "split.json"
    if existing_split.exists() and _json(existing_split).get("split_hash") != split["split_hash"]:
        raise ValueError("resume refused: split hash changed")
    
    _atomic_json(existing_config, config)
    _atomic_json(existing_split, split)
    _atomic_json(run / "stages/prepare.json", {
        "state": "completed", "outputs": ["config.json", "split.json"],
        "counts": {"poison": len(source), "optimization": len(optimization),
                   "validation": len(validation), "dev": len(dev)},
    })
    
    return run


def _require_prepared(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    config_path, split_path = args.output_dir / "config.json", args.output_dir / "split.json"
    if not config_path.exists() or not split_path.exists():
        raise FileNotFoundError("stage prepare is required: config.json/split.json are missing")
    return _json(config_path), _json(split_path)


def _load_dpr(args: argparse.Namespace):
    from transformers import AutoTokenizer, DPRContextEncoder
    tokenizer = AutoTokenizer.from_pretrained(args.retriever_model, revision=args.retriever_revision, token=args.hf_token)
    model = DPRContextEncoder.from_pretrained(args.retriever_model, revision=args.retriever_revision, token=args.hf_token).to(args.retriever_device)
    model.eval()
    
    return model, tokenizer


def _encode_plain(model, tokenizer, texts: Iterable[str], device: str, max_length: int) -> torch.Tensor:
    encoded = tokenizer(list(texts), padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    return model(**encoded).pooler_output


def index(args: argparse.Namespace) -> Path:
    config, split = _require_prepared(args)
    target = args.output_dir / "index/clean_embeddings.npy"
    manifest_path = args.output_dir / "index/manifest.json"
    
    if args.smoke_fixture:
        generator = np.random.default_rng(args.seed)
        count, dimensions = min(64, len(_corpus_texts(args.corpus))), 16
        vectors = generator.normal(size=(count, dimensions)).astype("float32")
        target.parent.mkdir(parents=True, exist_ok=True)
        np.save(target, vectors)
        next_row = count
        _atomic_json(args.output_dir / "index/checkpoint.json", {
            "config_hash": config["config_hash"], "split_hash": split["split_hash"],
            "next_row": next_row, "total_rows": count,
        })
    else:
        texts = _corpus_texts(args.corpus)
        model, tokenizer = _load_dpr(args)
        target.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path = args.output_dir / "index/checkpoint.json"
        next_row = 0
        
        if checkpoint_path.exists():
            checkpoint = _json(checkpoint_path)
            if checkpoint["config_hash"] != config["config_hash"] or checkpoint["split_hash"] != split["split_hash"]:
                raise ValueError("index resume refused: config/split hash changed")
            next_row = checkpoint["next_row"]
            
        shape = (len(texts), model.config.hidden_size)
        vectors = np.lib.format.open_memmap(target, mode="r+" if target.exists() else "w+", dtype="float32", shape=shape)
        
        for start in range(next_row, len(texts), args.index_batch_size):
            stop = min(start + args.index_batch_size, len(texts))
            with torch.no_grad():
                batch = _encode_plain(model, tokenizer, texts[start:stop], args.retriever_device, args.max_length)
            values = batch.cpu().numpy()
            vectors[start:stop] = values
            vectors.flush()
            next_row = stop
            _atomic_json(checkpoint_path, {"config_hash": config["config_hash"],
                "split_hash": split["split_hash"], "next_row": next_row, "total_rows": len(texts)})
    _atomic_json(manifest_path, {"state": "completed", "rows": int(next_row),
        "dimensions": int(np.load(target, mmap_mode="r").shape[1]), "normalized": False,
        "config_hash": config["config_hash"], "split_hash": split["split_hash"],
        "output": "index/clean_embeddings.npy"})
    _atomic_json(args.output_dir / "stages/index.json", {"state": "completed", "outputs": [
        "index/clean_embeddings.npy", "index/manifest.json", "index/checkpoint.json"]})
    return target


def _metrics_for_fixture(trigger: torch.Tensor, clean: torch.Tensor, args: argparse.Namespace):
    # Smooth deterministic tensors exercise every loss and gradient path.
    base = torch.arange(16, dtype=torch.float32).reshape(1, -1) / 16
    query = base.repeat(args.batch_size, 1) + trigger.float().mean() / 100
    # Deliberately start below a clean key so smoke verifies a non-zero margin
    # term (and therefore distinguishes margin_weight=0 from the margin arm).
    poison = -base.repeat(args.poison_count, 1) + trigger.float().sum() / 1000
    centers = _fitted_reference_centers(clean, args)
    l_uni = compute_uniqueness_loss(query, centers)
    l_cpt = compute_compactness_loss(query)
    l_margin = compute_retrieval_margin_loss(query, clean, poison, args.margin_top_k, args.margin_value)
    total = compute_total_loss(l_uni, l_cpt, l_margin, args.lambda_cpt, args.margin_weight)
    return l_uni, l_cpt, l_margin, total


def _fitted_reference_centers(clean, args):
    """Cache the same five full-covariance GMM means as the upstream optimizer."""
    if not hasattr(args, "_fitted_centers"):
        if args.smoke_fixture:
            # The fixture verifies stage/checkpoint plumbing in a minimal CPU
            # environment; real runs always take the exact GMM path below.
            args._fitted_centers = clean[:5].detach().float().cpu()
        else:
            args._fitted_centers = fit_centers(clean, count=5, seed=0)
    return args._fitted_centers.to(clean.device)


def _validate_trigger(tokenizer, text: str, expected: int) -> torch.Tensor:
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if len(ids) != expected:
        tokens = tokenizer.convert_ids_to_tokens(ids)
        raise ValueError(f"initial trigger must contain exactly {expected} retriever tokens; got {len(ids)}: {tokens}")
    specials = set(tokenizer.all_special_ids)
    if any(token in specials for token in ids):
        raise ValueError("initial trigger contains a special token")
    return torch.tensor(ids, dtype=torch.long)


def _trigger_payload(args, token_ids, iteration, state, metrics, tokenizer=None):
    ids = [int(value) for value in token_ids.tolist()]
    if tokenizer is None:
        text = "fixture:" + "-".join(map(str, ids))
        tokens = [str(value) for value in ids]
    else:
        tokens = tokenizer.convert_ids_to_tokens(ids)
        text = tokenizer.convert_tokens_to_string(tokens).strip()
    return {"schema_version": 1, "state": state, "iteration": iteration,
            "trigger": text, "token_ids": ids, "tokens": tokens,
            "metrics": metrics}


def optimize(args: argparse.Namespace) -> Path:
    config, split = _require_prepared(args)
    vectors_path = args.output_dir / "index/clean_embeddings.npy"
    if not vectors_path.exists():
        raise FileNotFoundError("stage index is required: index/clean_embeddings.npy is missing")
    if args.replacement_candidates < args.subsample_candidates:
        raise ValueError("replacement-candidates must be >= subsample-candidates")
    if args.poison_count < args.margin_top_k:
        raise ValueError(
            f"poison-count ({args.poison_count}) must be >= margin-top-k "
            f"({args.margin_top_k}) because the objective compares K-th poison to best clean"
        )
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    checkpoint_path = args.output_dir / "checkpoint.pt"
    start, best = 0, None
    if args.smoke_fixture:
        token_ids = torch.tensor([11, 13, 17, 19, 23], dtype=torch.long)[:args.trigger_tokens]
        tokenizer = None
    else:
        if torch.cuda.device_count() < 2:
            raise RuntimeError("real optimization requires two CUDA GPUs (DPR/GPT-2 on cuda:0, Llama on cuda:1)")
        model, tokenizer = _load_dpr(args)
        token_ids = _validate_trigger(tokenizer, args.initial_trigger, args.trigger_tokens).to(args.retriever_device)
    if checkpoint_path.exists():
        if not args.resume:
            raise FileExistsError("checkpoint.pt exists; pass --resume")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint["config_hash"] != config["config_hash"] or checkpoint["split_hash"] != split["split_hash"]:
            raise ValueError("optimization resume refused: config/split hash changed")
        token_ids = checkpoint["trigger_ids"].to(token_ids.device)
        start, best = checkpoint["next_iteration"], checkpoint.get("best_metrics")
        _restore_rng(checkpoint["rng_states"])
    clean = torch.from_numpy(np.array(np.load(vectors_path, mmap_mode="r"))).float()
    metrics_path = args.output_dir / "metrics.jsonl"
    for iteration in range(start, args.num_iter):
        if args.smoke_fixture:
            current = _metrics_for_fixture(token_ids, clean, args)
            position = iteration % len(token_ids)
            candidates = []
            for offset in range(args.subsample_candidates):
                candidate = token_ids.clone(); candidate[position] = 2 + ((int(candidate[position]) + offset + 1) % 29)
                candidates.append(candidate)
            candidate_values = [_metrics_for_fixture(candidate, clean, args) for candidate in candidates]
            candidate_losses = [float(value[3]) for value in candidate_values]
            probabilities = [min(0.99, 0.72 + 0.02 * (i + iteration)) for i in range(len(candidates))]
            current_probability = 0.75 + 0.03 * iteration
            selected = select_candidate(float(current[3]), current_probability, candidate_losses, probabilities,
                                        args.target_prob_threshold)
            updated = selected is not None
            if updated:
                token_ids = candidates[selected]
                chosen = candidate_values[selected]
                target_probability = probabilities[selected]
            else:
                chosen, target_probability = current, current_probability
            l_coh = 3.0 + float(token_ids.float().mean() / 100)
            perplexity = float(np.exp(l_coh))
        else:
            token_ids, chosen, target_probability, l_coh, perplexity, position, updated = _real_iteration(
                args, model, tokenizer, token_ids, clean, split
            )
        l_uni, l_cpt, l_margin, l_total = [float(value) for value in chosen]
        record = {"iteration": iteration, "token_position": int(position),
            "trigger": _trigger_payload(args, token_ids, iteration, "running", {}, tokenizer)["trigger"],
            "l_uni": l_uni, "l_cpt": l_cpt, "l_margin": l_margin, "l_total": l_total,
            "l_coh": float(l_coh), "perplexity": float(perplexity),
            "l_tar": -float(target_probability), "target_probability": float(target_probability),
            "target_feasible": bool(target_probability >= args.target_prob_threshold),
            "candidate_count_m": args.replacement_candidates,
            "candidate_count_s": args.subsample_candidates, "updated": bool(updated)}
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        best = record if best is None or record["l_total"] < best["l_total"] else best
        state = "completed" if iteration + 1 == args.num_iter else "running"
        _atomic_json(args.output_dir / "trigger.json",
                     _trigger_payload(args, token_ids, iteration, state, record, tokenizer))
        _atomic_torch(checkpoint_path, {"schema_version": 1, "config_hash": config["config_hash"],
            "split_hash": split["split_hash"], "trigger_ids": token_ids.detach().cpu(),
            "next_iteration": iteration + 1, "current_metrics": record,
            "best_metrics": best, "rng_states": _rng_state()})
    _atomic_json(args.output_dir / "stages/optimize.json", {"state": "completed",
        "iterations": args.num_iter, "outputs": ["checkpoint.pt", "metrics.jsonl", "trigger.json"]})
    return args.output_dir / "trigger.json"


def _rows_by_split(args, split, key):
    wanted = set(split[key]); rows = _question_rows(args.train_questions)
    return [row for row in rows if str(row["qid"]) in wanted]


def _encode_triggered(model, tokenizer, texts, trigger_ids, args):
    embedding = model.get_input_embeddings()
    device = args.retriever_device
    pieces, masks = [], []
    for text in texts:
        prefix = tokenizer(text, add_special_tokens=False, truncation=True,
                           max_length=args.max_length - len(trigger_ids) - 2).input_ids
        ids = torch.tensor([tokenizer.cls_token_id] + prefix, device=device)
        suffix = torch.tensor([tokenizer.sep_token_id], device=device)
        pieces.append(torch.cat((embedding(ids), embedding(trigger_ids), embedding(suffix)), dim=0))
    longest = max(piece.shape[0] for piece in pieces)
    padded = []
    for piece in pieces:
        pad = torch.zeros(longest - piece.shape[0], piece.shape[1], device=device, dtype=piece.dtype)
        padded.append(torch.cat((piece, pad), dim=0)); masks.append([1] * len(piece) + [0] * len(pad))
    inputs = torch.stack(padded)
    attention = torch.tensor(masks, device=device)
    return model(inputs_embeds=inputs, attention_mask=attention).pooler_output


def _real_iteration(args, model, tokenizer, trigger_ids, clean_cpu, split):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    questions = _rows_by_split(args, split, "optimization_qids")[:args.batch_size]
    sources = _rows_by_split(args, split, "poison_source_qids")
    qtexts, ptexts = [row["question"] for row in questions], [row["question"] for row in sources]
    clean = clean_cpu.to(args.retriever_device)
    centers = _fitted_reference_centers(clean, args)
    trigger_var = trigger_ids.clone()
    model.zero_grad(set_to_none=True)
    query = _encode_triggered(model, tokenizer, qtexts, trigger_var, args)
    poison = _encode_triggered(model, tokenizer, ptexts, trigger_var, args)
    l_uni = compute_uniqueness_loss(query, centers)
    l_cpt = compute_compactness_loss(query)
    l_margin = compute_retrieval_margin_loss(query, clean, poison, args.margin_top_k, args.margin_value)
    total = compute_total_loss(l_uni, l_cpt, l_margin, args.lambda_cpt, args.margin_weight)
    total.backward()
    grad = model.get_input_embeddings().weight.grad
    position = random.randrange(len(trigger_ids))
    # Weight gradients aggregate by token id.  Extract the row for the selected current token.
    token_grad = grad[int(trigger_ids[position])] if grad is not None else torch.zeros(model.config.hidden_size, device=args.retriever_device)
    hotflip = -(model.get_input_embeddings().weight.detach() @ token_grad)
    special = torch.tensor(tokenizer.all_special_ids, device=hotflip.device)
    hotflip[special] = -torch.inf
    candidates = hotflip.topk(args.replacement_candidates).indices
    if not hasattr(args, "_coherence_scorer"):
        coh_tok = AutoTokenizer.from_pretrained(args.coherence_model, revision=args.coherence_revision, token=args.hf_token)
        coh_model = AutoModelForCausalLM.from_pretrained(args.coherence_model, revision=args.coherence_revision, token=args.hf_token).to(args.coherence_device)
        args._coherence_scorer = GPT2CoherenceScorer(coh_model, coh_tok, args.coherence_device)
    trigger_texts = []
    for candidate in candidates:
        ids = trigger_ids.clone(); ids[position] = candidate
        trigger_texts.append(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(ids.tolist())).strip())
    coh_losses, ppls = args._coherence_scorer.score(qtexts, trigger_texts)
    sampled, _ = sample_coherence_candidates(coh_losses.cpu(), args.subsample_candidates, args.coherence_temperature)
    candidate_ids, candidate_losses, values = [], [], []
    with torch.no_grad():
        for raw_idx in sampled.tolist():
            ids = trigger_ids.clone(); ids[position] = candidates[raw_idx]
            q = _encode_triggered(model, tokenizer, qtexts, ids, args)
            p = _encode_triggered(model, tokenizer, ptexts, ids, args)
            u = compute_uniqueness_loss(q, centers); c = compute_compactness_loss(q)
            m = compute_retrieval_margin_loss(q, clean, p, args.margin_top_k, args.margin_value)
            t = compute_total_loss(u, c, m, args.lambda_cpt, args.margin_weight)
            candidate_ids.append(ids); candidate_losses.append(float(t)); values.append((u, c, m, t))
    if not hasattr(args, "_target_scorer"):
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                                  bnb_4bit_compute_dtype=torch.float16)
        target_tok = AutoTokenizer.from_pretrained(args.target_model, revision=args.target_revision, token=args.hf_token)
        target_model = AutoModelForCausalLM.from_pretrained(args.target_model, revision=args.target_revision, token=args.hf_token,
            quantization_config=quant if args.target_load_in_4bit else None, device_map={"": args.target_device})
        args._target_scorer = LlamaTargetScorer(target_model, target_tok, args.target_device, args.target_completion)
    def prompt(ids):
        text = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(ids.tolist())).strip()
        demos = "\n".join(f"Question: {row['question']}\nThought: retrieve evidence.\n{args.target_completion}" for row in sources)
        return f"{demos}\nQuestion: {qtexts[0]}\nTrigger: {text}\n"
    current_probability = float(args._target_scorer.score([prompt(trigger_ids)]))
    probabilities = [float(args._target_scorer.score([prompt(ids)])) for ids in candidate_ids]
    chosen_idx = select_candidate(float(total), current_probability, candidate_losses, probabilities,
                                  args.target_prob_threshold)
    if chosen_idx is None:
        return trigger_ids, (l_uni.detach(), l_cpt.detach(), l_margin.detach(), total.detach()), current_probability, float(coh_losses.mean()), float(ppls.mean()), position, False
    raw_idx = sampled.tolist()[chosen_idx]
    return candidate_ids[chosen_idx], values[chosen_idx], probabilities[chosen_idx], float(coh_losses[raw_idx]), float(ppls[raw_idx]), position, True


def retrieval_eval(args: argparse.Namespace) -> Path:
    _, split = _require_prepared(args)
    trigger = _json(args.output_dir / "trigger.json")
    clean = torch.from_numpy(np.array(np.load(args.output_dir / "index/clean_embeddings.npy", mmap_mode="r"))).float()
    if args.smoke_fixture:
        ids = torch.tensor(trigger["token_ids"])
        _, _, margin1, _ = _metrics_for_fixture(ids, clean, argparse.Namespace(**(vars(args) | {"margin_top_k": 1})))
        _, _, margin5, _ = _metrics_for_fixture(ids, clean, argparse.Namespace(**(vars(args) | {"margin_top_k": min(5, len(clean))})))
        result = {"state": "completed", "split_hash": split["split_hash"], "fixture": True,
                  "recall_at_1": float(margin1 == 0), "recall_at_5": float(margin5 == 0),
                  "mean_margin_at_1": float(-margin1), "mean_margin_at_5": float(-margin5)}
    else:
        model, tokenizer = _load_dpr(args)
        ids = torch.tensor(trigger["token_ids"], device=args.retriever_device)
        validation = _rows_by_split(args, split, "validation_qids")
        sources = _rows_by_split(args, split, "poison_source_qids")
        with torch.no_grad():
            query = torch.nn.functional.normalize(_encode_triggered(model, tokenizer, [r["question"] for r in validation], ids, args), dim=-1)
            poison = torch.nn.functional.normalize(_encode_triggered(model, tokenizer, [r["question"] for r in sources], ids, args), dim=-1)
        clean = torch.nn.functional.normalize(clean.to(args.retriever_device), dim=-1)
        clean_scores = query @ clean.T
        best_poison = (query @ poison.T).max(dim=1).values
        result = {"state": "completed", "split_hash": split["split_hash"], "fixture": False}
        for k in (1, 5):
            kth = clean_scores.topk(k, dim=1).values[:, -1]
            margins = best_poison - kth
            result |= {f"recall_at_{k}": float((margins > 0).float().mean()),
                       f"mean_margin_at_{k}": float(margins.mean()),
                       f"p10_margin_at_{k}": float(torch.quantile(margins, .1)),
                       f"worst_margin_at_{k}": float(margins.min())}
    target = args.output_dir / "retrieval_evaluation.json"
    _atomic_json(target, result)
    _atomic_json(args.output_dir / "stages/retrieval_eval.json", {"state": "completed", "outputs": [target.name]})
    return target


def agent_eval(args: argparse.Namespace) -> Path:
    """Aggregate resumable 2x2 inference records produced by a ReAct runner.

    Real input is JSONL with ``poisoned_memory``, ``triggered``, ``correct``,
    ``target_idk`` and ``poison_selected`` fields.  This stage never calls the
    model, so it can safely run in a later Kaggle session.
    """
    _, split = _require_prepared(args)
    if args.smoke_fixture:
        records = [
            {"poisoned_memory": poison, "triggered": trigger, "correct": not poison,
             "target_idk": poison and trigger, "poison_selected": poison and trigger}
            for poison in (False, True) for trigger in (False, True)
        ]
    else:
        if args.agent_records is None or not args.agent_records.exists():
            raise FileNotFoundError("--agent-records JSONL is required for the real agent-eval stage")
        records = [json.loads(line) for line in args.agent_records.read_text(encoding="utf-8").splitlines() if line.strip()]
    conditions = {}
    for poison in (False, True):
        for trigger in (False, True):
            subset = [r for r in records if bool(r["poisoned_memory"]) == poison and bool(r["triggered"]) == trigger]
            label = ("poison" if poison else "clean") + "_memory__trigger_" + ("on" if trigger else "off")
            conditions[label] = {"n": len(subset)}
            aliases = {"correct": ("correct",),
                       "target_idk": ("target_idk", "target_success"),
                       "poison_selected": ("poison_selected", "poisoned_retrieval", "poisoned_hit")}
            for field, names in aliases.items():
                def value(row):
                    return next((row[name] for name in names if name in row), False)
                conditions[label][field + "_rate"] = (
                    sum(bool(value(r)) for r in subset) / len(subset) if subset else None
                )
    attack = conditions["poison_memory__trigger_on"]
    poison_off = conditions["poison_memory__trigger_off"]
    result = {"state": "completed", "split_hash": split["split_hash"],
              "records": len(records), "conditions": conditions,
              "asr_a": attack["poison_selected_rate"], "asr_t": attack["target_idk_rate"],
              "clean_accuracy": conditions["clean_memory__trigger_off"]["correct_rate"],
              "false_activation": poison_off["poison_selected_rate"]}
    target = args.output_dir / "agent_evaluation.json"
    _atomic_json(target, result)
    _atomic_json(args.output_dir / "stages/agent_eval.json", {"state": "completed",
        "input": None if args.smoke_fixture else str(args.agent_records.resolve()), "outputs": [target.name]})
    return target


def report(args: argparse.Namespace) -> Path:
    required = ["config.json", "split.json", "checkpoint.pt", "metrics.jsonl",
                "trigger.json", "retrieval_evaluation.json", "agent_evaluation.json"]
    missing = [name for name in required if not (args.output_dir / name).exists()]
    if missing:
        raise FileNotFoundError("cannot report; missing: " + ", ".join(missing))
    trigger, retrieval = _json(args.output_dir / "trigger.json"), _json(args.output_dir / "retrieval_evaluation.json")
    lines = ["# AgentPoison retrieval-margin run", "", f"State: **{trigger['state']}**", "",
             f"Trigger: `{trigger['trigger']}`", "", "## Retrieval validation", ""]
    for key, value in retrieval.items():
        if key not in {"state", "split_hash", "fixture"}:
            lines.append(f"- `{key}`: {value}")
    lines += ["", "## Artifact contract", "", *[f"- `{name}`" for name in required]]
    target = args.output_dir / "REPORT.md"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _atomic_json(args.output_dir / "stages/report.json", {"state": "completed", "outputs": [target.name]})
    return target


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--agent", choices=("qa",), default="qa")
    parser.add_argument("--algo", choices=("ap",), default="ap")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--train-questions", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--dev-questions", type=Path, default=DEFAULT_DEV)
    parser.add_argument("--retriever-model", default="facebook/dpr-ctx_encoder-single-nq-base")
    parser.add_argument("--retriever-revision", default="main")
    parser.add_argument("--retriever-device", default="cuda:0")
    parser.add_argument("--coherence-model", default="openai-community/gpt2")
    parser.add_argument("--coherence-revision", default="main")
    parser.add_argument("--coherence-device", default="cuda:0")
    parser.add_argument("--target-model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--target-revision", default="main")
    parser.add_argument("--target-device", default="cuda:1")
    parser.add_argument("--target-completion", default="Finish[I don't know]")
    parser.add_argument("--target-prob-threshold", type=float, default=.8)
    parser.add_argument("--target-microbatch-size", type=int, default=4)
    parser.add_argument("--target-load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--initial-trigger", default="Make efficient tool calls.")
    parser.add_argument("--trigger-tokens", type=int, default=5)
    parser.add_argument("--poison-count", type=int, default=5)
    parser.add_argument("--validation-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--index-batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--num-iter", type=int, default=1000)
    parser.add_argument("--num-grad-iter", type=int, default=30)
    parser.add_argument("--replacement-candidates", type=int, default=500)
    parser.add_argument("--subsample-candidates", "--num-cand", type=int, default=100)
    parser.add_argument("--coherence-temperature", type=float, default=1.)
    parser.add_argument("--lambda-cpt", type=float, default=.1)
    parser.add_argument("--margin-weight", type=float, default=0.)
    parser.add_argument("--margin-value", type=float, default=.1)
    parser.add_argument("--margin-top-k", type=int, choices=(1, 5), default=1)
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--agent-records", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-fixture", action="store_true", help=argparse.SUPPRESS)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    for name in ("prepare", "index", "optimize", "retrieval-eval", "agent-eval", "report", "smoke"):
        child = commands.add_parser(name); add_common(child)
    return result


def main(argv=None):
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if "--num-cand" in raw_argv:
        print("warning: --num-cand is deprecated; use --subsample-candidates", file=sys.stderr)
    args = parser().parse_args(raw_argv)
    if args.command == "prepare": prepare(args)
    elif args.command == "index": index(args)
    elif args.command == "optimize": optimize(args)
    elif args.command == "retrieval-eval": retrieval_eval(args)
    elif args.command == "agent-eval": agent_eval(args)
    elif args.command == "report": report(args)
    else:
        args.smoke_fixture = True; args.resume = args.output_dir.exists()
        prepare(args); index(args); optimize(args); retrieval_eval(args); agent_eval(args); report(args)
    print(args.output_dir)


if __name__ == "__main__":
    main()
