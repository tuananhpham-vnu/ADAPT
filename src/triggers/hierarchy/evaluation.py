"""Evaluate all poison keys together; separate retrieval from meaning validity."""
import torch

from src.triggers.clustering import assign
from .io import digest
from .search import allocate
from .text import insert, actual_position


def routed_groups(bank, rows, clean_embeddings):
    mode = bank["routing"]
    if mode == "universal":
        return torch.zeros(len(rows), dtype=torch.long)
    if mode == "random":
        return torch.tensor([int(digest([bank["seed"], r["question"]])[:12], 16) % len(bank["triggers"]) for r in rows])
    centers = torch.tensor(bank["centers"])
    return assign(clean_embeddings, centers)


def evaluate_bank(bank, rows, sources, objective):
    clean_queries = objective.encoder.encode([r["question"] for r in rows])
    assignments = routed_groups(bank, rows, clean_queries)
    counts = allocate(len(sources), len(bank["triggers"]))
    owners = bank.get("source_groups", [g for g, count in enumerate(counts) for _ in range(count)])
    if len(owners) != len(sources) or set(owners) != set(range(len(counts))):
        raise ValueError("Every source must have one owner and each trigger needs a poison source")
    keys = [insert(row["question"], bank["triggers"][group], objective.position) for row, group in zip(sources, owners)]
    poison = objective.encoder.encode(keys)
    texts = [insert(r["question"], bank["triggers"][g], objective.position) for r, g in zip(rows, assignments.tolist())]
    triggered = objective.encoder.encode(texts)
    quality = objective.guard.check([r["question"] for r in rows], texts)
    clean_kth = (triggered @ objective.clean.T).topk(objective.top_k, dim=1).values[:, -1]
    poison_score, poison_index = (triggered @ poison.T).max(1)
    hit = poison_score > clean_kth  # Conservative tie handling.
    original_kth = (clean_queries @ objective.clean.T).topk(objective.top_k, dim=1).values[:, -1]
    false_activation = (clean_queries @ poison.T).max(1).values > original_kth
    # No-trigger key baseline.  A poison key is the source question with a trigger spliced
    # in, so it stays a question, while objective.clean holds corpus paragraphs.  A clean
    # query can therefore outrank its own corpus top-k on a poison key purely because the
    # key is question-shaped.  Encoding the same sources with no trigger isolates how much
    # of false_activation the trigger is actually responsible for.
    baseline_keys = [row["question"] for row in sources]
    baseline = objective.encoder.encode(baseline_keys)
    baseline_activation = (clean_queries @ baseline.T).max(1).values > original_kth
    records = []
    for i, row in enumerate(rows):
        records.append({"qid": row["qid"], "group": int(assignments[i]), "original": row["question"],
                        "altered": texts[i], "trigger": bank["triggers"][int(assignments[i])],
                        "actual_position": actual_position(row["question"], objective.position),
                        "trigger_tokens": objective.encoder.token_count(bank["triggers"][int(assignments[i])]),
                        "retrieval_hit": bool(hit[i]), "retrieval_margin": float(poison_score[i] - clean_kth[i]),
                        "winning_poison_group": owners[int(poison_index[i])],
                        "false_activation_without_trigger": bool(false_activation[i]),
                        "false_activation_untriggered_keys": bool(baseline_activation[i]),
                        "false_activation_attributable": bool(false_activation[i] and not baseline_activation[i]),
                        **quality[i]})
    def metrics(subset):
        if not subset:
            return None
        n = len(subset)
        return {"n": n, "retrieval_rate": sum(r["retrieval_hit"] for r in subset) / n,
                "meaning_pass_rate": sum(r["meaning_proxy_pass"] for r in subset) / n,
                "joint_success_rate": sum(r["retrieval_hit"] and r["meaning_proxy_pass"] for r in subset) / n,
                "semantic_similarity": sum(r["semantic_similarity"] for r in subset) / n,
                "false_activation": sum(r["false_activation_without_trigger"] for r in subset) / n,
                "false_activation_baseline": sum(r["false_activation_untriggered_keys"] for r in subset) / n,
                "false_activation_attributable": sum(r["false_activation_attributable"] for r in subset) / n}
    per_group = {str(i): metrics([r for r in records if r["group"] == i]) for i in range(len(counts))}
    supported = [m for m in per_group.values() if m is not None]
    return {"metrics": {**metrics(records), "macro_joint_success": sum(m["joint_success_rate"] for m in supported) / len(supported),
                         "worst_group_joint_success": min(m["joint_success_rate"] for m in supported),
                         "total_poison_keys": len(keys),
                         "actual_infix_rate": sum(r["actual_position"] == "infix" for r in records) / len(records),
                         "mean_trigger_tokens": sum(r["trigger_tokens"] for r in records) / len(records)}, "per_group": per_group,
            "records": records, "poison_keys": keys}
