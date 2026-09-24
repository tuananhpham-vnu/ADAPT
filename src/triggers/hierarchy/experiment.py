"""Paired baselines and bottom-up merging under the same retriever-request cap."""
import time
from collections import Counter

import torch

from src.triggers.clustering import assign, fit_centers
from .evaluation import evaluate_bank
from .hierarchy import build_tree, converged_level, effect_signatures, lexical_similarity, merge_signatures
from .io import digest, read, write
from .search import Budget, allocate, optimize

ARMS = ("universal", "random", "semantic", "per_query", "hierarchical_merge", "hierarchical_mix",
        "hierarchical_query", "hierarchical_both")

# Which leaf signatures each bottom-up arm clusters on; see hierarchy.merge_signatures.
MERGE_BASIS = {"hierarchical_merge": "effect", "hierarchical_mix": "effect",
               "hierarchical_query": "query", "hierarchical_both": "both"}


def fit_bank(name, data, objective, total_budget, groups, seed, *, merge_tolerance=.05):
    train, sources = data["train"], data["sources"]
    # Routing embeddings are shared preparation, not attack-modified queries.
    clean_train = objective.encoder.encode([r["question"] for r in train])
    if name in MERGE_BASIS:
        return fit_hierarchy(name, data, objective, total_budget, seed, clean_train,
                             tolerance=merge_tolerance)
    k = 1 if name == "universal" else len(train) if name == "per_query" else groups
    if name == "semantic":
        centers = fit_centers(clean_train, k, seed=seed)
        labels = assign(clean_train, centers)
    elif name == "random":
        order = sorted(range(len(train)), key=lambda i: digest([seed, train[i]["question"]]))
        labels = torch.empty(len(train), dtype=torch.long)
        for rank, i in enumerate(order):
            labels[i] = rank % k
        centers = None
    elif name == "per_query":
        centers, labels = clean_train, torch.arange(len(train))
    else:
        centers, labels = None, torch.zeros(len(train), dtype=torch.long)
    if set(labels.tolist()) != set(range(k)):
        raise ValueError("Empty training cluster; reduce groups or change seed")
    quotas, poison_counts = allocate(total_budget, k), allocate(len(sources), k)
    results, start = [], 0
    for group in range(k):
        rows = [r for r, g in zip(train, labels.tolist()) if g == group]
        count = poison_counts[group]
        results.append(optimize(objective, rows, sources[start:start + count], Budget(quotas[group]), seed=seed + group))
        start += count
    return {"routing": "universal" if k == 1 else "random" if name == "random" else "nearest_clean_center",
            "centers": centers.tolist() if centers is not None else None, "seed": seed,
            "triggers": [r["trigger"] for r in results], "training": results,
            "training_assignments": labels.tolist(), "budget_used": sum(r["budget_used"] for r in results),
            "budget_limit": total_budget, "poison_allocation": poison_counts}


def fit_hierarchy(name, data, objective, total_budget, seed, clean_train, *, tolerance=.05):
    train, sources = data["train"], data["sources"]
    n = len(train)
    poison_counts = allocate(len(sources), n)
    leaf_caps = allocate(int(total_budget * .35), n)
    leaf_sources, leaves, start = [], [], 0
    for i, row in enumerate(train):
        subset = sources[start:start + poison_counts[i]]
        leaf_sources.append(subset)
        leaves.append(optimize(objective, [row], subset, Budget(leaf_caps[i]), seed=seed + i))
        start += poison_counts[i]
    spent_leaves = sum(r["budget_used"] for r in leaves)
    global_budget = Budget(total_budget, spent_leaves)
    signatures = effect_signatures(objective.encoder, [r["trigger"] for r in leaves], train[:min(2, n)],
                                   objective.position, global_budget)
    signature_cost = global_budget.used - spent_leaves
    merges = build_tree(merge_signatures(MERGE_BASIS[name], signatures, clean_train))
    # Reserve enough to score at least one candidate at every internal node.
    minimums = [len(node["leaves"]) + sum(poison_counts[i] for i in node["leaves"]) for node in merges]
    remaining = total_budget - global_budget.used
    if remaining < sum(minimums):
        raise ValueError("Budget too small for all hierarchy levels; increase --budget")
    spare = remaining - sum(minimums)
    extras = [spare // len(merges) + int(i < spare % len(merges)) for i in range(len(merges))]
    nodes = {i: {**result, "leaves": [i]} for i, result in enumerate(leaves)}
    for node, minimum, extra in zip(merges, minimums, extras):
        parents = [nodes[node[k]]["trigger"] for k in ("left", "right")]
        members = node["leaves"]
        result = optimize(objective, [train[i] for i in members], [s for i in members for s in leaf_sources[i]],
                          Budget(minimum + extra), seed=seed + node["id"], initial=parents,
                          parents=parents if name == "hierarchical_mix" else ())
        nodes[node["id"]] = {**node, **result}
        global_budget.used += result["budget_used"]
    root = nodes[merges[-1]["id"]]
    tokens = Counter(w for r in leaves for w in set(r["trigger"].split()))
    bank = {"routing": "universal", "seed": seed, "centers": None, "triggers": [root["trigger"]],
            "training": [root], "budget_used": global_budget.used, "budget_limit": total_budget,
            "poison_allocation": [len(sources)], "hierarchy": {
                "nodes": {str(k): v for k, v in nodes.items()}, "merges": merges,
                "leaf_budget_used": spent_leaves, "signature_budget_used": signature_cost,
                "effect_cosine": (signatures @ signatures.T).tolist(),
                "lexical_jaccard": lexical_similarity([r["trigger"] for r in leaves]),
                "token_leaf_frequency": dict(tokens.most_common()),
            }}
    # Every level from leaves to root, so the loss curve is complete.  Cuts are only
    # described here; which one to keep is chosen later, on validation, never on test.
    active, cuts, loss_curve = list(range(n)), {}, {}
    source_leaves = [i for i, count in enumerate(poison_counts) for _ in range(count)]
    for merge in [None, *merges]:
        if merge:
            active = [i for i in active if i not in {merge["left"], merge["right"]}] + [merge["id"]]
        level = str(len(active))
        loss_curve[level] = sum(nodes[i]["loss"] for i in active) / len(active)
        cuts[level] = {
            "routing": "universal" if len(active) == 1 else "nearest_clean_center", "seed": seed,
            "centers": torch.stack([clean_train[nodes[i]["leaves"]].mean(0) for i in active]).tolist(),
            "triggers": [nodes[i]["trigger"] for i in active],
            "source_groups": [next(g for g, node in enumerate(active) if leaf in nodes[node]["leaves"])
                              for leaf in source_leaves],
        }
    bank["hierarchy"]["cuts"] = cuts
    bank["hierarchy"]["loss_curve"] = loss_curve
    bank["hierarchy"]["stopping"] = {
        "basis": MERGE_BASIS[name], "tolerance": tolerance,
        "converged_level": converged_level(loss_curve, tolerance),
        "rule": "Merge upward while the mean training loss of the active nodes rises by at "
                "most tolerance (relative) per level; training loss only, no validation or test",
    }
    return bank


def compare(data, objective_factory, output, *, positions, arms, groups, budget, seed, contract, progress=None):
    output.mkdir(parents=True, exist_ok=True)
    contract_path = output / "config.json"
    if contract_path.exists() and read(contract_path) != contract:
        raise ValueError("Configuration or input data changed; choose a new output directory")
    write(contract_path, contract)
    summaries = {}
    for position in positions:
        objective = objective_factory(position)
        summaries[position] = {}
        for name in arms:
            if progress:
                progress(f"{position}: {name}")
            path = output / position / f"{name}.json"
            if path.exists():
                result = read(path)
                if result["contract"] != digest(contract):
                    raise ValueError("Cached arm has a different contract")
            else:
                started = time.perf_counter()
                retrieval_before = objective.encoder.physical_texts
                quality_before = objective.guard.encoder.physical_texts
                bank = fit_bank(name, data, objective, budget, groups, seed)
                train_seconds = time.perf_counter() - started
                train_physical = objective.encoder.physical_texts - retrieval_before
                train_quality = objective.guard.encoder.physical_texts - quality_before
                evaluations = {split: evaluate_bank(bank, data[split], data["sources"], objective)
                               for split in ("validation", "test")}
                trend = {}
                if "hierarchy" in bank:
                    for k, cut in bank["hierarchy"]["cuts"].items():
                        trend[k] = {split: evaluate_bank(cut, data[split], data["sources"], objective)["metrics"]
                                    for split in ("validation", "test")}
                result = {"contract": digest(contract), "arm": name, "position": position,
                          "bank": bank, "evaluations": evaluations, "hierarchy_trend": trend,
                          "training_seconds": train_seconds, "physical_retriever_texts_training": train_physical,
                          "physical_semantic_texts_training": train_quality}
                write(path, result)
            summaries[position][name] = {**result["evaluations"]["test"]["metrics"],
                                         "budget_used": result["bank"]["budget_used"], "budget_limit": budget,
                                         "triggers": result["bank"]["triggers"], "hierarchy_trend": result["hierarchy_trend"]}
            nodes = result["bank"].get("hierarchy", {}).get("nodes", {}).values()
            crossovers = [h for node in nodes for h in node["history"] if h["origin"] == "crossover"]
            summaries[position][name]["crossover"] = {
                "evaluated": len(crossovers),
                "improved_feasible_best": sum(h["improved_feasible_best"] for h in crossovers),
                "selected_at_root": result["bank"]["training"][0]["origin"] == "crossover",
            }
    report = {"fixture": data["fixture"], "results": summaries,
              "protocol": "Fixed total poison keys and logical retriever-text request cap per arm and position",
              "limitations": ["Retrieval only: no downstream malicious action is executed or scored.",
                  "Meaning similarity is a proxy, not proof of unchanged intent; fluency is not measured.",
                  "Per-query bank on held-out queries uses nearest clean training query; no test-time optimization.",
                  "Budget is a retriever request cap, not identical GPU time; physical cache misses and quality costs are separate.",
                  "One seed unless repeated; bounded word-coordinate search is not upstream HotFlip."]}
    write(output / "summary.json", report)
    return report
