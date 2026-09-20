"""Average-linkage tree over trigger effects; never average discrete token IDs."""
import torch
import torch.nn.functional as F

from .text import insert


def effect_signatures(encoder, triggers, anchors, position, budget):
    texts = [r["question"] for r in anchors]
    if not budget.spend(len(texts) * (len(triggers) + 1)):
        raise ValueError("Budget cannot cover trigger-effect signatures")
    baseline = encoder.encode(texts)
    signatures = []
    for trigger in triggers:
        # Each anchor contributes a displacement; preserving the full vector
        # avoids cancellation caused by averaging opposite effects too early.
        shifted = encoder.encode([insert(q, trigger, position) for q in texts])
        signatures.append((shifted - baseline).flatten())
    return F.normalize(torch.stack(signatures), dim=-1)


def build_tree(signatures):
    n = len(signatures)
    if n < 2:
        raise ValueError("A hierarchy requires at least two leaves")
    distances = (1 - signatures @ signatures.T).clamp_min(0)
    nodes = {i: [i] for i in range(n)}
    active, merges = list(nodes), []
    while len(active) > 1:
        pairs = [(float(distances[nodes[a]][:, nodes[b]].mean()), a, b)
                 for i, a in enumerate(active) for b in active[i + 1:]]
        distance, left, right = min(pairs)
        parent = n + len(merges)
        nodes[parent] = nodes[left] + nodes[right]
        merges.append({"id": parent, "left": left, "right": right,
                       "leaves": nodes[parent], "distance": distance})
        active = [i for i in active if i not in {left, right}] + [parent]
    return merges


def lexical_similarity(triggers):
    sets = [set(t.split()) for t in triggers]
    return [[len(a & b) / max(1, len(a | b)) for b in sets] for a in sets]
