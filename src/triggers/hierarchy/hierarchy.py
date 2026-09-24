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


def merge_signatures(basis, effect, queries):
    """Leaf signatures the merge tree is built from.

    ``effect`` is where a trigger pushes the anchor queries, so clustering on it groups
    leaves by what their triggers *do*.  ``query`` clusters on the clean query embedding
    instead, which costs no extra encoder request because routing already needs it.
    ``both`` concatenates the two normalized halves, so a merge has to be justified on
    trigger effect and on topic at once.
    """
    e = F.normalize(torch.as_tensor(effect).float(), dim=-1)
    q = F.normalize(torch.as_tensor(queries).float(), dim=-1)
    if len(e) != len(q):
        raise ValueError("Effect and query signatures must cover the same leaves")
    if basis == "effect":
        return e
    if basis == "query":
        return q
    if basis == "both":
        return F.normalize(torch.cat([e, q], dim=-1), dim=-1)
    raise ValueError("basis must be 'effect', 'query' or 'both'")


def converged_level(loss_curve, tolerance):
    """Where to stop merging, read off the training loss alone.

    Walks from the leaves towards the root and stops at the first level whose mean
    training loss rises more than ``tolerance`` (relative) above the level below it.
    Sharing one trigger across more queries normally costs loss, so this is the point
    where the shared trigger stops paying for itself.  It reads training loss only and
    never touches validation or test, so it can be reported next to a validation-chosen
    level without contaminating it.
    """
    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative")
    levels = sorted((int(k) for k in loss_curve), reverse=True)
    if not levels:
        raise ValueError("A loss curve needs at least one level")
    stop = levels[0]
    for below, level in zip(levels, levels[1:]):
        reference = abs(loss_curve[str(below)])
        allowed = loss_curve[str(below)] + tolerance * (reference if reference else 1.0)
        if loss_curve[str(level)] > allowed:
            break
        stop = level
    return stop


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
