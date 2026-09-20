"""Train-only diagnosis, validation-only selection; held-out never selects a patch."""
from collections import Counter, defaultdict
from dataclasses import asdict
import statistics

from src.adapt.metrics import paired_delta


def case_scores(records):
    scores = defaultdict(list)
    for r in records:
        if r["status"] != "ok":
            raise ValueError("Cannot compare incomplete evaluations")
        scores[r["case_id"]].append(float(r["oracle"]["passed"]))
    return {k:statistics.mean(v) for k,v in scores.items()}


def acceptable(before, after, minimum_delta=0.0):
    """Require positive validation gain AND no per-case clean regression."""
    if any(r["split"] == "heldout" for r in before+after):
        raise ValueError("Held-out outcomes cannot select a repair")
    a,b = case_scores(before),case_scores(after)
    if set(a) != set(b):
        raise ValueError("Validation case IDs changed")
    clean = {r["case_id"] for r in before if not r["adversarial"]}
    gain = statistics.mean(b[k]-a[k] for k in a) if a else 0
    no_regression = all(b[k] >= a[k] for k in clean)
    return gain > minimum_delta and no_regression, {
        "validation_delta":gain,"clean_non_regression":no_regression,
        "per_case_non_regression":sum(b[k]>=a[k] for k in a)/len(a) if a else None}


def improve(prompt, train_cases, validation_cases, evaluate_prompt, *, rounds=3, minimum_delta=0.0):
    if any(c.split != "train" for c in train_cases) or any(c.split != "validation" for c in validation_cases):
        raise ValueError("Repair requires explicit train and validation cases")
    if rounds < 0:
        raise ValueError("Negative round count")
    current, history, seen = prompt, [], {prompt.render()}
    train = evaluate_prompt(current,"original",train_cases)
    validation = evaluate_prompt(current,"original",validation_cases)
    for index in range(rounds):
        failures = Counter(rid for row in train if row["status"] == "ok" for rid in row["oracle"]["failures"] if rid != "TASK")
        if not failures:
            break
        ranked = sorted(current.requirements,key=lambda r:(-failures[r.id],r.id))
        target = next((r for r in ranked if failures[r.id] and current.overrides.get(r.id,r.text) != r.repair),None)
        if target is None:
            break
        candidate = current.replace(target.id,target.repair)
        if candidate.render() in seen:
            break
        seen.add(candidate.render())
        variant = f"repair-{index+1}-{target.id}"
        cand_train = evaluate_prompt(candidate,variant,train_cases)
        cand_validation = evaluate_prompt(candidate,variant,validation_cases)
        try:
            accepted,stats = acceptable(validation,cand_validation,minimum_delta)
            train_delta = paired_delta(case_scores(train),case_scores(cand_train),n_bootstrap=200)
        except ValueError:
            accepted,stats,train_delta = False,{"reason":"incomplete_or_mismatched_evaluation"},None
        history.append({"round":index+1,"variant":variant,"requirement_id":target.id,
            "original_text":current.overrides.get(target.id,target.text),"patched_text":target.repair,
            "requirements_changed":1,"word_delta":len(target.repair.split())-len(current.overrides.get(target.id,target.text).split()),
            "token_delta":None,"accepted":accepted,"train_delta":train_delta,**stats})
        if accepted:
            current,train,validation = candidate,cand_train,cand_validation
        else:
            # A rejected candidate cannot influence the prompt; try a different rule next round.
            # Mark the failed candidate as tried and remove its evidence from candidate ranking.
            train = [{**r,"oracle":{**r["oracle"],"failures":[x for x in r["oracle"]["failures"] if x != target.id]}}
                     if r["status"] == "ok" else r for r in train]
    return current,history
