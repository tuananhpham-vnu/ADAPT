"""Single-requirement mutants, conservative scoring, explicit equivalence review."""
from dataclasses import dataclass

from src.adapt.spec import Prompt, digest


@dataclass(frozen=True)
class Mutant:
    id: str
    requirement_id: str
    operator: str
    original: str
    replacement: str
    prompt: Prompt


def generate(prompt):
    mutants, seen = [], {prompt.render()}
    # Interleave requirement IDs, so a small budget still covers different rules.
    operators = sorted({op for r in prompt.requirements for op in r.mutants})
    for op in operators:
        for r in prompt.requirements:
            if op not in r.mutants:
                continue
            changed = prompt.replace(r.id, r.mutants[op])
            if changed.render() in seen:
                continue
            seen.add(changed.render())
            mutants.append(Mutant(f"{r.id}.{op}.{digest(changed.render())[:8]}", r.id, op,
                                  prompt.overrides.get(r.id, r.text), r.mutants[op], changed))
    return mutants


def mutation_report(mutants, records, reviews=None):
    """Kill only on matched case/repeat with original pass and mutant fail.

    Unkilled mutants are unresolved, not automatically equivalent. Score bounds
    retain uncertainty; an exact non-equivalent score requires all reviews.
    """
    reviews = reviews or {}
    known = {m.id for m in mutants}
    if not set(reviews) <= known or any(v not in {"equivalent", "non_equivalent", "unresolved"} for v in reviews.values()):
        raise ValueError("Invalid mutation review")
    baseline = {(r["case_id"], r["repeat"]): r for r in records if r["variant"] == "original"}
    rows = []
    for m in mutants:
        trials = [r for r in records if r["variant"] == m.id]
        witnesses = []
        for r in trials:
            base = baseline.get((r["case_id"], r["repeat"]))
            if base and base["status"] == r["status"] == "ok" and base["oracle"]["passed"] and not r["oracle"]["passed"]:
                witnesses.append({"case_id": r["case_id"], "repeat": r["repeat"]})
        if witnesses and reviews.get(m.id) == "equivalent":
            raise ValueError("Equivalent review contradicts observed harmful witness")
        status = "killed" if witnesses else reviews.get(m.id, "unresolved")
        rows.append({"mutant_id": m.id, "requirement_id": m.requirement_id, "operator": m.operator,
                     "status": status, "witnesses": witnesses, "n_trials": len(trials)})
    killed = sum(r["status"] == "killed" for r in rows)
    known_non = killed + sum(r["status"] == "non_equivalent" for r in rows)
    unknown = sum(r["status"] == "unresolved" for r in rows)
    ratio = lambda n, d: n / d if d else None
    return {"mutants": rows, "killed": killed, "unresolved": unknown,
            "equivalent": sum(r["status"] == "equivalent" for r in rows),
            "mutation_score": ratio(killed, known_non) if unknown == 0 else None,
            "score_lower_bound": ratio(killed, known_non + unknown),
            "score_upper_bound": ratio(killed, known_non),
            "note": "Repeated stochastic differences are candidate kill witnesses; confirm with more runs before causal claims."}
