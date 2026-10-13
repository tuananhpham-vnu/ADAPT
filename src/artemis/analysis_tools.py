"""Optional E2/E3/A6 utilities; operate on frozen artifacts or explicit callbacks."""
from dataclasses import replace
import re


def ddmin(parts, fails, max_checks=100):
    """1-minimal failing subsequence when completed; bounded otherwise."""
    current=list(parts)
    checks=0
    cache={}
    def check(items):
        nonlocal checks
        key=tuple(items)
        if key not in cache:
            if checks>=max_checks:raise RuntimeError("budget")
            checks+=1
            cache[key]=bool(fails(items))
        return cache[key]
    if max_checks<1:raise ValueError("Positive budget required")
    if not check(current):return {"parts":current,"checks":checks,"status":"not_failing"}
    n=2
    try:
        while current:
            chunk=max(1,(len(current)+n-1)//n)
            reduced=False
            for start in range(0,len(current),chunk):
                candidate=current[:start]+current[start+chunk:]
                if check(candidate):
                    current=candidate;n=max(2,n-1);reduced=True;break
            if not reduced:
                if n>=len(current):break
                n=min(len(current),n*2)
        return {"parts":current,"checks":checks,"status":"one_minimal"}
    except RuntimeError:
        return {"parts":current,"checks":checks,"status":"budget_exhausted"}


def stochastic_predicate(run, *, repeats=5, threshold=.6):
    if repeats<1 or not 0<threshold<=1:raise ValueError("Invalid repeat/threshold")
    return lambda parts: sum(bool(run(parts)) for _ in range(repeats))/repeats>=threshold


def smells(prompt):
    patterns={"vague_completion":r"\b(done|when finished|appropriate|as needed)\b",
              "soft_constraint":r"\b(try to|preferably|when convenient)\b",
              "trusted_retrieval":r"(?i)retrieved.{0,30}(trusted|override)"}
    return [{"smell":name,"start":m.start(),"end":m.end(),"text":m.group()}
            for name,pat in patterns.items() for m in re.finditer(pat,prompt,re.I)]


def metamorphic_cases(case):
    """Semantics-preserving context transforms with unchanged independent labels."""
    chunks=case.context.split("\n\n")
    return [replace(case,id=case.id+"-reordered",context="\n\n".join(reversed(chunks))),
            replace(case,id=case.id+"-irrelevant",context=case.context+"\n\nOffice hours: Monday to Friday.")]


def metamorphic_verdict(original_calls, transformed_calls):
    return {"invariant":original_calls==transformed_calls,
            "note":"A violation may be stochastic; confirm over matched repeated runs. Equality does not imply correctness."}
