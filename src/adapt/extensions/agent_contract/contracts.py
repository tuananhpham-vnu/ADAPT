"""Executable contracts over actual upstream outputs; empirical reachability only."""
from dataclasses import dataclass,field


@dataclass(frozen=True)
class Contract:
    required_keys: dict[str,type] = field(default_factory=dict)
    forbidden_markers: tuple[str,...] = ()

    def check(self, output):
        errors=[]
        if self.required_keys:
            if not isinstance(output,dict):
                errors.append("OUTPUT_NOT_OBJECT")
            else:
                for key,typ in self.required_keys.items():
                    if key not in output or type(output[key]) is not typ:
                        errors.append("FIELD_"+key)
        for marker in self.forbidden_markers:
            if marker in str(output):errors.append("FORBIDDEN_MARKER:"+marker)
        return {"passed":not errors,"errors":errors}


def evaluate_edge(upstream, downstream, inputs, contract, candidate_inputs=()):
    rows=[]
    for value in inputs:
        actual=upstream(value)
        verdict=contract.check(actual)
        rows.append({"input":value,"upstream_output":actual,"contract":verdict,
                     "downstream_output":downstream(actual) if verdict["passed"] else None})
    return {"records":rows,"violation_rate":sum(not r["contract"]["passed"] for r in rows)/len(rows) if rows else None,
            "observed_reachability":sum(any(c==r["upstream_output"] for r in rows) for c in candidate_inputs)/len(candidate_inputs) if candidate_inputs else None,
            "note":"Unobserved candidate outputs are not proven unreachable."}
