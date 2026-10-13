"""Bounded real-callback graph runner; records contracts, early stops and propagation."""
from collections import Counter


def run_graph(nodes, router, start, value, *, terminals, max_steps=30, contracts=None):
    if start not in nodes or max_steps<1 or not set(terminals)<=set(nodes):
        raise ValueError("Invalid graph entry/terminals/budget")
    trace=[]
    current=start
    for _ in range(max_steps):
        before=value
        try:
            value=nodes[current](value)
            contract=(contracts or {}).get(current)
            verdict=contract.check(value) if contract else {"passed":True,"errors":[]}
            trace.append({"node":current,"input":before,"output":value,"contract":verdict})
            if not verdict["passed"]:
                return {"status":"contract_blocked","trace":trace,"output":None}
            if current in terminals:
                return {"status":"completed","trace":trace,"output":value}
            next_node=router(current,value)
            if next_node is None:
                return {"status":"early_stop","trace":trace,"output":value}
            if next_node not in nodes:
                return {"status":"invalid_route","trace":trace,"output":value}
            current=next_node
        except Exception as exc:
            trace.append({"node":current,"error_type":type(exc).__name__})
            return {"status":"error","trace":trace,"output":None}
    return {"status":"step_limit","trace":trace,"output":None}


def summarize_graph(runs, expected_paths=()):
    paths=Counter(tuple(s["node"] for s in r["trace"]) for r in runs)
    target={tuple(p) for p in expected_paths}
    return {"n_runs":len(runs),"statuses":dict(Counter(r["status"] for r in runs)),
            "observed_paths":[{"path":list(p),"count":n} for p,n in paths.items()],
            "declared_path_coverage":len(set(paths)&target)/len(target) if target else None}


def propagation(pairs):
    """Pairs carry independently labeled source_failure and sink_failure booleans."""
    affected=[p for p in pairs if p["source_failure"]]
    return {"source_failures":len(affected),"propagation_rate":sum(p["sink_failure"] for p in affected)/len(affected) if affected else None}
