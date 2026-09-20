"""Readable paired results, hierarchy cuts, and crossover attribution."""


def render(result):
    lines = ["# Trigger hierarchy comparison", "", f"Fixture: {result['fixture']}", "",
             "Retrieval-only pilot. Meaning is an independent similarity proxy; fluency is unmeasured.", "",
             "| Position | Arm | Retrieval | Meaning pass | Joint success | False activation | Requests / cap |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for position, arms in result["results"].items():
        for arm, m in arms.items():
            lines.append(f"| {position} | {arm} | {m['retrieval_rate']:.3f} | {m['meaning_pass_rate']:.3f} | {m['joint_success_rate']:.3f} | {m['false_activation']:.3f} | {m['budget_used']}/{m['budget_limit']} |")
    lines += ["", "## Hierarchy trend on held-out queries", "",
              "Cuts are predefined. Leaf cuts have less training budget than the separate per-query baseline.", "",
              "| Position | Arm | Number of triggers | Joint success | Meaning pass |",
              "|---|---|---:|---:|---:|"]
    for position, arms in result["results"].items():
        for arm, m in arms.items():
            for k, splits in m["hierarchy_trend"].items():
                t = splits["test"]
                lines.append(f"| {position} | {arm} | {k} | {t['joint_success_rate']:.3f} | {t['meaning_pass_rate']:.3f} |")
    lines += ["", "## Crossover contribution during training", "",
              "Counts exclude duplicate proposals. Improvements are local training-loss improvements, not causal evidence of better test performance.", "",
              "| Position | Arm | Evaluated | Improved feasible best | Selected at root |",
              "|---|---|---:|---:|---|"]
    for position, arms in result["results"].items():
        for arm, m in arms.items():
            if arm.startswith("hierarchical"):
                c = m["crossover"]
                lines.append(f"| {position} | {arm} | {c['evaluated']} | {c['improved_feasible_best']} | {c['selected_at_root']} |")
    lines += ["", "## Limits", "", *[f"- {item}" for item in result["limitations"]]]
    return "\n".join(lines) + "\n"
