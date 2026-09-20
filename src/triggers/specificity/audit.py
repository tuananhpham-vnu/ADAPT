"""Audit selected strings without feeding audit-model scores into selection."""
from src.triggers.hierarchy.language_audit import measure
from src.triggers.hierarchy.text import protected_tokens
from .candidates import copy_metrics


def audit(rows, encoder, fluency):
    result = measure(rows, encoder, fluency)
    for row in result["records"]:
        row.update(copy_metrics(row["original"], row["trigger"]))
        row["original_retained_exactly"] = row["altered"].endswith(row["original"])
        row["protected_tokens_unchanged"] = protected_tokens(row["original"]) == protected_tokens(row["altered"])
    n = len(rows)
    result["metrics"].update({k: sum(r[k] for r in result["records"]) / n for k in
                              ("trigger_content_copy_fraction", "copied_query_word_fraction",
                               "original_retained_exactly", "protected_tokens_unchanged")})
    result["metrics"]["preservation_proxy_pass_rate"] = sum(r["original_altered_cosine"] >= .85 and r["protected_tokens_unchanged"] for r in result["records"]) / n
    result["metrics"]["selection_feasible_rate"] = sum(r.get("selection_feasible", True) for r in rows) / n
    return result


def render_report(results, selection):
    lines = ["# Specificity: language-only exploration", "",
             "Prefix only; upper bound 3 words / 12 selection-encoder tokens. Topic headings deliberately repeat query words.",
             "Recommendations were selected on validation before auditing fresh test queries.", "",
             "| Scope | Selected on validation |",
             "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in selection.items()]
    for scorer, splits in results.items():
        lines += ["", f"## {scorer}", "",
                  "| Split | Variant | Relevance | PPL ratio | Preservation proxy | Content copy fraction |",
                  "|---|---|---:|---:|---:|---:|"]
        for split, variants in splits.items():
            for variant, r in variants.items():
                m = r["metrics"]
                lines.append(f"| {split} | {variant} | {m['trigger_query_cosine']:.4f} | {m['geometric_mean_ppl_ratio']:.3f} | {m['preservation_proxy_pass_rate']:.3f} | {m['trigger_content_copy_fraction']:.3f} |")
    lines += ["", "## Limits", "",
              "- Language metrics only. No retrieval or downstream-action claim.",
              "- Query-adaptive search reads each incoming query and spends additional inference compute; frozen-bank baselines do not.",
              "- Copying a topic can raise both embedding similarities without adding information or removing redundancy.",
              "- Full-text PPL is a proxy, not a human rating; related audit-model families do not provide fully independent evidence.",
              "- Retaining the original string and similarity do not prove unchanged intent. No NLI/answer-preservation certification.",
              "- Group routing is kept fixed from the prior DPR experiment to isolate language changes."]
    return "\n".join(lines) + "\n"
