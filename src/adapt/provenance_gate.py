"""Narrow deterministic policy for the demo's explicit English request grammar.

This checks authorization against a parsed user request, not causal provenance
inside a model. Unknown/ambiguous requests fail closed; retrieved text is never
an authorization source. All tools used by the demo are simulated.
"""
from __future__ import annotations

import math
import re


def authorized_call(query: str) -> dict | None:
    match = re.fullmatch(r"What is the balance of account (B\d+)\?", query)
    if match:
        return {"name": "lookup_balance", "args": {"account": match[1]}}
    match = re.fullmatch(r"Transfer (\d+(?:\.\d+)?) from account (B\d+) to account (B\d+)\.", query)
    if match and 0 < float(match[1]) < float("inf"):
        return {"name": "transfer_funds", "args": {
            "account": match[2], "amount": float(match[1]), "dest": match[3]}}
    return None


def check_calls(query: str, calls: list[dict]) -> dict:
    expected = authorized_call(query)
    reasons = []
    if expected is None:
        reasons.append("USER_INTENT_UNRESOLVED")
    if len(calls) != 1:
        reasons.append("TC_EXACTLY_ONE_CALL")
    for call in calls:
        name, args = call.get("name"), call.get("args")
        fields = {"lookup_balance": {"account"}, "transfer_funds": {"account", "amount", "dest"}}.get(name)
        if fields is None or not isinstance(args, dict) or set(args) != fields:
            reasons.append("TC_SCHEMA")
            continue
        if any(not isinstance(args[k], str) or not args[k] for k in fields - {"amount"}):
            reasons.append("TC_SCHEMA")
        if name == "transfer_funds":
            amount = args["amount"]
            if type(amount) not in (int, float) or not math.isfinite(amount) or amount <= 0:
                reasons.append("TC_AMOUNT_DOMAIN")
        if expected:
            if name != expected["name"]:
                reasons.append("TC_USER_AUTHORIZATION")
            for key, value in args.items():
                if key not in expected["args"] or value != expected["args"][key]:
                    reasons.append("GC_ARGUMENT_" + key.upper())
    return {"allowed": not reasons, "reasons": sorted(set(reasons)), "authorized": expected}
