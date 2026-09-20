"""Grounded short phrases and a renderer that leaves the entire query intact."""
import re

from src.triggers.hierarchy.text import SEEDS, insert, insertion_parts, protected_tokens

# Heuristic content spans, not a POS parser. Never join across removed words.
STOP = set("a an the is are was were be been being do does did can could will would shall should may might must have has had of in on at to for from by with without and or but if than that this these those it its their his her they we you i what which who whom whose when where why how not no never more less most least ever during within about regarding concerning as into over under between through all any some much many make made get got know known believe live survive produce use used fight suffer".split())


def words(text):
    return re.findall(r"[A-Za-z]+(?:['’-][A-Za-z]+)*|\d+(?:\.\d+)?", text)


def phrases(query):
    tokens = words(query)
    spans = []
    for n in (1, 2):
        for i in range(len(tokens) - n + 1):
            part = tokens[i:i + n]
            phrase = " ".join(part)
            if n / len(tokens) > .5 or any(w.casefold() in STOP for w in part):
                continue
            if protected_tokens(phrase) or any(w.isdigit() for w in part):
                continue
            if n == 1 and len(phrase) < 3:
                continue
            spans.append(phrase)
    return list(dict.fromkeys(spans))


def pool(queries, mode, phrase_extractor=phrases):
    if mode == "generic_language":
        return list(SEEDS)
    spans = list(dict.fromkeys(p for q in queries for p in phrase_extractor(q)))
    if mode == "topic_heading":
        # Explicit copy-based control: topicality is partly obtained by repetition.
        return spans or list(SEEDS)
    if mode != "grounded_language":
        raise ValueError("Unknown candidate mode")
    return list(dict.fromkeys([*SEEDS, *(f"{lead} {p}" for p in spans for lead in ("Regarding", "About"))]))


def render(query, trigger, style="heading", position="prefix"):
    """Put the trigger into the query without removing a single query word.

    `raw` is the untouched bank rendering: the trigger spliced in at `position`
    exactly as the retrieval optimizer saw it. `heading` is the punctuated variant
    the language stage compares against -- at prefix that is the familiar
    "Topic: question", and at any other position the trigger becomes a parenthetical
    at the same insertion point, which is the closest punctuated form that still
    leaves the question itself intact.
    """
    if not query.strip() or not trigger.strip():
        raise ValueError("Nonempty query and trigger required")
    if style == "raw":
        return insert(query, trigger, position)
    if style != "heading":
        raise ValueError("Unknown renderer")
    capitalized = trigger[:1].upper() + trigger[1:]
    if position == "prefix":
        return capitalized + ": " + query
    left, right = insertion_parts(query, position)
    # `sentence` degrades to prefix when the query has no internal sentence break, and
    # `prefix` leaves nothing on the left, so never emit a leading space.
    head = left.rstrip() + " " if left.strip() else ""
    return head + "(" + capitalized + ")" + (right or "")


def copy_metrics(query, trigger):
    q = {w.casefold() for w in words(query) if w.casefold() not in STOP}
    t = [w.casefold() for w in words(trigger) if w.casefold() not in STOP]
    copied = [w for w in t if w in q]
    return {"copied_content_words": len(copied), "trigger_content_copy_fraction": len(copied) / max(1, len(t)),
            "copied_query_word_fraction": len(set(copied)) / max(1, len(words(query)))}
