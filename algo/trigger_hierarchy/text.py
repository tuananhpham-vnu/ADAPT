"""Insertion and bounded-length crossover shared by every experiment arm."""
import re

SEEDS = ("use relevant information", "consider available evidence", "check useful details",
         "review related context", "consider practical details", "use reliable context",
         "check relevant evidence", "review available information")


def insertion_parts(question, position):
    if not question.strip():
        raise ValueError("Question must be nonempty")
    if position == "suffix":
        left, right = question.rstrip() + " ", ""
    elif position == "prefix":
        left, right = "", " " + question.lstrip()
    elif position == "sentence":
        match = re.search(r"[.!?;]\s+", question)
        left, right = (question[:match.end()], " " + question[match.end():]) if match else ("", " " + question.lstrip())
    elif position == "infix":
        boundaries = [m for m in re.finditer(r"\s+", question) if m.start() > 0 and m.end() < len(question)]
        if not boundaries:
            left, right = "", " " + question.lstrip()
        else:
            match = min(boundaries, key=lambda m: abs(m.start() - len(question) / 2))
            left, right = question[:match.end()], " " + question[match.end():]
    else:
        raise ValueError("Unknown insertion position")
    return left, right


def insert(question, trigger, position):
    if not trigger.strip():
        raise ValueError("Trigger must be nonempty")
    left, right = insertion_parts(question, position)
    return left + trigger + right


def actual_position(question, position):
    left, right = insertion_parts(question, position)
    return "prefix" if not left else "suffix" if not right else "infix"


def proposals(trigger, rng, max_words, parents=(), style="template"):
    """Word-coordinate search plus restarts and fixed-length parent crossover.

    This is a bounded discrete-search pilot, not a HotFlip reproduction.
    """
    words = trigger.split()
    vocabulary = sorted({w for seed in SEEDS for w in seed.split()} | {w for p in parents for w in p.split()})
    # Evaluate genuine recombinations early enough that small budgets can reach
    # them. Slot-preserving crossover does not make a longer trigger.
    for parent in parents:
        other = parent.split()
        for cut in range(1, min(len(words), len(other))):
            yield " ".join((words[:cut] + other[cut:])[:max_words]), "crossover"
    for i in range(len(words)):
        changed = words.copy()
        choices = sorted({s.split()[i] for s in SEEDS}) if style == "template" else vocabulary
        changed[i] = rng.choice(choices)
        yield " ".join(changed[:max_words]), "mutation"
    yield rng.choice(SEEDS), "restart"


def candidates(trigger, rng, max_words, parents=()):
    return (text for text, _ in proposals(trigger, rng, max_words, parents))


def protected_tokens(text):
    # A guard against obvious changes, not a semantic-equivalence certificate.
    return re.findall(r"\b(?:\d+(?:\.\d+)?|not|never|without|no)\b", text.casefold())
