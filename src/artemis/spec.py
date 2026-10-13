"""Serializable experiment contracts. Expected calls are labels, never model input."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import re


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class Requirement:
    id: str
    kind: str
    text: str
    repair: str
    mutants: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in {"T", "AC", "CC", "TC", "GC"} or not re.fullmatch(r"[A-Z][A-Z0-9_]*", self.id):
            raise ValueError("Invalid requirement kind/id")
        if not self.text.strip() or not self.repair.strip():
            raise ValueError("Requirement and repair must not be empty")


@dataclass(frozen=True)
class Case:
    id: str
    group: str
    split: str
    query: str
    expected: list[dict]
    family: str = "clean"
    context: str = ""
    trigger: str = ""
    factors: dict[str, str] = field(default_factory=dict)
    tool_note: str = ""
    text_criteria: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.split not in {"train", "validation", "heldout"}:
            raise ValueError("split must be train/validation/heldout")
        if not self.id or not self.group or not self.query:
            raise ValueError("Case identity/query required")
        if not isinstance(self.expected, list):
            raise ValueError("Expected calls must be a list")

    @property
    def adversarial(self):
        return self.family != "clean"


@dataclass(frozen=True)
class Prompt:
    intro: str
    requirements: tuple[Requirement, ...]
    overrides: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        ids = [r.id for r in self.requirements]
        if len(ids) != len(set(ids)) or not set(self.overrides) <= set(ids):
            raise ValueError("Duplicate or unknown requirement ID")

    def render(self):
        lines = [self.intro]
        for r in self.requirements:
            text = self.overrides.get(r.id, r.text)
            if text:
                lines.append(f"[{r.id}] {text}")
        return "\n".join(lines)

    def replace(self, requirement_id, text):
        if requirement_id not in {r.id for r in self.requirements}:
            raise ValueError("Unknown requirement")
        return Prompt(self.intro, self.requirements, {**self.overrides, requirement_id: text})


def validate_suite(cases: list[Case], heldout_families=()):
    ids, groups = set(), {}
    for c in cases:
        if c.id in ids:
            raise ValueError("Duplicate case ID")
        ids.add(c.id)
        if c.group in groups and groups[c.group] != c.split:
            raise ValueError("Related queries cross data splits")
        groups[c.group] = c.split
        if c.family in heldout_families and c.split != "heldout":
            raise ValueError("Held-out attack family leaked into development")
    if not cases:
        raise ValueError("Empty benchmark")


def dump_spec(prompt, cases):
    return {"prompt": asdict(prompt), "cases": [asdict(c) for c in cases]}
