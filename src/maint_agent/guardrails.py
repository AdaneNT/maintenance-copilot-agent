"""Responsible-AI guardrails: input safety screen, grounding check, human review flag.

These are deliberately simple, transparent rules. They don't replace a
content-safety service (e.g. Azure AI Content Safety); they encode
domain-specific rules for a safety-critical industrial setting:

1. The assistant is read-only and never helps bypass safety systems.
2. Every factual claim must be traceable to a tool output or a procedure.
3. Recommendations with operational consequences are flagged for a human.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

# ---------------------------------------------------------------- input guard

_SAFETY_SYSTEMS = r"(trip|trips|interlock|interlocks|alarm|alarms|esd|emergency shutdown|safety system|fire and gas|f&g|psv|relief valve)"
_DEFEAT_VERBS = r"(bypass|bypassing|disable|disabling|override|overriding|inhibit|inhibiting|defeat|silence|suppress|jumper|change|adjust)"

_UNSAFE_PATTERNS = [
    re.compile(rf"\b{_DEFEAT_VERBS}\b.{{0,40}}\b{_SAFETY_SYSTEMS}\b", re.I),
    re.compile(rf"\b{_SAFETY_SYSTEMS}\b.{{0,20}}\b(bypass|bypassed|override|overridden|disabled|inhibited)\b", re.I),
    re.compile(r"\b(start|stop|shut ?down|open|close|isolate)\b.{0,30}\b(the )?(pump|compressor|valve|equipment|unit)\b.{0,15}\b(now|for me|remotely)\b", re.I),
]
_INJECTION_PATTERNS = [
    re.compile(r"ignore (all |the )?(previous|prior|above) (instructions|rules)", re.I),
    re.compile(r"(reveal|print|show).{0,20}(system prompt|instructions)", re.I),
]

REFUSAL_UNSAFE = (
    "I can't help with bypassing or changing safety functions, or with operating equipment. "
    "Changes to trips, alarms or interlocks require an approved management-of-change process, "
    "and operational actions must be decided by operations and qualified personnel. "
    "I can summarise the equipment condition and the relevant procedures to support that discussion."
)
REFUSAL_INJECTION = "I can only help with questions about equipment condition, maintenance history and procedures."


@dataclass(frozen=True)
class InputCheck:
    allowed: bool
    category: str | None = None
    message: str | None = None


def check_input(text: str) -> InputCheck:
    for p in _INJECTION_PATTERNS:
        if p.search(text):
            return InputCheck(False, "prompt_injection", REFUSAL_INJECTION)
    for p in _UNSAFE_PATTERNS:
        if p.search(text):
            return InputCheck(False, "unsafe_operation", REFUSAL_UNSAFE)
    return InputCheck(True)


# ------------------------------------------------------------ grounding check

# Citations are written as [tool:...] or [doc:...]; variants are normalised below.
_BRACKET_RE = re.compile(r"\[([^\[\]\n]{1,200})\]")
_TOOL_NAMES = ("fleet_overview", "list_equipment", "sensor_summary", "detect_anomalies",
               "get_work_orders", "work_orders", "search_procedures")
_NUMBER_RE = re.compile(r"(?<![\w/#.-])-?\d+(?:\.\d+)?(?![\w/])")
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?")
_MONTH = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|"
          r"Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)")
_WRITTEN_DATE_RE = re.compile(
    rf"\b{_MONTH}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?\b"      # September 20th, 2026
    rf"|\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?{_MONTH}(?:,?\s+\d{{4}})?\b"  # 20 September 2026
    rf"|\b{_MONTH}\s+\d{{4}}\b",                                           # May 2026
    re.I,
)
# a number followed by one of these is a measured or derived value and is always checked
_UNIT_RE = re.compile(r"\s*(?:%|percent\b|mm/s|°\s*C|degrees|bar\b|m3/h|m³/h)", re.I)


def normalize_citation(raw: str) -> str | None:
    """Map citation variants to the canonical form, or None if the bracket is not a citation.

    Small models often write `[source: tool:x]` or drop the prefix (`[sensor_summary/P-102/...]`).
    Normalising them means valid variants are accepted *and* invented ones are checked,
    instead of silently ignored.
    """
    text = re.sub(r"^\s*sources?\s*:\s*", "", raw.strip(), flags=re.I).strip()
    if text.startswith(("tool:", "doc:")):
        return text
    if text.startswith(_TOOL_NAMES):
        return "tool:" + text
    return None


def find_citations(answer: str) -> list[str]:
    return [c for c in (normalize_citation(m) for m in _BRACKET_RE.findall(answer)) if c]


def _strip_citations(answer: str) -> str:
    return _BRACKET_RE.sub(lambda m: " " if normalize_citation(m.group(1)) else m.group(0), answer)


def _walk(obj: Any) -> Iterable[Any]:
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _walk(v)
    else:
        yield obj


def collect_sources(tool_outputs: list[dict]) -> set[str]:
    sources: set[str] = set()

    def visit(o: Any) -> None:
        if isinstance(o, dict):
            if isinstance(o.get("source"), str):
                sources.add(o["source"])
            for v in o.values():
                visit(v)
        elif isinstance(o, list):
            for v in o:
                visit(v)

    for out in tool_outputs:
        visit(out)
    return sources


def collect_numbers(tool_outputs: list[dict]) -> list[float]:
    nums: list[float] = []
    for v in _walk(tool_outputs):
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            nums.append(float(v))
        elif isinstance(v, str):
            nums.extend(float(m) for m in _NUMBER_RE.findall(_DATE_RE.sub(" ", v)))
    return nums


def _claims(answer: str) -> list[tuple[float, bool]]:
    """(value, has_unit) for every number stated in the answer, excluding citations,
    dates, equipment tags and list numbering."""
    text = _strip_citations(answer)
    text = _DATE_RE.sub(" ", text)
    text = _WRITTEN_DATE_RE.sub(" ", text)
    text = re.sub(r"\b[A-Z]{1,3}-\d+\b", " ", text)  # equipment ids like P-102, WO-1005
    text = re.sub(r"^\s*\d+[.)]\s", " ", text, flags=re.M)  # list numbering
    return [(float(m.group()), bool(_UNIT_RE.match(text, m.end()))) for m in _NUMBER_RE.finditer(text)]


def extract_claimed_numbers(answer: str) -> list[float]:
    return [v for v, _ in _claims(answer)]


@dataclass
class GroundingReport:
    has_citations: bool
    unknown_citations: list[str] = field(default_factory=list)
    ungrounded_numbers: list[float] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.has_citations and not self.unknown_citations and not self.ungrounded_numbers

    def feedback(self, valid_sources: Iterable[str] = ()) -> str:
        """Actionable revision instructions. Listing the valid source ids lets the model
        repair its citations instead of guessing again."""
        problems = []
        if not self.has_citations:
            problems.append("the answer has no citations; cite sources as [tool:...] or [doc:...]")
        if self.unknown_citations:
            problems.append(
                f"these citations do not match any retrieved source: {self.unknown_citations}; "
                "remove them, or call the tool first and then cite the source it returns"
            )
        if self.ungrounded_numbers:
            problems.append(
                f"these numbers do not appear in any tool output or procedure: {self.ungrounded_numbers}; "
                "remove them or call a tool to obtain them"
            )
        text = "Revise your answer: " + "; ".join(problems) + "."
        valid = sorted(valid_sources)
        if valid:
            text += " The only valid citations are: " + ", ".join(f"[{s}]" for s in valid) + "."
        else:
            text += (
                " No tools have been called yet. Call the relevant tools first (search_procedures for "
                "procedures and limits, sensor_summary or detect_anomalies for equipment data), then answer "
                "citing the sources they return."
            )
        return text


def _is_grounded(x: float, known: list[float], has_unit: bool = False, rel_tol: float = 0.02) -> bool:
    # bare small integers (counts, steps, days) are too ambiguous to check reliably,
    # but a value with a unit ("15%", "7 mm/s") is a measurement claim and is always checked
    if x.is_integer() and abs(x) <= 31 and not has_unit:
        return True
    return any(abs(x - k) <= max(rel_tol * abs(k), 0.011) for k in known)


def check_grounding(answer: str, tool_outputs: list[dict]) -> GroundingReport:
    cited = find_citations(answer)
    known_sources = collect_sources(tool_outputs)
    known_numbers = collect_numbers(tool_outputs)
    return GroundingReport(
        has_citations=bool(cited),
        unknown_citations=sorted({c for c in cited if c not in known_sources}),
        ungrounded_numbers=sorted({v for v, unit in _claims(answer) if not _is_grounded(v, known_numbers, unit)}),
    )


# ---------------------------------------------------------- human review flag

_REVIEW_RE = re.compile(
    r"\b(shut ?down|shutdown|stop the|isolat\w+|replace\w*|bearing replacement|zone d|trip|"
    r"take (it|the \w+) out of service|reduce (load|speed)|immediate\w*|urgent\w*|intrusive|"
    r"notify the (area )?technician)\b",
    re.I,
)


def needs_human_review(answer: str) -> bool:
    """Flag answers that recommend actions with operational or safety consequences."""
    return bool(_REVIEW_RE.search(_strip_citations(answer)))
