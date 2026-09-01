"""Structured-output schemas and a parser that survives real model output.

Nebius Token Factory supports guided JSON through the OpenAI-compatible
``response_format`` field, but support is uneven across the 60+ models it
serves, and reasoning models emit their scratchpad before the JSON. Two
mechanisms handle that:

* a **degradation ladder** -- if a model rejects a strict schema, retry with a
  progressively weaker constraint rather than losing the request entirely, and
  record which rung it landed on (that is a reportable finding in itself);
* a **lenient parser** -- strip reasoning blocks, code fences and trailing
  commas before parsing, then validate every candidate against the slot it
  claims to answer.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .normalize import CLUE_TYPES, normalize_answer

#: Cap on candidates per slot. The search benefits from a longer list;
#: the schema must allow what the prompt asks for.
MAX_CANDIDATES = 8

#: Rungs of the degradation ladder, strongest first.
STRICT = "strict_schema"
LOOSE = "loose_schema"
NO_STRICT = "schema_no_strict"
JSON_OBJECT = "json_object"
FREE_TEXT = "free_text"

LADDER = (STRICT, LOOSE, NO_STRICT, JSON_OBJECT, FREE_TEXT)

_THINK_RE = re.compile(
    r"<(think|thinking|reasoning|scratchpad)>.*?</\1>", re.DOTALL | re.IGNORECASE
)
_UNCLOSED_THINK_RE = re.compile(
    r"^.*?</(think|thinking|reasoning|scratchpad)>", re.DOTALL | re.IGNORECASE
)
_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


def candidates_schema(*, strict: bool = True, constrained: bool = True) -> dict:
    """The response schema for a batch of clue answers."""
    answer: dict = {"type": "string"}
    if constrained:
        answer["pattern"] = "^[A-Z]+$"
    candidate: dict = {
        "type": "object",
        "additionalProperties": False,
        "required": ["answer", "confidence"],
        "properties": {
            "answer": answer,
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "kind": {"type": "string", "enum": list(CLUE_TYPES)},
        },
    }
    candidate_list: dict = {"type": "array", "items": candidate}
    if constrained:
        candidate_list["maxItems"] = MAX_CANDIDATES
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["items"],
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["id", "candidates"],
                    "properties": {
                        "id": {"type": "string"},
                        "candidates": candidate_list,
                    },
                },
            }
        },
    }
    body: dict = {"name": "crossword_candidates", "schema": schema}
    if strict:
        body["strict"] = True
    return {"type": "json_schema", "json_schema": body}


def response_format_for(rung: str) -> dict | None:
    """The ``response_format`` payload for a given rung of the ladder."""
    if rung == STRICT:
        return candidates_schema(strict=True, constrained=True)
    if rung == LOOSE:
        return candidates_schema(strict=True, constrained=False)
    if rung == NO_STRICT:
        return candidates_schema(strict=False, constrained=False)
    if rung == JSON_OBJECT:
        return {"type": "json_object"}
    return None


def strip_reasoning(text: str) -> str:
    """Remove reasoning blocks and code fences, leaving the payload."""
    cleaned = _THINK_RE.sub("", text)
    # A truncated reasoning block leaves a closing tag with no opener.
    if re.search(r"</(think|thinking|reasoning|scratchpad)>", cleaned, re.IGNORECASE):
        cleaned = _UNCLOSED_THINK_RE.sub("", cleaned)
    fenced = _FENCE_RE.search(cleaned)
    if fenced:
        return fenced.group(1).strip()
    return cleaned.strip()


def extract_json(text: str) -> dict | None:
    """Find and parse the first balanced JSON object in a blob of text."""
    cleaned = strip_reasoning(text)
    for attempt in (cleaned, _TRAILING_COMMA_RE.sub(r"\1", cleaned)):
        try:
            value = json.loads(attempt)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
    # Fall back to scanning for a balanced object, ignoring braces in strings.
    start = cleaned.find("{")
    while start != -1:
        depth, in_string, escaped = 0, False, False
        for i in range(start, len(cleaned)):
            ch = cleaned[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    chunk = cleaned[start : i + 1]
                    for attempt in (chunk, _TRAILING_COMMA_RE.sub(r"\1", chunk)):
                        try:
                            value = json.loads(attempt)
                            if isinstance(value, dict):
                                return value
                        except json.JSONDecodeError:
                            pass
                    break
        start = cleaned.find("{", start + 1)
    return None


@dataclass(frozen=True)
class Candidate:
    """One proposed answer for one slot."""

    slot_id: str
    answer: str
    confidence: float
    kind: str = "definition"
    #: How many independent models proposed this answer.
    sources: int = 1

    def fits(self, length: int, pattern: str | None = None) -> bool:
        if len(self.answer) != length:
            return False
        if pattern:
            for got, want in zip(self.answer, pattern):
                if want not in ("?", got):
                    return False
        return True


def parse_candidates(
    text: str,
    *,
    expected: dict[str, int],
    patterns: dict[str, str] | None = None,
) -> tuple[list[Candidate], list[str]]:
    """Turn raw model output into validated candidates.

    ``expected`` maps slot id to required answer length; anything for an
    unknown slot, of the wrong length, or contradicting a known pattern is
    dropped. Returns (candidates, warnings) -- warnings feed the trace so the
    report can say how often each model needed rescuing.
    """
    warnings: list[str] = []
    payload = extract_json(text)
    if payload is None:
        return [], ["no JSON object found in response"]

    items = payload.get("items")
    if not isinstance(items, list):
        # Some models answer with a bare {slot_id: answer} mapping.
        if all(isinstance(v, str) for v in payload.values()) and payload:
            items = [
                {"id": k, "candidates": [{"answer": v, "confidence": 0.5}]}
                for k, v in payload.items()
            ]
            warnings.append("response used a flat mapping instead of 'items'")
        else:
            return [], ["response has no 'items' array"]

    patterns = patterns or {}
    best: dict[tuple[str, str], Candidate] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        slot_id = str(item.get("id", "")).strip().upper()
        if slot_id not in expected:
            warnings.append(f"unknown slot {slot_id!r}")
            continue
        raw_candidates = item.get("candidates")
        if isinstance(raw_candidates, (str, dict)):
            raw_candidates = [raw_candidates]
        if not isinstance(raw_candidates, list):
            continue
        for raw in raw_candidates:
            if isinstance(raw, str):
                raw = {"answer": raw, "confidence": 0.5}
            if not isinstance(raw, dict):
                continue
            answer = normalize_answer(str(raw.get("answer", "")))
            if not answer:
                continue
            try:
                confidence = float(raw.get("confidence", 0.5))
            except (TypeError, ValueError):
                confidence = 0.5
            # Clamp away from 0 and 1: the search works in log space, and a
            # model asserting certainty should not be able to veto backtracking.
            confidence = min(0.99, max(0.01, confidence))
            kind = str(raw.get("kind", "definition"))
            if kind not in CLUE_TYPES:
                kind = "definition"
            candidate = Candidate(slot_id, answer, confidence, kind)
            if not candidate.fits(expected[slot_id], patterns.get(slot_id)):
                warnings.append(
                    f"{slot_id}: {answer!r} does not fit "
                    f"({expected[slot_id]} cells, pattern {patterns.get(slot_id, '')})"
                )
                continue
            key = (slot_id, answer)
            existing = best.get(key)
            if existing is None or candidate.confidence > existing.confidence:
                best[key] = candidate

    missing = sorted(set(expected) - {slot for slot, _ in best})
    if missing:
        warnings.append(f"no usable candidate for {len(missing)} slot(s): {missing[:6]}")
    return sorted(
        best.values(), key=lambda c: (c.slot_id, -c.confidence, c.answer)
    ), warnings


def merge_candidates(
    groups: list[list[Candidate]], *, agreement_bonus: bool = True
) -> list[Candidate]:
    """Combine candidate lists from independent models.

    Two models agreeing is the strongest signal available here, so shared
    answers combine as noisy-OR rather than by averaging: independent
    confirmations should raise confidence, not dilute it.
    """
    merged: dict[tuple[str, str], Candidate] = {}
    for group in groups:
        for candidate in group:
            key = (candidate.slot_id, candidate.answer)
            existing = merged.get(key)
            if existing is None:
                merged[key] = candidate
                continue
            if agreement_bonus:
                combined = 1.0 - (1.0 - existing.confidence) * (
                    1.0 - candidate.confidence
                )
            else:
                combined = max(existing.confidence, candidate.confidence)
            merged[key] = Candidate(
                candidate.slot_id,
                candidate.answer,
                min(0.99, combined),
                existing.kind,
                existing.sources + candidate.sources,
            )
    return sorted(merged.values(), key=lambda c: (c.slot_id, -c.confidence, c.answer))
