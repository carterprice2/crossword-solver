"""Prompt construction.

Two prompts, and the difference between them is the point of the project.

The **first-pass** prompt asks for answers to clues with nothing else known.
The **repair** prompt asks again for a handful of slots, but now supplies the
letters the grid has since pinned down, the crossing answers that pinned them,
and the list of answers already ruled out. That extra context is what a human
solver uses on a second pass, and arm A3-minus-A2 measures what it is worth.
"""

from __future__ import annotations

import json

from ..model import Puzzle
from .constraints import SlotGraph

SYSTEM = """\
You are an expert crossword solver. You answer clues for an American-style \
crossword.

Rules:
- An answer is UPPERCASE LETTERS ONLY: no spaces, punctuation, or accents.
- The answer must be exactly the stated number of letters. This is absolute; a \
plausible answer of the wrong length is useless because it breaks every \
crossing entry.
- When a pattern is given, the answer must match it. "?" means unknown; any \
other character is a letter already fixed by a crossing answer.
- Give up to 5 candidates per clue, best first. Reason from the clue and the \
length; you are not given a word list.
- confidence is your honest probability that the candidate is correct, from \
0 to 1. Do not inflate it. A well-calibrated 0.4 is far more useful to the \
solver than a confident guess, because low-confidence answers are checked \
against crossings before being accepted.
- If you genuinely do not know, return fewer candidates rather than padding \
the list with filler.

Respond with JSON only, in this shape:
{"items": [{"id": "A1", "candidates": [{"answer": "OREO", "confidence": 0.82, \
"kind": "definition"}]}]}

kind is one of: definition, wordplay, fitb, abbrev, proper, crosswordese.
"""

SCHEMA_HINT = """\

Return a JSON object with an "items" array. Each element has "id" (the slot id \
exactly as given) and "candidates" (an array of at most 5 objects, each with \
"answer", "confidence", and optionally "kind").
"""


def _slot_payload(
    graph: SlotGraph,
    puzzle: Puzzle,
    slot_ids: list[str],
    patterns: dict[str, str],
    rejected: dict[str, list[str]] | None = None,
    issues: dict[str, list[str]] | None = None,
) -> list[dict]:
    rejected = rejected or {}
    issues = issues or {}
    items = []
    for slot_id in slot_ids:
        slot = puzzle.slot(slot_id)
        item: dict = {
            "id": slot_id,
            "clue": slot.clue,
            "len": slot.length,
        }
        pattern = patterns.get(slot_id)
        if pattern and pattern != "?" * slot.length:
            item["pattern"] = pattern
        already = rejected.get(slot_id)
        if already:
            item["rejected"] = sorted(already)[:12]
        notes = issues.get(slot_id)
        if notes:
            item["issues"] = notes[:6]
        items.append(item)
    return items


def first_pass_messages(
    puzzle: Puzzle,
    graph: SlotGraph,
    slot_ids: list[str],
    patterns: dict[str, str],
    *,
    schema_in_prompt: bool = False,
) -> list[dict]:
    payload = _slot_payload(graph, puzzle, slot_ids, patterns)
    title = puzzle.metadata.get("Title", "")
    lines = [
        f"Solve these {len(payload)} clues from a "
        f"{puzzle.grid.height}x{puzzle.grid.width} crossword.",
    ]
    if title and puzzle.metadata.get("Source") != "generated":
        lines.append(f'Puzzle: "{title}"')
    lines.append(
        "These clues cross each other in the grid, so their answers must be "
        "mutually consistent where they intersect."
    )
    lines.append("")
    lines.append(json.dumps({"slots": payload}, ensure_ascii=False))
    system = SYSTEM + (SCHEMA_HINT if schema_in_prompt else "")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(lines)},
    ]


def repair_messages(
    puzzle: Puzzle,
    graph: SlotGraph,
    slot_ids: list[str],
    patterns: dict[str, str],
    rejected: dict[str, list[str]],
    crossing_context: dict[str, list[tuple[str, str, str]]],
    *,
    likely: dict[str, str] | None = None,
    issues: dict[str, list[str]] | None = None,
    schema_in_prompt: bool = False,
) -> list[dict]:
    """Second-pass prompt: same clues, but now with what the grid has learned.

    Two patterns are supplied, and keeping them distinct matters. ``pattern``
    holds only letters confirmed by two agreeing confident entries -- a hard
    constraint. ``likely`` additionally includes letters from the current best
    but unconfirmed answers; those are usable hints and are labelled as such
    rather than being passed off as certain.
    """
    likely = likely or {}
    payload = _slot_payload(graph, puzzle, slot_ids, patterns, rejected, issues)
    for item in payload:
        context = crossing_context.get(item["id"]) or []
        if context:
            item["crossings"] = [
                {"id": other, "answer": answer, "clue": clue}
                for other, answer, clue in context[:6]
            ]
        guess = likely.get(item["id"])
        if guess and guess != item.get("pattern") and guess != "?" * item["len"]:
            item["likely"] = guess
    lines = [
        f"Second pass on {len(payload)} clues that are still unresolved.",
        "",
        "Since the first pass, crossing answers have fixed some letters. Each "
        "slot below shows:",
        "  pattern    - letters CONFIRMED by two agreeing entries ('?' = open)."
        " This is a hard constraint: the answer must match it.",
        "  likely     - the same, plus letters from crossing answers that are "
        "not yet confirmed. Strong hints, but any one of them may be wrong.",
        "  rejected   - answers already tried and ruled out; do not repeat them",
        "  issues     - why the current fill failed a check (length, blank, "
        "crossing, or not a real word)",
        "  crossings  - the crossing entries and what they currently say",
        "",
        "Use the pattern. It usually narrows the clue to a single answer even "
        "when the clue alone was ambiguous. If nothing fits the confirmed "
        "pattern, say so with low confidence rather than breaking it.",
        "",
        json.dumps({"slots": payload}, ensure_ascii=False),
    ]
    system = SYSTEM + (SCHEMA_HINT if schema_in_prompt else "")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(lines)},
    ]


def star_repair_messages(
    puzzle: Puzzle,
    graph: SlotGraph,
    star_slots: list[str],
    hubs: tuple[str, str],
    patterns: dict[str, str],
    rejected: dict[str, list[str]],
    current: dict[str, list[str]],
    *,
    likely: dict[str, str] | None = None,
    issues: dict[str, list[str]] | None = None,
    schema_in_prompt: bool = False,
) -> list[dict]:
    """Repair one clash: the two hubs plus the entries they touch.

    The model is asked to produce candidates that agree at every shared cell,
    not to re-answer a mixed bag of unrelated slots.
    """
    likely = likely or {}
    payload = _slot_payload(graph, puzzle, star_slots, patterns, rejected, issues)
    for item in payload:
        guess = likely.get(item["id"])
        if guess and guess != item.get("pattern") and guess != "?" * item["len"]:
            item["likely"] = guess
        have = current.get(item["id"]) or []
        if have:
            item["current"] = have[:6]
    crosses = []
    wanted = set(star_slots)
    for slot_id in star_slots:
        for crossing in graph.by_slot.get(slot_id, []):
            if crossing.b < slot_id or crossing.b not in wanted:
                continue
            crosses.append(
                {
                    "a": crossing.a,
                    "a_index": crossing.ai,
                    "b": crossing.b,
                    "b_index": crossing.bi,
                }
            )
    hub_a, hub_b = hubs
    lines = [
        f"Local repair around {hub_a} x {hub_b}.",
        "",
        "These entries share cells. A fill is only legal if every pair agrees "
        "on the shared letter AND every answer is a real word, abbreviation, "
        "or name -- not a leftover letter scrap.",
        "The current candidate lists do not mesh. Propose up to 5 NEW "
        "candidates per slot that can form a consistent local fill. "
        "Prefer an answer that agrees with a crossing candidate already listed.",
        "If you cannot satisfy a slot, return an empty candidate list for it "
        "rather than a guess that breaks a crossing.",
        "",
        "  pattern  - letters locked by two agreeing entries ('?' = open)",
        "  likely   - unconfirmed letters from other current answers",
        "  current  - candidates we already tried; they do not yet fit together",
        "  rejected - do not repeat these",
        "  crosses  - (slot, index) pairs that must hold the same letter",
        "",
        json.dumps(
            {"slots": payload, "crosses": crosses[:24]},
            ensure_ascii=False,
        ),
    ]
    system = SYSTEM + (SCHEMA_HINT if schema_in_prompt else "")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(lines)},
    ]


def whole_puzzle_messages(
    puzzle: Puzzle, known: dict | None = None
) -> list[dict]:
    """The naive baseline (arm A0): one prompt, whole grid, no loop."""
    # Clues are labelled with the full slot id rather than the bare number, so
    # the ids the model must echo back are the ones in front of it.
    across = [
        f"{s.id}. {s.clue} ({s.length})" for s in puzzle.slots if s.direction == "A"
    ]
    down = [
        f"{s.id}. {s.clue} ({s.length})" for s in puzzle.slots if s.direction == "D"
    ]
    grid = "\n".join(puzzle.grid.render(known=known, blank="."))
    content = (
        f"Solve this {puzzle.grid.height}x{puzzle.grid.width} crossword.\n\n"
        f"Grid ('#' is a blocked square, '.' is a letter to fill):\n{grid}\n\n"
        f"ACROSS\n" + "\n".join(across) + "\n\nDOWN\n" + "\n".join(down) + "\n\n"
        "Return JSON with every entry's answer:\n"
        '{"items": [{"id": "A1", "candidates": [{"answer": "...", '
        '"confidence": 0.9}]}]}\n'
        "Use the exact slot ids (A1, D1, ...) shown above, prefixed with A or D."
    )
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": content},
    ]


def blank_clue_messages(puzzle: Puzzle, slot_ids: list[str]) -> list[dict]:
    """Contamination probe: ask for answers with the clues withheld.

    If a model scores meaningfully above a length-and-letter-frequency prior
    here, it is recalling the puzzle rather than solving it.
    """
    payload = [
        {"id": sid, "len": puzzle.slot(sid).length, "clue": ""} for sid in slot_ids
    ]
    content = (
        f'Puzzle: "{puzzle.metadata.get("Title", "")}" '
        f'({puzzle.metadata.get("Date", "date unknown")})\n\n'
        "Give the answers for these entries. The clue text is not provided.\n\n"
        + json.dumps({"slots": payload}, ensure_ascii=False)
    )
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": content},
    ]
