"""Read and write the ``.xd`` crossword format.

Spec: https://github.com/century-arcade/xd/blob/master/doc/xd-format.md

A file is three sections -- metadata, grid, clues -- separated either by two or
more blank lines or by explicit ``## Section`` headers. Rather than trust the
ordering, we sniff each section by shape, which handles the many small
deviations found in real corpus files.

    Title: New York Times, Saturday, January 1, 1955
    Author: Anthony Morse
    Rebus: 1=HEART,2=DIAMOND


    1ACHE#ADAM#2LIL
    BLUER#GULL#MATA


    A1. Sadness. ~ HEARTACHE
    D1. Vital throb. ~ HEARTBEAT
"""

from __future__ import annotations

import re
from typing import Iterable

from .model import ACROSS, BLOCK, DOWN, Cell, Grid, Puzzle, PuzzleError, Slot

#: ``A17. Some clue ~ ANSWER``
CLUE_RE = re.compile(r"^\s*([AD])(\d+)\.\s*(.*)$")
HEADER_RE = re.compile(r"^\s*##\s*(.+?)\s*$", re.MULTILINE)
META_RE = re.compile(r"^([A-Za-z][A-Za-z0-9 _-]*):\s*(.*)$")
SECTION_SPLIT_RE = re.compile(r"\n\s*\n\s*\n+")
#: The clue/answer separator. Answers never contain " ~ ", clues sometimes do.
ANSWER_SEP = " ~ "


def _looks_like_grid(lines: list[str]) -> bool:
    if not lines:
        return False
    if any(CLUE_RE.match(ln) for ln in lines):
        return False
    if any(META_RE.match(ln) for ln in lines):
        return False
    widths = {len(ln) for ln in lines}
    return len(widths) == 1 and all(
        ch == BLOCK or ch.isalnum() or ch == "." for ln in lines for ch in ln
    )


def _parse_rebus(spec: str) -> dict[str, str]:
    """``1=HEART,2=DIAMOND`` or ``1=HEART 2=DIAMOND`` -> {"1": "HEART", ...}"""
    out: dict[str, str] = {}
    for part in re.split(r"[,\s]+", spec.strip()):
        if not part:
            continue
        key, _, value = part.partition("=")
        if value:
            out[key.strip()] = value.strip().upper()
    return out


def _split_sections(text: str) -> dict[str, list[str]]:
    """Group lines into metadata / grid / clues, honoring ## headers if present."""
    buckets: dict[str, list[str]] = {"metadata": [], "grid": [], "clues": []}

    def classify(lines: list[str]) -> str | None:
        if not lines:
            return None
        if any(CLUE_RE.match(ln) for ln in lines):
            return "clues"
        if _looks_like_grid(lines):
            return "grid"
        if any(META_RE.match(ln) for ln in lines):
            return "metadata"
        return None

    if HEADER_RE.search(text):
        current: str | None = None
        explicit: dict[str, list[str]] = {}
        for raw in text.splitlines():
            header = HEADER_RE.fullmatch(raw.rstrip())
            if header:
                name = header.group(1).lower()
                current = name if name in buckets else None
                if current:
                    explicit.setdefault(current, [])
                continue
            if current and raw.strip():
                explicit[current].append(raw.rstrip())
        if explicit:
            for name, lines in explicit.items():
                buckets[name].extend(lines)
            return buckets

    for chunk in SECTION_SPLIT_RE.split(text.strip("\n")):
        lines = [ln.rstrip() for ln in chunk.splitlines() if ln.strip()]
        kind = classify(lines)
        if kind:
            buckets[kind].extend(lines)
    return buckets


def parse_xd(text: str, *, puzzle_id: str | None = None) -> Puzzle:
    """Parse ``.xd`` source into a Puzzle, gold answers included when present."""
    buckets = _split_sections(text)
    grid_lines, clue_lines = buckets["grid"], buckets["clues"]
    if not grid_lines:
        raise PuzzleError("no grid section found")

    metadata: dict[str, str] = {}
    for line in buckets["metadata"]:
        m = META_RE.match(line)
        if m:
            metadata[m.group(1).strip()] = m.group(2).strip()
    if puzzle_id:
        metadata["id"] = puzzle_id

    rebus_map = _parse_rebus(metadata.get("Rebus", ""))

    width = len(grid_lines[0])
    if any(len(ln) != width for ln in grid_lines):
        raise PuzzleError("grid rows have inconsistent widths")

    blocks: set[Cell] = set()
    letters: dict[Cell, str] = {}
    rebus: dict[Cell, str] = {}
    for r, row in enumerate(grid_lines):
        for c, ch in enumerate(row):
            cell = (r, c)
            if ch == BLOCK:
                blocks.add(cell)
            elif ch in rebus_map:
                value = rebus_map[ch]
                letters[cell] = value
                rebus[cell] = value
            elif ch == ".":
                pass  # solution withheld
            else:
                letters[cell] = ch.upper()

    grid = Grid(len(grid_lines), width, blocks)
    clues: dict[str, str] = {}
    golds: dict[str, str] = {}
    for line in clue_lines:
        m = CLUE_RE.match(line)
        if not m:
            continue
        direction, number, rest = m.group(1), int(m.group(2)), m.group(3)
        slot_id = f"{direction}{number}"
        # The spec reserves "~" as the clue/answer separator, so the last one
        # wins. Splitting on the bare character (not " ~ ") keeps a clue-less
        # line like "A1. ~ HEARTACHE" parsing correctly.
        clue, sep, answer = rest.rpartition("~")
        if not sep:  # no gold answer on this line
            clue, answer = rest, ""
        clues[slot_id] = clue.strip()
        if answer.strip():
            golds[slot_id] = answer.strip().upper()

    slots: list[Slot] = []
    for slot in grid.slots():
        gold = golds.get(slot.id)
        if gold is None and letters:
            spelled = slot.spell(letters)
            gold = spelled
        slots.append(
            Slot(
                id=slot.id,
                direction=slot.direction,
                number=slot.number,
                cells=slot.cells,
                clue=clues.get(slot.id, ""),
                gold=gold,
            )
        )
    return Puzzle(grid=grid, slots=slots, metadata=metadata, rebus=rebus)


def dump_xd(puzzle: Puzzle, *, include_answers: bool = True) -> str:
    """Serialize a Puzzle back to ``.xd``. Round-trips with :func:`parse_xd`."""
    lines: list[str] = []
    rebus_codes: dict[str, str] = {}
    if puzzle.rebus:
        values = sorted({v for v in puzzle.rebus.values()})
        rebus_codes = {v: str(i + 1) for i, v in enumerate(values)}

    meta = {k: v for k, v in puzzle.metadata.items() if k != "Rebus"}
    for key, value in meta.items():
        lines.append(f"{key}: {value}")
    if rebus_codes:
        lines.append(
            "Rebus: " + ",".join(f"{code}={val}" for val, code in rebus_codes.items())
        )

    known = puzzle.gold_solution() if puzzle.has_gold() and include_answers else {}
    grid_rows = []
    for r in range(puzzle.grid.height):
        row = []
        for c in range(puzzle.grid.width):
            cell = (r, c)
            if puzzle.grid.is_block(cell):
                row.append(BLOCK)
                continue
            value = known.get(cell)
            if value is None:
                row.append(".")
            elif len(value) > 1:
                row.append(rebus_codes[value])
            else:
                row.append(value)
        grid_rows.append("".join(row))

    out = "\n".join(lines) + "\n\n\n" + "\n".join(grid_rows) + "\n\n\n"
    for direction in (ACROSS, DOWN):
        group = [s for s in puzzle.slots if s.direction == direction]
        for slot in sorted(group, key=lambda s: s.number):
            line = f"{slot.id}. {slot.clue}".rstrip()
            if include_answers and slot.gold:
                line += f"{ANSWER_SEP}{slot.gold}"
            out += line + "\n"
        out += "\n"
    return out.rstrip("\n") + "\n"


def load_xd(path: str) -> Puzzle:
    import os

    with open(path, encoding="utf-8") as fh:
        return parse_xd(fh.read(), puzzle_id=os.path.splitext(os.path.basename(path))[0])


def load_many(paths: Iterable[str]) -> list[Puzzle]:
    return [load_xd(p) for p in paths]
