"""Turn a grid mask plus clue text into a Puzzle.

Used by the web ingest API. No HTTP, no vision client — those live in
``crossword.api``. Letters in the mask become solver prefill, never gold.
"""

from __future__ import annotations

import base64
import os
import re
from dataclasses import dataclass, field, replace

from .client import DEFAULT_VISION_MODEL
from .model import ACROSS, BLOCK, DOWN, Cell, Grid, Puzzle, PuzzleError, Slot

MAX_SIZE = 15
MIN_SIZE = 2
MAX_IMAGE_BYTES = 4 * 1024 * 1024
PNG_MAGIC = b"\x89PNG"
JPEG_MAGIC = b"\xff\xd8"

#: ``1. clue``, ``A1. clue``, ``D14) clue``
NUMBERED_RE = re.compile(r"^\s*([ADad])?\s*(\d+)[.)]\s+(.*)$")


class IngestError(ValueError):
    """The mask or clue list is unusable (not a recoverable mismatch)."""


@dataclass
class IngestDraft:
    """Result of assembling a mask with Across/Down text."""

    status: str
    puzzle: Puzzle | None = None
    prefill: dict[Cell, str] = field(default_factory=dict)
    rows: list[str] = field(default_factory=list)
    across_slots: int = 0
    down_slots: int = 0
    across_clues: int = 0
    down_clues: int = 0
    unknown_numbers: list[str] = field(default_factory=list)
    message: str = ""


def parse_clue_list(text: str, *, direction: str) -> list[tuple[str, str]]:
    """Parse a textarea into ``(slot_id, clue)`` pairs.

    Numbered when any non-blank line matches ``1.`` / ``A1.`` / ``D2)``.
    Otherwise sequential: line *i* becomes ``{direction}{i+1}``.
    """
    direction = direction.upper()
    if direction not in (ACROSS, DOWN):
        raise IngestError(f"direction must be A or D, got {direction!r}")
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    if not lines:
        return []
    numbered = any(NUMBERED_RE.match(ln) for ln in lines)
    if not numbered:
        return [(f"{direction}{i + 1}", ln) for i, ln in enumerate(lines)]
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for ln in lines:
        match = NUMBERED_RE.match(ln)
        if match is None:
            raise IngestError(f"expected a numbered clue, got {ln!r}")
        prefix, number, clue = match.group(1), int(match.group(2)), match.group(3).strip()
        slot_id = f"{(prefix or direction).upper()}{number}"
        if slot_id in seen:
            raise IngestError(f"duplicate clue {slot_id}")
        seen.add(slot_id)
        out.append((slot_id, clue))
    return out


def rows_to_grid(rows: list[str]) -> tuple[Grid, dict[Cell, str]]:
    """Build a Grid and letter prefill from ``#`` / ``.`` / ``A–Z`` rows."""
    cleaned = [row.strip().upper().replace(" ", "") for row in rows if row.strip()]
    if not cleaned:
        raise IngestError("no grid rows")
    width = len(cleaned[0])
    height = len(cleaned)
    if any(len(row) != width for row in cleaned):
        raise IngestError("grid rows have inconsistent widths")
    if not (MIN_SIZE <= height <= MAX_SIZE and MIN_SIZE <= width <= MAX_SIZE):
        raise IngestError(f"grid must be {MIN_SIZE}–{MAX_SIZE} on each side, got {height}x{width}")
    blocks: set[Cell] = set()
    prefill: dict[Cell, str] = {}
    for r, row in enumerate(cleaned):
        for c, ch in enumerate(row):
            if ch == BLOCK:
                blocks.add((r, c))
            elif ch == ".":
                continue
            elif "A" <= ch <= "Z":
                prefill[(r, c)] = ch
            else:
                raise IngestError(f"illegal grid character {ch!r} at {(r, c)}")
    try:
        grid = Grid(height, width, blocks)
    except PuzzleError as exc:
        raise IngestError(str(exc)) from exc
    return grid, prefill


def assemble(
    rows: list[str],
    across_text: str,
    down_text: str,
    *,
    puzzle_id: str,
    title: str = "",
) -> IngestDraft:
    """Match clue lists to a mask. ``ready`` or ``needs_edit``; never gold."""
    try:
        grid, prefill = rows_to_grid(rows)
    except IngestError as exc:
        return IngestDraft(status="needs_edit", rows=list(rows), message=str(exc))

    slots = grid.slots()
    isolated = grid.isolated_cells()
    across_slots = [s for s in slots if s.direction == ACROSS]
    down_slots = [s for s in slots if s.direction == DOWN]
    try:
        across_pairs = parse_clue_list(across_text, direction=ACROSS)
        down_pairs = parse_clue_list(down_text, direction=DOWN)
    except IngestError as exc:
        return IngestDraft(
            status="needs_edit",
            prefill=prefill,
            rows=_render(grid, prefill),
            across_slots=len(across_slots),
            down_slots=len(down_slots),
            message=str(exc),
        )

    across_pairs = _bind_sequential(across_pairs, across_slots, ACROSS)
    down_pairs = _bind_sequential(down_pairs, down_slots, DOWN)

    known_ids = {s.id for s in slots}
    unknown = [
        sid for sid, _ in across_pairs + down_pairs if sid not in known_ids
    ]
    across_clues = len(across_pairs)
    down_clues = len(down_pairs)
    reasons: list[str] = []
    if isolated:
        reasons.append(f"{len(isolated)} isolated cell(s) belong to no entry")
    if across_clues != len(across_slots):
        reasons.append(
            f"across: {across_clues} clues for {len(across_slots)} slots"
        )
    if down_clues != len(down_slots):
        reasons.append(f"down: {down_clues} clues for {len(down_slots)} slots")
    if unknown:
        reasons.append("unknown numbers: " + ", ".join(unknown))

    rendered = _render(grid, prefill)
    if reasons:
        return IngestDraft(
            status="needs_edit",
            prefill=prefill,
            rows=rendered,
            across_slots=len(across_slots),
            down_slots=len(down_slots),
            across_clues=across_clues,
            down_clues=down_clues,
            unknown_numbers=unknown,
            message="; ".join(reasons),
        )

    clues = dict(across_pairs + down_pairs)
    filled: list[Slot] = [replace(slot, clue=clues.get(slot.id, "")) for slot in slots]
    metadata = {"id": puzzle_id, "Source": "upload"}
    if title:
        metadata["Title"] = title
    puzzle = Puzzle(grid=grid, slots=filled, metadata=metadata)
    return IngestDraft(
        status="ready",
        puzzle=puzzle,
        prefill=prefill,
        rows=rendered,
        across_slots=len(across_slots),
        down_slots=len(down_slots),
        across_clues=across_clues,
        down_clues=down_clues,
    )


def _bind_sequential(
    pairs: list[tuple[str, str]], slots: list[Slot], direction: str
) -> list[tuple[str, str]]:
    """If clues were unnumbered placeholders ``A1,A2,…``, zip onto real slot ids."""
    expected = [f"{direction}{i + 1}" for i in range(len(pairs))]
    if [sid for sid, _ in pairs] != expected:
        return pairs
    if len(pairs) != len(slots):
        return pairs
    return [(slot.id, clue) for slot, (_, clue) in zip(slots, pairs)]


def _render(grid: Grid, prefill: dict[Cell, str]) -> list[str]:
    return grid.render(prefill, blank=".")


def decode_image(image: str) -> tuple[bytes, str]:
    """Decode a data URL or raw base64 image. Returns (bytes, mime)."""
    raw = image.strip()
    mime = ""
    if raw.startswith("data:"):
        header, _, payload = raw.partition(",")
        mime = header[5:].split(";", 1)[0].strip().lower()
        raw = payload
    try:
        data = base64.b64decode(raw, validate=False)
    except Exception as exc:
        raise IngestError("image is not valid base64") from exc
    if not data:
        raise IngestError("empty image")
    if len(data) > MAX_IMAGE_BYTES:
        raise IngestError(f"image is larger than {MAX_IMAGE_BYTES} bytes")
    if data.startswith(PNG_MAGIC):
        sniffed = "image/png"
    elif data.startswith(JPEG_MAGIC):
        sniffed = "image/jpeg"
    else:
        raise IngestError("image must be png or jpeg")
    if mime and mime not in ("image/png", "image/jpeg") and mime != sniffed:
        raise IngestError(f"unsupported image type {mime}")
    return data, sniffed


VISION_PROMPT = """\
You are reading a crossword grid photograph. Return JSON with a "rows" array.
Each row is a string of the same length. Use # for a black square, . for an
empty white square, and A-Z for a letter already filled in. Ignore clue
numbers, puzzle title, and anything outside the grid. Do not invent letters.
The grid is at most 15 by 15.
"""


def vision_messages(image_bytes: bytes, mime: str, *, schema_in_prompt: bool = False) -> list[dict]:
    """Multimodal chat messages for Token Factory vision."""
    b64 = base64.b64encode(image_bytes).decode("ascii")
    prompt = VISION_PROMPT
    if schema_in_prompt:
        prompt += '\nReturn a JSON object: {"rows": ["#..", "..."]}.'
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                },
            ],
        }
    ]


def vision_model() -> str:
    return os.environ.get("NEBIUS_VISION_MODEL") or DEFAULT_VISION_MODEL
