"""Core puzzle data model.

The whole project agrees on three representations:

  Grid        the block mask plus the standard crossword numbering
  Puzzle      a Grid with clues attached to each slot (and gold answers, if known)
  Solution    dict[Cell, str] -- the canonical unit every metric compares

Keeping metrics on Solution rather than on slot strings is deliberate: rebus
squares hold multi-character values, and comparing at the cell level means that
never leaks into the metric code.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable, Iterator, Mapping

Cell = tuple[int, int]
Solution = dict[Cell, str]

ACROSS = "A"
DOWN = "D"

BLOCK = "#"
#: Placeholder used in patterns for a cell whose letter is not yet known.
UNKNOWN = "?"

#: Shortest run of open cells that counts as an entry. American crosswords use
#: 3; the numbering rule itself only requires 2, so parsing published puzzles
#: needs the lower bound while our generator enforces the higher one.
MIN_ENTRY = 2


class PuzzleError(ValueError):
    """Raised when a grid or puzzle is structurally invalid."""


@dataclass(frozen=True)
class Slot:
    """One numbered entry: its cells, its clue, and its answer if known."""

    id: str
    direction: str
    number: int
    cells: tuple[Cell, ...]
    clue: str = ""
    gold: str | None = None

    @property
    def length(self) -> int:
        return len(self.cells)

    def index_of(self, cell: Cell) -> int:
        return self.cells.index(cell)

    def pattern(self, known: Mapping[Cell, str]) -> str:
        """Render the slot as e.g. ``?E??R`` given the cells known so far.

        Multi-character (rebus) values collapse to their first character; the
        pattern is a per-cell display, not a spelling.
        """
        return "".join(known.get(c, UNKNOWN)[:1] or UNKNOWN for c in self.cells)

    def spell(self, known: Mapping[Cell, str]) -> str | None:
        """The full answer string, or None if any cell is unknown."""
        parts = [known.get(c) for c in self.cells]
        if any(p is None for p in parts):
            return None
        return "".join(parts)  # type: ignore[arg-type]

    def lay(self, answer: str) -> Solution:
        """Spread a single-character-per-cell answer across this slot's cells."""
        if len(answer) != self.length:
            raise PuzzleError(
                f"{self.id}: answer {answer!r} has length {len(answer)}, "
                f"slot has {self.length} cells"
            )
        return {cell: answer[i] for i, cell in enumerate(self.cells)}


@dataclass(frozen=True)
class Intersection:
    """A cell shared by one across slot and one down slot."""

    cell: Cell
    across: str
    across_index: int
    down: str
    down_index: int


class Grid:
    """The block mask, plus everything derivable from it."""

    def __init__(self, height: int, width: int, blocks: Iterable[Cell]):
        if height <= 0 or width <= 0:
            raise PuzzleError(f"grid must be non-empty, got {height}x{width}")
        self.height = height
        self.width = width
        self.blocks = frozenset(blocks)
        for r, c in self.blocks:
            if not (0 <= r < height and 0 <= c < width):
                raise PuzzleError(f"block {(r, c)} outside {height}x{width} grid")

    # -- basic queries ----------------------------------------------------

    def is_block(self, cell: Cell) -> bool:
        return cell in self.blocks

    def in_bounds(self, cell: Cell) -> bool:
        r, c = cell
        return 0 <= r < self.height and 0 <= c < self.width

    def open_cells(self) -> list[Cell]:
        return [
            (r, c)
            for r in range(self.height)
            for c in range(self.width)
            if (r, c) not in self.blocks
        ]

    @property
    def block_density(self) -> float:
        return len(self.blocks) / (self.height * self.width)

    def is_symmetric(self) -> bool:
        """True if the block mask has 180-degree rotational symmetry."""
        return all(
            (self.height - 1 - r, self.width - 1 - c) in self.blocks
            for r, c in self.blocks
        )

    # -- numbering and slots ----------------------------------------------

    def _runs(self, direction: str, min_entry: int) -> Iterator[tuple[Cell, tuple[Cell, ...]]]:
        """Yield (start_cell, cells) for every run of open cells."""
        outer, inner = (
            (self.height, self.width) if direction == ACROSS else (self.width, self.height)
        )
        for a in range(outer):
            b = 0
            while b < inner:
                cell = (a, b) if direction == ACROSS else (b, a)
                if cell in self.blocks:
                    b += 1
                    continue
                run: list[Cell] = []
                while b < inner:
                    cur = (a, b) if direction == ACROSS else (b, a)
                    if cur in self.blocks:
                        break
                    run.append(cur)
                    b += 1
                if len(run) >= min_entry:
                    yield run[0], tuple(run)

    def numbering(self, min_entry: int = MIN_ENTRY) -> dict[Cell, int]:
        """Standard crossword numbering: scan in reading order, number any cell
        that starts an across or down entry."""
        starts = {c for c, _ in self._runs(ACROSS, min_entry)}
        starts |= {c for c, _ in self._runs(DOWN, min_entry)}
        numbers: dict[Cell, int] = {}
        n = 0
        for r in range(self.height):
            for c in range(self.width):
                if (r, c) in starts:
                    n += 1
                    numbers[(r, c)] = n
        return numbers

    def slots(self, min_entry: int = MIN_ENTRY) -> list[Slot]:
        """All entries, across first then down, each in numbering order."""
        numbers = self.numbering(min_entry)
        out: list[Slot] = []
        for direction in (ACROSS, DOWN):
            runs = sorted(
                self._runs(direction, min_entry), key=lambda rc: numbers[rc[0]]
            )
            for start, cells in runs:
                num = numbers[start]
                out.append(
                    Slot(
                        id=f"{direction}{num}",
                        direction=direction,
                        number=num,
                        cells=cells,
                    )
                )
        return out

    def isolated_cells(self, min_entry: int = MIN_ENTRY) -> list[Cell]:
        """Open cells belonging to no entry -- always a grid construction bug."""
        covered: set[Cell] = set()
        for direction in (ACROSS, DOWN):
            for _, cells in self._runs(direction, min_entry):
                covered.update(cells)
        return [c for c in self.open_cells() if c not in covered]

    # -- rendering ---------------------------------------------------------

    def render(self, known: Mapping[Cell, str] | None = None, blank: str = ".") -> list[str]:
        known = known or {}
        rows = []
        for r in range(self.height):
            row = []
            for c in range(self.width):
                if (r, c) in self.blocks:
                    row.append(BLOCK)
                else:
                    row.append(known.get((r, c), blank)[:1] or blank)
            rows.append("".join(row))
        return rows

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Grid({self.height}x{self.width}, {len(self.blocks)} blocks)"


@dataclass
class Puzzle:
    """A grid plus clues, and the gold solution when we have one."""

    grid: Grid
    slots: list[Slot]
    metadata: dict[str, str] = field(default_factory=dict)
    #: Cells whose gold value is more than one character (rebus squares).
    rebus: dict[Cell, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._by_id = {s.id: s for s in self.slots}
        if len(self._by_id) != len(self.slots):
            dupes = [s.id for s in self.slots if list(self._by_id).count(s.id) > 1]
            raise PuzzleError(f"duplicate slot ids: {sorted(set(dupes))}")

    # -- lookups -----------------------------------------------------------

    @property
    def id(self) -> str:
        return self.metadata.get("id") or self.metadata.get("Title") or "puzzle"

    def slot(self, slot_id: str) -> Slot:
        try:
            return self._by_id[slot_id]
        except KeyError:
            raise PuzzleError(f"no slot {slot_id!r}") from None

    def has_gold(self) -> bool:
        return all(s.gold for s in self.slots)

    def gold_solution(self) -> Solution:
        """The known-correct letter for every open cell."""
        out: Solution = {}
        for slot in self.slots:
            if slot.gold is None:
                raise PuzzleError(f"{slot.id} has no gold answer")
            out.update(self._lay_gold(slot))
        return out

    def _lay_gold(self, slot: Slot) -> Solution:
        """Lay a gold answer across cells, honoring rebus squares."""
        assert slot.gold is not None
        if not self.rebus:
            return slot.lay(slot.gold)
        # With rebus present, walk the answer consuming each cell's true value.
        out: Solution = {}
        pos = 0
        for cell in slot.cells:
            value = self.rebus.get(cell, slot.gold[pos : pos + 1])
            if slot.gold[pos : pos + len(value)] != value:
                raise PuzzleError(
                    f"{slot.id}: gold {slot.gold!r} disagrees with rebus at {cell}"
                )
            out[cell] = value
            pos += len(value)
        if pos != len(slot.gold):
            raise PuzzleError(f"{slot.id}: gold {slot.gold!r} does not fill its cells")
        return out

    # -- structure ---------------------------------------------------------

    def intersections(self) -> list[Intersection]:
        owner: dict[Cell, dict[str, Slot]] = {}
        for slot in self.slots:
            for cell in slot.cells:
                owner.setdefault(cell, {})[slot.direction] = slot
        out = []
        for cell, dirs in sorted(owner.items()):
            a, d = dirs.get(ACROSS), dirs.get(DOWN)
            if a and d:
                out.append(
                    Intersection(cell, a.id, a.index_of(cell), d.id, d.index_of(cell))
                )
        return out

    def crossers(self, slot_id: str) -> list[tuple[int, str, int]]:
        """For a slot, the (my_index, other_slot_id, their_index) it crosses."""
        slot = self.slot(slot_id)
        cells = set(slot.cells)
        out = []
        for other in self.slots:
            if other.id == slot_id or other.direction == slot.direction:
                continue
            for cell in other.cells:
                if cell in cells:
                    out.append((slot.index_of(cell), other.id, other.index_of(cell)))
        return sorted(out)

    def neighbors(self, slot_id: str) -> list[str]:
        return [other for _, other, _ in self.crossers(slot_id)]

    def with_clues(self, clues: Mapping[str, str]) -> "Puzzle":
        slots = [replace(s, clue=clues.get(s.id, s.clue)) for s in self.slots]
        return Puzzle(self.grid, slots, dict(self.metadata), dict(self.rebus))

    def validate(self) -> list[str]:
        """Structural problems, as human-readable strings. Empty means clean."""
        problems: list[str] = []
        isolated = self.grid.isolated_cells()
        if isolated:
            problems.append(f"{len(isolated)} isolated open cell(s): {isolated[:5]}")
        for slot in self.slots:
            if slot.gold is None:
                continue
            expected = sum(len(self.rebus.get(c, "x")) for c in slot.cells)
            if len(slot.gold) != expected:
                problems.append(
                    f"{slot.id}: gold {slot.gold!r} is {len(slot.gold)} chars, "
                    f"cells need {expected}"
                )
        if self.has_gold():
            try:
                self.gold_solution()
            except PuzzleError as exc:
                problems.append(str(exc))
        return problems


def solution_from_rows(grid: Grid, rows: Iterable[str]) -> Solution:
    """Read a rendered grid back into a Solution, ignoring blocks and blanks."""
    out: Solution = {}
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if (r, c) in grid.blocks or ch in (BLOCK, UNKNOWN, ".", " "):
                continue
            out[(r, c)] = ch
    return out
