"""Block-mask templates: parsing, validation, and search.

A template is a plain-text mask -- ``#`` for a block, ``.`` for an open cell.
Templates are validated rather than trusted, because a bad mask fails in a way
that is expensive to debug: the filler simply grinds until its node budget runs
out. The rules encoded here come from a measured result (see REPORT.md): fill
difficulty tracks the *longest slot*, not the grid size, so capping run length
is what keeps generation fast.
"""

from __future__ import annotations

import random
from collections import deque

from ..model import ACROSS, DOWN, Cell, Grid

#: American convention: no entry shorter than three cells.
MIN_RUN = 3
#: Above this, the filler's search space explodes. An open 7x7 (max run 7)
#: never completed in 40s; every grid we ship stays at or under this.
DEFAULT_MAX_RUN = 6


def parse_template(text: str) -> Grid:
    rows = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not rows:
        raise ValueError("empty template")
    width = len(rows[0])
    if any(len(r) != width for r in rows):
        raise ValueError("template rows have inconsistent widths")
    blocks = {
        (r, c) for r, row in enumerate(rows) for c, ch in enumerate(row) if ch == "#"
    }
    return Grid(len(rows), width, blocks)


def dump_template(grid: Grid) -> str:
    return "\n".join(grid.render(blank=".")) + "\n"


def run_lengths(grid: Grid) -> list[int]:
    return [len(s.cells) for s in grid.slots(min_entry=1)]


def is_connected(grid: Grid) -> bool:
    """Every open cell reachable from every other, moving orthogonally."""
    open_cells = grid.open_cells()
    if not open_cells:
        return False
    start = open_cells[0]
    seen = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            cell = (nr, nc)
            if cell in seen or not grid.in_bounds(cell) or grid.is_block(cell):
                continue
            seen.add(cell)
            queue.append(cell)
    return len(seen) == len(open_cells)


def validate_template(
    grid: Grid, *, max_run: int = DEFAULT_MAX_RUN, min_run: int = MIN_RUN
) -> list[str]:
    """Structural problems with a mask. Empty list means it is usable."""
    problems: list[str] = []
    if not grid.is_symmetric():
        problems.append("block mask is not 180-degree symmetric")
    runs = run_lengths(grid)
    if not runs:
        problems.append("no entries at all")
        return problems
    short = [n for n in runs if n < min_run]
    if short:
        problems.append(f"{len(short)} run(s) shorter than {min_run}: {sorted(short)}")
    longest = max(runs)
    if longest > max_run:
        problems.append(f"longest run is {longest}, limit is {max_run}")
    if not is_connected(grid):
        problems.append("open cells are not all connected")
    # Every open cell must belong to both an across and a down entry, or the
    # crossing constraints the solver relies on do not exist there.
    covered = {ACROSS: set(), DOWN: set()}
    for slot in grid.slots(min_entry=min_run):
        covered[slot.direction].update(slot.cells)
    uncrossed = [c for c in grid.open_cells() if c not in covered[ACROSS] or c not in covered[DOWN]]
    if uncrossed:
        problems.append(f"{len(uncrossed)} cell(s) not fully crossed: {uncrossed[:4]}")
    return problems


def _long_runs(grid: Grid, max_run: int) -> list[tuple[Cell, ...]]:
    return [s.cells for s in grid.slots(min_entry=1) if len(s.cells) > max_run]


def _legal_splits(length: int, min_run: int) -> list[int]:
    """Positions within a run where a block leaves both halves legal.

    Blocking at index ``p`` yields pieces of length ``p`` and ``length-1-p``;
    each must be either empty or at least ``min_run`` long.
    """
    ok = []
    for p in range(length):
        left, right = p, length - 1 - p
        if (left == 0 or left >= min_run) and (right == 0 or right >= min_run):
            ok.append(p)
    return ok


def search_templates(
    size: int,
    *,
    max_run: int = DEFAULT_MAX_RUN,
    target_density: float = 0.18,
    seed: int = 0,
    attempts: int = 2000,
    want: int = 1,
    min_run: int = MIN_RUN,
) -> list[Grid]:
    """Build valid masks by repairing, not by guessing.

    Random symmetric block placement almost never lands on a legal mask -- the
    dominant failure is a row or column left uncut, so most samples have a run
    the full width of the grid. Instead, start from an empty grid and
    repeatedly split whichever run is still too long, always at a position that
    leaves both halves legal, mirroring each block to preserve symmetry.
    """
    rnd = random.Random(seed)
    found: list[Grid] = []
    seen: set[frozenset[Cell]] = set()
    max_blocks = round(target_density * size * size * 1.6)

    for _ in range(attempts):
        blocks: set[Cell] = set()
        grid = Grid(size, size, blocks)
        for _ in range(size * size):
            long_runs = _long_runs(grid, max_run)
            if not long_runs:
                break
            if len(blocks) > max_blocks:
                break
            # Attack the worst offender first.
            rnd.shuffle(long_runs)
            run = max(long_runs, key=len)
            splits = _legal_splits(len(run), min_run)
            rnd.shuffle(splits)
            # A split is legal for its own run, but its mirror lands somewhere
            # else and can strand a too-short run there. Try splits until one
            # leaves the whole grid clean, rather than discovering the problem
            # only at the final validation.
            progressed = False
            for position in splits:
                cell = run[position]
                mirror = (size - 1 - cell[0], size - 1 - cell[1])
                candidate = blocks | {cell, mirror}
                if candidate == blocks:
                    continue
                trial = Grid(size, size, candidate)
                if any(len(s.cells) < min_run for s in trial.slots(min_entry=1)):
                    continue
                blocks, grid, progressed = candidate, trial, True
                break
            if not progressed:
                break
        else:  # pragma: no cover - loop bound is generous
            continue

        key = frozenset(blocks)
        if key in seen:
            continue
        seen.add(key)
        if validate_template(grid, max_run=max_run, min_run=min_run):
            continue
        found.append(grid)
        if len(found) >= want:
            break
    return found
