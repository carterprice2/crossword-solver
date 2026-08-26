"""Fill a block mask with real words, then attach clues.

Standard CSP: slots are variables, bank words are values, crossing cells are
the constraints. MRV picks the most-constrained slot next and forward checking
prunes crossers after every assignment, which together keep the search shallow
enough that a 9x9 fills in about a second against a 3,800-word bank.
"""

from __future__ import annotations

import heapq
import random
import time
from dataclasses import dataclass, field, replace

from ..model import Grid, Puzzle, Slot
from .bank import Bank


class FillError(RuntimeError):
    """Raised when no fill was found within the budget."""


@dataclass
class FillStats:
    nodes: int = 0
    backtracks: int = 0
    seconds: float = 0.0
    restarts: int = 0


@dataclass
class _Cross:
    """slot A's cell index i is slot B's cell index j."""

    other: int
    i: int
    j: int


@dataclass
class _Problem:
    slots: list[Slot]
    crosses: dict[int, list[_Cross]] = field(default_factory=dict)


def _build_problem(grid: Grid, min_entry: int) -> _Problem:
    slots = grid.slots(min_entry=min_entry)
    index_of = {slot.id: n for n, slot in enumerate(slots)}
    crosses: dict[int, list[_Cross]] = {n: [] for n in range(len(slots))}
    cell_owner: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for n, slot in enumerate(slots):
        for i, cell in enumerate(slot.cells):
            cell_owner.setdefault(cell, []).append((n, i))
    for owners in cell_owner.values():
        for a, (na, ia) in enumerate(owners):
            for nb, ib in owners[a + 1 :]:
                crosses[na].append(_Cross(nb, ia, ib))
                crosses[nb].append(_Cross(na, ib, ia))
    del index_of
    return _Problem(slots=slots, crosses=crosses)


def fill_grid(
    grid: Grid,
    bank: Bank,
    *,
    seed: int = 0,
    min_entry: int = 3,
    time_limit: float = 20.0,
    max_nodes: int = 400_000,
    restarts: int = 4,
    beam: int = 40,
    propagate_max: int = 150,
) -> tuple[dict[str, str], FillStats]:
    """Assign a distinct bank word to every slot. Returns {slot_id: word}.

    ``beam`` caps how many words are tried per slot. Without a cap the search
    can spend its whole budget re-trying near-identical low-frequency words in
    one subtree instead of backtracking to a better decision higher up.
    """
    problem = _build_problem(grid, min_entry)
    slots = problem.slots
    if not slots:
        raise FillError("grid has no entries")

    stats = FillStats()
    started = time.monotonic()

    for attempt in range(restarts + 1):
        stats.restarts = attempt
        rnd = random.Random(seed * 1000 + attempt)
        domains: list[frozenset[str]] = [bank.words(len(s.cells)) for s in slots]
        if any(not d for d in domains):
            missing = {len(s.cells) for s, d in zip(slots, domains) if not d}
            raise FillError(f"bank has no words of length(s) {sorted(missing)}")

        lengths = [len(s.cells) for s in slots]
        assignment: dict[int, str] = {}
        used: set[str] = set()
        # Undo trail of (slot_index, previous_domain). Copying every domain at
        # every node made an 11x11 explore only ~400 nodes in 15s; recording
        # just what changed is what makes the larger grids tractable.
        trail: list[tuple[int, frozenset[str]]] = []

        def assign(n: int, domain: frozenset[str]) -> None:
            trail.append((n, domains[n]))
            domains[n] = domain

        def undo(mark: int) -> None:
            while len(trail) > mark:
                n, previous = trail.pop()
                domains[n] = previous

        def forward_check(n: int) -> bool:
            """Propagate slot n's domain to everything it crosses."""
            queue = [n]
            while queue:
                cur = queue.pop()
                # Deriving the letter set means walking the whole domain, so
                # doing it for a domain of thousands costs far more than it
                # saves: a large domain admits essentially every letter, and
                # the prune is a no-op. Skipping those sources is safe because
                # the slot we just assigned is always a singleton, and a
                # singleton always propagates -- that is what actually
                # enforces crossing consistency.
                if len(domains[cur]) > propagate_max:
                    continue
                letters_cache: dict[int, frozenset[str]] = {}
                for cross in problem.crosses[cur]:
                    letters = letters_cache.get(cross.i)
                    if letters is None:
                        letters = frozenset(w[cross.i] for w in domains[cur])
                        letters_cache[cross.i] = letters
                    current = domains[cross.other]
                    allowed = bank.with_any_letter(
                        lengths[cross.other], cross.j, letters
                    )
                    pruned = current & allowed
                    if not pruned:
                        return False
                    if len(pruned) < len(current):
                        assign(cross.other, pruned)
                        queue.append(cross.other)
            return True

        def recurse() -> bool:
            if time.monotonic() - started > time_limit or stats.nodes > max_nodes:
                raise TimeoutError
            unassigned = [n for n in range(len(slots)) if n not in assignment]
            if not unassigned:
                return True
            # MRV: fewest live words first.
            n = min(unassigned, key=lambda i: len(domains[i]))
            # Value ordering by commonness: the bank has to be large to fill a
            # dense grid at all, and an unordered large bank fills it with
            # Webster obscurities. Walking the precomputed frequency order and
            # stopping early keeps this off the hot path -- sorting each whole
            # domain here cost ~45ms per node and stalled the 11x11 grids.
            rank = bank.rank
            pool = heapq.nsmallest(
                beam * 3,
                (w for w in domains[n] if w not in used),
                key=rank.__getitem__,
            )
            # Jitter within the shortlist so different seeds give different
            # puzzles without abandoning the frequency preference.
            pool.sort(key=lambda w: -(bank.freq_for(w) * rnd.uniform(0.5, 1.5)))
            for word in pool[:beam]:
                stats.nodes += 1
                mark = len(trail)
                assign(n, frozenset({word}))
                assignment[n] = word
                used.add(word)
                if forward_check(n) and recurse():
                    return True
                del assignment[n]
                used.discard(word)
                undo(mark)
                stats.backtracks += 1
            return False

        try:
            if recurse():
                stats.seconds = time.monotonic() - started
                return {slots[n].id: w for n, w in assignment.items()}, stats
        except TimeoutError:
            if time.monotonic() - started > time_limit or stats.nodes > max_nodes:
                break

    stats.seconds = time.monotonic() - started
    raise FillError(
        f"no fill found in {stats.seconds:.1f}s "
        f"({stats.nodes} nodes, {stats.restarts + 1} attempt(s))"
    )


def build_puzzle(
    grid: Grid,
    bank: Bank,
    *,
    seed: int = 0,
    min_entry: int = 3,
    title: str = "",
    **fill_kwargs,
) -> tuple[Puzzle, FillStats]:
    """Fill a mask and dress the result up as a clued Puzzle."""
    words, stats = fill_grid(grid, bank, seed=seed, min_entry=min_entry, **fill_kwargs)
    slots: list[Slot] = []
    for slot in grid.slots(min_entry=min_entry):
        word = words[slot.id]
        slots.append(
            replace(slot, clue=bank.clue_for(word), gold=word)
        )
    metadata = {
        "Title": title or f"Generated {grid.height}x{grid.width} #{seed}",
        "Author": "crossword-agent generator",
        "Source": "generated",
        "Seed": str(seed),
    }
    puzzle = Puzzle(grid=grid, slots=slots, metadata=metadata)
    problems = puzzle.validate()
    if problems:
        raise FillError(f"generated puzzle is invalid: {problems}")
    return puzzle, stats
