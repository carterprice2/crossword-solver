"""Crossing constraints over LLM-proposed candidates.

The central difficulty: a crossword is a constraint-satisfaction problem, but
the values come from a language model, so the domains are *incomplete*. Textbook
arc consistency assumes every legal value is present and prunes anything without
support. Applied here it deletes correct answers, because the supporting word on
the crossing slot simply was never proposed.

Two design choices follow, and they are what make the layer work:

* **Soft AC-3.** A prune that would empty a domain is skipped and the crossing
  is recorded as a conflict site instead. Conflict sites are precisely the
  places worth spending another model call on.
* **A wildcard value.** Every slot can be left unfilled at a fixed probability
  cost, so the search can decline rather than commit to a wrong answer that
  would corrupt every crossing slot.

  Measured, this does not pay off. Arm ``a6`` disables it, and across the whole
  oracle sweep the two arms stay within 0.006 cell precision of each other; at
  candidate recall 0.50 disabling it is significantly *better* on WCR. The
  reason is that a declined slot does not stay blank -- ``endgame_fill`` fills
  it from letter marginals, a weaker inference than the search would have made,
  so the wildcard defers the decision rather than avoiding a bad guess. It is
  kept because arm ``a6`` measures it; see REPORT.md section 6 before relying
  on it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from ..model import Cell, Puzzle
from ..schemas import Candidate

#: The answer that means "leave this slot open".
WILDCARD = "\x00WILDCARD"
#: Probability mass reserved for the wildcard. Not tuned: the oracle sweep
#: found the mechanism worth roughly nothing either way, so there was no
#: optimum to find. See REPORT.md section 6.
DEFAULT_UNKNOWN_MASS = 0.15


@dataclass(frozen=True)
class Crossing:
    """Slot ``a`` position ``ai`` is the same cell as slot ``b`` position ``bi``."""

    a: str
    ai: int
    b: str
    bi: int
    cell: Cell


@dataclass
class ConflictSite:
    """A crossing where the candidate lists cannot be reconciled."""

    crossing: Crossing
    reason: str

    @property
    def slots(self) -> tuple[str, str]:
        return (self.crossing.a, self.crossing.b)


class SlotGraph:
    """Precomputed structure of a puzzle: lengths, crossings, neighbours."""

    def __init__(self, puzzle: Puzzle):
        self.puzzle = puzzle
        self.slot_ids = [s.id for s in puzzle.slots]
        self.length = {s.id: s.length for s in puzzle.slots}
        self.cells = {s.id: s.cells for s in puzzle.slots}
        self.crossings: list[Crossing] = []
        for inter in puzzle.intersections():
            self.crossings.append(
                Crossing(
                    inter.across, inter.across_index, inter.down, inter.down_index,
                    inter.cell,
                )
            )
        self.by_slot: dict[str, list[Crossing]] = {sid: [] for sid in self.slot_ids}
        for crossing in self.crossings:
            self.by_slot[crossing.a].append(crossing)
            self.by_slot[crossing.b].append(
                Crossing(crossing.b, crossing.bi, crossing.a, crossing.ai, crossing.cell)
            )

    def neighbors(self, slot_id: str) -> list[str]:
        return [c.b for c in self.by_slot.get(slot_id, [])]

    def pattern(self, slot_id: str, known: dict[Cell, str]) -> str:
        return "".join(
            (known.get(cell) or "?")[:1] for cell in self.cells[slot_id]
        )


@dataclass
class Domain:
    """The live candidate list for one slot."""

    slot_id: str
    candidates: list[Candidate] = field(default_factory=list)
    unknown_mass: float = DEFAULT_UNKNOWN_MASS

    def answers(self) -> list[str]:
        return [c.answer for c in self.candidates]

    def best(self) -> Candidate | None:
        return max(self.candidates, key=lambda c: c.confidence, default=None)

    def letters_at(self, index: int) -> set[str]:
        return {c.answer[index] for c in self.candidates if index < len(c.answer)}

    def is_empty(self) -> bool:
        return not self.candidates


Domains = dict[str, Domain]


def build_domains(
    graph: SlotGraph,
    candidates: list[Candidate],
    *,
    unknown_mass: float = DEFAULT_UNKNOWN_MASS,
) -> Domains:
    grouped: dict[str, list[Candidate]] = {sid: [] for sid in graph.slot_ids}
    for candidate in candidates:
        if candidate.slot_id in grouped:
            grouped[candidate.slot_id].append(candidate)
    return {
        sid: Domain(
            sid,
            sorted(items, key=lambda c: (-c.confidence, c.answer)),
            unknown_mass,
        )
        for sid, items in grouped.items()
    }


def merge_domains(base: Domains, incoming: Domains) -> Domains:
    """Union two candidate sets, keeping the higher confidence per answer."""
    out: Domains = {}
    for slot_id in set(base) | set(incoming):
        seen: dict[str, Candidate] = {}
        for source in (base.get(slot_id), incoming.get(slot_id)):
            if source is None:
                continue
            for candidate in source.candidates:
                existing = seen.get(candidate.answer)
                if existing is None or candidate.confidence > existing.confidence:
                    seen[candidate.answer] = candidate
        mass = (incoming.get(slot_id) or base.get(slot_id)).unknown_mass
        out[slot_id] = Domain(
            slot_id,
            sorted(seen.values(), key=lambda c: (-c.confidence, c.answer)),
            mass,
        )
    return out


def pattern_filter(
    domains: Domains, graph: SlotGraph, locked: dict[Cell, str]
) -> Domains:
    """Drop candidates that contradict a locked cell."""
    if not locked:
        return domains
    out: Domains = {}
    for slot_id, domain in domains.items():
        pattern = graph.pattern(slot_id, locked)
        kept = [c for c in domain.candidates if c.fits(graph.length[slot_id], pattern)]
        out[slot_id] = replace(domain, candidates=kept) if kept != domain.candidates else domain
    return out


def soft_ac3(
    domains: Domains, graph: SlotGraph, *, max_passes: int = 4
) -> tuple[Domains, list[ConflictSite]]:
    """Prune candidates unsupported by a crossing slot -- but never to empty.

    Returns the pruned domains and the crossings that could not be satisfied.
    """
    working = {sid: list(d.candidates) for sid, d in domains.items()}
    conflicts: list[ConflictSite] = []
    seen_conflicts: set[tuple[str, str, Cell]] = set()

    for _ in range(max_passes):
        changed = False
        for crossing in graph.crossings:
            for source, si, target, ti in (
                (crossing.a, crossing.ai, crossing.b, crossing.bi),
                (crossing.b, crossing.bi, crossing.a, crossing.ai),
            ):
                supports = {
                    c.answer[si]
                    for c in working.get(source, [])
                    if si < len(c.answer)
                }
                if not supports:
                    # An empty source constrains nothing; the wildcard covers it.
                    continue
                current = working.get(target, [])
                kept = [c for c in current if ti < len(c.answer) and c.answer[ti] in supports]
                if not kept:
                    if current:
                        key = (crossing.a, crossing.b, crossing.cell)
                        if key not in seen_conflicts:
                            seen_conflicts.add(key)
                            conflicts.append(
                                ConflictSite(
                                    crossing,
                                    f"{target} has no candidate agreeing with "
                                    f"{source} at {crossing.cell}",
                                )
                            )
                    # Soft: keep the domain rather than annihilate it.
                    continue
                if len(kept) < len(current):
                    working[target] = kept
                    changed = True
        if not changed:
            break

    pruned = {
        sid: replace(domains[sid], candidates=items) for sid, items in working.items()
    }
    return pruned, conflicts


def letter_marginals(
    domains: Domains, graph: SlotGraph
) -> dict[Cell, dict[str, float]]:
    """Confidence-weighted letter distribution per cell.

    Used to fill leftover cells once the search has done what it can, and as a
    tie-break signal. Every slot covering a cell contributes, so a cell whose
    across and down slots agree gets a sharply peaked distribution.
    """
    totals: dict[Cell, dict[str, float]] = {}
    for slot_id, domain in domains.items():
        cells = graph.cells.get(slot_id, ())
        if not domain.candidates:
            continue
        weight_sum = sum(c.confidence for c in domain.candidates) or 1.0
        for candidate in domain.candidates:
            share = candidate.confidence / weight_sum
            for index, cell in enumerate(cells):
                if index >= len(candidate.answer):
                    break
                bucket = totals.setdefault(cell, {})
                letter = candidate.answer[index]
                bucket[letter] = bucket.get(letter, 0.0) + share
    # Normalize so each cell sums to 1.
    for cell, bucket in totals.items():
        total = sum(bucket.values()) or 1.0
        totals[cell] = {k: v / total for k, v in bucket.items()}
    return totals


def intersection_consistency(
    assignment: dict[str, str], graph: SlotGraph
) -> tuple[int, int]:
    """(agreeing, total) crossings among assigned slots.

    Computable without the gold answers, which is why it doubles as the
    solver's own stopping signal and as a reported metric.
    """
    agree = total = 0
    for crossing in graph.crossings:
        a = assignment.get(crossing.a)
        b = assignment.get(crossing.b)
        if not a or not b:
            continue
        if crossing.ai >= len(a) or crossing.bi >= len(b):
            continue
        total += 1
        if a[crossing.ai] == b[crossing.bi]:
            agree += 1
    return agree, total


def cells_from_assignment(
    assignment: dict[str, str], graph: SlotGraph
) -> dict[Cell, str]:
    """Lay assigned answers onto the grid. Later slots do not overwrite earlier
    ones, so a disagreement leaves the first writer's letter in place."""
    out: dict[Cell, str] = {}
    for slot_id, answer in assignment.items():
        if answer == WILDCARD:
            continue
        for index, cell in enumerate(graph.cells.get(slot_id, ())):
            if index < len(answer) and cell not in out:
                out[cell] = answer[index]
    return out
