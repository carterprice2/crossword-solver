"""Local repair: treat a clash as a small CSP (a star), not a whole-grid re-ask.

A conflict is two entries that share a cell, plus everything those two touch.
That set is small enough to enumerate existing candidates before spending
another model call. If the lists cannot mesh, we ask the model about *this*
star. If that also fails, keep the higher-confidence hub and leave the other
blank -- do not invent a non-word from leftover letters.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..schemas import Candidate
from .constraints import WILDCARD, ConflictSite, Domains, SlotGraph, cells_from_assignment
from .search import SearchResult, solve as search_solve

#: Cap so a dense 15×15 clash does not become "the whole puzzle."
MAX_STAR_SLOTS = 12
MAX_STARS_PER_ROUND = 4


@dataclass(frozen=True)
class Star:
    """One local subproblem: ``hubs`` clashed; ``slots`` is the star around them."""

    slots: tuple[str, ...]
    hubs: tuple[str, str]


def star_slots(graph: SlotGraph, hub_a: str, hub_b: str) -> tuple[str, ...]:
    slots = {hub_a, hub_b}
    slots.update(graph.neighbors(hub_a))
    slots.update(graph.neighbors(hub_b))
    return tuple(sorted(slots))


def collect_stars(
    graph: SlotGraph,
    conflicts: list[ConflictSite],
    targets: list[str],
) -> list[Star]:
    """Build stars for clashes first, then for leftover unhappy slots."""
    stars: list[Star] = []
    seen: set[tuple[str, ...]] = set()

    def add(hub_a: str, hub_b: str) -> None:
        slots = star_slots(graph, hub_a, hub_b)
        if len(slots) > MAX_STAR_SLOTS:
            keep = {hub_a, hub_b}
            keep.update(s for s in graph.neighbors(hub_a) if s in targets)
            keep.update(s for s in graph.neighbors(hub_b) if s in targets)
            slots = tuple(sorted(keep))
        if slots in seen:
            return
        seen.add(slots)
        stars.append(Star(slots=slots, hubs=(hub_a, hub_b)))

    for site in conflicts:
        add(site.crossing.a, site.crossing.b)
    covered = {s for star in stars for s in star.slots}
    for slot_id in targets:
        if slot_id in covered:
            continue
        neighbors = graph.neighbors(slot_id)
        other = neighbors[0] if neighbors else slot_id
        add(slot_id, other)
        covered.update(stars[-1].slots if stars else ())

    stars.sort(key=lambda star: (len(star.slots), star.hubs, star.slots))
    return stars[:MAX_STARS_PER_ROUND]


def preset_cells(
    graph: SlotGraph,
    assignment: dict[str, str],
    locked: dict,
    star: Star,
) -> dict:
    """Letters from outside the star, plus locks, as hard constraints."""
    outside = {
        sid: answer
        for sid, answer in assignment.items()
        if sid not in star.slots and answer and answer != WILDCARD
    }
    cells = cells_from_assignment(outside, graph)
    cells.update(locked)
    return cells


def solve_star(
    domains: Domains,
    graph: SlotGraph,
    star: Star,
    assignment: dict[str, str],
    locked: dict,
    *,
    seed: int = 0,
    require_words: bool = True,
) -> SearchResult:
    return search_solve(
        domains,
        graph,
        seed=seed,
        slot_ids=list(star.slots),
        preset_cells=preset_cells(graph, assignment, locked, star),
        require_words=require_words,
        max_nodes=20_000,
        restarts=3,
        beam=6,
    )


def star_complete(result: SearchResult, star: Star) -> bool:
    return all(
        result.assignment.get(sid) not in (None, WILDCARD) for sid in star.slots
    )


def apply_star(
    result: SearchResult,
    star: Star,
    assignment: dict[str, str],
    confidence: dict[str, float],
    domains: Domains,
) -> None:
    """Write a locally consistent fill back and boost those candidates."""
    for slot_id in star.slots:
        answer = result.assignment.get(slot_id)
        if not answer or answer == WILDCARD:
            continue
        assignment[slot_id] = answer
        confidence[slot_id] = max(
            confidence.get(slot_id, 0.0), result.confidence.get(slot_id, 0.0)
        )
        _boost(domains, slot_id, answer, confidence[slot_id])


def fallback_star(
    star: Star,
    assignment: dict[str, str],
    confidence: dict[str, float],
    domains: Domains,
    rejected: dict[str, set[str]],
) -> str:
    """Keep the stronger hub; blank the other if they still disagree.

    Returns a short reason for the trace.
    """
    hub_a, hub_b = star.hubs
    pick_a = _hub_score(hub_a, assignment, confidence, domains)
    pick_b = _hub_score(hub_b, assignment, confidence, domains)
    if pick_b > pick_a:
        keep, drop = hub_b, hub_a
        keep_word, drop_word = pick_b[1], pick_a[1]
    else:
        keep, drop = hub_a, hub_b
        keep_word, drop_word = pick_a[1], pick_b[1]
    if keep_word:
        assignment[keep] = keep_word
        confidence[keep] = max(confidence.get(keep, 0.0), pick_a[0] if keep == hub_a else pick_b[0])
        _boost(domains, keep, keep_word, confidence[keep])
    if drop != keep and drop_word and drop_word != keep_word:
        rejected.setdefault(drop, set()).add(drop_word)
        assignment[drop] = WILDCARD
        confidence[drop] = 0.0
        domain = domains.get(drop)
        if domain is not None:
            domain.candidates = [
                c for c in domain.candidates if c.answer != drop_word
            ]
    return f"keep {keep}={keep_word or '?'}, blank {drop}"


def _hub_score(
    slot_id: str,
    assignment: dict[str, str],
    confidence: dict[str, float],
    domains: Domains,
) -> tuple[float, str]:
    answer = assignment.get(slot_id)
    if answer and answer != WILDCARD:
        return (confidence.get(slot_id, 0.0), answer)
    domain = domains.get(slot_id)
    best = domain.best() if domain is not None else None
    if best:
        return (best.confidence, best.answer)
    return (0.0, "")


def _boost(domains: Domains, slot_id: str, answer: str, confidence: float) -> None:
    domain = domains.get(slot_id)
    if domain is None:
        return
    found = False
    updated: list[Candidate] = []
    floor = max(0.85, confidence)
    for candidate in domain.candidates:
        if candidate.answer == answer:
            updated.append(
                replace(candidate, confidence=max(candidate.confidence, floor))
            )
            found = True
        else:
            updated.append(candidate)
    if not found:
        updated.append(Candidate(slot_id, answer, floor))
    domain.candidates = sorted(updated, key=lambda c: (-c.confidence, c.answer))
