"""Pick the highest-scoring consistent set of answers from the candidate lists.

Weighted DFS over slots. Every slot must take a value; the wildcard is always
available, so a solution always exists and the search is really an optimization
over which slots to commit to.

Scoring is additive in log space: ``sum(log p)`` over chosen candidates, with
the wildcard priced at ``log(unknown_mass)``. That makes "commit to a 0.9
answer" beat "decline", and "decline" beat "commit to a 0.05 answer".

Variable/value ordering follows the CS50 CSP recipe, adapted for incomplete
LLM domains:

* **MRV** (dynamic): at each node, fill the unassigned slot with the fewest
  candidates that still fit the letters on the grid.
* **Degree** (tie-break): among those, prefer the slot that still crosses the
  most unfilled slots, so a conflict surfaces while the tree is shallow.
* **LCV** (value tie-break): among similar scores, try the answer that kills
  the fewest candidates on neighbouring slots. Score stays primary -- this is
  branch-and-bound, not "find any fill".
* **Nogoods**: a partial assignment that spelled a contradiction is recorded
  so later restarts do not replay it.

That trade sounds right but does not measure out -- see REPORT.md section 6.
Declined slots stay blank unless crossings have already spelled a real word.
Filling leftovers from letter marginals produced non-words (LFA, ETNT) and
is no longer done.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from ..lexicon import is_valid_entry
from ..schemas import Candidate
from .constraints import WILDCARD, Domains, SlotGraph, intersection_consistency

#: Extra score for an answer proposed independently by more than one model.
AGREEMENT_BONUS = 0.35


#: Cap on recorded contradictions so a dense 15×15 cannot grow without bound.
MAX_NOGOODS = 4_096


@dataclass
class SearchResult:
    assignment: dict[str, str]
    score: float
    confidence: dict[str, float] = field(default_factory=dict)
    nodes: int = 0
    restarts: int = 0
    complete: bool = False
    nogoods: int = 0

    @property
    def open_slots(self) -> list[str]:
        return sorted(s for s, a in self.assignment.items() if a == WILDCARD)

    @property
    def filled(self) -> dict[str, str]:
        return {s: a for s, a in self.assignment.items() if a != WILDCARD}


def _score(candidate: Candidate) -> float:
    base = math.log(max(1e-6, min(0.999, candidate.confidence)))
    if candidate.sources > 1:
        base += AGREEMENT_BONUS * (candidate.sources - 1)
    return base


def _domain_words(domains: Domains, slot_id: str) -> set[str]:
    domain = domains.get(slot_id)
    if domain is None:
        return set()
    return {c.answer for c in domain.candidates}


def _acceptable(word: str, slot_id: str, allowed: dict[str, set[str]]) -> bool:
    if word in allowed.get(slot_id, ()):
        return True
    return is_valid_entry(word)


def _spelled(graph: SlotGraph, cells: dict, slot_id: str) -> str | None:
    letters = [cells.get(cell) for cell in graph.cells[slot_id]]
    if any(ch is None for ch in letters):
        return None
    return "".join(letters)  # type: ignore[arg-type]


def fits_cells(slot_id: str, answer: str, cells: dict, graph: SlotGraph) -> bool:
    """True if ``answer`` agrees with every letter already on the grid."""
    for index, cell in enumerate(graph.cells[slot_id]):
        if index >= len(answer):
            return False
        existing = cells.get(cell)
        if existing is not None and existing != answer[index]:
            return False
    return True


def remaining_legal(
    slot_id: str,
    options: dict[str, list[tuple[str, float, float]]],
    cells: dict,
    graph: SlotGraph,
) -> int:
    """Non-wildcard candidates for ``slot_id`` that still fit ``cells``."""
    count = 0
    for answer, _, _ in options.get(slot_id, ()):
        if answer == WILDCARD:
            continue
        if fits_cells(slot_id, answer, cells, graph):
            count += 1
    return count


def next_slot(
    unassigned: list[str] | set[str],
    options: dict[str, list[tuple[str, float, float]]],
    cells: dict,
    graph: SlotGraph,
) -> str:
    """MRV, then degree (most unfilled crossings), then slot id."""
    open_set = set(unassigned)

    def key(slot_id: str) -> tuple[int, int, str]:
        legal = remaining_legal(slot_id, options, cells, graph)
        degree = sum(1 for neighbor in graph.neighbors(slot_id) if neighbor in open_set)
        return (legal, -degree, slot_id)

    return min(unassigned, key=key)


def eliminated_count(
    slot_id: str,
    answer: str,
    options: dict[str, list[tuple[str, float, float]]],
    assignment: dict[str, str],
    graph: SlotGraph,
) -> int:
    """How many neighbour candidates this value would delete. Lower is better (LCV)."""
    dropped = 0
    for crossing in graph.by_slot.get(slot_id, ()):
        other = crossing.b
        other_answer = assignment.get(other)
        if other_answer and other_answer != WILDCARD:
            continue
        if crossing.ai >= len(answer):
            continue
        letter = answer[crossing.ai]
        for other_word, _, _ in options.get(other, ()):
            if other_word == WILDCARD:
                continue
            if crossing.bi < len(other_word) and other_word[crossing.bi] != letter:
                dropped += 1
    return dropped


def value_order(
    slot_id: str,
    options: dict[str, list[tuple[str, float, float]]],
    assignment: dict[str, str],
    graph: SlotGraph,
    cells: dict | None = None,
) -> list[tuple[str, float, float]]:
    """High score first, then least-constraining, with the wildcard last."""
    known = cells if cells is not None else {}
    words: list[tuple[str, float, float]] = []
    wild: list[tuple[str, float, float]] = []
    for entry in options.get(slot_id, ()):
        answer = entry[0]
        if answer == WILDCARD:
            wild.append(entry)
            continue
        if fits_cells(slot_id, answer, known, graph):
            words.append(entry)
    words.sort(
        key=lambda entry: (
            -entry[1],
            eliminated_count(slot_id, entry[0], options, assignment, graph),
            entry[0],
        )
    )
    return words + wild


class NogoodCache:
    """Partial assignments that have already been shown to contradict."""

    def __init__(self, limit: int = MAX_NOGOODS):
        self._nogoods: set[frozenset[tuple[str, str]]] = set()
        self._limit = limit

    def __len__(self) -> int:
        return len(self._nogoods)

    def record(self, nogood: frozenset[tuple[str, str]]) -> None:
        if not nogood or len(self._nogoods) >= self._limit:
            return
        self._nogoods.add(nogood)

    def hits(
        self,
        assignment: dict[str, str],
        extra: tuple[str, str] | None = None,
    ) -> bool:
        assigned = {
            (slot_id, answer)
            for slot_id, answer in assignment.items()
            if answer and answer != WILDCARD
        }
        if extra is not None:
            assigned.add(extra)
        if not assigned:
            return False
        return any(nogood <= assigned for nogood in self._nogoods)


def _conflict_nogood(
    slot_id: str,
    answer: str,
    assignment: dict[str, str],
    cells: dict,
    graph: SlotGraph,
) -> frozenset[tuple[str, str]]:
    """Smallest assignment that made ``answer`` illegal in ``slot_id``."""
    items: list[tuple[str, str]] = [(slot_id, answer)]
    letter_hit = False
    for index, cell in enumerate(graph.cells[slot_id]):
        if index >= len(answer):
            break
        existing = cells.get(cell)
        if existing is None or existing == answer[index]:
            continue
        letter_hit = True
        for other, other_answer in assignment.items():
            if not other_answer or other_answer == WILDCARD:
                continue
            other_cells = graph.cells.get(other, ())
            if cell in other_cells:
                items.append((other, other_answer))
                break
    if not letter_hit:
        for other, other_answer in assignment.items():
            if other_answer and other_answer != WILDCARD:
                items.append((other, other_answer))
    return frozenset(items)


def solve(
    domains: Domains,
    graph: SlotGraph,
    *,
    seed: int = 0,
    max_nodes: int = 100_000,
    restarts: int = 5,
    beam: int = 8,
    require_words: bool = True,
    slot_ids: list[str] | None = None,
    preset_cells: dict | None = None,
) -> SearchResult:
    """Search for the best consistent assignment.

    Restarts with jittered tie-breaks matter more than they look: a single
    greedy descent commits early to a confident-but-wrong answer and then
    cannot escape, because every crossing it touches has been narrowed to
    agree with it.

    Slot order is recomputed at every node (MRV + degree). Values are tried
    high-score first, with least-constraining-value as the tie-break. Partial
    assignments that contradict are cached as nogoods across restarts.

    When ``require_words`` is set, a complete slot (every cell filled, even
    if the slot itself was declined) must be a real word, abbreviation, or
    a candidate the model actually proposed for that slot. That is what
    stops LINE + FEST from standing when they force the down LFA.

    ``slot_ids`` limits the search to a local star; ``preset_cells`` holds
    letters already committed outside that star (and locked cells).
    """
    search_ids = list(slot_ids) if slot_ids is not None else list(graph.slot_ids)
    if not search_ids:
        return SearchResult(assignment={}, score=0.0, complete=True)

    allowed = (
        {sid: _domain_words(domains, sid) for sid in graph.slot_ids}
        if require_words
        else {}
    )
    preset = dict(preset_cells or {})
    oov_ids = list(graph.slot_ids)
    best: SearchResult | None = None
    total_nodes = 0
    total_nogoods = 0

    for attempt in range(max(1, restarts)):
        rnd = random.Random(seed * 977 + attempt)
        options: dict[str, list[tuple[str, float, float]]] = {}
        for slot_id in search_ids:
            domain = domains.get(slot_id)
            entries: list[tuple[str, float, float]] = []
            if domain is not None:
                for candidate in domain.candidates[: beam * 2]:
                    jitter = rnd.uniform(0.97, 1.03) if attempt else 1.0
                    entries.append(
                        (candidate.answer, _score(candidate) * jitter, candidate.confidence)
                    )
            mass = domain.unknown_mass if domain else 0.15
            entries.sort(key=lambda e: -e[1])
            entries = entries[:beam]
            # The wildcard goes last: declining is a fallback, not a first choice.
            entries.append((WILDCARD, math.log(max(1e-6, mass)), 0.0))
            options[slot_id] = entries

        max_of = {sid: max(entry[1] for entry in options[sid]) for sid in search_ids}
        remaining_best0 = sum(max_of[sid] for sid in search_ids)

        assignment: dict[str, str] = {}
        cells: dict[tuple[int, int], str] = dict(preset)
        confidence: dict[str, float] = {}
        unassigned: set[str] = set(search_ids)
        nogoods = NogoodCache()
        state = {"nodes": 0, "best_score": -math.inf, "best": None}

        def place(slot_id: str, answer: str) -> list[tuple[int, int]] | None:
            """Write an answer's letters, or None if it contradicts the grid
            or completes a crossing as a non-word."""
            written: list[tuple[int, int]] = []
            for index, cell in enumerate(graph.cells[slot_id]):
                if index >= len(answer):
                    break
                existing = cells.get(cell)
                if existing is None:
                    cells[cell] = answer[index]
                    written.append(cell)
                elif existing != answer[index]:
                    for cell_to_clear in written:
                        del cells[cell_to_clear]
                    return None
            if require_words:
                for other in oov_ids:
                    spelling = _spelled(graph, cells, other)
                    if spelling is None:
                        continue
                    if not _acceptable(spelling, other, allowed):
                        for cell_to_clear in written:
                            del cells[cell_to_clear]
                        return None
            return written

        def recurse(score: float, remaining_best: float) -> None:
            if state["nodes"] >= max_nodes:
                return
            if not unassigned:
                chosen = dict(assignment)
                conf = dict(confidence)
                if require_words:
                    for sid in oov_ids:
                        spelling = _spelled(graph, cells, sid)
                        if spelling is None:
                            continue
                        if not _acceptable(spelling, sid, allowed):
                            return
                        if sid in search_ids and chosen.get(sid) == WILDCARD:
                            chosen[sid] = spelling
                            conf[sid] = max(conf.get(sid, 0.0), 0.5)
                if score > state["best_score"]:
                    state["best_score"] = score
                    state["best"] = (chosen, conf)
                return
            # Bound: even a perfect remainder cannot beat the incumbent.
            if score + remaining_best <= state["best_score"]:
                return
            if nogoods.hits(assignment):
                return

            slot_id = next_slot(unassigned, options, cells, graph)
            child_remaining = remaining_best - max_of[slot_id]
            unassigned.remove(slot_id)
            for answer, value, conf in value_order(
                slot_id, options, assignment, graph, cells
            ):
                state["nodes"] += 1
                if state["nodes"] >= max_nodes:
                    break
                if answer == WILDCARD:
                    assignment[slot_id] = WILDCARD
                    confidence[slot_id] = 0.0
                    recurse(score + value, child_remaining)
                    del assignment[slot_id]
                    del confidence[slot_id]
                    continue
                if nogoods.hits(assignment, (slot_id, answer)):
                    continue
                written = place(slot_id, answer)
                if written is None:
                    nogoods.record(
                        _conflict_nogood(slot_id, answer, assignment, cells, graph)
                    )
                    continue
                assignment[slot_id] = answer
                confidence[slot_id] = conf
                recurse(score + value, child_remaining)
                del assignment[slot_id]
                del confidence[slot_id]
                for cell in written:
                    del cells[cell]
            unassigned.add(slot_id)

        recurse(0.0, remaining_best0)
        total_nodes += state["nodes"]
        total_nogoods += len(nogoods)
        if state["best"] is None:
            continue
        chosen, confidences = state["best"]
        result = SearchResult(
            assignment=chosen,
            score=state["best_score"],
            confidence=confidences,
            nodes=total_nodes,
            restarts=attempt,
            complete=all(chosen.get(s) not in (None, WILDCARD) for s in search_ids),
            nogoods=total_nogoods,
        )
        if best is None or result.score > best.score:
            best = result
        if best.complete:
            agree, total = intersection_consistency(best.filled, graph)
            if total and agree == total:
                break

    if best is None:
        # Nothing consistent at all: decline everything rather than guess.
        return SearchResult(
            assignment={s: WILDCARD for s in search_ids},
            score=-math.inf,
            nodes=total_nodes,
            nogoods=total_nogoods,
        )
    best.nodes = total_nodes
    best.nogoods = total_nogoods
    return best


def endgame_fill(
    assignment: dict[str, str],
    graph: SlotGraph,
    marginals: dict[tuple[int, int], dict[str, float]] | None = None,
    *,
    require_words: bool = True,
) -> dict[tuple[int, int], str]:
    """Keep letters from assigned slots; promote a declined slot only when
    crossings have already spelled a real word.

    Letter-by-letter guesses used to fill leftovers from per-cell marginals
    and produced strings like LFA that are not words. A blank is honest.
    ``marginals`` is accepted for call-site compatibility and ignored.
    """
    del marginals
    from .constraints import cells_from_assignment

    cells = cells_from_assignment(assignment, graph)
    for slot_id, answer in assignment.items():
        if answer != WILDCARD:
            continue
        spelling = _spelled(graph, cells, slot_id)
        if spelling is None:
            continue
        if require_words and not is_valid_entry(spelling):
            continue
        assignment[slot_id] = spelling
        for index, cell in enumerate(graph.cells.get(slot_id, ())):
            cells[cell] = spelling[index]
    return cells
