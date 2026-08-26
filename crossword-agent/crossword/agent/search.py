"""Pick the highest-scoring consistent set of answers from the candidate lists.

Weighted DFS over slots. Every slot must take a value; the wildcard is always
available, so a solution always exists and the search is really an optimization
over which slots to commit to.

Scoring is additive in log space: ``sum(log p)`` over chosen candidates, with
the wildcard priced at ``log(unknown_mass)``. That makes "commit to a 0.9
answer" beat "decline", and "decline" beat "commit to a 0.05 answer".

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


@dataclass
class SearchResult:
    assignment: dict[str, str]
    score: float
    confidence: dict[str, float] = field(default_factory=dict)
    nodes: int = 0
    restarts: int = 0
    complete: bool = False

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

        # Most-constrained first, so conflicts surface near the root where
        # backtracking is cheap.
        order = sorted(
            search_ids,
            key=lambda s: (
                len(options[s]),
                -max((e[2] for e in options[s]), default=0.0),
                -len(graph.by_slot.get(s, [])),
            ),
        )

        # suffix_best[d] = the most score the slots from depth d onwards could
        # possibly contribute, used to prune hopeless subtrees.
        suffix_best = [0.0] * (len(order) + 1)
        for depth in range(len(order) - 1, -1, -1):
            suffix_best[depth] = suffix_best[depth + 1] + max(
                entry[1] for entry in options[order[depth]]
            )

        assignment: dict[str, str] = {}
        cells: dict[tuple[int, int], str] = dict(preset)
        confidence: dict[str, float] = {}
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

        def recurse(depth: int, score: float) -> None:
            if state["nodes"] >= max_nodes:
                return
            if depth == len(order):
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
            # suffix_best is precomputed; recomputing this sum at every node
            # cost O(slots x options) per node, which dominated the search on
            # larger grids.
            if score + suffix_best[depth] <= state["best_score"]:
                return

            slot_id = order[depth]
            for answer, value, conf in options[slot_id]:
                state["nodes"] += 1
                if state["nodes"] >= max_nodes:
                    return
                if answer == WILDCARD:
                    assignment[slot_id] = WILDCARD
                    confidence[slot_id] = 0.0
                    recurse(depth + 1, score + value)
                    del assignment[slot_id]
                    del confidence[slot_id]
                    continue
                written = place(slot_id, answer)
                if written is None:
                    continue
                assignment[slot_id] = answer
                confidence[slot_id] = conf
                recurse(depth + 1, score + value)
                del assignment[slot_id]
                del confidence[slot_id]
                for cell in written:
                    del cells[cell]

        recurse(0, 0.0)
        total_nodes += state["nodes"]
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
        )
    best.nodes = total_nodes
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
