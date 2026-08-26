"""Check a fill the way a human would: length, blanks, and crossings.

Candidates come from the model. This module never searches a word list by
clue — that would invert the constructor's answer key. The optional dictionary
path is the other direction: given a word the model already proposed, look up
*that word's* definition and see whether it agrees with the clue.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping

from ..lexicon import is_valid_entry
from ..model import Puzzle
from ..normalize import normalize_answer, normalize_clue
from ..schemas import Candidate
from .constraints import (
    WILDCARD,
    SlotGraph,
    cells_from_assignment,
    intersection_consistency,
)

_STOP = frozenset(
    "a an the of or and to in for with from by as on at is be that which "
    "this these those its it's not".split()
)


@dataclass
class SlotIssue:
    slot_id: str
    kind: str
    detail: str


@dataclass
class VerifyReport:
    issues: list[SlotIssue] = field(default_factory=list)
    complete: bool = False

    @property
    def ok(self) -> bool:
        return not self.issues and self.complete

    def for_slot(self, slot_id: str) -> list[SlotIssue]:
        return [i for i in self.issues if i.slot_id == slot_id]

    def slot_ids(self) -> list[str]:
        return sorted({i.slot_id for i in self.issues})


def definition_score(clue: str, dictionary_clue: str) -> float:
    """1.0 on a normalised exact match, otherwise Jaccard over content words."""
    a = normalize_clue(clue)
    b = normalize_clue(dictionary_clue)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    ta = {t for t in a.split() if t not in _STOP}
    tb = {t for t in b.split() if t not in _STOP}
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def rescore_with_dictionary(
    candidates: list[Candidate],
    puzzle: Puzzle,
    definitions: Mapping[str, str],
) -> list[Candidate]:
    """Boost a proposed word when *its* dictionary sense matches the clue.

    Words the dictionary does not know are left alone. Nothing is added or
    dropped — the model still has to propose the word.
    """
    out: list[Candidate] = []
    for candidate in candidates:
        sense = definitions.get(candidate.answer)
        if not sense:
            out.append(candidate)
            continue
        try:
            clue = puzzle.slot(candidate.slot_id).clue
        except Exception:
            out.append(candidate)
            continue
        score = definition_score(clue, sense)
        if score <= 0.0:
            out.append(candidate)
            continue
        confidence = min(0.97, candidate.confidence + 0.15 * score)
        if score >= 1.0:
            confidence = max(confidence, min(0.97, candidate.confidence + 0.2))
        out.append(replace(candidate, confidence=confidence))
    return out


def implied_spellings(
    assignment: dict[str, str], graph: SlotGraph
) -> dict[str, str]:
    """Slot → letters, including downs fully determined by assigned acrosses."""
    cells = cells_from_assignment(assignment, graph)
    out: dict[str, str] = {}
    for slot_id, slot_cells in graph.cells.items():
        answer = assignment.get(slot_id)
        if answer and answer != WILDCARD:
            out[slot_id] = answer
            continue
        letters = [cells.get(cell) for cell in slot_cells]
        if all(letters):
            out[slot_id] = "".join(letters)  # type: ignore[arg-type]
    return out


def verify_fill(
    puzzle: Puzzle,
    graph: SlotGraph,
    assignment: dict[str, str],
    *,
    definitions: Mapping[str, str] | None = None,
    proposed: Mapping[str, set[str]] | None = None,
    require_words: bool = True,
) -> VerifyReport:
    """Hard structural checks. Empty issues + complete ⇒ the grid is consistent.

    A slot that is declined but whose letters are fully determined by crossings
    is still checked: LFA is not a word just because LINE and FEST made it.
    """
    issues: list[SlotIssue] = []
    filled = 0
    proposed = proposed or {}
    spelled = implied_spellings(assignment, graph)
    for slot in puzzle.slots:
        answer = spelled.get(slot.id) or assignment.get(slot.id)
        if not answer or answer == WILDCARD:
            issues.append(SlotIssue(slot.id, "blank", "slot was left open"))
            continue
        filled += 1
        word = normalize_answer(answer)
        if len(word) != slot.length:
            issues.append(
                SlotIssue(
                    slot.id,
                    "length",
                    f"{word!r} has {len(word)} letters, slot is {slot.length}",
                )
            )
        if require_words and not is_valid_entry(
            word, proposed=word in proposed.get(slot.id, ())
        ):
            issues.append(
                SlotIssue(
                    slot.id,
                    "word",
                    f"{word} is not a word, abbreviation, or known crossword entry",
                )
            )
        if definitions is not None and word in definitions:
            score = definition_score(slot.clue, definitions[word])
            if score < 0.25:
                issues.append(
                    SlotIssue(
                        slot.id,
                        "definition",
                        f"{word}'s dictionary sense does not match the clue "
                        f"(overlap {score:.2f})",
                    )
                )

    filled_map = {s: a for s, a in assignment.items() if a and a != WILDCARD}
    agree, total = intersection_consistency(filled_map, graph)
    if total and agree != total:
        for crossing in graph.crossings:
            a = filled_map.get(crossing.a)
            b = filled_map.get(crossing.b)
            if not a or not b:
                continue
            if crossing.ai >= len(a) or crossing.bi >= len(b):
                continue
            if a[crossing.ai] != b[crossing.bi]:
                issues.append(
                    SlotIssue(
                        crossing.a,
                        "crossing",
                        f"{crossing.a}={a} conflicts with {crossing.b}={b} "
                        f"at {crossing.cell}",
                    )
                )
                issues.append(
                    SlotIssue(
                        crossing.b,
                        "crossing",
                        f"{crossing.b}={b} conflicts with {crossing.a}={a} "
                        f"at {crossing.cell}",
                    )
                )

    complete = filled == len(puzzle.slots) and (total == 0 or agree == total)
    if issues:
        complete = False
    return VerifyReport(issues=issues, complete=complete)
