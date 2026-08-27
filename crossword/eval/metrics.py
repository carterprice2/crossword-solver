"""Scoring a solved puzzle.

Metric names follow CrossWordBench (arXiv 2504.00043) for WCR/LCR/ICR and the
Berkeley Crossword Solver (arXiv 2205.09665) for exact-puzzle accuracy, so
numbers here are comparable to the published literature.

One metric deserves a note. **ICR is computed without the gold answers** -- it
asks whether the across and down *slot assignments* agree wherever they cross.
That is only meaningful if it is scored on the slot strings the solver committed
to, not on a cell-keyed grid: a cell cannot disagree with itself, so ICR on a
filled Solution is always 1. Pass ``assignment`` to :func:`score_solution`.

Two more exist because a solver that declines is not the same as one that
guesses wrong, and averaging them together hides the difference:

    cell_precision  of the cells it filled, how many were right
    cell_recall     of the cells it could have filled, how many it got right
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping

from ..model import Cell, Puzzle, Solution

#: Must match crossword.agent.constraints.WILDCARD; kept here so scoring
#: does not import the agent package (eval/metrics is in the hermetic set).
_WILDCARD = "\x00WILDCARD"


@dataclass
class Scores:
    """Every metric for one solved puzzle."""

    wcr: float = 0.0
    lcr: float = 0.0
    icr: float = 1.0
    exact: bool = False
    cell_precision: float = 0.0
    cell_recall: float = 0.0
    cells_filled: int = 0
    cells_total: int = 0
    cells_correct: int = 0
    slots_correct: int = 0
    slots_total: int = 0
    slots_attempted: int = 0
    per_slot: dict[str, bool] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "wcr": round(self.wcr, 6),
            "lcr": round(self.lcr, 6),
            "icr": round(self.icr, 6),
            "exact": self.exact,
            "cell_precision": round(self.cell_precision, 6),
            "cell_recall": round(self.cell_recall, 6),
            "cells_filled": self.cells_filled,
            "cells_correct": self.cells_correct,
            "cells_total": self.cells_total,
            "slots_correct": self.slots_correct,
            "slots_attempted": self.slots_attempted,
            "slots_total": self.slots_total,
        }


def score_solution(
    puzzle: Puzzle,
    predicted: Solution,
    *,
    assignment: Mapping[str, str] | None = None,
) -> Scores:
    """Compare a predicted grid against the puzzle's gold answers.

    ``assignment`` is the per-slot fill the search produced. ICR is computed
    from those strings (across vs down) so a solver that ignores crossings
    can score below 1. Without it, ICR falls back to the cell grid, which is
    tautological once every crossing cell has a letter.
    """
    gold = puzzle.gold_solution()
    open_cells = [c for c in puzzle.grid.open_cells()]
    total = len(open_cells)

    filled = 0
    correct = 0
    for cell in open_cells:
        got = predicted.get(cell)
        if got is None or got == "":
            continue
        filled += 1
        if got.upper() == gold[cell].upper():
            correct += 1

    per_slot: dict[str, bool] = {}
    attempted = 0
    for slot in puzzle.slots:
        values = [predicted.get(cell) for cell in slot.cells]
        if any(v is None or v == "" for v in values):
            per_slot[slot.id] = False
            continue
        attempted += 1
        answer = "".join(v.upper() for v in values)  # type: ignore[union-attr]
        per_slot[slot.id] = answer == (slot.gold or "").upper()

    slots_total = len(puzzle.slots)
    slots_correct = sum(1 for ok in per_slot.values() if ok)

    return Scores(
        wcr=slots_correct / slots_total if slots_total else 0.0,
        lcr=correct / total if total else 0.0,
        icr=(
            assignment_consistency(puzzle, assignment)
            if assignment is not None
            else grid_consistency(puzzle, predicted)
        ),
        exact=correct == total and total > 0,
        cell_precision=correct / filled if filled else 0.0,
        cell_recall=correct / total if total else 0.0,
        cells_filled=filled,
        cells_total=total,
        cells_correct=correct,
        slots_correct=slots_correct,
        slots_total=slots_total,
        slots_attempted=attempted,
        per_slot=per_slot,
    )


def assignment_consistency(
    puzzle: Puzzle, assignment: Mapping[str, str]
) -> float:
    """Fraction of crossings where the two slot strings agree. Needs no gold.

    Open / declined slots (missing or wildcard) do not count. This is the ICR
    that can actually go below 1 for an inconsistent solver.
    """
    agree = total = 0
    for inter in puzzle.intersections():
        a = assignment.get(inter.across)
        b = assignment.get(inter.down)
        if not a or not b or a == _WILDCARD or b == _WILDCARD:
            continue
        if inter.across_index >= len(a) or inter.down_index >= len(b):
            continue
        total += 1
        if a[inter.across_index] == b[inter.down_index]:
            agree += 1
    return agree / total if total else 1.0


def grid_consistency(puzzle: Puzzle, predicted: Solution) -> float:
    """Fraction of *filled* crossings whose cell value is present.

    A cell-keyed Solution cannot disagree with itself. Prefer
    :func:`assignment_consistency` for the reported ICR.
    """
    agree = total = 0
    for inter in puzzle.intersections():
        value = predicted.get(inter.cell)
        if value is None or value == "":
            continue
        across = puzzle.slot(inter.across)
        down = puzzle.slot(inter.down)
        a_letters = [predicted.get(c) for c in across.cells]
        d_letters = [predicted.get(c) for c in down.cells]
        if any(v is None for v in a_letters) or any(v is None for v in d_letters):
            continue
        total += 1
        if a_letters[inter.across_index] == d_letters[inter.down_index]:
            agree += 1
    return agree / total if total else 1.0


# -- calibration ----------------------------------------------------------


@dataclass
class Calibration:
    """How well self-reported confidence tracks being right.

    This matters more here than in most benchmarks: the constraint layer
    consumes those confidences as probabilities, so a confidently-wrong model
    actively damages the search, while an uncertainly-right one does not.
    """

    ece: float = 0.0
    brier: float = 0.0
    bins: list[dict] = field(default_factory=list)
    n: int = 0

    def as_dict(self) -> dict:
        return {
            "ece": round(self.ece, 6),
            "brier": round(self.brier, 6),
            "n": self.n,
            "bins": self.bins,
        }


def calibration(pairs: list[tuple[float, bool]], *, n_bins: int = 10) -> Calibration:
    """Expected calibration error and Brier score over (confidence, correct)."""
    if not pairs:
        return Calibration()
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for confidence, correct in pairs:
        clamped = min(0.999999, max(0.0, confidence))
        buckets[int(clamped * n_bins)].append((clamped, correct))

    total = len(pairs)
    ece = 0.0
    bins = []
    for index, bucket in enumerate(buckets):
        if not bucket:
            continue
        mean_conf = sum(c for c, _ in bucket) / len(bucket)
        accuracy = sum(1 for _, ok in bucket if ok) / len(bucket)
        ece += (len(bucket) / total) * abs(mean_conf - accuracy)
        bins.append(
            {
                "lower": round(index / n_bins, 3),
                "upper": round((index + 1) / n_bins, 3),
                "n": len(bucket),
                "mean_confidence": round(mean_conf, 4),
                "accuracy": round(accuracy, 4),
            }
        )
    brier = sum((c - (1.0 if ok else 0.0)) ** 2 for c, ok in pairs) / total
    return Calibration(ece=ece, brier=brier, bins=bins, n=total)


# -- difficulty control ----------------------------------------------------


def prefill_cells(puzzle: Puzzle, ratio: float, *, seed: int = 0) -> dict[Cell, str]:
    """Reveal a fraction of the grid, CrossWordBench-style.

    Cells are spread out rather than sampled independently: a random draw
    clusters, and a clustered reveal hands over whole answers instead of
    uniformly reducing difficulty.
    """
    if ratio <= 0:
        return {}
    import random

    gold = puzzle.gold_solution()
    cells = sorted(puzzle.grid.open_cells())
    want = int(round(ratio * len(cells)))
    if want <= 0:
        return {}
    if want >= len(cells):
        return dict(gold)

    rnd = random.Random(seed)
    chosen: list[Cell] = []
    remaining = cells[:]
    rnd.shuffle(remaining)
    for cell in remaining:
        if len(chosen) >= want:
            break
        # Prefer cells away from those already revealed.
        if chosen and any(
            abs(cell[0] - c[0]) + abs(cell[1] - c[1]) <= 1 for c in chosen[-4:]
        ):
            continue
        chosen.append(cell)
    for cell in remaining:  # top up if spreading was too strict
        if len(chosen) >= want:
            break
        if cell not in chosen:
            chosen.append(cell)
    return {cell: gold[cell] for cell in chosen[:want]}


def aggregate(scores: list[Scores]) -> dict:
    """Mean of each metric across puzzles, plus exact-solve count."""
    if not scores:
        return {}
    n = len(scores)
    return {
        "n": n,
        "wcr": sum(s.wcr for s in scores) / n,
        "lcr": sum(s.lcr for s in scores) / n,
        "icr": sum(s.icr for s in scores) / n,
        "exact": sum(1 for s in scores if s.exact) / n,
        "exact_count": sum(1 for s in scores if s.exact),
        "cell_precision": sum(s.cell_precision for s in scores) / n,
        "cell_recall": sum(s.cell_recall for s in scores) / n,
    }


def mean_and_sd(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(variance)
