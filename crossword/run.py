"""Shared solve runner used by the CLI and the web API.

The agent loop stays in ``crossword.agent``. This module is the wiring:
which corpus file, which client, which ablation arm, then ``Solver.solve``.
"""

from __future__ import annotations

import glob
import os
from typing import Any

from .agent.solver import Solver, SolverConfig, SolveResult
from .agent.trace import Tracer
from .client import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_REASONING_MODEL,
    NebiusClient,
    OracleClient,
    OracleConfig,
    RecordingClient,
    ReplayClient,
)
from .eval.harness import Arm, build_arms, solve_one_shot
from .eval.metrics import Scores, prefill_cells, score_solution
from .model import ACROSS, DOWN, Puzzle, Solution
from .xd import load_xd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(HERE, "corpus")

ENSEMBLE_MODEL = "meta-llama/Llama-3.3-70B-Instruct"


class RunError(ValueError):
    """User-facing failure: unknown suite, missing gold, bad arm, etc."""


def suite_paths(suite: str) -> list[str]:
    if os.path.isdir(suite):
        return sorted(glob.glob(os.path.join(suite, "*.xd")))
    directory = os.path.join(CORPUS, suite)
    if os.path.isdir(directory):
        return sorted(glob.glob(os.path.join(directory, "*.xd")))
    available = sorted(os.listdir(CORPUS)) if os.path.isdir(CORPUS) else []
    raise RunError(
        f"no suite {suite!r}. Expected a directory, or one of: "
        f"{', '.join(available) if available else '(none)'}"
    )


def load_puzzles(paths: list[str], limit: int = 0) -> list[Puzzle]:
    puzzles = [load_xd(p) for p in paths]
    return puzzles[:limit] if limit else puzzles


def list_suites() -> list[dict[str, Any]]:
    """Suites the web UI can pick from. Mini is committed; NYT is materialized."""
    suites = [
        {
            "id": "mini",
            "label": "Mini",
            "count": len(suite_paths("mini")),
            "description": "Generated 7×7, 9×9, and 11×11 puzzles. Never published.",
        }
    ]
    nyt_paths = _ensure_nyt()
    if nyt_paths:
        suites.append(
            {
                "id": "nyt",
                "label": "NYT",
                "count": len(nyt_paths),
                "description": "Friday May 28 2021, 15×15. Live solves spend tokens.",
                "warning": "A live 15×15 Nebius solve is slow and costs real tokens. Oracle is free.",
            }
        )
    return suites


def _ensure_nyt() -> list[str]:
    """Return nyt .xd paths, writing the local fixture if it is missing."""
    directory = os.path.join(CORPUS, "nyt")
    existing = sorted(glob.glob(os.path.join(directory, "*.xd")))
    if existing:
        return existing
    from .eval.nyt import write_corpus

    write_corpus(directory)
    return sorted(glob.glob(os.path.join(directory, "*.xd")))


def list_puzzles(suite: str) -> list[dict[str, Any]]:
    if suite == "nyt":
        paths = _ensure_nyt()
        if not paths:
            raise RunError("no NYT fixture")
    else:
        paths = suite_paths(suite)
    out = []
    for path in paths:
        puzzle = load_xd(path)
        out.append(puzzle_summary(puzzle, suite=suite))
    return out


def puzzle_summary(puzzle: Puzzle, *, suite: str = "") -> dict[str, Any]:
    return {
        "id": puzzle.id,
        "suite": suite,
        "title": puzzle.metadata.get("Title", puzzle.id),
        "author": puzzle.metadata.get("Author", ""),
        "height": puzzle.grid.height,
        "width": puzzle.grid.width,
        "size": f"{puzzle.grid.height}x{puzzle.grid.width}",
        "slots": len(puzzle.slots),
        "provenance": puzzle.metadata.get("Source", "unknown"),
        "has_gold": puzzle.has_gold(),
    }


def find_puzzle(puzzle_id: str) -> Puzzle:
    for suite in ("mini", "nyt"):
        try:
            paths = _ensure_nyt() if suite == "nyt" else suite_paths(suite)
        except RunError:
            continue
        for path in paths:
            puzzle = load_xd(path)
            if puzzle.id == puzzle_id:
                return puzzle
    raise RunError(f"no puzzle {puzzle_id!r}")


def serialize_puzzle(puzzle: Puzzle) -> dict[str, Any]:
    """Public puzzle view: grid, numbers, clues. No gold letters."""
    numbers = puzzle.grid.numbering()
    blocks = [[r, c] for r, c in sorted(puzzle.grid.blocks)]
    number_list = [[r, c, n] for (r, c), n in sorted(numbers.items())]
    clues = {"across": [], "down": []}
    for slot in puzzle.slots:
        entry = {
            "id": slot.id,
            "number": slot.number,
            "clue": slot.clue,
            "length": slot.length,
            "cells": [[r, c] for r, c in slot.cells],
        }
        if slot.direction == ACROSS:
            clues["across"].append(entry)
        elif slot.direction == DOWN:
            clues["down"].append(entry)
    return {
        "id": puzzle.id,
        "title": puzzle.metadata.get("Title", puzzle.id),
        "author": puzzle.metadata.get("Author", ""),
        "provenance": puzzle.metadata.get("Source", "unknown"),
        "height": puzzle.grid.height,
        "width": puzzle.grid.width,
        "size": f"{puzzle.grid.height}x{puzzle.grid.width}",
        "slots": len(puzzle.slots),
        "has_gold": puzzle.has_gold(),
        "grid": {
            "height": puzzle.grid.height,
            "width": puzzle.grid.width,
            "blocks": blocks,
            "numbers": number_list,
        },
        "clues": clues,
    }


def cell_correctness(puzzle: Puzzle, predicted: Solution) -> list[dict[str, Any]]:
    """Per-cell gold-check flags. Wrong cells do not include the gold letter."""
    gold = puzzle.gold_solution() if puzzle.has_gold() else {}
    out = []
    for cell in puzzle.grid.open_cells():
        letter = predicted.get(cell)
        if not letter:
            continue
        row = {"r": cell[0], "c": cell[1], "letter": letter[:1]}
        if gold:
            row["correct"] = letter.upper() == gold[cell].upper()
        out.append(row)
    return out


def gold_cells(puzzle: Puzzle) -> list[dict[str, Any]]:
    """Answer-key letters for every open cell. Empty if the puzzle has no gold."""
    if not puzzle.has_gold():
        return []
    gold = puzzle.gold_solution()
    return [
        {"r": cell[0], "c": cell[1], "letter": gold[cell][:1]}
        for cell in puzzle.grid.open_cells()
    ]


def annotate_candidate_event(event: dict[str, Any], puzzle: Puzzle) -> dict[str, Any]:
    """Add gold answer and hit/miss to a ``candidates`` trace event."""
    gold_by_slot = {slot.id: (slot.gold or "").upper() for slot in puzzle.slots}
    data = dict(event.get("data") or {})
    slots = []
    for slot in data.get("slots") or []:
        slot_id = slot.get("id", "")
        gold = gold_by_slot.get(slot_id, "")
        answers = {
            str(item.get("answer", "")).upper()
            for item in slot.get("candidates") or []
        }
        row = dict(slot)
        row["gold"] = gold
        row["hit"] = bool(gold) and gold in answers
        slots.append(row)
    out = dict(event)
    out["data"] = {**data, "slots": slots}
    return out


def make_client(
    puzzle: Puzzle | None,
    *,
    backend: str = "nebius",
    seed: int = 7,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 120.0,
    replay: str | None = None,
    replay_loose: bool = False,
    oracle_recall: float = 0.8,
    oracle_top1_error: float = 0.35,
    oracle_independent: bool = False,
    record: str | None = None,
):
    """Pick a client. Only ``nebius`` touches the network."""
    if backend == "replay":
        if not replay:
            raise RunError("--backend replay needs a replay path")
        return ReplayClient(replay, strict=not replay_loose)
    if backend == "oracle":
        if puzzle is None or not puzzle.has_gold():
            raise RunError("oracle backend needs a puzzle with gold answers")
        gold = {s.id: s.gold or "" for s in puzzle.slots}
        return OracleClient(
            gold,
            OracleConfig(
                recall=oracle_recall,
                top1_error=oracle_top1_error,
                conf_noise=0.2,
                seed=seed,
                pattern_aware=not oracle_independent,
            ),
        )
    client = NebiusClient(base_url=base_url, timeout=timeout)
    if record:
        return RecordingClient(client, record)
    return client


def solver_config(
    *,
    arm: str = "a3",
    model: str = DEFAULT_MODEL,
    repair_model: str = DEFAULT_REASONING_MODEL,
    ensemble_model: str = ENSEMBLE_MODEL,
    seed: int = 7,
    workers: int = 8,
    rounds: int = 0,
) -> tuple[Arm, SolverConfig]:
    arms = build_arms(model, repair_model, ensemble_model)
    chosen = arms.get(arm)
    if chosen is None:
        raise RunError(f"unknown arm {arm!r}; choose from {', '.join(arms)}")
    config = chosen.config
    config.model = model
    config.repair_model = repair_model
    config.seed = seed
    config.max_workers = workers
    if rounds:
        config.max_rounds = rounds
    return chosen, config


def run_solve(
    puzzle: Puzzle,
    *,
    client,
    config: SolverConfig,
    one_shot: bool = False,
    tracer: Tracer | None = None,
    prefill: dict | None = None,
) -> tuple[SolveResult, Scores | None]:
    if one_shot:
        result = solve_one_shot(client, puzzle, config, prefill=prefill)
    else:
        result = Solver(client, config, tracer=tracer or Tracer()).solve(
            puzzle, prefill=prefill
        )
    scores = (
        score_solution(puzzle, result.solution, assignment=result.assignment)
        if puzzle.has_gold()
        else None
    )
    return result, scores


def default_backend() -> str:
    """Oracle when there is no key, so the UI works on a fresh clone."""
    return "nebius" if os.environ.get("NEBIUS_API_KEY") else "oracle"


def prefill_for(puzzle: Puzzle, ratio: float, seed: int) -> dict:
    return prefill_cells(puzzle, ratio, seed=seed) if ratio else {}
