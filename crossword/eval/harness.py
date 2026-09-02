"""Ablation arms and the evaluation runner.

The arms are designed so that each adjacent pair isolates one mechanism:

    A1 - A0   does asking clue-by-clue beat one whole-grid prompt?
    A2 - A1   what does constraint propagation add?
    A3 - A2   what does the agentic re-query loop add?   <- the headline claim
    A4 - A3   what does a second model family add?
    A5 - A3   what does using the big model everywhere cost, and buy?
    A3 - A6   what does letting the solver decline a slot add?

Every arm runs the identical puzzles at the identical seeds, so all comparisons
are paired and `stats.paired_bootstrap` applies.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, replace

from ..agent.constraints import SlotGraph
from ..agent.prompts import whole_puzzle_messages
from ..agent.solver import SolveResult, Solver, SolverConfig
from ..agent.trace import Tracer
from ..progress import enabled as progress_enabled
from ..progress import log as progress_log
from ..client import DEFAULT_MODEL, DEFAULT_REASONING_MODEL, ModelClient, Usage
from ..model import Puzzle
from ..normalize import classify_clue, length_bucket
from ..schemas import parse_candidates
from .metrics import Scores, calibration, prefill_cells, score_solution


@dataclass
class Arm:
    name: str
    label: str
    description: str
    config: SolverConfig
    #: A0 does not use the agent loop at all.
    one_shot: bool = False


def build_arms(
    model: str = DEFAULT_MODEL,
    reasoning_model: str = DEFAULT_REASONING_MODEL,
    ensemble_model: str = "meta-llama/Llama-3.3-70B-Instruct",
) -> dict[str, Arm]:
    base = SolverConfig(model=model, repair_model=reasoning_model)
    return {
        "a0": Arm(
            "a0",
            "one-shot whole puzzle",
            "Single prompt containing the grid and every clue. The naive baseline.",
            replace(base, max_rounds=1),
            one_shot=True,
        ),
        "a1": Arm(
            "a1",
            "per-clue, no constraints",
            "Ask clue by clue, take each top answer, ignore crossings.",
            replace(base, max_rounds=1, use_constraints=False, use_repair=False),
        ),
        "a2": Arm(
            "a2",
            "+ constraint propagation",
            "Add soft AC-3 and the weighted search over candidates. No re-query.",
            replace(base, max_rounds=1, use_constraints=True, use_repair=False),
        ),
        "a3": Arm(
            "a3",
            "+ repair rounds (full agent)",
            "Re-query unresolved and conflicting slots with the letters the "
            "grid has since pinned down.",
            replace(base, max_rounds=5, use_constraints=True, use_repair=True),
        ),
        "a4": Arm(
            "a4",
            "+ two-model ensemble",
            "A3, plus a second model family in round 0; agreement raises "
            "confidence by noisy-OR.",
            replace(base, max_rounds=5, ensemble_model=ensemble_model),
        ),
        "a5": Arm(
            "a5",
            "reasoning model throughout",
            "A3 with the large reasoning model at every stage. The cost ceiling.",
            replace(base, max_rounds=5, model=reasoning_model,
                    repair_model=reasoning_model),
        ),
        "a6": Arm(
            "a6",
            "no wildcard (must guess)",
            "A3 with unknown_mass=0, so the search can never decline a slot.",
            replace(base, max_rounds=5, unknown_mass=1e-9),
        ),
    }


DEFAULT_ARMS = ("a0", "a1", "a2", "a3")


def solve_one_shot(
    client: ModelClient,
    puzzle: Puzzle,
    config: SolverConfig,
    *,
    prefill: dict | None = None,
) -> SolveResult:
    """Arm A0: one prompt, whole puzzle, take what comes back."""
    started = time.monotonic()
    graph = SlotGraph(puzzle)
    usage = Usage()
    expected = {s.id: s.length for s in puzzle.slots}
    warnings: list[str] = []
    assignment: dict[str, str] = {}
    confidence: dict[str, float] = {}
    locked = dict(prefill or {})
    try:
        completion = client.complete(
            model=config.model,
            messages=whole_puzzle_messages(puzzle, known=locked),
            temperature=config.temperature,
            max_tokens=max(2048, 40 * len(puzzle.slots)),
            seed=config.seed,
        )
        usage.record(completion)
        candidates, warnings = parse_candidates(completion.text, expected=expected)
        for candidate in candidates:
            current = confidence.get(candidate.slot_id, -1.0)
            if candidate.confidence > current:
                assignment[candidate.slot_id] = candidate.answer
                confidence[candidate.slot_id] = candidate.confidence
    except Exception as exc:
        warnings.append(str(exc))

    cells: dict = dict(locked)
    for slot_id, answer in assignment.items():
        for index, cell in enumerate(graph.cells.get(slot_id, ())):
            if index < len(answer):
                cells.setdefault(cell, answer[index])

    pairs = []
    if puzzle.has_gold():
        for slot_id, answer in assignment.items():
            gold = (puzzle.slot(slot_id).gold or "").upper()
            pairs.append((confidence.get(slot_id, 0.0), answer == gold))

    return SolveResult(
        puzzle=puzzle,
        solution=cells,
        assignment=assignment,
        confidence=confidence,
        locked=locked,
        rounds=1,
        usage=usage,
        seconds=time.monotonic() - started,
        warnings=warnings,
        calibration_pairs=pairs,
    )


@dataclass
class RunRecord:
    """One (puzzle, model, arm, seed) result."""

    puzzle_id: str
    arm: str
    seed: int
    prefill: float
    scores: Scores | None
    solve: SolveResult
    strata: dict[str, object] = field(default_factory=dict)
    model: str = ""
    error: str | None = None

    def as_dict(self) -> dict:
        return {
            "puzzle_id": self.puzzle_id,
            "model": self.model,
            "arm": self.arm,
            "seed": self.seed,
            "prefill": self.prefill,
            "strata": self.strata,
            "scores": None if self.scores is None else self.scores.as_dict(),
            "solve": self.solve.as_dict(),
            "error": self.error,
            "per_slot": {} if self.scores is None else self.scores.per_slot,
        }


def puzzle_strata(puzzle: Puzzle) -> dict[str, object]:
    """Descriptive tags used to slice results."""
    lengths = [s.length for s in puzzle.slots]
    types: dict[str, int] = {}
    for slot in puzzle.slots:
        kind = classify_clue(slot.clue, slot.gold or "")
        types[kind] = types.get(kind, 0) + 1
    return {
        "size": f"{puzzle.grid.height}x{puzzle.grid.width}",
        "slots": len(puzzle.slots),
        "block_density": round(puzzle.grid.block_density, 4),
        "mean_answer_length": round(sum(lengths) / len(lengths), 2) if lengths else 0,
        "clue_types": dict(sorted(types.items())),
        "provenance": puzzle.metadata.get("Source", "unknown"),
        "date": puzzle.metadata.get("Date", ""),
    }


def slot_records(puzzle: Puzzle, scores: Scores) -> list[dict]:
    """Per-slot outcomes, so the report can break results down by clue type."""
    out = []
    for slot in puzzle.slots:
        out.append(
            {
                "slot": slot.id,
                "length": slot.length,
                "length_bucket": length_bucket(slot.length),
                "clue_type": classify_clue(slot.clue, slot.gold or ""),
                "correct": bool(scores.per_slot.get(slot.id, False)),
            }
        )
    return out


def _stderr_trace(event) -> None:
    if event.kind == "candidates":
        return
    progress_log(f"{event.kind}: {event.message}" if event.message else event.kind)


def _arm_for_model(arm: Arm, model: str) -> Arm:
    if arm.name == "a5":
        config = replace(arm.config, model=model, repair_model=model)
    else:
        config = replace(arm.config, model=model)
    return replace(arm, config=config)


def _cell_key(record: dict) -> tuple:
    return (
        record["puzzle_id"],
        record.get("model") or "",
        record["arm"],
        int(record["seed"]),
        float(record.get("prefill") or 0.0),
    )


def _load_cells(path: str) -> dict[tuple, dict]:
    found: dict[tuple, dict] = {}
    if not os.path.isfile(path):
        return found
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            found[_cell_key(payload)] = payload
    return found


def _append_cell(path: str, record: RunRecord, slots: list[dict]) -> None:
    payload = record.as_dict()
    payload["slot_records"] = slots
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload) + "\n")


class Harness:
    """Runs the arm x puzzle x seed matrix and writes a results directory."""

    def __init__(
        self,
        client_factory,
        arms: dict[str, Arm],
        *,
        out_dir: str = "results",
        trace: bool = False,
    ):
        self.client_factory = client_factory
        self.arms = arms
        self.out_dir = out_dir
        self.trace = trace

    def run(
        self,
        puzzles: list[Puzzle],
        arm_names: list[str],
        *,
        seeds: list[int] | None = None,
        prefill_ratios: list[float] | None = None,
        run_id: str | None = None,
        progress=None,
        models: list[str] | None = None,
        retry_errors: bool = False,
        stage: str = "",
        carry_arms: list[str] | None = None,
        rank_by: str = "arm",
    ) -> dict:
        seeds = seeds or [0]
        prefill_ratios = prefill_ratios or [0.0]
        run_id = run_id or time.strftime("run-%Y%m%d-%H%M%S")
        directory = os.path.join(self.out_dir, run_id)
        os.makedirs(directory, exist_ok=True)
        jsonl_path = os.path.join(directory, "cells.jsonl")
        prior = _load_cells(jsonl_path)
        model_axis = list(models) if models else [None]

        record_dicts: list[dict] = []
        live_records: list[RunRecord] = []
        per_slot: list[dict] = []

        for puzzle in puzzles:
            strata = puzzle_strata(puzzle)
            for arm_name in arm_names:
                stock = self.arms[arm_name]
                for seed in seeds:
                    for ratio in prefill_ratios:
                        for model in model_axis:
                            cell_model = model or stock.config.model
                            key = (
                                puzzle.id,
                                cell_model,
                                arm_name,
                                seed,
                                float(ratio),
                            )
                            existing = prior.get(key)
                            if existing and not (
                                retry_errors and existing.get("error")
                            ):
                                rec = {
                                    k: v
                                    for k, v in existing.items()
                                    if k != "slot_records"
                                }
                                record_dicts.append(rec)
                                per_slot.extend(existing.get("slot_records") or [])
                                progress_log(
                                    f"skip {puzzle.id} {arm_name} {cell_model} "
                                    f"(already in cells.jsonl)"
                                )
                                continue

                            live_arm = _arm_for_model(stock, cell_model)
                            prefill = prefill_cells(puzzle, ratio, seed=seed)
                            config = replace(live_arm.config, seed=seed)
                            listeners = []
                            if progress_enabled():
                                listeners.append(_stderr_trace)
                            tracer = Tracer(
                                os.path.join(directory, "trace.jsonl")
                                if self.trace
                                else None,
                                listeners=listeners,
                            )
                            progress_log(
                                f"start {puzzle.id} {arm_name} {cell_model} "
                                f"seed={seed} max_rounds={config.max_rounds} "
                                f"workers={config.max_workers}"
                            )
                            started = time.monotonic()
                            error = None
                            try:
                                client = self.client_factory(puzzle, live_arm, seed)
                                if live_arm.one_shot:
                                    solved = solve_one_shot(
                                        client, puzzle, config, prefill=prefill
                                    )
                                else:
                                    solved = Solver(
                                        client, config, tracer=tracer
                                    ).solve(puzzle, prefill=prefill)
                                scores = score_solution(
                                    puzzle,
                                    solved.solution,
                                    assignment=solved.assignment,
                                )
                            except Exception as exc:
                                error = str(exc)
                                solved = SolveResult(
                                    puzzle=puzzle,
                                    solution={},
                                    seconds=time.monotonic() - started,
                                    warnings=[error],
                                )
                                scores = None

                            record = RunRecord(
                                puzzle_id=puzzle.id,
                                arm=arm_name,
                                seed=seed,
                                prefill=ratio,
                                scores=scores,
                                solve=solved,
                                strata=strata,
                                model=cell_model,
                                error=error,
                            )
                            slots = []
                            if scores is not None:
                                for row in slot_records(puzzle, scores):
                                    row.update(
                                        {
                                            "puzzle_id": puzzle.id,
                                            "arm": arm_name,
                                            "seed": seed,
                                            "model": cell_model,
                                        }
                                    )
                                    slots.append(row)
                            _append_cell(jsonl_path, record, slots)
                            record_dicts.append(record.as_dict())
                            live_records.append(record)
                            per_slot.extend(slots)
                            if progress:
                                progress(record)

        payload = {
            "run_id": run_id,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "stage": stage,
            "rank_by": rank_by,
            "carry_arms": list(carry_arms or []),
            "models": [m for m in model_axis if m] or sorted(
                {r.get("model") for r in record_dicts if r.get("model")}
            ),
            "arms": {
                name: {
                    "label": self.arms[name].label,
                    "description": self.arms[name].description,
                    "model": self.arms[name].config.model,
                    "repair_model": self.arms[name].config.repair_model,
                    "ensemble_model": self.arms[name].config.ensemble_model,
                    "max_rounds": self.arms[name].config.max_rounds,
                    "use_constraints": self.arms[name].config.use_constraints,
                    "use_repair": self.arms[name].config.use_repair,
                    "unknown_mass": self.arms[name].config.unknown_mass,
                }
                for name in arm_names
            },
            "seeds": seeds,
            "prefill_ratios": prefill_ratios,
            "puzzles": [
                {"id": p.id, **puzzle_strata(p)} for p in puzzles
            ],
            "records": record_dicts,
            "calibration": {
                arm: calibration(
                    [
                        pair
                        for r in live_records
                        if r.arm == arm
                        for pair in r.solve.calibration_pairs
                    ]
                ).as_dict()
                for arm in arm_names
            },
            "slot_records": per_slot,
        }
        with open(os.path.join(directory, "results.json"), "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
            fh.write("\n")
        return payload
