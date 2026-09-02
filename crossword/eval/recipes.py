"""Named eval recipes and WCR ranking for the pause-gated tournament."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from collections import defaultdict

from ..client import DEFAULT_MODEL, KNOWN_MODELS

SCREEN_PUZZLE = "mini-07-00-0"
FINAL_PUZZLES = ("mini-07-00-0", "mini-09-00-0", "mini-11-04-0")
STRATEGY_ARMS = ("a0", "a1", "a2", "a3", "a4", "a5", "a6")
MISSING_WINNERS = "no winners.json; pass --from or --arms/--models"


class RecipeError(ValueError):
    """A recipe cannot be expanded from the flags and winners on hand."""


@dataclass
class EvalSpec:
    models: list[str]
    arms: list[str]
    puzzle_ids: list[str]
    stage: str


def _as_list(value) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item)]
    return [part.strip() for part in str(value).split(",") if part.strip()]


def expand_recipe(
    name: str,
    *,
    models=None,
    arms=None,
    puzzle_ids=None,
    winners: dict | None = None,
) -> EvalSpec:
    models_u = _as_list(models)
    arms_u = _as_list(arms)
    puzzles_u = _as_list(puzzle_ids)
    winners = winners or {}

    if name == "screen-arms":
        return EvalSpec(
            models=models_u or [DEFAULT_MODEL],
            arms=arms_u or list(STRATEGY_ARMS),
            puzzle_ids=puzzles_u or [SCREEN_PUZZLE],
            stage=name,
        )
    if name == "screen-models":
        if arms_u:
            run_arms = arms_u
        else:
            prior = list(winners.get("arms") or [])
            if not prior:
                raise RecipeError(MISSING_WINNERS)
            run_arms = [prior[0]]
        return EvalSpec(
            models=models_u or list(KNOWN_MODELS),
            arms=run_arms,
            puzzle_ids=puzzles_u or [SCREEN_PUZZLE],
            stage=name,
        )
    if name == "final-grid":
        run_models = models_u or list(winners.get("models") or [])
        run_arms = arms_u or list(winners.get("arms") or [])
        if not run_models or not run_arms:
            raise RecipeError(MISSING_WINNERS)
        return EvalSpec(
            models=run_models,
            arms=run_arms,
            puzzle_ids=puzzles_u or list(FINAL_PUZZLES),
            stage=name,
        )
    raise RecipeError(f"unknown recipe {name!r}")


def rank_keys(records: list[dict], key: str, *, limit: int = 3) -> list[str]:
    """Mean WCR descending, then cheaper cost_usd, then name. Skip failed cells."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        scores = record.get("scores")
        if not scores:
            continue
        groups[record[key]].append(record)

    def sort_tuple(name: str) -> tuple:
        rows = groups[name]
        wcr = sum(r["scores"]["wcr"] for r in rows) / len(rows)
        costs = [r.get("solve", {}).get("cost_usd") for r in rows]
        if any(c is None for c in costs):
            cost = float("inf")
        else:
            cost = sum(costs) / len(costs)
        return (-wcr, cost, name)

    return sorted(groups, key=sort_tuple)[:limit]


def winners_payload(stage: str, *, arms: list[str], models: list[str]) -> dict:
    return {
        "stage": stage,
        "ranking": "wcr",
        "pick": arms[0] if arms else "",
        "arms": list(arms),
        "models": list(models),
    }


def write_winners(directory: str, payload: dict) -> str:
    path = os.path.join(directory, "winners.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    return path


def load_winners(path: str) -> dict:
    """`path` is a run directory or a winners.json file."""
    if os.path.isdir(path):
        path = os.path.join(path, "winners.json")
    if not os.path.isfile(path):
        raise RecipeError(MISSING_WINNERS)
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
