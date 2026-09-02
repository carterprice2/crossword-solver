import os
import tempfile
import unittest

from crossword.client import DEFAULT_MODEL, KNOWN_MODELS
from crossword.eval.recipes import (
    FINAL_PUZZLES,
    SCREEN_PUZZLE,
    EvalSpec,
    RecipeError,
    expand_recipe,
    load_winners,
    rank_keys,
    winners_payload,
    write_winners,
)


class TestExpandRecipe(unittest.TestCase):
    def test_screen_arms_defaults(self):
        spec = expand_recipe("screen-arms")
        self.assertEqual(spec.stage, "screen-arms")
        self.assertEqual(spec.models, [DEFAULT_MODEL])
        self.assertEqual(spec.arms, ["a0", "a1", "a2", "a3", "a4", "a5", "a6"])
        self.assertEqual(spec.puzzle_ids, [SCREEN_PUZZLE])

    def test_screen_arms_user_flags_win(self):
        spec = expand_recipe(
            "screen-arms", models=["m"], arms=["a3"], puzzle_ids=["p"]
        )
        self.assertEqual(spec.models, ["m"])
        self.assertEqual(spec.arms, ["a3"])
        self.assertEqual(spec.puzzle_ids, ["p"])

    def test_screen_models_requires_from_or_arms(self):
        with self.assertRaises(RecipeError) as ctx:
            expand_recipe("screen-models")
        self.assertIn("winners.json", str(ctx.exception))

    def test_screen_models_uses_best_arm_from_winners(self):
        spec = expand_recipe(
            "screen-models",
            winners={"arms": ["a5", "a3", "a4"], "models": [DEFAULT_MODEL]},
        )
        self.assertEqual(spec.arms, ["a5"])
        self.assertEqual(spec.models, list(KNOWN_MODELS))
        self.assertEqual(spec.puzzle_ids, [SCREEN_PUZZLE])

    def test_screen_models_arms_override_skips_winners(self):
        spec = expand_recipe("screen-models", arms=["a3"])
        self.assertEqual(spec.arms, ["a3"])
        self.assertEqual(spec.models, list(KNOWN_MODELS))

    def test_final_grid_uses_winner_lists(self):
        spec = expand_recipe(
            "final-grid",
            winners={
                "arms": ["a3", "a5", "a4"],
                "models": ["m1", "m2", "m3"],
            },
        )
        self.assertEqual(spec.arms, ["a3", "a5", "a4"])
        self.assertEqual(spec.models, ["m1", "m2", "m3"])
        self.assertEqual(spec.puzzle_ids, list(FINAL_PUZZLES))

    def test_final_grid_requires_both_axes(self):
        with self.assertRaises(RecipeError):
            expand_recipe("final-grid", winners={"arms": ["a3"]})
        with self.assertRaises(RecipeError):
            expand_recipe("final-grid")

    def test_unknown_recipe(self):
        with self.assertRaises(RecipeError):
            expand_recipe("nope")


class TestRankKeys(unittest.TestCase):
    def test_top3_by_wcr(self):
        records = [
            _cell("a1", 0.5, 0.01),
            _cell("a3", 0.9, 0.10),
            _cell("a2", 0.7, 0.02),
            _cell("a4", 0.4, 0.01),
        ]
        self.assertEqual(rank_keys(records, "arm"), ["a3", "a2", "a1"])

    def test_tie_break_lower_cost_then_name(self):
        records = [
            _cell("b", 0.8, 0.20),
            _cell("a", 0.8, 0.20),
            _cell("c", 0.8, 0.05),
        ]
        self.assertEqual(rank_keys(records, "arm"), ["c", "a", "b"])

    def test_skips_failed_cells(self):
        records = [
            {"arm": "a0", "scores": None, "solve": {"cost_usd": None}},
            _cell("a3", 1.0, 0.01),
        ]
        self.assertEqual(rank_keys(records, "arm"), ["a3"])

    def test_none_cost_sorts_last_on_tie(self):
        records = [
            _cell("cheap", 0.5, 0.01),
            _cell("unknown", 0.5, None),
        ]
        self.assertEqual(rank_keys(records, "arm"), ["cheap", "unknown"])


class TestWinnersIo(unittest.TestCase):
    def test_round_trip(self):
        payload = winners_payload(
            "screen-arms", arms=["a3", "a5"], models=[DEFAULT_MODEL]
        )
        with tempfile.TemporaryDirectory() as tmp:
            write_winners(tmp, payload)
            loaded = load_winners(tmp)
        self.assertEqual(loaded["stage"], "screen-arms")
        self.assertEqual(loaded["arms"], ["a3", "a5"])
        self.assertEqual(loaded["ranking"], "wcr")


def _cell(arm, wcr, cost):
    return {
        "arm": arm,
        "model": "m",
        "scores": {"wcr": wcr},
        "solve": {"cost_usd": cost},
    }
