"""Shared solve runner: corpus listing, puzzle JSON, oracle solve."""

from __future__ import annotations

import os
import unittest

from crossword.xd import parse_xd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MINI = os.path.join(ROOT, "corpus", "mini")


class TestSuiteListing(unittest.TestCase):
    def test_suite_paths_finds_mini_xd_files(self):
        from crossword.run import suite_paths

        paths = suite_paths("mini")
        names = {os.path.basename(p) for p in paths}
        self.assertGreaterEqual(len(paths), 6)
        self.assertTrue(all(p.endswith(".xd") for p in paths))
        self.assertNotIn("mini-07-00-1.xd", names)
        self.assertIn("mini-11-06-1.xd", names)

    def test_unknown_suite_raises_run_error(self):
        from crossword.run import RunError, suite_paths

        with self.assertRaises(RunError):
            suite_paths("does-not-exist")

    def test_list_puzzles_includes_mini_ids(self):
        from crossword.run import list_puzzles

        puzzles = list_puzzles("mini")
        ids = {p["id"] for p in puzzles}
        self.assertIn("mini-07-00-0", ids)
        self.assertIn("mini-11-04-0", ids)
        seven = next(p for p in puzzles if p["id"] == "mini-07-00-0")
        self.assertEqual(seven["height"], 7)
        self.assertEqual(seven["width"], 7)
        self.assertGreater(seven["slots"], 0)
        self.assertTrue(seven["has_gold"])


class TestSerializePuzzle(unittest.TestCase):
    def test_public_payload_omits_gold_answers(self):
        from crossword.run import find_puzzle, serialize_puzzle

        puzzle = find_puzzle("mini-07-00-0")
        payload = serialize_puzzle(puzzle)
        self.assertEqual(payload["id"], "mini-07-00-0")
        self.assertEqual(payload["height"], 7)
        self.assertIn("blocks", payload["grid"])
        self.assertIn("numbers", payload["grid"])
        dumped = str(payload)
        self.assertNotIn('"gold"', dumped)
        across = payload["clues"]["across"]
        self.assertTrue(across)
        self.assertIn("clue", across[0])
        self.assertIn("cells", across[0])
        self.assertNotIn("gold", across[0])
        # The gold fill itself must not be on the public puzzle.
        gold = puzzle.gold_solution()
        fill = "".join(gold[cell] for cell in sorted(gold))
        self.assertNotIn(fill, dumped)


class TestRunSolve(unittest.TestCase):
    def test_unknown_arm_raises_run_error(self):
        from crossword.run import RunError, solver_config

        with self.assertRaises(RunError):
            solver_config(arm="zz")

    def test_oracle_solve_returns_scores(self):
        from crossword.run import make_client, run_solve, solver_config
        from crossword.xd import load_xd

        path = os.path.join(MINI, "mini-07-00-0.xd")
        puzzle = load_xd(path)
        client = make_client(puzzle, backend="oracle", seed=7, oracle_recall=0.95)
        arm, config = solver_config(arm="a3", seed=7)
        result, scores = run_solve(puzzle, client=client, config=config, one_shot=arm.one_shot)
        self.assertIsNotNone(scores)
        self.assertGreater(scores.wcr, 0)
        self.assertGreaterEqual(result.rounds, 1)

    def test_a5_uses_selected_model_for_repair(self):
        from crossword.run import solver_config

        _, config = solver_config(arm="a5", model="picked", repair_model="ignored")
        self.assertEqual(config.model, "picked")
        self.assertEqual(config.repair_model, "picked")

    def test_a4_keeps_ensemble_model(self):
        from crossword.run import solver_config

        _, config = solver_config(
            arm="a4", model="primary", ensemble_model="second"
        )
        self.assertEqual(config.model, "primary")
        self.assertEqual(config.ensemble_model, "second")
        self.assertTrue(config.use_repair)

    def test_a6_turns_off_wildcard(self):
        from crossword.run import solver_config

        _, a3 = solver_config(arm="a3")
        _, a6 = solver_config(arm="a6")
        self.assertGreater(a3.unknown_mass, a6.unknown_mass)
        self.assertLess(a6.unknown_mass, 1e-6)

    def test_repair_arms_keep_a_high_safety_cap(self):
        from crossword.run import solver_config

        _, a2 = solver_config(arm="a2")
        _, a3 = solver_config(arm="a3")
        self.assertEqual(a2.max_rounds, 1)
        self.assertGreaterEqual(a3.max_rounds, 20)
        self.assertGreaterEqual(a3.max_calls, 200)


class TestCellCorrectness(unittest.TestCase):
    def test_marks_wrong_cells_without_revealing_gold_letter(self):
        from crossword.run import cell_correctness

        puzzle = parse_xd(
            "Title: t\nid: t\n\n\nCAT\nARE\nBEE\n\n\n"
            "A1. Feline ~ CAT\nA4. Exist ~ ARE\nA5. Buzzer ~ BEE\n"
            "D1. Taxi ~ CAB\nD2. Exist ~ ARE\nD3. Golf peg ~ TEE\n"
        )
        predicted = dict(puzzle.gold_solution())
        predicted[(0, 0)] = "X"
        cells = cell_correctness(puzzle, predicted)
        by_pos = {(c["r"], c["c"]): c for c in cells}
        self.assertEqual(by_pos[(0, 0)]["letter"], "X")
        self.assertFalse(by_pos[(0, 0)]["correct"])
        self.assertNotIn("gold", by_pos[(0, 0)])
        self.assertTrue(by_pos[(0, 1)]["correct"])
        self.assertEqual(by_pos[(0, 1)]["letter"], "A")

    def test_gold_cells_are_the_answer_key(self):
        from crossword.run import gold_cells

        puzzle = parse_xd(
            "Title: t\nid: t\n\n\nCAT\nARE\nBEE\n\n\n"
            "A1. Feline ~ CAT\nA4. Exist ~ ARE\nA5. Buzzer ~ BEE\n"
            "D1. Taxi ~ CAB\nD2. Exist ~ ARE\nD3. Golf peg ~ TEE\n"
        )
        cells = {(c["r"], c["c"]): c["letter"] for c in gold_cells(puzzle)}
        self.assertEqual(cells[(0, 0)], "C")
        self.assertEqual(cells[(2, 2)], "E")

    def test_annotate_candidates_marks_gold_hit(self):
        from crossword.run import annotate_candidate_event, find_puzzle

        puzzle = find_puzzle("mini-07-00-0")
        gold = puzzle.slots[0].gold
        event = {
            "kind": "candidates",
            "data": {
                "slots": [
                    {
                        "id": puzzle.slots[0].id,
                        "pattern": "?????",
                        "candidates": [
                            {"answer": gold, "confidence": 0.9},
                            {"answer": "XXXXX", "confidence": 0.1},
                        ],
                    }
                ]
            },
        }
        annotated = annotate_candidate_event(event, puzzle)
        slot = annotated["data"]["slots"][0]
        self.assertEqual(slot["gold"], (gold or "").upper())
        self.assertTrue(slot["hit"])


if __name__ == "__main__":
    unittest.main()
