import unittest

from crossword.eval.metrics import (
    aggregate,
    assignment_consistency,
    calibration,
    grid_consistency,
    mean_and_sd,
    prefill_cells,
    score_solution,
)
from crossword.eval.stats import (
    mcnemar,
    paired_bootstrap,
    required_n,
    wilson,
)
from crossword.xd import parse_xd

MINI = """\
Title: Metric Mini


CAT
ARE
BEE


A1. Feline ~ CAT
A4. Exist ~ ARE
A5. Buzzer ~ BEE

D1. Taxi ~ CAB
D2. Exist ~ ARE
D3. Golf peg ~ TEE
"""


class TestScoreSolution(unittest.TestCase):
    def setUp(self):
        self.puzzle = parse_xd(MINI)
        self.gold = self.puzzle.gold_solution()

    def test_perfect_solution(self):
        s = score_solution(self.puzzle, dict(self.gold))
        self.assertEqual(s.wcr, 1.0)
        self.assertEqual(s.lcr, 1.0)
        self.assertEqual(s.icr, 1.0)
        self.assertTrue(s.exact)
        self.assertEqual(s.cell_precision, 1.0)
        self.assertEqual(s.cell_recall, 1.0)

    def test_empty_solution(self):
        s = score_solution(self.puzzle, {})
        self.assertEqual(s.wcr, 0.0)
        self.assertEqual(s.lcr, 0.0)
        self.assertFalse(s.exact)
        self.assertEqual(s.cells_filled, 0)
        # Precision is undefined with nothing filled; report 0 rather than NaN.
        self.assertEqual(s.cell_precision, 0.0)

    def test_one_wrong_letter_breaks_exact_and_two_slots(self):
        predicted = dict(self.gold)
        predicted[(0, 0)] = "X"  # breaks A1 (CAT) and D1 (CAB)
        s = score_solution(self.puzzle, predicted)
        self.assertFalse(s.exact)
        self.assertAlmostEqual(s.lcr, 8 / 9)
        self.assertAlmostEqual(s.wcr, 4 / 6)
        self.assertFalse(s.per_slot["A1"])
        self.assertFalse(s.per_slot["D1"])
        self.assertTrue(s.per_slot["A4"])

    def test_precision_and_recall_separate_wrong_from_blank(self):
        """A solver that declines must not score the same as one that guesses."""
        blank = {c: v for c, v in self.gold.items() if c != (0, 0)}
        wrong = dict(self.gold)
        wrong[(0, 0)] = "X"

        blank_scores = score_solution(self.puzzle, blank)
        wrong_scores = score_solution(self.puzzle, wrong)

        self.assertAlmostEqual(blank_scores.lcr, wrong_scores.lcr)
        self.assertAlmostEqual(blank_scores.cell_recall, wrong_scores.cell_recall)
        # But precision distinguishes them: declining kept precision perfect.
        self.assertEqual(blank_scores.cell_precision, 1.0)
        self.assertLess(wrong_scores.cell_precision, 1.0)

    def test_icr_needs_no_gold(self):
        """A fully filled cell grid is always consistent with itself."""
        wrong_but_consistent = {c: "Z" for c in self.puzzle.grid.open_cells()}
        self.assertEqual(grid_consistency(self.puzzle, wrong_but_consistent), 1.0)
        scores = score_solution(self.puzzle, wrong_but_consistent)
        self.assertEqual(scores.icr, 1.0)
        self.assertEqual(scores.lcr, 0.0)

    def test_icr_from_assignment_detects_crossing_disagreement(self):
        """The reported ICR must use slot strings, not the cell dict."""
        assignment = {
            "A1": "CAT",
            "A4": "ARE",
            "A5": "BEE",
            "D1": "XYZ",  # disagrees with CAT at (0,0)
            "D2": "ARE",
            "D3": "TEE",
        }
        gold = self.puzzle.gold_solution()
        # First-writer cell grid is consistent; assignment is not.
        scores = score_solution(self.puzzle, gold, assignment=assignment)
        self.assertLess(scores.icr, 1.0)
        self.assertLess(assignment_consistency(self.puzzle, assignment), 1.0)
        self.assertEqual(
            assignment_consistency(self.puzzle, {s.id: s.gold for s in self.puzzle.slots}),
            1.0,
        )


class TestPrefill(unittest.TestCase):
    def setUp(self):
        self.puzzle = parse_xd(MINI)

    def test_zero_ratio_reveals_nothing(self):
        self.assertEqual(prefill_cells(self.puzzle, 0.0), {})

    def test_ratio_controls_count(self):
        revealed = prefill_cells(self.puzzle, 0.5, seed=1)
        self.assertEqual(len(revealed), round(0.5 * 9))

    def test_revealed_letters_are_correct(self):
        gold = self.puzzle.gold_solution()
        for cell, letter in prefill_cells(self.puzzle, 0.4, seed=2).items():
            self.assertEqual(letter, gold[cell])

    def test_full_ratio_reveals_everything(self):
        self.assertEqual(
            prefill_cells(self.puzzle, 1.0), self.puzzle.gold_solution()
        )

    def test_deterministic_for_a_seed(self):
        self.assertEqual(
            prefill_cells(self.puzzle, 0.4, seed=5),
            prefill_cells(self.puzzle, 0.4, seed=5),
        )


class TestCalibration(unittest.TestCase):
    def test_perfectly_calibrated(self):
        pairs = [(0.9, True)] * 9 + [(0.9, False)]
        result = calibration(pairs)
        self.assertAlmostEqual(result.ece, 0.0, places=6)

    def test_overconfident_model_scores_badly(self):
        pairs = [(0.95, False)] * 10
        result = calibration(pairs)
        self.assertAlmostEqual(result.ece, 0.95, places=6)
        self.assertAlmostEqual(result.brier, 0.9025, places=4)

    def test_empty_is_safe(self):
        self.assertEqual(calibration([]).n, 0)

    def test_confidence_of_one_lands_in_last_bin(self):
        result = calibration([(1.0, True)], n_bins=10)
        self.assertEqual(len(result.bins), 1)
        self.assertEqual(result.bins[0]["n"], 1)


class TestAggregate(unittest.TestCase):
    def test_averages_and_counts(self):
        puzzle = parse_xd(MINI)
        gold = puzzle.gold_solution()
        wrong = dict(gold)
        wrong[(0, 0)] = "X"
        summary = aggregate([score_solution(puzzle, gold), score_solution(puzzle, wrong)])
        self.assertEqual(summary["n"], 2)
        self.assertEqual(summary["exact_count"], 1)
        self.assertAlmostEqual(summary["exact"], 0.5)

    def test_empty(self):
        self.assertEqual(aggregate([]), {})

    def test_mean_and_sd(self):
        mean, sd = mean_and_sd([1.0, 2.0, 3.0])
        self.assertAlmostEqual(mean, 2.0)
        self.assertAlmostEqual(sd, 1.0)
        self.assertEqual(mean_and_sd([5.0]), (5.0, 0.0))


class TestPairedBootstrap(unittest.TestCase):
    def test_clear_improvement_is_detected(self):
        a = [0.9] * 20
        b = [0.5] * 20
        result = paired_bootstrap(a, b, resamples=2000, seed=1)
        self.assertAlmostEqual(result.delta.point, 0.4)
        self.assertTrue(result.significant)
        self.assertEqual(result.p_better, 1.0)
        self.assertEqual(result.wins, 20)

    def test_no_difference_is_not_significant(self):
        values = [0.5, 0.6, 0.4, 0.7, 0.5, 0.55, 0.45, 0.6]
        result = paired_bootstrap(values, list(values), resamples=2000, seed=1)
        self.assertAlmostEqual(result.delta.point, 0.0)
        self.assertFalse(result.significant)
        self.assertEqual(result.ties, len(values))

    def test_pairing_survives_high_between_puzzle_variance(self):
        """The point of pairing: a consistent +0.05 is detectable even when
        puzzle difficulty swings far more than the effect being measured."""
        base = [0.1, 0.9, 0.3, 0.8, 0.2, 0.95, 0.15, 0.75, 0.4, 0.6]
        a = [min(1.0, x + 0.05) for x in base]
        result = paired_bootstrap(a, base, resamples=4000, seed=3)
        self.assertTrue(result.significant)

    def test_mismatched_lengths_raise(self):
        with self.assertRaises(ValueError):
            paired_bootstrap([1.0], [1.0, 2.0])

    def test_empty_is_safe(self):
        self.assertEqual(paired_bootstrap([], []).n, 0)


class TestWilson(unittest.TestCase):
    def test_interval_contains_point(self):
        interval = wilson(8, 10)
        self.assertLess(interval.low, 0.8)
        self.assertGreater(interval.high, 0.8)

    def test_stays_inside_zero_one_at_extremes(self):
        """The normal approximation escapes [0,1] here; Wilson must not."""
        self.assertGreaterEqual(wilson(0, 10).low, 0.0)
        self.assertLessEqual(wilson(10, 10).high, 1.0)

    def test_exact_solve_at_n40_is_wide(self):
        """Documents why exact-solve is reported as descriptive only."""
        interval = wilson(32, 40)  # 80%
        self.assertGreater(interval.high - interval.low, 0.20)

    def test_zero_n(self):
        self.assertEqual(wilson(0, 0).point, 0.0)


class TestMcNemar(unittest.TestCase):
    def test_all_discordant_one_way_is_significant(self):
        a = [True] * 10
        b = [False] * 10
        result = mcnemar(a, b)
        self.assertEqual(result["a_only"], 10)
        self.assertLess(result["p_value"], 0.01)

    def test_agreement_gives_no_information(self):
        result = mcnemar([True, False, True], [True, False, True])
        self.assertEqual(result["discordant"], 0)
        self.assertEqual(result["p_value"], 1.0)

    def test_balanced_discordance_is_not_significant(self):
        a = [True, False, True, False]
        b = [False, True, False, True]
        self.assertGreater(mcnemar(a, b)["p_value"], 0.5)

    def test_mismatched_lengths_raise(self):
        with self.assertRaises(ValueError):
            mcnemar([True], [True, False])


class TestRequiredN(unittest.TestCase):
    def test_small_difference_needs_many_puzzles(self):
        """80% vs 70% exact-solve needs a few hundred puzzles at 80% power."""
        n = required_n(0.8, 0.7)
        self.assertGreater(n, 200)
        self.assertLess(n, 500)

    def test_large_difference_needs_fewer(self):
        self.assertLess(required_n(0.9, 0.3), 30)

    def test_identical_rates(self):
        self.assertEqual(required_n(0.5, 0.5), 0)


if __name__ == "__main__":
    unittest.main()
