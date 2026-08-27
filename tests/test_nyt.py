import os
import tempfile
import unittest

from crossword.eval.nyt import ACROSS, DOWN, GRID, build_puzzle, write_corpus
from crossword.eval.metrics import score_solution


class TestNytFriday20210528(unittest.TestCase):
    def setUp(self):
        self.puzzle = build_puzzle()

    def test_grid_shape_and_fill(self):
        self.assertEqual(self.puzzle.grid.height, 15)
        self.assertEqual(self.puzzle.grid.width, 15)
        self.assertTrue(self.puzzle.grid.is_symmetric())
        self.assertEqual(len(GRID.strip().splitlines()), 15)
        self.assertTrue(all(len(row) == 15 for row in GRID.strip().splitlines()))

    def test_slot_counts_and_theme_entries(self):
        self.assertEqual(len(self.puzzle.slots), 70)
        self.assertEqual(len(ACROSS), 33)
        self.assertEqual(len(DOWN), 37)
        a1 = self.puzzle.slot("A1")
        self.assertEqual(a1.length, 7)
        self.assertEqual(a1.gold, "BEERBAR")
        self.assertEqual(a1.clue, "Building with many drafts")
        self.assertEqual(self.puzzle.slot("A27").gold, "TRANSICON")
        self.assertEqual(self.puzzle.slot("A27").length, 9)
        self.assertEqual(self.puzzle.slot("D15").gold, "EDITORINCHIEF")
        self.assertEqual(self.puzzle.slot("D30").gold, "COMESUP")
        self.assertEqual(self.puzzle.slot("A40").gold, "DICKCHENEY")

    def test_every_clue_is_attached(self):
        missing = [s.id for s in self.puzzle.slots if not s.clue]
        self.assertEqual(missing, [])
        gold_cells = self.puzzle.gold_solution()
        self.assertEqual(len(gold_cells), len(self.puzzle.grid.open_cells()))
        self.assertEqual(self.puzzle.validate(), [])

    def test_gold_scores_exactly(self):
        scores = score_solution(self.puzzle, self.puzzle.gold_solution())
        self.assertTrue(scores.exact)
        self.assertEqual(scores.wcr, 1.0)

    def test_write_corpus_round_trips(self):
        from crossword.xd import load_xd

        with tempfile.TemporaryDirectory() as tmp:
            path = write_corpus(tmp)
            self.assertTrue(os.path.isfile(path))
            loaded = load_xd(path)
            self.assertEqual(loaded.slot("A1").gold, "BEERBAR")
            self.assertEqual(loaded.slot("D1").clue, "Roll in the hay?")
            self.assertEqual(len(loaded.slots), 70)
