import unittest

from crossword.model import PuzzleError
from crossword.xd import dump_xd, parse_xd

SIMPLE = """\
Title: Test Mini
Author: Reno Tahoe AI
Date: 2026-01-01


CAT
ARE
BEE


A1. Feline ~ CAT
A4. "You ___ here" ~ ARE
A5. Hive dweller ~ BEE

D1. Taxi ~ CAB
D2. Exist ~ ARE
D3. Golf peg ~ TEE
"""

WITH_BLOCKS = """\
Title: Blocked


AB#
C#D
EFG


A1. First two ~ AB
A4. Left ~ C
A5. Bottom row ~ EFG

D1. Down left ~ ACE
D3. Down right ~ DG
"""

EXPLICIT_HEADERS = """\
## Metadata
Title: Header Style

## Grid
CAT
ARE
BEE

## Clues
A1. Feline ~ CAT
D1. Taxi ~ CAB
"""

REBUS = """\
Title: Rebus Mini
Rebus: 1=CAT


1S
ON


A1. Feline plus S ~ CATS
A3. Switched on ~ ON

D1. Feline concern ~ CATO
D2. Yes, in Marseille ~ SN
"""


class TestParse(unittest.TestCase):
    def test_metadata_grid_and_clues(self):
        puzzle = parse_xd(SIMPLE)
        self.assertEqual(puzzle.metadata["Title"], "Test Mini")
        self.assertEqual(puzzle.metadata["Author"], "Reno Tahoe AI")
        self.assertEqual(puzzle.grid.height, 3)
        self.assertEqual(puzzle.grid.width, 3)
        self.assertEqual(len(puzzle.slots), 6)
        self.assertEqual(puzzle.validate(), [])

    def test_clue_text_and_gold(self):
        puzzle = parse_xd(SIMPLE)
        self.assertEqual(puzzle.slot("A1").clue, "Feline")
        self.assertEqual(puzzle.slot("A1").gold, "CAT")
        self.assertEqual(puzzle.slot("D3").clue, "Golf peg")
        self.assertEqual(puzzle.slot("D3").gold, "TEE")

    def test_clue_containing_quotes_and_underscores(self):
        self.assertEqual(parse_xd(SIMPLE).slot("A4").clue, '"You ___ here"')

    def test_gold_is_cross_consistent(self):
        solution = parse_xd(SIMPLE).gold_solution()
        self.assertEqual(solution[(0, 0)], "C")
        self.assertEqual(solution[(1, 1)], "R")
        self.assertEqual(len(solution), 9)

    def test_blocks_become_blocks(self):
        puzzle = parse_xd(WITH_BLOCKS)
        self.assertIn((0, 2), puzzle.grid.blocks)
        self.assertIn((1, 1), puzzle.grid.blocks)
        self.assertEqual(len(puzzle.grid.blocks), 2)

    def test_explicit_section_headers(self):
        puzzle = parse_xd(EXPLICIT_HEADERS)
        self.assertEqual(puzzle.metadata["Title"], "Header Style")
        self.assertEqual(puzzle.grid.height, 3)
        self.assertEqual(puzzle.slot("A1").clue, "Feline")

    def test_rebus_cell_holds_multiple_letters(self):
        puzzle = parse_xd(REBUS)
        self.assertEqual(puzzle.rebus, {(0, 0): "CAT"})
        self.assertEqual(puzzle.slot("A1").gold, "CATS")
        solution = puzzle.gold_solution()
        self.assertEqual(solution[(0, 0)], "CAT")
        self.assertEqual(solution[(0, 1)], "S")

    def test_grid_without_answers_parses(self):
        blank = SIMPLE.replace("CAT\nARE\nBEE", "...\n...\n...")
        puzzle = parse_xd(blank)
        self.assertEqual(puzzle.slot("A1").gold, "CAT")  # recovered from clue lines
        self.assertEqual(puzzle.slot("A1").clue, "Feline")

    def test_clue_line_without_answer(self):
        text = SIMPLE.replace("A1. Feline ~ CAT", "A1. Feline")
        puzzle = parse_xd(text)
        self.assertEqual(puzzle.slot("A1").clue, "Feline")
        self.assertEqual(puzzle.slot("A1").gold, "CAT")  # still in the grid

    def test_missing_grid_raises(self):
        with self.assertRaises(PuzzleError):
            parse_xd("Title: Nothing\n")

    def test_ragged_grid_raises(self):
        with self.assertRaises(PuzzleError):
            parse_xd("Title: Bad\n\n\nCAT\nAR\nBEE\n\n\nA1. x ~ CAT\n")

    def test_puzzle_id_recorded(self):
        self.assertEqual(parse_xd(SIMPLE, puzzle_id="mini-01").id, "mini-01")


class TestRoundTrip(unittest.TestCase):
    def assert_round_trips(self, text):
        first = parse_xd(text)
        second = parse_xd(dump_xd(first))
        self.assertEqual(first.gold_solution(), second.gold_solution())
        self.assertEqual(
            [(s.id, s.clue, s.gold) for s in first.slots],
            [(s.id, s.clue, s.gold) for s in second.slots],
        )
        self.assertEqual(first.grid.blocks, second.grid.blocks)
        self.assertEqual(first.rebus, second.rebus)

    def test_simple(self):
        self.assert_round_trips(SIMPLE)

    def test_with_blocks(self):
        self.assert_round_trips(WITH_BLOCKS)

    def test_rebus(self):
        self.assert_round_trips(REBUS)

    def test_empty_clues_round_trip(self):
        """A grid-only file (common in the xd corpus) must survive a round trip.

        This is the case that first broke: dumping an empty clue left the line
        as "A1. ~ CAT", and splitting on " ~ " then swallowed the separator.
        """
        text = "Title: Grid Only\n\n\nCAT\nARE\nBEE\n"
        puzzle = parse_xd(text)
        self.assertEqual(puzzle.slot("A1").clue, "")
        self.assertEqual(puzzle.slot("A1").gold, "CAT")
        self.assert_round_trips(text)

    def test_dump_without_answers_hides_gold(self):
        text = dump_xd(parse_xd(SIMPLE), include_answers=False)
        self.assertNotIn("CAT", text)
        self.assertIn("A1. Feline", text)
        self.assertIn("...", text)


if __name__ == "__main__":
    unittest.main()
