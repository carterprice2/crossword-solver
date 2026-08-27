import unittest

from crossword.model import ACROSS, DOWN, Grid, Puzzle, PuzzleError, Slot, solution_from_rows


def grid_from_rows(rows):
    blocks = {
        (r, c) for r, row in enumerate(rows) for c, ch in enumerate(row) if ch == "#"
    }
    return Grid(len(rows), len(rows[0]), blocks)


class TestGrid(unittest.TestCase):
    def test_open_5x5_has_ten_slots(self):
        grid = grid_from_rows(["....."] * 5)
        slots = grid.slots()
        self.assertEqual(len(slots), 10)
        self.assertEqual(sum(1 for s in slots if s.direction == ACROSS), 5)
        self.assertTrue(all(s.length == 5 for s in slots))

    def test_numbering_matches_standard_rules(self):
        #  0 1 2
        #  . . #
        #  . # .
        #  . . .
        grid = grid_from_rows(["..#", ".#.", "..."])
        numbers = grid.numbering()
        # (0,0) starts both A1 and D1. (1,2) starts a 2-cell down entry.
        # (2,0) starts the bottom across entry.
        self.assertEqual(numbers, {(0, 0): 1, (1, 2): 2, (2, 0): 3})
        # (0,1) gets no number: its across run starts at (0,0), and its down
        # run is a single cell because (1,1) is a block.
        self.assertNotIn((0, 1), numbers)
        self.assertNotIn((1, 0), numbers)

    def test_slot_ids_and_cells(self):
        grid = grid_from_rows(["..#", ".#.", "..."])
        by_id = {s.id: s for s in grid.slots()}
        self.assertEqual(by_id["A1"].cells, ((0, 0), (0, 1)))
        self.assertEqual(by_id["D1"].cells, ((0, 0), (1, 0), (2, 0)))
        self.assertEqual(by_id["D2"].cells, ((1, 2), (2, 2)))
        self.assertEqual(by_id["A3"].cells, ((2, 0), (2, 1), (2, 2)))

    def test_min_entry_excludes_short_runs(self):
        grid = grid_from_rows(["..#", ".#.", "..."])
        ids = {s.id for s in grid.slots(min_entry=3)}
        self.assertNotIn("A1", ids)  # only 2 cells long
        self.assertIn("D1", ids)

    def test_symmetry_and_density(self):
        grid = grid_from_rows(["#..", "...", "..#"])
        self.assertTrue(grid.is_symmetric())
        self.assertAlmostEqual(grid.block_density, 2 / 9)
        self.assertFalse(grid_from_rows(["#..", "...", "..."]).is_symmetric())

    def test_isolated_cell_is_reported(self):
        grid = grid_from_rows(["..#", "..#", "###"])
        self.assertEqual(grid.isolated_cells(), [])
        # (0,2) is open but touches nothing long enough to be an entry.
        lonely = grid_from_rows(["...", ".##", ".##"])
        self.assertEqual(lonely.isolated_cells(min_entry=3), [])
        boxed = grid_from_rows(["#.#", "###", "###"])
        self.assertEqual(boxed.isolated_cells(), [(0, 1)])

    def test_rejects_out_of_bounds_block(self):
        with self.assertRaises(PuzzleError):
            Grid(2, 2, [(5, 5)])

    def test_render_round_trips_through_solution(self):
        grid = grid_from_rows(["..#", ".#.", "..."])
        known = {(0, 0): "A", (0, 1): "B", (2, 2): "Z"}
        rows = grid.render(known)
        self.assertEqual(rows[0], "AB#")
        self.assertEqual(solution_from_rows(grid, rows), known)


class TestSlot(unittest.TestCase):
    def setUp(self):
        self.slot = Slot("A1", ACROSS, 1, ((0, 0), (0, 1), (0, 2)))

    def test_pattern_marks_unknown_cells(self):
        self.assertEqual(self.slot.pattern({(0, 1): "E"}), "?E?")
        self.assertEqual(self.slot.pattern({}), "???")

    def test_pattern_collapses_rebus_to_one_char(self):
        self.assertEqual(self.slot.pattern({(0, 0): "HEART"}), "H??")

    def test_spell_needs_every_cell(self):
        self.assertIsNone(self.slot.spell({(0, 0): "C"}))
        full = {(0, 0): "C", (0, 1): "A", (0, 2): "T"}
        self.assertEqual(self.slot.spell(full), "CAT")

    def test_lay_rejects_wrong_length(self):
        self.assertEqual(self.slot.lay("CAT"), {(0, 0): "C", (0, 1): "A", (0, 2): "T"})
        with self.assertRaises(PuzzleError):
            self.slot.lay("CATS")


class TestPuzzle(unittest.TestCase):
    def build(self, golds=None):
        grid = grid_from_rows(["...", "...", "..."])
        golds = golds or {}
        slots = [
            Slot(s.id, s.direction, s.number, s.cells, clue=f"clue {s.id}",
                 gold=golds.get(s.id))
            for s in grid.slots()
        ]
        return Puzzle(grid=grid, slots=slots)

    def test_intersections_are_complete(self):
        puzzle = self.build()
        # A 3x3 open grid: every one of the 9 cells is an across/down crossing.
        self.assertEqual(len(puzzle.intersections()), 9)

    def test_crossers_report_both_indices(self):
        puzzle = self.build()
        crossers = puzzle.crossers("A1")
        self.assertEqual(crossers, [(0, "D1", 0), (1, "D2", 0), (2, "D3", 0)])
        self.assertEqual(puzzle.neighbors("A1"), ["D1", "D2", "D3"])

    def test_gold_solution_agrees_across_and_down(self):
        golds = {"A1": "CAT", "A4": "ARE", "A5": "BEE",
                 "D1": "CAB", "D2": "ARE", "D3": "TEE"}
        puzzle = self.build(golds)
        self.assertEqual(puzzle.validate(), [])
        solution = puzzle.gold_solution()
        self.assertEqual(len(solution), 9)
        self.assertEqual(solution[(0, 0)], "C")
        self.assertEqual(solution[(2, 2)], "E")

    def test_validate_flags_wrong_length_gold(self):
        puzzle = self.build({"A1": "TOOLONG"})
        self.assertTrue(any("A1" in p for p in puzzle.validate()))

    def test_duplicate_slot_ids_rejected(self):
        grid = grid_from_rows(["...", "...", "..."])
        dupe = Slot("A1", ACROSS, 1, ((0, 0), (0, 1), (0, 2)))
        with self.assertRaises(PuzzleError):
            Puzzle(grid=grid, slots=[dupe, dupe])

    def test_missing_slot_lookup_is_clear(self):
        with self.assertRaises(PuzzleError):
            self.build().slot("A99")

    def test_with_clues_replaces_text_only(self):
        puzzle = self.build({"A1": "CAT"})
        updated = puzzle.with_clues({"A1": "Feline"})
        self.assertEqual(updated.slot("A1").clue, "Feline")
        self.assertEqual(updated.slot("A1").gold, "CAT")
        self.assertEqual(puzzle.slot("A1").clue, "clue A1")  # original untouched


class TestImportPurity(unittest.TestCase):
    """The solver core and the eval harness must never import the API client.

    This is what keeps the test suite hermetic: metrics and constraint code
    cannot accidentally acquire a network dependency.
    """

    #: Scoring and constraint logic must not touch the API client at all.
    PURE = (
        "model.py",
        "xd.py",
        "normalize.py",
        "agent/constraints.py",
        "agent/search.py",
        "eval/metrics.py",
        "eval/stats.py",
    )
    #: The harness orchestrates solves, so it may name the ModelClient type --
    #: but it must never construct a live client. It takes a factory instead,
    #: which is what lets the whole eval matrix run against the oracle offline.
    NO_LIVE_CLIENT = ("eval/harness.py", "eval/report.py")

    def _root(self):
        import pathlib

        return pathlib.Path(__file__).resolve().parent.parent / "crossword"

    def test_scoring_and_constraints_never_import_the_client(self):
        import ast

        root = self._root()
        offenders = []
        for rel in self.PURE:
            path = root / rel
            if not path.exists():
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                mod = None
                if isinstance(node, ast.ImportFrom):
                    mod = node.module or ""
                elif isinstance(node, ast.Import):
                    mod = ",".join(a.name for a in node.names)
                if mod and "client" in mod:
                    offenders.append(f"{rel} imports {mod}")
        self.assertEqual(offenders, [])

    def test_harness_never_constructs_a_live_client(self):
        root = self._root()
        offenders = []
        for rel in self.NO_LIVE_CLIENT:
            path = root / rel
            if not path.exists():
                continue
            source = path.read_text(encoding="utf-8")
            if "NebiusClient(" in source:
                offenders.append(f"{rel} constructs NebiusClient")
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
