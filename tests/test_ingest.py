import unittest

from crossword.ingest import IngestError, assemble, decode_image, parse_clue_list, rows_to_grid
from crossword.xd import parse_xd

# 3×3 with a center block. Numbering: A1/D1 at (0,0), D2 at (0,2), A3 at (2,0).
MINI = [
    "C.T",
    ".#.",
    "T.G",
]


class TestParseClues(unittest.TestCase):
    def test_numbered_across(self):
        pairs = parse_clue_list("1. Feline\n3. Canine", direction="A")
        self.assertEqual(pairs, [("A1", "Feline"), ("A3", "Canine")])

    def test_prefixed_direction(self):
        pairs = parse_clue_list("D1. Vertical\nD2. Other", direction="A")
        self.assertEqual(pairs, [("D1", "Vertical"), ("D2", "Other")])

    def test_sequential_when_unnumbered(self):
        pairs = parse_clue_list("Feline\nCanine", direction="A")
        self.assertEqual([p[1] for p in pairs], ["Feline", "Canine"])
        self.assertEqual([p[0] for p in pairs], ["A1", "A2"])

    def test_skips_blank_lines(self):
        pairs = parse_clue_list("1. One\n\n2. Two\n", direction="D")
        self.assertEqual(pairs, [("D1", "One"), ("D2", "Two")])


class TestRows(unittest.TestCase):
    def test_letters_are_prefill_not_gold(self):
        grid, prefill = rows_to_grid(MINI)
        self.assertEqual(grid.height, 3)
        self.assertEqual(grid.width, 3)
        self.assertIn((1, 1), grid.blocks)
        self.assertEqual(prefill[(0, 0)], "C")
        self.assertNotIn((0, 1), prefill)

    def test_rejects_non_rectangular(self):
        with self.assertRaises(IngestError):
            rows_to_grid(["..", "..."])

    def test_rejects_too_large(self):
        with self.assertRaises(IngestError):
            rows_to_grid(["." * 16] * 16)

    def test_dots_only_have_no_prefill(self):
        grid, prefill = rows_to_grid(["...", ".#.", "..."])
        self.assertEqual(prefill, {})
        self.assertEqual(len(grid.slots()), 4)


class TestAssemble(unittest.TestCase):
    def test_ready_when_counts_match(self):
        across = "1. C to T\n3. T to G"
        down = "1. C to T\n2. T to G"
        draft = assemble(MINI, across, down, puzzle_id="upload-ab")
        self.assertEqual(draft.status, "ready")
        self.assertIsNotNone(draft.puzzle)
        assert draft.puzzle is not None
        self.assertFalse(draft.puzzle.has_gold())
        self.assertEqual(draft.puzzle.id, "upload-ab")
        self.assertEqual(draft.puzzle.slot("A1").clue, "C to T")
        self.assertEqual(draft.puzzle.slot("A3").clue, "T to G")
        self.assertEqual(draft.prefill[(0, 0)], "C")
        self.assertIsNone(draft.puzzle.slot("A1").gold)

    def test_sequential_zips_in_slot_order(self):
        draft = assemble(MINI, "First across\nSecond across", "First down\nSecond down", puzzle_id="x")
        self.assertEqual(draft.status, "ready")
        assert draft.puzzle is not None
        self.assertEqual(draft.puzzle.slot("A1").clue, "First across")
        self.assertEqual(draft.puzzle.slot("A3").clue, "Second across")
        self.assertEqual(draft.puzzle.slot("D1").clue, "First down")
        self.assertEqual(draft.puzzle.slot("D2").clue, "Second down")

    def test_needs_edit_on_count_mismatch(self):
        draft = assemble(MINI, "1. only one", "1. a\n2. b", puzzle_id="x")
        self.assertEqual(draft.status, "needs_edit")
        self.assertIsNone(draft.puzzle)
        self.assertEqual(draft.across_slots, 2)
        self.assertEqual(draft.across_clues, 1)

    def test_unknown_number(self):
        draft = assemble(MINI, "1. a\n9. missing", "1. a\n2. b", puzzle_id="x")
        self.assertEqual(draft.status, "needs_edit")
        self.assertIn("A9", draft.unknown_numbers)

    def test_isolated_cell_needs_edit(self):
        rows = ["###", "#.#", "###"]
        draft = assemble(rows, "1. nope", "1. nope", puzzle_id="x")
        self.assertEqual(draft.status, "needs_edit")


class TestDecodeImage(unittest.TestCase):
    PNG = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )

    def test_data_url(self):
        data, mime = decode_image(f"data:image/png;base64,{self.PNG}")
        self.assertEqual(mime, "image/png")
        self.assertTrue(data.startswith(b"\x89PNG"))

    def test_raw_base64(self):
        data, mime = decode_image(self.PNG)
        self.assertEqual(mime, "image/png")
        self.assertTrue(data.startswith(b"\x89PNG"))

    def test_rejects_garbage(self):
        with self.assertRaises(IngestError):
            decode_image("not-an-image")
    def test_xd_without_gold(self):
        text = (
            "Title: X\n\n\n"
            "...\n.#.\n...\n\n\n"
            "A1. One\nA3. Two\nD1. Three\nD2. Four\n"
        )
        puzzle = parse_xd(text, puzzle_id="xd-1")
        self.assertFalse(puzzle.has_gold())
        self.assertEqual(puzzle.slot("A1").clue, "One")
