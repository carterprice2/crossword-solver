import glob
import json
import os
import unittest

from crossword.gen.bank import Bank, Entry, load_bank
from crossword.gen.fill import FillError, build_puzzle, fill_grid
from crossword.gen.grids import (
    DEFAULT_MAX_RUN,
    dump_template,
    is_connected,
    parse_template,
    run_lengths,
    search_templates,
    validate_template,
)
from crossword.xd import load_xd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(ROOT, "corpus")


def toy_bank():
    words = [
        ("CAT", "Feline", 90), ("CAB", "Taxi", 80), ("ARE", "Exist", 95),
        ("BEE", "Buzzer", 70), ("TEE", "Golf peg", 60), ("ART", "Craft", 85),
        ("RAT", "Rodent", 50), ("TAR", "Pitch", 40), ("EAR", "Listener", 65),
        ("ERA", "Age", 75), ("ATE", "Consumed", 55), ("ETA", "Arrival figure", 45),
    ]
    return Bank([Entry(w, c, "definition", f) for w, c, f in words])


class TestBank(unittest.TestCase):
    def setUp(self):
        self.bank = toy_bank()

    def test_groups_by_length(self):
        self.assertEqual(self.bank.length_counts(), {3: 12})
        self.assertEqual(len(self.bank.words(3)), 12)
        self.assertEqual(self.bank.words(5), frozenset())

    def test_matching_pattern(self):
        self.assertEqual(self.bank.matching("CA?"), frozenset({"CAT", "CAB"}))
        self.assertEqual(self.bank.matching("???"), self.bank.words(3))
        self.assertEqual(self.bank.matching("ZZZ"), frozenset())

    def test_with_any_letter_is_a_union(self):
        got = self.bank.with_any_letter(3, 0, frozenset({"C", "A"}))
        self.assertEqual(got, frozenset({"CAT", "CAB", "ARE", "ART", "ATE"}))

    def test_ordered_is_most_common_first(self):
        order = self.bank.ordered(3)
        self.assertEqual(order[0], "ARE")  # freq 95
        self.assertEqual(len(order), 12)

    def test_clue_and_freq_lookup(self):
        self.assertEqual(self.bank.clue_for("CAT"), "Feline")
        self.assertEqual(self.bank.freq_for("CAT"), 90)
        self.assertEqual(self.bank.clue_for("NOPE"), "")


class TestTemplates(unittest.TestCase):
    def test_parse_and_dump_round_trip(self):
        text = "...#...\n...#...\n.......\n"
        grid = parse_template(text)
        self.assertEqual(grid.height, 3)
        self.assertEqual(dump_template(grid), text)

    def test_ragged_template_rejected(self):
        with self.assertRaises(ValueError):
            parse_template("...\n..\n")

    def test_validate_catches_asymmetry(self):
        grid = parse_template("#..\n...\n...")
        self.assertTrue(any("symmetric" in p for p in validate_template(grid)))

    def test_validate_catches_a_long_run(self):
        grid = parse_template("." * 9 + "\n" + ("." * 9 + "\n") * 8)
        problems = validate_template(grid, max_run=6)
        self.assertTrue(any("longest run" in p for p in problems))

    def test_validate_catches_a_short_run(self):
        grid = parse_template("..#..\n.....\n.....\n.....\n..#..")
        self.assertTrue(any("shorter than" in p for p in validate_template(grid)))

    def test_connectivity(self):
        self.assertTrue(is_connected(parse_template("...\n...\n...")))
        split = parse_template("...\n###\n...")
        self.assertFalse(is_connected(split))

    def test_search_finds_valid_templates(self):
        found = search_templates(9, max_run=6, target_density=0.18, seed=1,
                                 attempts=4000, want=2)
        self.assertTrue(found, "expected at least one valid 9x9 template")
        for grid in found:
            self.assertEqual(validate_template(grid, max_run=6), [])
            self.assertTrue(grid.is_symmetric())
            self.assertLessEqual(max(run_lengths(grid)), 6)

    def test_search_is_deterministic(self):
        a = search_templates(9, seed=4, attempts=2000, want=1)
        b = search_templates(9, seed=4, attempts=2000, want=1)
        self.assertEqual([g.blocks for g in a], [g.blocks for g in b])


class TestFill(unittest.TestCase):
    def test_fills_a_3x3(self):
        grid = parse_template("...\n...\n...")
        words, stats = fill_grid(grid, toy_bank(), seed=0, time_limit=10)
        self.assertEqual(len(words), 6)
        self.assertGreater(stats.nodes, 0)

    def test_words_are_distinct(self):
        grid = parse_template("...\n...\n...")
        words, _ = fill_grid(grid, toy_bank(), seed=1, time_limit=10)
        self.assertEqual(len(set(words.values())), len(words))

    def test_crossings_agree(self):
        grid = parse_template("...\n...\n...")
        words, _ = fill_grid(grid, toy_bank(), seed=0, time_limit=10)
        by_id = {slot.id: slot for slot in grid.slots()}
        for slot in by_id.values():
            for index, cell in enumerate(slot.cells):
                for other in by_id.values():
                    if other.direction == slot.direction or cell not in other.cells:
                        continue
                    self.assertEqual(
                        words[slot.id][index],
                        words[other.id][other.index_of(cell)],
                        f"{slot.id} and {other.id} disagree at {cell}",
                    )

    def test_missing_length_reports_clearly(self):
        grid = parse_template(".....\n.....\n.....\n.....\n.....")
        with self.assertRaises(FillError) as ctx:
            fill_grid(grid, toy_bank(), seed=0, time_limit=5)
        self.assertIn("length", str(ctx.exception))

    def test_build_puzzle_is_self_consistent(self):
        grid = parse_template("...\n...\n...")
        puzzle, _ = build_puzzle(grid, toy_bank(), seed=0, time_limit=10)
        self.assertEqual(puzzle.validate(), [])
        # Crossing letters must agree, which gold_solution enforces.
        solution = puzzle.gold_solution()
        self.assertEqual(len(solution), 9)
        for slot in puzzle.slots:
            self.assertEqual(slot.spell(solution), slot.gold)
            self.assertTrue(slot.clue)

    def test_deterministic_for_a_seed(self):
        grid = parse_template("...\n...\n...")
        a, _ = fill_grid(grid, toy_bank(), seed=3, time_limit=10)
        b, _ = fill_grid(grid, toy_bank(), seed=3, time_limit=10)
        self.assertEqual(a, b)


class TestCommittedCorpus(unittest.TestCase):
    """The shipped corpus must be valid, self-consistent, and described."""

    def setUp(self):
        self.paths = sorted(glob.glob(os.path.join(CORPUS, "mini", "*.xd")))
        if not self.paths:
            self.skipTest("corpus not generated")

    def test_every_puzzle_parses_and_validates(self):
        for path in self.paths:
            with self.subTest(path=os.path.basename(path)):
                puzzle = load_xd(path)
                self.assertEqual(puzzle.validate(), [])
                self.assertTrue(puzzle.has_gold())

    def test_every_clue_is_present(self):
        for path in self.paths:
            puzzle = load_xd(path)
            for slot in puzzle.slots:
                with self.subTest(path=os.path.basename(path), slot=slot.id):
                    self.assertTrue(slot.clue.strip(), f"{slot.id} has no clue")

    def test_gold_agrees_at_every_crossing(self):
        for path in self.paths:
            puzzle = load_xd(path)
            solution = puzzle.gold_solution()
            for inter in puzzle.intersections():
                across = puzzle.slot(inter.across).gold
                down = puzzle.slot(inter.down).gold
                with self.subTest(path=os.path.basename(path), cell=inter.cell):
                    self.assertEqual(
                        across[inter.across_index], down[inter.down_index]
                    )
                    self.assertEqual(solution[inter.cell], across[inter.across_index])

    def test_grids_are_symmetric_and_well_formed(self):
        for path in self.paths:
            puzzle = load_xd(path)
            with self.subTest(path=os.path.basename(path)):
                self.assertTrue(puzzle.grid.is_symmetric())
                self.assertEqual(puzzle.grid.isolated_cells(), [])
                self.assertLessEqual(
                    max(s.length for s in puzzle.slots), DEFAULT_MAX_RUN
                )

    def test_no_duplicate_answers_within_a_puzzle(self):
        for path in self.paths:
            puzzle = load_xd(path)
            answers = [s.gold for s in puzzle.slots]
            with self.subTest(path=os.path.basename(path)):
                self.assertEqual(len(answers), len(set(answers)))

    def test_clue_never_contains_its_own_answer(self):
        """A clue must not hand over its answer as a word.

        Matched on word boundaries, not as a substring: LAR clued "A tutelary
        deity" is fine -- "lar" inside "tutelary" is a coincidence no solver
        could use -- whereas "A lar is..." would give the answer away.
        """
        import re

        for path in self.paths:
            puzzle = load_xd(path)
            for slot in puzzle.slots:
                answer = (slot.gold or "").lower()
                with self.subTest(path=os.path.basename(path), slot=slot.id):
                    self.assertIsNone(
                        re.search(rf"\b{re.escape(answer)}\b", slot.clue.lower()),
                        f"{slot.id}: clue {slot.clue!r} contains {answer!r}",
                    )

    def test_manifest_describes_every_puzzle(self):
        manifest_path = os.path.join(CORPUS, "manifest.json")
        if not os.path.isfile(manifest_path):
            self.skipTest("no manifest")
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)
        listed = {entry["id"] for entry in manifest["puzzles"]}
        on_disk = {os.path.splitext(os.path.basename(p))[0] for p in self.paths}
        self.assertEqual(listed, on_disk)
        # Provenance and licensing must be recorded, not assumed.
        self.assertTrue(manifest["bank"]["sources"])
        for source in manifest["bank"]["sources"]:
            self.assertIn("license", source)

    def test_templates_are_committed_alongside(self):
        templates = glob.glob(os.path.join(CORPUS, "grids", "*.txt"))
        self.assertTrue(templates, "expected committed grid templates")
        for path in templates:
            with self.subTest(path=os.path.basename(path)):
                with open(path, encoding="utf-8") as fh:
                    grid = parse_template(fh.read())
                self.assertEqual(validate_template(grid, max_run=DEFAULT_MAX_RUN), [])


class TestShippedBank(unittest.TestCase):
    def setUp(self):
        path = os.path.join(CORPUS, "bank", "words.tsv")
        if not os.path.isfile(path):
            self.skipTest("bank not built")
        self.bank = load_bank(path)

    def test_has_usable_supply_at_every_length(self):
        counts = self.bank.length_counts()
        for length in range(3, 9):
            with self.subTest(length=length):
                self.assertGreater(counts.get(length, 0), 50)

    def test_entries_are_uppercase_letters_only(self):
        for entry in self.bank.entries[:500]:
            with self.subTest(word=entry.word):
                self.assertTrue(entry.word.isalpha() and entry.word.isupper())
                self.assertTrue(entry.clue)


if __name__ == "__main__":
    unittest.main()
