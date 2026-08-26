import os
import unittest

from crossword.agent.constraints import SlotGraph, WILDCARD
from crossword.agent.solver import Solver, SolverConfig
from crossword.agent.verify import (
    definition_score,
    rescore_with_dictionary,
    verify_fill,
)
from crossword.client import ScriptedClient
from crossword.schemas import Candidate
from crossword.xd import load_xd, parse_xd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MINI07 = os.path.join(ROOT, "corpus", "mini", "mini-07-00-0.xd")

TINY = """\
Title: Verifier Mini
Source: generated


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


class TestDefinitionScore(unittest.TestCase):
    def test_exact_match_is_one(self):
        clue = "In a row, line, or rank"
        self.assertEqual(definition_score(clue, clue), 1.0)

    def test_unrelated_is_low(self):
        self.assertLess(
            definition_score("In a row, line, or rank", "A kind of close carriage"),
            0.2,
        )


class TestRescoreWithDictionary(unittest.TestCase):
    def test_boosts_a_proposed_word_whose_sense_matches(self):
        puzzle = parse_xd(TINY)
        candidates = [Candidate("A1", "CAT", 0.5), Candidate("A1", "DOG", 0.8)]
        out = rescore_with_dictionary(
            candidates, puzzle, {"CAT": "Feline", "DOG": "A barking animal"}
        )
        by_word = {c.answer: c.confidence for c in out}
        self.assertGreater(by_word["CAT"], 0.5)
        self.assertEqual(by_word["DOG"], 0.8)

    def test_does_not_inject_unproposed_words(self):
        puzzle = parse_xd(TINY)
        out = rescore_with_dictionary(
            [Candidate("A1", "DOG", 0.8)],
            puzzle,
            {"CAT": "Feline", "DOG": "A barking animal"},
        )
        self.assertEqual([c.answer for c in out], ["DOG"])


class TestVerifyFill(unittest.TestCase):
    def setUp(self):
        self.puzzle = parse_xd(TINY)
        self.graph = SlotGraph(self.puzzle)

    def test_gold_assignment_is_complete(self):
        gold = {s.id: s.gold for s in self.puzzle.slots}
        report = verify_fill(self.puzzle, self.graph, gold)
        self.assertTrue(report.complete)
        self.assertEqual(report.issues, [])

    def test_blank_is_an_issue(self):
        gold = {s.id: WILDCARD for s in self.puzzle.slots}
        report = verify_fill(self.puzzle, self.graph, gold)
        self.assertFalse(report.ok)
        self.assertIn("blank", {i.kind for i in report.issues})

    def test_length_mismatch(self):
        gold = {s.id: s.gold for s in self.puzzle.slots}
        gold["A1"] = "CATTY"
        report = verify_fill(self.puzzle, self.graph, gold)
        self.assertIn("length", {i.kind for i in report.issues})

    def test_crossing_conflict(self):
        gold = {s.id: s.gold for s in self.puzzle.slots}
        gold["D1"] = "XYZ"
        report = verify_fill(self.puzzle, self.graph, gold)
        self.assertIn("crossing", {i.kind for i in report.issues})

    def test_implied_non_word_from_crossings_is_flagged(self):
        """LINE/FEST/OATEN are all words; the down they spell is not."""
        from crossword.xd import load_xd

        path = MINI07
        if not os.path.isfile(path):
            self.skipTest("corpus not generated")
        puzzle = load_xd(path)
        graph = SlotGraph(puzzle)
        gold = {s.id: WILDCARD for s in puzzle.slots}
        gold["A1"] = "LINE"
        gold["A5"] = "FEST"
        gold["A6"] = "OATEN"
        report = verify_fill(puzzle, graph, gold)
        kinds = {i.kind for i in report.issues}
        self.assertIn("word", kinds)
        self.assertTrue(any("LFA" in i.detail for i in report.issues))

    def test_dictionary_mismatch_is_flagged_only_for_proposed_word(self):
        gold = {s.id: s.gold for s in self.puzzle.slots}
        gold["A1"] = "DOG"
        report = verify_fill(
            self.puzzle,
            self.graph,
            gold,
            definitions={"DOG": "A barking animal", "CAT": "Feline"},
        )
        self.assertIn("definition", {i.kind for i in report.issues})
        self.assertTrue(any("DOG" in i.detail for i in report.issues))
        self.assertFalse(any("CAT" in i.detail for i in report.issues))


class TestGeneratedSolveDoesNotConsultBank(unittest.TestCase):
    def test_model_guess_is_not_replaced_by_bank_lookup(self):
        """The constructor's word is AROW, but the solver must not retrieve it
        from the bank. LINE is a reasonable English guess and it has to stand
        unless the model itself proposes something else."""
        if not os.path.isfile(MINI07):
            self.skipTest("corpus not generated")
        puzzle = load_xd(MINI07)
        client = ScriptedClient(
            [
                '{"items": [{"id": "A1", "candidates": '
                '[{"answer": "LINE", "confidence": 0.9}]}]}'
            ]
        )
        result = Solver(
            client,
            SolverConfig(model="x", max_workers=1, max_rounds=1, check_definitions=False),
        ).solve(puzzle)
        self.assertGreater(result.usage.calls, 0)
        self.assertEqual(result.assignment.get("A1"), "LINE")
