import os
import unittest

from crossword.agent.constraints import (
    WILDCARD,
    ConflictSite,
    Crossing,
    SlotGraph,
    build_domains,
)
from crossword.agent.star import (
    Star,
    apply_star,
    collect_stars,
    fallback_star,
    solve_star,
    star_complete,
    star_slots,
)
from crossword.schemas import Candidate
from crossword.xd import load_xd, parse_xd

MINI = """\
Title: Star Mini


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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MINI07 = os.path.join(ROOT, "corpus", "mini", "mini-07-00-0.xd")


def mini():
    return parse_xd(MINI)


def gold_candidates(puzzle, confidence=0.9):
    return [Candidate(s.id, s.gold, confidence) for s in puzzle.slots]


class TestStarShape(unittest.TestCase):
    def test_star_is_the_two_hubs_plus_everything_they_touch(self):
        graph = SlotGraph(mini())
        slots = star_slots(graph, "A1", "D1")
        self.assertEqual(slots, ("A1", "A4", "A5", "D1", "D2", "D3"))

    def test_collect_stars_from_a_conflict(self):
        graph = SlotGraph(mini())
        site = ConflictSite(
            Crossing("A1", 0, "D1", 0, (0, 0)),
            "clash",
        )
        stars = collect_stars(graph, [site], ["A1", "D1"])
        self.assertEqual(len(stars), 1)
        self.assertEqual(stars[0].hubs, ("A1", "D1"))
        self.assertIn("A1", stars[0].slots)
        self.assertIn("D1", stars[0].slots)


class TestStarSolve(unittest.TestCase):
    def test_existing_candidates_mesh_without_a_model_call(self):
        puzzle = mini()
        graph = SlotGraph(puzzle)
        domains = build_domains(graph, gold_candidates(puzzle))
        star = Star(slots=tuple(graph.slot_ids), hubs=("A1", "D1"))
        result = solve_star(domains, graph, star, {}, {}, seed=0)
        self.assertTrue(star_complete(result, star))
        self.assertEqual(result.assignment["A1"], "CAT")
        self.assertEqual(result.assignment["D1"], "CAB")

    def test_line_fest_cannot_complete_the_star(self):
        if not os.path.isfile(MINI07):
            self.skipTest("corpus not generated")
        puzzle = load_xd(MINI07)
        graph = SlotGraph(puzzle)
        candidates = [
            c for c in gold_candidates(puzzle, 0.8) if c.slot_id not in ("A1", "A5")
        ]
        candidates.append(Candidate("A1", "LINE", 0.95))
        candidates.append(Candidate("A5", "FEST", 0.95))
        domains = build_domains(graph, candidates)
        star = Star(slots=star_slots(graph, "A1", "D1"), hubs=("A1", "D1"))
        result = solve_star(domains, graph, star, {}, {}, seed=0)
        combo = (
            result.assignment.get("A1"),
            result.assignment.get("A5"),
            result.assignment.get("A6"),
        )
        self.assertNotEqual(combo, ("LINE", "FEST", "OATEN"))


class TestStarFallback(unittest.TestCase):
    def test_keeps_the_higher_confidence_hub(self):
        puzzle = mini()
        graph = SlotGraph(puzzle)
        domains = build_domains(
            graph,
            [
                Candidate("A1", "LINE", 0.9),
                Candidate("D1", "CAB", 0.4),
            ],
        )
        star = Star(slots=("A1", "D1"), hubs=("A1", "D1"))
        assignment = {"A1": "LINE", "D1": "CAB"}
        confidence = {"A1": 0.9, "D1": 0.4}
        rejected: dict[str, set[str]] = {}
        reason = fallback_star(star, assignment, confidence, domains, rejected)
        self.assertEqual(assignment["A1"], "LINE")
        self.assertEqual(assignment["D1"], WILDCARD)
        self.assertIn("CAB", rejected["D1"])
        self.assertIn("keep A1=LINE", reason)

    def test_apply_boosts_the_chosen_answers(self):
        puzzle = mini()
        graph = SlotGraph(puzzle)
        domains = build_domains(graph, [Candidate("A1", "CAT", 0.5)])
        star = Star(slots=("A1",), hubs=("A1", "A1"))
        from crossword.agent.search import SearchResult

        result = SearchResult(assignment={"A1": "CAT"}, score=0.0, confidence={"A1": 0.5})
        assignment: dict[str, str] = {}
        confidence: dict[str, float] = {}
        apply_star(result, star, assignment, confidence, domains)
        self.assertEqual(assignment["A1"], "CAT")
        self.assertGreaterEqual(domains["A1"].best().confidence, 0.85)
