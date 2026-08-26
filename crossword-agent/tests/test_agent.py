import unittest

from crossword.agent.candidates import batch_by_locality
from crossword.agent.constraints import (
    DEFAULT_UNKNOWN_MASS,
    WILDCARD,
    SlotGraph,
    build_domains,
    cells_from_assignment,
    intersection_consistency,
    letter_marginals,
    merge_domains,
    pattern_filter,
    soft_ac3,
)
from crossword.agent.search import solve as search_solve
from crossword.agent.solver import Solver, SolverConfig
from crossword.client import OracleClient, OracleConfig
from crossword.eval.metrics import score_solution
from crossword.schemas import Candidate
from crossword.xd import parse_xd

MINI = """\
Title: Agent Mini


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


def mini():
    return parse_xd(MINI)


def gold_candidates(puzzle, confidence=0.9):
    return [Candidate(s.id, s.gold, confidence) for s in puzzle.slots]


class TestSlotGraph(unittest.TestCase):
    def setUp(self):
        self.puzzle = mini()
        self.graph = SlotGraph(self.puzzle)

    def test_crossings_cover_every_cell(self):
        self.assertEqual(len(self.graph.crossings), 9)

    def test_neighbors_are_perpendicular(self):
        self.assertEqual(sorted(self.graph.neighbors("A1")), ["D1", "D2", "D3"])
        self.assertEqual(sorted(self.graph.neighbors("D1")), ["A1", "A4", "A5"])

    def test_pattern_reflects_known_cells(self):
        self.assertEqual(self.graph.pattern("A1", {}), "???")
        self.assertEqual(self.graph.pattern("A1", {(0, 1): "A"}), "?A?")


class TestDomains(unittest.TestCase):
    def setUp(self):
        self.puzzle = mini()
        self.graph = SlotGraph(self.puzzle)

    def test_build_sorts_by_confidence(self):
        domains = build_domains(
            self.graph,
            [Candidate("A1", "COT", 0.3), Candidate("A1", "CAT", 0.9)],
        )
        self.assertEqual(domains["A1"].answers(), ["CAT", "COT"])
        self.assertEqual(domains["A1"].best().answer, "CAT")

    def test_every_slot_gets_a_domain(self):
        domains = build_domains(self.graph, [])
        self.assertEqual(set(domains), set(self.graph.slot_ids))
        self.assertTrue(all(d.is_empty() for d in domains.values()))

    def test_merge_keeps_higher_confidence(self):
        a = build_domains(self.graph, [Candidate("A1", "CAT", 0.4)])
        b = build_domains(self.graph, [Candidate("A1", "CAT", 0.8), Candidate("A1", "COT", 0.2)])
        merged = merge_domains(a, b)
        by_answer = {c.answer: c.confidence for c in merged["A1"].candidates}
        self.assertAlmostEqual(by_answer["CAT"], 0.8)
        self.assertIn("COT", by_answer)

    def test_pattern_filter_drops_contradictions(self):
        domains = build_domains(
            self.graph, [Candidate("A1", "CAT", 0.9), Candidate("A1", "BAT", 0.8)]
        )
        filtered = pattern_filter(domains, self.graph, {(0, 0): "C"})
        self.assertEqual(filtered["A1"].answers(), ["CAT"])


class TestSoftAC3(unittest.TestCase):
    def setUp(self):
        self.puzzle = mini()
        self.graph = SlotGraph(self.puzzle)

    def test_prunes_unsupported_candidates(self):
        candidates = gold_candidates(self.puzzle)
        # ZZZ agrees with nothing crossing A1.
        candidates.append(Candidate("A1", "ZZZ", 0.5))
        domains = build_domains(self.graph, candidates)
        pruned, conflicts = soft_ac3(domains, self.graph)
        self.assertNotIn("ZZZ", pruned["A1"].answers())
        self.assertEqual(conflicts, [])

    def test_never_empties_a_domain(self):
        """The defining behaviour: LLM candidate lists are incomplete, so a
        prune that would empty a domain must be skipped, not applied."""
        domains = build_domains(
            self.graph,
            [Candidate("A1", "CAT", 0.9), Candidate("D1", "XYZ", 0.9)],
        )
        pruned, conflicts = soft_ac3(domains, self.graph)
        self.assertEqual(pruned["D1"].answers(), ["XYZ"])
        self.assertEqual(pruned["A1"].answers(), ["CAT"])
        self.assertTrue(conflicts)

    def test_conflict_names_both_slots(self):
        domains = build_domains(
            self.graph,
            [Candidate("A1", "CAT", 0.9), Candidate("D1", "XYZ", 0.9)],
        )
        _, conflicts = soft_ac3(domains, self.graph)
        pairs = {frozenset(c.slots) for c in conflicts}
        self.assertIn(frozenset({"A1", "D1"}), pairs)

    def test_gold_candidates_survive_untouched(self):
        domains = build_domains(self.graph, gold_candidates(self.puzzle))
        pruned, conflicts = soft_ac3(domains, self.graph)
        self.assertEqual(conflicts, [])
        for slot in self.puzzle.slots:
            self.assertEqual(pruned[slot.id].answers(), [slot.gold])


class TestSearch(unittest.TestCase):
    def setUp(self):
        self.puzzle = mini()
        self.graph = SlotGraph(self.puzzle)

    def test_recovers_the_exact_grid(self):
        domains = build_domains(self.graph, gold_candidates(self.puzzle))
        result = search_solve(domains, self.graph, seed=0)
        self.assertTrue(result.complete)
        self.assertEqual(
            result.filled, {s.id: s.gold for s in self.puzzle.slots}
        )

    def test_prefers_consistency_over_a_confident_wrong_answer(self):
        """A high-confidence answer that breaks three crossings must lose to
        the consistent set -- this is the whole point of the search."""
        candidates = gold_candidates(self.puzzle, confidence=0.7)
        candidates.append(Candidate("A1", "ZZZ", 0.95))
        domains = build_domains(self.graph, candidates)
        result = search_solve(domains, self.graph, seed=0)
        self.assertEqual(result.assignment["A1"], "CAT")

    def test_declines_rather_than_guessing_wrong(self):
        """With the true answer absent, the search prefers the wildcard to a
        low-confidence answer.

        This asserts the scoring behaves as designed, not that the behaviour
        helps: the ablation (arm a6) found declining worth roughly nothing end
        to end, because endgame_fill fills the slot anyway. See REPORT.md
        section 6.
        """
        candidates = [
            c for c in gold_candidates(self.puzzle) if c.slot_id != "A1"
        ]
        candidates.append(Candidate("A1", "ZZZ", 0.2))
        domains = build_domains(self.graph, candidates)
        result = search_solve(domains, self.graph, seed=0)
        self.assertNotEqual(result.assignment["A1"], "ZZZ")
        self.assertIn(result.assignment["A1"], {WILDCARD, "CAT"})

    def test_no_candidates_declines_everything(self):
        domains = build_domains(self.graph, [])
        result = search_solve(domains, self.graph, seed=0)
        self.assertEqual(result.open_slots, sorted(self.graph.slot_ids))

    def test_deterministic_for_a_seed(self):
        domains = build_domains(self.graph, gold_candidates(self.puzzle))
        a = search_solve(domains, self.graph, seed=5)
        b = search_solve(domains, self.graph, seed=5)
        self.assertEqual(a.assignment, b.assignment)

    def test_respects_the_node_budget(self):
        domains = build_domains(self.graph, gold_candidates(self.puzzle))
        result = search_solve(domains, self.graph, seed=0, max_nodes=5, restarts=1)
        self.assertLessEqual(result.nodes, 20)

    def test_agreement_bonus_breaks_a_tie(self):
        candidates = [
            c for c in gold_candidates(self.puzzle, 0.5) if c.slot_id != "A1"
        ]
        candidates.append(Candidate("A1", "CAT", 0.5, sources=2))
        candidates.append(Candidate("A1", "CAB", 0.5, sources=1))
        domains = build_domains(self.graph, candidates)
        self.assertEqual(search_solve(domains, self.graph, seed=0).assignment["A1"], "CAT")

    def test_refuses_line_fest_when_they_spell_lfa(self):
        import os

        from crossword.xd import load_xd

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(root, "corpus", "mini", "mini-07-00-0.xd")
        if not os.path.isfile(path):
            self.skipTest("corpus not generated")
        puzzle = load_xd(path)
        graph = SlotGraph(puzzle)
        candidates = [
            c for c in gold_candidates(puzzle, 0.8) if c.slot_id not in ("A1", "A5")
        ]
        candidates.append(Candidate("A1", "LINE", 0.95))
        candidates.append(Candidate("A5", "FEST", 0.95))
        result = search_solve(build_domains(graph, candidates), graph, seed=0)
        combo = (
            result.assignment.get("A1"),
            result.assignment.get("A5"),
            result.assignment.get("A6"),
        )
        self.assertNotEqual(combo, ("LINE", "FEST", "OATEN"))
        spelled = puzzle.slot("D1").spell(
            cells_from_assignment(result.assignment, graph)
        )
        if spelled:
            self.assertNotEqual(spelled, "LFA")


class TestHelpers(unittest.TestCase):
    def setUp(self):
        self.puzzle = mini()
        self.graph = SlotGraph(self.puzzle)

    def test_intersection_consistency(self):
        gold = {s.id: s.gold for s in self.puzzle.slots}
        self.assertEqual(intersection_consistency(gold, self.graph), (9, 9))
        broken = dict(gold, A1="ZZZ")
        agree, total = intersection_consistency(broken, self.graph)
        self.assertEqual(total, 9)
        self.assertLess(agree, 9)

    def test_cells_from_assignment_skips_wildcards(self):
        cells = cells_from_assignment({"A1": "CAT", "A4": WILDCARD}, self.graph)
        self.assertEqual(cells, {(0, 0): "C", (0, 1): "A", (0, 2): "T"})

    def test_letter_marginals_peak_on_agreement(self):
        domains = build_domains(self.graph, gold_candidates(self.puzzle))
        marginals = letter_marginals(domains, self.graph)
        self.assertAlmostEqual(marginals[(0, 0)]["C"], 1.0)
        self.assertAlmostEqual(sum(marginals[(0, 0)].values()), 1.0)


class TestBatching(unittest.TestCase):
    def test_batches_hold_crossing_slots(self):
        puzzle = mini()
        graph = SlotGraph(puzzle)
        batches = batch_by_locality(graph, graph.slot_ids, size=3)
        self.assertEqual(sum(len(b) for b in batches), len(graph.slot_ids))
        self.assertEqual(
            sorted(s for b in batches for s in b), sorted(graph.slot_ids)
        )
        # Every batch after the first element should contain a crossing slot.
        for batch in batches:
            if len(batch) > 1:
                neighbors = set(graph.neighbors(batch[0]))
                self.assertTrue(neighbors & set(batch[1:]))

    def test_deterministic(self):
        graph = SlotGraph(mini())
        self.assertEqual(
            batch_by_locality(graph, graph.slot_ids, 3),
            batch_by_locality(graph, graph.slot_ids, 3),
        )

    def test_size_zero_means_one_batch(self):
        graph = SlotGraph(mini())
        self.assertEqual(len(batch_by_locality(graph, graph.slot_ids, 0)), 1)


class TestSolverEndToEnd(unittest.TestCase):
    """The whole loop, offline, against synthetic candidate lists."""

    def solve_with(self, puzzle, **oracle):
        gold = {s.id: s.gold for s in puzzle.slots}
        client = OracleClient(gold, OracleConfig(seed=3, **oracle))
        config = SolverConfig(model="oracle", max_workers=1)
        return Solver(client, config).solve(puzzle)

    def test_perfect_candidates_solve_exactly(self):
        puzzle = mini()
        result = self.solve_with(puzzle, recall=1.0, top1_error=0.0, conf_noise=0.0)
        self.assertTrue(
            score_solution(puzzle, result.solution, assignment=result.assignment).exact
        )

    def test_recovers_from_missing_answers(self):
        puzzle = mini()
        result = self.solve_with(puzzle, recall=0.6, top1_error=0.4)
        scores = score_solution(puzzle, result.solution)
        # Crossing constraints should recover most of what round 0 missed.
        self.assertGreater(scores.lcr, 0.6)

    def test_repair_rounds_actually_run(self):
        puzzle = mini()
        gold = {s.id: s.gold for s in puzzle.slots}
        client = OracleClient(gold, OracleConfig(recall=0.5, top1_error=0.5, seed=1))
        config = SolverConfig(model="oracle", max_rounds=3, max_workers=1)
        result = Solver(client, config).solve(puzzle)
        self.assertGreaterEqual(result.rounds, 1)
        self.assertLessEqual(result.rounds, 3)

    def test_prefill_is_respected(self):
        puzzle = mini()
        gold = puzzle.gold_solution()
        client = OracleClient(
            {s.id: s.gold for s in puzzle.slots}, OracleConfig(recall=0.0, seed=1)
        )
        config = SolverConfig(model="oracle", max_workers=1)
        prefill = {(0, 0): gold[(0, 0)], (1, 1): gold[(1, 1)]}
        result = Solver(client, config).solve(puzzle, prefill=prefill)
        for cell, letter in prefill.items():
            self.assertEqual(result.solution.get(cell), letter)

    def test_no_candidates_does_not_invent_letters(self):
        """Declined slots stay blank. Filling them with 'E' or letter scraps
        produced non-words like LFA; a blank is the honest result."""
        puzzle = mini()
        client = OracleClient({}, OracleConfig(recall=0.0))
        config = SolverConfig(model="oracle", max_workers=1)
        result = Solver(client, config).solve(puzzle)
        from crossword.lexicon import is_valid_entry

        for slot in puzzle.slots:
            spelled = slot.spell(result.solution)
            if spelled:
                self.assertTrue(is_valid_entry(spelled), spelled)

    def test_usage_is_tracked(self):
        result = self.solve_with(mini(), recall=0.8)
        self.assertGreater(result.usage.calls, 0)
        self.assertGreater(result.usage.total_tokens, 0)


if __name__ == "__main__":
    unittest.main()
