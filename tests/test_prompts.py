import unittest

from crossword.agent.prompts import SYSTEM, first_pass_messages, star_repair_messages
from crossword.agent.constraints import SlotGraph
from crossword.xd import parse_xd

MINI = """\
Title: Prompt Mini


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


class TestPrompts(unittest.TestCase):
    def test_system_mentions_two_word_mashups(self):
        self.assertIn("mashups of two words", SYSTEM)
        self.assertIn("ICEAGE, not ICE AGE", SYSTEM)

    def test_first_pass_uses_system_prompt(self):
        puzzle = parse_xd(MINI)
        graph = SlotGraph(puzzle)
        messages = first_pass_messages(puzzle, graph, ["A1"], {})
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("mashups of two words", messages[0]["content"])

    def test_star_repair_allows_mashups(self):
        puzzle = parse_xd(MINI)
        graph = SlotGraph(puzzle)
        messages = star_repair_messages(
            puzzle,
            graph,
            ["A1", "D1"],
            ("A1", "D1"),
            {},
            {},
            {},
        )
        self.assertIn("mashup of two words", messages[1]["content"])
