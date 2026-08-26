import json
import unittest

from crossword.schemas import (
    FREE_TEXT,
    JSON_OBJECT,
    LADDER,
    STRICT,
    Candidate,
    candidates_schema,
    extract_json,
    merge_candidates,
    parse_candidates,
    response_format_for,
    strip_reasoning,
)

GOOD = json.dumps(
    {
        "items": [
            {
                "id": "A1",
                "candidates": [
                    {"answer": "CAT", "confidence": 0.9, "kind": "definition"},
                    {"answer": "COT", "confidence": 0.4},
                ],
            },
            {"id": "D1", "candidates": [{"answer": "CAB", "confidence": 0.7}]},
        ]
    }
)

EXPECTED = {"A1": 3, "D1": 3}


class TestStripReasoning(unittest.TestCase):
    def test_removes_think_block(self):
        text = "<think>Let me consider ORE vs ORT.</think>" + GOOD
        self.assertEqual(strip_reasoning(text), GOOD)

    def test_removes_unclosed_think_block(self):
        """A truncated reasoning block leaves a stray closing tag."""
        text = "reasoning spilled here</think>" + GOOD
        self.assertEqual(strip_reasoning(text), GOOD)

    def test_unwraps_code_fence(self):
        self.assertEqual(strip_reasoning(f"```json\n{GOOD}\n```"), GOOD)

    def test_unwraps_bare_fence(self):
        self.assertEqual(strip_reasoning(f"```\n{GOOD}\n```"), GOOD)

    def test_leaves_plain_json_alone(self):
        self.assertEqual(strip_reasoning(GOOD), GOOD)


class TestExtractJson(unittest.TestCase):
    def test_plain(self):
        self.assertEqual(extract_json('{"a": 1}'), {"a": 1})

    def test_leading_prose(self):
        self.assertEqual(extract_json('Here you go:\n{"a": 1}'), {"a": 1})

    def test_trailing_commas(self):
        self.assertEqual(extract_json('{"a": 1,}'), {"a": 1})

    def test_trailing_prose_after_object(self):
        self.assertEqual(extract_json('{"a": 1}\nHope that helps!'), {"a": 1})

    def test_braces_inside_strings_do_not_confuse_the_scan(self):
        text = 'note:\n{"clue": "a { brace", "b": 2}\ntrailing'
        self.assertEqual(extract_json(text), {"clue": "a { brace", "b": 2})

    def test_escaped_quote_inside_string(self):
        text = 'x {"clue": "say \\"hi\\"", "n": 1} y'
        self.assertEqual(extract_json(text), {"clue": 'say "hi"', "n": 1})

    def test_returns_none_when_absent(self):
        self.assertIsNone(extract_json("no json here at all"))


class TestParseCandidates(unittest.TestCase):
    def test_happy_path(self):
        candidates, warnings = parse_candidates(GOOD, expected=EXPECTED)
        self.assertEqual(warnings, [])
        answers = {(c.slot_id, c.answer) for c in candidates}
        self.assertEqual(answers, {("A1", "CAT"), ("A1", "COT"), ("D1", "CAB")})

    def test_survives_reasoning_and_fence(self):
        messy = f"<think>hmm</think>\n```json\n{GOOD}\n```\nDone."
        candidates, _ = parse_candidates(messy, expected=EXPECTED)
        self.assertEqual(len(candidates), 3)

    def test_drops_wrong_length(self):
        text = json.dumps(
            {"items": [{"id": "A1", "candidates": [{"answer": "TIGER", "confidence": 1}]}]}
        )
        candidates, warnings = parse_candidates(text, expected=EXPECTED)
        self.assertEqual([c for c in candidates if c.slot_id == "A1"], [])
        self.assertTrue(any("does not fit" in w for w in warnings))

    def test_drops_pattern_violation(self):
        candidates, _ = parse_candidates(
            GOOD, expected=EXPECTED, patterns={"A1": "C?T", "D1": "???"}
        )
        a1 = {c.answer for c in candidates if c.slot_id == "A1"}
        self.assertEqual(a1, {"CAT", "COT"})
        candidates, _ = parse_candidates(
            GOOD, expected=EXPECTED, patterns={"A1": "CA?", "D1": "???"}
        )
        a1 = {c.answer for c in candidates if c.slot_id == "A1"}
        self.assertEqual(a1, {"CAT"})

    def test_normalizes_case_and_punctuation(self):
        text = json.dumps(
            {"items": [{"id": "a1", "candidates": [{"answer": "c-a-t", "confidence": 0.8}]}]}
        )
        candidates, _ = parse_candidates(text, expected=EXPECTED)
        self.assertEqual(candidates[0].answer, "CAT")

    def test_clamps_confidence(self):
        text = json.dumps(
            {
                "items": [
                    {"id": "A1", "candidates": [{"answer": "CAT", "confidence": 1.0}]},
                    {"id": "D1", "candidates": [{"answer": "CAB", "confidence": 0.0}]},
                ]
            }
        )
        candidates, _ = parse_candidates(text, expected=EXPECTED)
        by_slot = {c.slot_id: c.confidence for c in candidates}
        # Never exactly 0 or 1: the search works in log space and a model
        # asserting certainty must not be able to veto backtracking.
        self.assertLess(by_slot["A1"], 1.0)
        self.assertGreater(by_slot["D1"], 0.0)

    def test_bad_confidence_falls_back(self):
        text = json.dumps(
            {"items": [{"id": "A1", "candidates": [{"answer": "CAT", "confidence": "high"}]}]}
        )
        candidates, _ = parse_candidates(text, expected=EXPECTED)
        self.assertAlmostEqual(candidates[0].confidence, 0.5)

    def test_flat_mapping_fallback(self):
        candidates, warnings = parse_candidates(
            json.dumps({"A1": "CAT", "D1": "CAB"}), expected=EXPECTED
        )
        self.assertEqual({c.answer for c in candidates}, {"CAT", "CAB"})
        self.assertTrue(any("flat mapping" in w for w in warnings))

    def test_bare_string_candidate(self):
        text = json.dumps({"items": [{"id": "A1", "candidates": ["CAT"]}]})
        candidates, _ = parse_candidates(text, expected=EXPECTED)
        self.assertEqual(candidates[0].answer, "CAT")

    def test_unknown_slot_is_dropped(self):
        text = json.dumps({"items": [{"id": "A99", "candidates": ["XYZ"]}]})
        candidates, warnings = parse_candidates(text, expected=EXPECTED)
        self.assertEqual(candidates, [])
        self.assertTrue(any("A99" in w for w in warnings))

    def test_garbage_reports_cleanly(self):
        candidates, warnings = parse_candidates("total nonsense", expected=EXPECTED)
        self.assertEqual(candidates, [])
        self.assertIn("no JSON object found in response", warnings)

    def test_duplicate_answers_keep_highest_confidence(self):
        text = json.dumps(
            {
                "items": [
                    {
                        "id": "A1",
                        "candidates": [
                            {"answer": "CAT", "confidence": 0.3},
                            {"answer": "CAT", "confidence": 0.8},
                        ],
                    }
                ]
            }
        )
        candidates, _ = parse_candidates(text, expected=EXPECTED)
        a1 = [c for c in candidates if c.slot_id == "A1"]
        self.assertEqual(len(a1), 1)
        self.assertAlmostEqual(a1[0].confidence, 0.8)


class TestMergeCandidates(unittest.TestCase):
    def test_agreement_raises_confidence(self):
        one = [Candidate("A1", "CAT", 0.6)]
        two = [Candidate("A1", "CAT", 0.6)]
        merged = merge_candidates([one, two])
        self.assertEqual(len(merged), 1)
        # Noisy-OR: two independent 0.6 confirmations should exceed 0.6.
        self.assertGreater(merged[0].confidence, 0.6)
        self.assertAlmostEqual(merged[0].confidence, 0.84, places=2)
        self.assertEqual(merged[0].sources, 2)

    def test_disagreement_keeps_both(self):
        merged = merge_candidates(
            [[Candidate("A1", "CAT", 0.6)], [Candidate("A1", "COT", 0.5)]]
        )
        self.assertEqual({c.answer for c in merged}, {"CAT", "COT"})
        self.assertTrue(all(c.sources == 1 for c in merged))

    def test_confidence_never_reaches_one(self):
        groups = [[Candidate("A1", "CAT", 0.99)] for _ in range(5)]
        merged = merge_candidates(groups)
        self.assertLess(merged[0].confidence, 1.0)


class TestSchemaLadder(unittest.TestCase):
    def test_strict_schema_is_constrained(self):
        schema = candidates_schema(strict=True, constrained=True)
        self.assertTrue(schema["json_schema"]["strict"])
        body = schema["json_schema"]["schema"]
        answer = body["properties"]["items"]["items"]["properties"]["candidates"]
        self.assertEqual(answer["maxItems"], 5)
        self.assertIn("pattern", answer["items"]["properties"]["answer"])

    def test_ladder_weakens_monotonically(self):
        formats = [response_format_for(rung) for rung in LADDER]
        self.assertIsNotNone(formats[0])
        self.assertEqual(response_format_for(JSON_OBJECT), {"type": "json_object"})
        self.assertIsNone(response_format_for(FREE_TEXT))

    def test_every_rung_is_handled(self):
        for rung in LADDER:
            response_format_for(rung)  # must not raise
        self.assertEqual(LADDER[0], STRICT)


if __name__ == "__main__":
    unittest.main()
