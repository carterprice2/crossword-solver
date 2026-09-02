import unittest

from crossword.agent.candidates import request_with_ladder
from crossword.client import Completion
from crossword.schemas import STRICT, looks_like_candidates

JSON = '{"items":[{"id":"A1","candidates":[{"answer":"CAT","confidence":0.9}]}]}'
THINKING = "Let me reason about birds and genera without emitting any object."


class QueueClient:
    def __init__(self, replies: list[tuple[str, int]]):
        self.replies = list(replies)
        self.calls: list[dict] = []

    def complete(self, **kwargs) -> Completion:
        self.calls.append(kwargs)
        text, tokens = self.replies.pop(0)
        return Completion(
            text=text,
            model=kwargs["model"],
            completion_tokens=tokens,
        )


def _messages(*, schema_in_prompt: bool):
    return [{"role": "user", "content": "clues"}]


class TestLooksLikeCandidates(unittest.TestCase):
    def test_items_payload(self):
        self.assertTrue(looks_like_candidates(JSON))

    def test_thinking_prose_is_not_candidates(self):
        self.assertFalse(looks_like_candidates(THINKING))


class TestRequestWithLadder(unittest.TestCase):
    def test_skips_rung_that_returns_no_json(self):
        client = QueueClient(
            [(THINKING, 80), (THINKING, 80), (THINKING, 80), (JSON, 40)]
        )
        completion, rung = request_with_ladder(
            client,
            model="m",
            build_messages=_messages,
            temperature=0.3,
            max_tokens=2048,
            seed=0,
        )
        self.assertEqual(completion.text, JSON)
        self.assertNotEqual(rung, STRICT)
        self.assertGreaterEqual(len(client.calls), 2)

    def test_retries_truncated_thinking_on_same_rung(self):
        client = QueueClient([(THINKING, 2048), (JSON, 200)])
        completion, rung = request_with_ladder(
            client,
            model="m",
            build_messages=_messages,
            temperature=0.3,
            max_tokens=2048,
            seed=0,
        )
        self.assertEqual(completion.text, JSON)
        self.assertEqual(rung, STRICT)
        self.assertEqual(client.calls[0]["max_tokens"], 2048)
        self.assertGreater(client.calls[1]["max_tokens"], 2048)
        self.assertEqual(
            client.calls[0]["response_format"],
            client.calls[1]["response_format"],
        )
