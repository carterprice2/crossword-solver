import unittest

from crossword.client import Completion
from crossword.eval.smoke import SMOKE_SLOTS, smoke_catalog, smoke_one
from crossword.schemas import STRICT


JSON = (
    '{"items":['
    '{"id":"A1","candidates":[{"answer":"CAT","confidence":0.9}]},'
    '{"id":"A4","candidates":[{"answer":"ARE","confidence":0.8}]}'
    "]}"
)


class FakeClient:
    def __init__(self, text=JSON, error=None):
        self.text = text
        self.error = error
        self.calls = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return Completion(text=self.text, model=kwargs["model"], completion_tokens=20)


class TestSmokeOne(unittest.TestCase):
    def test_ok_when_json_parses(self):
        result = smoke_one(FakeClient(), "m")
        self.assertTrue(result.ok)
        self.assertGreaterEqual(result.n_candidates, 2)
        self.assertIn("A1", result.slots)
        self.assertEqual(result.rung, STRICT)

    def test_fail_when_prose(self):
        result = smoke_one(FakeClient(text="I am thinking about cats."), "m")
        self.assertFalse(result.ok)
        self.assertEqual(result.n_candidates, 0)

    def test_records_client_error(self):
        result = smoke_one(FakeClient(error=RuntimeError("down")), "m")
        self.assertFalse(result.ok)
        self.assertIn("down", result.error)


class TestSmokeCatalog(unittest.TestCase):
    def test_skips_models_not_on_the_account(self):
        results = smoke_catalog(
            FakeClient(),
            models=["good", "missing"],
            available=["good"],
        )
        by_name = {r.model: r for r in results}
        self.assertTrue(by_name["good"].ok)
        self.assertTrue(by_name["missing"].skipped)
        self.assertFalse(by_name["missing"].ok)

    def test_smoke_slots_are_tiny(self):
        self.assertEqual(len(SMOKE_SLOTS), 2)
