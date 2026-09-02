import io
import json
import os
import tempfile
import unittest
import urllib.error

from crossword.client import (
    CacheMiss,
    Completion,
    ModelError,
    NebiusClient,
    OracleClient,
    OracleConfig,
    RecordingClient,
    ReplayClient,
    SchemaRejected,
    ScriptedClient,
    Usage,
    effective_recall,
    load_env_file,
    request_key,
)


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def chat_payload(text="hi", model="m", prompt=11, completion=7):
    return json.dumps(
        {
            "model": model,
            "choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": completion},
        }
    ).encode()


class FakeOpener:
    """Stands in for urllib's opener so no socket is ever created."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []

    def open(self, request, timeout=None):
        self.requests.append(request)
        item = self.script.pop(0) if self.script else chat_payload()
        if isinstance(item, Exception):
            raise item
        return FakeResponse(item)


def http_error(code, body=b"boom", headers=None):
    return urllib.error.HTTPError(
        "https://x/", code, "err", headers or {}, io.BytesIO(body)
    )


class TestNebiusClient(unittest.TestCase):
    def build(self, script, **kwargs):
        return NebiusClient(
            api_key="test-key",
            opener=FakeOpener(script),
            sleep=lambda _: None,
            **kwargs,
        )

    def test_requires_api_key(self):
        saved = os.environ.pop("NEBIUS_API_KEY", None)
        try:
            with self.assertRaises(ModelError) as ctx:
                NebiusClient(opener=FakeOpener([]))
            self.assertIn("NEBIUS_API_KEY", str(ctx.exception))
        finally:
            if saved is not None:
                os.environ["NEBIUS_API_KEY"] = saved

    def test_successful_completion_records_usage(self):
        client = self.build([chat_payload("answer", "Qwen/X", 30, 12)])
        result = client.complete(model="Qwen/X", messages=[{"role": "user", "content": "q"}])
        self.assertEqual(result.text, "answer")
        self.assertEqual(result.prompt_tokens, 30)
        self.assertEqual(result.completion_tokens, 12)
        self.assertEqual(result.total_tokens, 42)
        self.assertEqual(client.calls, 1)

    def test_request_targets_chat_completions(self):
        opener = FakeOpener([chat_payload()])
        client = NebiusClient(api_key="k", opener=opener, sleep=lambda _: None)
        client.complete(model="m", messages=[{"role": "user", "content": "q"}])
        request = opener.requests[0]
        self.assertTrue(request.full_url.endswith("/v1/chat/completions"))
        self.assertEqual(request.headers["Authorization"], "Bearer k")
        body = json.loads(request.data)
        self.assertEqual(body["model"], "m")

    def test_response_format_is_forwarded(self):
        opener = FakeOpener([chat_payload()])
        client = NebiusClient(api_key="k", opener=opener, sleep=lambda _: None)
        fmt = {"type": "json_object"}
        client.complete(model="m", messages=[], response_format=fmt)
        self.assertEqual(json.loads(opener.requests[0].data)["response_format"], fmt)

    def test_400_becomes_schema_rejected(self):
        """400 is how guided decoding refuses a schema; the caller degrades."""
        client = self.build([http_error(400, b"unsupported schema")])
        with self.assertRaises(SchemaRejected):
            client.complete(model="m", messages=[])

    def test_401_is_not_retried(self):
        opener = FakeOpener([http_error(401), chat_payload()])
        client = NebiusClient(api_key="k", opener=opener, sleep=lambda _: None)
        with self.assertRaises(ModelError) as ctx:
            client.complete(model="m", messages=[])
        self.assertIn("NEBIUS_API_KEY", str(ctx.exception))
        self.assertEqual(len(opener.requests), 1)

    def test_429_is_retried_then_succeeds(self):
        opener = FakeOpener([http_error(429), http_error(429), chat_payload("ok")])
        client = NebiusClient(api_key="k", opener=opener, sleep=lambda _: None)
        self.assertEqual(client.complete(model="m", messages=[]).text, "ok")
        self.assertEqual(len(opener.requests), 3)

    def test_retry_after_header_is_honored(self):
        waits = []
        opener = FakeOpener([http_error(429, headers={"Retry-After": "7"}), chat_payload()])
        client = NebiusClient(
            api_key="k", opener=opener, sleep=lambda s: waits.append(s)
        )
        client.complete(model="m", messages=[])
        self.assertTrue(waits and waits[0] >= 4.9, waits)

    def test_500_is_retried(self):
        opener = FakeOpener([http_error(503), chat_payload("ok")])
        client = NebiusClient(api_key="k", opener=opener, sleep=lambda _: None)
        self.assertEqual(client.complete(model="m", messages=[]).text, "ok")

    def test_gives_up_after_max_retries(self):
        opener = FakeOpener([http_error(500)] * 10)
        client = NebiusClient(
            api_key="k", opener=opener, sleep=lambda _: None, max_retries=2
        )
        with self.assertRaises(ModelError):
            client.complete(model="m", messages=[])
        self.assertEqual(len(opener.requests), 3)

    def test_timeout_is_retried(self):
        opener = FakeOpener([TimeoutError("slow"), chat_payload("ok")])
        client = NebiusClient(api_key="k", opener=opener, sleep=lambda _: None)
        self.assertEqual(client.complete(model="m", messages=[]).text, "ok")

    def test_empty_choices_raises(self):
        client = self.build([json.dumps({"choices": []}).encode()])
        with self.assertRaises(ModelError):
            client.complete(model="m", messages=[])


class TestRequestKey(unittest.TestCase):
    def test_stable_across_key_order(self):
        a = request_key("m", [{"role": "user", "content": "x"}], None, 0.3, 1)
        b = request_key("m", [{"content": "x", "role": "user"}], None, 0.3, 1)
        self.assertEqual(a, b)

    def test_sensitive_to_every_field(self):
        base = ("m", [{"role": "user", "content": "x"}], None, 0.3, 1)
        key = request_key(*base)
        self.assertNotEqual(key, request_key("other", *base[1:]))
        self.assertNotEqual(key, request_key(base[0], [], *base[2:]))
        self.assertNotEqual(key, request_key(*base[:3], 0.9, base[4]))
        self.assertNotEqual(key, request_key(*base[:4], 2))


class TestRecordReplay(unittest.TestCase):
    def test_round_trip(self):
        inner = ScriptedClient(['{"items": []}'])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "trace.jsonl")
            recorder = RecordingClient(inner, path)
            kwargs = dict(model="m", messages=[{"role": "user", "content": "q"}])
            original = recorder.complete(**kwargs)

            replay = ReplayClient(path)
            again = replay.complete(**kwargs)
            self.assertEqual(again.text, original.text)

    def test_strict_replay_raises_on_miss(self):
        inner = ScriptedClient(["x"])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "trace.jsonl")
            RecordingClient(inner, path).complete(model="m", messages=[])
            replay = ReplayClient(path, strict=True)
            with self.assertRaises(CacheMiss):
                replay.complete(model="different", messages=[])

    def test_non_strict_replay_falls_back_to_order(self):
        inner = ScriptedClient(["first"])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "trace.jsonl")
            RecordingClient(inner, path).complete(model="m", messages=[])
            replay = ReplayClient(path, strict=False)
            self.assertEqual(replay.complete(model="other", messages=[]).text, "first")


class TestOracleClient(unittest.TestCase):
    GOLD = {"A1": "CAT", "D1": "CAB", "A4": "ARE"}

    def prompt(self, slots=("A1", "D1", "A4")):
        items = [
            {"id": s, "clue": "some clue", "len": len(self.GOLD[s])} for s in slots
        ]
        return [{"role": "user", "content": json.dumps({"slots": items})}]

    def test_perfect_oracle_returns_truth_first(self):
        client = OracleClient(
            self.GOLD, OracleConfig(recall=1.0, top1_error=0.0, conf_noise=0.0, width=3)
        )
        payload = json.loads(client.complete(model="oracle", messages=self.prompt()).text)
        by_slot = {item["id"]: item["candidates"] for item in payload["items"]}
        self.assertEqual(set(by_slot), set(self.GOLD))
        for slot, candidates in by_slot.items():
            best = max(candidates, key=lambda c: c["confidence"])
            self.assertEqual(best["answer"], self.GOLD[slot])

    def test_zero_recall_never_includes_truth(self):
        client = OracleClient(self.GOLD, OracleConfig(recall=0.0, width=3))
        payload = json.loads(client.complete(model="oracle", messages=self.prompt()).text)
        for item in payload["items"]:
            answers = {c["answer"] for c in item["candidates"]}
            self.assertNotIn(self.GOLD[item["id"]], answers)

    def test_distractors_have_the_right_length(self):
        """Random strings would be trivially pruned; distractors must be
        plausible enough that crossing constraints do real work."""
        client = OracleClient(self.GOLD, OracleConfig(recall=0.5, width=4, seed=3))
        payload = json.loads(client.complete(model="oracle", messages=self.prompt()).text)
        for item in payload["items"]:
            for candidate in item["candidates"]:
                self.assertEqual(len(candidate["answer"]), len(self.GOLD[item["id"]]))
                self.assertTrue(candidate["answer"].isupper())

    def test_deterministic_for_a_seed(self):
        one = OracleClient(self.GOLD, OracleConfig(seed=11)).complete(
            model="o", messages=self.prompt()
        )
        two = OracleClient(self.GOLD, OracleConfig(seed=11)).complete(
            model="o", messages=self.prompt()
        )
        self.assertEqual(one.text, two.text)

    def test_only_requested_slots_come_back(self):
        client = OracleClient(self.GOLD, OracleConfig(recall=1.0, width=1))
        payload = json.loads(
            client.complete(model="o", messages=self.prompt(("A1",))).text
        )
        self.assertEqual([item["id"] for item in payload["items"]], ["A1"])

    def test_full_pattern_with_zero_recall_includes_gold(self):
        client = OracleClient(
            self.GOLD, OracleConfig(recall=0.0, width=3, pattern_aware=True, seed=1)
        )
        messages = [
            {
                "role": "user",
                "content": json.dumps(
                    {"slots": [{"id": "A1", "clue": "x", "len": 3, "pattern": "CAT"}]}
                ),
            }
        ]
        payload = json.loads(client.complete(model="o", messages=messages).text)
        answers = {c["answer"] for c in payload["items"][0]["candidates"]}
        self.assertIn("CAT", answers)

    def test_independent_mode_ignores_pattern(self):
        client = OracleClient(
            self.GOLD, OracleConfig(recall=0.0, width=3, pattern_aware=False, seed=1)
        )
        messages = [
            {
                "role": "user",
                "content": json.dumps(
                    {"slots": [{"id": "A1", "clue": "x", "len": 3, "pattern": "CAT"}]}
                ),
            }
        ]
        payload = json.loads(client.complete(model="o", messages=messages).text)
        answers = {c["answer"] for c in payload["items"][0]["candidates"]}
        self.assertNotIn("CAT", answers)

    def test_contradicting_pattern_never_includes_gold(self):
        client = OracleClient(
            self.GOLD, OracleConfig(recall=1.0, width=3, pattern_aware=True, seed=1)
        )
        messages = [
            {
                "role": "user",
                "content": json.dumps(
                    {"slots": [{"id": "A1", "clue": "x", "len": 3, "pattern": "DOG"}]}
                ),
            }
        ]
        payload = json.loads(client.complete(model="o", messages=messages).text)
        answers = {c["answer"] for c in payload["items"][0]["candidates"]}
        self.assertNotIn("CAT", answers)

    def test_pattern_aware_distractors_match_pattern(self):
        client = OracleClient(
            self.GOLD, OracleConfig(recall=0.0, width=4, pattern_aware=True, seed=2)
        )
        messages = [
            {
                "role": "user",
                "content": json.dumps(
                    {"slots": [{"id": "A1", "clue": "x", "len": 3, "pattern": "C?T"}]}
                ),
            }
        ]
        payload = json.loads(client.complete(model="o", messages=messages).text)
        for candidate in payload["items"][0]["candidates"]:
            self.assertTrue(candidate["answer"].startswith("C"))
            self.assertTrue(candidate["answer"].endswith("T"))

    def test_rejected_gold_is_not_returned(self):
        client = OracleClient(
            self.GOLD, OracleConfig(recall=1.0, width=3, pattern_aware=True, seed=1)
        )
        messages = [
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "slots": [
                            {
                                "id": "A1",
                                "clue": "x",
                                "len": 3,
                                "rejected": ["CAT"],
                            }
                        ]
                    }
                ),
            }
        ]
        payload = json.loads(client.complete(model="o", messages=messages).text)
        answers = {c["answer"] for c in payload["items"][0]["candidates"]}
        self.assertNotIn("CAT", answers)


class TestEffectiveRecall(unittest.TestCase):
    def test_no_pattern_leaves_base_unchanged(self):
        self.assertEqual(effective_recall(0.4, None, "CAT"), 0.4)
        self.assertEqual(effective_recall(0.4, "???", "CAT"), 0.4)

    def test_full_match_is_certain(self):
        self.assertEqual(effective_recall(0.0, "CAT", "CAT"), 1.0)

    def test_partial_interpolates(self):
        self.assertAlmostEqual(effective_recall(0.0, "C?T", "CAT"), 2 / 3)

    def test_contradiction_is_zero(self):
        self.assertEqual(effective_recall(1.0, "DOG", "CAT"), 0.0)


class TestLoadEnvFile(unittest.TestCase):
    def test_does_not_override_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, ".env")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("CROSSWORD_TEST_KEY=fromfile\n")
            os.environ["CROSSWORD_TEST_KEY"] = "already"
            try:
                loaded = load_env_file(path)
                self.assertEqual(loaded, [os.path.abspath(path)])
                self.assertEqual(os.environ["CROSSWORD_TEST_KEY"], "already")
            finally:
                os.environ.pop("CROSSWORD_TEST_KEY", None)

    def test_fills_missing_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, ".env")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("CROSSWORD_TEST_KEY=fromfile\n")
            os.environ.pop("CROSSWORD_TEST_KEY", None)
            try:
                load_env_file(path)
                self.assertEqual(os.environ["CROSSWORD_TEST_KEY"], "fromfile")
            finally:
                os.environ.pop("CROSSWORD_TEST_KEY", None)


class TestUsage(unittest.TestCase):
    def test_accumulates_per_model(self):
        usage = Usage()
        usage.record(Completion(text="", model="a", prompt_tokens=10, completion_tokens=5))
        usage.record(Completion(text="", model="a", prompt_tokens=1, completion_tokens=1))
        usage.record(Completion(text="", model="b", prompt_tokens=2, completion_tokens=2))
        self.assertEqual(usage.calls, 3)
        self.assertEqual(usage.total_tokens, 21)
        self.assertEqual(
            usage.by_model,
            {"a": {"prompt": 11, "completion": 6}, "b": {"prompt": 2, "completion": 2}},
        )


if __name__ == "__main__":
    unittest.main()
