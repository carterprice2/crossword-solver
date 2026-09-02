"""HTTP API for listing corpus puzzles and streaming a live solve.

Skipped entirely when FastAPI is not installed so ``make test`` stays
stdlib-only.
"""

from __future__ import annotations

import importlib.util
import json
import unittest

from crossword.api.limits import RateLimiter
from crossword.client import ScriptedClient

HAS_FASTAPI = importlib.util.find_spec("fastapi") is not None


@unittest.skipUnless(HAS_FASTAPI, "fastapi not installed")
class TestApi(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient

        from crossword.api.app import create_app

        self.app = create_app(limiter=RateLimiter(hourly=1000, daily=1000))
        self.client = TestClient(self.app)

    def test_lists_mini_suite_and_puzzles(self):
        suites = self.client.get("/api/suites")
        self.assertEqual(suites.status_code, 200)
        ids = {s["id"] for s in suites.json()}
        self.assertIn("mini", ids)

        puzzles = self.client.get("/api/puzzles", params={"suite": "mini"})
        self.assertEqual(puzzles.status_code, 200)
        payload = puzzles.json()
        self.assertTrue(any(p["id"] == "mini-07-00-0" for p in payload))

    def test_defaults_include_nebius_models(self):
        from crossword.client import DEFAULT_MODEL, KNOWN_MODELS

        response = self.client.get("/api/defaults")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["model"], DEFAULT_MODEL)
        self.assertEqual(
            body["models"],
            [
                "Qwen/Qwen3-30B-A3B-Instruct-2507",
                "Qwen/Qwen3-235B-A22B-Instruct-2507",
                "Qwen/Qwen3.5-397B-A17B",
                "meta-llama/Llama-3.3-70B-Instruct",
                "openai/gpt-oss-120b",
                "deepseek-ai/DeepSeek-V4-Pro",
                "zai-org/GLM-5.2",
                "MiniMaxAI/MiniMax-M3",
            ],
        )
        self.assertEqual(body["models"], list(KNOWN_MODELS))
        self.assertEqual(body["ensemble_model"], "meta-llama/Llama-3.3-70B-Instruct")
        self.assertEqual(
            [arm["id"] for arm in body["arms"]],
            ["a2", "a3", "a4", "a5", "a6"],
        )

    def test_puzzle_payload_strips_gold(self):
        response = self.client.get("/api/puzzles/mini-07-00-0")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["id"], "mini-07-00-0")
        self.assertIn("clues", body)
        self.assertNotIn("gold", str(body["clues"]))
        for clue in body["clues"]["across"]:
            self.assertNotIn("gold", clue)

    def test_oracle_solve_streams_finished_with_scores(self):
        start = self.client.post(
            "/api/solves",
            json={"puzzle_id": "mini-07-00-0", "backend": "oracle", "arm": "a3"},
        )
        self.assertEqual(start.status_code, 200, start.text)
        job_id = start.json()["job_id"]
        self.assertTrue(job_id)

        with self.client.stream("GET", f"/api/solves/{job_id}/events") as stream:
            kinds = []
            events = []
            finished = None
            for line in stream.iter_lines():
                if not line or line.startswith(":"):
                    continue
                if line.startswith("data:"):
                    event = json.loads(line[5:].strip())
                    events.append(event)
                    kinds.append(event["kind"])
                    if event["kind"] == "finished":
                        finished = event
                        break
            self.assertIn("search", kinds)
            self.assertIsNotNone(finished)
            self.assertIn("scores", finished)
            self.assertIn("wcr", finished["scores"])
            self.assertGreater(finished["scores"]["wcr"], 0)
            self.assertIn("cells", finished)
            self.assertTrue(finished["cells"])
            self.assertIn("gold", finished)
            self.assertTrue(finished["gold"])
            self.assertIn("letter", finished["gold"][0])
            self.assertIn("candidates", kinds)
            cand = next(e for e in events if e["kind"] == "candidates")
            slot = cand["data"]["slots"][0]
            self.assertNotIn("gold", slot)
            self.assertNotIn("hit", slot)

        status = self.client.get(f"/api/solves/{job_id}")
        self.assertEqual(status.status_code, 200)
        self.assertEqual(status.json()["status"], "done")
        self.assertIn("scores", status.json())

    def test_second_solve_while_busy_returns_409(self):
        self.app.state.store.occupy("held")
        second = self.client.post(
            "/api/solves",
            json={"puzzle_id": "mini-07-01-0", "backend": "oracle", "arm": "a3"},
        )
        self.assertEqual(second.status_code, 409)
        self.assertIn("busy", second.json()["detail"].lower())

    def test_debug_solve_annotates_candidate_hits(self):
        start = self.client.post(
            "/api/solves",
            json={
                "puzzle_id": "mini-07-00-0",
                "backend": "oracle",
                "arm": "a3",
                "debug": True,
            },
        )
        self.assertEqual(start.status_code, 200, start.text)
        job_id = start.json()["job_id"]
        with self.client.stream("GET", f"/api/solves/{job_id}/events") as stream:
            cand = None
            for line in stream.iter_lines():
                if not line or line.startswith(":"):
                    continue
                if line.startswith("data:"):
                    event = json.loads(line[5:].strip())
                    if event["kind"] == "candidates" and cand is None:
                        cand = event
                    if event["kind"] == "finished":
                        break
        self.assertIsNotNone(cand)
        slot = cand["data"]["slots"][0]
        self.assertIn("gold", slot)
        self.assertIn("hit", slot)
        self.assertTrue(slot["gold"])
        self.assertIn("candidates", slot)

    def test_debug_finished_includes_candidate_batches(self):
        start = self.client.post(
            "/api/solves",
            json={
                "puzzle_id": "mini-09-00-0",
                "backend": "oracle",
                "arm": "a3",
                "debug": True,
            },
        )
        self.assertEqual(start.status_code, 200, start.text)
        job_id = start.json()["job_id"]
        finished = None
        with self.client.stream("GET", f"/api/solves/{job_id}/events") as stream:
            for line in stream.iter_lines():
                if not line or line.startswith(":"):
                    continue
                if line.startswith("data:"):
                    event = json.loads(line[5:].strip())
                    if event["kind"] == "finished":
                        finished = event
                        break
        self.assertIsNotNone(finished)
        batches = finished["candidate_batches"]
        self.assertGreater(len(batches), 1)
        total = sum(len(batch["slots"]) for batch in batches)
        self.assertGreaterEqual(total, 32)
        slot = batches[0]["slots"][0]
        self.assertIn("gold", slot)
        self.assertIn("hit", slot)
        self.assertIn("id", slot)

    def test_unknown_puzzle_is_404(self):
        response = self.client.get("/api/puzzles/no-such-puzzle")
        self.assertEqual(response.status_code, 404)

    def test_serves_built_frontend_index(self):
        response = self.client.get("/")
        if response.status_code == 404:
            self.skipTest("web/dist not built")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Crossword Agent", response.text)


PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
MINI_ROWS = ["C.T", ".#.", "T.G"]
ACROSS = "1. C to T\n3. T to G"
DOWN = "1. C to T\n2. T to G"
XD_NO_GOLD = (
    "Title: X\n\n\n"
    "...\n.#.\n...\n\n\n"
    "A1. One\nA3. Two\nD1. Three\nD2. Four\n"
)


@unittest.skipUnless(HAS_FASTAPI, "fastapi not installed")
class TestIngestApi(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient

        from crossword.api.app import create_app

        self.app = create_app(
            vision=lambda *_: list(MINI_ROWS),
            client_factory=lambda *a, **k: ScriptedClient(),
            require_key=False,
            limiter=RateLimiter(hourly=1000, daily=1000),
        )
        self.client = TestClient(self.app)

    def _wait_finished(self, job_id: str) -> dict:
        finished = None
        with self.client.stream("GET", f"/api/solves/{job_id}/events") as stream:
            for line in stream.iter_lines():
                if not line or line.startswith(":"):
                    continue
                if line.startswith("data:"):
                    event = json.loads(line[5:].strip())
                    if event["kind"] in ("finished", "error"):
                        finished = event
                        break
        self.assertIsNotNone(finished)
        return finished

    def test_health(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["ok"])
        self.assertIn("has_key", body)
        self.assertIn("busy", body)

    def test_ingest_ready_starts_solve(self):
        response = self.client.post(
            "/api/ingest",
            json={"image": PNG, "across": ACROSS, "down": DOWN},
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["status"], "ready")
        self.assertTrue(body["job_id"])
        self.assertFalse(body["puzzle"]["has_gold"])
        finished = self._wait_finished(body["job_id"])
        self.assertEqual(finished["kind"], "finished")
        self.assertEqual(finished["gold"], [])
        self.assertIsNone(finished["scores"])

    def test_needs_edit_then_grid_fix(self):
        from crossword.api.app import create_app
        from fastapi.testclient import TestClient

        app = create_app(
            vision=lambda *_: ["###", "###", "###"],
            client_factory=lambda *a, **k: ScriptedClient(),
            require_key=False,
            limiter=RateLimiter(hourly=1000, daily=1000),
        )
        client = TestClient(app)
        first = client.post(
            "/api/ingest",
            json={"image": PNG, "across": ACROSS, "down": DOWN},
        )
        self.assertEqual(first.status_code, 200, first.text)
        body = first.json()
        self.assertEqual(body["status"], "needs_edit")
        self.assertTrue(body["draft_id"])
        self.assertIsNone(body.get("job_id"))
        fixed = client.post(
            f"/api/ingest/{body['draft_id']}/grid",
            json={"rows": MINI_ROWS},
        )
        self.assertEqual(fixed.status_code, 200, fixed.text)
        self.assertEqual(fixed.json()["status"], "ready")
        self.assertTrue(fixed.json()["job_id"])
        finished = None
        job_id = fixed.json()["job_id"]
        with client.stream("GET", f"/api/solves/{job_id}/events") as stream:
            for line in stream.iter_lines():
                if not line or line.startswith(":"):
                    continue
                if line.startswith("data:"):
                    event = json.loads(line[5:].strip())
                    if event["kind"] in ("finished", "error"):
                        finished = event
                        break
        self.assertIsNotNone(finished)
        self.assertEqual(finished["kind"], "finished")

    def test_xd_paste_without_gold(self):
        response = self.client.post("/api/ingest", json={"xd": XD_NO_GOLD})
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["status"], "ready")
        self.assertFalse(body["puzzle"]["has_gold"])
        finished = self._wait_finished(body["job_id"])
        self.assertEqual(finished["gold"], [])
        self.assertIsNone(finished["scores"])

    def test_oracle_on_upload_is_400(self):
        ingested = self.client.post(
            "/api/ingest",
            json={"image": PNG, "across": ACROSS, "down": DOWN},
        )
        self.assertEqual(ingested.status_code, 200, ingested.text)
        puzzle_id = ingested.json()["puzzle"]["id"]
        # Drain the auto-started job so the store is free.
        self._wait_finished(ingested.json()["job_id"])
        blocked = self.client.post(
            "/api/solves",
            json={"puzzle_id": puzzle_id, "backend": "oracle", "arm": "a3"},
        )
        self.assertEqual(blocked.status_code, 400)
        self.assertIn("answer key", blocked.json()["detail"].lower())

    def test_missing_key_is_503(self):
        from crossword.api.app import create_app
        from fastapi.testclient import TestClient

        app = create_app(
            vision=lambda *_: list(MINI_ROWS),
            require_key=True,
            limiter=RateLimiter(hourly=1000, daily=1000),
        )
        client = TestClient(app)
        # Force the key check regardless of the developer's .env.
        previous = app.state.require_key
        app.state.require_key = True
        # has_key() reads env; patch by using require_key True and clearing env.
        import os

        old = os.environ.pop("NEBIUS_API_KEY", None)
        try:
            response = client.post(
                "/api/ingest",
                json={"image": PNG, "across": ACROSS, "down": DOWN},
            )
            self.assertEqual(response.status_code, 503)
            self.assertIn("Token Factory", response.json()["detail"])
        finally:
            if old is not None:
                os.environ["NEBIUS_API_KEY"] = old
            app.state.require_key = previous

    def test_hourly_limit_is_429(self):
        from fastapi.testclient import TestClient

        from crossword.api.app import create_app

        app = create_app(
            vision=lambda *_: list(MINI_ROWS),
            client_factory=lambda *a, **k: ScriptedClient(),
            require_key=False,
            limiter=RateLimiter(hourly=1, daily=40),
        )
        client = TestClient(app)
        payload = {"image": PNG, "across": ACROSS, "down": DOWN}
        first = client.post("/api/ingest", json=payload)
        self.assertEqual(first.status_code, 200, first.text)
        self._drain(client, first.json()["job_id"])
        second = client.post("/api/ingest", json=payload)
        self.assertEqual(second.status_code, 429)
        self.assertIn("rate-limited", second.json()["detail"].lower())

    def _drain(self, client, job_id: str) -> None:
        with client.stream("GET", f"/api/solves/{job_id}/events") as stream:
            for line in stream.iter_lines():
                if not line or line.startswith(":"):
                    continue
                if line.startswith("data:"):
                    event = json.loads(line[5:].strip())
                    if event["kind"] in ("finished", "error"):
                        return


if __name__ == "__main__":
    unittest.main()
