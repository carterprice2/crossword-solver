"""HTTP API for listing corpus puzzles and streaming a live solve.

Skipped entirely when FastAPI is not installed so ``make test`` stays
stdlib-only.
"""

from __future__ import annotations

import importlib.util
import json
import unittest

HAS_FASTAPI = importlib.util.find_spec("fastapi") is not None


@unittest.skipUnless(HAS_FASTAPI, "fastapi not installed")
class TestApi(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient

        from crossword.api.app import create_app

        self.app = create_app()
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
        self.assertEqual(body["models"], list(KNOWN_MODELS))
        self.assertTrue(body["models"])

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


if __name__ == "__main__":
    unittest.main()
