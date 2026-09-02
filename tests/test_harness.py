import json
import os
import tempfile
import unittest

from crossword.client import OracleClient, OracleConfig
from crossword.eval.harness import Harness, build_arms
from crossword.xd import parse_xd

MINI = """\
Title: Harness Mini


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


class TestHarnessResume(unittest.TestCase):
    def setUp(self):
        self.puzzle = parse_xd(MINI, puzzle_id="harness-mini")
        self.arms = build_arms()
        self.calls = {"n": 0}

    def factory(self, puzzle, arm, seed):
        self.calls["n"] += 1
        gold = {s.id: s.gold or "" for s in puzzle.slots}
        return OracleClient(gold, OracleConfig(recall=1.0, top1_error=0.0, seed=seed))

    def test_two_models_run_twice(self):
        with tempfile.TemporaryDirectory() as tmp:
            harness = Harness(self.factory, self.arms, out_dir=tmp)
            payload = harness.run(
                [self.puzzle],
                ["a1"],
                models=["m-a", "m-b"],
                run_id="r",
            )
            self.assertEqual(self.calls["n"], 2)
            self.assertEqual(len(payload["records"]), 2)
            self.assertEqual({r["model"] for r in payload["records"]}, {"m-a", "m-b"})
            self.assertTrue(os.path.isfile(os.path.join(tmp, "r", "cells.jsonl")))

    def test_resume_skips_completed_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            harness = Harness(self.factory, self.arms, out_dir=tmp)
            harness.run(
                [self.puzzle], ["a1"], models=["m-a", "m-b"], run_id="r"
            )
            self.assertEqual(self.calls["n"], 2)
            harness.run(
                [self.puzzle], ["a1"], models=["m-a", "m-b"], run_id="r"
            )
            self.assertEqual(self.calls["n"], 2)

    def test_failed_cell_is_written(self):
        def boom_factory(puzzle, arm, seed):
            self.calls["n"] += 1
            raise RuntimeError("boom")

        with tempfile.TemporaryDirectory() as tmp:
            harness = Harness(boom_factory, self.arms, out_dir=tmp)
            payload = harness.run(
                [self.puzzle], ["a1"], models=["m-a"], run_id="r"
            )
            path = os.path.join(tmp, "r", "cells.jsonl")
            with open(path, encoding="utf-8") as fh:
                line = json.loads(fh.readline())
        self.assertEqual(len(payload["records"]), 1)
        self.assertIsNone(payload["records"][0]["scores"])
        self.assertIn("boom", payload["records"][0]["error"])
        self.assertIsNone(line["scores"])
        self.assertIn("boom", line["error"])

    def test_retry_errors_re_runs_failed_key(self):
        state = {"fail": True}

        def flaky(puzzle, arm, seed):
            self.calls["n"] += 1
            if state["fail"]:
                raise RuntimeError("boom")
            gold = {s.id: s.gold or "" for s in puzzle.slots}
            return OracleClient(gold, OracleConfig(recall=1.0, top1_error=0.0, seed=seed))

        with tempfile.TemporaryDirectory() as tmp:
            harness = Harness(flaky, self.arms, out_dir=tmp)
            harness.run([self.puzzle], ["a1"], models=["m-a"], run_id="r")
            self.assertEqual(self.calls["n"], 1)
            state["fail"] = False
            payload = harness.run(
                [self.puzzle],
                ["a1"],
                models=["m-a"],
                run_id="r",
                retry_errors=True,
            )
            self.assertEqual(self.calls["n"], 2)
            self.assertIsNotNone(payload["records"][0]["scores"])

    def test_a5_uses_cell_model_for_repair(self):
        seen = []

        def capture(puzzle, arm, seed):
            seen.append((arm.config.model, arm.config.repair_model))
            gold = {s.id: s.gold or "" for s in puzzle.slots}
            return OracleClient(gold, OracleConfig(recall=1.0, top1_error=0.0, seed=seed))

        with tempfile.TemporaryDirectory() as tmp:
            Harness(capture, self.arms, out_dir=tmp).run(
                [self.puzzle], ["a5"], models=["cell-model"], run_id="r"
            )
        self.assertEqual(seen, [("cell-model", "cell-model")])
