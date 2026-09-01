import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from crossword.cli import build_parser, main
from crossword.eval.report import summarize
from crossword.ui.live import LiveView
from crossword.xd import parse_xd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MINI = os.path.join(ROOT, "corpus", "mini")

SMALL = """\
Title: CLI Mini


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


def a_puzzle_path():
    import glob

    paths = sorted(glob.glob(os.path.join(MINI, "*.xd")))
    return paths[0] if paths else None


class TestParser(unittest.TestCase):
    def test_solve_defaults(self):
        args = build_parser().parse_args(["solve", "p.xd"])
        self.assertEqual(args.arm, "a3")
        self.assertEqual(args.backend, "nebius")
        self.assertFalse(args.live)

    def test_serve_defaults(self):
        args = build_parser().parse_args(["serve"])
        self.assertEqual(args.port, 8000)
        self.assertEqual(args.host, "127.0.0.1")
        self.assertFalse(args.build)

    def test_eval_arms_parse(self):
        args = build_parser().parse_args(["eval", "--arms", "a0,a3"])
        self.assertEqual(args.arms, "a0,a3")

    def test_unknown_command_exits(self):
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["nonsense"])

    def test_missing_command_exits(self):
        with self.assertRaises(SystemExit):
            build_parser().parse_args([])


class TestSolveOffline(unittest.TestCase):
    """The CLI must work end to end with no network and no API key."""

    def setUp(self):
        self.path = a_puzzle_path()
        if not self.path:
            self.skipTest("corpus not generated")

    def run_cli(self, argv):
        stdout, stderr = io.StringIO(), io.StringIO()
        saved = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = stdout, stderr
        try:
            code = main(argv)
        finally:
            sys.stdout, sys.stderr = saved
        return code, stdout.getvalue(), stderr.getvalue()

    def test_oracle_solve_succeeds(self):
        code, out, err = self.run_cli(
            ["solve", self.path, "--backend", "oracle", "--oracle-recall", "0.95"]
        )
        self.assertEqual(code, 0)
        self.assertIn("WCR", out)

    def test_json_output_is_valid(self):
        code, out, _ = self.run_cli(
            ["solve", self.path, "--backend", "oracle", "--json"]
        )
        self.assertEqual(code, 0)
        payload = json.loads(out[out.index("{") :])
        self.assertIn("scores", payload)
        self.assertIn("rounds", payload)

    def test_arm_selection_changes_behaviour(self):
        _, a1_out, _ = self.run_cli(
            ["solve", self.path, "--backend", "oracle", "--arm", "a1", "--json"]
        )
        _, a3_out, _ = self.run_cli(
            ["solve", self.path, "--backend", "oracle", "--arm", "a3", "--json"]
        )
        a1 = json.loads(a1_out[a1_out.index("{") :])
        a3 = json.loads(a3_out[a3_out.index("{") :])
        # The full agent should spend more calls than the single-pass arm.
        self.assertGreaterEqual(a3["calls"], a1["calls"])

    def test_unknown_arm_is_rejected(self):
        with self.assertRaises(SystemExit):
            self.run_cli(["solve", self.path, "--backend", "oracle", "--arm", "zz"])

    def test_missing_key_reports_clearly(self):
        saved = os.environ.pop("NEBIUS_API_KEY", None)
        try:
            with patch("crossword.cli.load_env_file", lambda *a, **k: []):
                code, _, err = self.run_cli(["solve", self.path])
            self.assertEqual(code, 2)
            self.assertIn("NEBIUS_API_KEY", err)
        finally:
            if saved is not None:
                os.environ["NEBIUS_API_KEY"] = saved


class TestEvalOffline(unittest.TestCase):
    def test_matrix_runs_and_writes_results(self):
        if not a_puzzle_path():
            self.skipTest("corpus not generated")
        with tempfile.TemporaryDirectory() as tmp:
            stdout, stderr = io.StringIO(), io.StringIO()
            saved = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = stdout, stderr
            try:
                code = main(
                    [
                        "eval", "--suite", "mini", "--arms", "a1,a3",
                        "--backend", "oracle", "--limit", "2", "--out", tmp,
                        "--run-id", "t",
                    ]
                )
            finally:
                sys.stdout, sys.stderr = saved
            self.assertEqual(code, 0)
            directory = os.path.join(tmp, "t")
            self.assertTrue(os.path.isfile(os.path.join(directory, "results.json")))
            self.assertTrue(os.path.isfile(os.path.join(directory, "summary.md")))
            with open(os.path.join(directory, "results.json")) as fh:
                payload = json.load(fh)
            self.assertEqual(len(payload["records"]), 4)  # 2 puzzles x 2 arms


class TestReportRendering(unittest.TestCase):
    def test_summarize_handles_a_minimal_payload(self):
        payload = {
            "run_id": "r",
            "generated_at": "now",
            "arms": {"a3": {"label": "full", "description": "", "model": "m",
                            "repair_model": "m", "ensemble_model": "",
                            "max_rounds": 3, "use_constraints": True,
                            "use_repair": True, "unknown_mass": 0.15}},
            "seeds": [0],
            "prefill_ratios": [0.0],
            "puzzles": [{"id": "p", "size": "9x9"}],
            "records": [
                {
                    "puzzle_id": "p", "arm": "a3", "seed": 0, "prefill": 0.0,
                    "strata": {"size": "9x9", "provenance": "generated"},
                    "scores": {"wcr": 1.0, "lcr": 1.0, "icr": 1.0, "exact": True,
                               "cell_precision": 1.0, "cell_recall": 1.0},
                    "solve": {"calls": 3, "prompt_tokens": 10,
                              "completion_tokens": 5, "seconds": 1.0,
                              "open_slots": [], "rungs": {"m": "strict_schema"}},
                    "per_slot": {},
                }
            ],
            "calibration": {"a3": {"ece": 0.1, "brier": 0.05, "n": 4, "bins": []}},
            "slot_records": [
                {"puzzle_id": "p", "arm": "a3", "seed": 0, "slot": "A1",
                 "length": 3, "length_bucket": "3", "clue_type": "definition",
                 "correct": True}
            ],
        }
        text = summarize(payload)
        self.assertIn("Evaluation summary", text)
        self.assertIn("WCR", text)
        self.assertIn("Exact-solve", text)
        self.assertIn("strict_schema", text)

    def test_summarize_survives_no_records(self):
        payload = {
            "run_id": "empty", "arms": {}, "records": [], "puzzles": [],
            "calibration": {}, "slot_records": [],
        }
        self.assertIn("empty", summarize(payload))


class TestLiveView(unittest.TestCase):
    def test_renders_without_a_terminal(self):
        puzzle = parse_xd(SMALL)
        stream = io.StringIO()
        view = LiveView(puzzle, stream=stream, color=False)
        view.render(force=True)
        output = stream.getvalue()
        self.assertIn("Reno Crossword Agent", output)
        self.assertIn("##", output) if puzzle.grid.blocks else None
        self.assertIn("round 0", output)

    def test_absorbs_grid_events(self):
        puzzle = parse_xd(SMALL)
        stream = io.StringIO()
        view = LiveView(puzzle, stream=stream, color=False, min_interval=0)
        from crossword.agent.trace import SEARCH, SolveEvent

        view.handle(
            SolveEvent(kind=SEARCH, round=1, data={"filled": 3, "icr": 1.0,
                                                   "grid": ["CAT", "ARE", "BEE"]})
        )
        self.assertEqual(view.cells[(0, 0)], "C")
        self.assertIn("C A T", stream.getvalue())

    def _finish(self, solution, *, color=False):
        puzzle = parse_xd(SMALL)
        stream = io.StringIO()
        view = LiveView(puzzle, stream=stream, color=color, min_interval=0)
        from crossword.agent.solver import SolveResult
        from crossword.agent.trace import REPAIR, SolveEvent
        from crossword.eval.metrics import score_solution

        view.handle(
            SolveEvent(
                kind=REPAIR, round=2,
                data={"slots": ["A12", "D4"], "model": "oracle"},
            )
        )
        result = SolveResult(puzzle=puzzle, solution=solution)
        scores = score_solution(puzzle, solution)
        view.finish(result, scores)
        return stream.getvalue()

    def test_gold_check_replaces_the_repair_log(self):
        puzzle = parse_xd(SMALL)
        text = self._finish(puzzle.gold_solution())
        after = text[text.rfind("gold check"):]
        self.assertIn("gold check", after)
        self.assertIn("WCR", after)
        self.assertNotIn("repair", after.lower())
        self.assertNotIn("round 2", after)

    def test_gold_check_paints_correct_green_and_wrong_red(self):
        puzzle = parse_xd(SMALL)
        wrong = dict(puzzle.gold_solution())
        wrong[(0, 0)] = "X"
        from crossword.ui.live import GREEN, RED

        text = self._finish(wrong, color=True)
        after = text[text.rfind("gold check"):]
        self.assertIn(RED, after)
        self.assertIn(GREEN, after)
        self.assertIn("PARTIAL", after)

    def test_a_broken_listener_does_not_kill_the_solve(self):
        from crossword.agent.trace import Tracer

        def explode(event):
            raise RuntimeError("ui bug")

        tracer = Tracer(listeners=[explode])
        tracer.emit("round_start", "fine")  # must not raise
        self.assertEqual(len(tracer.events), 1)


class TestModuleEntryPoint(unittest.TestCase):
    def test_python_m_crossword_runs(self):
        result = subprocess.run(
            [sys.executable, "-m", "crossword", "--version"],
            cwd=ROOT, capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("0.1", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
