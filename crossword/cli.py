"""Command line interface.

    crossword solve   corpus/mini/mini-09-00-0.xd --live
    crossword eval    --suite mini --arms a0,a1,a2,a3
    crossword report  results/run-.../
    crossword generate --size 9 --seed 3
    crossword models  ping
    crossword serve
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

from . import __version__
from .agent.trace import Tracer
from .client import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_REASONING_MODEL,
    KNOWN_MODELS,
    ModelError,
    NebiusClient,
    load_env_file,
)
from .eval.harness import Harness, build_arms
from .eval.report import write_summary
from .run import (
    HERE,
    RunError,
    load_puzzles,
    make_client,
    prefill_for,
    run_solve,
    solver_config,
    suite_paths,
)
from .ui.live import LiveView
from .xd import load_xd


def _make_client(args, puzzle=None):
    try:
        return make_client(
            puzzle,
            backend=args.backend,
            seed=args.seed,
            base_url=args.base_url,
            timeout=args.timeout,
            replay=getattr(args, "replay", None),
            replay_loose=getattr(args, "replay_loose", False),
            oracle_recall=getattr(args, "oracle_recall", 0.8),
            oracle_top1_error=getattr(args, "oracle_top1_error", 0.35),
            oracle_independent=getattr(args, "oracle_independent", False),
            record=getattr(args, "record", None),
        )
    except RunError as exc:
        raise SystemExit(str(exc)) from exc


def _solver_config(args):
    try:
        arm, config = solver_config(
            arm=args.arm,
            model=args.model,
            repair_model=args.repair_model,
            seed=args.seed,
            workers=args.workers,
            rounds=getattr(args, "rounds", 0) or 0,
        )
    except RunError as exc:
        raise SystemExit(str(exc)) from exc
    return arm, config


# -- commands --------------------------------------------------------------


def cmd_solve(args) -> int:
    puzzle = load_xd(args.puzzle)
    client = _make_client(args, puzzle)
    arm, config = _solver_config(args)

    prefill = prefill_for(puzzle, args.prefill, args.seed)
    view = LiveView(puzzle) if args.live else None
    tracer = Tracer(args.trace, listeners=[view] if view else [])

    print(
        f"{puzzle.id}  {puzzle.grid.height}x{puzzle.grid.width}  "
        f"{len(puzzle.slots)} slots  arm {args.arm} ({arm.label})",
        file=sys.stderr,
    )
    if args.backend == "nebius":
        print(f"model {args.model} -> repair {args.repair_model}", file=sys.stderr)

    context = view if view else _NullContext()
    with context:
        result, scores = run_solve(
            puzzle,
            client=client,
            config=config,
            one_shot=arm.one_shot,
            tracer=tracer,
            prefill=prefill,
        )
    if view:
        view.finish(result, scores)
    else:
        for row in puzzle.grid.render(result.solution):
            print(row)
        if scores:
            print(
                f"\nWCR {scores.wcr:.3f}  LCR {scores.lcr:.3f}  ICR {scores.icr:.3f}  "
                f"exact={scores.exact}  {result.usage.total_tokens:,} tokens  "
                f"{result.usage.calls} calls  {result.seconds:.1f}s"
            )
    if result.open_slots:
        print(
            f"declined {len(result.open_slots)} slot(s): "
            f"{', '.join(result.open_slots[:10])}",
            file=sys.stderr,
        )
    if args.json:
        payload = result.as_dict()
        if scores:
            payload["scores"] = scores.as_dict()
        print(json.dumps(payload, indent=2))
    return 0 if (scores is None or scores.wcr > 0) else 1


def cmd_eval(args) -> int:
    try:
        paths = suite_paths(args.suite)
    except RunError as exc:
        raise SystemExit(str(exc)) from exc
    puzzles = load_puzzles(paths, args.limit)
    if not puzzles:
        raise SystemExit(f"no puzzles in suite {args.suite!r}")
    arm_names = [a.strip() for a in args.arms.split(",") if a.strip()]
    arms = build_arms(args.model, args.repair_model, args.ensemble_model)
    for name in arm_names:
        if name not in arms:
            raise SystemExit(f"unknown arm {name!r}; choose from {', '.join(arms)}")

    seeds = [int(s) for s in str(args.seeds).split(",")] if "," in str(args.seeds) else list(range(int(args.seeds)))
    ratios = [float(r) for r in str(args.prefill).split(",")]

    total = len(puzzles) * len(arm_names) * len(seeds) * len(ratios)
    print(
        f"{len(puzzles)} puzzles x {len(arm_names)} arms x {len(seeds)} seed(s) "
        f"x {len(ratios)} prefill = {total} solves",
        file=sys.stderr,
    )
    if args.backend == "nebius":
        print(
            "This will spend real tokens. Ctrl-C within 3s to stop.", file=sys.stderr
        )
        time.sleep(3)

    done = {"n": 0}

    def progress(record):
        done["n"] += 1
        print(
            f"  [{done['n']}/{total}] {record.puzzle_id} {record.arm} "
            f"seed={record.seed} WCR={record.scores.wcr:.3f}",
            file=sys.stderr,
        )

    def factory(puzzle, arm, seed):
        args.seed = seed
        return _make_client(args, puzzle)

    harness = Harness(factory, arms, out_dir=args.out, trace=args.trace)
    payload = harness.run(
        puzzles, arm_names, seeds=seeds, prefill_ratios=ratios,
        run_id=args.run_id, progress=progress,
    )
    directory = os.path.join(args.out, payload["run_id"])
    path = write_summary(directory)
    print(f"\nresults -> {directory}", file=sys.stderr)
    print(open(path, encoding="utf-8").read())
    return 0


def cmd_report(args) -> int:
    directory = args.directory
    if not os.path.isfile(os.path.join(directory, "results.json")):
        raise SystemExit(f"no results.json in {directory}")
    path = write_summary(directory)
    print(open(path, encoding="utf-8").read())
    print(f"wrote {path}", file=sys.stderr)
    return 0


def cmd_generate(args) -> int:
    from .gen.bank import load_bank
    from .gen.fill import build_puzzle
    from .gen.grids import search_templates
    from .xd import dump_xd

    bank = load_bank(args.bank)
    templates = search_templates(
        args.size, max_run=args.max_run, target_density=args.density,
        seed=args.seed, attempts=8000, want=1,
    )
    if not templates:
        raise SystemExit(
            f"no valid {args.size}x{args.size} template with max run {args.max_run}"
        )
    puzzle, stats = build_puzzle(templates[0], bank, seed=args.seed, title=args.title)
    text = dump_xd(puzzle)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"wrote {args.out} ({stats.seconds:.1f}s, {stats.nodes} nodes)",
              file=sys.stderr)
    else:
        print(text)
    return 0


def cmd_models(args) -> int:
    try:
        client = NebiusClient(base_url=args.base_url, timeout=args.timeout)
    except ModelError as exc:
        print(f"{exc}", file=sys.stderr)
        return 2
    try:
        available = client.list_models()
    except Exception as exc:
        print(f"could not list models: {exc}", file=sys.stderr)
        return 2
    print(f"{len(available)} model(s) reachable at {args.base_url}")
    for name in available:
        marker = " *" if name in KNOWN_MODELS else "  "
        print(f"{marker} {name}")
    missing = [m for m in KNOWN_MODELS if m not in available]
    if missing:
        print(
            f"\nnot reachable on this account: {', '.join(missing)}", file=sys.stderr
        )
    return 0


def cmd_serve(args) -> int:
    try:
        import uvicorn
    except ImportError:
        print(
            "web extra missing. From the repo root:\n"
            "  python3 -m venv .venv && .venv/bin/pip install -e '.[web]'",
            file=sys.stderr,
        )
        return 2

    web_dir = os.path.join(HERE, "web")
    dist = os.path.join(web_dir, "dist")
    if args.build:
        built = subprocess.run(["npm", "run", "build"], cwd=web_dir)
        if built.returncode:
            return built.returncode
    elif not os.path.isdir(dist):
        print(
            "no web/dist yet — serving the API only.\n"
            "  python3 -m crossword serve --build\n"
            "  or: cd web && npm run dev   (proxies /api to this server)",
            file=sys.stderr,
        )

    print(f"Crossword Agent  http://{args.host}:{args.port}", file=sys.stderr)
    uvicorn.run(
        "crossword.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# -- argument parsing ------------------------------------------------------


def add_common(parser):
    parser.add_argument(
        "--backend",
        default="nebius",
        choices=("nebius", "oracle", "replay"),
        help="nebius = live API; oracle = offline synthetic candidates; "
        "replay = a recorded trace",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--repair-model", default=DEFAULT_REASONING_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--replay", help="path to a recorded trace (jsonl)")
    parser.add_argument("--replay-loose", action="store_true",
                        help="serve recordings in order when the key misses")
    parser.add_argument("--oracle-recall", type=float, default=0.8)
    parser.add_argument("--oracle-top1-error", type=float, default=0.35)
    parser.add_argument(
        "--oracle-independent",
        action="store_true",
        help="oracle: do not raise recall from confirmed letters (old sweep)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="crossword",
        description="A crossword-solving agent on Nebius Token Factory.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    solve = sub.add_parser("solve", help="solve one puzzle")
    solve.add_argument("puzzle")
    solve.add_argument("--arm", default="a3")
    solve.add_argument("--live", action="store_true", help="animated grid view")
    solve.add_argument("--rounds", type=int, default=0, help="override max rounds")
    solve.add_argument("--prefill", type=float, default=0.0)
    solve.add_argument("--trace", help="write a solve trace to this path")
    solve.add_argument("--record", help="record API calls for later replay")
    solve.add_argument("--json", action="store_true")
    add_common(solve)
    solve.set_defaults(func=cmd_solve)

    run = sub.add_parser("eval", help="run the arm x puzzle matrix")
    run.add_argument("--suite", default="mini")
    run.add_argument("--arms", default="a0,a1,a2,a3")
    run.add_argument("--seeds", default="1", help="count, or a comma list")
    run.add_argument("--prefill", default="0.0", help="comma list of ratios")
    run.add_argument("--limit", type=int, default=0)
    run.add_argument("--out", default="results")
    run.add_argument("--run-id")
    run.add_argument("--trace", action="store_true")
    run.add_argument("--ensemble-model", default="meta-llama/Llama-3.3-70B-Instruct")
    run.add_argument("--record", help="record API calls for later replay")
    add_common(run)
    run.set_defaults(func=cmd_eval)

    report = sub.add_parser("report", help="regenerate summary.md from results.json")
    report.add_argument("directory")
    report.set_defaults(func=cmd_report)

    generate = sub.add_parser("generate", help="generate a puzzle")
    generate.add_argument("--size", type=int, default=9)
    generate.add_argument("--max-run", type=int, default=6)
    generate.add_argument("--density", type=float, default=0.18)
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--bank")
    generate.add_argument("--title", default="")
    generate.add_argument("--out")
    generate.set_defaults(func=cmd_generate)

    models = sub.add_parser("models", help="check the API key and list models")
    models.add_argument("action", nargs="?", default="ping", choices=("ping", "list"))
    models.add_argument("--base-url", default=DEFAULT_BASE_URL)
    models.add_argument("--timeout", type=float, default=30.0)
    models.set_defaults(func=cmd_models)

    serve = sub.add_parser("serve", help="open the web UI")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--build", action="store_true", help="npm run build, then serve web/dist")
    serve.add_argument("--reload", action="store_true", help="reload the API on Python changes")
    serve.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    load_env_file()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ModelError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
