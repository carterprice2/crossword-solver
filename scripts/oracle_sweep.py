#!/usr/bin/env python3
"""Measure the architecture offline, with no API key and no network.

The solver's job is to turn *noisy candidate lists* into a correct grid. That
input can be simulated: `OracleClient` builds candidate lists from a known
solution with a controlled probability that the correct answer is missing
(``recall``) and that it is not ranked first (``top1_error``).

Sweeping those knobs answers the question the ablation is really asking --
*how much model error does the constraint and repair machinery absorb?* -- and
it answers it reproducibly, without spending a token. A live model run then
tells you where on this curve a given Nebius model actually sits.

    python3 scripts/oracle_sweep.py --out results/synthetic-sweep.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from crossword.client import OracleClient, OracleConfig  # noqa: E402
from crossword.eval.harness import Harness, build_arms  # noqa: E402
from crossword.eval.stats import paired_bootstrap  # noqa: E402
from crossword.xd import load_xd  # noqa: E402

#: (recall, top1_error) -- from a strong model to a weak one.
LEVELS = [
    (0.95, 0.20),
    (0.80, 0.35),
    (0.65, 0.45),
    (0.50, 0.55),
    (0.35, 0.65),
]

ARMS = ["a0", "a1", "a2", "a3", "a6"]

CONTRASTS = [
    ("a1", "a0", "a1_minus_a0"),
    ("a2", "a1", "a2_minus_a1"),
    ("a3", "a2", "a3_minus_a2"),
    ("a3", "a6", "a3_minus_a6"),
]


def mean(values):
    return sum(values) / len(values) if values else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/pattern-aware-sweep.json")
    ap.add_argument("--suite", default="corpus/mini")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--resamples", type=int, default=4000)
    ap.add_argument(
        "--independent",
        action="store_true",
        help="Disable pattern-aware recall (original sweep; cannot test repair prompts).",
    )
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.suite, "*.xd")))
    puzzles = [load_xd(p) for p in paths]
    if not puzzles:
        raise SystemExit(f"no puzzles in {args.suite}")
    seeds = list(range(args.seeds))
    arms = build_arms(model="oracle", reasoning_model="oracle", ensemble_model="oracle")

    sweep = []
    for recall, top1_error in LEVELS:

        def factory(puzzle, arm, seed, _r=recall, _t=top1_error):
            gold = {s.id: s.gold or "" for s in puzzle.slots}
            return OracleClient(
                gold,
                OracleConfig(
                    recall=_r, top1_error=_t, conf_noise=0.2, width=4,
                    seed=seed * 31 + 7,
                    pattern_aware=not args.independent,
                ),
            )

        payload = Harness(factory, arms, out_dir="/tmp/oracle-sweep").run(
            puzzles, ARMS, seeds=seeds, run_id=f"recall-{recall}"
        )
        by_arm: dict[str, list] = {}
        for record in payload["records"]:
            by_arm.setdefault(record["arm"], []).append(record)

        entry = {"recall": recall, "top1_error": top1_error, "arms": {}, "deltas": {}}
        for name in ARMS:
            group = by_arm.get(name, [])
            entry["arms"][name] = {
                "wcr": round(mean([r["scores"]["wcr"] for r in group]), 4),
                "lcr": round(mean([r["scores"]["lcr"] for r in group]), 4),
                "icr": round(mean([r["scores"]["icr"] for r in group]), 4),
                "exact": round(
                    mean([1.0 if r["scores"]["exact"] else 0.0 for r in group]), 4
                ),
                "cell_precision": round(
                    mean([r["scores"]["cell_precision"] for r in group]), 4
                ),
                "open_slots": round(
                    mean([len(r["solve"]["open_slots"]) for r in group]), 2
                ),
                "calls": round(mean([r["solve"]["calls"] for r in group]), 2),
            }
        for high, low, key in CONTRASTS:
            comparison = paired_bootstrap(
                [r["scores"]["wcr"] for r in by_arm.get(high, [])],
                [r["scores"]["wcr"] for r in by_arm.get(low, [])],
                resamples=args.resamples,
                seed=1,
            )
            entry["deltas"][key] = comparison.as_dict()
        sweep.append(entry)
        print(
            f"recall={recall:.2f}  "
            + "  ".join(f"{a}={entry['arms'][a]['wcr']:.3f}" for a in ARMS),
            file=sys.stderr,
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "what": (
                    "Offline sweep over synthetic candidate quality. recall is "
                    "the probability the correct answer appears at all; "
                    "top1_error is the probability it is not ranked first. "
                    "Measures how much model error the constraint and repair "
                    "layers absorb, with no API key and no network."
                ),
                "caveat": (
                    "Independent mode (--independent) answers each slot without "
                    "reading the repair pattern, so it cannot model the claimed "
                    "repair mechanism. Pattern-aware mode raises P(gold) as "
                    "confirmed letters accumulate. The a1-a0 contrast is only "
                    "meaningful in a live run either way."
                ),
                "pattern_aware": not args.independent,
                "puzzles": len(puzzles),
                "seeds": seeds,
                "arms": {name: arms[name].label for name in ARMS},
                "sweep": sweep,
            },
            fh,
            indent=2,
        )
        fh.write("\n")
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
