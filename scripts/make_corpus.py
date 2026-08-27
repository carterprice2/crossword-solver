#!/usr/bin/env python3
"""Generate the committed mini corpus.

Templates are searched and fill-tested together, because a mask that validates
structurally can still be unfillable against our bank -- and an unfillable mask
fails slowly, which is the worst way to find out. Only masks that fill at
several independent seeds are kept, and both the mask and the puzzles it
produced are written out.

    python3 scripts/make_corpus.py --out corpus

Deterministic: the same seeds produce the same puzzles, so the committed corpus
can be regenerated and diffed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from crossword.gen.bank import load_bank  # noqa: E402
from crossword.gen.fill import FillError, build_puzzle  # noqa: E402
from crossword.gen.grids import dump_template, search_templates  # noqa: E402
from crossword.normalize import classify_clue  # noqa: E402
from crossword.xd import dump_xd  # noqa: E402

#: (size, max_run, target_density, puzzles_to_keep)
PLAN = [
    (7, 5, 0.22, 4),
    (9, 6, 0.18, 4),
    (11, 6, 0.20, 4),
]


def try_template(grid, bank, seeds, *, time_limit, min_entry):
    """Fill a mask at several seeds. Returns the puzzles that succeeded."""
    puzzles = []
    for seed in seeds:
        try:
            puzzle, stats = build_puzzle(
                grid,
                bank,
                seed=seed,
                min_entry=min_entry,
                time_limit=time_limit,
                restarts=1,
            )
        except FillError:
            return []  # a template must be reliable, not lucky
        puzzles.append((puzzle, stats))
    return puzzles


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="corpus")
    ap.add_argument("--bank")
    ap.add_argument("--time-limit", type=float, default=30.0)
    ap.add_argument("--templates-per-size", type=int, default=12)
    ap.add_argument("--min-entry", type=int, default=3)
    args = ap.parse_args()

    bank = load_bank(args.bank)
    print(f"bank: {len(bank)} entries {bank.length_counts()}", file=sys.stderr)

    mini_dir = os.path.join(args.out, "mini")
    grid_dir = os.path.join(args.out, "grids")
    os.makedirs(mini_dir, exist_ok=True)
    os.makedirs(grid_dir, exist_ok=True)

    manifest: list[dict] = []
    for size, max_run, density, want in PLAN:
        candidates = search_templates(
            size,
            max_run=max_run,
            target_density=density,
            seed=size,
            attempts=8000,
            want=args.templates_per_size,
        )
        print(
            f"{size}x{size}: {len(candidates)} candidate template(s)", file=sys.stderr
        )
        kept = 0
        for index, grid in enumerate(candidates):
            if kept >= want:
                break
            seeds = [0, 1]
            results = try_template(
                grid, bank, seeds, time_limit=args.time_limit, min_entry=args.min_entry
            )
            if not results:
                continue
            template_name = f"{size}x{size}-{index:02d}"
            with open(os.path.join(grid_dir, f"{template_name}.txt"), "w") as fh:
                fh.write(dump_template(grid))
            for seed, (puzzle, stats) in zip(seeds, results):
                if kept >= want:
                    break
                puzzle_id = f"mini-{size:02d}-{index:02d}-{seed}"
                puzzle.metadata["Title"] = f"Generated {size}x{size} ({puzzle_id})"
                puzzle.metadata["id"] = puzzle_id
                path = os.path.join(mini_dir, f"{puzzle_id}.xd")
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(dump_xd(puzzle))
                types: dict[str, int] = {}
                for slot in puzzle.slots:
                    kind = classify_clue(slot.clue, slot.gold or "")
                    types[kind] = types.get(kind, 0) + 1
                manifest.append(
                    {
                        "id": puzzle_id,
                        "path": os.path.relpath(path, args.out),
                        "size": size,
                        "template": template_name,
                        "seed": seed,
                        "slots": len(puzzle.slots),
                        "block_density": round(grid.block_density, 4),
                        "max_slot_length": max(len(s.cells) for s in puzzle.slots),
                        "clue_types": dict(sorted(types.items())),
                        "provenance": "generated",
                        "fill_seconds": round(stats.seconds, 3),
                        "fill_nodes": stats.nodes,
                    }
                )
                kept += 1
                print(f"  wrote {puzzle_id} ({stats.seconds:.1f}s)", file=sys.stderr)

    payload = {
        "puzzles": manifest,
        "bank": {
            "entries": len(bank),
            "by_length": bank.length_counts(),
            "sources": [
                {
                    "name": "dwyl/english-words",
                    "file": "words_alpha.txt",
                    "license": "Unlicense (public domain)",
                },
                {
                    "name": "matthewreagan/WebstersEnglishDictionary",
                    "file": "dictionary.json",
                    "license": "Webster's 1913, public domain via Project Gutenberg",
                },
            ],
        },
        "note": (
            "Generated puzzles, never published anywhere. They are the "
            "contamination-free slice of the evaluation: no model can have "
            "memorized them. Clues are condensed Webster 1913 definitions, so "
            "they are definitional rather than NYT-style -- this set is a "
            "control, not a difficulty proxy for real crosswords."
        ),
    }
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    print(f"\n{len(manifest)} puzzles -> {args.out}", file=sys.stderr)
    return 0 if manifest else 1


if __name__ == "__main__":
    raise SystemExit(main())
