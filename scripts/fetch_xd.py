#!/usr/bin/env python3
"""Fetch the real crossword corpus from xd.saul.pw.

    python3 scripts/fetch_xd.py --what puzzles

`xd-puzzles.zip` holds 6,000+ pre-1965 New York Times puzzles, from the era
now in the public domain. They are the only source of real 15x15 grids here --
the generator does not attempt them, because a 15x15 needs a constructor-grade
word list rather than the 8,500-entry public-domain bank we ship.

Nothing fetched is committed. The pre-1965 puzzles are public domain, but the
packaged archives are not ours to redistribute, so `corpus/xd/` is gitignored.

Two cautions that belong on every number computed from this corpus:

1. These puzzles are famous and long-published, so they are almost certainly in
   every model's training data. Run `crossword eval --arms ...` against the
   generated `mini` suite alongside them and compare -- that gap is the
   contamination signal.
2. Pre-1965 puzzles are stylistically unlike modern ones: heavy on obscure
   vocabulary, essentially no themes or wordplay. They are a different task,
   not an easier version of the same one. Keep the eras as separate strata.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import urllib.error
import urllib.request
import zipfile

SOURCES = {
    "puzzles": "https://xd.saul.pw/xd-puzzles.zip",
    "clues": "https://xd.saul.pw/xd-clues.zip",
}


def fetch(url: str, timeout: float) -> bytes:
    print(f"downloading {url}", file=sys.stderr)
    request = urllib.request.Request(url, headers={"User-Agent": "crossword-agent"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.URLError as exc:
        raise SystemExit(
            f"could not reach {url}: {exc}\n"
            "This step needs unrestricted network access. Everything else in "
            "the project -- tests, the offline demo, the oracle sweep -- runs "
            "without it."
        ) from exc


def extract(payload: bytes, out_dir: str, *, limit: int = 0, pattern: str = ".xd") -> int:
    os.makedirs(out_dir, exist_ok=True)
    written = 0
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for info in archive.infolist():
            if info.is_dir() or not info.filename.endswith(pattern):
                continue
            name = os.path.basename(info.filename)
            if not name:
                continue
            with archive.open(info) as source:
                data = source.read()
            with open(os.path.join(out_dir, name), "wb") as fh:
                fh.write(data)
            written += 1
            if limit and written >= limit:
                break
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--what", default="puzzles", help="puzzles, clues, or both")
    ap.add_argument("--out", default="corpus")
    ap.add_argument("--limit", type=int, default=0, help="stop after N files")
    ap.add_argument("--timeout", type=float, default=300.0)
    args = ap.parse_args()

    wanted = ["puzzles", "clues"] if args.what == "both" else [args.what]
    for name in wanted:
        if name not in SOURCES:
            raise SystemExit(f"unknown --what {name!r}; use puzzles, clues, or both")
        payload = fetch(SOURCES[name], args.timeout)
        target = os.path.join(args.out, "xd" if name == "puzzles" else "clues")
        count = extract(
            payload, target, limit=args.limit, pattern=".xd" if name == "puzzles" else ".tsv"
        )
        print(f"extracted {count} file(s) to {target}", file=sys.stderr)

    print(
        "\nNext: python3 -m crossword eval --suite xd --arms a0,a1,a2,a3 --limit 40\n"
        "Report generated and published puzzles as separate strata -- see the "
        "contamination note in REPORT.md.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
