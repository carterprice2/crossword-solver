#!/usr/bin/env python3
"""Write the May 28 2021 NYT Friday into corpus/nyt/ (gitignored)."""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from crossword.eval.nyt import write_corpus  # noqa: E402


def main() -> int:
    path = write_corpus()
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
