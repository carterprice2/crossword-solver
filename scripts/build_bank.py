#!/usr/bin/env python3
"""Derive the committed word/clue bank from two public-domain sources.

    dwyl/english-words                  words_alpha.txt   (Unlicense)
    matthewreagan/WebstersEnglishDictionary  dictionary.json
                                        (Webster's 1913, public domain via
                                         Project Gutenberg; the repo's GPL
                                         covers its Swift tool, not this text)

Intersecting a word list with a public-domain dictionary gives entries that are
both fillable and cluable without hand-authoring anything, and without shipping
a single copyrighted crossword clue.

    python3 scripts/build_bank.py --out corpus/bank/words.tsv

Sources are cloned to a cache directory on first run. Requires network; the
output is committed so nobody else needs to run this.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from crossword.normalize import classify_clue  # noqa: E402

SOURCES = {
    "english-words": "https://github.com/dwyl/english-words",
    "webster": "https://github.com/matthewreagan/WebstersEnglishDictionary",
}

MIN_LEN, MAX_LEN = 3, 8
MIN_CLUE, MAX_CLUE = 14, 88

#: Pure function words. Common enough to pass the frequency filter, but they
#: make poor entries and worse clues.
STOPWORDS = frozenset(
    """
    the and with for from that also who are not any his its but was all
    some their they have this she him her had been which when what where
    while would could should there then than them who whom whose you your
    our out its into onto upon such very much many most more each other
    """.split()
)


def definition_frequency(definitions: dict[str, str]) -> dict[str, int]:
    """Count how many dictionary entries use each word in their definition.

    This is a commonness proxy built from the dictionary itself. Common words
    define other words -- WATER appears in 2,203 definitions, ABDEST in none --
    so it separates real crossword vocabulary from Webster's deep obscurities
    without pulling in a frequency list under a restrictive license.
    """
    token = re.compile(r"[a-z]{3,}")
    counts: dict[str, int] = {}
    for text in definitions.values():
        if not isinstance(text, str):
            continue
        for word in set(token.findall(text.lower())):
            counts[word] = counts.get(word, 0) + 1
    return counts

#: Webster marks obsolete/rare/dialect senses inline. Those make miserable
#: clues -- unguessable for the wrong reason -- so drop them.
_REJECT_MARKERS = re.compile(
    r"\[(obs|archaic|r|colloq|scot|prov|poet|obsolete)\.?[^\]]*\]", re.IGNORECASE
)
#: Leading field-of-study tags: "(Mining)", "(Zoöl.)", "(Bot.)"
_FIELD_TAG = re.compile(r"^\s*\((?:[^)]{1,24})\)\s*")
_SENSE_NUM = re.compile(r"^\s*\d+\.\s*")
_DEFN = re.compile(r"^\s*Defn:\s*", re.IGNORECASE)
_WS = re.compile(r"\s+")
#: A capitalized word mid-sentence usually starts a citation ("Shak.", "Milton").
_CITATION = re.compile(r"\s+[A-Z][a-z]*\.\s*$")


def clone(cache: str, name: str, url: str) -> str:
    path = os.path.join(cache, name)
    if not os.path.isdir(os.path.join(path, ".git")):
        os.makedirs(cache, exist_ok=True)
        print(f"cloning {url} -> {path}", file=sys.stderr)
        subprocess.run(
            ["git", "clone", "--depth", "1", "--quiet", url, path], check=True
        )
    return path


def clean_definition(word: str, raw: str) -> str | None:
    """Condense a Webster entry into one short clue, or None if unusable."""
    text = raw.strip()
    if not text:
        return None

    # Keep only the first numbered sense.
    parts = re.split(r"(?m)(?=^\s*\d+\.\s)", text)
    first = next((p for p in parts if p.strip()), text)
    if "\n2." in text or re.search(r"(?m)^\s*2\.\s", text):
        first = re.split(r"(?m)^\s*2\.\s", text)[0]

    first = _SENSE_NUM.sub("", first)
    first = _DEFN.sub("", first)
    if _REJECT_MARKERS.search(first):
        return None
    first = _FIELD_TAG.sub("", first)

    # Cut at the first strong break: sentence end, semicolon, or dash gloss.
    first = _WS.sub(" ", first).strip()
    for sep in (";", " -- ", "--"):
        idx = first.find(sep)
        if idx > 0:
            first = first[:idx]
    match = re.search(r"\.(\s|$)", first)
    if match:
        first = first[: match.start()]
    first = _CITATION.sub("", first).strip().strip(",;: ")

    if not first or not (MIN_CLUE <= len(first) <= MAX_CLUE):
        return None
    # A definition that names its own headword gives the answer away. Check
    # shared prefixes too, or "To alien or alienate" survives as a clue for
    # ALIENE.
    lower = word.lower()
    for token in re.findall(r"[a-z]+", first.lower()):
        if token.startswith(lower[:4]) or lower.startswith(token[:4]):
            if len(token) >= 4 and len(lower) >= 4:
                return None
        if token == lower:
            return None
    if any(ch in first for ch in '"“”[]()'):
        return None
    if first.lower().startswith(("see ", "same as", "one who", "that which")):
        return None
    # Must read as a phrase, not a fragment of a longer sentence.
    if first[0].isalpha() is False:
        return None
    return first[0].upper() + first[1:]


def build(
    words_path: str, dict_path: str, limit: int, min_freq: int
) -> list[tuple[str, str, str, int]]:
    with open(dict_path, encoding="utf-8") as fh:
        definitions = json.load(fh)
    lookup = {k.upper(): v for k, v in definitions.items() if isinstance(v, str)}
    freq = definition_frequency(definitions)

    scored: list[tuple[int, str, str, str]] = []
    seen: set[str] = set()
    with open(words_path, encoding="utf-8") as fh:
        for line in fh:
            word = line.strip().upper()
            if not (MIN_LEN <= len(word) <= MAX_LEN) or not word.isalpha():
                continue
            if word in seen or word not in lookup:
                continue
            if word.lower() in STOPWORDS:
                continue
            count = freq.get(word.lower(), 0)
            if count < min_freq:
                continue
            clue = clean_definition(word, lookup[word])
            if clue is None:
                continue
            seen.add(word)
            scored.append((count, word, clue, classify_clue(clue, word)))

    # Most common first, so any cap keeps the best entries.
    scored.sort(key=lambda r: (-r[0], r[1]))
    rows = [(w, c, k, n) for n, w, c, k in scored]
    if limit and len(rows) > limit:
        by_len: dict[int, list[tuple[str, str, str, int]]] = {}
        for row in rows:
            by_len.setdefault(len(row[0]), []).append(row)
        # Even quota per length: fill difficulty is dominated by whichever
        # length is scarcest, not by the total.
        quota = max(1, limit // len(by_len))
        rows = [row for length in sorted(by_len) for row in by_len[length][:quota]]
    return sorted(rows, key=lambda r: (len(r[0]), r[0]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="corpus/bank/words.tsv")
    ap.add_argument("--cache", default=os.path.expanduser("~/.cache/crossword-agent"))
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="cap total rows, balanced across lengths (0 = keep all)",
    )
    ap.add_argument(
        "--words", help="local words_alpha.txt (skips cloning english-words)"
    )
    ap.add_argument("--dict", help="local dictionary.json (skips cloning webster)")
    ap.add_argument(
        "--min-freq",
        type=int,
        default=1,
        help="minimum number of Webster definitions a word must appear in "
        "(commonness filter; default 1 -- frequency is used for value\n        ordering during fill, not as a hard cut)",
    )
    args = ap.parse_args()

    words_path = args.words or os.path.join(
        clone(args.cache, "english-words", SOURCES["english-words"]), "words_alpha.txt"
    )
    dict_path = args.dict or os.path.join(
        clone(args.cache, "webster", SOURCES["webster"]), "dictionary.json"
    )

    rows = build(words_path, dict_path, args.limit, args.min_freq)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("# word\tclue\tclue_type\tfreq\n")
        fh.write("# Derived from dwyl/english-words (Unlicense) and Webster's\n")
        fh.write("# 1913 via matthewreagan/WebstersEnglishDictionary (public\n")
        fh.write("# domain). Regenerate with scripts/build_bank.py.\n")
        for word, clue, kind, freq in rows:
            fh.write(f"{word}\t{clue}\t{kind}\t{freq}\n")

    counts: dict[int, int] = {}
    for word, *_ in rows:
        counts[len(word)] = counts.get(len(word), 0) + 1
    print(f"wrote {len(rows)} entries to {args.out}", file=sys.stderr)
    print(f"by length: {dict(sorted(counts.items()))}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
