"""The word/clue bank and the positional index the filler searches over."""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_BANK = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "corpus",
    "bank",
    "words.tsv",
)


@dataclass(frozen=True)
class Entry:
    word: str
    clue: str
    clue_type: str
    #: How many Webster definitions use this word -- a commonness proxy. Used
    #: to order the filler's choices, never to exclude a word: a small bank
    #: cannot fill a dense grid, but an unordered large one fills it with
    #: obscurities. Ordering gets both.
    freq: int = 0


class Bank:
    """Words grouped by length, with a (length, position, letter) index.

    The index is what makes the filler's forward checking cheap: intersecting
    a few precomputed sets is far faster than scanning the word list per slot.
    """

    def __init__(self, entries: list[Entry]):
        self.entries = entries
        self.by_word = {e.word: e for e in entries}
        self.by_length: dict[int, list[str]] = {}
        for entry in entries:
            self.by_length.setdefault(len(entry.word), []).append(entry.word)
        self._index: dict[tuple[int, int, str], frozenset[str]] = {}
        staging: dict[tuple[int, int, str], set[str]] = {}
        for entry in entries:
            word = entry.word
            for i, ch in enumerate(word):
                staging.setdefault((len(word), i, ch), set()).add(word)
        self._index = {k: frozenset(v) for k, v in staging.items()}
        self._sets = {n: frozenset(w) for n, w in self.by_length.items()}
        self._union_cache: dict[tuple[int, int, frozenset[str]], frozenset[str]] = {}
        self._ordered: dict[int, tuple[str, ...]] = {}
        # Global commonness rank, most common first. A plain dict lookup is
        # the cheapest possible sort key on the filler's hot path.
        self.rank: dict[str, int] = {
            e.word: i
            for i, e in enumerate(sorted(entries, key=lambda e: (-e.freq, e.word)))
        }

    def __len__(self) -> int:
        return len(self.entries)

    def words(self, length: int) -> frozenset[str]:
        return self._sets.get(length, frozenset())

    def matching(self, pattern: str) -> frozenset[str]:
        """Every word matching a pattern like ``?E??R`` (``?`` = any letter)."""
        length = len(pattern)
        pool: frozenset[str] | None = None
        for i, ch in enumerate(pattern):
            if ch == "?":
                continue
            candidates = self._index.get((length, i, ch), frozenset())
            pool = candidates if pool is None else pool & candidates
            if not pool:
                return frozenset()
        return pool if pool is not None else self.words(length)

    def ordered(self, length: int) -> tuple[str, ...]:
        """Words of a given length, most common first.

        Precomputed once so the filler can walk this list and stop as soon as
        it has enough live candidates, instead of sorting a whole domain at
        every search node.
        """
        cached = self._ordered.get(length)
        if cached is None:
            words = self.by_length.get(length, [])
            cached = tuple(sorted(words, key=lambda w: (-self.freq_for(w), w)))
            self._ordered[length] = cached
        return cached

    def with_letter(self, length: int, position: int, letter: str) -> frozenset[str]:
        return self._index.get((length, position, letter), frozenset())

    def with_any_letter(
        self, length: int, position: int, letters: frozenset[str]
    ) -> frozenset[str]:
        """Every word of ``length`` whose ``position`` holds one of ``letters``.

        Cached, because the filler asks the same question at the same crossing
        thousands of times. Answering by set union beats filtering a domain in
        Python -- that difference is what makes an 11x11 fill finish.
        """
        key = (length, position, letters)
        hit = self._union_cache.get(key)
        if hit is None:
            hit = frozenset().union(
                *(self._index.get((length, position, ch), frozenset()) for ch in letters)
            ) if letters else frozenset()
            self._union_cache[key] = hit
        return hit

    def clue_for(self, word: str) -> str:
        entry = self.by_word.get(word)
        return entry.clue if entry else ""

    def clue_type_for(self, word: str) -> str:
        entry = self.by_word.get(word)
        return entry.clue_type if entry else "definition"

    def freq_for(self, word: str) -> int:
        entry = self.by_word.get(word)
        return entry.freq if entry else 0

    def length_counts(self) -> dict[int, int]:
        return {n: len(w) for n, w in sorted(self.by_length.items())}


def load_bank(path: str | None = None) -> Bank:
    path = path or DEFAULT_BANK
    entries: list[Entry] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            word, clue = parts[0].strip().upper(), parts[1].strip()
            kind = parts[2].strip() if len(parts) > 2 else "definition"
            try:
                freq = int(parts[3]) if len(parts) > 3 else 0
            except ValueError:
                freq = 0
            if word and clue:
                entries.append(Entry(word, clue, kind, freq))
    if not entries:
        raise ValueError(f"no entries loaded from {path}")
    return Bank(entries)
