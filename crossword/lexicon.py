"""English/crossword entry check: is this string a real word?

Used to reject nonsense like LFA or ETNT. This is membership, not retrieval:
we never search the list by clue. A word the model proposed is also allowed,
so names, brands, and abbreviations that are missing from the dictionary still
count when they came from the clue rather than from leftover letters.
"""

from __future__ import annotations

import os
from functools import lru_cache

SYSTEM_WORDS = "/usr/share/dict/words"
# Ubuntu/Debian ship the list under these names; `wamerican` also
# provides the /usr/share/dict/words symlink via update-alternatives.
SYSTEM_WORD_PATHS = (
    SYSTEM_WORDS,
    "/usr/share/dict/american-english",
    "/usr/share/dict/british-english",
)
VOWELS = frozenset("AEIOUY")


def _load_system_words(path: str | None = None) -> set[str]:
    words: set[str] = set()
    candidates = (path,) if path is not None else SYSTEM_WORD_PATHS
    for candidate in candidates:
        if not candidate or not os.path.isfile(candidate):
            continue
        with open(candidate, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                token = line.strip()
                if token.isalpha():
                    words.add(token.upper())
        if words:
            return words
    return words


def _load_bank_words() -> set[str]:
    try:
        from .gen.bank import load_bank

        return set(load_bank().by_word)
    except (OSError, ValueError, ImportError):
        return set()


@lru_cache(maxsize=1)
def word_set() -> frozenset[str]:
    return frozenset(_load_system_words() | _load_bank_words())


def _in_list(word: str, words: frozenset[str]) -> bool:
    if word in words:
        return True
    n = len(word)
    if n >= 4 and word.endswith("S") and word[:-1] in words:
        return True
    if n >= 5 and word.endswith("ES") and word[:-2] in words:
        return True
    if n >= 4 and word.endswith("ED") and (
        word[:-2] in words or word[:-1] in words
    ):
        return True
    if n >= 5 and word.endswith("ING"):
        stem = word[:-3]
        if stem in words or (stem + "E") in words:
            return True
    return False


def _is_abbrev(word: str) -> bool:
    """GDP, NFL, CSS: short strings with no vowel, treated as abbreviations."""
    return 2 <= len(word) <= 4 and not (set(word) & VOWELS)


def _is_compound(word: str, words: frozenset[str]) -> bool:
    """Two or three dictionary parts, at least one of length 3+ (BEER+BAR)."""
    n = len(word)
    if n < 5:
        return False
    for i in range(2, n - 1):
        a, b = word[:i], word[i:]
        if min(len(a), len(b)) < 2:
            continue
        if max(len(a), len(b)) < 3:
            continue
        if _in_list(a, words) and _in_list(b, words):
            return True
    for i in range(2, n - 3):
        for j in range(i + 2, n - 1):
            a, b, c = word[:i], word[i:j], word[j:]
            if min(len(a), len(b), len(c)) < 2:
                continue
            if max(len(a), len(b), len(c)) < 3:
                continue
            if _in_list(a, words) and _in_list(b, words) and _in_list(c, words):
                return True
    return False


def is_valid_entry(
    word: str,
    *,
    words: frozenset[str] | None = None,
    proposed: bool = False,
) -> bool:
    """True if ``word`` is a dictionary word, a short abbrev, a compound, or
    (when ``proposed``) something the model actually offered for this slot."""
    token = "".join(ch for ch in word.upper() if ch.isalpha())
    if len(token) < 2 or token != word.upper():
        return False
    if proposed:
        return True
    lexicon = words if words is not None else word_set()
    if _in_list(token, lexicon):
        return True
    if _is_abbrev(token):
        return True
    if _is_compound(token, lexicon):
        return True
    return False
