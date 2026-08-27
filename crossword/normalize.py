"""Answer normalization and clue-type classification.

Clue types drive the per-stratum breakdown in the evaluation. The heuristics
here are deliberately shallow and cheap; they are applied identically to
generated and published puzzles so the strata mean the same thing in both.
Their agreement with hand labels is measured and reported rather than assumed
-- see ``crossword eval --audit-clue-types``.
"""

from __future__ import annotations

import re
import unicodedata

#: Clue-type labels, in the order they are tried.
DEFINITION = "definition"
FITB = "fitb"
ABBREV = "abbrev"
WORDPLAY = "wordplay"
PROPER = "proper"
CROSSWORDESE = "crosswordese"

CLUE_TYPES = (FITB, ABBREV, WORDPLAY, PROPER, CROSSWORDESE, DEFINITION)

_FITB_RE = re.compile(r"_{2,}|\.{3,}|\b___\b")
_ABBREV_RE = re.compile(
    r"\b(abbr|abbrev|acronym|initials|for short|briefly|in brief|"
    r"letters|init|symbol|sci\.|e\.g\.)\b\.?",
    re.IGNORECASE,
)
_ABBREV_TAIL_RE = re.compile(r"[:,]\s*(abbr|var)\.?\s*$", re.IGNORECASE)
_PROPER_RE = re.compile(
    r"\b(actor|actress|singer|author|poet|painter|composer|director|"
    r"president|senator|king|queen|saint|river|city|capital|county|state|"
    r"island|mountain|lake|sea|star of|player|team|novelist)\b",
    re.IGNORECASE,
)

#: Very short entries that recur constantly in American crosswords. Not a
#: judgment about the words -- just that they are solved by convention rather
#: than by reasoning, which is exactly what the stratum is meant to isolate.
CROSSWORDESE_WORDS = frozenset(
    """
    ADIT ALAI ALEE ANOA AREA ARIA ASEA ATOLL AVER EDDA EPEE ERNE ESNE ETUI
    EWER IOTA ISLE OBOE OGEE OLEO OLIO ORLE ORT ROC SLOE SMEE STOA TSAR ULNA
    UNAU UREA ESE ETA IRE ODE OLE ORE ANE ELS ENS ERS ESS OES
    """.split()
)


def normalize_answer(text: str) -> str:
    """Strip an answer down to the letters that actually go in the grid.

    Crossword answers are uppercase A-Z with no spaces or punctuation, so
    "St. Elmo's Fire" and "STELMOSFIRE" are the same answer. Accents are
    folded rather than dropped so "CAFE" and "CAFÉ" agree.
    """
    folded = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in folded if not unicodedata.combining(ch))
    return re.sub(r"[^A-Z]", "", stripped.upper())


def normalize_clue(text: str) -> str:
    """Collapse a clue to a comparable key (for dedupe and retrieval)."""
    lowered = unicodedata.normalize("NFKD", text.strip().lower())
    lowered = "".join(ch for ch in lowered if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9_ ]+", " ", lowered)).strip()


def classify_clue(clue: str, answer: str = "") -> str:
    """Best-guess clue type. Order matters: the most specific test wins."""
    text = clue.strip()
    if _FITB_RE.search(text):
        return FITB
    if _ABBREV_RE.search(text) or _ABBREV_TAIL_RE.search(text):
        return ABBREV
    # A trailing "?" is the standard American signal for a pun or misdirection.
    if text.endswith("?"):
        return WORDPLAY
    if answer and normalize_answer(answer) in CROSSWORDESE_WORDS:
        return CROSSWORDESE
    if _PROPER_RE.search(text):
        return PROPER
    # A capitalized word that is not the first word usually names something.
    words = text.split()
    if any(w[:1].isupper() and w[:1].isalpha() for w in words[1:]):
        return PROPER
    return DEFINITION


def length_bucket(length: int) -> str:
    if length <= 3:
        return "3"
    if length <= 5:
        return "4-5"
    if length <= 8:
        return "6-8"
    return "9+"
