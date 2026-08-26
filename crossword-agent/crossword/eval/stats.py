"""Statistics for comparing arms.

Every arm runs the *same* puzzles, so all comparisons are paired. That is the
whole reason the numbers here are usable at the sample sizes an LLM eval can
afford: between-puzzle variance dominates -- some grids are simply harder --
and pairing removes it entirely.

Two cautions this module exists to enforce:

* **Exact-solve is brutally low-powered.** It is one Bernoulli trial per
  puzzle. At n=40 a Wilson interval spans roughly +/-15 points, so an unpaired
  comparison of exact-solve rates between two arms is nearly meaningless.
  McNemar's test on the discordant pairs is the paired alternative and is far
  more sensitive.
* **Seed variance and puzzle variance are different things.** Sampling
  temperature makes the same arm score differently on the same puzzle. That is
  not the same uncertainty as "how does this arm do on a new puzzle", and
  averaging them into one error bar overstates confidence. Report them apart.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass
class Interval:
    point: float
    low: float
    high: float

    def as_dict(self) -> dict:
        return {
            "point": round(self.point, 6),
            "low": round(self.low, 6),
            "high": round(self.high, 6),
        }

    def __str__(self) -> str:
        return f"{self.point:.3f} [{self.low:.3f}, {self.high:.3f}]"


@dataclass
class PairedComparison:
    """The difference between two arms on the same puzzles."""

    delta: Interval
    p_better: float
    n: int
    wins: int
    losses: int
    ties: int

    @property
    def significant(self) -> bool:
        """95% CI of the difference excludes zero."""
        return self.delta.low > 0 or self.delta.high < 0

    def as_dict(self) -> dict:
        return {
            "delta": self.delta.as_dict(),
            "p_better": round(self.p_better, 4),
            "n": self.n,
            "wins": self.wins,
            "losses": self.losses,
            "ties": self.ties,
            "significant": self.significant,
        }


def paired_bootstrap(
    a: list[float],
    b: list[float],
    *,
    resamples: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> PairedComparison:
    """Bootstrap the mean difference ``a - b`` over paired observations.

    Resampling *puzzles* (not observations independently) is what keeps the
    pairing intact -- each resample draws whole (a_i, b_i) pairs.
    """
    if len(a) != len(b):
        raise ValueError(f"paired inputs must match: {len(a)} vs {len(b)}")
    if not a:
        return PairedComparison(Interval(0.0, 0.0, 0.0), 0.5, 0, 0, 0, 0)

    diffs = [x - y for x, y in zip(a, b)]
    point = sum(diffs) / len(diffs)

    rnd = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(resamples):
        total = 0.0
        for _ in range(n):
            total += diffs[rnd.randrange(n)]
        means.append(total / n)
    means.sort()
    low = means[max(0, int((alpha / 2) * resamples) - 1)]
    high = means[min(resamples - 1, int((1 - alpha / 2) * resamples))]
    p_better = sum(1 for m in means if m > 0) / resamples

    return PairedComparison(
        delta=Interval(point, low, high),
        p_better=p_better,
        n=n,
        wins=sum(1 for d in diffs if d > 0),
        losses=sum(1 for d in diffs if d < 0),
        ties=sum(1 for d in diffs if d == 0),
    )


def wilson(successes: int, n: int, *, z: float = 1.96) -> Interval:
    """Wilson score interval for a proportion.

    Preferred over the normal approximation because exact-solve rates land near
    0 or 1, where the normal interval runs outside [0, 1] and badly understates
    uncertainty.
    """
    if n == 0:
        return Interval(0.0, 0.0, 0.0)
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    spread = (z / denominator) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return Interval(p, max(0.0, center - spread), min(1.0, center + spread))


def mcnemar(a: list[bool], b: list[bool]) -> dict:
    """Paired test for two binary outcomes on the same items.

    Only the discordant pairs carry information: puzzles both arms solved, or
    both failed, say nothing about which is better. Uses the exact binomial
    test, which stays valid when discordant counts are small -- and with 40
    puzzles they usually are.
    """
    if len(a) != len(b):
        raise ValueError(f"paired inputs must match: {len(a)} vs {len(b)}")
    a_only = sum(1 for x, y in zip(a, b) if x and not y)
    b_only = sum(1 for x, y in zip(a, b) if y and not x)
    discordant = a_only + b_only
    if discordant == 0:
        return {
            "a_only": 0,
            "b_only": 0,
            "discordant": 0,
            "p_value": 1.0,
            "note": "no discordant pairs -- the arms agree on every puzzle",
        }
    smaller = min(a_only, b_only)
    # Two-sided exact binomial with p=0.5.
    tail = sum(math.comb(discordant, k) for k in range(smaller + 1))
    p_value = min(1.0, 2.0 * tail / (2**discordant))
    return {
        "a_only": a_only,
        "b_only": b_only,
        "discordant": discordant,
        "p_value": round(p_value, 6),
    }


def required_n(p1: float, p2: float, *, power: float = 0.8, alpha: float = 0.05) -> int:
    """Roughly how many puzzles are needed to tell two rates apart.

    Reported alongside exact-solve numbers so a null result is read as "not
    enough puzzles" rather than "no difference".
    """
    if p1 == p2:
        return 0
    z_alpha = 1.959964 if alpha == 0.05 else 2.575829
    z_power = 0.8416 if power == 0.8 else 1.2816
    p_bar = (p1 + p2) / 2
    numerator = (
        z_alpha * math.sqrt(2 * p_bar * (1 - p_bar))
        + z_power * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2
    return math.ceil(numerator / ((p1 - p2) ** 2))
