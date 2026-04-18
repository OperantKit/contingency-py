"""Fleshler & Hoffman variable-schedule generators.

References
----------
- Fleshler, M., & Hoffman, H. S. (1962). A progression for generating
  variable-interval schedules. *Journal of the Experimental Analysis of
  Behavior*, 5(4), 529-530. https://doi.org/10.1901/jeab.1962.5-529
- Hantula, D. A. (1991). A simple BASIC program to generate values for
  variable-interval schedules of reinforcement. *Journal of Applied
  Behavior Analysis*, 24(4), 799-801.
  https://doi.org/10.1901/jaba.1991.24-799

Design notes
------------
* The original 1962 algorithm rounds each FH value to an integer and
  applies a remnant correction so that ``sum == v * n`` exactly. This
  port preserves that mean-preservation property for both the float and
  integer code paths.
* RNG is injected via ``random.Random(seed)`` so tests can pin a seed
  and obtain bit-identical schedules.
* Hantula's (1991) BASIC program used a GOTO loop for random placement.
  We reproduce the semantics without unbounded recursion.
"""

from __future__ import annotations

import math
import random

__all__ = [
    "generate_intervals",
    "generate_ratios",
    "generate_intervals_hantula1991",
]


def _fh_progression(v: float, n: int) -> list[float]:
    """Compute the raw Fleshler-Hoffman progression (unshuffled)."""
    if n == 0:
        return []
    intervals: list[float] = []
    dn = float(n)
    for m in range(1, n + 1):
        dm = float(m)
        if m == n:
            value = v * (1.0 + math.log(dn))
        else:
            s1 = (1.0 + math.log(dn)) + (dn - dm) * math.log(dn - dm)
            s2 = (dn - dm + 1.0) * math.log(dn - dm + 1.0)
            value = v * (s1 - s2)
        intervals.append(value)
    return intervals


def _shuffled(values: list[float], seed: int | None) -> list[float]:
    rng = random.Random(seed)
    out = list(values)
    rng.shuffle(out)
    return out


def generate_intervals(
    v: float,
    n: int = 12,
    seed: int | None = None,
) -> list[float]:
    """Generate a Variable-Interval (VI) schedule with mean ``v``.

    Parameters
    ----------
    v:
        Target mean interval (in whatever unit the caller uses).
    n:
        Number of intervals in one cycle. Defaults to 12, matching the
        Swift original and conventional practice.
    seed:
        Optional RNG seed. When supplied, output is deterministic.

    Returns
    -------
    list[float]
        ``n`` shuffled intervals whose arithmetic mean equals ``v``
        within float rounding error.
    """
    if n == 0:
        return []
    raw = _fh_progression(v, n)
    target_sum = v * n
    actual_sum = math.fsum(raw)
    raw[0] += target_sum - actual_sum
    return _shuffled(raw, seed)


def generate_ratios(
    v: float,
    n: int = 12,
    seed: int | None = None,
) -> list[int]:
    """Generate a Variable-Ratio (VR) schedule with mean ``v``.

    Each value is rounded to a positive integer (``>= 1``) and the final
    element absorbs any surplus so that the integer mean stays exactly
    ``v``. If the surplus would be non-positive, we walk the list from
    the tail decrementing entries ``>= 2`` until balance is restored —
    matching the Swift fallback.
    """
    if n == 0:
        return []
    floats = _fh_progression(v, n)
    rd: list[int] = [max(1, round(x)) for x in floats]
    target_total = int(round(v * n))
    head_sum = sum(rd[:-1])
    surplus = target_total - head_sum
    if surplus >= 1:
        rd[-1] = surplus
    else:
        rd[-1] = 1
        surplus -= 1
        for i in range(len(rd) - 1, -1, -1):
            if rd[i] >= 2:
                rd[i] -= 1
                surplus += 1
                if surplus >= 0:
                    break
    shuffled: list[float] = _shuffled([float(x) for x in rd], seed)
    return [int(x) for x in shuffled]


def generate_intervals_hantula1991(
    v: float,
    n: int = 12,
    seed: int | None = None,
) -> list[int]:
    """Hantula (1991) BASIC-program variant of the FH progression.

    The arithmetic is identical to :func:`generate_intervals`; the
    difference is in the random-placement loop, which Hantula expressed
    as a GOTO retry. We reproduce that semantics without unbounded
    recursion.
    """
    if n == 0:
        return []
    rng = random.Random(seed)
    rd: list[int] = [0] * n
    raw = _fh_progression(v, n)
    for m in range(n):
        value = int(round(raw[m]))
        while True:
            order = rng.randrange(0, n)
            if rd[order] == 0:
                rd[order] = value if value != 0 else 1
                break
    return rd
