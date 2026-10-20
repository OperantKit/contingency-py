"""Tests for Fleshler-Hoffman VI/VR generators.

Reference: Fleshler & Hoffman (1962); Hantula (1991).
"""

from __future__ import annotations

import math

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from contingency.helpers.fleshler_hoffman import (
    _fh_progression,
    generate_intervals,
    generate_intervals_hantula1991,
    generate_ratios,
)


class TestProgression:
    def test_empty_for_zero_iterations(self) -> None:
        assert _fh_progression(30.0, 0) == []

    def test_last_element_matches_formula(self) -> None:
        n = 12
        v = 30.0
        raw = _fh_progression(v, n)
        expected_last = v * (1.0 + math.log(n))
        assert raw[-1] == pytest.approx(expected_last)


class TestGenerateIntervals:
    def test_empty_for_zero_iterations(self) -> None:
        assert generate_intervals(30.0, 0) == []

    def test_length_matches_n(self) -> None:
        out = generate_intervals(30.0, n=12, seed=0)
        assert len(out) == 12

    def test_mean_preserved(self) -> None:
        v = 30.0
        n = 12
        out = generate_intervals(v, n=n, seed=0)
        assert math.fsum(out) == pytest.approx(v * n)

    def test_deterministic_under_seed(self) -> None:
        a = generate_intervals(30.0, n=12, seed=42)
        b = generate_intervals(30.0, n=12, seed=42)
        assert a == b

    def test_different_seeds_differ(self) -> None:
        a = generate_intervals(30.0, n=12, seed=1)
        b = generate_intervals(30.0, n=12, seed=2)
        assert a != b

    @given(
        v=st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        n=st.integers(min_value=2, max_value=50),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(max_examples=50, deadline=None)
    def test_mean_preserved_property(self, v: float, n: int, seed: int) -> None:
        out = generate_intervals(v, n=n, seed=seed)
        assert math.fsum(out) == pytest.approx(v * n, rel=1e-9, abs=1e-9)
        assert len(out) == n


class TestGenerateRatios:
    def test_empty_for_zero_iterations(self) -> None:
        assert generate_ratios(30, 0) == []

    def test_mean_preserved(self) -> None:
        v = 30.0
        n = 12
        out = generate_ratios(v, n=n, seed=0)
        assert sum(out) == int(round(v * n))

    def test_all_values_positive(self) -> None:
        out = generate_ratios(30.0, n=12, seed=0)
        assert all(x >= 1 for x in out)

    def test_deterministic_under_seed(self) -> None:
        a = generate_ratios(30.0, n=12, seed=7)
        b = generate_ratios(30.0, n=12, seed=7)
        assert a == b

    @given(
        v=st.integers(min_value=2, max_value=200),
        n=st.integers(min_value=2, max_value=30),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(max_examples=50, deadline=None)
    def test_mean_preserved_property(self, v: int, n: int, seed: int) -> None:
        out = generate_ratios(float(v), n=n, seed=seed)
        assert sum(out) == v * n
        assert all(x >= 1 for x in out)
        assert len(out) == n


class TestHantula1991:
    def test_empty_for_zero_iterations(self) -> None:
        assert generate_intervals_hantula1991(30.0, 0) == []

    def test_length_matches_n(self) -> None:
        out = generate_intervals_hantula1991(30.0, n=12, seed=0)
        assert len(out) == 12

    def test_all_slots_filled(self) -> None:
        out = generate_intervals_hantula1991(30.0, n=12, seed=0)
        assert all(x > 0 for x in out)

    def test_deterministic_under_seed(self) -> None:
        a = generate_intervals_hantula1991(30.0, n=12, seed=99)
        b = generate_intervals_hantula1991(30.0, n=12, seed=99)
        assert a == b

    def test_mean_within_rounding_tolerance(self) -> None:
        # Hantula rounds each slot independently; the mean is only
        # approximately preserved (bounded by n/2 in the worst case).
        v = 30.0
        n = 12
        out = generate_intervals_hantula1991(v, n=n, seed=0)
        deviation = abs(sum(out) - v * n)
        assert deviation <= n / 2


# ---------------------------------------------------------------------------
# Shape of the distribution
#
# Fleshler, M., & Hoffman, H. S. (1962). A progression for generating
# variable-interval schedules. Journal of the Experimental Analysis of
# Behavior, 5(4), 529-530. https://doi.org/10.1901/jeab.1962.5-529
#
# Equation (2), with the constant [-ln(1 - p)]^-1 replaced by the VI mean T:
#
#     t_n = T * [1 + ln N + (N - n) ln(N - n) - (N - n + 1) ln(N - n + 1)]
#     t_N = T * (1 + ln N)                      (limit, n = N)
#
# t_n is the mean of the n-th of N equal-area classes of the exponential
# distribution with mean T, so class n spans
#
#     [-T ln(1 - (n - 1)/N), -T ln(1 - n/N)].
#
# Hantula, D. A. (1991). A simple BASIC program to generate values for
# variable-interval schedules of reinforcement. Journal of Applied Behavior
# Analysis, 24(4), 799-801. https://doi.org/10.1901/jaba.1991.24-799
#
# Hantula restates the same equation and places each t_n at a random
# unused slot. The BASIC listing stores t_n unrounded; the Swift original
# (and this port) rounds each slot to an integer.
#
# Generators only shuffle the progression, so the multiset of values is
# deterministic and is compared as a sorted list.
# ---------------------------------------------------------------------------


def _closed_form(t: float, n_terms: int) -> list[float]:
    """Equation (2) of Fleshler & Hoffman (1962), coded separately from the module.

    The float operations happen in the same order as ``_fh_progression``, so
    the two agree bit for bit; the Hantula rounding check relies on that.
    """
    big_n = float(n_terms)
    out: list[float] = []
    for n in range(1, n_terms + 1):
        if n == n_terms:
            out.append(t * (1.0 + math.log(big_n)))
            continue
        k = big_n - n
        out.append(t * (1.0 + math.log(big_n) + k * math.log(k) - (k + 1.0) * math.log(k + 1.0)))
    return out


# Values printed in Fleshler & Hoffman (1962, p. 530) for VI 30 s, N = 12.
# The printed values deviate from Equation (2) by up to 0.0023 s with no
# consistent sign (term 3: printed 7.020, equation 7.0223), which is larger
# than 3-decimal rounding; the tolerance below allows for that.
_PUBLISHED_VI30_N12 = [
    1.287,
    4.017,
    7.020,
    10.364,
    14.123,
    18.423,
    23.447,
    29.486,
    37.067,
    47.261,
    62.958,
    104.547,
]
_PUBLISHED_ABS_TOL = 0.003


class TestValueSetMatchesClosedForm:
    @pytest.mark.parametrize("seed", [0, 1, 42, 12345])
    def test_vi30_n12_matches_published_progression(self, seed: int) -> None:
        out = sorted(generate_intervals(30.0, n=12, seed=seed))
        assert out == pytest.approx(_PUBLISHED_VI30_N12, abs=_PUBLISHED_ABS_TOL)

    def test_closed_form_matches_published_progression(self) -> None:
        assert _closed_form(30.0, 12) == pytest.approx(_PUBLISHED_VI30_N12, abs=_PUBLISHED_ABS_TOL)

    @given(
        v=st.floats(min_value=0.5, max_value=600.0, allow_nan=False, allow_infinity=False),
        n=st.integers(min_value=1, max_value=200),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(max_examples=100, deadline=None)
    def test_intervals_are_the_closed_form_set(self, v: float, n: int, seed: int) -> None:
        # generate_intervals folds the float residual of sum(t_n) - v*n into one
        # term to keep the mean exact, so allow an error proportional to v*n.
        out = sorted(generate_intervals(v, n=n, seed=seed))
        assert out == pytest.approx(_closed_form(v, n), rel=1e-12, abs=1e-12 * v * n)

    @given(
        v=st.integers(min_value=1, max_value=300),
        n=st.integers(min_value=1, max_value=50),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(max_examples=100, deadline=None)
    def test_hantula1991_is_the_rounded_closed_form_set(self, v: int, n: int, seed: int) -> None:
        out = sorted(generate_intervals_hantula1991(float(v), n=n, seed=seed))
        expected = sorted(max(1, int(round(x))) for x in _closed_form(float(v), n))
        assert out == expected

    @given(
        v=st.integers(min_value=2, max_value=300),
        n=st.integers(min_value=2, max_value=50),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(max_examples=100, deadline=None)
    def test_ratios_track_the_closed_form_set(self, v: int, n: int, seed: int) -> None:
        # VR rounds each of t_1..t_{N-1} and replaces t_N with whatever keeps
        # the integer mean exact (the Swift remnant rule). When small v and
        # large n make that surplus < 1, the Swift fallback decrements terms
        # instead; its mean is covered by TestGenerateRatios.
        head = [max(1, int(round(x))) for x in _closed_form(float(v), n)[:-1]]
        surplus = v * n - sum(head)
        assume(surplus >= 1)
        out = sorted(generate_ratios(float(v), n=n, seed=seed))
        assert out == sorted([*head, surplus])


def _hazards(values: list[float], width: float, min_at_risk: int) -> list[float]:
    """Conditional probability of reinforcement in successive equal-width bins.

    For bin k = (k*width, (k+1)*width], the hazard is the number of intervals
    ending in the bin divided by the number of intervals still running at its
    start. Bins are evaluated while at least ``min_at_risk`` intervals remain.
    """
    hazards: list[float] = []
    k = 0
    while True:
        lo, hi = k * width, (k + 1) * width
        at_risk = sum(1 for x in values if x > lo)
        if at_risk < min_at_risk:
            return hazards
        ending = sum(1 for x in values if lo < x <= hi)
        hazards.append(ending / at_risk)
        k += 1


class TestConstantHazard:
    def test_each_term_lies_in_its_equal_area_class(self) -> None:
        v, n = 30.0, 200
        terms = _fh_progression(v, n)
        for i, t in enumerate(terms, start=1):
            lower = -v * math.log(1.0 - (i - 1) / n)
            upper = math.inf if i == n else -v * math.log(1.0 - i / n)
            assert lower < t < upper

    @pytest.mark.parametrize("width", [5.0, 15.0, 30.0])
    def test_hazard_is_flat_across_equal_bins(self, width: float) -> None:
        # Exponential with mean T: P(rf in a bin of width w | not yet) =
        # 1 - exp(-w/T), independent of the time since the last rf. The
        # hazard does not depend on order, so the seed does not matter.
        # Observed max deviation: 0.0054 / 0.0011 / 0.0025 for w = 5 / 15 / 30.
        v, n = 30.0, 1000
        expected = 1.0 - math.exp(-width / v)
        hazards = _hazards(generate_intervals(v, n=n, seed=0), width, min_at_risk=n // 10)
        assert len(hazards) >= 3
        assert max(abs(h - expected) for h in hazards) < 0.007

    def test_hazard_check_rejects_arithmetic_progression(self) -> None:
        # Fleshler & Hoffman's counterexample: an arithmetic progression with
        # the same mean makes P(rf) rise with time since rf.
        v, n, width = 30.0, 1000, 15.0
        arithmetic = [2.0 * v * (i + 0.5) / n for i in range(n)]
        expected = 1.0 - math.exp(-width / v)
        hazards = _hazards(arithmetic, width, min_at_risk=n // 10)
        assert hazards == sorted(hazards)
        assert max(abs(h - expected) for h in hazards) > 0.1
