"""Tests for Fleshler-Hoffman VI/VR generators.

Reference: Fleshler & Hoffman (1962); Hantula (1991).
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
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
