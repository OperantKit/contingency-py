"""Tests for interval-family schedules: FI, VI, RI, LimitedHold."""

from __future__ import annotations

import math
import random
import statistics

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from contingency.entities import ResponseEvent
from contingency.errors import ScheduleConfigError, ScheduleStateError
from contingency.schedules.interval import FI, RI, VI, LimitedHold

# ---------------------------------------------------------------------------
# FI
# ---------------------------------------------------------------------------


class TestFIConfig:
    @pytest.mark.parametrize("bad", [0, -1, -0.001, -100.0])
    def test_non_positive_interval_rejected(self, bad: float) -> None:
        with pytest.raises(ScheduleConfigError):
            FI(bad)

    def test_positive_interval_accepted(self) -> None:
        FI(1.0)
        FI(0.001)
        FI(3600.0)


class TestFIBehavior:
    def test_tick_only_never_reinforces(self) -> None:
        sch = FI(10.0)
        for t in (0.0, 1.0, 5.0, 9.99, 10.0, 15.0, 100.0):
            out = sch.step(t)
            assert not out.reinforced
            assert out.reinforcer is None

    def test_response_before_interval_not_reinforced(self) -> None:
        sch = FI(10.0)
        for t in (1.0, 2.0, 5.0, 9.999):
            out = sch.step(t, ResponseEvent(time=t))
            assert not out.reinforced

    def test_first_response_after_interval_reinforces(self) -> None:
        sch = FI(10.0)
        sch.step(5.0, ResponseEvent(time=5.0))  # too early
        out = sch.step(10.5, ResponseEvent(time=10.5))
        assert out.reinforced
        assert out.reinforcer is not None
        assert out.reinforcer.time == 10.5

    def test_response_during_interval_does_not_reset_timer(self) -> None:
        sch = FI(10.0)
        # Many responses during the first interval.
        for t in (1.0, 3.0, 5.0, 7.0, 9.0):
            out = sch.step(t, ResponseEvent(time=t))
            assert not out.reinforced
        # A response at exactly the interval boundary qualifies.
        out = sch.step(10.0, ResponseEvent(time=10.0))
        assert out.reinforced

    def test_timer_restarts_from_reinforcement_time(self) -> None:
        sch = FI(10.0)
        out = sch.step(12.0, ResponseEvent(time=12.0))
        assert out.reinforced
        # New interval starts from now=12, so 20.0 is still too early.
        out = sch.step(20.0, ResponseEvent(time=20.0))
        assert not out.reinforced
        out = sch.step(22.5, ResponseEvent(time=22.5))
        assert out.reinforced

    def test_tick_past_interval_does_not_reinforce_without_response(
        self,
    ) -> None:
        sch = FI(5.0)
        out = sch.step(10.0)  # past interval but no response
        assert not out.reinforced
        out = sch.step(10.0, ResponseEvent(time=10.0))
        assert out.reinforced

    def test_reset_returns_to_post_construction(self) -> None:
        sch = FI(10.0)
        sch.step(15.0, ResponseEvent(time=15.0))  # reinforced
        sch.reset()
        # Back to: first reinforcer requires now >= 10 and a response.
        assert not sch.step(9.0, ResponseEvent(time=9.0)).reinforced
        assert sch.step(10.0, ResponseEvent(time=10.0)).reinforced


class TestFIStateErrors:
    def test_non_monotonic_time(self) -> None:
        sch = FI(5.0)
        sch.step(10.0)
        with pytest.raises(ScheduleStateError):
            sch.step(9.0)

    def test_event_time_mismatch(self) -> None:
        sch = FI(5.0)
        with pytest.raises(ScheduleStateError):
            sch.step(10.0, ResponseEvent(time=9.5))

    def test_monotonic_equality_ok(self) -> None:
        sch = FI(5.0)
        sch.step(10.0)
        sch.step(10.0)  # same time is fine


# ---------------------------------------------------------------------------
# VI
# ---------------------------------------------------------------------------


class TestVIConfig:
    @pytest.mark.parametrize("bad", [0, -0.5, -10])
    def test_non_positive_mean_rejected(self, bad: float) -> None:
        with pytest.raises(ScheduleConfigError):
            VI(bad)

    @pytest.mark.parametrize("bad_n", [0, -1, -10])
    def test_non_positive_n_intervals_rejected(self, bad_n: int) -> None:
        with pytest.raises(ScheduleConfigError):
            VI(5.0, n_intervals=bad_n)


def _run_vi_until_reinforcements(
    sch: VI, target: int, dt: float = 0.1
) -> list[float]:
    """Drive ``sch`` forward emitting a response every ``dt`` until
    ``target`` reinforcements have been observed. Returns the list
    of reinforcement times.
    """
    times: list[float] = []
    t = 0.0
    # Hard cap to keep pathological cases finite in test duration.
    max_steps = target * 10_000
    steps = 0
    while len(times) < target and steps < max_steps:
        t += dt
        out = sch.step(t, ResponseEvent(time=t))
        if out.reinforced:
            times.append(t)
        steps += 1
    assert len(times) == target, "VI failed to produce enough reinforcements"
    return times


class TestVIBehavior:
    def test_mean_rate_matches_mean_interval(self) -> None:
        mean = 5.0
        sch = VI(mean, n_intervals=12, seed=42)
        times = _run_vi_until_reinforcements(sch, target=200, dt=0.1)
        # Inter-reinforcement intervals
        iri = [b - a for a, b in zip(times, times[1:], strict=False)]
        empirical_mean = statistics.fmean(iri)
        # Since responses are continuous at dt=0.1 the empirical mean
        # should be very close to the configured mean (plus a small
        # positive bias from the discrete dt).
        assert abs(empirical_mean - mean) < 0.25

    def test_determinism_under_seed(self) -> None:
        sch_a = VI(3.0, n_intervals=8, seed=12345)
        sch_b = VI(3.0, n_intervals=8, seed=12345)
        times_a = _run_vi_until_reinforcements(sch_a, target=30, dt=0.05)
        times_b = _run_vi_until_reinforcements(sch_b, target=30, dt=0.05)
        assert times_a == times_b

    def test_different_seeds_diverge(self) -> None:
        sch_a = VI(3.0, n_intervals=8, seed=1)
        sch_b = VI(3.0, n_intervals=8, seed=2)
        times_a = _run_vi_until_reinforcements(sch_a, target=20, dt=0.05)
        times_b = _run_vi_until_reinforcements(sch_b, target=20, dt=0.05)
        assert times_a != times_b

    def test_reset_deterministic(self) -> None:
        sch = VI(4.0, n_intervals=10, seed=7)
        first = _run_vi_until_reinforcements(sch, target=25, dt=0.1)
        sch.reset()
        second = _run_vi_until_reinforcements(sch, target=25, dt=0.1)
        assert first == second

    def test_tick_only_never_reinforces(self) -> None:
        sch = VI(1.0, n_intervals=5, seed=0)
        for t in (0.1, 0.5, 1.0, 5.0, 10.0, 50.0):
            assert not sch.step(t).reinforced

    def test_sequence_regenerates_after_exhaustion(self) -> None:
        # n_intervals=3 forces the schedule to regenerate multiple
        # times over the course of the run.
        sch = VI(2.0, n_intervals=3, seed=11)
        times = _run_vi_until_reinforcements(sch, target=20, dt=0.05)
        # Just check we survived regeneration without crashing and
        # produced plausible rates.
        iri = [b - a for a, b in zip(times, times[1:], strict=False)]
        assert abs(statistics.fmean(iri) - 2.0) < 0.5

    def test_state_errors(self) -> None:
        sch = VI(2.0, seed=1)
        sch.step(1.0)
        with pytest.raises(ScheduleStateError):
            sch.step(0.5)
        with pytest.raises(ScheduleStateError):
            sch.step(2.0, ResponseEvent(time=1.5))


@settings(
    max_examples=8,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    mean=st.floats(min_value=1.0, max_value=10.0),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_vi_mean_rate_property(mean: float, seed: int) -> None:
    sch = VI(mean, n_intervals=12, seed=seed)
    times = _run_vi_until_reinforcements(sch, target=120, dt=0.1)
    iri = [b - a for a, b in zip(times, times[1:], strict=False)]
    empirical = statistics.fmean(iri)
    # Allow 25% tolerance for the small-sample empirical mean; with
    # n=120 intervals this is comfortably wide.
    assert abs(empirical - mean) / mean < 0.25


# ---------------------------------------------------------------------------
# RI
# ---------------------------------------------------------------------------


class TestRIConfig:
    @pytest.mark.parametrize("bad", [0, -1, -0.5])
    def test_non_positive_mean_rejected(self, bad: float) -> None:
        with pytest.raises(ScheduleConfigError):
            RI(bad)


def _run_ri_until_reinforcements(
    sch: RI, target: int, dt: float = 0.05
) -> list[float]:
    times: list[float] = []
    t = 0.0
    max_steps = target * 20_000
    steps = 0
    while len(times) < target and steps < max_steps:
        t += dt
        out = sch.step(t, ResponseEvent(time=t))
        if out.reinforced:
            times.append(t)
        steps += 1
    assert len(times) == target
    return times


class TestRIBehavior:
    def test_mean_rate_approximates_mean_interval(self) -> None:
        mean = 2.0
        sch = RI(mean, rng=random.Random(42))
        times = _run_ri_until_reinforcements(sch, target=500, dt=0.02)
        iri = [b - a for a, b in zip(times, times[1:], strict=False)]
        empirical = statistics.fmean(iri)
        # Law of large numbers on 500 samples — should be within ~10%.
        assert abs(empirical - mean) / mean < 0.15

    def test_exponential_distribution_chi_square(self) -> None:
        """Bin inter-reinforcement intervals and compare to exponential."""
        mean = 1.0
        sch = RI(mean, rng=random.Random(2024))
        # Use discrete responses dense enough that the response-side
        # discretization is negligible relative to the interval.
        times = _run_ri_until_reinforcements(sch, target=10_000, dt=0.005)
        iri = [b - a for a, b in zip(times, times[1:], strict=False)]
        # Bin edges at the quantiles of Exp(1): F^-1(p) = -ln(1-p).
        # Use 10 equal-probability bins so expected count per bin = N/10.
        n = len(iri)
        n_bins = 10
        edges = [-math.log(1.0 - k / n_bins) for k in range(1, n_bins)]
        expected = n / n_bins
        observed = [0] * n_bins
        for x in iri:
            for i, edge in enumerate(edges):
                if x <= edge:
                    observed[i] += 1
                    break
            else:
                observed[-1] += 1
        chi_sq = sum(
            (o - expected) ** 2 / expected for o in observed
        )
        # df = 9, 0.999 critical value ~= 27.88. We give generous
        # headroom to avoid flakiness across Python versions.
        assert chi_sq < 35.0, (
            f"chi-square={chi_sq:.2f} rejects Exp(1); "
            f"observed counts {observed}"
        )

    def test_determinism_under_seeded_rng(self) -> None:
        sch_a = RI(1.5, rng=random.Random(7))
        sch_b = RI(1.5, rng=random.Random(7))
        times_a = _run_ri_until_reinforcements(sch_a, target=50, dt=0.05)
        times_b = _run_ri_until_reinforcements(sch_b, target=50, dt=0.05)
        assert times_a == times_b

    def test_reset_restores_initial_rng_state(self) -> None:
        sch = RI(1.0, rng=random.Random(99))
        first = _run_ri_until_reinforcements(sch, target=30, dt=0.05)
        sch.reset()
        second = _run_ri_until_reinforcements(sch, target=30, dt=0.05)
        assert first == second

    def test_default_rng_is_non_deterministic(self) -> None:
        # Without a seeded RNG two instances should diverge (with
        # overwhelming probability). This documents the API contract.
        sch_a = RI(1.0)
        sch_b = RI(1.0)
        times_a = _run_ri_until_reinforcements(sch_a, target=10, dt=0.05)
        times_b = _run_ri_until_reinforcements(sch_b, target=10, dt=0.05)
        assert times_a != times_b

    def test_tick_only_never_reinforces(self) -> None:
        sch = RI(0.5, rng=random.Random(1))
        for t in (0.1, 1.0, 100.0):
            assert not sch.step(t).reinforced

    def test_state_errors(self) -> None:
        sch = RI(1.0, rng=random.Random(0))
        sch.step(5.0)
        with pytest.raises(ScheduleStateError):
            sch.step(4.5)
        with pytest.raises(ScheduleStateError):
            sch.step(6.0, ResponseEvent(time=5.999))


# ---------------------------------------------------------------------------
# LimitedHold
# ---------------------------------------------------------------------------


class TestLimitedHoldConfig:
    @pytest.mark.parametrize("bad", [0, -0.1, -10])
    def test_non_positive_hold_rejected(self, bad: float) -> None:
        with pytest.raises(ScheduleConfigError):
            LimitedHold(FI(5.0), bad)

    def test_requires_interval_family_inner(self) -> None:
        class Bogus:
            def step(self, now, event=None):  # pragma: no cover
                return None

            def reset(self):  # pragma: no cover
                return None

        with pytest.raises(ScheduleConfigError):
            LimitedHold(Bogus(), 2.0)


class TestLimitedHoldBehavior:
    def test_within_hold_response_reinforced(self) -> None:
        sch = LimitedHold(FI(10.0), hold=2.0)
        # Before interval: no reinforcement.
        assert not sch.step(5.0, ResponseEvent(time=5.0)).reinforced
        # Inside the hold window: reinforced.
        out = sch.step(10.5, ResponseEvent(time=10.5))
        assert out.reinforced
        assert out.reinforcer is not None
        assert out.reinforcer.time == 10.5

    def test_edge_of_hold_window_still_reinforced(self) -> None:
        sch = LimitedHold(FI(10.0), hold=2.0)
        out = sch.step(12.0, ResponseEvent(time=12.0))
        assert out.reinforced

    def test_past_hold_response_not_reinforced(self) -> None:
        sch = LimitedHold(FI(10.0), hold=2.0)
        # Tick past the hold window — opportunity withdrawn.
        sch.step(15.0)
        out = sch.step(15.0, ResponseEvent(time=15.0))
        assert not out.reinforced

    def test_schedule_re_arms_after_missed_opportunity(self) -> None:
        sch = LimitedHold(FI(10.0), hold=2.0)
        # Miss the first opportunity (response at 15.0, hold ended at 12.0).
        missed = sch.step(15.0, ResponseEvent(time=15.0))
        assert not missed.reinforced
        # A new interval is armed at now=15.0 → next opportunity at 25.0.
        assert not sch.step(20.0, ResponseEvent(time=20.0)).reinforced
        out = sch.step(25.5, ResponseEvent(time=25.5))
        assert out.reinforced

    def test_tick_without_response_during_hold_no_reinforcement(self) -> None:
        sch = LimitedHold(FI(10.0), hold=2.0)
        # Ticks inside the hold window don't trigger reinforcement;
        # they also don't close the window prematurely.
        assert not sch.step(10.0).reinforced
        assert not sch.step(11.0).reinforced
        out = sch.step(11.5, ResponseEvent(time=11.5))
        assert out.reinforced

    def test_response_before_interval_ignored(self) -> None:
        sch = LimitedHold(FI(10.0), hold=2.0)
        for t in (1.0, 5.0, 9.0):
            assert not sch.step(t, ResponseEvent(time=t)).reinforced

    def test_multiple_reinforcement_cycles(self) -> None:
        sch = LimitedHold(FI(5.0), hold=1.0)
        # First reinforcement inside the window.
        assert sch.step(5.5, ResponseEvent(time=5.5)).reinforced
        # New interval anchored at 5.5 → next arm at 10.5, hold ends at 11.5.
        assert not sch.step(10.0, ResponseEvent(time=10.0)).reinforced
        assert sch.step(11.0, ResponseEvent(time=11.0)).reinforced

    def test_reset_delegates_to_inner(self) -> None:
        inner = FI(10.0)
        sch = LimitedHold(inner, hold=2.0)
        assert sch.step(10.5, ResponseEvent(time=10.5)).reinforced
        sch.reset()
        # After reset the first opportunity is again at t=10.
        assert not sch.step(9.0, ResponseEvent(time=9.0)).reinforced
        assert sch.step(10.0, ResponseEvent(time=10.0)).reinforced

    def test_wraps_vi(self) -> None:
        inner = VI(2.0, n_intervals=5, seed=3)
        sch = LimitedHold(inner, hold=0.5)
        # Drive long enough to cover several intervals.
        t = 0.0
        reinforcements = 0
        for _ in range(2000):
            t += 0.05
            if sch.step(t, ResponseEvent(time=t)).reinforced:
                reinforcements += 1
        assert reinforcements > 0

    def test_wraps_ri(self) -> None:
        inner = RI(1.0, rng=random.Random(5))
        sch = LimitedHold(inner, hold=0.3)
        t = 0.0
        reinforcements = 0
        for _ in range(2000):
            t += 0.05
            if sch.step(t, ResponseEvent(time=t)).reinforced:
                reinforcements += 1
        assert reinforcements > 0

    def test_state_errors(self) -> None:
        sch = LimitedHold(FI(5.0), hold=1.0)
        sch.step(10.0)
        with pytest.raises(ScheduleStateError):
            sch.step(9.0)
        with pytest.raises(ScheduleStateError):
            sch.step(11.0, ResponseEvent(time=10.5))

    def test_vi_withdraw_and_rearm_path(self) -> None:
        """Force VI through _withdraw_and_rearm by missing an opportunity."""
        inner = VI(2.0, n_intervals=5, seed=13)
        sch = LimitedHold(inner, hold=0.1)
        # Tick far past any plausible VI interval, withdrawing once.
        sch.step(100.0)
        # Inner VI has been re-armed at 100.0 with a new interval.
        assert inner._arm_time > 100.0

    def test_ri_withdraw_and_rearm_path(self) -> None:
        """Force RI through _withdraw_and_rearm by missing an opportunity."""
        inner = RI(1.0, rng=random.Random(21))
        sch = LimitedHold(inner, hold=0.05)
        sch.step(50.0)
        assert inner._arm_time > 50.0
