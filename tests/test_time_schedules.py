"""Tests for time-based and extinction schedules (FT, VT, RT, EXT)."""

from __future__ import annotations

import math
import random
import statistics

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from contingency.entities import Outcome, ResponseEvent
from contingency.errors import ScheduleConfigError, ScheduleStateError
from contingency.interfaces import Schedule
from contingency.schedules.time_based import EXT, FT, RT, VT

# ---------------------------------------------------------------------------
# FT — Fixed Time
# ---------------------------------------------------------------------------


class TestFTConstruction:
    def test_valid_interval(self) -> None:
        ft = FT(5.0)
        assert ft.interval == 5.0

    def test_int_interval_accepted(self) -> None:
        ft = FT(3)
        assert ft.interval == 3.0

    def test_zero_interval_rejected(self) -> None:
        with pytest.raises(ScheduleConfigError):
            FT(0)

    def test_negative_interval_rejected(self) -> None:
        with pytest.raises(ScheduleConfigError):
            FT(-1.0)

    def test_non_numeric_rejected(self) -> None:
        with pytest.raises(ScheduleConfigError):
            FT("5")  # type: ignore[arg-type]

    def test_bool_rejected(self) -> None:
        with pytest.raises(ScheduleConfigError):
            FT(True)  # type: ignore[arg-type]

    def test_is_schedule(self) -> None:
        assert isinstance(FT(1.0), Schedule)


class TestFTBehavior:
    def test_first_step_anchors_and_does_not_reinforce(self) -> None:
        ft = FT(1.0)
        assert ft.step(0.0) == Outcome()

    def test_reinforces_when_interval_elapsed(self) -> None:
        ft = FT(1.0)
        ft.step(0.0)
        result = ft.step(1.0)
        assert result.reinforced is True
        assert result.reinforcer is not None
        assert result.reinforcer.time == 1.0

    def test_does_not_reinforce_before_interval(self) -> None:
        ft = FT(1.0)
        ft.step(0.0)
        assert ft.step(0.5).reinforced is False
        assert ft.step(0.999).reinforced is False

    def test_response_is_ignored(self) -> None:
        """FT is response-independent: events do not trigger reinforcement."""
        ft = FT(1.0)
        ft.step(0.0)
        # Many responses, no time elapsed
        for t in [0.0, 0.1, 0.2, 0.3]:
            out = ft.step(t, event=ResponseEvent(time=t))
            assert out.reinforced is False

    def test_reinforcer_fires_without_any_events(self) -> None:
        ft = FT(1.0)
        ft.step(0.0)
        out = ft.step(1.0)  # no event supplied
        assert out.reinforced is True

    def test_response_events_do_not_affect_timing(self) -> None:
        """A run with events and without events produces the same reinforcement pattern."""
        ft_a = FT(1.0)
        ft_b = FT(1.0)
        times = [0.0, 0.3, 0.6, 1.0, 1.4, 1.7, 2.0]
        results_a = [ft_a.step(t).reinforced for t in times]
        results_b = [
            ft_b.step(t, event=ResponseEvent(time=t)).reinforced for t in times
        ]
        assert results_a == results_b

    def test_one_reinforcer_per_step_even_for_large_leap(self) -> None:
        """Stepping far past multiple intervals still yields one reinforcer."""
        ft = FT(1.0)
        ft.step(0.0)
        out = ft.step(10.0)
        assert out.reinforced is True
        # Next step should need a full fresh interval — not fire immediately.
        assert ft.step(10.5).reinforced is False
        assert ft.step(11.0).reinforced is True

    def test_repeated_reinforcements_on_grid(self) -> None:
        ft = FT(1.0)
        ft.step(0.0)
        fired = [ft.step(float(t)).reinforced for t in range(1, 6)]
        assert fired == [True, True, True, True, True]

    def test_reinforcement_exactly_at_boundary(self) -> None:
        ft = FT(2.5)
        ft.step(0.0)
        assert ft.step(2.4999).reinforced is False
        assert ft.step(2.5).reinforced is True

    def test_reset_clears_state(self) -> None:
        ft = FT(1.0)
        ft.step(0.0)
        ft.step(1.0)
        ft.reset()
        # After reset, first step anchors again — no reinforcement until one interval later
        assert ft.step(100.0).reinforced is False
        assert ft.step(101.0).reinforced is True


class TestFTStateErrors:
    def test_non_monotonic_now_raises(self) -> None:
        ft = FT(1.0)
        ft.step(1.0)
        with pytest.raises(ScheduleStateError):
            ft.step(0.5)

    def test_event_time_mismatch_raises(self) -> None:
        ft = FT(1.0)
        with pytest.raises(ScheduleStateError):
            ft.step(1.0, event=ResponseEvent(time=0.5))

    def test_event_time_equal_within_tol_ok(self) -> None:
        ft = FT(1.0)
        ft.step(0.0, event=ResponseEvent(time=1e-12))  # within 1e-9


# ---------------------------------------------------------------------------
# VT — Variable Time
# ---------------------------------------------------------------------------


class TestVTConstruction:
    def test_valid_params(self) -> None:
        vt = VT(10.0, n_intervals=12, seed=42)
        assert vt.mean_interval == 10.0
        assert vt.n_intervals == 12

    def test_default_n_intervals(self) -> None:
        assert VT(5.0).n_intervals == 12

    def test_zero_mean_rejected(self) -> None:
        with pytest.raises(ScheduleConfigError):
            VT(0.0)

    def test_negative_mean_rejected(self) -> None:
        with pytest.raises(ScheduleConfigError):
            VT(-1.0)

    def test_non_numeric_mean_rejected(self) -> None:
        with pytest.raises(ScheduleConfigError):
            VT("5")  # type: ignore[arg-type]

    def test_bool_mean_rejected(self) -> None:
        with pytest.raises(ScheduleConfigError):
            VT(True)  # type: ignore[arg-type]

    def test_non_int_n_intervals_rejected(self) -> None:
        with pytest.raises(ScheduleConfigError):
            VT(5.0, n_intervals=1.5)  # type: ignore[arg-type]

    def test_bool_n_intervals_rejected(self) -> None:
        with pytest.raises(ScheduleConfigError):
            VT(5.0, n_intervals=True)  # type: ignore[arg-type]

    def test_zero_n_intervals_rejected(self) -> None:
        with pytest.raises(ScheduleConfigError):
            VT(5.0, n_intervals=0)

    def test_is_schedule(self) -> None:
        assert isinstance(VT(1.0, seed=0), Schedule)


def _run_vt_for_reinforcers(
    vt: VT, total_time: float, dt: float = 0.01
) -> list[float]:
    """Drive a VT schedule with uniform ``dt`` steps and return reinforcer times."""
    times: list[float] = []
    steps = int(total_time / dt)
    for i in range(steps + 1):
        t = i * dt
        out = vt.step(t)
        if out.reinforced:
            assert out.reinforcer is not None
            times.append(out.reinforcer.time)
    return times


class TestVTBehavior:
    def test_is_response_independent(self) -> None:
        """VT ignores events for the reinforcement decision."""
        vt_a = VT(1.0, n_intervals=6, seed=0)
        vt_b = VT(1.0, n_intervals=6, seed=0)
        times = [i * 0.05 for i in range(400)]
        count_a = sum(vt_a.step(t).reinforced for t in times)
        count_b = sum(
            vt_b.step(t, event=ResponseEvent(time=t)).reinforced for t in times
        )
        assert count_a == count_b

    def test_determinism_with_seed(self) -> None:
        vt_a = VT(2.0, n_intervals=6, seed=123)
        vt_b = VT(2.0, n_intervals=6, seed=123)
        times_a = _run_vt_for_reinforcers(vt_a, 60.0)
        times_b = _run_vt_for_reinforcers(vt_b, 60.0)
        assert times_a == times_b

    def test_different_seeds_differ(self) -> None:
        vt_a = VT(2.0, n_intervals=6, seed=1)
        vt_b = VT(2.0, n_intervals=6, seed=2)
        times_a = _run_vt_for_reinforcers(vt_a, 60.0)
        times_b = _run_vt_for_reinforcers(vt_b, 60.0)
        assert times_a != times_b

    def test_reset_restores_sequence(self) -> None:
        vt = VT(2.0, n_intervals=6, seed=7)
        times_a = _run_vt_for_reinforcers(vt, 60.0)
        vt.reset()
        times_b = _run_vt_for_reinforcers(vt, 60.0)
        assert times_a == times_b

    def test_sequence_cycles(self) -> None:
        """When the n_intervals pool is exhausted a new pool is drawn; reinforcement continues."""
        vt = VT(1.0, n_intervals=4, seed=0)
        times = _run_vt_for_reinforcers(vt, 40.0)
        # Should have substantially more reinforcers than n_intervals=4
        assert len(times) > 10

    @settings(
        deadline=None,
        max_examples=5,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    @given(seed=st.integers(min_value=0, max_value=10_000))
    def test_mean_rate_matches_configured(self, seed: int) -> None:
        """Over many cycles, reinforcement rate approximates 1 / mean_interval."""
        mean = 2.0
        vt = VT(mean, n_intervals=12, seed=seed)
        total_time = 2400.0  # ~100 cycles
        dt = 0.01
        times = _run_vt_for_reinforcers(vt, total_time, dt=dt)
        # Empirical mean IRI should be close to configured mean
        iris = [t2 - t1 for t1, t2 in zip(times, times[1:], strict=False)]
        assert len(iris) > 50
        empirical_mean = statistics.fmean(iris)
        # Loose bound — FH intervals are deterministic per cycle, so with
        # 100 cycles the average is very close to the configured mean.
        assert abs(empirical_mean - mean) < 0.2


class TestVTStateErrors:
    def test_non_monotonic_now_raises(self) -> None:
        vt = VT(1.0, seed=0)
        vt.step(1.0)
        with pytest.raises(ScheduleStateError):
            vt.step(0.5)

    def test_event_time_mismatch_raises(self) -> None:
        vt = VT(1.0, seed=0)
        with pytest.raises(ScheduleStateError):
            vt.step(1.0, event=ResponseEvent(time=0.0))


# ---------------------------------------------------------------------------
# RT — Random Time
# ---------------------------------------------------------------------------


class TestRTConstruction:
    def test_valid_mean(self) -> None:
        rt = RT(5.0)
        assert rt.mean_interval == 5.0

    def test_int_mean_accepted(self) -> None:
        rt = RT(3)
        assert rt.mean_interval == 3.0

    def test_zero_mean_rejected(self) -> None:
        with pytest.raises(ScheduleConfigError):
            RT(0.0)

    def test_negative_mean_rejected(self) -> None:
        with pytest.raises(ScheduleConfigError):
            RT(-1.0)

    def test_non_numeric_mean_rejected(self) -> None:
        with pytest.raises(ScheduleConfigError):
            RT("5")  # type: ignore[arg-type]

    def test_bool_mean_rejected(self) -> None:
        with pytest.raises(ScheduleConfigError):
            RT(True)  # type: ignore[arg-type]

    def test_is_schedule(self) -> None:
        assert isinstance(RT(1.0, rng=random.Random(0)), Schedule)


def _run_rt_for_reinforcers(
    rt: RT, total_time: float, dt: float = 0.01
) -> list[float]:
    times: list[float] = []
    steps = int(total_time / dt)
    for i in range(steps + 1):
        t = i * dt
        out = rt.step(t)
        if out.reinforced:
            assert out.reinforcer is not None
            times.append(out.reinforcer.time)
    return times


class TestRTBehavior:
    def test_determinism_via_rng(self) -> None:
        rt_a = RT(1.0, rng=random.Random(42))
        rt_b = RT(1.0, rng=random.Random(42))
        times_a = _run_rt_for_reinforcers(rt_a, 200.0)
        times_b = _run_rt_for_reinforcers(rt_b, 200.0)
        assert times_a == times_b

    def test_reset_restores_draw_sequence(self) -> None:
        rt = RT(1.0, rng=random.Random(42))
        times_a = _run_rt_for_reinforcers(rt, 200.0)
        rt.reset()
        times_b = _run_rt_for_reinforcers(rt, 200.0)
        assert times_a == times_b

    def test_is_response_independent(self) -> None:
        rt_a = RT(1.0, rng=random.Random(0))
        rt_b = RT(1.0, rng=random.Random(0))
        times = [i * 0.01 for i in range(5000)]
        count_a = sum(rt_a.step(t).reinforced for t in times)
        count_b = sum(
            rt_b.step(t, event=ResponseEvent(time=t)).reinforced for t in times
        )
        assert count_a == count_b

    def test_iri_mean_approximates_configured(self) -> None:
        """Inter-reinforcement intervals have empirical mean near configured mean."""
        mean = 1.0
        rt = RT(mean, rng=random.Random(12345))
        dt = 0.01
        total_time = 5000.0
        times = _run_rt_for_reinforcers(rt, total_time, dt=dt)
        iris = [t2 - t1 for t1, t2 in zip(times, times[1:], strict=False)]
        assert len(iris) > 1000
        empirical_mean = statistics.fmean(iris)
        # With n > 1000, sample mean of an exponential should be within a few percent
        # (small positive bias possible because dt discretisation rounds up).
        assert abs(empirical_mean - mean) < 0.1

    def test_iri_variance_approximates_exponential(self) -> None:
        """For Exp(1/m): var == m^2 (memoryless hallmark)."""
        mean = 1.0
        rt = RT(mean, rng=random.Random(777))
        dt = 0.005
        total_time = 5000.0
        times = _run_rt_for_reinforcers(rt, total_time, dt=dt)
        iris = [t2 - t1 for t1, t2 in zip(times, times[1:], strict=False)]
        assert len(iris) > 1000
        empirical_var = statistics.variance(iris)
        # Exponential variance equals mean**2; allow loose tolerance.
        assert abs(empirical_var - mean * mean) < 0.2

    def test_unseeded_default_rng_works(self) -> None:
        """RT() with no rng argument still functions (non-deterministic)."""
        rt = RT(1.0)
        fired = 0
        for i in range(500):
            if rt.step(i * 0.1).reinforced:
                fired += 1
        # With mean_interval=1 over 50 time units, expect ~50 reinforcers
        assert fired > 10


class TestRTStateErrors:
    def test_non_monotonic_now_raises(self) -> None:
        rt = RT(1.0, rng=random.Random(0))
        rt.step(1.0)
        with pytest.raises(ScheduleStateError):
            rt.step(0.5)

    def test_event_time_mismatch_raises(self) -> None:
        rt = RT(1.0, rng=random.Random(0))
        with pytest.raises(ScheduleStateError):
            rt.step(1.0, event=ResponseEvent(time=0.9))


# ---------------------------------------------------------------------------
# EXT — Extinction
# ---------------------------------------------------------------------------


class TestEXTConstruction:
    def test_construct(self) -> None:
        EXT()

    def test_is_schedule(self) -> None:
        assert isinstance(EXT(), Schedule)


class TestEXTBehavior:
    def test_never_reinforces_across_many_steps_no_events(self) -> None:
        ext = EXT()
        for i in range(1000):
            out = ext.step(i * 0.1)
            assert out.reinforced is False
            assert out.reinforcer is None

    def test_never_reinforces_with_events(self) -> None:
        ext = EXT()
        for i in range(1000):
            t = i * 0.1
            out = ext.step(t, event=ResponseEvent(time=t))
            assert out.reinforced is False
            assert out.reinforcer is None

    def test_large_time_leap_no_reinforcement(self) -> None:
        ext = EXT()
        out = ext.step(1e9)
        assert out.reinforced is False

    def test_reset_is_noop_for_reinforcement(self) -> None:
        ext = EXT()
        for t in [0.0, 1.0, 2.0]:
            assert ext.step(t).reinforced is False
        ext.reset()
        for t in [0.0, 1.0, 2.0]:  # can replay times after reset
            assert ext.step(t).reinforced is False

    def test_reset_clears_monotonic_lock(self) -> None:
        """After reset, a previously-seen ``now`` should be acceptable again."""
        ext = EXT()
        ext.step(5.0)
        ext.reset()
        ext.step(0.0)  # should not raise


class TestEXTStateErrors:
    def test_non_monotonic_now_raises(self) -> None:
        ext = EXT()
        ext.step(1.0)
        with pytest.raises(ScheduleStateError):
            ext.step(0.5)

    def test_event_time_mismatch_raises(self) -> None:
        ext = EXT()
        with pytest.raises(ScheduleStateError):
            ext.step(1.0, event=ResponseEvent(time=0.0))


# ---------------------------------------------------------------------------
# Schedule protocol re-exports
# ---------------------------------------------------------------------------


class TestExports:
    def test_all_symbols_exported(self) -> None:
        from contingency import schedules

        for name in ("FT", "VT", "RT", "EXT"):
            assert hasattr(schedules, name)
            assert name in schedules.__all__


# Make hypothesis' math imports referenced (quiets unused-import warnings from linters).
_ = math
