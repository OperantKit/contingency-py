"""Tests for ratio-family reinforcement schedules (FR, CRF, VR, RR).

References
----------
Fleshler, M., & Hoffman, H. S. (1962). A progression for generating
variable-interval schedules. *Journal of the Experimental Analysis of
Behavior*, 5(4), 529-530. https://doi.org/10.1901/jeab.1962.5-529

Skinner, B. F. (1957). *Schedules of reinforcement* (with C. B.
Ferster). Appleton-Century-Crofts.
"""

from __future__ import annotations

import random

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from contingency import (
    Outcome,
    Reinforcer,
    ResponseEvent,
    Schedule,
    ScheduleConfigError,
    ScheduleStateError,
)
from contingency.helpers.fleshler_hoffman import generate_ratios
from contingency.schedules import CRF, FR, RR, VR


def _respond(schedule: Schedule, now: float) -> Outcome:
    """Helper: step the schedule with a response at ``now``."""
    return schedule.step(now, ResponseEvent(time=now))


class TestProtocolCompliance:
    def test_fr_satisfies_schedule_protocol(self) -> None:
        assert isinstance(FR(1), Schedule)

    def test_vr_satisfies_schedule_protocol(self) -> None:
        assert isinstance(VR(5.0, seed=0), Schedule)

    def test_rr_satisfies_schedule_protocol(self) -> None:
        assert isinstance(RR(0.5, rng=random.Random(0)), Schedule)


class TestFR:
    def test_reinforces_on_exact_nth_response(self) -> None:
        s = FR(5)
        outcomes = [_respond(s, float(i)) for i in range(1, 16)]
        reinforced = [i + 1 for i, o in enumerate(outcomes) if o.reinforced]
        assert reinforced == [5, 10, 15]

    def test_reinforcer_timestamp_equals_now(self) -> None:
        s = FR(3)
        _respond(s, 1.0)
        _respond(s, 2.0)
        o = _respond(s, 3.0)
        assert o.reinforced is True
        assert o.reinforcer is not None
        assert o.reinforcer.time == 3.0

    def test_fr1_reinforces_every_response(self) -> None:
        s = FR(1)
        for i in range(1, 11):
            o = _respond(s, float(i))
            assert o.reinforced is True
            assert o.reinforcer is not None
            assert o.reinforcer.time == float(i)

    def test_non_response_steps_never_reinforce(self) -> None:
        s = FR(3)
        for t in (0.1, 0.2, 0.3, 0.4, 0.5):
            o = s.step(t)
            assert o.reinforced is False
            assert o.reinforcer is None

    def test_empty_outcome_fields(self) -> None:
        s = FR(2)
        o = _respond(s, 1.0)
        assert o.reinforced is False
        assert o.reinforcer is None
        assert o.meta == {}

    def test_reset_restarts_count(self) -> None:
        s = FR(4)
        _respond(s, 1.0)
        _respond(s, 2.0)
        _respond(s, 3.0)  # count at 3
        s.reset()
        # With count restarted we need 4 more responses.
        for i in range(1, 4):
            o = _respond(s, float(i))
            assert o.reinforced is False
        o = _respond(s, 4.0)
        assert o.reinforced is True

    def test_reset_clears_last_time(self) -> None:
        s = FR(2)
        _respond(s, 100.0)
        s.reset()
        # After reset, stepping "earlier" again is valid.
        o = _respond(s, 0.0)
        assert o.reinforced is False

    def test_n_property(self) -> None:
        assert FR(7).n == 7

    def test_invalid_n_zero_raises(self) -> None:
        with pytest.raises(ScheduleConfigError):
            FR(0)

    def test_invalid_n_negative_raises(self) -> None:
        with pytest.raises(ScheduleConfigError):
            FR(-3)

    def test_invalid_n_non_int_raises(self) -> None:
        with pytest.raises(ScheduleConfigError):
            FR(1.5)  # type: ignore[arg-type]

    def test_invalid_n_bool_raises(self) -> None:
        # bool is a subclass of int but is semantically nonsense here.
        with pytest.raises(ScheduleConfigError):
            FR(True)  # type: ignore[arg-type]


class TestCRF:
    def test_crf_returns_fr_instance(self) -> None:
        s = CRF()
        assert isinstance(s, FR)
        assert s.n == 1

    def test_crf_reinforces_every_response(self) -> None:
        s = CRF()
        for i in range(1, 21):
            o = _respond(s, float(i))
            assert o.reinforced is True
            assert o.reinforcer is not None
            assert o.reinforcer.time == float(i)

    def test_crf_non_response_steps(self) -> None:
        s = CRF()
        assert s.step(1.0).reinforced is False


class TestVR:
    def test_invalid_mean_zero_raises(self) -> None:
        with pytest.raises(ScheduleConfigError):
            VR(0.0)

    def test_invalid_mean_negative_raises(self) -> None:
        with pytest.raises(ScheduleConfigError):
            VR(-2.0)

    def test_invalid_mean_non_numeric_raises(self) -> None:
        with pytest.raises(ScheduleConfigError):
            VR("fast")  # type: ignore[arg-type]

    def test_invalid_mean_bool_raises(self) -> None:
        with pytest.raises(ScheduleConfigError):
            VR(True)  # type: ignore[arg-type]

    def test_invalid_n_intervals_zero_raises(self) -> None:
        with pytest.raises(ScheduleConfigError):
            VR(5.0, n_intervals=0)

    def test_invalid_n_intervals_type_raises(self) -> None:
        with pytest.raises(ScheduleConfigError):
            VR(5.0, n_intervals=3.5)  # type: ignore[arg-type]

    def test_invalid_n_intervals_bool_raises(self) -> None:
        with pytest.raises(ScheduleConfigError):
            VR(5.0, n_intervals=True)  # type: ignore[arg-type]

    def test_mean_property(self) -> None:
        s = VR(7.0, n_intervals=12, seed=0)
        assert s.mean == 7.0
        assert s.n_intervals == 12

    def test_deterministic_under_seed(self) -> None:
        a = VR(8.0, n_intervals=12, seed=123)
        b = VR(8.0, n_intervals=12, seed=123)
        trace_a = _run_until_reinforcers(a, target=30)
        trace_b = _run_until_reinforcers(b, target=30)
        assert trace_a == trace_b

    def test_different_seeds_differ(self) -> None:
        a = VR(8.0, n_intervals=12, seed=1)
        b = VR(8.0, n_intervals=12, seed=2)
        trace_a = _run_until_reinforcers(a, target=30)
        trace_b = _run_until_reinforcers(b, target=30)
        assert trace_a != trace_b

    def test_reset_restarts_sequence_deterministically(self) -> None:
        s = VR(6.0, n_intervals=12, seed=42)
        trace_before = _run_until_reinforcers(s, target=20)
        s.reset()
        trace_after = _run_until_reinforcers(s, target=20)
        assert trace_before == trace_after

    def test_mean_over_many_cycles(self) -> None:
        mean = 10.0
        n_intervals = 12
        cycles = 10
        s = VR(mean, n_intervals=n_intervals, seed=0)
        counts = _run_until_reinforcers(s, target=cycles * n_intervals)
        expected_total = int(round(mean * n_intervals)) * cycles
        assert sum(counts) == expected_total

    def test_sequence_regenerates_after_exhaustion(self) -> None:
        # Use small n_intervals to force a reload within the test.
        s = VR(4.0, n_intervals=3, seed=0)
        counts = _run_until_reinforcers(s, target=20)
        assert len(counts) == 20
        assert all(c >= 1 for c in counts)

    @pytest.mark.parametrize("n_intervals", [3, 4, 5])
    def test_each_cycle_is_one_full_sequence_of_n_intervals(
        self, n_intervals: int
    ) -> None:
        # Each block of n_intervals realised requirements must be one
        # permutation of the Fleshler-Hoffman ratios for n_intervals
        # values, not for the default cycle length.
        mean = 10.0
        cycles = 4
        s = VR(mean, n_intervals=n_intervals, seed=5)
        counts = _run_until_reinforcers(s, target=cycles * n_intervals)
        expected = sorted(generate_ratios(mean, n=n_intervals, seed=0))
        for c in range(cycles):
            block = counts[c * n_intervals : (c + 1) * n_intervals]
            assert sorted(block) == expected

    def test_non_response_steps_do_not_reinforce(self) -> None:
        s = VR(5.0, n_intervals=12, seed=0)
        for t in (0.1, 0.2, 0.3):
            o = s.step(t)
            assert o.reinforced is False

    def test_reinforcer_timestamp(self) -> None:
        s = VR(2.0, n_intervals=12, seed=0)
        t = 0.0
        while True:
            t += 1.0
            o = _respond(s, t)
            if o.reinforced:
                assert o.reinforcer is not None
                assert o.reinforcer.time == t
                break

    @given(
        mean=st.integers(min_value=2, max_value=20),
        cycles=st.integers(min_value=10, max_value=15),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(max_examples=20, deadline=None)
    def test_mean_preservation_property(
        self, mean: int, cycles: int, seed: int
    ) -> None:
        n_intervals = 12
        s = VR(float(mean), n_intervals=n_intervals, seed=seed)
        counts = _run_until_reinforcers(s, target=cycles * n_intervals)
        # Each reload preserves the integer mean exactly (see
        # fleshler_hoffman.generate_ratios); therefore a number of
        # reinforcers that is an exact multiple of n_intervals yields
        # an exact mean equal to ``mean``.
        assert sum(counts) == mean * n_intervals * cycles


def _run_until_reinforcers(schedule: Schedule, target: int) -> list[int]:
    """Run the schedule with one response per tick until ``target``
    reinforcers have been delivered. Returns the response count
    between successive reinforcers (i.e. the realised ratio
    requirements)."""
    counts: list[int] = []
    responses_since_last: int = 0
    t = 0.0
    safety_cap = max(target * 1000, 10_000)
    ticks = 0
    while len(counts) < target:
        ticks += 1
        if ticks > safety_cap:
            raise AssertionError(
                f"Schedule failed to produce {target} reinforcers in {safety_cap} steps"
            )
        t += 1.0
        responses_since_last += 1
        outcome = _respond(schedule, t)
        if outcome.reinforced:
            counts.append(responses_since_last)
            responses_since_last = 0
    return counts


class TestRR:
    def test_invalid_probability_zero_raises(self) -> None:
        with pytest.raises(ScheduleConfigError):
            RR(0.0)

    def test_invalid_probability_negative_raises(self) -> None:
        with pytest.raises(ScheduleConfigError):
            RR(-0.1)

    def test_invalid_probability_gt_one_raises(self) -> None:
        with pytest.raises(ScheduleConfigError):
            RR(1.5)

    def test_invalid_probability_type_raises(self) -> None:
        with pytest.raises(ScheduleConfigError):
            RR("half")  # type: ignore[arg-type]

    def test_invalid_probability_bool_raises(self) -> None:
        with pytest.raises(ScheduleConfigError):
            RR(True)  # type: ignore[arg-type]

    def test_probability_one_reinforces_every_response(self) -> None:
        s = RR(1.0, rng=random.Random(0))
        for i in range(1, 21):
            o = _respond(s, float(i))
            assert o.reinforced is True
            assert o.reinforcer is not None
            assert o.reinforcer.time == float(i)

    def test_probability_property(self) -> None:
        assert RR(0.25, rng=random.Random(0)).probability == 0.25

    def test_monte_carlo_convergence(self) -> None:
        p = 0.3
        trials = 10_000
        s = RR(p, rng=random.Random(12345))
        hits = 0
        for i in range(1, trials + 1):
            o = _respond(s, float(i))
            if o.reinforced:
                hits += 1
        observed = hits / trials
        assert abs(observed - p) < 0.05 * p + 0.005

    def test_deterministic_under_seeded_rng(self) -> None:
        a = RR(0.4, rng=random.Random(777))
        b = RR(0.4, rng=random.Random(777))
        trace_a = [_respond(a, float(i)).reinforced for i in range(1, 201)]
        trace_b = [_respond(b, float(i)).reinforced for i in range(1, 201)]
        assert trace_a == trace_b

    def test_reset_restores_draw_sequence(self) -> None:
        s = RR(0.4, rng=random.Random(777))
        trace_before = [_respond(s, float(i)).reinforced for i in range(1, 101)]
        s.reset()
        trace_after = [_respond(s, float(i)).reinforced for i in range(1, 101)]
        assert trace_before == trace_after

    def test_non_response_steps_do_not_reinforce(self) -> None:
        s = RR(1.0, rng=random.Random(0))
        for t in (0.1, 0.2, 0.3):
            assert s.step(t).reinforced is False

    def test_default_rng_constructs(self) -> None:
        # Smoke test: without an explicit rng we still produce valid
        # boolean outcomes.
        s = RR(0.5)
        for i in range(1, 11):
            o = _respond(s, float(i))
            assert isinstance(o, Outcome)

    def test_reinforcer_timestamp(self) -> None:
        s = RR(1.0, rng=random.Random(0))
        o = _respond(s, 42.0)
        assert o.reinforced is True
        assert o.reinforcer == Reinforcer(time=42.0)


class TestStateErrors:
    def test_fr_rejects_non_monotonic_time(self) -> None:
        s = FR(2)
        _respond(s, 5.0)
        with pytest.raises(ScheduleStateError):
            _respond(s, 4.0)

    def test_fr_allows_equal_time(self) -> None:
        s = FR(2)
        _respond(s, 5.0)
        # Equal timestamps are valid (within tolerance).
        o = _respond(s, 5.0)
        assert o.reinforced is True

    def test_vr_rejects_non_monotonic_time(self) -> None:
        s = VR(3.0, seed=0)
        _respond(s, 5.0)
        with pytest.raises(ScheduleStateError):
            _respond(s, 4.0)

    def test_rr_rejects_non_monotonic_time(self) -> None:
        s = RR(0.5, rng=random.Random(0))
        _respond(s, 5.0)
        with pytest.raises(ScheduleStateError):
            _respond(s, 4.0)

    def test_fr_rejects_mismatched_event_time(self) -> None:
        s = FR(2)
        with pytest.raises(ScheduleStateError):
            s.step(1.0, ResponseEvent(time=0.5))

    def test_vr_rejects_mismatched_event_time(self) -> None:
        s = VR(3.0, seed=0)
        with pytest.raises(ScheduleStateError):
            s.step(1.0, ResponseEvent(time=0.9))

    def test_rr_rejects_mismatched_event_time(self) -> None:
        s = RR(0.5, rng=random.Random(0))
        with pytest.raises(ScheduleStateError):
            s.step(1.0, ResponseEvent(time=2.0))

    def test_tolerance_accepts_tiny_drift(self) -> None:
        s = FR(1)
        # Drift within _TIME_TOL is accepted.
        o = s.step(1.0, ResponseEvent(time=1.0 + 1e-10))
        assert o.reinforced is True
