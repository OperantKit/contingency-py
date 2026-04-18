"""Tests for the Progressive-Ratio schedule and step-function helpers.

References
----------
Hodos, W. (1961). Progressive ratio as a measure of reward strength.
*Science*, 134(3483), 943-944.
https://doi.org/10.1126/science.134.3483.943

Hursh, S. R. (1980). Economic concepts for the analysis of behavior.
*Journal of the Experimental Analysis of Behavior*, 34(2), 219-238.
https://doi.org/10.1901/jeab.1980.34-219

Richardson, N. R., & Roberts, D. C. S. (1996). Progressive ratio
schedules in drug self-administration studies in rats: A method to
evaluate reinforcing efficacy. *Journal of Neuroscience Methods*,
66(1), 1-11. https://doi.org/10.1016/0165-0270(95)00153-0
"""

from __future__ import annotations

import pytest

from contingency import (
    Outcome,
    ResponseEvent,
    Schedule,
    ScheduleConfigError,
    ScheduleStateError,
)
from contingency.schedules import (
    ProgressiveRatio,
    arithmetic,
    geometric,
    richardson_roberts,
)


def _respond(schedule: Schedule, now: float) -> Outcome:
    """Helper: step the schedule with a response at ``now``."""
    return schedule.step(now, ResponseEvent(time=now))


# ---------------------------------------------------------------------------
# Step function helpers
# ---------------------------------------------------------------------------


class TestArithmetic:
    def test_default_start1_step1_series(self) -> None:
        fn = arithmetic()  # start=1, step=1
        assert [fn(n) for n in range(5)] == [1, 2, 3, 4, 5]

    def test_start2_step3(self) -> None:
        fn = arithmetic(start=2, step=3)
        assert [fn(n) for n in range(5)] == [2, 5, 8, 11, 14]

    def test_returns_positive_int(self) -> None:
        fn = arithmetic(start=7, step=4)
        value = fn(0)
        assert isinstance(value, int)
        assert value == 7

    def test_invalid_start_zero_raises(self) -> None:
        with pytest.raises(ScheduleConfigError):
            arithmetic(start=0)

    def test_invalid_start_negative_raises(self) -> None:
        with pytest.raises(ScheduleConfigError):
            arithmetic(start=-3)

    def test_invalid_step_zero_raises(self) -> None:
        with pytest.raises(ScheduleConfigError):
            arithmetic(start=1, step=0)

    def test_invalid_step_negative_raises(self) -> None:
        with pytest.raises(ScheduleConfigError):
            arithmetic(start=1, step=-1)

    def test_invalid_start_type_raises(self) -> None:
        with pytest.raises(ScheduleConfigError):
            arithmetic(start=1.5)  # type: ignore[arg-type]

    def test_invalid_step_type_raises(self) -> None:
        with pytest.raises(ScheduleConfigError):
            arithmetic(step="one")  # type: ignore[arg-type]

    def test_invalid_start_bool_raises(self) -> None:
        with pytest.raises(ScheduleConfigError):
            arithmetic(start=True)  # type: ignore[arg-type]

    def test_invalid_step_bool_raises(self) -> None:
        with pytest.raises(ScheduleConfigError):
            arithmetic(step=True)  # type: ignore[arg-type]


class TestGeometric:
    def test_doubling_series(self) -> None:
        fn = geometric(start=1, ratio=2.0)
        assert [fn(n) for n in range(5)] == [1, 2, 4, 8, 16]

    def test_start4_ratio_1_5_rounded(self) -> None:
        fn = geometric(start=4, ratio=1.5)
        # 4, 6, 9, 13.5->14, 20.25->20
        assert [fn(n) for n in range(5)] == [4, 6, 9, 14, 20]

    def test_integer_ratio_accepted(self) -> None:
        fn = geometric(start=1, ratio=3)
        assert [fn(n) for n in range(4)] == [1, 3, 9, 27]

    def test_invalid_start_zero_raises(self) -> None:
        with pytest.raises(ScheduleConfigError):
            geometric(start=0, ratio=2.0)

    def test_invalid_start_type_raises(self) -> None:
        with pytest.raises(ScheduleConfigError):
            geometric(start=2.0, ratio=2.0)  # type: ignore[arg-type]

    def test_invalid_start_bool_raises(self) -> None:
        with pytest.raises(ScheduleConfigError):
            geometric(start=True, ratio=2.0)  # type: ignore[arg-type]

    def test_invalid_ratio_one_raises(self) -> None:
        with pytest.raises(ScheduleConfigError):
            geometric(start=1, ratio=1.0)

    def test_invalid_ratio_below_one_raises(self) -> None:
        with pytest.raises(ScheduleConfigError):
            geometric(start=1, ratio=0.5)

    def test_invalid_ratio_type_raises(self) -> None:
        with pytest.raises(ScheduleConfigError):
            geometric(start=1, ratio="x")  # type: ignore[arg-type]

    def test_invalid_ratio_bool_raises(self) -> None:
        with pytest.raises(ScheduleConfigError):
            geometric(start=1, ratio=True)  # type: ignore[arg-type]


class TestRichardsonRoberts:
    def test_first_five_values(self) -> None:
        fn = richardson_roberts()
        assert [fn(n) for n in range(5)] == [1, 2, 4, 6, 9]

    def test_full_canonical_series(self) -> None:
        fn = richardson_roberts()
        expected = [
            1, 2, 4, 6, 9, 12, 16, 20, 25, 32,
            40, 50, 62, 77, 95, 118, 145, 178, 219, 268,
            328, 402, 492, 603, 737, 901, 1102, 1347, 1647, 2012,
        ]
        assert [fn(n) for n in range(30)] == expected

    def test_returns_positive_int(self) -> None:
        fn = richardson_roberts()
        for n in range(30):
            value = fn(n)
            assert isinstance(value, int)
            assert value >= 1

    def test_monotonically_increasing(self) -> None:
        fn = richardson_roberts()
        values = [fn(n) for n in range(30)]
        assert all(a < b for a, b in zip(values, values[1:]))

    def test_extrapolation_beyond_index_29_continues_geometrically(self) -> None:
        fn = richardson_roberts()
        # Series ends with 1647, 2012. Ratio ≈ 2012/1647 ≈ 1.2217.
        last_ratio = 2012 / 1647
        v30 = fn(30)
        v31 = fn(31)
        # Geometric extrapolation from 2012.
        assert v30 == round(2012 * last_ratio)
        assert v31 == round(2012 * last_ratio**2)
        assert v30 > 2012
        assert v31 > v30

    def test_extrapolation_far_index_still_positive(self) -> None:
        fn = richardson_roberts()
        for n in (40, 60, 100):
            value = fn(n)
            assert isinstance(value, int)
            assert value > fn(n - 1)

    def test_negative_index_raises(self) -> None:
        fn = richardson_roberts()
        with pytest.raises(ScheduleConfigError):
            fn(-1)


# ---------------------------------------------------------------------------
# ProgressiveRatio — protocol compliance
# ---------------------------------------------------------------------------


class TestProtocolCompliance:
    def test_satisfies_schedule_protocol(self) -> None:
        assert isinstance(ProgressiveRatio(arithmetic()), Schedule)


# ---------------------------------------------------------------------------
# ProgressiveRatio — construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_non_callable_step_fn_raises(self) -> None:
        with pytest.raises(ScheduleConfigError):
            ProgressiveRatio(step_fn=42)  # type: ignore[arg-type]

    def test_non_callable_none_raises(self) -> None:
        with pytest.raises(ScheduleConfigError):
            ProgressiveRatio(step_fn=None)  # type: ignore[arg-type]

    def test_current_requirement_initially_matches_step_fn_zero(self) -> None:
        pr = ProgressiveRatio(arithmetic(start=3, step=2))
        assert pr.current_requirement == 3

    def test_reinforcement_index_initially_zero(self) -> None:
        pr = ProgressiveRatio(arithmetic())
        assert pr.reinforcement_index == 0


# ---------------------------------------------------------------------------
# ProgressiveRatio — arithmetic progression
# ---------------------------------------------------------------------------


class TestProgressiveRatioArithmetic:
    def test_arithmetic_1_1_series_timings(self) -> None:
        """PR with arithmetic(1,1): reinforcers at responses 1, 3, 6, 10, 15."""
        pr = ProgressiveRatio(arithmetic(start=1, step=1))
        reinforced_at: list[int] = []
        t = 0.0
        for i in range(1, 21):
            t = float(i)
            o = _respond(pr, t)
            if o.reinforced:
                reinforced_at.append(i)
        # Cumulative responses at each reinforcer: 1, 1+2, 1+2+3, 1+2+3+4, 1+2+3+4+5
        assert reinforced_at == [1, 3, 6, 10, 15]

    def test_arithmetic_reinforcer_timestamp_equals_now(self) -> None:
        pr = ProgressiveRatio(arithmetic(start=1, step=1))
        o = _respond(pr, 1.0)
        assert o.reinforced is True
        assert o.reinforcer is not None
        assert o.reinforcer.time == 1.0

    def test_index_advances_after_reinforcement(self) -> None:
        pr = ProgressiveRatio(arithmetic(start=1, step=1))
        assert pr.reinforcement_index == 0
        _respond(pr, 1.0)  # meets criterion 1 → reinforce, index→1
        assert pr.reinforcement_index == 1
        assert pr.current_requirement == 2
        _respond(pr, 2.0)
        assert pr.reinforcement_index == 1
        _respond(pr, 3.0)  # second reinforcer earned
        assert pr.reinforcement_index == 2

    def test_non_response_steps_never_reinforce(self) -> None:
        pr = ProgressiveRatio(arithmetic())
        for t in (0.1, 0.5, 1.0, 1.5, 2.0):
            o = pr.step(t)
            assert o.reinforced is False
            assert o.reinforcer is None

    def test_empty_outcome_fields(self) -> None:
        pr = ProgressiveRatio(arithmetic(start=2, step=1))
        o = _respond(pr, 1.0)  # below criterion 2
        assert o.reinforced is False
        assert o.reinforcer is None
        assert o.meta == {}


# ---------------------------------------------------------------------------
# ProgressiveRatio — geometric progression
# ---------------------------------------------------------------------------


class TestProgressiveRatioGeometric:
    def test_geometric_2x_pattern(self) -> None:
        """geometric(1, 2.0): requirements 1, 2, 4, 8 → reinforced at 1, 3, 7, 15."""
        pr = ProgressiveRatio(geometric(start=1, ratio=2.0))
        reinforced_at: list[int] = []
        for i in range(1, 16):
            o = _respond(pr, float(i))
            if o.reinforced:
                reinforced_at.append(i)
        assert reinforced_at == [1, 3, 7, 15]


# ---------------------------------------------------------------------------
# ProgressiveRatio — Richardson-Roberts
# ---------------------------------------------------------------------------


class TestProgressiveRatioRichardsonRoberts:
    def test_richardson_roberts_first_requirements_match_series(self) -> None:
        pr = ProgressiveRatio(richardson_roberts())
        expected = [1, 2, 4, 6, 9]
        observed: list[int] = []
        response_count = 0
        # Enough responses to earn 5 reinforcers: 1+2+4+6+9 = 22
        for i in range(1, 25):
            observed_req = pr.current_requirement
            o = _respond(pr, float(i))
            response_count += 1
            if o.reinforced:
                observed.append(observed_req)
            if len(observed) == 5:
                break
        assert observed == expected


# ---------------------------------------------------------------------------
# ProgressiveRatio — reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_restarts_from_index_zero(self) -> None:
        pr = ProgressiveRatio(arithmetic(start=1, step=1))
        for t in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
            _respond(pr, t)
        assert pr.reinforcement_index > 0
        pr.reset()
        assert pr.reinforcement_index == 0
        assert pr.current_requirement == 1
        # First response after reset reinforces again.
        o = _respond(pr, 7.0)
        assert o.reinforced is True

    def test_reset_clears_last_time(self) -> None:
        pr = ProgressiveRatio(arithmetic())
        _respond(pr, 100.0)
        pr.reset()
        # After reset, stepping "earlier" again is valid.
        o = _respond(pr, 0.0)
        assert o.reinforced is True


# ---------------------------------------------------------------------------
# ProgressiveRatio — invalid step-function values
# ---------------------------------------------------------------------------


class TestInvalidStepFn:
    def test_step_fn_returning_zero_raises_on_trigger(self) -> None:
        pr = ProgressiveRatio(lambda n: 0)
        with pytest.raises(ScheduleConfigError):
            _respond(pr, 1.0)

    def test_step_fn_returning_negative_raises_on_trigger(self) -> None:
        pr = ProgressiveRatio(lambda n: -1)
        with pytest.raises(ScheduleConfigError):
            _respond(pr, 1.0)

    def test_step_fn_returning_non_int_raises_on_trigger(self) -> None:
        pr = ProgressiveRatio(lambda n: 1.5)  # type: ignore[return-value,arg-type]
        with pytest.raises(ScheduleConfigError):
            _respond(pr, 1.0)

    def test_step_fn_returning_bool_raises_on_trigger(self) -> None:
        pr = ProgressiveRatio(lambda n: True)  # type: ignore[return-value,arg-type]
        with pytest.raises(ScheduleConfigError):
            _respond(pr, 1.0)

    def test_current_requirement_property_raises_on_bad_value(self) -> None:
        pr = ProgressiveRatio(lambda n: 0)
        with pytest.raises(ScheduleConfigError):
            _ = pr.current_requirement


# ---------------------------------------------------------------------------
# ProgressiveRatio — state errors
# ---------------------------------------------------------------------------


class TestStateErrors:
    def test_non_monotonic_time_raises(self) -> None:
        pr = ProgressiveRatio(arithmetic(start=2))
        _respond(pr, 10.0)
        with pytest.raises(ScheduleStateError):
            _respond(pr, 5.0)

    def test_event_time_mismatch_raises(self) -> None:
        pr = ProgressiveRatio(arithmetic())
        with pytest.raises(ScheduleStateError):
            pr.step(1.0, ResponseEvent(time=2.0))

    def test_monotonic_within_tolerance_allowed(self) -> None:
        pr = ProgressiveRatio(arithmetic(start=2))
        _respond(pr, 10.0)
        # Small backward step within tolerance should not raise.
        _respond(pr, 10.0 - 1e-12)


# ---------------------------------------------------------------------------
# Integration: 10 successive reinforcers on arithmetic
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_ten_reinforcers_arithmetic_1_1(self) -> None:
        """On arithmetic(1,1), response counts between reinforcers = [1..10]."""
        pr = ProgressiveRatio(arithmetic(start=1, step=1))
        response_counts: list[int] = []
        count = 0
        t = 0.0
        # Expected cumulative responses: 1, 3, 6, 10, 15, 21, 28, 36, 45, 55.
        total_needed = sum(range(1, 11))  # 55
        for i in range(1, total_needed + 1):
            count += 1
            t = float(i)
            o = _respond(pr, t)
            if o.reinforced:
                response_counts.append(count)
                count = 0
        assert response_counts == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        assert pr.reinforcement_index == 10

    def test_ten_reinforcers_arithmetic_2_3(self) -> None:
        """arithmetic(2,3): requirements 2, 5, 8, 11, ...; counts match."""
        pr = ProgressiveRatio(arithmetic(start=2, step=3))
        expected = [2 + 3 * n for n in range(10)]  # [2, 5, 8, ..., 29]
        total = sum(expected)
        response_counts: list[int] = []
        count = 0
        for i in range(1, total + 1):
            count += 1
            o = _respond(pr, float(i))
            if o.reinforced:
                response_counts.append(count)
                count = 0
        assert response_counts == expected
