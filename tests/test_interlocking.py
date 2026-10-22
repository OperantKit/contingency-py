"""Tests for the interlocking schedule (``InterlockingSchedule``).

The effective requirement at elapsed time ``t`` since the last
reinforcer is ``R(t) = max(0, int(R_0 * (1 - t / T)))``. These tests
pin the boundaries described in the class docstring: the start of the
schedule, satisfying it by waiting only, satisfying it by responding
only, the mix of the two, and ``reset``.

Most tests use ``R_0 = 8`` and ``T = 8.0`` so that ``R(t) = int(8 - t)``
is exact in binary floating point for the elapsed times used.

References
----------
Ferster, C. B., & Skinner, B. F. (1957). *Schedules of
reinforcement* (pp. 501-502). Appleton-Century-Crofts.

Berryman, R., & Nevin, J. A. (1962). Interlocking schedules of
reinforcement. *Journal of the Experimental Analysis of Behavior*,
5(2), 213-223. https://doi.org/10.1901/jeab.1962.5-213
"""

from __future__ import annotations

import math
from fractions import Fraction

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
from contingency.schedules import InterlockingSchedule


def _respond(schedule: InterlockingSchedule, now: float) -> Outcome:
    """Helper: step the schedule with a response at ``now``."""
    return schedule.step(now, ResponseEvent(time=now))


def _responses_to_reinforce(schedule: InterlockingSchedule, now: float) -> int:
    """Respond repeatedly at ``now`` and return the index of the reinforced one."""
    for n in range(1, 10_000):
        if _respond(schedule, now).reinforced:
            return n
    raise AssertionError("schedule never reinforced")


class TestConstruction:
    def test_satisfies_schedule_protocol(self) -> None:
        assert isinstance(InterlockingSchedule(8, 8.0), Schedule)

    def test_exposes_parameters(self) -> None:
        s = InterlockingSchedule(8, 30)
        assert s.initial_ratio == 8
        assert s.decay_time == 30.0
        assert isinstance(s.decay_time, float)
        assert s.response_count == 0

    @pytest.mark.parametrize("bad", [0, -1, 2.0, True, "8", None])
    def test_rejects_bad_initial_ratio(self, bad: object) -> None:
        with pytest.raises(ScheduleConfigError):
            InterlockingSchedule(bad, 8.0)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad", [0, 0.0, -1.0, True, "8", None, math.nan])
    def test_rejects_bad_decay_time(self, bad: object) -> None:
        with pytest.raises(ScheduleConfigError):
            InterlockingSchedule(8, bad)  # type: ignore[arg-type]


class TestStart:
    def test_requirement_before_any_step_is_initial_ratio(self) -> None:
        s = InterlockingSchedule(8, 8.0)
        assert s.current_requirement(0.0) == 8
        assert s.current_requirement(100.0) == 8

    def test_first_reinforcer_needs_initial_ratio_responses_at_start(self) -> None:
        s = InterlockingSchedule(8, 8.0)
        for _ in range(7):
            assert not _respond(s, 0.0).reinforced
        assert s.response_count == 7
        out = _respond(s, 0.0)
        assert out.reinforced
        assert out.reinforcer == Reinforcer(time=0.0)
        assert s.response_count == 0

    def test_no_implicit_reinforcement_at_start(self) -> None:
        s = InterlockingSchedule(1, 8.0)
        assert s.step(0.0) == Outcome()
        assert s.response_count == 0

    def test_first_tick_anchors_elapsed_time(self) -> None:
        # The anchor is the first observed tick, not t = 0.
        s = InterlockingSchedule(8, 8.0)
        s.step(100.0)
        assert s.current_requirement(100.0) == 8
        assert s.current_requirement(104.0) == 4
        assert _responses_to_reinforce(s, 104.0) == 4

    def test_first_response_anchors_when_it_is_the_first_step(self) -> None:
        s = InterlockingSchedule(8, 8.0)
        assert not _respond(s, 50.0).reinforced
        # Anchored at 50: at 54 the requirement is 4 and one response
        # has already been counted.
        assert s.current_requirement(54.0) == 4
        assert _responses_to_reinforce(s, 54.0) == 3


class TestTimeOnly:
    def test_bare_ticks_never_reinforce_even_past_decay_time(self) -> None:
        s = InterlockingSchedule(8, 8.0)
        for t in [0.0, 4.0, 8.0, 16.0, 1000.0]:
            assert s.step(t) == Outcome()
        assert s.response_count == 0

    def test_requirement_decays_linearly(self) -> None:
        s = InterlockingSchedule(8, 8.0)
        s.step(0.0)
        assert [s.current_requirement(float(t)) for t in range(10)] == [
            8, 7, 6, 5, 4, 3, 2, 1, 0, 0,
        ]

    def test_requirement_is_truncated_toward_zero(self) -> None:
        s = InterlockingSchedule(8, 8.0)
        s.step(0.0)
        assert s.current_requirement(0.5) == 7
        assert s.current_requirement(6.5) == 1
        # Truncation makes the requirement reach 0 once R_0 * (1 - t/T) < 1,
        # i.e. for t > T * (1 - 1/R_0) = 7, before t = T.
        assert s.current_requirement(7.5) == 0

    def test_single_response_reinforces_at_decay_time(self) -> None:
        s = InterlockingSchedule(8, 8.0)
        s.step(0.0)
        out = _respond(s, 8.0)
        assert out.reinforced
        assert out.reinforcer == Reinforcer(time=8.0)

    def test_single_response_reinforces_long_after_decay_time(self) -> None:
        s = InterlockingSchedule(8, 8.0)
        s.step(0.0)
        assert _respond(s, 500.0).reinforced


class TestResponseOnly:
    def test_fixed_ratio_when_responding_at_one_instant(self) -> None:
        s = InterlockingSchedule(5, 8.0)
        reinforced = [_respond(s, 0.0).reinforced for _ in range(15)]
        assert reinforced == [False, False, False, False, True] * 3

    def test_initial_ratio_one_reinforces_every_response(self) -> None:
        s = InterlockingSchedule(1, 8.0)
        assert all(_respond(s, float(t)).reinforced for t in range(5))


class TestResponsesAndTime:
    def test_earlier_responses_count_toward_a_decayed_requirement(self) -> None:
        s = InterlockingSchedule(8, 8.0)
        for _ in range(3):
            assert not _respond(s, 0.0).reinforced
        # At t = 4 the requirement is 4; the 4th response reinforces.
        out = _respond(s, 4.0)
        assert out.reinforced
        assert out.reinforcer == Reinforcer(time=4.0)

    def test_too_few_responses_for_the_decayed_requirement(self) -> None:
        s = InterlockingSchedule(8, 8.0)
        for _ in range(2):
            _respond(s, 0.0)
        # At t = 4 the requirement is 4; the 3rd response falls short.
        assert not _respond(s, 4.0).reinforced
        assert s.response_count == 3
        assert _respond(s, 4.0).reinforced

    def test_requirement_is_evaluated_at_each_response_time(self) -> None:
        s = InterlockingSchedule(8, 8.0)
        _respond(s, 0.0)  # count 1, requirement 8
        _respond(s, 1.0)  # count 2, requirement 7
        _respond(s, 2.0)  # count 3, requirement 6
        assert not _respond(s, 3.0).reinforced  # count 4, requirement 5
        assert _respond(s, 4.0).reinforced  # count 5, requirement 4

    def test_reinforcement_reanchors_time_and_count(self) -> None:
        s = InterlockingSchedule(8, 8.0)
        for _ in range(3):
            _respond(s, 0.0)
        assert _respond(s, 4.0).reinforced
        assert s.response_count == 0
        assert s.current_requirement(4.0) == 8
        assert s.current_requirement(8.0) == 4
        # Elapsed time is now measured from 4.0, not 0.0.
        assert s.current_requirement(11.0) == 1
        assert s.current_requirement(12.0) == 0
        assert _respond(s, 12.0).reinforced
        assert s.current_requirement(12.0) == 8


class TestReset:
    def test_reset_clears_count_anchor_and_time(self) -> None:
        s = InterlockingSchedule(8, 8.0)
        for _ in range(5):
            _respond(s, 6.0)
        s.reset()
        assert s.response_count == 0
        # The anchor is cleared, so the requirement is R_0 at any time.
        assert s.current_requirement(1000.0) == 8
        # The time guard is cleared, so time may restart from 0.
        assert not _respond(s, 0.0).reinforced
        assert s.current_requirement(0.0) == 8

    def test_reset_then_the_next_step_reanchors(self) -> None:
        s = InterlockingSchedule(8, 8.0)
        s.step(0.0)
        s.reset()
        s.step(20.0)
        assert s.current_requirement(24.0) == 4
        assert _responses_to_reinforce(s, 24.0) == 4

    def test_reset_keeps_parameters(self) -> None:
        s = InterlockingSchedule(8, 30.0)
        s.reset()
        assert s.initial_ratio == 8
        assert s.decay_time == 30.0


class TestTimeContract:
    def test_rejects_non_monotonic_time(self) -> None:
        s = InterlockingSchedule(8, 8.0)
        s.step(5.0)
        with pytest.raises(ScheduleStateError):
            s.step(4.0)

    def test_accepts_equal_time(self) -> None:
        s = InterlockingSchedule(8, 8.0)
        s.step(5.0)
        s.step(5.0)

    def test_rejects_event_time_mismatch(self) -> None:
        s = InterlockingSchedule(8, 8.0)
        with pytest.raises(ScheduleStateError):
            s.step(1.0, ResponseEvent(time=2.0))


def _exact_requirement(r0: int, t: int, elapsed: int) -> int:
    return max(0, math.floor(Fraction(r0) * (1 - Fraction(elapsed, t))))


class TestDocumentedFormula:
    @settings(max_examples=200, deadline=None)
    @given(
        r0=st.integers(min_value=1, max_value=64),
        log2_t=st.integers(min_value=0, max_value=6),
        data=st.data(),
    )
    def test_reinforced_response_index_matches_formula(
        self, r0: int, log2_t: int, data: st.DataObject
    ) -> None:
        # Power-of-two T keeps elapsed / T exact, so this checks the
        # documented R(t) without floating-point noise.
        t = 2**log2_t
        elapsed = data.draw(st.integers(min_value=0, max_value=2 * t))
        s = InterlockingSchedule(r0, float(t))
        s.step(0.0)
        expected = max(1, _exact_requirement(r0, t, elapsed))
        assert _responses_to_reinforce(s, float(elapsed)) == expected

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "R_0 * (1 - t/T) is computed in floating point and truncated, "
            "so an exact integer such as 10 * (1 - 8/10) = 2 becomes "
            "1.9999999999999996 and the requirement drops to 1"
        ),
    )
    def test_requirement_at_an_exact_integer_value(self) -> None:
        s = InterlockingSchedule(10, 10.0)
        s.step(0.0)
        assert _exact_requirement(10, 10, 8) == 2
        assert s.current_requirement(8.0) == 2
