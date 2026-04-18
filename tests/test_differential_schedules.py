"""Tests for differential reinforcement schedules: DRO, DRL, DRH.

References
----------
Ferster, C. B., & Skinner, B. F. (1957). *Schedules of reinforcement*.
Appleton-Century-Crofts.

Reynolds, G. S. (1961). Behavioral contrast. *Journal of the
Experimental Analysis of Behavior*, 4(1), 57-71.
https://doi.org/10.1901/jeab.1961.4-57

Reynolds, G. S. (1964). Accurate and rapid reconditioning of
spaced-responding by differential reinforcement of other behavior.
*Journal of the Experimental Analysis of Behavior*, 7(3), 223-224.
https://doi.org/10.1901/jeab.1964.7-223

Zeiler, M. D. (1977). Schedules of reinforcement: The controlling
variables. In W. K. Honig & J. E. R. Staddon (Eds.), *Handbook of
operant behavior* (pp. 201-232). Prentice-Hall.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from contingency.entities import Outcome, ResponseEvent
from contingency.errors import ScheduleConfigError, ScheduleStateError
from contingency.interfaces import Schedule
from contingency.schedules import DRH, DRL, DRO


def _respond(schedule: Schedule, now: float) -> Outcome:
    """Helper: step the schedule with a response at ``now``."""
    return schedule.step(now, ResponseEvent(time=now))


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestProtocolCompliance:
    def test_dro_resetting_satisfies_schedule_protocol(self) -> None:
        assert isinstance(DRO(1.0), Schedule)

    def test_dro_momentary_satisfies_schedule_protocol(self) -> None:
        assert isinstance(DRO(1.0, dro_type="momentary"), Schedule)

    def test_drl_satisfies_schedule_protocol(self) -> None:
        assert isinstance(DRL(1.0), Schedule)

    def test_drh_satisfies_schedule_protocol(self) -> None:
        assert isinstance(DRH(3, 5.0), Schedule)


# ---------------------------------------------------------------------------
# DRO — resetting variant
# ---------------------------------------------------------------------------


class TestDROResettingConfig:
    @pytest.mark.parametrize("bad", [0, -1, -0.001, -100.0])
    def test_non_positive_interval_rejected(self, bad: float) -> None:
        with pytest.raises(ScheduleConfigError):
            DRO(bad)

    def test_non_numeric_interval_rejected(self) -> None:
        with pytest.raises(ScheduleConfigError):
            DRO("1.0")  # type: ignore[arg-type]

    def test_bool_interval_rejected(self) -> None:
        with pytest.raises(ScheduleConfigError):
            DRO(True)  # type: ignore[arg-type]

    def test_invalid_dro_type_rejected(self) -> None:
        with pytest.raises(ScheduleConfigError):
            DRO(1.0, dro_type="invalid")  # type: ignore[arg-type]

    def test_defaults_to_resetting(self) -> None:
        s = DRO(1.0)
        assert s.dro_type == "resetting"
        assert s.interval == 1.0


class TestDROResettingBehavior:
    def test_first_step_anchors_no_reinforcement(self) -> None:
        s = DRO(5.0)
        out = s.step(10.0)
        assert not out.reinforced

    def test_tick_before_interval_no_reinforcement(self) -> None:
        s = DRO(5.0)
        s.step(0.0)
        for t in (1.0, 2.0, 4.999):
            out = s.step(t)
            assert not out.reinforced

    def test_tick_exactly_at_interval_reinforces(self) -> None:
        s = DRO(5.0)
        s.step(0.0)
        out = s.step(5.0)
        assert out.reinforced
        assert out.reinforcer is not None
        assert out.reinforcer.time == 5.0

    def test_tick_past_interval_reinforces(self) -> None:
        s = DRO(5.0)
        s.step(0.0)
        out = s.step(7.5)
        assert out.reinforced

    def test_event_during_interval_resets_timer(self) -> None:
        s = DRO(5.0)
        s.step(0.0)
        # Response at t=3 resets the clock.
        out = _respond(s, 3.0)
        assert not out.reinforced
        # t=7 is only 4s after the reset → no reinforcement.
        out = s.step(7.0)
        assert not out.reinforced
        # t=8 is 5s after the reset → reinforcement.
        out = s.step(8.0)
        assert out.reinforced
        assert out.reinforcer is not None
        assert out.reinforcer.time == 8.0

    def test_consecutive_reinforcements_on_successive_quiet_intervals(
        self,
    ) -> None:
        s = DRO(5.0)
        s.step(0.0)
        reinforced_times: list[float] = []
        # Tick every 1s out to t=20; no responses.
        for i in range(1, 21):
            out = s.step(float(i))
            if out.reinforced:
                assert out.reinforcer is not None
                reinforced_times.append(out.reinforcer.time)
        # Reinforcer at t=5, then after restart at t=5 → t=10, t=15, t=20.
        assert reinforced_times == [5.0, 10.0, 15.0, 20.0]

    def test_timer_restarts_from_reinforcement_time(self) -> None:
        s = DRO(5.0)
        s.step(0.0)
        out = s.step(5.0)
        assert out.reinforced
        # New window starts at t=5; t=9 is too early, t=10 is on boundary.
        out = s.step(9.0)
        assert not out.reinforced
        out = s.step(10.0)
        assert out.reinforced

    def test_event_after_reinforcement_resets_again(self) -> None:
        s = DRO(5.0)
        s.step(0.0)
        s.step(5.0)  # reinforced, new window from 5
        _respond(s, 7.0)  # response resets to 7
        out = s.step(11.0)  # 4s after reset → no reinforcement
        assert not out.reinforced
        out = s.step(12.0)  # 5s after reset → reinforcement
        assert out.reinforced


# ---------------------------------------------------------------------------
# DRO — momentary variant
# ---------------------------------------------------------------------------


class TestDROMomentaryBehavior:
    def test_first_step_anchors(self) -> None:
        s = DRO(5.0, dro_type="momentary")
        out = s.step(10.0)
        assert not out.reinforced

    def test_event_during_interval_does_not_reset_timer(self) -> None:
        s = DRO(5.0, dro_type="momentary")
        s.step(0.0)
        _respond(s, 2.0)
        # At t=5, interval boundary — but a response occurred → no SR+.
        out = s.step(5.0)
        assert not out.reinforced
        # Next window runs 5..10 with no responses → SR+ at 10.
        out = s.step(10.0)
        assert out.reinforced
        assert out.reinforcer is not None
        assert out.reinforcer.time == 10.0

    def test_boundary_with_no_event_reinforces(self) -> None:
        s = DRO(5.0, dro_type="momentary")
        s.step(0.0)
        out = s.step(5.0)
        assert out.reinforced

    def test_timer_advances_regardless_of_outcome(self) -> None:
        s = DRO(5.0, dro_type="momentary")
        s.step(0.0)
        _respond(s, 1.0)  # response blocks next SR+
        out = s.step(5.0)  # boundary, blocked
        assert not out.reinforced
        # Timer advanced → next window 5..10
        out = s.step(10.0)
        assert out.reinforced

    def test_multiple_consecutive_intervals(self) -> None:
        s = DRO(5.0, dro_type="momentary")
        s.step(0.0)
        # Stream of responses every 1s keeps each window polluted.
        for i in range(1, 11):
            _respond(s, float(i))
        # After this stream the most recent window-close was at t=10,
        # and the event at t=10 belongs to the [10, 15) window per the
        # half-open rule. So at t=15 SR+ is still blocked by the
        # response at t=10. The first clean window is [15, 20).
        out = s.step(15.0)
        assert not out.reinforced
        out = s.step(20.0)
        assert out.reinforced

    def test_response_at_boundary_belongs_to_next_window(self) -> None:
        """Half-open window semantics: a response exactly at ``now``
        counts for the next window, not the one just closing."""
        s = DRO(5.0, dro_type="momentary")
        s.step(0.0)
        # No response during [0, 5); the boundary-and-response step at
        # t=5 should reinforce (window closing was clean) and the
        # response is registered for the [5, 10) window.
        out = _respond(s, 5.0)
        assert out.reinforced
        # Next boundary at t=10 should NOT reinforce because the
        # response at t=5 is in [5, 10).
        out = s.step(10.0)
        assert not out.reinforced

    def test_large_jump_only_one_reinforcer(self) -> None:
        """Stepping far past the boundary still emits one SR+ only."""
        s = DRO(5.0, dro_type="momentary")
        s.step(0.0)
        out = s.step(100.0)
        assert out.reinforced
        # After the jump the anchor is at 100; next SR+ requires 5s.
        out = s.step(104.0)
        assert not out.reinforced
        out = s.step(105.0)
        assert out.reinforced


class TestDROAnchorWithEvent:
    def test_resetting_first_step_with_event_anchors(self) -> None:
        s = DRO(5.0)
        out = _respond(s, 10.0)
        assert not out.reinforced
        # Clock now anchored at 10; need another 5s of quiet.
        out = s.step(14.999)
        assert not out.reinforced
        out = s.step(15.0)
        assert out.reinforced

    def test_momentary_first_step_with_event_anchors_with_response_flag(
        self,
    ) -> None:
        s = DRO(5.0, dro_type="momentary")
        _respond(s, 0.0)
        # Window [0, 5) now contains the anchor response → no SR+ at 5.
        out = s.step(5.0)
        assert not out.reinforced


# ---------------------------------------------------------------------------
# DRL
# ---------------------------------------------------------------------------


class TestDRLConfig:
    @pytest.mark.parametrize("bad", [0, -1, -0.001, -100.0])
    def test_non_positive_interval_rejected(self, bad: float) -> None:
        with pytest.raises(ScheduleConfigError):
            DRL(bad)

    def test_non_numeric_interval_rejected(self) -> None:
        with pytest.raises(ScheduleConfigError):
            DRL("1.0")  # type: ignore[arg-type]

    def test_bool_interval_rejected(self) -> None:
        with pytest.raises(ScheduleConfigError):
            DRL(True)  # type: ignore[arg-type]

    def test_interval_property(self) -> None:
        assert DRL(3.5).interval == 3.5


class TestDRLBehavior:
    def test_tick_only_never_reinforces(self) -> None:
        s = DRL(3.0)
        for t in (0.0, 1.0, 3.0, 10.0, 100.0):
            out = s.step(t)
            assert not out.reinforced

    def test_first_event_always_reinforces(self) -> None:
        s = DRL(3.0)
        out = _respond(s, 0.5)
        assert out.reinforced
        assert out.reinforcer is not None
        assert out.reinforcer.time == 0.5

    def test_event_within_interval_not_reinforced(self) -> None:
        s = DRL(3.0)
        _respond(s, 0.0)  # first → reinforced
        out = _respond(s, 2.0)  # IRT=2 < 3
        assert not out.reinforced

    def test_event_within_interval_still_resets_irt_clock(self) -> None:
        s = DRL(3.0)
        _respond(s, 0.0)  # reinforced
        _respond(s, 1.0)  # IRT=1 → not reinforced, bookmark advances to 1
        _respond(s, 2.0)  # IRT=1 → not reinforced, bookmark to 2
        out = _respond(s, 4.9)  # IRT=2.9 from t=2 → still short
        assert not out.reinforced
        out = _respond(s, 8.0)  # IRT=3.1 from t=4.9 → reinforced
        assert out.reinforced

    def test_event_exactly_at_interval_boundary_reinforces(self) -> None:
        s = DRL(3.0)
        _respond(s, 0.0)
        out = _respond(s, 3.0)
        assert out.reinforced

    def test_event_after_interval_reinforces(self) -> None:
        s = DRL(3.0)
        _respond(s, 0.0)
        out = _respond(s, 5.0)
        assert out.reinforced


# ---------------------------------------------------------------------------
# DRH
# ---------------------------------------------------------------------------


class TestDRHConfig:
    @pytest.mark.parametrize("bad", [0, -1, -5])
    def test_non_positive_response_count_rejected(self, bad: int) -> None:
        with pytest.raises(ScheduleConfigError):
            DRH(bad, 5.0)

    def test_non_int_response_count_rejected(self) -> None:
        with pytest.raises(ScheduleConfigError):
            DRH(3.0, 5.0)  # type: ignore[arg-type]

    def test_bool_response_count_rejected(self) -> None:
        with pytest.raises(ScheduleConfigError):
            DRH(True, 5.0)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad", [0, -1, -0.001])
    def test_non_positive_window_rejected(self, bad: float) -> None:
        with pytest.raises(ScheduleConfigError):
            DRH(3, bad)

    def test_non_numeric_window_rejected(self) -> None:
        with pytest.raises(ScheduleConfigError):
            DRH(3, "5.0")  # type: ignore[arg-type]

    def test_bool_window_rejected(self) -> None:
        with pytest.raises(ScheduleConfigError):
            DRH(3, True)  # type: ignore[arg-type]

    def test_properties(self) -> None:
        s = DRH(3, 5.0)
        assert s.response_count == 3
        assert s.time_window == 5.0


class TestDRHBehavior:
    def test_tick_only_never_reinforces(self) -> None:
        s = DRH(3, 5.0)
        for t in (0.0, 1.0, 5.0, 100.0):
            out = s.step(t)
            assert not out.reinforced

    def test_first_events_below_count_not_reinforced(self) -> None:
        s = DRH(3, 5.0)
        out = _respond(s, 0.0)
        assert not out.reinforced
        out = _respond(s, 1.0)
        assert not out.reinforced

    def test_third_event_within_window_reinforces(self) -> None:
        s = DRH(3, 5.0)
        _respond(s, 0.0)
        _respond(s, 1.0)
        out = _respond(s, 2.0)
        assert out.reinforced
        assert out.reinforcer is not None
        assert out.reinforcer.time == 2.0

    def test_old_events_drop_out_of_window(self) -> None:
        s = DRH(3, 5.0)
        _respond(s, 0.0)
        _respond(s, 1.0)
        # These two drop out when the window cutoff passes.
        # Responses at t=10, 11 → only 2 in window.
        out = _respond(s, 10.0)
        assert not out.reinforced
        out = _respond(s, 11.0)
        assert not out.reinforced
        # Third recent response → reinforced.
        out = _respond(s, 12.0)
        assert out.reinforced

    def test_window_uses_closed_lower_bound(self) -> None:
        """A response exactly ``time_window`` old is still in the window."""
        s = DRH(3, 5.0)
        _respond(s, 0.0)
        _respond(s, 2.0)
        # At t=5 the response at t=0 is exactly window-old; window [0,5].
        out = _respond(s, 5.0)
        assert out.reinforced

    def test_continuous_high_rate_keeps_reinforcing(self) -> None:
        s = DRH(3, 5.0)
        # Three responses already produced one SR+ at t=2.
        _respond(s, 0.0)
        _respond(s, 1.0)
        _respond(s, 2.0)  # first SR+
        # Each subsequent response still sees >= 3 timestamps within 5s.
        out = _respond(s, 3.0)
        assert out.reinforced
        out = _respond(s, 4.0)
        assert out.reinforced

    def test_tick_drops_old_responses(self) -> None:
        """Ticks without events still evict old timestamps."""
        s = DRH(3, 5.0)
        _respond(s, 0.0)
        _respond(s, 0.5)
        # Tick past the window; both should evict.
        s.step(10.0)
        out = _respond(s, 10.0)
        assert not out.reinforced
        # One response in window so far; need 2 more.
        out = _respond(s, 10.5)
        assert not out.reinforced
        out = _respond(s, 11.0)
        assert out.reinforced

    def test_drh_count_one_always_reinforces(self) -> None:
        s = DRH(1, 5.0)
        out = _respond(s, 0.0)
        assert out.reinforced
        out = _respond(s, 1.0)
        assert out.reinforced


class TestDRHHypothesis:
    @settings(max_examples=50, deadline=None)
    @given(
        times=st.lists(
            st.floats(
                min_value=0.0,
                max_value=100.0,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=1,
            max_size=40,
        ),
        response_count=st.integers(min_value=1, max_value=6),
        time_window=st.floats(
            min_value=0.5, max_value=10.0, allow_nan=False, allow_infinity=False
        ),
    )
    def test_monotonic_window_correctness(
        self,
        times: list[float],
        response_count: int,
        time_window: float,
    ) -> None:
        """Each reinforcement should correspond to a moment where the
        number of timestamps within ``[now - time_window, now]`` is at
        least ``response_count``."""
        # Ensure monotonic non-decreasing sequence.
        times = sorted(times)
        s = DRH(response_count, time_window)
        processed: list[float] = []
        for now in times:
            out = _respond(s, now)
            processed.append(now)
            # Independent reference: count of processed events whose
            # timestamps lie within [now - time_window, now].
            in_window = sum(
                1
                for t in processed
                if t <= now + 1e-9 and t >= now - time_window - 1e-9
            )
            if out.reinforced:
                assert in_window >= response_count
            else:
                # Not reinforced → strictly fewer than count in window.
                assert in_window < response_count


# ---------------------------------------------------------------------------
# State errors (monotonicity / event timestamp)
# ---------------------------------------------------------------------------


class TestStateErrors:
    @pytest.mark.parametrize(
        "factory",
        [
            lambda: DRO(5.0),
            lambda: DRO(5.0, dro_type="momentary"),
            lambda: DRL(3.0),
            lambda: DRH(3, 5.0),
        ],
    )
    def test_non_monotonic_now_rejected(self, factory) -> None:  # type: ignore[no-untyped-def]
        s = factory()
        s.step(5.0)
        with pytest.raises(ScheduleStateError):
            s.step(4.0)

    @pytest.mark.parametrize(
        "factory",
        [
            lambda: DRO(5.0),
            lambda: DRO(5.0, dro_type="momentary"),
            lambda: DRL(3.0),
            lambda: DRH(3, 5.0),
        ],
    )
    def test_mismatched_event_time_rejected(self, factory) -> None:  # type: ignore[no-untyped-def]
        s = factory()
        with pytest.raises(ScheduleStateError):
            s.step(5.0, ResponseEvent(time=4.99))


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_dro_resetting_reset_matches_fresh(self) -> None:
        s = DRO(5.0)
        s.step(0.0)
        _respond(s, 3.0)
        s.step(10.0)  # would reinforce etc.
        s.reset()
        # Post-reset identical to fresh: first step anchors; 5s passes then SR+.
        fresh = DRO(5.0)
        for t in (100.0, 104.0, 105.0):
            assert s.step(t).reinforced == fresh.step(t).reinforced

    def test_dro_momentary_reset_matches_fresh(self) -> None:
        s = DRO(5.0, dro_type="momentary")
        s.step(0.0)
        _respond(s, 2.0)
        s.step(5.0)
        s.reset()
        fresh = DRO(5.0, dro_type="momentary")
        for t in (0.0, 5.0, 10.0):
            assert s.step(t).reinforced == fresh.step(t).reinforced

    def test_drl_reset_matches_fresh(self) -> None:
        s = DRL(3.0)
        _respond(s, 0.0)
        _respond(s, 1.0)
        _respond(s, 5.0)
        s.reset()
        fresh = DRL(3.0)
        # First event after reset should reinforce, like fresh.
        out_s = _respond(s, 100.0)
        out_f = _respond(fresh, 100.0)
        assert out_s.reinforced is True
        assert out_f.reinforced is True

    def test_drh_reset_clears_window(self) -> None:
        s = DRH(3, 5.0)
        _respond(s, 0.0)
        _respond(s, 1.0)
        _respond(s, 2.0)  # SR+
        s.reset()
        # Fresh schedule: first two not reinforced.
        out = _respond(s, 0.0)
        assert not out.reinforced
        out = _respond(s, 1.0)
        assert not out.reinforced
        out = _respond(s, 2.0)
        assert out.reinforced

    def test_reset_restores_monotonic_check(self) -> None:
        """After reset the schedule should accept a smaller ``now``."""
        s = DRL(3.0)
        s.step(100.0)
        s.reset()
        # Fresh: accepts any now.
        s.step(0.0)
