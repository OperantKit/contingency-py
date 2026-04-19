"""Tests for :class:`contingency.schedules.AdjustingSchedule`.

References
----------
Mazur, J. E. (1987). An adjusting procedure for studying delayed
reinforcement. In M. L. Commons, J. E. Mazur, J. A. Nevin, & H.
Rachlin (Eds.), *Quantitative analyses of behavior, Vol. 5*
(pp. 55-73). Lawrence Erlbaum Associates.
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
from contingency.schedules import AdjustingSchedule


def _respond(schedule: Schedule, now: float) -> Outcome:
    return schedule.step(now, ResponseEvent(time=now))


class TestProtocol:
    def test_ratio_instance_is_schedule(self) -> None:
        assert isinstance(
            AdjustingSchedule("ratio", start=2, step=1), Schedule
        )

    def test_interval_instance_is_schedule(self) -> None:
        assert isinstance(
            AdjustingSchedule("interval", start=5.0, step=1.0), Schedule
        )


class TestConstructionValidation:
    def test_unknown_target_raises(self) -> None:
        with pytest.raises(ScheduleConfigError, match="target must be"):
            AdjustingSchedule("speed", start=1, step=1)

    def test_non_string_target_raises(self) -> None:
        with pytest.raises(ScheduleConfigError, match="str target"):
            AdjustingSchedule(42, start=1, step=1)  # type: ignore[arg-type]

    def test_delay_target_is_deferred(self) -> None:
        with pytest.raises(
            ScheduleConfigError, match="runtime not yet implemented"
        ):
            AdjustingSchedule("delay", start=1.0, step=1.0)

    def test_amount_target_is_deferred(self) -> None:
        with pytest.raises(
            ScheduleConfigError, match="runtime not yet implemented"
        ):
            AdjustingSchedule("amount", start=1.0, step=1.0)

    def test_negative_start_raises(self) -> None:
        with pytest.raises(ScheduleConfigError, match="start must be > 0"):
            AdjustingSchedule("ratio", start=-3, step=1)

    def test_zero_start_raises(self) -> None:
        with pytest.raises(ScheduleConfigError, match="start must be > 0"):
            AdjustingSchedule("ratio", start=0, step=1)

    def test_bool_start_raises(self) -> None:
        with pytest.raises(ScheduleConfigError, match="numeric start"):
            AdjustingSchedule(
                "ratio", start=True, step=1
            )  # type: ignore[arg-type]

    def test_string_step_raises(self) -> None:
        with pytest.raises(ScheduleConfigError, match="numeric step"):
            AdjustingSchedule(
                "ratio", start=1, step="one"
            )  # type: ignore[arg-type]

    def test_bool_step_raises(self) -> None:
        with pytest.raises(ScheduleConfigError, match="numeric step"):
            AdjustingSchedule(
                "ratio", start=1, step=False
            )  # type: ignore[arg-type]

    def test_minimum_greater_than_maximum_raises(self) -> None:
        with pytest.raises(
            ScheduleConfigError, match="minimum <= maximum"
        ):
            AdjustingSchedule(
                "ratio", start=5, step=1, minimum=10, maximum=2
            )

    def test_non_numeric_minimum_raises(self) -> None:
        with pytest.raises(ScheduleConfigError, match="minimum"):
            AdjustingSchedule(
                "ratio", start=5, step=1, minimum="lo"
            )  # type: ignore[arg-type]

    def test_non_numeric_maximum_raises(self) -> None:
        with pytest.raises(ScheduleConfigError, match="maximum"):
            AdjustingSchedule(
                "ratio", start=5, step=1, maximum="hi"
            )  # type: ignore[arg-type]

    def test_ratio_start_below_one_after_rounding_raises(self) -> None:
        with pytest.raises(ScheduleConfigError, match="ratio start"):
            AdjustingSchedule("ratio", start=0.3, step=1)

    def test_properties_expose_params(self) -> None:
        s = AdjustingSchedule(
            "ratio", start=3, step=2, minimum=1, maximum=10
        )
        assert s.target == "ratio"
        assert s.start == 3.0
        assert s.step_size == 2.0
        assert s.minimum == 1.0
        assert s.maximum == 10.0
        assert s.current_value == 3.0


class TestRatioSemantics:
    def test_first_reinforcement_uses_start_value(self) -> None:
        s = AdjustingSchedule("ratio", start=3, step=1)
        outcomes = [_respond(s, float(i)) for i in range(1, 4)]
        assert [o.reinforced for o in outcomes] == [False, False, True]

    def test_increment_applies_after_reinforcement(self) -> None:
        s = AdjustingSchedule("ratio", start=2, step=1)
        # First reinforcement at count 2 → requirement advances to 3.
        _respond(s, 1.0)
        out = _respond(s, 2.0)
        assert out.reinforced
        assert s.current_value == 3.0
        # Next three responses for the new requirement of 3.
        _respond(s, 3.0)
        _respond(s, 4.0)
        out = _respond(s, 5.0)
        assert out.reinforced
        assert s.current_value == 4.0

    def test_decrement_with_negative_step(self) -> None:
        s = AdjustingSchedule("ratio", start=4, step=-1, minimum=1)
        # Burn through three reinforcements: 4 → 3 → 2 → 1.
        t = 0.0
        delivered = 0
        while delivered < 3:
            t += 1.0
            if _respond(s, t).reinforced:
                delivered += 1
        # After 3 reinforcements current_value is 1 (clamped).
        assert s.current_value == 1.0

    def test_clamp_upper_bound(self) -> None:
        s = AdjustingSchedule(
            "ratio", start=2, step=5, maximum=6
        )
        # 2 → 7 but clamped to 6 → 11 but clamped to 6.
        while _respond(s, 1.0).reinforced is False:
            pass
        assert s.current_value == 6.0
        # A single reinforcement should have been emitted.
        while _respond(s, 2.0).reinforced is False:
            pass
        assert s.current_value == 6.0

    def test_clamp_lower_bound(self) -> None:
        s = AdjustingSchedule("ratio", start=3, step=-5, minimum=1)
        # 3 → -2 clamped to 1.
        while _respond(s, 1.0).reinforced is False:
            pass
        assert s.current_value == 1.0

    def test_rounding_for_non_integer_current(self) -> None:
        # start=2, step=0.4 ⇒ after first reinforcement current=2.4.
        # The rounded requirement is max(1, round(2.4)) = 2. The second
        # reinforcement then moves current to 2.8 ⇒ requirement 3.
        s = AdjustingSchedule("ratio", start=2, step=0.4)
        outcomes = []
        for t in range(1, 10):
            outcomes.append(_respond(s, float(t)))
        reinforced_steps = [i + 1 for i, o in enumerate(outcomes) if o.reinforced]
        # Reinforcements at: 2 (req 2), 4 (req 2), 7 (req 3) — current values
        # 2.4, 2.8, 3.2 → reqs 2, 3, 3.
        assert reinforced_steps == [2, 4, 7]

    def test_non_response_ticks_do_not_reinforce(self) -> None:
        s = AdjustingSchedule("ratio", start=2, step=1)
        assert s.step(1.0).reinforced is False
        assert s.step(2.0).reinforced is False


class TestIntervalSemantics:
    def test_first_reinforcement_after_interval(self) -> None:
        s = AdjustingSchedule("interval", start=2.0, step=1.0)
        # Anchor establishes at t=0.0.
        _respond(s, 0.0)
        # A response at 1.0 is before the interval — no reinforcer.
        assert _respond(s, 1.0).reinforced is False
        # A response exactly at the interval: reinforced.
        out = _respond(s, 2.0)
        assert out.reinforced
        assert out.reinforcer is not None
        assert out.reinforcer.time == 2.0
        assert s.current_value == 3.0

    def test_increment_interval_after_reinforcement(self) -> None:
        s = AdjustingSchedule("interval", start=1.0, step=0.5)
        # Anchor at 0.0.
        _respond(s, 0.0)
        # First reinforcement at t=1.0 → current becomes 1.5.
        assert _respond(s, 1.0).reinforced
        assert s.current_value == pytest.approx(1.5)
        # No reinforcement at t=2.0 (elapsed 1.0 < 1.5).
        assert _respond(s, 2.0).reinforced is False
        # Reinforcement at t=2.5 (elapsed 1.5 ≥ 1.5).
        assert _respond(s, 2.5).reinforced

    def test_clamp_interval(self) -> None:
        s = AdjustingSchedule(
            "interval", start=1.0, step=1.0, maximum=2.0
        )
        # Anchor.
        _respond(s, 0.0)
        # Reinforce at t=1.0 → current becomes 2.0 (clamped).
        assert _respond(s, 1.0).reinforced
        assert s.current_value == 2.0
        # Next interval 2.0: reinforce at t=3.0 (elapsed 2.0) → stays 2.0.
        assert _respond(s, 3.0).reinforced
        assert s.current_value == 2.0

    def test_anchor_is_set_on_first_step_even_without_event(self) -> None:
        s = AdjustingSchedule("interval", start=1.0, step=1.0)
        # A bare tick at 10.0 anchors the internal clock.
        s.step(10.0)
        # A response at 10.5 is still inside the interval.
        assert _respond(s, 10.5).reinforced is False
        # A response at 11.0 has elapsed=1.0 from the anchor.
        assert _respond(s, 11.0).reinforced


class TestReset:
    def test_reset_restores_ratio_start(self) -> None:
        s = AdjustingSchedule("ratio", start=2, step=1)
        _respond(s, 1.0)
        out = _respond(s, 2.0)
        assert out.reinforced
        assert s.current_value == 3.0
        s.reset()
        assert s.current_value == 2.0
        # Next reinforcement takes 2 responses again.
        _respond(s, 3.0)
        out = _respond(s, 4.0)
        assert out.reinforced

    def test_reset_restores_interval_state(self) -> None:
        s = AdjustingSchedule("interval", start=1.0, step=1.0)
        _respond(s, 0.0)
        _respond(s, 1.0)
        assert s.current_value == 2.0
        s.reset()
        assert s.current_value == 1.0
        # After reset the anchor is cleared; stepping from a new time works.
        _respond(s, 100.0)
        assert _respond(s, 101.0).reinforced

    def test_reset_clears_last_time(self) -> None:
        s = AdjustingSchedule("ratio", start=2, step=1)
        _respond(s, 100.0)
        s.reset()
        # Stepping "earlier" again is legal after reset.
        _respond(s, 1.0)

    def test_reset_applies_clamp_to_initial_value(self) -> None:
        # Construct with a start just above the maximum — the clamp
        # kicks in at construction and again on reset.
        s = AdjustingSchedule(
            "ratio", start=5, step=1, minimum=2, maximum=3
        )
        assert s.current_value == 3.0
        # Run a reinforcement then reset.
        while _respond(s, 1.0).reinforced is False:
            pass
        s.reset()
        assert s.current_value == 3.0


class TestStateErrors:
    def test_non_monotonic_time_raises(self) -> None:
        s = AdjustingSchedule("ratio", start=2, step=1)
        _respond(s, 5.0)
        with pytest.raises(ScheduleStateError, match="Non-monotonic"):
            _respond(s, 4.0)

    def test_event_time_mismatch_raises(self) -> None:
        s = AdjustingSchedule("ratio", start=2, step=1)
        with pytest.raises(ScheduleStateError, match="event.time"):
            s.step(1.0, ResponseEvent(time=0.5))

    def test_equal_time_allowed(self) -> None:
        s = AdjustingSchedule("ratio", start=2, step=1)
        _respond(s, 5.0)
        # Second response at same time is legal (within tolerance);
        # it completes the requirement of 2.
        out = _respond(s, 5.0)
        assert out.reinforced

    def test_non_monotonic_interval_raises(self) -> None:
        s = AdjustingSchedule("interval", start=1.0, step=1.0)
        _respond(s, 5.0)
        with pytest.raises(ScheduleStateError):
            _respond(s, 4.0)


class TestEdgeCases:
    def test_ratio_requirement_never_drops_below_one(self) -> None:
        # Even with a negative step and no minimum, the effective
        # ratio requirement is max(1, round(current)).
        s = AdjustingSchedule("ratio", start=1, step=-10)
        out = _respond(s, 1.0)
        assert out.reinforced
        # current would be -9 → round(-9) = -9 → max(1, -9) = 1.
        out = _respond(s, 2.0)
        assert out.reinforced

    def test_step_zero_keeps_current_constant(self) -> None:
        # A zero step is a degenerate FR — the adjusting trajectory is
        # flat. This is permitted.
        s = AdjustingSchedule("ratio", start=2, step=0)
        for _ in range(3):
            _respond(s, 1.0)  # legal: equal timestamps within tol
            out = _respond(s, 1.0)
            assert out.reinforced
            assert s.current_value == 2.0
