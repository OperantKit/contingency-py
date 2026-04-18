"""Tests for the Alternative compound schedule.

An Alternative schedule presents two component schedules concurrently
on a single operandum and delivers reinforcement as soon as *either*
component's criterion is satisfied. After reinforcement, both
components reset.

References
----------
Ferster, C. B., & Skinner, B. F. (1957). *Schedules of reinforcement*.
Appleton-Century-Crofts.
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
from contingency.schedules import FR, FT, VI, VR, Alternative


def _respond(schedule: Schedule, now: float) -> Outcome:
    """Helper: step the schedule with a response at ``now``."""
    return schedule.step(now, ResponseEvent(time=now))


class TestProtocolCompliance:
    def test_alternative_satisfies_schedule_protocol(self) -> None:
        assert isinstance(Alternative(FR(3), FT(5.0)), Schedule)


class TestConstruction:
    def test_accepts_two_schedule_components(self) -> None:
        alt = Alternative(FR(3), FT(5.0))
        assert isinstance(alt, Schedule)

    def test_component_properties_expose_originals(self) -> None:
        first = FR(3)
        second = FT(5.0)
        alt = Alternative(first, second)
        assert alt.first is first
        assert alt.second is second

    def test_rejects_non_schedule_first_component(self) -> None:
        with pytest.raises(ScheduleConfigError):
            Alternative("not a schedule", FT(5.0))  # type: ignore[arg-type]

    def test_rejects_non_schedule_second_component(self) -> None:
        with pytest.raises(ScheduleConfigError):
            Alternative(FR(3), 42)  # type: ignore[arg-type]

    def test_rejects_both_non_schedule(self) -> None:
        with pytest.raises(ScheduleConfigError):
            Alternative(None, None)  # type: ignore[arg-type]


class TestFRFT:
    def test_fr_wins_when_response_criterion_met_first(self) -> None:
        """Alternative(FR3, FT5.0): fires on 3rd response before 5 s elapse."""
        alt = Alternative(FR(3), FT(5.0))

        o1 = _respond(alt, 1.0)
        assert o1.reinforced is False

        o2 = _respond(alt, 2.0)
        assert o2.reinforced is False

        o3 = _respond(alt, 3.0)
        assert o3.reinforced is True
        assert o3.reinforcer is not None
        assert o3.reinforcer.time == 3.0
        assert o3.meta.get("alternative_winner") == "first"

    def test_ft_wins_when_time_criterion_met_first(self) -> None:
        """Alternative(FR10, FT5.0): fires when time elapses before 10 responses."""
        alt = Alternative(FR(10), FT(5.0))

        # Anchor FT's clock at t=0 via a plain step.
        alt.step(0.0)

        # A handful of responses, not enough to trigger FR10.
        for t in (1.0, 2.0, 3.0, 4.0):
            o = _respond(alt, t)
            assert o.reinforced is False

        # At t=5.0, FT5.0 has elapsed.
        o = alt.step(5.0)
        assert o.reinforced is True
        assert o.reinforcer is not None
        assert o.reinforcer.time == 5.0
        assert o.meta.get("alternative_winner") == "second"

    def test_reset_after_response_win_rearms_both(self) -> None:
        """After a response-win, the time component is anchored at the win."""
        alt = Alternative(FR(3), FT(5.0))

        # Trigger FR3 at t=3.0.
        _respond(alt, 1.0)
        _respond(alt, 2.0)
        o = _respond(alt, 3.0)
        assert o.reinforced is True
        assert o.meta["alternative_winner"] == "first"

        # Now no responses for 5 s → FT5 should fire (second wins).
        # Re-anchor FT via a plain step at t=3.0 (same as win time).
        o_tick = alt.step(3.0)
        assert o_tick.reinforced is False

        # At t=8.0 (5 s later), FT should fire.
        o_time = alt.step(8.0)
        assert o_time.reinforced is True
        assert o_time.meta["alternative_winner"] == "second"


class TestFR10FT1:
    def test_fr10_wins_before_ft1_with_fast_responding(self) -> None:
        """Alternative(FR10, FT1.0): with responses every 0.5 s, FT1.0 fires first at t=1.0."""
        alt = Alternative(FR(10), FT(1.0))

        # Step 0.0, 0.5, 1.0 ... responses at every step (2 per second).
        # FT1.0 arms at t=0.0, fires at t=1.0. After 1 s we've only had 2 responses.
        o_anchor = alt.step(0.0, ResponseEvent(time=0.0))
        assert o_anchor.reinforced is False

        o_half = alt.step(0.5, ResponseEvent(time=0.5))
        assert o_half.reinforced is False

        # At t=1.0 with only the 3rd response, FT1 wins.
        o_win = alt.step(1.0, ResponseEvent(time=1.0))
        assert o_win.reinforced is True
        assert o_win.meta["alternative_winner"] == "second"

    def test_fr10_can_win_before_ft_with_dense_responding(self) -> None:
        """Alternative(FR10, FT1.0) with 20 responses in 0.5 s: FR10 fires first."""
        alt = Alternative(FR(10), FT(1.0))

        # 10 responses at sub-second intervals; FR10 should fire on the 10th.
        reinforced_at: list[float] = []
        for i in range(1, 15):
            t = i * 0.05  # 20 Hz: 10 responses in 0.5 s.
            o = _respond(alt, t)
            if o.reinforced:
                reinforced_at.append(t)
                assert o.meta["alternative_winner"] == "first"
                break

        assert reinforced_at == [pytest.approx(0.5)]


class TestTieBreak:
    def test_first_wins_on_simultaneous_satisfaction(self) -> None:
        """When both components would fire on the same step, first wins."""
        # FR1 fires on any response; FT0-like behaviour via FT(1e-9) fires on
        # second step. We construct FR1 + FT with a tiny interval so both
        # would be satisfied at the same step.
        alt = Alternative(FR(1), FT(0.001))

        # Anchor FT at t=0.0 (no response → no FR fire).
        o_anchor = alt.step(0.0)
        assert o_anchor.reinforced is False

        # At t=0.002: FT's 0.001 interval has elapsed AND a response comes in.
        # Both components would reinforce; first (FR) must win.
        o = _respond(alt, 0.002)
        assert o.reinforced is True
        assert o.meta["alternative_winner"] == "first"


class TestResetSemantics:
    def test_after_win_both_components_reset(self) -> None:
        """After a win, subsequent count toward FR restarts from zero."""
        alt = Alternative(FR(3), FT(100.0))  # very long FT so FR wins.

        # First cycle: FR3 fires at the 3rd response.
        _respond(alt, 1.0)
        _respond(alt, 2.0)
        win1 = _respond(alt, 3.0)
        assert win1.reinforced is True
        assert win1.meta["alternative_winner"] == "first"

        # Second cycle: FR3 must require 3 *new* responses.
        o4 = _respond(alt, 4.0)
        assert o4.reinforced is False
        o5 = _respond(alt, 5.0)
        assert o5.reinforced is False
        o6 = _respond(alt, 6.0)
        assert o6.reinforced is True
        assert o6.meta["alternative_winner"] == "first"

    def test_explicit_reset_clears_internal_state(self) -> None:
        """Explicit reset() re-anchors time and clears response counts."""
        alt = Alternative(FR(3), FT(5.0))

        # Accumulate some state.
        alt.step(0.0)
        _respond(alt, 1.0)
        _respond(alt, 2.0)

        alt.reset()

        # After reset, both FR count and FT anchor are cleared. The first
        # step re-anchors FT; subsequent responses count from zero for FR.
        o_anchor = alt.step(10.0)
        assert o_anchor.reinforced is False

        _respond(alt, 11.0)
        _respond(alt, 12.0)
        o_win = _respond(alt, 13.0)
        assert o_win.reinforced is True
        assert o_win.meta["alternative_winner"] == "first"


class TestEventForwarding:
    def test_event_forwarded_to_both_components(self) -> None:
        """Both components must see every response for their own validation."""
        # Construct an Alternative where the *second* component is response-based.
        # If the event were only sent to the first, the second's FR would never fire.
        alt = Alternative(FT(100.0), FR(3))

        _respond(alt, 1.0)
        _respond(alt, 2.0)
        o = _respond(alt, 3.0)
        assert o.reinforced is True
        assert o.meta["alternative_winner"] == "second"


class TestStateErrors:
    def test_non_monotonic_now_raises(self) -> None:
        alt = Alternative(FR(3), FT(5.0))
        alt.step(1.0)
        with pytest.raises(ScheduleStateError):
            alt.step(0.5)

    def test_event_time_not_equal_now_raises(self) -> None:
        alt = Alternative(FR(3), FT(5.0))
        with pytest.raises(ScheduleStateError):
            alt.step(1.0, ResponseEvent(time=2.0))


class TestWinnerMeta:
    def test_winner_meta_first(self) -> None:
        alt = Alternative(FR(1), FT(100.0))
        o = _respond(alt, 0.5)
        assert o.reinforced is True
        assert o.meta["alternative_winner"] == "first"

    def test_winner_meta_second(self) -> None:
        alt = Alternative(FR(100), FT(1.0))
        alt.step(0.0)
        o = alt.step(1.0)
        assert o.reinforced is True
        assert o.meta["alternative_winner"] == "second"

    def test_non_reinforced_outcome_has_empty_meta(self) -> None:
        alt = Alternative(FR(3), FT(5.0))
        o = alt.step(0.5)
        assert o.reinforced is False
        assert "alternative_winner" not in o.meta


class TestCompositionWithVariableSchedules:
    def test_alternative_with_vr_and_vi_runs(self) -> None:
        """Alternative composes with variable-family schedules without error."""
        alt = Alternative(VR(5.0, seed=0), VI(3.0, seed=0))

        # Drive some responses; just ensure no error and some reinforcement occurs.
        n_reinforced = 0
        for i in range(1, 101):
            t = i * 0.1
            o = _respond(alt, t)
            if o.reinforced:
                n_reinforced += 1
                assert o.meta["alternative_winner"] in ("first", "second")

        assert n_reinforced > 0


class TestChainingAlternatives:
    def test_three_component_chain_via_nested_alternatives(self) -> None:
        """For 3+ components, chain Alternatives; the contract still holds."""
        # Alternative(Alternative(FR3, FT100), FR5): fires at min of 3 responses,
        # 100 s, or 5 responses (trivially the FR3 with these params).
        alt = Alternative(Alternative(FR(3), FT(100.0)), FR(5))

        _respond(alt, 1.0)
        _respond(alt, 2.0)
        o = _respond(alt, 3.0)
        assert o.reinforced is True
        # The inner Alternative wins → outer reports "first".
        assert o.meta["alternative_winner"] == "first"
