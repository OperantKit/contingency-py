"""Tests for the Concurrent schedule with COD / FRCO gating.

References
----------
Catania, A. C. (1966). Concurrent performances: Reinforcement
interaction and response independence. *Journal of the Experimental
Analysis of Behavior*, 9(3), 253-263.
https://doi.org/10.1901/jeab.1966.9-253

Herrnstein, R. J. (1961). Relative and absolute strength of response
as a function of frequency of reinforcement. *Journal of the
Experimental Analysis of Behavior*, 4(3), 267-272.
https://doi.org/10.1901/jeab.1961.4-267
"""

from __future__ import annotations

import pytest

from contingency.entities import Outcome, ResponseEvent
from contingency.errors import ScheduleConfigError, ScheduleStateError
from contingency.schedules import Concurrent
from contingency.schedules.differential import DRO
from contingency.schedules.interval import FI
from contingency.schedules.ratio import FR
from contingency.schedules.time_based import EXT, FT

# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------


class TestConcurrentConfig:
    def test_requires_at_least_two_components(self) -> None:
        with pytest.raises(ScheduleConfigError):
            Concurrent({"left": FR(1)})

    def test_zero_components_rejected(self) -> None:
        with pytest.raises(ScheduleConfigError):
            Concurrent({})

    def test_components_must_be_dict(self) -> None:
        with pytest.raises(ScheduleConfigError):
            Concurrent([FR(1), FR(2)])  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad", [-0.1, -1.0, -100.0])
    def test_negative_cod_rejected(self, bad: float) -> None:
        with pytest.raises(ScheduleConfigError):
            Concurrent({"a": FR(1), "b": FR(1)}, cod=bad)

    def test_non_numeric_cod_rejected(self) -> None:
        with pytest.raises(ScheduleConfigError):
            Concurrent({"a": FR(1), "b": FR(1)}, cod="2.0")  # type: ignore[arg-type]

    def test_bool_cod_rejected(self) -> None:
        with pytest.raises(ScheduleConfigError):
            Concurrent({"a": FR(1), "b": FR(1)}, cod=True)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad", [-1, -5, -100])
    def test_negative_cor_rejected(self, bad: int) -> None:
        with pytest.raises(ScheduleConfigError):
            Concurrent({"a": FR(1), "b": FR(1)}, frco=bad)

    def test_non_int_cor_rejected(self) -> None:
        with pytest.raises(ScheduleConfigError):
            Concurrent({"a": FR(1), "b": FR(1)}, frco=1.5)  # type: ignore[arg-type]

    def test_bool_cor_rejected(self) -> None:
        with pytest.raises(ScheduleConfigError):
            Concurrent({"a": FR(1), "b": FR(1)}, frco=True)  # type: ignore[arg-type]

    def test_defaults_accepted(self) -> None:
        sch = Concurrent({"a": FR(1), "b": FR(1)})
        assert sch.cod == 0.0
        assert sch.frco == 0

    def test_positive_cod_and_cor_accepted(self) -> None:
        sch = Concurrent({"a": FR(1), "b": FR(1)}, cod=2.0, frco=3)
        assert sch.cod == 2.0
        assert sch.frco == 3

    def test_components_property_returns_copy(self) -> None:
        a, b = FR(1), FR(1)
        sch = Concurrent({"a": a, "b": b})
        snapshot = sch.components
        snapshot["c"] = FR(1)
        # Mutating the snapshot must not alter the schedule.
        assert "c" not in sch.components


# ---------------------------------------------------------------------------
# Basic routing (COD=0)
# ---------------------------------------------------------------------------


class TestConcurrentRouting:
    def test_two_fr_components_independent_reinforcement(self) -> None:
        """conc FR3 FR5: each operandum reinforces on its own count only."""
        sch = Concurrent({"left": FR(3), "right": FR(5)})

        # Two left responses: no reinforcement yet.
        assert not sch.step(0.1, ResponseEvent(0.1, operandum="left")).reinforced
        assert not sch.step(0.2, ResponseEvent(0.2, operandum="left")).reinforced
        # Third left response: FR3 fires.
        out = sch.step(0.3, ResponseEvent(0.3, operandum="left"))
        assert out.reinforced
        assert out.reinforcer is not None
        assert out.reinforcer.time == 0.3

        # Right responses must accumulate independently of the left
        # component's completed ratio; we need 5 on the right.
        for t in (0.4, 0.5, 0.6, 0.7):
            assert not sch.step(t, ResponseEvent(t, operandum="right")).reinforced
        out = sch.step(0.8, ResponseEvent(0.8, operandum="right"))
        assert out.reinforced
        assert out.reinforcer is not None
        assert out.reinforcer.time == 0.8

    def test_three_component_routing(self) -> None:
        """Keys 'A', 'B', 'C' route correctly to their components."""
        sch = Concurrent(
            {"A": FR(1), "B": FR(2), "C": FR(3)},
        )
        # Every A response reinforces (FR1).
        assert sch.step(0.1, ResponseEvent(0.1, operandum="A")).reinforced
        # First B response does not (FR2 needs 2).
        assert not sch.step(0.2, ResponseEvent(0.2, operandum="B")).reinforced
        # Second B response reinforces.
        assert sch.step(0.3, ResponseEvent(0.3, operandum="B")).reinforced
        # C needs 3.
        assert not sch.step(0.4, ResponseEvent(0.4, operandum="C")).reinforced
        assert not sch.step(0.5, ResponseEvent(0.5, operandum="C")).reinforced
        assert sch.step(0.6, ResponseEvent(0.6, operandum="C")).reinforced

    def test_unknown_operandum_raises_config_error(self) -> None:
        sch = Concurrent({"left": FR(1), "right": FR(1)})
        with pytest.raises(ScheduleConfigError):
            sch.step(0.1, ResponseEvent(0.1, operandum="center"))

    def test_tick_without_event_no_reinforcement_for_response_based(self) -> None:
        sch = Concurrent({"left": FR(1), "right": FR(1)})
        for t in (0.0, 1.0, 10.0, 100.0):
            out = sch.step(t)
            assert not out.reinforced
            assert out.reinforcer is None


# ---------------------------------------------------------------------------
# COD behavior
# ---------------------------------------------------------------------------


class TestConcurrentCOD:
    def test_first_response_not_gated(self) -> None:
        """The very first response of a session is never a changeover."""
        sch = Concurrent(
            {"left": FR(1), "right": FR(1)},
            cod=1.0,
        )
        out = sch.step(0.0, ResponseEvent(0.0, operandum="left"))
        assert out.reinforced
        assert out.reinforcer is not None

    def test_switch_within_cod_suppresses_reinforcement(self) -> None:
        """Switch from left to right at t=0.5: right is gated until t=1.5."""
        sch = Concurrent(
            {"left": FR(1), "right": FR(1)},
            cod=1.0,
        )
        # Establish left as the active operandum.
        sch.step(0.0, ResponseEvent(0.0, operandum="left"))

        # Switch to right at t=0.5 — within COD, must NOT reinforce.
        out = sch.step(0.5, ResponseEvent(0.5, operandum="right"))
        assert out.reinforced is False
        assert out.reinforcer is None
        assert out.meta.get("cod_suppressed") is True
        assert out.meta.get("operandum") == "right"

    def test_response_on_new_operandum_after_cod_reinforces(self) -> None:
        sch = Concurrent(
            {"left": FR(1), "right": FR(1)},
            cod=1.0,
        )
        sch.step(0.0, ResponseEvent(0.0, operandum="left"))
        sch.step(0.5, ResponseEvent(0.5, operandum="right"))  # suppressed
        out = sch.step(1.6, ResponseEvent(1.6, operandum="right"))
        assert out.reinforced
        assert out.reinforcer is not None
        assert out.reinforcer.time == 1.6

    def test_repeated_switches_reset_cod_timer(self) -> None:
        """Each switch re-anchors the COD clock."""
        sch = Concurrent(
            {"left": FR(1), "right": FR(1)},
            cod=1.0,
        )
        sch.step(0.0, ResponseEvent(0.0, operandum="left"))
        # Switch #1 at t=0.2 → armed at 0.2.
        out = sch.step(0.2, ResponseEvent(0.2, operandum="right"))
        assert not out.reinforced
        # Switch #2 at t=0.5 back to left → armed at 0.5.
        out = sch.step(0.5, ResponseEvent(0.5, operandum="left"))
        assert not out.reinforced
        # Switch #3 at t=0.9 to right → armed at 0.9.
        out = sch.step(0.9, ResponseEvent(0.9, operandum="right"))
        assert not out.reinforced
        # At t=1.95, 1.05 has elapsed since last switch → reinforced.
        out = sch.step(1.95, ResponseEvent(1.95, operandum="right"))
        assert out.reinforced

    def test_non_switching_responses_not_gated(self) -> None:
        """Responses that stay on the current operandum are never gated."""
        sch = Concurrent(
            {"left": FR(1), "right": FR(1)},
            cod=5.0,
        )
        # First response establishes left.
        out = sch.step(0.0, ResponseEvent(0.0, operandum="left"))
        assert out.reinforced
        # Subsequent left responses should keep reinforcing even though
        # cod=5.0 is non-trivial: no switch → no COD.
        for t in (0.1, 0.2, 0.3):
            out = sch.step(t, ResponseEvent(t, operandum="left"))
            assert out.reinforced

    def test_cod_zero_never_suppresses(self) -> None:
        sch = Concurrent(
            {"left": FR(1), "right": FR(1)},
            cod=0.0,
        )
        sch.step(0.0, ResponseEvent(0.0, operandum="left"))
        out = sch.step(0.0001, ResponseEvent(0.0001, operandum="right"))
        assert out.reinforced
        assert "cod_suppressed" not in out.meta

    def test_cod_exactly_at_boundary_is_not_suppressed(self) -> None:
        """At now - switch_time == cod, the lockout has just ended."""
        sch = Concurrent(
            {"left": FR(1), "right": FR(1)},
            cod=1.0,
        )
        sch.step(0.0, ResponseEvent(0.0, operandum="left"))
        sch.step(0.5, ResponseEvent(0.5, operandum="right"))  # arms at 0.5
        # At t=1.5, delta is exactly 1.0 — treat as past the edge.
        out = sch.step(1.5, ResponseEvent(1.5, operandum="right"))
        assert out.reinforced


# ---------------------------------------------------------------------------
# FRCO behavior
# ---------------------------------------------------------------------------


class TestConcurrentCOR:
    def test_cor_two_first_response_does_not_arm_cod(self) -> None:
        """With frco=2, the first response on new operandum does not arm COD."""
        sch = Concurrent(
            {"left": FR(1), "right": FR(1)},
            cod=1.0,
            frco=2,
        )
        # Left reinforces normally.
        out = sch.step(0.0, ResponseEvent(0.0, operandum="left"))
        assert out.reinforced

        # First response on right (frco=2): does not yet count as a
        # changeover, so COD is not armed. Reinforcement flows through.
        out = sch.step(0.5, ResponseEvent(0.5, operandum="right"))
        assert out.reinforced
        assert "cod_suppressed" not in out.meta

    def test_cor_two_nth_response_arms_cod(self) -> None:
        """With frco=2, the 2nd consecutive response on new operandum arms COD."""
        sch = Concurrent(
            {"left": FR(1), "right": FR(1)},
            cod=1.0,
            frco=2,
        )
        sch.step(0.0, ResponseEvent(0.0, operandum="left"))
        # 1st right: not a changeover yet.
        sch.step(0.5, ResponseEvent(0.5, operandum="right"))
        # 2nd right at t=0.6: completes the changeover, arms COD at 0.6.
        # That response itself is covered by COD and thus suppressed.
        out = sch.step(0.6, ResponseEvent(0.6, operandum="right"))
        assert not out.reinforced
        assert out.meta.get("cod_suppressed") is True

    def test_cor_three_requires_three_consecutive(self) -> None:
        sch = Concurrent(
            {"left": FR(1), "right": FR(1)},
            cod=1.0,
            frco=3,
        )
        sch.step(0.0, ResponseEvent(0.0, operandum="left"))
        # 1st and 2nd right — no changeover yet.
        assert sch.step(0.1, ResponseEvent(0.1, operandum="right")).reinforced
        assert sch.step(0.2, ResponseEvent(0.2, operandum="right")).reinforced
        # 3rd right: completes the changeover, arms COD at 0.3, and is
        # therefore suppressed.
        out = sch.step(0.3, ResponseEvent(0.3, operandum="right"))
        assert not out.reinforced
        assert out.meta.get("cod_suppressed") is True

    def test_cor_streak_broken_by_returning_to_old_operandum(self) -> None:
        """Going back to the previous operandum resets the FRCO streak."""
        sch = Concurrent(
            {"left": FR(1), "right": FR(1)},
            cod=1.0,
            frco=3,
        )
        sch.step(0.0, ResponseEvent(0.0, operandum="left"))
        sch.step(0.1, ResponseEvent(0.1, operandum="right"))
        sch.step(0.2, ResponseEvent(0.2, operandum="right"))
        # Break streak by going back to left.
        sch.step(0.3, ResponseEvent(0.3, operandum="left"))
        # Start over on right — need 3 fresh responses.
        sch.step(0.4, ResponseEvent(0.4, operandum="right"))
        sch.step(0.5, ResponseEvent(0.5, operandum="right"))
        out = sch.step(0.6, ResponseEvent(0.6, operandum="right"))
        # This 3rd-in-a-row response arms COD → suppressed.
        assert not out.reinforced
        assert out.meta.get("cod_suppressed") is True

    def test_cor_streak_on_third_operandum_not_gated_by_prior_cod(self) -> None:
        """Responses on the in-progress streak operandum are not gated.

        After a complete changeover to B (so switch_time is armed),
        partial streak responses toward a third operandum A do not
        sit on B and therefore are not gated by the existing COD. This
        confirms that ``_cod_active`` returns False when the queried
        operandum does not match ``_last_operandum``.
        """
        sch = Concurrent(
            {"A": FR(1), "B": FR(1), "C": FR(1)},
            cod=10.0,
            frco=3,
        )
        # Establish current operandum = A (no prior switch).
        sch.step(0.0, ResponseEvent(0.0, operandum="A"))
        # Complete a changeover A → B (frco=3).
        sch.step(0.1, ResponseEvent(0.1, operandum="B"))
        sch.step(0.2, ResponseEvent(0.2, operandum="B"))
        # 3rd B response arms COD on B at t=0.3 and is suppressed.
        out = sch.step(0.3, ResponseEvent(0.3, operandum="B"))
        assert not out.reinforced
        # Begin a new streak B → C. The 1st C response does not yet
        # complete the changeover, so _last_operandum is still B. A
        # response on C therefore is not gated by the B-anchored COD.
        out = sch.step(0.4, ResponseEvent(0.4, operandum="C"))
        assert out.reinforced
        assert "cod_suppressed" not in out.meta

    def test_cor_zero_immediate_switch(self) -> None:
        """frco=0 means every switch is an immediate changeover."""
        sch = Concurrent(
            {"left": FR(1), "right": FR(1)},
            cod=1.0,
            frco=0,
        )
        sch.step(0.0, ResponseEvent(0.0, operandum="left"))
        # 1st switch response is itself the changeover (no FRCO buffer).
        out = sch.step(0.5, ResponseEvent(0.5, operandum="right"))
        assert not out.reinforced
        assert out.meta.get("cod_suppressed") is True


# ---------------------------------------------------------------------------
# Time advancement for time-based components
# ---------------------------------------------------------------------------


class TestConcurrentTimeAdvancement:
    def test_tick_advances_ft_component(self) -> None:
        """A pure tick advances an FT component, which may fire without responses."""
        sch = Concurrent(
            {"left": FT(1.0), "right": FR(1)},
        )
        # Anchor both components at t=0.
        sch.step(0.0)
        # At t=1.0, the FT component on "left" fires on its own.
        out = sch.step(1.0)
        assert out.reinforced
        assert out.reinforcer is not None
        assert out.reinforcer.time == 1.0

    def test_event_on_right_still_ticks_left_ft(self) -> None:
        """Even if the event targets 'right', 'left' (FT) continues to tick."""
        sch = Concurrent(
            {"left": FT(1.0), "right": FR(2)},
        )
        sch.step(0.0)  # anchor
        sch.step(0.5, ResponseEvent(0.5, operandum="right"))  # tick both
        # t=1.0: bare tick, FT should fire on left.
        out = sch.step(1.0)
        assert out.reinforced

    def test_ext_component_never_reinforces(self) -> None:
        sch = Concurrent(
            {"left": EXT(), "right": FR(1)},
        )
        for t in range(10):
            out = sch.step(float(t), ResponseEvent(float(t), operandum="left"))
            assert not out.reinforced

    def test_event_on_unknown_even_with_time_component_raises(self) -> None:
        sch = Concurrent({"left": FT(1.0), "right": FR(1)})
        sch.step(0.0)
        with pytest.raises(ScheduleConfigError):
            sch.step(0.5, ResponseEvent(0.5, operandum="nowhere"))

    def test_tick_reinforcement_surfaces_even_when_event_present(self) -> None:
        """FT on left fires at t=1.0; a concurrent response on right must
        not silently consume that reinforcement."""
        sch = Concurrent({"left": FT(1.0), "right": FR(3)})
        sch.step(0.0)  # anchor left's FT clock at 0.0
        out = sch.step(1.0, ResponseEvent(1.0, operandum="right"))
        assert out.reinforced
        assert out.reinforcer is not None
        assert out.reinforcer.time == 1.0

    def test_tick_reinforcement_not_gated_by_event_cod(self) -> None:
        """A tick reinforcement from FT on one operandum must not be
        COD-gated by the event operandum's changeover state."""
        sch = Concurrent(
            {"left": FT(1.0), "right": FR(3)},
            cod=5.0,
        )
        sch.step(0.0, ResponseEvent(0.0, operandum="right"))
        # Right is now _last_operandum with no switch yet; FT on left
        # arms a reinforcer at t=1.0 independent of the right component.
        out = sch.step(1.0, ResponseEvent(1.0, operandum="right"))
        assert out.reinforced
        assert "cod_suppressed" not in out.meta

    def test_tick_reinforcement_survives_hold_on_event_side(self) -> None:
        """On a step where COD holds the event-side reinforcer, a
        reinforcer another component delivers on its tick is still
        returned."""
        sch = Concurrent({"a": FT(5.0), "b": FI(3.0)}, cod=2.0)
        results = [
            sch.step(t, ResponseEvent(t, operandum=op)).reinforced
            for t, op in [(0.0, "a"), (4.0, "b"), (5.0, "b"), (7.0, "b")]
        ]
        assert results == [False, False, True, True]

    def test_tick_reinforcer_on_hold_step_records_hold(self) -> None:
        sch = Concurrent({"a": FT(5.0), "b": FI(3.0)}, cod=2.0)
        sch.step(0.0, ResponseEvent(0.0, operandum="a"))
        sch.step(4.0, ResponseEvent(4.0, operandum="b"))
        out = sch.step(5.0, ResponseEvent(5.0, operandum="b"))
        assert out.reinforcer is not None
        assert out.reinforcer.time == 5.0
        assert out.meta.get("cod_suppressed") is True
        assert "operandum" not in out.meta

    def test_tick_reinforcement_survives_queued_hold(self) -> None:
        """Same as above when the event-side reinforcer is queued by
        :class:`Concurrent` (a component it does not hold in place)."""
        sch = Concurrent({"a": FT(5.0), "b": FT(1.0)}, cod=2.0)
        results = [
            sch.step(t, ResponseEvent(t, operandum=op)).reinforced
            for t, op in [(0.0, "a"), (4.0, "b"), (5.0, "b"), (7.0, "b")]
        ]
        assert results == [False, False, True, True]

    def test_punisher_on_hold_step_still_wins_over_tick(self) -> None:
        """Unchanged behaviour: when a punisher fires on the hold step,
        the punisher outcome is returned and the tick reinforcer is not."""
        sch = Concurrent(
            {"a": FT(5.0), "b": FI(3.0)}, cod=2.0, punish={"b": FR(1)}
        )
        sch.step(0.0, ResponseEvent(0.0, operandum="a"))
        sch.step(4.0, ResponseEvent(4.0, operandum="b"))
        out = sch.step(5.0, ResponseEvent(5.0, operandum="b"))
        assert out.reinforcer is not None
        assert out.reinforcer.label == "SR-"
        assert out.meta.get("cod_suppressed") is True


# ---------------------------------------------------------------------------
# State-level error handling
# ---------------------------------------------------------------------------


class TestConcurrentStateErrors:
    def test_non_monotonic_time_raises(self) -> None:
        sch = Concurrent({"a": FR(1), "b": FR(1)})
        sch.step(1.0)
        with pytest.raises(ScheduleStateError):
            sch.step(0.5)

    def test_non_monotonic_time_with_event_raises(self) -> None:
        sch = Concurrent({"a": FR(1), "b": FR(1)})
        sch.step(1.0, ResponseEvent(1.0, operandum="a"))
        with pytest.raises(ScheduleStateError):
            sch.step(0.5, ResponseEvent(0.5, operandum="a"))

    def test_event_time_mismatches_now_raises(self) -> None:
        sch = Concurrent({"a": FR(1), "b": FR(1)})
        with pytest.raises(ScheduleStateError):
            sch.step(1.0, ResponseEvent(1.1, operandum="a"))

    def test_time_within_tolerance_accepted(self) -> None:
        sch = Concurrent({"a": FR(1), "b": FR(1)})
        sch.step(1.0)
        sch.step(1.0 - 1e-12)  # within tolerance


# ---------------------------------------------------------------------------
# Reset semantics
# ---------------------------------------------------------------------------


class TestConcurrentReset:
    def test_reset_clears_changeover_state(self) -> None:
        sch = Concurrent(
            {"left": FR(1), "right": FR(1)},
            cod=1.0,
        )
        sch.step(0.0, ResponseEvent(0.0, operandum="left"))
        sch.step(0.5, ResponseEvent(0.5, operandum="right"))  # suppressed
        sch.reset()
        # After reset, first response on any operandum is not a changeover.
        out = sch.step(0.0, ResponseEvent(0.0, operandum="right"))
        assert out.reinforced

    def test_reset_resets_component_state(self) -> None:
        """Components are reset; e.g. an FR3 counter goes back to 0."""
        sch = Concurrent({"left": FR(3), "right": FR(3)})
        # Push left partway through an FR3.
        sch.step(0.0, ResponseEvent(0.0, operandum="left"))
        sch.step(0.1, ResponseEvent(0.1, operandum="left"))
        # Not yet reinforced (only 2 of 3 responses).
        sch.reset()
        # After reset, the next 3 should be needed again.
        assert not sch.step(0.2, ResponseEvent(0.2, operandum="left")).reinforced
        assert not sch.step(0.3, ResponseEvent(0.3, operandum="left")).reinforced
        assert sch.step(0.4, ResponseEvent(0.4, operandum="left")).reinforced

    def test_reset_clears_last_now(self) -> None:
        """After reset, stepping backwards in time is allowed (fresh clock)."""
        sch = Concurrent({"a": FR(1), "b": FR(1)})
        sch.step(10.0)
        sch.reset()
        # Stepping back to a small now does not raise after reset.
        sch.step(0.0)

    def test_reset_clears_cor_streak(self) -> None:
        sch = Concurrent(
            {"left": FR(1), "right": FR(1)},
            cod=1.0,
            frco=2,
        )
        sch.step(0.0, ResponseEvent(0.0, operandum="left"))
        sch.step(0.1, ResponseEvent(0.1, operandum="right"))  # streak=1
        sch.reset()
        # After reset, no streak and no prior operandum: first response
        # on right reinforces, then 'right' is the current operandum.
        assert sch.step(0.0, ResponseEvent(0.0, operandum="right")).reinforced


# ---------------------------------------------------------------------------
# Meta payload on suppression
# ---------------------------------------------------------------------------


class TestConcurrentSuppressionMeta:
    def test_suppressed_outcome_has_meta(self) -> None:
        sch = Concurrent(
            {"left": FR(1), "right": FR(1)},
            cod=1.0,
        )
        sch.step(0.0, ResponseEvent(0.0, operandum="left"))
        out = sch.step(0.3, ResponseEvent(0.3, operandum="right"))
        assert isinstance(out, Outcome)
        assert out.reinforced is False
        assert out.reinforcer is None
        assert out.meta == {"cod_suppressed": True, "operandum": "right"}

    def test_non_suppressed_outcome_has_no_cod_meta(self) -> None:
        sch = Concurrent(
            {"left": FR(1), "right": FR(1)},
            cod=1.0,
        )
        out = sch.step(0.0, ResponseEvent(0.0, operandum="left"))
        assert "cod_suppressed" not in out.meta

    def test_suppressed_reinforcer_is_held_until_cod_ends(self) -> None:
        """A reinforcer set up on B is delivered by the first B response after COD."""
        sch = Concurrent({"a": FR(1), "b": FI(5.0)}, cod=1.5)
        sch.step(0.0, ResponseEvent(0.0, operandum="a"))
        # FI 5 on b is set up at t=5. Switch a -> b at t=6: within COD.
        out = sch.step(6.0, ResponseEvent(6.0, operandum="b"))
        assert not out.reinforced
        assert out.meta.get("cod_suppressed") is True
        # First b response after the 1.5-s COD collects the held reinforcer.
        out = sch.step(7.6, ResponseEvent(7.6, operandum="b"))
        assert out.reinforced
        assert out.reinforcer is not None
        assert out.reinforcer.time == 7.6

    def test_held_reinforcer_survives_switching_back(self) -> None:
        """Leaving B during COD does not forfeit the reinforcer set up on B."""
        sch = Concurrent({"a": FR(100), "b": FI(5.0)}, cod=1.5)
        sch.step(0.0, ResponseEvent(0.0, operandum="a"))
        assert not sch.step(6.0, ResponseEvent(6.0, operandum="b")).reinforced
        sch.step(6.5, ResponseEvent(6.5, operandum="a"))
        # Back to b: a new COD starts at t=7.
        assert not sch.step(7.0, ResponseEvent(7.0, operandum="b")).reinforced
        assert sch.step(8.6, ResponseEvent(8.6, operandum="b")).reinforced

    def test_next_interval_starts_at_delivery_not_at_setup(self) -> None:
        """While the reinforcer is held, the component's next interval does not run."""
        sch = Concurrent({"a": FR(1), "b": FI(5.0)}, cod=1.5)
        sch.step(0.0, ResponseEvent(0.0, operandum="a"))
        sch.step(6.0, ResponseEvent(6.0, operandum="b"))  # held
        assert sch.step(7.6, ResponseEvent(7.6, operandum="b")).reinforced
        # Next FI 5 is timed from the delivery at 7.6, so it sets up at 12.6.
        assert not sch.step(12.0, ResponseEvent(12.0, operandum="b")).reinforced
        assert sch.step(12.7, ResponseEvent(12.7, operandum="b")).reinforced

    def test_time_based_reinforcer_is_held_not_lost(self) -> None:
        """An FT reinforcer that falls on a response inside COD is paid after it."""
        sch = Concurrent({"a": FR(100), "b": FT(5.0)}, cod=2.0)
        sch.step(0.0, ResponseEvent(0.0, operandum="a"))
        # FT 5 fires on the b response at 5.5, which opens a 2-s COD.
        out = sch.step(5.5, ResponseEvent(5.5, operandum="b"))
        assert not out.reinforced
        assert out.meta.get("cod_suppressed") is True
        out = sch.step(7.6, ResponseEvent(7.6, operandum="b"))
        assert out.reinforced
        assert out.reinforcer is not None
        assert out.reinforcer.time == 7.6

    def test_responses_inside_cod_still_reach_dro(self) -> None:
        """A response inside COD resets DRO even when it coincides with a DRO payout."""
        sch = Concurrent(
            {"a": FR(100), "b": DRO(2.0, dro_type="momentary")}, cod=1.0
        )
        sch.step(0.0, ResponseEvent(0.0, operandum="a"))
        # The b response at 2.0 lands on the DRO boundary inside COD.
        out = sch.step(2.0, ResponseEvent(2.0, operandum="b"))
        assert not out.reinforced
        sch.step(3.5)
        # [2, 4) contains the b response at 2.0, so DRO must not pay at 4.0.
        assert not sch.step(4.0).reinforced
        # The DRO reinforcer set up at 2.0 is collected by the next b response.
        assert sch.step(4.5, ResponseEvent(4.5, operandum="b")).reinforced

    def test_held_ratio_reinforcer_restarts_count_after_delivery(self) -> None:
        """For FR, the ratio completed during COD is paid on the first response after it."""
        sch = Concurrent({"left": FR(1), "right": FR(2)}, cod=10.0)
        sch.step(0.0, ResponseEvent(0.0, operandum="left"))
        sch.step(1.0, ResponseEvent(1.0, operandum="right"))
        # Ratio completed inside COD: held, not delivered.
        out = sch.step(1.5, ResponseEvent(1.5, operandum="right"))
        assert not out.reinforced
        assert out.meta.get("cod_suppressed") is True
        # First response after COD collects it.
        assert sch.step(12.0, ResponseEvent(12.0, operandum="right")).reinforced
        # The count then restarts: two fresh responses are needed.
        assert not sch.step(12.1, ResponseEvent(12.1, operandum="right")).reinforced
        assert sch.step(12.2, ResponseEvent(12.2, operandum="right")).reinforced


# ---------------------------------------------------------------------------
# DirectionalCOD
# ---------------------------------------------------------------------------


class TestConcurrentDirectionalCOD:
    def test_rejects_non_dict_cod_directional(self) -> None:
        with pytest.raises(ScheduleConfigError):
            Concurrent(
                {"a": FR(1), "b": FR(1)},
                cod_directional=[("a", "b", 2.0)],  # type: ignore[arg-type]
            )

    def test_rejects_non_tuple_key(self) -> None:
        with pytest.raises(ScheduleConfigError):
            Concurrent(
                {"a": FR(1), "b": FR(1)},
                cod_directional={"a->b": 2.0},  # type: ignore[dict-item]
            )

    def test_rejects_self_transition(self) -> None:
        with pytest.raises(ScheduleConfigError):
            Concurrent(
                {"a": FR(1), "b": FR(1)},
                cod_directional={("a", "a"): 2.0},
            )

    def test_rejects_negative_directional_value(self) -> None:
        with pytest.raises(ScheduleConfigError):
            Concurrent(
                {"a": FR(1), "b": FR(1)},
                cod_directional={("a", "b"): -1.0},
            )

    def test_directional_cod_governs_each_direction(self) -> None:
        """A->B COD = 2s, B->A COD = 1s."""
        sch = Concurrent(
            {"a": FR(1), "b": FR(1)},
            cod=0.0,
            cod_directional={("a", "b"): 2.0, ("b", "a"): 1.0},
        )
        # Anchor on 'a'.
        assert sch.step(0.0, ResponseEvent(0.0, operandum="a")).reinforced
        # Switch a->b at t=1.5: within 2s COD -> suppressed.
        out = sch.step(1.5, ResponseEvent(1.5, operandum="b"))
        assert not out.reinforced
        assert out.meta.get("cod_suppressed") is True
        # Wait past the 2s COD (a->b): t=2.5 = 1.0s after switch at 1.5,
        # still suppressed. t=3.6 => 2.1s after -> allowed.
        out = sch.step(3.6, ResponseEvent(3.6, operandum="b"))
        assert out.reinforced
        # Switch b->a at t=4.0: 1s COD applies.
        out = sch.step(4.0, ResponseEvent(4.0, operandum="a"))
        assert not out.reinforced
        assert out.meta.get("cod_suppressed") is True
        # After 1s (t=5.1) -> allowed.
        out = sch.step(5.1, ResponseEvent(5.1, operandum="a"))
        assert out.reinforced

    def test_base_cod_fallback_when_directional_entry_missing(self) -> None:
        """When directional entry is missing, fall back to base cod."""
        sch = Concurrent(
            {"a": FR(1), "b": FR(1), "c": FR(1)},
            cod=0.5,
            cod_directional={("a", "b"): 2.0},  # only a->b overridden
        )
        # Anchor on 'a', then switch a->c: uses base cod (0.5).
        assert sch.step(0.0, ResponseEvent(0.0, operandum="a")).reinforced
        out = sch.step(0.1, ResponseEvent(0.1, operandum="c"))
        # 0.1s < 0.5s -> suppressed by base cod.
        assert not out.reinforced
        assert out.meta.get("cod_suppressed") is True
        # Past base cod at t=0.7.
        out = sch.step(0.7, ResponseEvent(0.7, operandum="c"))
        assert out.reinforced

    def test_empty_dict_equivalent_to_no_directional(self) -> None:
        sch = Concurrent(
            {"a": FR(1), "b": FR(1)},
            cod=0.0,
            cod_directional={},
        )
        assert sch.cod_directional is None

    def test_property_returns_copy(self) -> None:
        sch = Concurrent(
            {"a": FR(1), "b": FR(1)},
            cod_directional={("a", "b"): 2.0},
        )
        snap = sch.cod_directional
        assert snap is not None
        snap[("x", "y")] = 9.0
        # Original must be unaffected.
        assert ("x", "y") not in (sch.cod_directional or {})


# ---------------------------------------------------------------------------
# PunishParam (component form)
# ---------------------------------------------------------------------------


class TestConcurrentPunish:
    def test_rejects_non_dict_punish(self) -> None:
        with pytest.raises(ScheduleConfigError):
            Concurrent(
                {"a": FR(1), "b": FR(1)},
                punish=[("a", FR(1))],  # type: ignore[arg-type]
            )

    def test_rejects_punish_for_unknown_operandum(self) -> None:
        with pytest.raises(ScheduleConfigError):
            Concurrent(
                {"a": FR(1), "b": FR(1)},
                punish={"c": FR(2)},
            )

    def test_rejects_non_schedule_punish_value(self) -> None:
        with pytest.raises(ScheduleConfigError):
            Concurrent(
                {"a": FR(1), "b": FR(1)},
                punish={"a": 99},  # type: ignore[dict-item]
            )

    def test_reinforcement_and_punishment_net_magnitude(self) -> None:
        """Reinforcement on A and punishment on A on the same response."""
        sch = Concurrent(
            {"a": FR(1), "b": FR(1)},
            punish={"a": FR(1)},
        )
        out = sch.step(0.0, ResponseEvent(0.0, operandum="a"))
        assert out.reinforced
        assert out.meta.get("base_fired") is True
        assert out.meta.get("punisher_fired") is True
        assert out.meta.get("net_magnitude") == pytest.approx(0.0)
        assert out.reinforcer is not None
        assert out.reinforcer.label == "SR0"

    def test_punisher_only_fires_when_base_does_not(self) -> None:
        """FR(2) base, FR(1) punisher: first response fires only punisher."""
        sch = Concurrent(
            {"a": FR(2), "b": FR(1)},
            punish={"a": FR(1)},
        )
        out = sch.step(0.0, ResponseEvent(0.0, operandum="a"))
        assert out.reinforced
        assert out.meta.get("punisher_fired") is True
        assert out.meta.get("base_fired") is False
        assert out.reinforcer is not None
        assert out.reinforcer.label == "SR-"
        assert out.reinforcer.magnitude == pytest.approx(-1.0)

    def test_punish_on_other_operandum_does_not_trigger(self) -> None:
        """Responding on A must not advance B's punisher."""
        sch = Concurrent(
            {"a": FR(1), "b": FR(1)},
            punish={"b": FR(1)},
        )
        out = sch.step(0.0, ResponseEvent(0.0, operandum="a"))
        assert out.reinforced
        assert out.reinforcer is not None
        assert out.reinforcer.label == "SR+"

    def test_punish_dict_property_returns_copy(self) -> None:
        sch = Concurrent(
            {"a": FR(1), "b": FR(1)},
            punish={"a": FR(1)},
        )
        snap = sch.punish
        assert snap is not None
        snap["b"] = FR(9)
        assert "b" not in (sch.punish or {})

    def test_reset_clears_punish_components(self) -> None:
        sch = Concurrent(
            {"a": FR(2), "b": FR(1)},
            punish={"a": FR(2)},
        )
        sch.step(0.0, ResponseEvent(0.0, operandum="a"))
        sch.reset()
        # After reset, first response on 'a': FR(2) base needs 2, FR(2)
        # punisher needs 2 -> neither fires.
        out = sch.step(1.0, ResponseEvent(1.0, operandum="a"))
        assert not out.reinforced
