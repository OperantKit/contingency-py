"""Tests for the Concurrent schedule with COD / COR gating.

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
            Concurrent({"a": FR(1), "b": FR(1)}, cor=bad)

    def test_non_int_cor_rejected(self) -> None:
        with pytest.raises(ScheduleConfigError):
            Concurrent({"a": FR(1), "b": FR(1)}, cor=1.5)  # type: ignore[arg-type]

    def test_bool_cor_rejected(self) -> None:
        with pytest.raises(ScheduleConfigError):
            Concurrent({"a": FR(1), "b": FR(1)}, cor=True)  # type: ignore[arg-type]

    def test_defaults_accepted(self) -> None:
        sch = Concurrent({"a": FR(1), "b": FR(1)})
        assert sch.cod == 0.0
        assert sch.cor == 0

    def test_positive_cod_and_cor_accepted(self) -> None:
        sch = Concurrent({"a": FR(1), "b": FR(1)}, cod=2.0, cor=3)
        assert sch.cod == 2.0
        assert sch.cor == 3

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
# COR behavior
# ---------------------------------------------------------------------------


class TestConcurrentCOR:
    def test_cor_two_first_response_does_not_arm_cod(self) -> None:
        """With cor=2, the first response on new operandum does not arm COD."""
        sch = Concurrent(
            {"left": FR(1), "right": FR(1)},
            cod=1.0,
            cor=2,
        )
        # Left reinforces normally.
        out = sch.step(0.0, ResponseEvent(0.0, operandum="left"))
        assert out.reinforced

        # First response on right (cor=2): does not yet count as a
        # changeover, so COD is not armed. Reinforcement flows through.
        out = sch.step(0.5, ResponseEvent(0.5, operandum="right"))
        assert out.reinforced
        assert "cod_suppressed" not in out.meta

    def test_cor_two_nth_response_arms_cod(self) -> None:
        """With cor=2, the 2nd consecutive response on new operandum arms COD."""
        sch = Concurrent(
            {"left": FR(1), "right": FR(1)},
            cod=1.0,
            cor=2,
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
            cor=3,
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
        """Going back to the previous operandum resets the COR streak."""
        sch = Concurrent(
            {"left": FR(1), "right": FR(1)},
            cod=1.0,
            cor=3,
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
            cor=3,
        )
        # Establish current operandum = A (no prior switch).
        sch.step(0.0, ResponseEvent(0.0, operandum="A"))
        # Complete a changeover A → B (cor=3).
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
        """cor=0 means every switch is an immediate changeover."""
        sch = Concurrent(
            {"left": FR(1), "right": FR(1)},
            cod=1.0,
            cor=0,
        )
        sch.step(0.0, ResponseEvent(0.0, operandum="left"))
        # 1st switch response is itself the changeover (no COR buffer).
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
            cor=2,
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

    def test_suppression_consumes_component_reinforcement(self) -> None:
        """A suppressed reinforcement still advances the component's state."""
        left = FR(1)
        right = FR(2)
        sch = Concurrent(
            {"left": left, "right": right},
            cod=10.0,
        )
        # First response on left reinforces (no prior changeover).
        sch.step(0.0, ResponseEvent(0.0, operandum="left"))
        # Switch to right: first response on right is part of FR(2) so
        # does not reinforce anyway.
        sch.step(1.0, ResponseEvent(1.0, operandum="right"))
        # Second response on right would reinforce but COD gates it. The
        # component's count still resets (spec: component advances).
        out = sch.step(1.5, ResponseEvent(1.5, operandum="right"))
        assert not out.reinforced
        assert out.meta.get("cod_suppressed") is True
        # To prove the reinforcement was consumed: accumulate more right
        # responses after the COD window. If the FR(2) counter were
        # still at 2, the next response would reinforce. It should
        # instead need two fresh responses to reinforce.
        # First response after COD expires — count goes to 1.
        out = sch.step(12.0, ResponseEvent(12.0, operandum="right"))
        assert not out.reinforced
        # Second response after COD expires — FR(2) fires.
        out = sch.step(12.1, ResponseEvent(12.1, operandum="right"))
        assert out.reinforced
