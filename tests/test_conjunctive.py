"""Tests for the Conjunctive (CONJ, AND) compound schedule.

A conjunctive schedule reinforces when BOTH component schedules
satisfy their criteria on the same step. Contrast with
:class:`Alternative` (OR).

References
----------
Ferster, C. B., & Skinner, B. F. (1957). *Schedules of reinforcement*
(pp. 508-509). Appleton-Century-Crofts.
"""

from __future__ import annotations

import pytest

from contingency.entities import Outcome, ResponseEvent
from contingency.errors import ScheduleConfigError, ScheduleStateError
from contingency.interfaces import Schedule
from contingency.schedules import FI, FR, FT, Conjunctive


def _respond(s: Schedule, now: float) -> Outcome:
    return s.step(now, ResponseEvent(time=now))


class TestProtocolCompliance:
    def test_conjunctive_satisfies_schedule_protocol(self) -> None:
        assert isinstance(Conjunctive(FR(3), FI(5.0)), Schedule)


class TestConstruction:
    def test_accepts_two_schedule_components(self) -> None:
        conj = Conjunctive(FR(3), FI(5.0))
        assert isinstance(conj, Schedule)

    def test_component_properties_expose_originals(self) -> None:
        first = FR(3)
        second = FI(5.0)
        conj = Conjunctive(first, second)
        assert conj.first is first
        assert conj.second is second

    def test_rejects_non_schedule_first_component(self) -> None:
        with pytest.raises(ScheduleConfigError):
            Conjunctive("not a schedule", FI(5.0))  # type: ignore[arg-type]

    def test_rejects_non_schedule_second_component(self) -> None:
        with pytest.raises(ScheduleConfigError):
            Conjunctive(FR(3), 42)  # type: ignore[arg-type]

    def test_rejects_both_non_schedule(self) -> None:
        with pytest.raises(ScheduleConfigError):
            Conjunctive(None, None)  # type: ignore[arg-type]


class TestFRFI:
    def test_no_reinforcement_until_both_criteria_met(self) -> None:
        """Conj(FR3, FI5): nothing before 3 responses AND 5 s elapsed."""
        conj = Conjunctive(FR(3), FI(5.0))
        # Anchor FI clock.
        conj.step(0.0)
        # Three responses quickly — FR3 would fire, but FI5 hasn't elapsed.
        for t in (1.0, 2.0, 3.0):
            out = _respond(conj, t)
            assert out.reinforced is False

    def test_reinforces_when_both_fire_same_step(self) -> None:
        """Conj(FR3, FI5): on the 3rd response AT t>=5s, both fire → reinforced.

        After the 3 quick responses above the FR3 has reset its count.
        To make both fire on the same step, arrange a fresh Conjunctive
        where the response crossing the FI threshold is also the FR-th.
        """
        conj = Conjunctive(FR(3), FI(5.0))
        conj.step(0.0)
        # Two responses during the FI interval — FR counts to 2 but does
        # not reinforce; FI does not reinforce (interval not yet elapsed).
        _respond(conj, 1.0)
        _respond(conj, 2.0)
        # Third response at t=5.0: FR fires on response 3 AND FI fires
        # (5.0 >= arm_time). Both reinforced → Conj reinforces.
        out = _respond(conj, 5.0)
        assert out.reinforced is True
        assert out.reinforcer is not None
        assert out.reinforcer.time == 5.0
        assert out.meta.get("conjunctive") is True

    def test_does_not_reinforce_when_only_fi_fires(self) -> None:
        """A plain step after FI elapses with no response → no FR satisfaction."""
        conj = Conjunctive(FR(3), FI(5.0))
        conj.step(0.0)
        # No responses; FI can't fire without a response either, but a
        # plain tick past 5 s certainly does not produce reinforcement.
        out = conj.step(6.0)
        assert out.reinforced is False


class TestFRFT:
    def test_fr_alone_firing_is_suppressed(self) -> None:
        """Conj(FR3, FT5): FR fires on response 3 before FT elapses → no reinforcement."""
        conj = Conjunctive(FR(3), FT(5.0))
        conj.step(0.0)
        _respond(conj, 1.0)
        _respond(conj, 2.0)
        out = _respond(conj, 3.0)
        # FR fired, FT did not → suppressed.
        assert out.reinforced is False

    def test_ft_alone_firing_is_suppressed(self) -> None:
        """Conj(FR10, FT1): FT elapses with response, but FR not met → suppressed."""
        conj = Conjunctive(FR(10), FT(1.0))
        conj.step(0.0)
        out = _respond(conj, 1.0)
        # FT would fire (1.0 s elapsed), FR has only 1 response < 10.
        assert out.reinforced is False


class TestResetSemantics:
    def test_both_components_reset_after_joint_win(self) -> None:
        conj = Conjunctive(FR(3), FI(5.0))
        conj.step(0.0)
        _respond(conj, 1.0)
        _respond(conj, 2.0)
        win = _respond(conj, 5.0)
        assert win.reinforced is True

        # After a win, FR's counter is cleared; we need 3 new responses
        # AND a fresh FI interval.
        out = _respond(conj, 6.0)
        assert out.reinforced is False
        out = _respond(conj, 7.0)
        assert out.reinforced is False
        # Third new response at t=11.0: FR=3 AND FI elapsed (5.0+5.0=10.0).
        out = _respond(conj, 11.0)
        assert out.reinforced is True

    def test_explicit_reset_clears_internal_state(self) -> None:
        conj = Conjunctive(FR(3), FI(5.0))
        conj.step(0.0)
        _respond(conj, 1.0)
        _respond(conj, 2.0)

        conj.reset()

        # After reset, monotonic-time bookkeeping is cleared — a step at
        # a time earlier than the previous one should not raise.
        conj.step(0.0)
        _respond(conj, 1.0)
        _respond(conj, 2.0)
        # Third response at t=5.0: both fire.
        out = _respond(conj, 5.0)
        assert out.reinforced is True


class TestEventForwarding:
    def test_event_forwarded_to_both_components(self) -> None:
        """Both components see every response so each maintains its own state."""
        # FR on both sides: joint satisfaction requires both counters
        # to hit their thresholds on the same response.
        conj = Conjunctive(FR(3), FR(3))
        _respond(conj, 1.0)
        _respond(conj, 2.0)
        out = _respond(conj, 3.0)
        # Both FRs fire on the 3rd response simultaneously.
        assert out.reinforced is True


class TestStateErrors:
    def test_non_monotonic_now_raises(self) -> None:
        conj = Conjunctive(FR(3), FI(5.0))
        conj.step(1.0)
        with pytest.raises(ScheduleStateError, match="monotonic"):
            conj.step(0.5)

    def test_event_time_mismatch_raises(self) -> None:
        conj = Conjunctive(FR(3), FI(5.0))
        with pytest.raises(ScheduleStateError, match="event.time"):
            conj.step(1.0, ResponseEvent(time=2.0))


class TestMeta:
    def test_reinforced_outcome_has_conjunctive_flag(self) -> None:
        conj = Conjunctive(FR(1), FR(1))
        out = _respond(conj, 1.0)
        assert out.reinforced is True
        assert out.meta.get("conjunctive") is True

    def test_non_reinforced_outcome_has_empty_meta(self) -> None:
        conj = Conjunctive(FR(3), FI(5.0))
        out = conj.step(0.0)
        assert out.reinforced is False
        assert "conjunctive" not in out.meta


class TestChainingConjunctives:
    def test_three_component_conjunctive_via_nesting(self) -> None:
        """Conj(Conj(FR1, FR1), FR1): all three must fire same response."""
        conj = Conjunctive(Conjunctive(FR(1), FR(1)), FR(1))
        out = _respond(conj, 1.0)
        # On the first response, all FR(1)s reinforce simultaneously.
        assert out.reinforced is True
