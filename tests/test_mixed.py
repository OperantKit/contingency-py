"""Tests for the Mixed (MIX) compound schedule.

A mixed schedule alternates components on reinforcement, exactly like
:class:`Multiple` but without distinctive stimuli. The integer active
index is reported in ``Outcome.meta["current_component"]``.

References
----------
Ferster, C. B., & Skinner, B. F. (1957). *Schedules of reinforcement*
(pp. 623-626). Appleton-Century-Crofts.
"""

from __future__ import annotations

import pytest

from contingency.entities import Outcome, ResponseEvent
from contingency.errors import ScheduleConfigError, ScheduleStateError
from contingency.interfaces import Schedule
from contingency.schedules import FR, Mixed


def _respond(s: Schedule, now: float) -> Outcome:
    return s.step(now, ResponseEvent(time=now))


class TestProtocolCompliance:
    def test_mixed_satisfies_schedule_protocol(self) -> None:
        assert isinstance(Mixed([FR(1), FR(1)]), Schedule)


class TestConstruction:
    def test_accepts_list_of_schedules(self) -> None:
        m = Mixed([FR(2), FR(3)])
        assert m.n_components == 2

    def test_accepts_three_components(self) -> None:
        m = Mixed([FR(1), FR(2), FR(3)])
        assert m.n_components == 3

    def test_requires_at_least_two_components(self) -> None:
        with pytest.raises(ScheduleConfigError, match=">= 2"):
            Mixed([FR(1)])

    def test_rejects_empty_components(self) -> None:
        with pytest.raises(ScheduleConfigError):
            Mixed([])

    def test_rejects_non_sequence_components(self) -> None:
        with pytest.raises(ScheduleConfigError, match="sequence"):
            Mixed(42)  # type: ignore[arg-type]

    def test_rejects_string_components(self) -> None:
        with pytest.raises(ScheduleConfigError, match="sequence"):
            Mixed("ab")  # type: ignore[arg-type]

    def test_rejects_non_schedule_component(self) -> None:
        with pytest.raises(ScheduleConfigError, match="Schedule protocol"):
            Mixed([FR(1), "not-a-schedule"])  # type: ignore[list-item]

    def test_active_index_starts_at_zero(self) -> None:
        m = Mixed([FR(1), FR(1)])
        assert m.active_index == 0


class TestAlternationOnReinforcement:
    def test_advances_on_each_reinforcement(self) -> None:
        """Mix(FR1, FR1): every response reinforces; active index alternates."""
        m = Mixed([FR(1), FR(1)])
        assert m.active_index == 0

        out = _respond(m, 1.0)
        assert out.reinforced is True
        assert m.active_index == 1

        out = _respond(m, 2.0)
        assert out.reinforced is True
        assert m.active_index == 0  # wrapped.

    def test_three_components_cycle(self) -> None:
        m = Mixed([FR(1), FR(1), FR(1)])
        seen_indices = []
        for t in range(1, 7):
            out = _respond(m, float(t))
            seen_indices.append(out.meta["current_component"])
        # First 3 responses at components 0, 1, 2; next 3 at 0, 1, 2.
        assert seen_indices == [0, 1, 2, 0, 1, 2]

    def test_non_reinforced_step_does_not_advance(self) -> None:
        m = Mixed([FR(3), FR(3)])
        _respond(m, 1.0)
        _respond(m, 2.0)
        # First two responses on component 0 don't reinforce yet.
        assert m.active_index == 0

    def test_only_active_component_is_stepped(self) -> None:
        """Inactive components do not advance — classical MIX semantics."""
        m = Mixed([FR(3), FR(2)])
        # Three responses → component 0 reinforces, switches to component 1.
        _respond(m, 1.0)
        _respond(m, 2.0)
        _respond(m, 3.0)
        assert m.active_index == 1
        # Now component 1's FR(2) still requires 2 fresh responses.
        assert _respond(m, 4.0).reinforced is False
        assert _respond(m, 5.0).reinforced is True


class TestMetaPayload:
    def test_current_component_is_int_not_str(self) -> None:
        """MIX reports current_component as int, differentiating from Multiple."""
        m = Mixed([FR(1), FR(1)])
        out = _respond(m, 1.0)
        assert isinstance(out.meta["current_component"], int)
        assert out.meta["current_component"] == 0

    def test_non_reinforced_step_still_reports_index(self) -> None:
        m = Mixed([FR(3), FR(3)])
        out = _respond(m, 1.0)
        assert out.reinforced is False
        assert out.meta["current_component"] == 0

    def test_reinforced_step_reports_the_firing_component(self) -> None:
        """The index reported is the component that fired, not the next one."""
        m = Mixed([FR(1), FR(1)])
        out = _respond(m, 1.0)
        assert out.reinforced is True
        assert out.meta["current_component"] == 0
        # The next step will report 1.
        out = _respond(m, 2.0)
        assert out.meta["current_component"] == 1


class TestResetSemantics:
    def test_reset_returns_to_initial_index(self) -> None:
        m = Mixed([FR(1), FR(1), FR(1)])
        _respond(m, 1.0)
        _respond(m, 2.0)
        assert m.active_index == 2
        m.reset()
        assert m.active_index == 0

    def test_reset_clears_component_state(self) -> None:
        m = Mixed([FR(3), FR(3)])
        _respond(m, 1.0)
        _respond(m, 2.0)
        m.reset()
        # After reset, we need 3 fresh responses on component 0.
        assert _respond(m, 10.0).reinforced is False
        assert _respond(m, 11.0).reinforced is False
        assert _respond(m, 12.0).reinforced is True

    def test_reset_allows_time_to_restart(self) -> None:
        m = Mixed([FR(1), FR(1)])
        m.step(10.0)
        m.reset()
        # After reset, an earlier time should not raise non-monotonic.
        m.step(1.0)


class TestStateErrors:
    def test_non_monotonic_now_raises(self) -> None:
        m = Mixed([FR(1), FR(1)])
        m.step(1.0)
        with pytest.raises(ScheduleStateError, match="monotonic"):
            m.step(0.5)

    def test_event_time_mismatch_raises(self) -> None:
        m = Mixed([FR(1), FR(1)])
        with pytest.raises(ScheduleStateError, match="event.time"):
            m.step(1.0, ResponseEvent(time=2.0))


class TestReinforcerTiming:
    def test_reinforcer_timestamped_at_now(self) -> None:
        m = Mixed([FR(1), FR(1)])
        out = _respond(m, 3.14)
        assert out.reinforced is True
        assert out.reinforcer is not None
        assert out.reinforcer.time == 3.14
