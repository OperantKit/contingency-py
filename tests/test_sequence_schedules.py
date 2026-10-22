"""Tests for sequence-family compound schedules (Multiple, Chained, Tandem).

References
----------
Ferster, C. B., & Skinner, B. F. (1957). *Schedules of reinforcement*.
Appleton-Century-Crofts.

Kelleher, R. T., & Gollub, L. R. (1962). A review of positive
conditioned reinforcement. *Journal of the Experimental Analysis of
Behavior*, 5(4 Suppl), 543-597. https://doi.org/10.1901/jeab.1962.5-s543
"""

from __future__ import annotations

import pytest

from contingency.entities import Outcome, ResponseEvent
from contingency.errors import ScheduleConfigError, ScheduleStateError
from contingency.interfaces import Schedule
from contingency.schedules import FR, FT, Chained, Multiple, Tandem


def _respond(s: Schedule, now: float) -> Outcome:
    return s.step(now, ResponseEvent(time=now))


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestProtocolCompliance:
    def test_multiple_satisfies_schedule_protocol(self) -> None:
        assert isinstance(Multiple([FR(1), FR(1)]), Schedule)

    def test_chained_satisfies_schedule_protocol(self) -> None:
        assert isinstance(Chained([FR(1), FR(1)]), Schedule)

    def test_tandem_satisfies_schedule_protocol(self) -> None:
        assert isinstance(Tandem([FR(1), FR(1)]), Schedule)


# ---------------------------------------------------------------------------
# Construction / validation (shared across Multiple, Chained, Tandem)
# ---------------------------------------------------------------------------


class TestConfigErrors:
    def test_multiple_requires_at_least_two_components(self) -> None:
        with pytest.raises(ScheduleConfigError):
            Multiple([FR(1)])

    def test_chained_requires_at_least_two_components(self) -> None:
        with pytest.raises(ScheduleConfigError):
            Chained([FR(1)])

    def test_tandem_requires_at_least_two_components(self) -> None:
        with pytest.raises(ScheduleConfigError):
            Tandem([FR(1)])

    def test_empty_components_rejected(self) -> None:
        with pytest.raises(ScheduleConfigError):
            Multiple([])
        with pytest.raises(ScheduleConfigError):
            Chained([])
        with pytest.raises(ScheduleConfigError):
            Tandem([])

    def test_non_sequence_components_rejected(self) -> None:
        with pytest.raises(ScheduleConfigError):
            Multiple(42)  # type: ignore[arg-type]

    def test_string_as_components_rejected(self) -> None:
        with pytest.raises(ScheduleConfigError):
            Multiple("ab")  # type: ignore[arg-type]

    def test_non_schedule_component_rejected(self) -> None:
        with pytest.raises(ScheduleConfigError):
            Multiple([FR(1), "not-a-schedule"])  # type: ignore[list-item]

    def test_multiple_stimuli_length_mismatch_rejected(self) -> None:
        with pytest.raises(ScheduleConfigError):
            Multiple([FR(1), FR(1)], stimuli=["a"])

    def test_chained_stimuli_length_mismatch_rejected(self) -> None:
        with pytest.raises(ScheduleConfigError):
            Chained([FR(1), FR(1)], stimuli=["a", "b", "c"])

    def test_duplicate_stimuli_rejected_multiple(self) -> None:
        with pytest.raises(ScheduleConfigError):
            Multiple([FR(1), FR(1)], stimuli=["red", "red"])

    def test_duplicate_stimuli_rejected_chained(self) -> None:
        with pytest.raises(ScheduleConfigError):
            Chained([FR(1), FR(1)], stimuli=["green", "green"])

    def test_stimuli_must_be_sequence(self) -> None:
        with pytest.raises(ScheduleConfigError):
            Multiple([FR(1), FR(1)], stimuli=42)  # type: ignore[arg-type]

    def test_stimuli_string_rejected_as_non_sequence_of_strings(self) -> None:
        # A bare string could be interpreted as a sequence of characters —
        # reject to avoid silent surprise.
        with pytest.raises(ScheduleConfigError):
            Multiple([FR(1), FR(1)], stimuli="ab")  # type: ignore[arg-type]

    def test_non_string_stimulus_rejected(self) -> None:
        with pytest.raises(ScheduleConfigError):
            Multiple([FR(1), FR(1)], stimuli=["a", 2])  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# Multiple
# ---------------------------------------------------------------------------


class TestMultipleDefaults:
    def test_default_stimulus_names(self) -> None:
        m = Multiple([FR(1), FR(1), FR(1)])
        assert m.current_stimulus == "comp_0"
        assert m.n_components == 3
        assert m.active_index == 0

    def test_custom_stimulus_names(self) -> None:
        m = Multiple([FR(1), FR(1)], stimuli=["red", "green"])
        assert m.current_stimulus == "red"


class TestMultipleBehavior:
    def test_responses_route_to_active_component_only(self) -> None:
        # FR3 active, FR5 idle. Reinforcement should happen at the 3rd
        # response even though FR5 would require 5.
        m = Multiple([FR(3), FR(5)], stimuli=["red", "green"])
        o1 = _respond(m, 1.0)
        o2 = _respond(m, 2.0)
        o3 = _respond(m, 3.0)
        assert o1.reinforced is False
        assert o2.reinforced is False
        assert o3.reinforced is True
        assert o3.reinforcer is not None
        assert o3.reinforcer.time == 3.0

    def test_advances_to_next_component_after_reinforcement(self) -> None:
        m = Multiple([FR(2), FR(3)], stimuli=["red", "green"])
        _respond(m, 1.0)
        o_rf = _respond(m, 2.0)
        assert o_rf.reinforced is True
        # After reinforcement, active component is component 1 (FR3, "green").
        assert m.active_index == 1
        assert m.current_stimulus == "green"

        # Now two responses should not reinforce under FR3.
        assert _respond(m, 3.0).reinforced is False
        assert _respond(m, 4.0).reinforced is False
        o2 = _respond(m, 5.0)
        assert o2.reinforced is True
        # Wraps back to component 0.
        assert m.active_index == 0
        assert m.current_stimulus == "red"

    def test_wraps_around_to_zero_after_terminal(self) -> None:
        m = Multiple([FR(1), FR(1), FR(1)], stimuli=["a", "b", "c"])
        stims = []
        for i in range(1, 7):
            o = _respond(m, float(i))
            assert o.reinforced is True
            stims.append(o.meta["current_component"])
        # Each FR1 reinforces on every response, cycling a → b → c → a ...
        assert stims == ["a", "b", "c", "a", "b", "c"]

    def test_current_component_in_meta_on_non_reinforced_step(self) -> None:
        m = Multiple([FR(3), FR(3)], stimuli=["red", "green"])
        o = _respond(m, 1.0)
        assert o.reinforced is False
        assert o.meta["current_component"] == "red"

    def test_current_component_reflects_next_component_on_reinforcement(
        self,
    ) -> None:
        # After a reinforced Outcome, the contract says `current_component`
        # is the stimulus that was active when reinforcement was delivered.
        # (The new component becomes the active one for the *next* step.)
        m = Multiple([FR(1), FR(1)], stimuli=["red", "green"])
        o = _respond(m, 1.0)
        assert o.reinforced is True
        assert o.meta["current_component"] == "red"
        # Next step runs under "green".
        o2 = m.step(2.0)
        assert o2.reinforced is False
        assert o2.meta["current_component"] == "green"

    def test_non_response_tick_does_not_advance_component(self) -> None:
        m = Multiple([FR(2), FR(2)], stimuli=["red", "green"])
        o = m.step(1.0)
        assert o.reinforced is False
        assert m.active_index == 0

    def test_with_time_based_component_ft_fires_while_active(self) -> None:
        # Mixed FR + FT: FT must fire reinforcement autonomously when it
        # is the active component.
        m = Multiple([FR(2), FT(1.0)], stimuli=["red", "green"])
        # First, drive the FR component to reinforcement.
        _respond(m, 0.0)
        o_rf = _respond(m, 1.0)
        assert o_rf.reinforced is True
        # Now FT is active. First step anchors; next step 1s later reinforces.
        o_anchor = m.step(1.0)
        assert o_anchor.reinforced is False
        assert m.active_index == 1
        # Step forward by 1 full interval.
        o_ft = m.step(2.0)
        assert o_ft.reinforced is True
        assert o_ft.reinforcer is not None
        assert o_ft.reinforcer.time == 2.0
        # After FT reinforces, cycle wraps back to FR.
        assert m.active_index == 0
        assert m.current_stimulus == "red"


class TestMultipleReset:
    def test_reset_returns_to_component_zero(self) -> None:
        m = Multiple([FR(1), FR(1)], stimuli=["a", "b"])
        _respond(m, 1.0)
        assert m.active_index == 1
        m.reset()
        assert m.active_index == 0
        assert m.current_stimulus == "a"

    def test_reset_clears_last_now(self) -> None:
        m = Multiple([FR(1), FR(1)])
        m.step(5.0)
        m.reset()
        # After reset we can step at an earlier time again.
        o = m.step(0.0)
        assert o.reinforced is False

    def test_reset_resets_each_component(self) -> None:
        m = Multiple([FR(3), FR(3)], stimuli=["a", "b"])
        _respond(m, 1.0)
        _respond(m, 2.0)
        # FR(3) has count=2 internally, ready to reinforce on next response.
        m.reset()
        # After reset, need 3 fresh responses.
        assert _respond(m, 1.0).reinforced is False
        assert _respond(m, 2.0).reinforced is False
        assert _respond(m, 3.0).reinforced is True


class TestMultipleStateErrors:
    def test_non_monotonic_time_rejected(self) -> None:
        m = Multiple([FR(1), FR(1)])
        m.step(5.0)
        with pytest.raises(ScheduleStateError):
            m.step(1.0)

    def test_event_time_mismatch_rejected(self) -> None:
        m = Multiple([FR(1), FR(1)])
        with pytest.raises(ScheduleStateError):
            m.step(1.0, ResponseEvent(time=2.0))


# ---------------------------------------------------------------------------
# Chained
# ---------------------------------------------------------------------------


class TestChainedBehavior:
    def test_non_terminal_reinforcement_is_a_transition(self) -> None:
        # FR2 → FR3. Completing FR2 should NOT produce a Reinforcer; only
        # the FR3 terminal does.
        c = Chained([FR(2), FR(3)], stimuli=["red", "green"])
        _respond(c, 1.0)
        o_trans = _respond(c, 2.0)
        assert o_trans.reinforced is False
        assert o_trans.reinforcer is None
        assert o_trans.meta["chain_transition"] is True
        assert o_trans.meta["current_component"] == "green"
        assert c.active_index == 1
        assert c.is_terminal is True

    def test_terminal_produces_primary_reinforcer(self) -> None:
        c = Chained([FR(2), FR(3)], stimuli=["red", "green"])
        # Transition out of link 0.
        _respond(c, 1.0)
        _respond(c, 2.0)
        # Three more responses on terminal → primary reinforcer.
        _respond(c, 3.0)
        _respond(c, 4.0)
        o = _respond(c, 5.0)
        assert o.reinforced is True
        assert o.reinforcer is not None
        assert o.reinforcer.time == 5.0
        assert "chain_transition" not in o.meta
        # Cycles back to link 0.
        assert c.active_index == 0
        assert c.current_stimulus == "red"
        assert o.meta["current_component"] == "red"

    def test_non_terminal_step_without_transition_carries_stimulus(
        self,
    ) -> None:
        c = Chained([FR(3), FR(2)], stimuli=["a", "b"])
        o = _respond(c, 1.0)
        assert o.reinforced is False
        assert o.meta["current_component"] == "a"
        assert "chain_transition" not in o.meta

    def test_chain_transition_flag_only_on_non_terminal(self) -> None:
        c = Chained([FR(1), FR(1), FR(1)], stimuli=["a", "b", "c"])
        # Link 0 → 1 (transition)
        o0 = _respond(c, 1.0)
        assert o0.reinforced is False
        assert o0.meta.get("chain_transition") is True
        assert o0.meta["current_component"] == "b"
        # Link 1 → 2 (transition)
        o1 = _respond(c, 2.0)
        assert o1.reinforced is False
        assert o1.meta.get("chain_transition") is True
        assert o1.meta["current_component"] == "c"
        # Link 2 (terminal): primary SR+, no chain_transition flag.
        o2 = _respond(c, 3.0)
        assert o2.reinforced is True
        assert "chain_transition" not in o2.meta
        # Back to link 0.
        assert c.active_index == 0
        assert o2.meta["current_component"] == "a"

    def test_cycles_back_to_zero_after_terminal(self) -> None:
        c = Chained([FR(1), FR(1)], stimuli=["x", "y"])
        # Transition x → y.
        _respond(c, 1.0)
        # Terminal reinforces.
        o = _respond(c, 2.0)
        assert o.reinforced is True
        assert c.active_index == 0
        # Next full cycle.
        o2 = _respond(c, 3.0)
        assert o2.reinforced is False
        assert o2.meta["chain_transition"] is True
        o3 = _respond(c, 4.0)
        assert o3.reinforced is True

    def test_default_stimulus_names(self) -> None:
        c = Chained([FR(1), FR(1)])
        assert c.current_stimulus == "comp_0"

    def test_n_components(self) -> None:
        c = Chained([FR(1), FR(1), FR(1)])
        assert c.n_components == 3

    def test_non_response_tick_does_not_transition(self) -> None:
        c = Chained([FR(2), FR(2)], stimuli=["a", "b"])
        o = c.step(1.0)
        assert o.reinforced is False
        assert o.meta["current_component"] == "a"
        assert "chain_transition" not in o.meta
        assert c.active_index == 0


class TestChainedReset:
    def test_reset_returns_to_link_zero(self) -> None:
        c = Chained([FR(1), FR(1)], stimuli=["a", "b"])
        _respond(c, 1.0)  # transition to link 1
        assert c.active_index == 1
        c.reset()
        assert c.active_index == 0
        assert c.current_stimulus == "a"

    def test_reset_resets_components(self) -> None:
        c = Chained([FR(3), FR(1)], stimuli=["a", "b"])
        _respond(c, 1.0)
        _respond(c, 2.0)
        # FR(3) internal counter = 2.
        c.reset()
        # After reset, need 3 fresh responses for the transition.
        o1 = _respond(c, 1.0)
        o2 = _respond(c, 2.0)
        o3 = _respond(c, 3.0)
        assert o1.reinforced is False and "chain_transition" not in o1.meta
        assert o2.reinforced is False and "chain_transition" not in o2.meta
        assert o3.meta.get("chain_transition") is True


class TestChainedStateErrors:
    def test_non_monotonic_time_rejected(self) -> None:
        c = Chained([FR(1), FR(1)])
        c.step(5.0)
        with pytest.raises(ScheduleStateError):
            c.step(1.0)

    def test_event_time_mismatch_rejected(self) -> None:
        c = Chained([FR(1), FR(1)])
        with pytest.raises(ScheduleStateError):
            c.step(1.0, ResponseEvent(time=2.0))


# ---------------------------------------------------------------------------
# Tandem
# ---------------------------------------------------------------------------


class TestTandemBehavior:
    def test_behaves_like_chained_without_stimuli(self) -> None:
        t = Tandem([FR(2), FR(3)])
        assert t.n_components == 2
        assert t.active_index == 0
        _respond(t, 1.0)
        o_trans = _respond(t, 2.0)
        assert o_trans.reinforced is False
        assert o_trans.meta["chain_transition"] is True
        # Tandem reports the component by integer index.
        assert o_trans.meta["current_component"] == 1
        assert t.active_index == 1
        assert t.is_terminal is True

    def test_terminal_primary_reinforcer(self) -> None:
        t = Tandem([FR(1), FR(2)])
        _respond(t, 1.0)  # transition
        _respond(t, 2.0)  # FR2 response 1
        o = _respond(t, 3.0)
        assert o.reinforced is True
        assert o.reinforcer is not None
        assert o.reinforcer.time == 3.0
        # Cycles back.
        assert t.active_index == 0
        assert o.meta["current_component"] == 0
        assert "chain_transition" not in o.meta

    def test_tandem_has_no_stimuli_argument(self) -> None:
        with pytest.raises(TypeError):
            Tandem([FR(1), FR(1)], stimuli=["a", "b"])  # type: ignore[call-arg]

    def test_meta_component_is_int_on_non_reinforced_step(self) -> None:
        t = Tandem([FR(3), FR(1)])
        o = _respond(t, 1.0)
        assert o.reinforced is False
        assert o.meta["current_component"] == 0
        assert isinstance(o.meta["current_component"], int)

    def test_non_response_tick(self) -> None:
        t = Tandem([FR(2), FR(2)])
        o = t.step(1.0)
        assert o.reinforced is False
        assert o.meta["current_component"] == 0
        assert t.active_index == 0


class TestTandemReset:
    def test_reset_returns_to_link_zero(self) -> None:
        t = Tandem([FR(1), FR(1)])
        _respond(t, 1.0)
        assert t.active_index == 1
        t.reset()
        assert t.active_index == 0

    def test_reset_resets_components(self) -> None:
        t = Tandem([FR(3), FR(1)])
        _respond(t, 1.0)
        _respond(t, 2.0)
        t.reset()
        o1 = _respond(t, 1.0)
        o2 = _respond(t, 2.0)
        o3 = _respond(t, 3.0)
        assert o1.reinforced is False and "chain_transition" not in o1.meta
        assert o2.reinforced is False and "chain_transition" not in o2.meta
        assert o3.meta.get("chain_transition") is True


class TestTandemStateErrors:
    def test_non_monotonic_time_rejected(self) -> None:
        t = Tandem([FR(1), FR(1)])
        t.step(5.0)
        with pytest.raises(ScheduleStateError):
            t.step(1.0)

    def test_event_time_mismatch_rejected(self) -> None:
        t = Tandem([FR(1), FR(1)])
        with pytest.raises(ScheduleStateError):
            t.step(1.0, ResponseEvent(time=2.0))


def _reinforcement_times(s: Schedule, until: int) -> list[float]:
    """Respond once per second from t=0 to ``until`` and collect SR+ times."""
    out: list[float] = []
    for i in range(until + 1):
        t = float(i)
        if s.step(t, ResponseEvent(time=t)).reinforced:
            out.append(t)
    return out


class TestTimeBasedLinkAnchor:
    """Time-based links count from their first step after activation.

    A component's clock stops while it is inactive, so neither session
    time nor time spent in other links counts toward its interval.
    """

    def test_chained_fi_terminal_counts_from_link_entry(self) -> None:
        from contingency.schedules import FI

        c = Chained([FR(5), FI(30.0)], ["s0", "s1"])
        # FR 5 completes at t=4; FI link first stepped at 5 -> SR+ at 35.
        # FR 5 again at 36..40; FI link first stepped at 41 -> SR+ at 71.
        assert _reinforcement_times(c, 75) == [35.0, 71.0]

    def test_tandem_fi_terminal_counts_from_link_entry(self) -> None:
        from contingency.schedules import FI

        t = Tandem([FR(5), FI(30.0)])
        assert _reinforcement_times(t, 75) == [35.0, 71.0]

    def test_chained_ft_terminal_counts_from_link_entry(self) -> None:
        c = Chained([FR(5), FT(10.0)], ["s0", "s1"])
        out: list[float] = []
        for i in range(40):
            t = float(i)
            # Respond only while link 0 is active, tick otherwise.
            ev = ResponseEvent(time=t) if c.active_index == 0 else None
            if c.step(t, ev).reinforced:
                out.append(t)
        # FT link first stepped at 5 -> SR+ at 15; FR 5 at 16..20;
        # FT link first stepped at 21 -> SR+ at 31.
        assert out == [15.0, 31.0]

    def test_multiple_fi_component_clock_pauses_while_inactive(self) -> None:
        from contingency.schedules import FI

        m = Multiple([FI(10.0), FR(20)], ["a", "b"])
        # FI from 0 -> SR+ at 10; FR 20 at 11..30; FI resumes at 31 with
        # 10 s already used -> SR+ at 41.
        assert _reinforcement_times(m, 45) == [10.0, 30.0, 41.0]

    def test_mixed_fi_component_clock_pauses_while_inactive(self) -> None:
        from contingency.schedules import FI, Mixed

        m = Mixed([FI(10.0), FR(20)])
        assert _reinforcement_times(m, 45) == [10.0, 30.0, 41.0]

    def test_chained_escape_link_delivers_onset_on_entry(self) -> None:
        from contingency.schedules import Escape

        c = Chained([FR(1), Escape(5.0, 10.0)], ["s0", "s1"])
        c.step(0.0, ResponseEvent(time=0.0))
        o = c.step(1.0)
        assert o.meta.get("aversive_event") == "shock_onset"
