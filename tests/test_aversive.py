"""Tests for aversive schedules — Sidman, DiscriminatedAvoidance, Escape."""

from __future__ import annotations

import pytest

from contingency.entities import ResponseEvent
from contingency.errors import ScheduleConfigError, ScheduleStateError
from contingency.schedules import DiscriminatedAvoidance, Escape, Sidman

# ---------------------------------------------------------------------------
# Sidman
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_sidman_construction_valid() -> None:
    s = Sidman(shock_shock_interval=20.0, response_shock_interval=5.0)
    assert s.shock_shock_interval == 20.0
    assert s.response_shock_interval == 5.0
    assert s.shock_magnitude == 1.0
    assert s.next_shock_time == 20.0


@pytest.mark.unit
def test_sidman_rejects_nonpositive_ssi() -> None:
    with pytest.raises(ScheduleConfigError, match="shock_shock_interval"):
        Sidman(shock_shock_interval=0.0, response_shock_interval=5.0)
    with pytest.raises(ScheduleConfigError, match="shock_shock_interval"):
        Sidman(shock_shock_interval=-1.0, response_shock_interval=5.0)


@pytest.mark.unit
def test_sidman_rejects_nonpositive_rsi() -> None:
    with pytest.raises(ScheduleConfigError, match="response_shock_interval"):
        Sidman(shock_shock_interval=20.0, response_shock_interval=0.0)


@pytest.mark.unit
def test_sidman_rejects_nonnumeric_params() -> None:
    with pytest.raises(ScheduleConfigError):
        Sidman(shock_shock_interval="20", response_shock_interval=5.0)  # type: ignore[arg-type]
    with pytest.raises(ScheduleConfigError):
        Sidman(shock_shock_interval=20.0, response_shock_interval=True)  # type: ignore[arg-type]


@pytest.mark.unit
def test_sidman_rejects_nonpositive_magnitude() -> None:
    with pytest.raises(ScheduleConfigError, match="shock_magnitude"):
        Sidman(shock_shock_interval=20.0, response_shock_interval=5.0, shock_magnitude=0.0)
    with pytest.raises(ScheduleConfigError, match="shock_magnitude"):
        Sidman(shock_shock_interval=20.0, response_shock_interval=5.0, shock_magnitude=-1.0)


@pytest.mark.unit
def test_sidman_ssi_shock_no_response() -> None:
    s = Sidman(shock_shock_interval=10.0, response_shock_interval=5.0)
    # No responses; shock should fire at t=10.
    assert not s.step(0.0).reinforced
    assert not s.step(5.0).reinforced
    outcome = s.step(10.0)
    assert outcome.reinforced
    assert outcome.reinforcer is not None
    assert outcome.reinforcer.magnitude == -1.0
    assert outcome.reinforcer.label == "SR-"
    assert outcome.meta["aversive_event"] == "shock"


@pytest.mark.unit
def test_sidman_successive_shocks_use_ssi() -> None:
    s = Sidman(shock_shock_interval=10.0, response_shock_interval=5.0)
    o1 = s.step(10.0)
    assert o1.reinforced
    # Next shock is at 10 + 10 = 20.
    assert not s.step(15.0).reinforced
    o2 = s.step(20.0)
    assert o2.reinforced


@pytest.mark.unit
def test_sidman_rsi_override_postpones_shock() -> None:
    # With SSI=10, response at t=5 postpones shock to t=5+RSI=15.
    s = Sidman(shock_shock_interval=10.0, response_shock_interval=10.0)
    assert not s.step(5.0, ResponseEvent(time=5.0)).reinforced
    # At t=10 (the old SSI deadline) no shock, because RSI pushed it to 15.
    assert not s.step(10.0).reinforced
    assert not s.step(14.0).reinforced
    o = s.step(15.0)
    assert o.reinforced


@pytest.mark.unit
def test_sidman_continuous_responding_suppresses_shocks() -> None:
    s = Sidman(shock_shock_interval=10.0, response_shock_interval=5.0)
    # Respond every 2 time-units; RSI=5 so next_shock always pushed beyond now.
    t = 0.0
    while t < 30.0:
        t += 2.0
        outcome = s.step(t, ResponseEvent(time=t))
        assert not outcome.reinforced, f"unexpected shock at t={t}"


@pytest.mark.unit
def test_sidman_reset() -> None:
    s = Sidman(shock_shock_interval=10.0, response_shock_interval=5.0)
    s.step(5.0, ResponseEvent(time=5.0))
    assert s.next_shock_time == 10.0
    s.reset()
    assert s.next_shock_time == 10.0
    # After reset, step sequence works from t=0 again.
    assert not s.step(0.0).reinforced
    assert s.step(10.0).reinforced


@pytest.mark.unit
def test_sidman_non_monotonic_time_raises() -> None:
    s = Sidman(shock_shock_interval=10.0, response_shock_interval=5.0)
    s.step(5.0)
    with pytest.raises(ScheduleStateError, match="Non-monotonic"):
        s.step(4.0)


@pytest.mark.unit
def test_sidman_event_time_mismatch_raises() -> None:
    s = Sidman(shock_shock_interval=10.0, response_shock_interval=5.0)
    with pytest.raises(ScheduleStateError, match="event.time"):
        s.step(5.0, ResponseEvent(time=4.0))


@pytest.mark.unit
def test_sidman_custom_magnitude() -> None:
    s = Sidman(shock_shock_interval=5.0, response_shock_interval=2.0, shock_magnitude=3.5)
    o = s.step(5.0)
    assert o.reinforced
    assert o.reinforcer is not None
    assert o.reinforcer.magnitude == -3.5


# ---------------------------------------------------------------------------
# DiscriminatedAvoidance
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_discrimav_construction_valid() -> None:
    d = DiscriminatedAvoidance(warning_duration=5.0, iti=10.0)
    assert d.warning_duration == 5.0
    assert d.iti == 10.0
    assert d.phase == "iti"


@pytest.mark.unit
def test_discrimav_rejects_invalid_params() -> None:
    with pytest.raises(ScheduleConfigError, match="warning_duration"):
        DiscriminatedAvoidance(warning_duration=0.0, iti=10.0)
    with pytest.raises(ScheduleConfigError, match="iti"):
        DiscriminatedAvoidance(warning_duration=5.0, iti=-1.0)
    with pytest.raises(ScheduleConfigError, match="shock_magnitude"):
        DiscriminatedAvoidance(warning_duration=5.0, iti=10.0, shock_magnitude=0.0)


@pytest.mark.unit
def test_discrimav_iti_phase_reports_iti() -> None:
    d = DiscriminatedAvoidance(warning_duration=5.0, iti=10.0)
    o = d.step(2.0)
    assert not o.reinforced
    assert o.meta.get("phase") == "iti"


@pytest.mark.unit
def test_discrimav_warning_starts_after_iti() -> None:
    d = DiscriminatedAvoidance(warning_duration=5.0, iti=10.0)
    # Before ITI elapses.
    d.step(9.0)
    assert d.phase == "iti"
    # At t=10 the warning starts.
    o = d.step(10.0)
    assert d.phase == "warning"
    # No shock yet, only phase transition.
    assert not o.reinforced


@pytest.mark.unit
def test_discrimav_avoidance_response_in_warning() -> None:
    d = DiscriminatedAvoidance(warning_duration=5.0, iti=10.0)
    d.step(10.0)  # enter warning
    o = d.step(12.0, ResponseEvent(time=12.0))
    assert not o.reinforced
    assert o.meta["aversive_event"] == "avoidance"
    assert o.meta["trial_outcome"] == "avoid"
    assert d.phase == "iti"


@pytest.mark.unit
def test_discrimav_shock_on_warning_timeout() -> None:
    d = DiscriminatedAvoidance(warning_duration=5.0, iti=10.0)
    d.step(10.0)  # enter warning at t=10, expires at t=15
    o = d.step(15.0)
    assert o.reinforced
    assert o.reinforcer is not None
    assert o.reinforcer.magnitude == -1.0
    assert o.reinforcer.label == "SR-"
    assert o.meta["aversive_event"] == "shock"
    assert o.meta["trial_outcome"] == "shock"
    assert d.phase == "iti"


@pytest.mark.unit
def test_discrimav_second_trial_after_shock() -> None:
    d = DiscriminatedAvoidance(warning_duration=5.0, iti=10.0)
    d.step(10.0)  # warning begins
    d.step(15.0)  # shock -> iti; next warning at 25
    d.step(20.0)
    assert d.phase == "iti"
    d.step(25.0)
    assert d.phase == "warning"


@pytest.mark.unit
def test_discrimav_reset() -> None:
    d = DiscriminatedAvoidance(warning_duration=5.0, iti=10.0)
    d.step(10.0)
    d.step(12.0, ResponseEvent(time=12.0))
    d.reset()
    assert d.phase == "iti"
    o = d.step(2.0)
    assert o.meta.get("phase") == "iti"


@pytest.mark.unit
def test_discrimav_non_monotonic_raises() -> None:
    d = DiscriminatedAvoidance(warning_duration=5.0, iti=10.0)
    d.step(5.0)
    with pytest.raises(ScheduleStateError):
        d.step(4.0)


# ---------------------------------------------------------------------------
# Escape
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_escape_construction_valid() -> None:
    e = Escape(trial_duration=5.0, iti=3.0)
    assert e.trial_duration == 5.0
    assert e.iti == 3.0
    assert e.phase == "shock"


@pytest.mark.unit
def test_escape_rejects_invalid_params() -> None:
    with pytest.raises(ScheduleConfigError, match="trial_duration"):
        Escape(trial_duration=0.0, iti=3.0)
    with pytest.raises(ScheduleConfigError, match="iti"):
        Escape(trial_duration=5.0, iti=0.0)
    with pytest.raises(ScheduleConfigError, match="shock_magnitude"):
        Escape(trial_duration=5.0, iti=3.0, shock_magnitude=-2.0)


@pytest.mark.unit
def test_escape_emits_shock_on_first_tick() -> None:
    e = Escape(trial_duration=5.0, iti=3.0)
    o = e.step(0.0)
    assert o.reinforced
    assert o.reinforcer is not None
    assert o.reinforcer.magnitude == -1.0
    assert o.meta["aversive_event"] == "shock_onset"
    assert o.meta["phase"] == "shock"


@pytest.mark.unit
def test_escape_emits_shock_every_tick() -> None:
    e = Escape(trial_duration=10.0, iti=3.0)
    ticks = [e.step(t) for t in (0.0, 1.0, 2.0, 3.0)]
    assert all(o.reinforced for o in ticks)
    # First is onset, rest are 'shock'.
    assert ticks[0].meta["aversive_event"] == "shock_onset"
    for o in ticks[1:]:
        assert o.meta["aversive_event"] == "shock"


@pytest.mark.unit
def test_escape_response_terminates_shock() -> None:
    e = Escape(trial_duration=10.0, iti=3.0)
    e.step(0.0)
    e.step(1.0)
    o = e.step(2.0, ResponseEvent(time=2.0))
    assert not o.reinforced
    assert o.meta["aversive_event"] == "shock_termination"
    assert o.meta["trial_outcome"] == "escape"
    assert e.phase == "iti"


@pytest.mark.unit
def test_escape_trial_duration_bound_on_failure() -> None:
    e = Escape(trial_duration=3.0, iti=2.0)
    e.step(0.0)
    e.step(1.0)
    e.step(2.0)
    # At t=3 trial ends (no response).
    o = e.step(3.0)
    assert not o.reinforced
    assert o.meta["aversive_event"] == "trial_end"
    assert o.meta["trial_outcome"] == "failed"
    assert e.phase == "iti"


@pytest.mark.unit
def test_escape_iti_then_new_trial() -> None:
    e = Escape(trial_duration=10.0, iti=3.0)
    e.step(0.0)
    e.step(1.0, ResponseEvent(time=1.0))  # terminate -> iti
    # ITI ends at t = 1 + 3 = 4.
    o_iti = e.step(3.0)
    assert o_iti.meta.get("phase") == "iti"
    assert not o_iti.reinforced
    # At t=4, new trial shock onset.
    o_new = e.step(4.0)
    assert o_new.reinforced
    assert o_new.meta["aversive_event"] == "shock_onset"


@pytest.mark.unit
def test_escape_reset() -> None:
    e = Escape(trial_duration=5.0, iti=2.0)
    e.step(0.0)
    e.step(1.0, ResponseEvent(time=1.0))
    e.reset()
    assert e.phase == "shock"
    o = e.step(0.0)
    assert o.reinforced
    assert o.meta["aversive_event"] == "shock_onset"


@pytest.mark.unit
def test_escape_non_monotonic_raises() -> None:
    e = Escape(trial_duration=5.0, iti=2.0)
    e.step(1.0)
    with pytest.raises(ScheduleStateError):
        e.step(0.5)


@pytest.mark.unit
def test_escape_event_time_mismatch_raises() -> None:
    e = Escape(trial_duration=5.0, iti=2.0)
    with pytest.raises(ScheduleStateError):
        e.step(1.0, ResponseEvent(time=0.5))


@pytest.mark.unit
def test_escape_custom_magnitude() -> None:
    e = Escape(trial_duration=5.0, iti=2.0, shock_magnitude=2.5)
    o = e.step(0.0)
    assert o.reinforcer is not None
    assert o.reinforcer.magnitude == -2.5
