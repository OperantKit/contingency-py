"""Tests for :class:`contingency.schedules.Timeout`.

A Timeout wrapper suspends the inner schedule for ``duration`` time
units after each reinforcer. Within the timeout window, responses do
not advance the inner schedule and no reinforcement can be delivered;
when ``reset_on_response`` is true, a response during timeout extends
the window. The semantics follow Leitenberg (1965)'s review of
time-out from positive reinforcement as a conceptual frame.

References
----------
Leitenberg, H. (1965). Is time-out from positive reinforcement an
aversive event? A review of the experimental evidence. *Psychological
Bulletin*, 64(6), 428-441. https://doi.org/10.1037/h0022657
"""

from __future__ import annotations

import pytest

from contingency.entities import ResponseEvent
from contingency.errors import ScheduleConfigError, ScheduleStateError
from contingency.schedules import FR, Timeout


@pytest.mark.unit
def test_timeout_fr1_suspends_responses_within_window() -> None:
    """After a reinforcer, responses within ``duration`` do not fire."""
    s = Timeout(FR(1), duration=10.0)
    # First response triggers FR(1) reinforcement; timeout begins.
    out = s.step(1.0, ResponseEvent(time=1.0))
    assert out.reinforced is True
    assert out.reinforcer is not None
    assert out.meta.get("timeout_started") is True

    # During the timeout, responses are ignored: not reinforced, and
    # the wrapper advertises the active-timeout state via meta.
    for t in (2.0, 5.0, 10.0):
        out = s.step(t, ResponseEvent(time=t))
        assert out.reinforced is False
        assert out.meta.get("timeout_active") is True


@pytest.mark.unit
def test_timeout_clears_after_duration_elapses() -> None:
    """Once ``now >= timeout_ends``, the next response can reinforce again."""
    s = Timeout(FR(1), duration=10.0)
    assert s.step(1.0, ResponseEvent(time=1.0)).reinforced is True  # start TO
    # Tick past timeout end without a response.
    assert s.step(11.5, None).reinforced is False
    assert s.in_timeout is False
    # The next response on FR(1) reinforces again.
    out = s.step(12.0, ResponseEvent(time=12.0))
    assert out.reinforced is True
    assert out.meta.get("timeout_started") is True


@pytest.mark.unit
def test_timeout_response_at_exact_end_fires_inner() -> None:
    """A response at ``now == timeout_ends`` is past the boundary."""
    s = Timeout(FR(1), duration=5.0)
    s.step(0.0, ResponseEvent(time=0.0))  # arms TO until 5.0
    # At the boundary, timeout clears first, then the response fires.
    out = s.step(5.0, ResponseEvent(time=5.0))
    assert out.reinforced is True


@pytest.mark.unit
def test_timeout_reset_on_response_true_extends_window() -> None:
    """reset_on_response=True pushes ``timeout_ends`` forward each response."""
    s = Timeout(FR(1), duration=10.0, reset_on_response=True)
    s.step(1.0, ResponseEvent(time=1.0))  # TO ends at 11.0 initially.
    # A response at t=6.0 extends TO to 16.0.
    out = s.step(6.0, ResponseEvent(time=6.0))
    assert out.reinforced is False
    assert out.meta.get("timeout_active") is True
    # At t=12.0, without reset_on_response the window would have ended
    # already; with reset_on_response it is still active (TO now ends at 22.0).
    out = s.step(12.0, ResponseEvent(time=12.0))
    assert out.reinforced is False
    assert out.meta.get("timeout_active") is True
    # A quiet period past the most recent extension clears the timeout.
    s.step(22.5, None)
    assert s.in_timeout is False


@pytest.mark.unit
def test_timeout_reset_on_response_false_does_not_extend() -> None:
    """reset_on_response=False: responses during TO do not extend it."""
    s = Timeout(FR(1), duration=10.0, reset_on_response=False)
    s.step(1.0, ResponseEvent(time=1.0))  # TO ends at 11.0.
    s.step(5.0, ResponseEvent(time=5.0))  # ignored, no extension.
    out = s.step(11.5, ResponseEvent(time=11.5))
    # Timeout cleared at or before 11.0, so this response reinforces.
    assert out.reinforced is True


@pytest.mark.unit
def test_timeout_tick_without_event_propagates_to_inner_when_not_in_timeout() -> None:
    """A pure tick outside timeout is delegated to the inner schedule."""
    s = Timeout(FR(2), duration=3.0)
    out = s.step(0.0, None)
    assert out.reinforced is False
    assert out.meta == {}


@pytest.mark.unit
def test_timeout_non_monotonic_time_raises() -> None:
    s = Timeout(FR(1), duration=5.0)
    s.step(2.0, None)
    with pytest.raises(ScheduleStateError, match="Non-monotonic"):
        s.step(1.0, None)


@pytest.mark.unit
def test_timeout_event_mismatch_raises() -> None:
    s = Timeout(FR(1), duration=5.0)
    with pytest.raises(ScheduleStateError, match="event.time"):
        s.step(1.0, ResponseEvent(time=2.0))


@pytest.mark.unit
def test_timeout_reset_clears_timeout_and_restarts_inner() -> None:
    s = Timeout(FR(1), duration=10.0)
    s.step(1.0, ResponseEvent(time=1.0))  # arms TO.
    assert s.in_timeout is True
    s.reset()
    assert s.in_timeout is False
    # After reset the inner FR(1) is fresh — the next response reinforces.
    out = s.step(1.0, ResponseEvent(time=1.0))
    assert out.reinforced is True


@pytest.mark.unit
def test_timeout_config_error_duration_zero() -> None:
    with pytest.raises(ScheduleConfigError, match="duration > 0"):
        Timeout(FR(1), duration=0.0)


@pytest.mark.unit
def test_timeout_config_error_duration_negative() -> None:
    with pytest.raises(ScheduleConfigError, match="duration > 0"):
        Timeout(FR(1), duration=-1.0)


@pytest.mark.unit
def test_timeout_config_error_duration_non_numeric() -> None:
    with pytest.raises(ScheduleConfigError, match="numeric duration"):
        Timeout(FR(1), duration="ten")  # type: ignore[arg-type]


@pytest.mark.unit
def test_timeout_config_error_duration_bool_rejected() -> None:
    with pytest.raises(ScheduleConfigError, match="numeric duration"):
        Timeout(FR(1), duration=True)  # type: ignore[arg-type]


@pytest.mark.unit
def test_timeout_properties() -> None:
    s = Timeout(FR(1), duration=7.5, reset_on_response=True)
    assert s.duration == pytest.approx(7.5)
    assert s.reset_on_response is True
    assert s.in_timeout is False


@pytest.mark.unit
def test_timeout_with_fr_ratio_suspends_partial_run() -> None:
    """FR(3) under Timeout: the 3rd response reinforces, then 3s TO."""
    s = Timeout(FR(3), duration=3.0)
    assert s.step(1.0, ResponseEvent(time=1.0)).reinforced is False
    assert s.step(2.0, ResponseEvent(time=2.0)).reinforced is False
    out = s.step(3.0, ResponseEvent(time=3.0))
    assert out.reinforced is True
    # During timeout the inner's response counter is untouched.
    s.step(4.0, ResponseEvent(time=4.0))
    s.step(5.0, ResponseEvent(time=5.0))
    # After TO ends (at 6.0), three more responses are required.
    assert s.step(6.5, ResponseEvent(time=6.5)).reinforced is False
    assert s.step(7.0, ResponseEvent(time=7.0)).reinforced is False
    assert s.step(7.5, ResponseEvent(time=7.5)).reinforced is True
