"""Timeout (TO) schedule wrapper.

A timeout (TO) is a period following reinforcement during which the
underlying reinforcement schedule is suspended: responses do not advance
the inner schedule, and no reinforcement is available. Classically,
time-out from positive reinforcement has been studied both as a
punishment procedure (when contingent on a target response) and as a
simple interruption of the reinforcement process (see Leitenberg, 1965).

The wrapper implemented here places the inner schedule on timeout for
``duration`` time units after *every* reinforcer the inner schedule
produces. Responses emitted during the timeout interval are silently
dropped — they are not passed through to the inner schedule's
:meth:`step`, and no reinforcement can be delivered. When
``reset_on_response`` is true, any response emitted within the timeout
window extends the window (``timeout_ends`` is pushed to ``now +
duration``); this models a punishment contingency for responding in
timeout.

References
----------
Leitenberg, H. (1965). Is time-out from positive reinforcement an
aversive event? A review of the experimental evidence. *Psychological
Bulletin*, 64(6), 428-441. https://doi.org/10.1037/h0022657
"""

from __future__ import annotations

from contingency.entities import Outcome, ResponseEvent
from contingency.errors import ScheduleConfigError, ScheduleStateError
from contingency.interfaces import Schedule

__all__ = ["Timeout"]


_TIME_TOL = 1e-9


def _check_time(last_now: float | None, now: float) -> None:
    if last_now is not None and now < last_now - _TIME_TOL:
        raise ScheduleStateError(
            f"Non-monotonic time: new now={now!r} precedes previous now={last_now!r}"
        )


def _check_event(now: float, event: ResponseEvent | None) -> None:
    if event is not None and abs(event.time - now) > _TIME_TOL:
        raise ScheduleStateError(
            f"event.time ({event.time!r}) must equal now ({now!r}) within {_TIME_TOL}"
        )


class Timeout:
    """Timeout (TO) wrapper — Leitenberg (1965).

    After a reinforcement, the wrapped schedule is placed on timeout
    for ``duration`` time units. During the timeout, responses are
    ignored (do not advance the inner schedule) and no reinforcement
    is possible.

    If ``reset_on_response`` is True, a response during the timeout
    restarts the timeout clock (punishment for responding in timeout).

    Parameters
    ----------
    inner:
        The schedule to wrap.
    duration:
        Timeout duration in unit-agnostic monotonic time. Must be > 0.
    reset_on_response:
        Whether a response during timeout restarts the timeout clock.

    References
    ----------
    Leitenberg, H. (1965). Is time-out from positive reinforcement an
    aversive event? A review of the experimental evidence. *Psychological
    Bulletin*, 64(6), 428-441. https://doi.org/10.1037/h0022657
    """

    __slots__ = (
        "_inner",
        "_duration",
        "_reset_on_response",
        "_timeout_ends",
        "_last_now",
    )

    def __init__(
        self,
        inner: Schedule,
        duration: float,
        reset_on_response: bool = False,
    ) -> None:
        if isinstance(duration, bool) or not isinstance(duration, (int, float)):
            raise ScheduleConfigError(
                f"Timeout requires a numeric duration, got {type(duration).__name__}"
            )
        if not (duration > 0):
            raise ScheduleConfigError(
                f"Timeout requires duration > 0, got {duration!r}"
            )
        self._inner: Schedule = inner
        self._duration: float = float(duration)
        self._reset_on_response: bool = bool(reset_on_response)
        self._timeout_ends: float | None = None
        self._last_now: float | None = None

    @property
    def duration(self) -> float:
        return self._duration

    @property
    def reset_on_response(self) -> bool:
        return self._reset_on_response

    @property
    def in_timeout(self) -> bool:
        """Whether the wrapper is currently in a timeout window."""
        return self._timeout_ends is not None

    def step(self, now: float, event: ResponseEvent | None = None) -> Outcome:
        _check_time(self._last_now, now)
        _check_event(now, event)
        self._last_now = now

        # Clear the timeout if the clock has advanced past its end.
        if (
            self._timeout_ends is not None
            and now + _TIME_TOL >= self._timeout_ends
        ):
            self._timeout_ends = None

        # Still in timeout: optionally extend on response, never step inner.
        if self._timeout_ends is not None:
            if event is not None and self._reset_on_response:
                self._timeout_ends = now + self._duration
            return Outcome(meta={"timeout_active": True})

        # Not in timeout — step the inner schedule normally.
        outcome = self._inner.step(now, event)

        if outcome.reinforced:
            self._timeout_ends = now + self._duration
            meta: dict[str, object] = dict(outcome.meta)
            meta["timeout_started"] = True
            return Outcome(
                reinforced=True,
                reinforcer=outcome.reinforcer,
                meta=meta,
            )
        return outcome

    def reset(self) -> None:
        self._inner.reset()
        self._timeout_ends = None
        self._last_now = None
