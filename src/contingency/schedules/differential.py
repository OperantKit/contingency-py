"""Differential reinforcement schedules.

This module implements the three canonical differential-reinforcement
families of the experimental analysis of behavior:

* :class:`DRO` — Differential Reinforcement of Other behavior.
  Reinforces the *absence* of a response over an interval. Two
  variants are supported: the resetting (default) variant in which
  every response resets the DRO timer, and the ``"momentary"`` variant,
  which is a non-resetting whole-interval DRO: the timer runs
  continuously and, at each interval boundary, reinforcement depends on
  whether any response occurred during the whole interval. Despite its
  ``dro_type`` name it is not the momentary DRO of Lindberg et al.
  (1999), which checks only the instant at the end of the interval.
* :class:`DRL` — Differential Reinforcement of Low rate behavior.
  Reinforces a response whose inter-response time (IRT) is at least
  ``interval`` time units.
* :class:`DRH` — Differential Reinforcement of High rate behavior.
  Reinforces a response whenever at least ``response_count``
  responses have been emitted within the last ``time_window`` time
  units.

Each schedule is unit-agnostic with respect to time (``now`` is a
monotonic float on a caller-declared clock) and follows the same
:class:`~contingency.interfaces.Schedule` protocol (``step`` / ``reset``)
as the rest of :mod:`contingency.schedules`.

References
----------
Ferster, C. B., & Skinner, B. F. (1957). *Schedules of reinforcement*.
Appleton-Century-Crofts.

Reynolds, G. S. (1961). Behavioral contrast. *Journal of the
Experimental Analysis of Behavior*, 4(1), 57-71.
https://doi.org/10.1901/jeab.1961.4-57

Reynolds, G. S. (1964). Accurate and rapid reconditioning of
spaced responding. *Journal of the Experimental Analysis of
Behavior*, 7(3), 273-275. https://doi.org/10.1901/jeab.1964.7-273

Lindberg, J. S., Iwata, B. A., Kahng, S., & DeLeon, I. G. (1999). DRO
contingencies: An analysis of variable-momentary schedules. *Journal of
Applied Behavior Analysis*, 32(2), 123-136.
https://doi.org/10.1901/jaba.1999.32-123

Zeiler, M. D. (1977). Schedules of reinforcement: The controlling
variables. In W. K. Honig & J. E. R. Staddon (Eds.), *Handbook of
operant behavior* (pp. 201-232). Prentice-Hall.
"""

from __future__ import annotations

from collections import deque
from typing import Literal

from contingency.entities import Outcome, Reinforcer, ResponseEvent
from contingency.errors import ScheduleConfigError, ScheduleStateError

__all__ = ["DRO", "DRL", "DRH"]

_TIME_TOL = 1e-9

_DROType = Literal["resetting", "momentary"]


def _check_time(last_now: float | None, now: float) -> None:
    """Raise :class:`ScheduleStateError` if ``now`` moves backwards."""
    if last_now is not None and now < last_now - _TIME_TOL:
        raise ScheduleStateError(
            f"Non-monotonic time: new now={now!r} precedes previous now={last_now!r}"
        )


def _check_event(now: float, event: ResponseEvent | None) -> None:
    """Raise :class:`ScheduleStateError` if ``event.time`` disagrees with ``now``."""
    if event is not None and abs(event.time - now) > _TIME_TOL:
        raise ScheduleStateError(
            f"event.time ({event.time!r}) must equal now ({now!r}) within {_TIME_TOL}"
        )


class DRO:
    """Differential Reinforcement of Other behavior.

    Parameters
    ----------
    interval:
        Length of the DRO interval. Must be ``> 0``.
    dro_type:
        Either ``"resetting"`` (default) or ``"momentary"``.

        * ``"resetting"``: every response resets the DRO timer to
          ``now``. A reinforcer is delivered on the first step whose
          ``now`` satisfies ``now - timer_start >= interval`` with no
          intervening response. After reinforcement the timer is
          restarted from ``now``.
        * ``"momentary"`` (a non-resetting *whole-interval* DRO; not
          the momentary DRO of Lindberg et al., 1999, which checks
          only the instant at the end of each interval): the timer
          runs continuously, independent of responses. At every
          interval boundary (``now >= interval_start + interval``) a
          reinforcer is delivered iff no response occurred in
          ``[interval_start, now)``. The anchor ``interval_start`` is
          advanced to ``now`` on every boundary regardless of outcome.

    Notes
    -----
    The schedule's anchor is established on the *first* call to
    :meth:`step` — before that first call the schedule has no notion
    of "now". This mirrors :class:`~contingency.schedules.time_based.FT`
    and the other time-anchored schedules in this package.

    For the resetting variant, if a response arrives exactly on or
    after the interval boundary, the step performs the reinforcement
    *and* registers the response, meaning the next DRO timer is
    anchored at ``now``. In other words, the reinforcement is
    delivered and the response simultaneously restarts the clock.

    For the momentary variant, responses never reset the timer but
    they do block reinforcement on the *next* boundary check. The
    membership interval is half-open: a response whose timestamp
    equals the boundary counts for the *following* window, not the
    window just closing.

    References
    ----------
    Reynolds, G. S. (1961). Behavioral contrast. *Journal of the
    Experimental Analysis of Behavior*, 4(1), 57-71.
    https://doi.org/10.1901/jeab.1961.4-57

    Zeiler, M. D. (1977). Schedules of reinforcement: The controlling
    variables. In W. K. Honig & J. E. R. Staddon (Eds.), *Handbook of
    operant behavior* (pp. 201-232). Prentice-Hall.
    """

    __slots__ = (
        "_interval",
        "_type",
        "_anchor",
        "_has_response_in_window",
        "_last_now",
    )

    def __init__(
        self,
        interval: float,
        dro_type: _DROType = "resetting",
    ) -> None:
        if isinstance(interval, bool) or not isinstance(interval, (int, float)):
            raise ScheduleConfigError(
                f"DRO requires a numeric interval, got {type(interval).__name__}"
            )
        if interval <= 0:
            raise ScheduleConfigError(
                f"DRO requires interval > 0, got {interval}"
            )
        if dro_type not in ("resetting", "momentary"):
            raise ScheduleConfigError(
                "DRO dro_type must be 'resetting' or 'momentary', "
                f"got {dro_type!r}"
            )
        self._interval: float = float(interval)
        self._type: _DROType = dro_type
        self._anchor: float | None = None
        self._has_response_in_window: bool = False
        self._last_now: float | None = None

    @property
    def interval(self) -> float:
        """The configured DRO interval."""
        return self._interval

    @property
    def dro_type(self) -> _DROType:
        """The configured DRO variant."""
        return self._type

    def step(self, now: float, event: ResponseEvent | None = None) -> Outcome:
        _check_time(self._last_now, now)
        _check_event(now, event)
        self._last_now = now

        # First step anchors the timer / interval_start at ``now``.
        if self._anchor is None:
            self._anchor = now
            if event is not None:
                # An event arriving on the anchor step starts the
                # first window with a response already recorded (for
                # the momentary variant) and also resets the resetting
                # timer to ``now`` (which we just set).
                self._has_response_in_window = True
            return Outcome()

        if self._type == "resetting":
            return self._step_resetting(now, event)
        return self._step_momentary(now, event)

    def _step_resetting(
        self, now: float, event: ResponseEvent | None
    ) -> Outcome:
        if event is not None:
            # Any response resets the DRO timer. No reinforcement.
            self._anchor = now
            return Outcome()
        if now - self._anchor + _TIME_TOL >= self._interval:  # type: ignore[operator]
            self._anchor = now
            return Outcome(reinforced=True, reinforcer=Reinforcer(time=now))
        return Outcome()

    def _step_momentary(
        self, now: float, event: ResponseEvent | None
    ) -> Outcome:
        # Evaluate the boundary first, BEFORE recording the incoming
        # event: the window [interval_start, now) is half-open, so a
        # response timestamped at ``now`` belongs to the next window.
        reinforced = False
        if now - self._anchor + _TIME_TOL >= self._interval:  # type: ignore[operator]
            if not self._has_response_in_window:
                reinforced = True
            # Advance to the next interval regardless of outcome; the
            # in-window flag is reset because we are opening a fresh
            # window anchored at ``now``.
            self._anchor = now
            self._has_response_in_window = False
        if event is not None:
            self._has_response_in_window = True
        if reinforced:
            return Outcome(reinforced=True, reinforcer=Reinforcer(time=now))
        return Outcome()

    def reset(self) -> None:
        self._anchor = None
        self._has_response_in_window = False
        self._last_now = None


class DRL:
    """Differential Reinforcement of Low rate behavior.

    Parameters
    ----------
    interval:
        Minimum inter-response time (IRT) required for reinforcement.
        Must be ``> 0``.

    Notes
    -----
    A response at ``now`` is reinforced iff either:

    * it is the very first response since construction or
      :meth:`reset`, *or*
    * the previous response occurred at least ``interval`` time units
      before ``now`` (i.e. ``now - last_response_time >= interval``).

    Every response — reinforced or not — updates the internal
    last-response bookmark to ``now``. The IRT clock is therefore
    always measured between consecutive responses. Steps with
    ``event is None`` never modify state beyond the monotonic-time
    check and never produce reinforcement.

    References
    ----------
    Ferster, C. B., & Skinner, B. F. (1957). *Schedules of
    reinforcement*. Appleton-Century-Crofts.

    Reynolds, G. S. (1964). Accurate and rapid reconditioning of
    spaced responding. *Journal of the Experimental Analysis of
    Behavior*, 7(3), 273-275. https://doi.org/10.1901/jeab.1964.7-273
    """

    __slots__ = ("_interval", "_last_response_time", "_last_now")

    def __init__(self, interval: float) -> None:
        if isinstance(interval, bool) or not isinstance(interval, (int, float)):
            raise ScheduleConfigError(
                f"DRL requires a numeric interval, got {type(interval).__name__}"
            )
        if interval <= 0:
            raise ScheduleConfigError(
                f"DRL requires interval > 0, got {interval}"
            )
        self._interval: float = float(interval)
        self._last_response_time: float | None = None
        self._last_now: float | None = None

    @property
    def interval(self) -> float:
        """The configured minimum IRT."""
        return self._interval

    def step(self, now: float, event: ResponseEvent | None = None) -> Outcome:
        _check_time(self._last_now, now)
        _check_event(now, event)
        self._last_now = now
        if event is None:
            return Outcome()
        prev = self._last_response_time
        # Always update the last-response bookmark; the IRT clock
        # advances with every response regardless of reinforcement.
        self._last_response_time = now
        if prev is None or now - prev + _TIME_TOL >= self._interval:
            return Outcome(reinforced=True, reinforcer=Reinforcer(time=now))
        return Outcome()

    def reset(self) -> None:
        self._last_response_time = None
        self._last_now = None


class DRH:
    """Differential Reinforcement of High rate behavior.

    Parameters
    ----------
    response_count:
        Number of responses required within the sliding window. Must
        be ``>= 1``.
    time_window:
        Duration of the sliding window. Must be ``> 0``.

    Notes
    -----
    The schedule maintains a FIFO deque of recent response
    timestamps. On every :meth:`step` — whether or not it carries an
    event — timestamps strictly older than ``now - time_window`` are
    evicted from the left of the deque. When an event arrives, its
    timestamp is appended to the right of the deque; if the deque
    then contains at least ``response_count`` entries, a reinforcer
    is delivered at ``now``.

    The window bound uses ``>=``: a response whose timestamp equals
    ``now - time_window`` is considered still within the window
    (within ``_TIME_TOL``). This matches the intuitive reading "rate
    of at least ``response_count`` per ``time_window``".

    The deque is *not* emptied on reinforcement. DRH cares about
    rate, so a sustained high-rate train will continue to produce
    reinforcers on each qualifying response; the caller is free to
    chain a further schedule on top if a different policy is
    desired.

    References
    ----------
    Ferster, C. B., & Skinner, B. F. (1957). *Schedules of
    reinforcement*. Appleton-Century-Crofts.
    """

    __slots__ = ("_response_count", "_time_window", "_window", "_last_now")

    def __init__(self, response_count: int, time_window: float) -> None:
        if isinstance(response_count, bool) or not isinstance(response_count, int):
            raise ScheduleConfigError(
                "DRH requires an int response_count, got "
                f"{type(response_count).__name__}"
            )
        if response_count < 1:
            raise ScheduleConfigError(
                f"DRH requires response_count >= 1, got {response_count}"
            )
        if isinstance(time_window, bool) or not isinstance(time_window, (int, float)):
            raise ScheduleConfigError(
                "DRH requires a numeric time_window, got "
                f"{type(time_window).__name__}"
            )
        if time_window <= 0:
            raise ScheduleConfigError(
                f"DRH requires time_window > 0, got {time_window}"
            )
        self._response_count: int = response_count
        self._time_window: float = float(time_window)
        self._window: deque[float] = deque()
        self._last_now: float | None = None

    @property
    def response_count(self) -> int:
        """Number of responses required within the window."""
        return self._response_count

    @property
    def time_window(self) -> float:
        """Duration of the sliding window."""
        return self._time_window

    def _evict_old(self, now: float) -> None:
        cutoff = now - self._time_window
        w = self._window
        while w and w[0] < cutoff - _TIME_TOL:
            w.popleft()

    def step(self, now: float, event: ResponseEvent | None = None) -> Outcome:
        _check_time(self._last_now, now)
        _check_event(now, event)
        self._last_now = now
        self._evict_old(now)
        if event is None:
            return Outcome()
        self._window.append(now)
        if len(self._window) >= self._response_count:
            return Outcome(reinforced=True, reinforcer=Reinforcer(time=now))
        return Outcome()

    def reset(self) -> None:
        self._window.clear()
        self._last_now = None
