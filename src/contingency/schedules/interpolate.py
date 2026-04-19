"""Interpolate compound schedule — mid-session probe windows.

An Interpolate runs a ``base`` schedule as the default for the session
and periodically swaps to a ``probe`` schedule for a brief window
(``probe_duration``). Only one component is "on" at a time: during a
probe window, only ``probe`` receives events and ticks; outside the
window, only ``base`` does. The meta dictionary reports the current
window via ``in_probe: bool``.

The first probe window opens at ``first_probe_at`` (default: ``interval``
after the session starts, where "session start" is the first
:meth:`step` call — its ``now`` becomes the anchor). Successive probe
windows open every ``interval`` seconds from that anchor; i.e. probe
starts occur at ``anchor + first_probe_at + k * interval`` for
``k >= 0``. A probe window covers the half-open range
``[probe_start, probe_start + probe_duration)`` in the schedule's time
coordinates.

References
----------
Catania, A. C., & Reynolds, G. S. (1968). A quantitative analysis of
responding maintained by interval schedules of reinforcement. *Journal
of the Experimental Analysis of Behavior*, 11(3s), 327-383.
https://doi.org/10.1901/jeab.1968.11-s327
"""

from __future__ import annotations

from contingency.entities import Outcome, ResponseEvent
from contingency.errors import ScheduleConfigError, ScheduleStateError
from contingency.interfaces import Schedule

__all__ = ["Interpolate"]

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


class Interpolate:
    """Interpolate compound schedule — mid-session probe.

    Parameters
    ----------
    base:
        Default schedule controlling behavior outside probe windows.
    probe:
        Schedule active during probe windows.
    interval:
        Period between probe window starts (same time unit as
        :meth:`step`'s ``now``). Must be ``> 0``.
    probe_duration:
        Duration of each probe window. Must be ``> 0`` and less than
        ``interval`` so the probe window cannot overlap the next one.
    first_probe_at:
        Time from the session anchor (the first :meth:`step`'s ``now``)
        at which the first probe window opens. Must be ``>= 0``.
        Defaults to ``interval`` so the session begins in ``base`` for
        one full interval before the first probe.

    Notes
    -----
    *One active schedule at a time.* During a probe window only
    ``probe`` receives :meth:`step`; outside the window only ``base``
    does. The inactive schedule's clock does not advance during its
    off-phase. Because schedule state persists across re-entry, a
    subject may re-enter ``base`` at the point it left off, which is
    the intended "mid-session change" semantics for probe studies.

    *Meta reporting.* Each returned :class:`Outcome` carries
    ``meta["in_probe"] = bool`` indicating which component handled the
    step. When the active component reinforces, its own ``meta`` is
    preserved and ``in_probe`` is merged in.

    References
    ----------
    Catania, A. C., & Reynolds, G. S. (1968). A quantitative analysis
    of responding maintained by interval schedules of reinforcement.
    *Journal of the Experimental Analysis of Behavior*, 11(3s),
    327-383. https://doi.org/10.1901/jeab.1968.11-s327
    """

    __slots__ = (
        "_base",
        "_probe",
        "_interval",
        "_probe_duration",
        "_first_probe_at",
        "_anchor",
        "_last_now",
    )

    def __init__(
        self,
        base: Schedule,
        probe: Schedule,
        interval: float,
        probe_duration: float,
        first_probe_at: float | None = None,
    ) -> None:
        if not isinstance(base, Schedule):
            raise ScheduleConfigError(
                f"Interpolate requires base to implement Schedule, "
                f"got {type(base).__name__}"
            )
        if not isinstance(probe, Schedule):
            raise ScheduleConfigError(
                f"Interpolate requires probe to implement Schedule, "
                f"got {type(probe).__name__}"
            )
        if isinstance(interval, bool) or not isinstance(interval, (int, float)):
            raise ScheduleConfigError(
                f"Interpolate requires a numeric interval, got "
                f"{type(interval).__name__}"
            )
        if interval <= 0:
            raise ScheduleConfigError(
                f"Interpolate requires interval > 0, got {interval}"
            )
        if isinstance(probe_duration, bool) or not isinstance(
            probe_duration, (int, float)
        ):
            raise ScheduleConfigError(
                f"Interpolate requires a numeric probe_duration, got "
                f"{type(probe_duration).__name__}"
            )
        if probe_duration <= 0:
            raise ScheduleConfigError(
                f"Interpolate requires probe_duration > 0, got {probe_duration}"
            )
        if probe_duration >= interval:
            raise ScheduleConfigError(
                f"Interpolate requires probe_duration ({probe_duration}) < "
                f"interval ({interval}) so windows do not overlap"
            )
        if first_probe_at is None:
            first_probe_at = float(interval)
        elif isinstance(first_probe_at, bool) or not isinstance(
            first_probe_at, (int, float)
        ):
            raise ScheduleConfigError(
                f"Interpolate requires a numeric first_probe_at, got "
                f"{type(first_probe_at).__name__}"
            )
        elif first_probe_at < 0:
            raise ScheduleConfigError(
                f"Interpolate requires first_probe_at >= 0, got {first_probe_at}"
            )

        self._base: Schedule = base
        self._probe: Schedule = probe
        self._interval: float = float(interval)
        self._probe_duration: float = float(probe_duration)
        self._first_probe_at: float = float(first_probe_at)
        self._anchor: float | None = None
        self._last_now: float | None = None

    @property
    def base(self) -> Schedule:
        return self._base

    @property
    def probe(self) -> Schedule:
        return self._probe

    @property
    def interval(self) -> float:
        return self._interval

    @property
    def probe_duration(self) -> float:
        return self._probe_duration

    @property
    def first_probe_at(self) -> float:
        return self._first_probe_at

    def _in_probe_window(self, now: float) -> bool:
        """Return True iff ``now`` falls inside a probe window."""
        if self._anchor is None:
            return False
        t = now - self._anchor - self._first_probe_at
        if t < -_TIME_TOL:
            return False
        # Within the current interval, the window spans [0, probe_duration).
        frac = t - (int(t // self._interval)) * self._interval
        return frac < self._probe_duration - _TIME_TOL

    def step(self, now: float, event: ResponseEvent | None = None) -> Outcome:
        _check_time(self._last_now, now)
        _check_event(now, event)
        if self._anchor is None:
            self._anchor = now
        self._last_now = now

        in_probe = self._in_probe_window(now)
        active = self._probe if in_probe else self._base

        inner = active.step(now, event)
        meta = dict(inner.meta)
        meta["in_probe"] = in_probe
        return Outcome(
            reinforced=inner.reinforced,
            reinforcer=inner.reinforcer,
            meta=meta,
        )

    def reset(self) -> None:
        self._base.reset()
        self._probe.reset()
        self._anchor = None
        self._last_now = None
