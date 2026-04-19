"""Interlocking schedule (Ferster & Skinner, 1957).

Under an interlocking schedule the ratio requirement ``R_0`` at the
moment of the last reinforcement decays linearly toward zero over a
time window ``T``. The effective requirement at elapsed time ``t``
is::

    R(t) = max(0, R_0 * (1 - t / T))

so the subject can always earn the next reinforcer by emitting
``R_0`` responses immediately, or by waiting up to ``T`` time units
during which the required count shrinks. The two parameters
interlock the ratio and the interval: neither pure responding nor
pure waiting is optimal in general.

The first step a subject takes anchors the "time of last
reinforcement" — there is no implicit reinforcement at ``t = 0``.

References
----------
Ferster, C. B., & Skinner, B. F. (1957). *Schedules of
reinforcement* (pp. 501-502). Appleton-Century-Crofts.

Berryman, R., & Nevin, J. A. (1962). Interlocking schedules of
reinforcement. *Journal of the Experimental Analysis of Behavior*,
5(2), 213-223. https://doi.org/10.1901/jeab.1962.5-213
"""

from __future__ import annotations

from contingency.entities import Outcome, Reinforcer, ResponseEvent
from contingency.errors import ScheduleConfigError, ScheduleStateError

__all__ = ["InterlockingSchedule"]

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


class InterlockingSchedule:
    """Linearly-decaying interlocking ratio/interval schedule.

    Parameters
    ----------
    initial_ratio:
        The requirement ``R_0`` at the moment of the last reinforcer.
        Must be an ``int >= 1``.
    decay_time:
        The time window ``T`` over which ``R`` decays to zero.
        Must be strictly positive.

    Notes
    -----
    The effective requirement at elapsed time ``t`` since the last
    reinforcer (or, on the very first step, since the schedule's
    first tick) is ``R(t) = max(0, int(R_0 * (1 - t/T)))``. A
    response reinforces when the response count since the last
    reinforcer reaches that effective requirement.

    When ``R(t) == 0`` any response reinforces immediately; the
    schedule is still gated by the presence of a response (a bare
    tick with ``event is None`` never reinforces).

    The emitted :class:`Reinforcer` timestamp equals ``now``; the
    schedule then re-anchors the response counter and the
    last-reinforcement time to ``now``.
    """

    __slots__ = (
        "_R0",
        "_T",
        "_count",
        "_anchor",
        "_last_now",
    )

    def __init__(self, initial_ratio: int, decay_time: float) -> None:
        if not isinstance(initial_ratio, int) or isinstance(initial_ratio, bool):
            raise ScheduleConfigError(
                f"InterlockingSchedule requires an int initial_ratio, got "
                f"{type(initial_ratio).__name__}"
            )
        if initial_ratio < 1:
            raise ScheduleConfigError(
                f"InterlockingSchedule requires initial_ratio >= 1, got "
                f"{initial_ratio}"
            )
        if isinstance(decay_time, bool) or not isinstance(
            decay_time, (int, float)
        ):
            raise ScheduleConfigError(
                f"InterlockingSchedule requires a numeric decay_time, got "
                f"{type(decay_time).__name__}"
            )
        if not (decay_time > 0):
            raise ScheduleConfigError(
                f"InterlockingSchedule requires decay_time > 0, got "
                f"{decay_time!r}"
            )

        self._R0: int = initial_ratio
        self._T: float = float(decay_time)
        self._count: int = 0
        self._anchor: float | None = None
        self._last_now: float | None = None

    # --- Properties -------------------------------------------------------

    @property
    def initial_ratio(self) -> int:
        """The ratio requirement ``R_0`` at ``t = 0`` since last reinforcement."""
        return self._R0

    @property
    def decay_time(self) -> float:
        """The time ``T`` over which ``R`` decays to zero."""
        return self._T

    @property
    def response_count(self) -> int:
        """Responses emitted since the last reinforcement (or anchor)."""
        return self._count

    def current_requirement(self, now: float) -> int:
        """Return the effective requirement at ``now``.

        Before any step has anchored the schedule the reference time
        is implicitly ``now`` (elapsed ``0``) so ``R_0`` is returned.
        """
        anchor = self._anchor if self._anchor is not None else now
        elapsed = now - anchor
        if elapsed < 0:
            # Guard: should not happen in practice because step() enforces
            # monotonic ``now``, but ``current_requirement`` is callable
            # directly (e.g. for instrumentation).
            elapsed = 0.0
        raw = self._R0 * (1.0 - elapsed / self._T)
        if raw <= 0:
            return 0
        return int(raw)

    # --- Schedule protocol -----------------------------------------------

    def step(self, now: float, event: ResponseEvent | None = None) -> Outcome:
        _check_time(self._last_now, now)
        _check_event(now, event)
        self._last_now = now

        # Anchor on the first observed tick so elapsed time is measured
        # from when the schedule "came online" rather than from 0.
        if self._anchor is None:
            self._anchor = now

        if event is None:
            return Outcome()

        requirement = self._effective_requirement(now)
        self._count += 1
        if self._count >= requirement:
            self._count = 0
            self._anchor = now
            return Outcome(reinforced=True, reinforcer=Reinforcer(time=now))
        return Outcome()

    def reset(self) -> None:
        self._count = 0
        self._anchor = None
        self._last_now = None

    # --- Internals --------------------------------------------------------

    def _effective_requirement(self, now: float) -> int:
        """Compute ``R(t)`` at the internal anchor."""
        assert self._anchor is not None  # anchored in step()
        elapsed = now - self._anchor
        if elapsed < 0:
            elapsed = 0.0
        raw = self._R0 * (1.0 - elapsed / self._T)
        if raw <= 0:
            return 0
        return int(raw)
