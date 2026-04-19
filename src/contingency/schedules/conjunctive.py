"""Conjunctive compound schedule (``CONJ``).

A conjunctive schedule (Ferster & Skinner, 1957, pp. 508-509)
reinforces only when **both** component schedules simultaneously
satisfy their individual criteria. The canonical example is
``Conj(FR5, FI10)``: reinforcement occurs on a response that is both
the 5th (or later) since the last reinforcement *and* emitted after
the 10-second interval has elapsed.

Contrast with related compounds
-------------------------------
* :class:`~contingency.schedules.Alternative` (OR) — reinforces on the
  first component to satisfy its criterion; both reset.
* :class:`~contingency.schedules.Concurrent` — components are
  independent and each maintains its own reinforcement stream.
* :class:`Conjunctive` (AND) — reinforcement is delivered only when
  both components fire on the same step; both reset.

Binary shape
------------
``Conjunctive`` is strictly binary. To compose three or more
conjuncts, nest: ``Conjunctive(Conjunctive(a, b), c)``. Binary shape
keeps reset semantics unambiguous.

Event forwarding & simultaneity semantics
-----------------------------------------
Each :meth:`step` forwards ``(now, event)`` to both components so
each maintains its own internal state (identical to
:class:`~contingency.schedules.Alternative`). A conjunctive
reinforcement is emitted **only when both components would reinforce
on the same step**. In that case both are reset.

This mirrors the historical CONJ definition — "a response is
reinforced only after two requirements have been met" (Ferster &
Skinner, 1957, p. 508) — under the simultaneity interpretation:
satisfaction is evaluated in lock-step at each (time, event) tuple.
Combinations where satisfaction of the two components is naturally
synchronous on the terminal response (FR + FI, FR + FT, etc.) work
cleanly; exotic combinations where one component "consumes" its own
satisfaction on a step where the other does not (e.g. RR + RI where
RR can fire probabilistically on any response) may under-deliver
conjunctive reinforcement because the RR win is suppressed without
being re-armed. For such combinations prefer
:class:`~contingency.schedules.Alternative` or
:class:`~contingency.schedules.Concurrent`, or wrap components in
ratio/interval shells that only fire on joint satisfaction.

References
----------
Ferster, C. B., & Skinner, B. F. (1957). *Schedules of reinforcement*
(pp. 508-509). Appleton-Century-Crofts.
"""

from __future__ import annotations

from contingency.entities import Outcome, Reinforcer, ResponseEvent
from contingency.errors import ScheduleConfigError, ScheduleStateError
from contingency.interfaces import Schedule

__all__ = ["Conjunctive"]

_TIME_TOL = 1e-9


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


class Conjunctive:
    """Conjunctive (AND) compound schedule of two components.

    Parameters
    ----------
    first:
        The first component schedule.
    second:
        The second component schedule.

    Notes
    -----
    Both components are stepped on every call with the same
    ``(now, event)``. Reinforcement is delivered *only* when both
    components would reinforce on the same step. Both are reset
    together after a joint win. See the module docstring for the
    simultaneity semantics and their practical limits.

    References
    ----------
    Ferster, C. B., & Skinner, B. F. (1957). *Schedules of
    reinforcement* (pp. 508-509). Appleton-Century-Crofts.
    """

    __slots__ = ("_first", "_second", "_last_now")

    def __init__(self, first: Schedule, second: Schedule) -> None:
        if not isinstance(first, Schedule):
            raise ScheduleConfigError(
                f"Conjunctive requires first to implement Schedule, "
                f"got {type(first).__name__}"
            )
        if not isinstance(second, Schedule):
            raise ScheduleConfigError(
                f"Conjunctive requires second to implement Schedule, "
                f"got {type(second).__name__}"
            )
        self._first: Schedule = first
        self._second: Schedule = second
        self._last_now: float | None = None

    @property
    def first(self) -> Schedule:
        """The first component schedule."""
        return self._first

    @property
    def second(self) -> Schedule:
        """The second component schedule."""
        return self._second

    def step(self, now: float, event: ResponseEvent | None = None) -> Outcome:
        _check_time(self._last_now, now)
        _check_event(now, event)
        self._last_now = now

        first_outcome = self._first.step(now, event)
        second_outcome = self._second.step(now, event)

        if first_outcome.reinforced and second_outcome.reinforced:
            self._first.reset()
            self._second.reset()
            return Outcome(
                reinforced=True,
                reinforcer=Reinforcer(time=now),
                meta={"conjunctive": True},
            )
        return Outcome()

    def reset(self) -> None:
        self._first.reset()
        self._second.reset()
        self._last_now = None
