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

from typing import Any

from contingency.entities import Outcome, ResponseEvent
from contingency.errors import ScheduleConfigError
from contingency.interfaces import Schedule
from contingency.schedules._rust import component_specs, compound_schedule

__all__ = ["Conjunctive"]

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

    The compound runs on the Rust engine. Its components must be
    schedules of this package that run there, not yet stepped, and
    distinct objects; otherwise the constructor raises
    :class:`ScheduleConfigError`.

    References
    ----------
    Ferster, C. B., & Skinner, B. F. (1957). *Schedules of
    reinforcement* (pp. 508-509). Appleton-Century-Crofts.
    """

    __slots__ = ("_first", "_second", "_rust")

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
        self._rust: Any = compound_schedule(self._rust_spec())

    @property
    def first(self) -> Schedule:
        """The first component schedule."""
        return self._first

    @property
    def second(self) -> Schedule:
        """The second component schedule."""
        return self._second

    def _rust_spec(self) -> dict[str, Any]:
        first, second = component_specs("Conjunctive", [self._first, self._second])
        return {"type": "Conjunctive", "params": {"first": first, "second": second}}

    @property
    def engine(self) -> str:
        """Always ``"rust"``: the Rust engine runs this schedule."""
        return "rust"

    def step(self, now: float, event: ResponseEvent | None = None) -> Outcome:
        return self._rust.step(now, event)  # type: ignore[no-any-return]

    def reset(self) -> None:
        for component in (self._first, self._second):
            component.reset()
        self._rust = compound_schedule(self._rust_spec())
