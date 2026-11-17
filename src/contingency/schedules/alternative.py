"""Alternative compound schedule — whichever-first of two components.

An Alternative schedule presents two component schedules concurrently
on a single operandum. Both components run in parallel; reinforcement
is delivered as soon as **either** component's criterion is met. When
a reinforcer is delivered, **both** components are reset so that the
next cycle starts cleanly.

The canonical example combines a response-based schedule with a
time-based schedule, e.g. ``Alternative(FR(10), FT(60.0))`` — the
subject is reinforced after the 10th response *or* after 60 seconds,
whichever comes first.

Binary shape
------------
``Alternative`` is strictly binary. To compose three or more
components, chain Alternatives: ``Alternative(Alternative(a, b), c)``.
Keeping the shape binary simplifies reasoning about tie-breaking and
avoids an ambiguous ordering over arbitrary-arity components.

Event forwarding
----------------
On every :meth:`step` call, the provided ``now`` and ``event`` are
forwarded to **both** components unchanged. This is the semantics
required by the definition: a response that satisfies ``FR(10)`` does
not *affect* ``FT(60)``'s decision (``FT`` is response-independent),
but ``FT`` must still advance its clock to ``now`` and validate that
``event.time == now``. Both components therefore maintain their own
internal state independently; the ``Alternative`` wrapper only gates
which outcome is surfaced.

Tie-breaking
------------
If both components would reinforce on the same step — a rare but
possible situation — the first component wins deterministically. The
second component is still reset along with the first.

Reset semantics
---------------
After a win, :meth:`reset` is invoked on **both** components so that
each returns to its post-construction state. An external
:meth:`reset` call clears the same state plus the Alternative's own
monotonic-time bookkeeping.

References
----------
Ferster, C. B., & Skinner, B. F. (1957). *Schedules of reinforcement*.
Appleton-Century-Crofts.
"""

from __future__ import annotations

from typing import Any

from contingency.entities import Outcome, ResponseEvent
from contingency.errors import ScheduleConfigError
from contingency.interfaces import Schedule
from contingency.schedules._rust import component_specs, compound_schedule

__all__ = ["Alternative"]


class Alternative:
    """Alternative compound schedule of two components (whichever-first).

    Parameters
    ----------
    first:
        The first component schedule. Wins on simultaneous satisfaction.
    second:
        The second component schedule.

    Notes
    -----
    Both components are stepped on every call with the same ``now`` and
    ``event``. If either returns a reinforced :class:`Outcome`, both
    components are reset and the reinforced outcome is propagated with
    ``meta["alternative_winner"]`` set to ``"first"`` or ``"second"``.
    If both would reinforce simultaneously, the first component wins;
    both are still reset.

    The compound runs on the Rust engine. Its components must be
    schedules of this package that run there, not yet stepped, and
    distinct objects; otherwise the constructor raises
    :class:`ScheduleConfigError`. The component objects themselves are
    not stepped: the engine runs its own copies built from their specs.

    References
    ----------
    Ferster, C. B., & Skinner, B. F. (1957). *Schedules of
    reinforcement*. Appleton-Century-Crofts.
    """

    __slots__ = ("_first", "_second", "_rust")

    def __init__(self, first: Schedule, second: Schedule) -> None:
        if not isinstance(first, Schedule):
            raise ScheduleConfigError(
                f"Alternative requires first to implement Schedule, "
                f"got {type(first).__name__}"
            )
        if not isinstance(second, Schedule):
            raise ScheduleConfigError(
                f"Alternative requires second to implement Schedule, "
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
        first, second = component_specs("Alternative", [self._first, self._second])
        return {"type": "Alternative", "params": {"first": first, "second": second}}

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
