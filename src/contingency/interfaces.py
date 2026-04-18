"""Schedule protocol — the single abstraction every schedule implements.

A schedule is a stateful object driven forward by :meth:`step`. Each
step advances an internal clock to ``now`` and optionally registers a
:class:`ResponseEvent`. The schedule returns an :class:`Outcome`
describing whether the step produced reinforcement.

Design choices
--------------
* **Unified step API**. Response-based schedules (FR, VR) ignore the
  ``now`` argument beyond monotonicity checks; time-based schedules
  (FT, VT) ignore ``event``; interval schedules (FI, VI) use both.
  Having one method keeps composition straightforward.
* **Synchronous**. Async/IO concerns belong to the session runner. A
  schedule is a pure state machine.
* **Reset semantics**. :meth:`reset` returns the schedule to the state
  it was in immediately after construction; it does *not* re-sample
  random parameters. Re-sampling is the responsibility of variable
  schedules' internal logic.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from contingency.entities import Outcome, ResponseEvent

__all__ = ["Schedule"]


@runtime_checkable
class Schedule(Protocol):
    """A reinforcement schedule."""

    def step(self, now: float, event: ResponseEvent | None = None) -> Outcome:
        """Advance to ``now`` and optionally register a response.

        Parameters
        ----------
        now:
            Current monotonic time. Must be ``>=`` the previous step's
            ``now`` — non-monotonic input raises
            :class:`~contingency.errors.ScheduleStateError`.
        event:
            A :class:`ResponseEvent` if the subject responded at
            ``now``; otherwise ``None``. If provided, ``event.time``
            must equal ``now`` (within a small tolerance).

        Returns
        -------
        Outcome
            Describes whether reinforcement is delivered on this step.
        """
        ...

    def reset(self) -> None:
        """Return the schedule to its post-construction state."""
        ...
