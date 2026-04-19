"""Mixed compound schedule (``MIX``).

A mixed schedule (Ferster & Skinner, 1957, pp. 623-626) alternates
between two or more component schedules on each reinforcement,
exactly like :class:`~contingency.schedules.Multiple` — but **without
distinctive stimuli**. The subject receives no external cue indicating
which component is currently in effect; only the experimenter's
record (the integer ``current_component`` index in ``meta``) tracks
the active component.

Relation to :class:`Multiple` and :class:`Tandem`
-------------------------------------------------
* :class:`~contingency.schedules.Multiple` alternates components with
  distinctive S^Ds and reports ``current_component`` as the string
  stimulus name.
* :class:`~contingency.schedules.Tandem` is the chained (sequential)
  counterpart of :class:`~contingency.schedules.Chained` without
  S^Ds — primary reinforcement only on the terminal link.
* :class:`Mixed` is the alternating (multiple) counterpart of
  :class:`~contingency.schedules.Multiple` without S^Ds — every
  component's reinforcement is a primary reinforcer.

The runtime logic is identical to :class:`Multiple` with
``stimuli=None``; only the ``meta`` payload differs. :class:`Mixed`
reports the active index as an ``int`` (matching :class:`Tandem`'s
convention) rather than a ``str``, so a downstream consumer can
distinguish "discriminated alternation" from "undiscriminated
alternation" purely from the ``meta`` type.

References
----------
Ferster, C. B., & Skinner, B. F. (1957). *Schedules of reinforcement*
(pp. 623-626). Appleton-Century-Crofts.
"""

from __future__ import annotations

from collections.abc import Sequence

from contingency.entities import Outcome, ResponseEvent
from contingency.errors import ScheduleConfigError, ScheduleStateError
from contingency.interfaces import Schedule

__all__ = ["Mixed"]

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


def _validate_components(components: Sequence[Schedule]) -> list[Schedule]:
    if not isinstance(components, Sequence) or isinstance(components, (str, bytes)):
        raise ScheduleConfigError(
            f"components must be a sequence of schedules, got {type(components).__name__}"
        )
    comp_list = list(components)
    if len(comp_list) < 2:
        raise ScheduleConfigError(
            f"Mixed requires >= 2 components, got {len(comp_list)}"
        )
    for i, c in enumerate(comp_list):
        if not isinstance(c, Schedule):
            raise ScheduleConfigError(
                f"component {i} does not satisfy the Schedule protocol: "
                f"{type(c).__name__}"
            )
    return comp_list


class Mixed:
    """Mixed (``mix``) compound schedule.

    Two or more independent component schedules alternate on
    reinforcement, with no distinctive stimuli exposed to the subject.
    Exactly one component is active at a time; responses route only
    to that component. When the active component produces a
    reinforcer, the primary reinforcer is delivered and the active
    index advances (wrapping to 0 after the last component).

    Parameters
    ----------
    components:
        Two or more schedules, each satisfying the
        :class:`~contingency.interfaces.Schedule` protocol.

    Notes
    -----
    Inactive components are **not** stepped (matching
    :class:`~contingency.schedules.Multiple`). The integer active
    index is reported in ``Outcome.meta["current_component"]`` on
    every step; this is the key difference from :class:`Multiple`,
    which reports a string stimulus name.

    References
    ----------
    Ferster, C. B., & Skinner, B. F. (1957). *Schedules of
    reinforcement* (pp. 623-626). Appleton-Century-Crofts.
    """

    __slots__ = ("_components", "_active", "_last_now")

    def __init__(self, components: Sequence[Schedule]) -> None:
        self._components: list[Schedule] = _validate_components(components)
        self._active: int = 0
        self._last_now: float | None = None

    @property
    def n_components(self) -> int:
        """Number of component schedules."""
        return len(self._components)

    @property
    def active_index(self) -> int:
        """Index of the currently active component."""
        return self._active

    def step(self, now: float, event: ResponseEvent | None = None) -> Outcome:
        _check_time(self._last_now, now)
        _check_event(now, event)
        self._last_now = now

        inner = self._components[self._active].step(now, event)
        current = self._active

        if inner.reinforced:
            self._active = (self._active + 1) % len(self._components)
            meta: dict[str, object] = dict(inner.meta)
            meta["current_component"] = current
            return Outcome(
                reinforced=True, reinforcer=inner.reinforcer, meta=meta
            )

        meta = dict(inner.meta)
        meta["current_component"] = current
        return Outcome(meta=meta)

    def reset(self) -> None:
        self._active = 0
        self._last_now = None
        for c in self._components:
            c.reset()
