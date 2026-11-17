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
from typing import Any

from contingency.entities import Outcome, ResponseEvent
from contingency.errors import ScheduleConfigError
from contingency.interfaces import Schedule
from contingency.schedules._rust import component_specs, compound_schedule

__all__ = ["Mixed"]

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
    Inactive components are **not** stepped, and each component runs on
    its own clock that stops while it is inactive (matching
    :class:`~contingency.schedules.Multiple`). The integer active
    index is reported in ``Outcome.meta["current_component"]`` on
    every step; this is the key difference from :class:`Multiple`,
    which reports a string stimulus name.

    The compound runs on the Rust engine. Its components must be
    schedules of this package that run there, not yet stepped, and
    distinct objects; otherwise the constructor raises
    :class:`ScheduleConfigError`.

    References
    ----------
    Ferster, C. B., & Skinner, B. F. (1957). *Schedules of
    reinforcement* (pp. 623-626). Appleton-Century-Crofts.
    """

    __slots__ = ("_components", "_rust")

    def __init__(self, components: Sequence[Schedule]) -> None:
        self._components: list[Schedule] = _validate_components(components)
        self._rust: Any = compound_schedule(self._rust_spec(), needs_state=True)

    @property
    def n_components(self) -> int:
        """Number of component schedules."""
        return len(self._components)

    @property
    def active_index(self) -> int:
        """Index of the currently active component."""
        return int(self._rust.state()["active_index"])

    def _rust_spec(self) -> dict[str, Any]:
        specs = component_specs("Mixed", self._components)
        return {"type": "Mixed", "params": {"components": specs}}

    @property
    def engine(self) -> str:
        """Always ``"rust"``: the Rust engine runs this schedule."""
        return "rust"

    def step(self, now: float, event: ResponseEvent | None = None) -> Outcome:
        return self._rust.step(now, event)  # type: ignore[no-any-return]

    def reset(self) -> None:
        for component in self._components:
            component.reset()
        self._rust = compound_schedule(self._rust_spec(), needs_state=True)
