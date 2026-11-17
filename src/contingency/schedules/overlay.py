"""Overlay compound schedule — punishment overlay (Azrin & Holz, 1966).

An Overlay runs two schedules in parallel on the same operandum: a
primary reinforcement schedule (``base``) and a punishment schedule
(``punisher``). Every :meth:`step` call forwards ``(now, event)`` to
both. Whatever ``base`` would have delivered is a positive reinforcer
(``SR+``); whatever ``punisher`` would have delivered is re-labelled as
a punisher (``SR-``) with its magnitude negated.

A single step may therefore emit:

* nothing (neither fired);
* a positive reinforcement (base only);
* a punishment (punisher only);
* a compound outcome (both fired on the same step) — the meta dictionary
  records ``base_fired=True``, ``punisher_fired=True`` and
  ``net_magnitude = base.magnitude - punisher.magnitude`` (the magnitude
  field carries ``net_magnitude`` and the label is ``"SR+"`` when
  ``net_magnitude > 0``, ``"SR-"`` when ``< 0``, and ``"SR0"`` when
  exactly zero).

Reset clears both components.

References
----------
Azrin, N. H., & Holz, W. C. (1966). Punishment. In W. K. Honig (Ed.),
*Operant behavior: Areas of research and application* (pp. 380-447).
Appleton-Century-Crofts.
"""

from __future__ import annotations

from typing import Any

from contingency.entities import Outcome, ResponseEvent
from contingency.errors import ScheduleConfigError
from contingency.interfaces import Schedule
from contingency.schedules._rust import component_specs, compound_schedule

__all__ = ["Overlay"]

class Overlay:
    """Punishment overlay — Azrin & Holz (1966).

    Runs a primary reinforcement schedule (``base``) and a punishment
    schedule (``punisher``) in parallel on the same operandum. Each step
    forwards ``(now, event)`` to both.

    Parameters
    ----------
    base:
        Primary reinforcement schedule (its outcomes surface as
        positive :class:`~contingency.entities.Reinforcer` objects).
    punisher:
        A schedule whose reinforcer acts as a punisher; its magnitude
        is negated and its label replaced with ``"SR-"``.

    Returns
    -------
    :class:`Outcome` with:

    * ``reinforced=False`` when neither fires;
    * ``reinforced=True`` with positive magnitude and label ``"SR+"``
      when only ``base`` fires;
    * ``reinforced=True`` with negative magnitude and label ``"SR-"``
      when only ``punisher`` fires;
    * ``reinforced=True`` with ``meta["base_fired"]=True``,
      ``meta["punisher_fired"]=True`` and ``meta["net_magnitude"]``
      when both fire on the same step. The surfaced
      :class:`Reinforcer` magnitude equals ``net_magnitude``.

    The compound runs on the Rust engine. Its components must be
    schedules of this package that run there, not yet stepped, and
    distinct objects; otherwise the constructor raises
    :class:`ScheduleConfigError`.

    References
    ----------
    Azrin, N. H., & Holz, W. C. (1966). Punishment. In W. K. Honig
    (Ed.), *Operant behavior: Areas of research and application*
    (pp. 380-447). Appleton-Century-Crofts.
    """

    __slots__ = ("_base", "_punisher", "_rust")

    def __init__(self, base: Schedule, punisher: Schedule) -> None:
        if not isinstance(base, Schedule):
            raise ScheduleConfigError(
                f"Overlay requires base to implement Schedule, "
                f"got {type(base).__name__}"
            )
        if not isinstance(punisher, Schedule):
            raise ScheduleConfigError(
                f"Overlay requires punisher to implement Schedule, "
                f"got {type(punisher).__name__}"
            )
        self._base: Schedule = base
        self._punisher: Schedule = punisher
        self._rust: Any = compound_schedule(self._rust_spec())

    @property
    def base(self) -> Schedule:
        """The primary reinforcement schedule."""
        return self._base

    @property
    def punisher(self) -> Schedule:
        """The punishment schedule."""
        return self._punisher

    def _rust_spec(self) -> dict[str, Any]:
        base, punisher = component_specs("Overlay", [self._base, self._punisher])
        return {"type": "Overlay", "params": {"base": base, "punisher": punisher}}

    @property
    def engine(self) -> str:
        """Always ``"rust"``: the Rust engine runs this schedule."""
        return "rust"

    def step(self, now: float, event: ResponseEvent | None = None) -> Outcome:
        return self._rust.step(now, event)  # type: ignore[no-any-return]

    def reset(self) -> None:
        for component in (self._base, self._punisher):
            component.reset()
        self._rust = compound_schedule(self._rust_spec())
