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

from contingency.entities import Outcome, Reinforcer, ResponseEvent
from contingency.errors import ScheduleConfigError, ScheduleStateError
from contingency.interfaces import Schedule

__all__ = ["Overlay"]

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

    References
    ----------
    Azrin, N. H., & Holz, W. C. (1966). Punishment. In W. K. Honig
    (Ed.), *Operant behavior: Areas of research and application*
    (pp. 380-447). Appleton-Century-Crofts.
    """

    __slots__ = ("_base", "_punisher", "_last_now")

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
        self._last_now: float | None = None

    @property
    def base(self) -> Schedule:
        """The primary reinforcement schedule."""
        return self._base

    @property
    def punisher(self) -> Schedule:
        """The punishment schedule."""
        return self._punisher

    def step(self, now: float, event: ResponseEvent | None = None) -> Outcome:
        _check_time(self._last_now, now)
        _check_event(now, event)
        self._last_now = now

        base_outcome = self._base.step(now, event)
        punisher_outcome = self._punisher.step(now, event)

        base_fired = base_outcome.reinforced
        punisher_fired = punisher_outcome.reinforced

        if not base_fired and not punisher_fired:
            return Outcome()

        if base_fired and not punisher_fired:
            assert base_outcome.reinforcer is not None
            return Outcome(
                reinforced=True,
                reinforcer=base_outcome.reinforcer,
                meta={"base_fired": True, "punisher_fired": False},
            )

        if punisher_fired and not base_fired:
            assert punisher_outcome.reinforcer is not None
            p = punisher_outcome.reinforcer
            return Outcome(
                reinforced=True,
                reinforcer=Reinforcer(
                    time=p.time, magnitude=-p.magnitude, label="SR-"
                ),
                meta={"base_fired": False, "punisher_fired": True},
            )

        # Both fired on the same step. Surface a compound outcome.
        assert base_outcome.reinforcer is not None
        assert punisher_outcome.reinforcer is not None
        net = base_outcome.reinforcer.magnitude - punisher_outcome.reinforcer.magnitude
        if net > 0:
            label = "SR+"
        elif net < 0:
            label = "SR-"
        else:
            label = "SR0"
        return Outcome(
            reinforced=True,
            reinforcer=Reinforcer(
                time=base_outcome.reinforcer.time,
                magnitude=net,
                label=label,
            ),
            meta={
                "base_fired": True,
                "punisher_fired": True,
                "net_magnitude": net,
            },
        )

    def reset(self) -> None:
        self._base.reset()
        self._punisher.reset()
        self._last_now = None
