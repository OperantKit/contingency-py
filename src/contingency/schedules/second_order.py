"""Second-order schedules (Kelleher, 1966).

A second-order schedule nests an inner "unit" schedule inside an
outer "overall" schedule. Completions of the unit schedule count as
single responses toward the overall schedule's requirement. A
canonical example is ``FR5(FI30)``: the subject must complete five
FI-30 intervals, and only the fifth unit completion produces the
subject-visible reinforcer.

Semantics implemented here:

1. Every :meth:`step` is delegated to the unit schedule. If the
   unit does not reinforce, the overall schedule is also advanced
   with the same (possibly ``None``) event so that time-based
   overalls (``FI``, ``VI``, ``FT``, ``VT``) accumulate elapsed
   time correctly — this is the "overall-also-ticked" semantics
   noted in the design.
2. When the unit schedule reinforces, the reinforcer is **suppressed**
   (it is only the unit-completion signal, not a subject-visible
   delivery). The unit is :meth:`~Schedule.reset` and an internal
   synthesised :class:`ResponseEvent` tagged with
   ``operandum="__unit_completion__"`` is fed to the overall
   schedule on the same ``now``.
3. If the overall schedule reinforces on that synthetic event, the
   resulting :class:`Outcome` is what the second-order step emits;
   its ``meta`` carries ``{"second_order": True}`` so downstream
   consumers can distinguish it.

The ``Reinforcer`` emitted is always the overall schedule's — its
timestamp is ``now``.

References
----------
Kelleher, R. T. (1966). Chaining and conditioned reinforcement. In
W. K. Honig (Ed.), *Operant behavior: Areas of research and
application* (pp. 160-212). Appleton-Century-Crofts.
"""

from __future__ import annotations

from contingency.entities import Outcome, ResponseEvent
from contingency.errors import ScheduleConfigError, ScheduleStateError
from contingency.interfaces import Schedule

__all__ = ["SecondOrder"]

_TIME_TOL = 1e-9

UNIT_COMPLETION_OPERANDUM = "__unit_completion__"


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


class SecondOrder:
    """Nested second-order schedule ``overall(unit)``.

    Parameters
    ----------
    overall:
        The outer schedule whose reinforcement is the subject-visible
        delivery. Atomic schedules (FR, VR, FI, VI, FT, VT, RR, RI,
        RT) are the documented use case; compound or decorated
        schedules are accepted but undefined behaviour.
    unit:
        The inner schedule whose completions are counted as overall
        responses. Atomic schedules are the documented use case.

    Notes
    -----
    The second-order step implementation is "overall-also-ticked": on
    every call, after delegating to the unit schedule, the overall
    schedule is also advanced with the same ``now``. This ensures
    time-based overalls accumulate elapsed time correctly even when
    the unit does not complete on a given step. A synthesised
    :class:`ResponseEvent` is injected into the overall schedule only
    on the step where the unit reinforces.

    Because the overall's reinforcement is evaluated exclusively on
    unit-completion events, strict ratio overalls (FR, VR, RR) count
    one unit completion as one response. Interval/time overalls use
    the unit completion both as the gating response *and* as the
    time-progression tick.

    References
    ----------
    Kelleher, R. T. (1966). Chaining and conditioned reinforcement.
    In W. K. Honig (Ed.), *Operant behavior: Areas of research and
    application* (pp. 160-212). Appleton-Century-Crofts.
    """

    __slots__ = (
        "_overall",
        "_unit",
        "_last_now",
        "_unit_completions",
    )

    def __init__(self, overall: Schedule, unit: Schedule) -> None:
        if not isinstance(overall, Schedule):
            raise ScheduleConfigError(
                f"SecondOrder requires a Schedule overall, got "
                f"{type(overall).__name__}"
            )
        if not isinstance(unit, Schedule):
            raise ScheduleConfigError(
                f"SecondOrder requires a Schedule unit, got "
                f"{type(unit).__name__}"
            )
        if overall is unit:
            raise ScheduleConfigError(
                "SecondOrder requires distinct overall and unit schedule "
                "instances (shared state would entangle counters)"
            )
        self._overall: Schedule = overall
        self._unit: Schedule = unit
        self._last_now: float | None = None
        self._unit_completions: int = 0

    # --- Properties -------------------------------------------------------

    @property
    def overall(self) -> Schedule:
        """The outer schedule whose reinforcer is the subject-visible delivery."""
        return self._overall

    @property
    def unit(self) -> Schedule:
        """The inner schedule whose completions count as overall responses."""
        return self._unit

    @property
    def unit_completions(self) -> int:
        """Cumulative unit completions since construction or last :meth:`reset`."""
        return self._unit_completions

    # --- Schedule protocol -----------------------------------------------

    def step(self, now: float, event: ResponseEvent | None = None) -> Outcome:
        _check_time(self._last_now, now)
        _check_event(now, event)
        self._last_now = now

        # Step the unit schedule with the caller's event.
        unit_outcome = self._unit.step(now, event)

        # Decide what to feed the overall schedule on this tick.
        if unit_outcome.reinforced:
            # Unit completion: swallow its reinforcer, reset the unit,
            # and synthesise an internal event for the overall.
            self._unit.reset()
            self._unit_completions += 1
            synth_event: ResponseEvent | None = ResponseEvent(
                time=now, operandum=UNIT_COMPLETION_OPERANDUM
            )
        else:
            # No unit completion: still tick the overall so time-based
            # overalls advance their clocks, but with no event.
            synth_event = None

        overall_outcome = self._overall.step(now, synth_event)
        if overall_outcome.reinforced:
            # Preserve the overall's Reinforcer and enrich meta.
            new_meta: dict[str, object] = dict(overall_outcome.meta)
            new_meta["second_order"] = True
            return Outcome(
                reinforced=True,
                reinforcer=overall_outcome.reinforcer,
                meta=new_meta,
            )

        # No subject-visible reinforcement on this step; surface a
        # plain Outcome, but flag a unit completion in meta when one
        # occurred so downstream tooling can log it.
        if unit_outcome.reinforced:
            return Outcome(meta={"unit_completed": True})
        return Outcome()

    def reset(self) -> None:
        self._overall.reset()
        self._unit.reset()
        self._last_now = None
        self._unit_completions = 0
