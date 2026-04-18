"""Concurrent compound schedule with Changeover Delay (COD).

A concurrent schedule presents two or more component schedules on
distinct *operanda* (keys, levers, buttons). Each component is a
standalone :class:`~contingency.interfaces.Schedule`; responses on
operandum ``k`` are routed exclusively to ``components[k]``. The
composite returns the :class:`~contingency.entities.Outcome` produced by
the component that received the event, subject to changeover gating.

Changeover Delay (Catania, 1966)
--------------------------------
When the subject switches from operandum A to operandum B, the
reinforcer that would otherwise be delivered on B is *suppressed* for
``cod`` time units after the switch. This discourages rapid
alternation and superstitious changeover responding. The timer resets
on every subsequent changeover: "since the last changeover".

If no changeover has yet occurred (e.g. the very first response of a
session), COD is not active. This module implements the standard
"since-last-changeover" semantics of Catania (1966): each fresh
switch re-anchors the COD clock.

Changeover Ratio (COR)
----------------------
With ``cor > 0``, a switch is not counted as a changeover until the
subject has emitted ``cor`` *consecutive* responses on the new
operandum. The COD timer arms only on the ``cor``-th such response
(and at that response's timestamp). Responses 1..cor-1 on the new
operandum are not yet "a changeover" and are therefore not gated;
reinforcement on those early responses is allowed. This mirrors the
``cor`` semantics of ``contingency-dsl``.

Choice analytics NOT in scope
-----------------------------
By design this class handles only reinforcement *gating*. Choice
allocation, exclusive-choice detection, and other analytics live in
``session-analyzer`` and consume the event log, not the schedule
itself (see Issue #40).

Asymmetric COD — not implemented
--------------------------------
``cod_asymmetric: dict[tuple[str, str], float]`` (per-direction COD)
is deferred. When added, it will be a drop-in extension of the
scalar ``cod`` parameter; the single ``_cod_active`` helper below is
the extension point.

References
----------
Catania, A. C. (1966). Concurrent performances: Reinforcement
interaction and response independence. *Journal of the Experimental
Analysis of Behavior*, 9(3), 253-263.
https://doi.org/10.1901/jeab.1966.9-253

Herrnstein, R. J. (1961). Relative and absolute strength of response
as a function of frequency of reinforcement. *Journal of the
Experimental Analysis of Behavior*, 4(3), 267-272.
https://doi.org/10.1901/jeab.1961.4-267
"""

from __future__ import annotations

from contingency.entities import Outcome, ResponseEvent
from contingency.errors import ScheduleConfigError, ScheduleStateError
from contingency.interfaces import Schedule

__all__ = ["Concurrent"]

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


class Concurrent:
    """Concurrent schedule with changeover gating.

    Parameters
    ----------
    components:
        Mapping from operandum identifier (e.g. ``"left"``) to a
        :class:`Schedule`. Must contain at least two entries. Insertion
        order is preserved.
    cod:
        Changeover Delay in the caller's time units. After a changeover,
        reinforcement on the new operandum is suppressed for ``cod``
        time units. Default ``0.0`` disables COD. Must be ``>= 0``.
    cor:
        Changeover Ratio: number of consecutive responses on a new
        operandum required for the switch to count as a changeover.
        Default ``0`` treats every switch as immediate. Must be ``>= 0``.

    Notes
    -----
    Time advancement. On every :meth:`step` each component receives a
    :meth:`Schedule.step` call so that time-based components (FT, VT,
    RT) can advance their internal clocks. Only the component keyed by
    ``event.operandum`` receives the :class:`ResponseEvent`; all other
    components receive ``None`` at the same ``now``.

    Gating. If a component returns a reinforced outcome while COD is
    active, :class:`Concurrent` silently consumes the reinforcement and
    returns an un-reinforced outcome carrying
    ``meta["cod_suppressed"] = True`` alongside
    ``meta["operandum"] = <key>``. This matches standard operant-box
    behavior: the reinforcer is not delivered to the subject even
    though the component's internal state has advanced.
    """

    __slots__ = (
        "_components",
        "_cod",
        "_cor",
        "_last_operandum",
        "_switch_time",
        "_consecutive_new_count",
        "_last_now",
    )

    def __init__(
        self,
        components: dict[str, Schedule],
        cod: float = 0.0,
        cor: int = 0,
    ) -> None:
        if not isinstance(components, dict):
            raise ScheduleConfigError(
                f"Concurrent requires a dict of components, got "
                f"{type(components).__name__}"
            )
        if len(components) < 2:
            raise ScheduleConfigError(
                f"Concurrent requires at least 2 components, got {len(components)}"
            )
        if isinstance(cod, bool) or not isinstance(cod, (int, float)):
            raise ScheduleConfigError(
                f"Concurrent requires a numeric cod, got {type(cod).__name__}"
            )
        if cod < 0:
            raise ScheduleConfigError(f"Concurrent requires cod >= 0, got {cod}")
        if not isinstance(cor, int) or isinstance(cor, bool):
            raise ScheduleConfigError(
                f"Concurrent requires an int cor, got {type(cor).__name__}"
            )
        if cor < 0:
            raise ScheduleConfigError(f"Concurrent requires cor >= 0, got {cor}")

        # dict(...) to take a shallow copy; preserves insertion order.
        self._components: dict[str, Schedule] = dict(components)
        self._cod: float = float(cod)
        self._cor: int = cor
        self._last_operandum: str | None = None
        self._switch_time: float | None = None
        self._consecutive_new_count: int = 0
        self._last_now: float | None = None

    @property
    def components(self) -> dict[str, Schedule]:
        """The component mapping (read-only view; modifying has no effect)."""
        return dict(self._components)

    @property
    def cod(self) -> float:
        """The configured changeover delay."""
        return self._cod

    @property
    def cor(self) -> int:
        """The configured changeover ratio."""
        return self._cor

    def _advance_components(
        self, now: float, event_operandum: str | None, event: ResponseEvent | None
    ) -> Outcome:
        """Call step() on every component; only ``event_operandum`` receives ``event``.

        Returns the outcome produced by the component that received the
        event; otherwise the last-reinforced outcome observed on any
        component during the tick (time-based components may fire on a
        bare tick). If multiple components fire simultaneously on a
        bare tick, the first one (by insertion order) wins and the
        others are silently ignored — the caller should step often
        enough that this does not occur.
        """
        outcome_for_event: Outcome | None = None
        tick_outcome: Outcome | None = None
        for key, component in self._components.items():
            if key == event_operandum:
                outcome_for_event = component.step(now, event)
            else:
                out = component.step(now, None)
                if tick_outcome is None and out.reinforced:
                    tick_outcome = out
        if outcome_for_event is not None:
            return outcome_for_event
        return tick_outcome if tick_outcome is not None else Outcome()

    def _register_event(self, operandum: str, now: float) -> None:
        """Update changeover state for a response at ``operandum`` / ``now``.

        Implements the ``cor`` semantics: a changeover is confirmed
        only after ``cor`` consecutive responses on a new operandum.
        Once confirmed, ``_switch_time`` is anchored at ``now`` so the
        COD lockout starts from the confirming response.
        """
        if self._last_operandum is None:
            # First response ever: not a changeover.
            self._last_operandum = operandum
            self._consecutive_new_count = 0
            return

        if operandum == self._last_operandum:
            # Staying on the same operandum. Resets the streak counter.
            self._consecutive_new_count = 0
            return

        # Operandum differs from the last response's operandum.
        if self._cor == 0:
            # Immediate changeover: arm COD now and bind to new operandum.
            self._switch_time = now
            self._last_operandum = operandum
            self._consecutive_new_count = 0
            return

        # cor > 0: count this response as part of the switching streak.
        # _last_operandum stays as the previous confirmed operandum
        # until the streak reaches `cor`.
        self._consecutive_new_count += 1
        if self._consecutive_new_count >= self._cor:
            self._switch_time = now
            self._last_operandum = operandum
            self._consecutive_new_count = 0

    def _cod_active(self, operandum: str, now: float) -> bool:
        """Return True iff reinforcement on ``operandum`` is currently gated.

        COD gating applies to the operandum that the subject *has
        switched to* (i.e. the current ``_last_operandum`` after a
        confirmed changeover). Responses on other operanda are never
        gated by this scalar-COD implementation.
        """
        if self._cod <= 0.0:
            return False
        if self._switch_time is None:
            return False
        if operandum != self._last_operandum:
            return False
        return (now - self._switch_time) < self._cod - _TIME_TOL

    def step(self, now: float, event: ResponseEvent | None = None) -> Outcome:
        _check_time(self._last_now, now)
        _check_event(now, event)
        self._last_now = now

        if event is None:
            # Pure tick: advance every component's clock. A time-based
            # component may fire here; its outcome is returned as-is
            # because no changeover is involved.
            return self._advance_components(now, None, None)

        operandum = event.operandum
        if operandum not in self._components:
            raise ScheduleConfigError(
                f"Response event references unknown operandum {operandum!r}; "
                f"known operanda: {sorted(self._components.keys())!r}"
            )

        # Step all components; only the matched one sees ``event``. We
        # advance the component *before* deciding about gating so that
        # a suppressed reinforcer still advances the component's state
        # (the cost of the response is paid by the component regardless
        # of whether the subject receives the food).
        outcome = self._advance_components(now, operandum, event)

        # Update changeover bookkeeping. Registration happens *before*
        # gating so that the response which completes a changeover is
        # itself covered by the COD window it opens: the changeover
        # response is the first "on the new operandum" response and
        # COD starts at that instant (Catania, 1966).
        self._register_event(operandum, now)

        cod_active = self._cod_active(operandum, now)

        if outcome.reinforced and cod_active:
            # Absorb the reinforcement: component advanced but the
            # subject does not receive the reinforcer.
            return Outcome(
                reinforced=False,
                reinforcer=None,
                meta={"cod_suppressed": True, "operandum": operandum},
            )
        return outcome

    def reset(self) -> None:
        for component in self._components.values():
            component.reset()
        self._last_operandum = None
        self._switch_time = None
        self._consecutive_new_count = 0
        self._last_now = None
