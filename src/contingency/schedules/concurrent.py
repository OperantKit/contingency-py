"""Concurrent compound schedule with Changeover Delay (COD).

A concurrent schedule presents two or more component schedules on
distinct *operanda* (keys, levers, buttons). Each component is a
standalone :class:`~contingency.interfaces.Schedule`; responses on
operandum ``k`` are routed exclusively to ``components[k]``. The
composite returns the :class:`~contingency.entities.Outcome` produced by
the component that received the event, subject to changeover gating.

Changeover Delay (Catania, 1966)
--------------------------------
When the subject switches from operandum A to operandum B, no response
on B is reinforced for ``cod`` time units after the switch. This
discourages rapid alternation and superstitious changeover responding.
The timer resets on every subsequent changeover: "since the last
changeover".

A reinforcer that B's component sets up during the COD is *held*, not
lost: it is delivered by the first response on B after the COD ends,
and B's component does not start its next interval (or ratio) until
that delivery. This is the procedure of Herrnstein (1961): "no
response was reinforced within 1.5 seconds of a changeover", with
the programmed reinforcer kept until collected.

If no changeover has yet occurred (e.g. the very first response of a
session), COD is not active. This module implements the standard
"since-last-changeover" semantics of Catania (1966): each fresh
switch re-anchors the COD clock.

Fixed-Ratio Changeover (FRCO)
----------------------
With ``frco > 0``, a switch is not counted as a changeover until the
subject has emitted ``frco`` *consecutive* responses on the new
operandum. The COD timer arms only on the ``frco``-th such response
(and at that response's timestamp). Responses 1..frco-1 on the new
operandum are not yet "a changeover" and are therefore not gated;
reinforcement on those early responses is allowed. This mirrors the
``frco`` semantics of ``contingency-dsl``.

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

import copy

from contingency.entities import Outcome, Reinforcer, ResponseEvent
from contingency.errors import ScheduleConfigError, ScheduleStateError
from contingency.interfaces import Schedule
from contingency.schedules.interval import FI, RI, VI
from contingency.schedules.ratio import FR, RR, VR

__all__ = ["Concurrent"]

_TIME_TOL = 1e-9

# Components whose reinforcer can be held *inside* the component: a
# time tick never reinforces them and responses matter only through
# the reinforcing one, so withholding that response keeps the
# reinforcer set up and delays the next interval or ratio.
_HOLD_IN_COMPONENT: tuple[type, ...] = (FR, VR, RR, FI, VI, RI)


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
        no response on the new operandum is reinforced for ``cod`` time
        units; a reinforcer set up meanwhile is held until the first
        response after the COD. Default ``0.0`` disables COD. Must be
        ``>= 0``.
    frco:
        fixed-ratio changeover: number of consecutive responses on a new
        operandum required for the switch to count as a changeover.
        Default ``0`` treats every switch as immediate. Must be ``>= 0``.

    Notes
    -----
    Time advancement. On every :meth:`step` each component receives a
    :meth:`Schedule.step` call so that time-based components (FT, VT,
    RT) can advance their internal clocks. Only the component keyed by
    ``event.operandum`` receives the :class:`ResponseEvent`; all other
    components receive ``None`` at the same ``now``.

    Gating. While COD is active on the responded operandum, a
    reinforcer that the response would produce is held and
    :class:`Concurrent` returns an un-reinforced outcome carrying
    ``meta["cod_suppressed"] = True`` alongside
    ``meta["operandum"] = <key>``. The first response on that operandum
    after the COD collects the held reinforcer. If another component
    delivers a reinforcer on its tick at that step and no punisher
    fires, that reinforcer is returned instead, with
    ``meta["cod_suppressed"] = True`` and no ``"operandum"`` key. How it
    is held depends on the component:

    - FR, VR, RR, FI, VI, RI: the response is first tried on a copy of
      the component. If the copy would reinforce, the real component
      receives only a time tick, so the reinforcer stays set up and its
      next interval or ratio does not start until delivery. A response
      that would not reinforce is applied as usual, so it counts toward
      a ratio.
    - Any other component (time-based, differential, nested compound):
      the response is applied as usual, so the component sees every
      response, and the reinforcer it produces is kept by
      :class:`Concurrent` until delivery. The component's own next
      cycle starts at once.
    """

    __slots__ = (
        "_components",
        "_cod",
        "_cod_directional",
        "_frco",
        "_punish",
        "_last_operandum",
        "_last_confirmed_operandum",
        "_switch_time",
        "_active_cod",
        "_consecutive_new_count",
        "_last_now",
        "_pending",
    )

    def __init__(
        self,
        components: dict[str, Schedule],
        cod: float = 0.0,
        frco: int = 0,
        cod_directional: dict[tuple[str, str], float] | None = None,
        punish: dict[str, Schedule] | None = None,
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
        if not isinstance(frco, int) or isinstance(frco, bool):
            raise ScheduleConfigError(
                f"Concurrent requires an int frco, got {type(frco).__name__}"
            )
        if frco < 0:
            raise ScheduleConfigError(f"Concurrent requires frco >= 0, got {frco}")

        # dict(...) to take a shallow copy; preserves insertion order.
        self._components: dict[str, Schedule] = dict(components)
        self._cod: float = float(cod)
        self._frco: int = frco

        # Validate directional COD: keys must be (str, str), no self-
        # transitions, no duplicates (duplicates are intrinsic to dict
        # keys; we also reject the (A, A) form).
        self._cod_directional: dict[tuple[str, str], float] | None = None
        if cod_directional is not None:
            if not isinstance(cod_directional, dict):
                raise ScheduleConfigError(
                    f"Concurrent cod_directional must be a dict, got "
                    f"{type(cod_directional).__name__}"
                )
            validated: dict[tuple[str, str], float] = {}
            for key, value in cod_directional.items():
                if (
                    not isinstance(key, tuple)
                    or len(key) != 2
                    or not all(isinstance(k, str) for k in key)
                ):
                    raise ScheduleConfigError(
                        f"Concurrent cod_directional keys must be "
                        f"(from, to) tuples of str, got {key!r}"
                    )
                from_op, to_op = key
                if from_op == to_op:
                    raise ScheduleConfigError(
                        f"Concurrent cod_directional rejects self-"
                        f"transition {key!r}"
                    )
                if isinstance(value, bool) or not isinstance(
                    value, (int, float)
                ):
                    raise ScheduleConfigError(
                        f"Concurrent cod_directional values must be "
                        f"numeric, got {type(value).__name__} for {key!r}"
                    )
                if value < 0:
                    raise ScheduleConfigError(
                        f"Concurrent cod_directional values must be "
                        f">= 0, got {value} for {key!r}"
                    )
                validated[(from_op, to_op)] = float(value)
            if validated:
                self._cod_directional = validated

        # Validate punish dict.
        self._punish: dict[str, Schedule] | None = None
        if punish is not None:
            if not isinstance(punish, dict):
                raise ScheduleConfigError(
                    f"Concurrent punish must be a dict, got "
                    f"{type(punish).__name__}"
                )
            punish_validated: dict[str, Schedule] = {}
            for op_key, sched in punish.items():
                if not isinstance(op_key, str):
                    raise ScheduleConfigError(
                        f"Concurrent punish keys must be operandum "
                        f"strings, got {type(op_key).__name__}"
                    )
                if op_key not in self._components:
                    raise ScheduleConfigError(
                        f"Concurrent punish references unknown operandum "
                        f"{op_key!r}; known operanda: "
                        f"{sorted(self._components.keys())!r}"
                    )
                if not isinstance(sched, Schedule):
                    raise ScheduleConfigError(
                        f"Concurrent punish[{op_key!r}] must implement "
                        f"Schedule, got {type(sched).__name__}"
                    )
                punish_validated[op_key] = sched
            if punish_validated:
                self._punish = punish_validated

        self._last_operandum: str | None = None
        # Previously confirmed operandum, captured at the moment a new
        # changeover is confirmed. Used to resolve directional COD.
        self._last_confirmed_operandum: str | None = None
        self._switch_time: float | None = None
        # Currently armed COD duration (resolved directional or base).
        self._active_cod: float = float(cod)
        self._consecutive_new_count: int = 0
        self._last_now: float | None = None
        # Reinforcers held by the compound (components outside
        # ``_HOLD_IN_COMPONENT``), per operandum, oldest first.
        self._pending: dict[str, list[Reinforcer]] = {}

    @property
    def components(self) -> dict[str, Schedule]:
        """The component mapping (read-only view; modifying has no effect)."""
        return dict(self._components)

    @property
    def cod(self) -> float:
        """The configured changeover delay."""
        return self._cod

    @property
    def frco(self) -> int:
        """The configured fixed-ratio changeover."""
        return self._frco

    @property
    def cod_directional(self) -> dict[tuple[str, str], float] | None:
        """The configured per-direction COD overrides, if any."""
        if self._cod_directional is None:
            return None
        return dict(self._cod_directional)

    @property
    def punish(self) -> dict[str, Schedule] | None:
        """The configured per-operandum punishment schedules, if any."""
        if self._punish is None:
            return None
        return dict(self._punish)

    def _resolve_cod(self, from_op: str | None, to_op: str) -> float:
        """Return the COD to arm for a switch ``from_op -> to_op``.

        Uses the directional override when present; otherwise falls
        back to the base ``cod``. ``from_op=None`` (no prior operandum)
        never arms directional COD.
        """
        if (
            self._cod_directional is not None
            and from_op is not None
            and (from_op, to_op) in self._cod_directional
        ):
            return self._cod_directional[(from_op, to_op)]
        return self._cod

    def _step_gated(
        self, component: Schedule, now: float, event: ResponseEvent
    ) -> tuple[Outcome, bool]:
        """Step ``component`` with ``event`` while COD gates its operandum.

        For :data:`_HOLD_IN_COMPONENT` types the response is first tried
        on a copy. If the copy would reinforce, the real component only
        receives a time tick, so the reinforcer stays set up (and its
        next interval or ratio does not start) until the first response
        after the COD. For any other component the response is applied
        as usual and a reinforcer it produces is queued in
        ``_pending[event.operandum]`` for later delivery.

        Returns ``(outcome, held)``.
        """
        if isinstance(component, _HOLD_IN_COMPONENT):
            probe = copy.deepcopy(component)
            if probe.step(now, event).reinforced:
                component.step(now, None)
                return Outcome(), True
            return component.step(now, event), False
        out = component.step(now, event)
        if out.reinforced and out.reinforcer is not None:
            self._pending.setdefault(event.operandum, []).append(out.reinforcer)
            return Outcome(), True
        return out, False

    def _deliver_pending(
        self, operandum: str, now: float, outcome: Outcome, from_event: bool
    ) -> tuple[Outcome, bool]:
        """Deliver the oldest reinforcer held by :class:`Concurrent` on ``operandum``.

        Called for an ungated response. If the component reinforced on
        this response too, that reinforcer joins the back of the queue,
        so each response delivers at most one reinforcer.
        """
        queue = self._pending.get(operandum)
        if not queue:
            return outcome, from_event
        if from_event and outcome.reinforced and outcome.reinforcer is not None:
            queue.append(outcome.reinforcer)
        held = queue.pop(0)
        if not queue:
            del self._pending[operandum]
        delivered = Reinforcer(time=now, magnitude=held.magnitude, label=held.label)
        return Outcome(reinforced=True, reinforcer=delivered), True

    def _advance_components(
        self,
        now: float,
        event_operandum: str | None,
        event: ResponseEvent | None,
        gated: bool = False,
    ) -> tuple[Outcome, bool, bool]:
        """Call step() on every component; only ``event_operandum`` receives ``event``.

        Returns ``(outcome, from_event, held)`` where ``from_event`` is
        ``True`` iff the returned outcome originated from the
        event-matched component, and ``held`` is ``True`` iff that
        component set up a reinforcer which COD is holding back
        (only possible when ``gated``). Priority order:

        1. The event-matched component's outcome if it reinforced
           (``from_event=True``).
        2. Otherwise, the first reinforced tick outcome observed on any
           other component, in insertion order (``from_event=False``).
        3. Otherwise, the event-matched component's (un-reinforced)
           outcome, to preserve its ``meta`` (``from_event=True``).
        4. Otherwise, an empty :class:`Outcome` (``from_event=False``).

        Rationale: a time-based component (FT/VT/RT) that fires on a
        step must not be silently dropped just because another operandum
        also saw a response on the same step. Preferring the event-side
        outcome when both reinforce matches the single-outcome API; when
        the event side does not reinforce the tick-side reinforcement is
        surfaced rather than consumed. If multiple non-event components
        fire on the same tick, the first one (by insertion order) wins
        and the others are silently dropped — the caller should step
        often enough that this does not occur.
        """
        outcome_for_event: Outcome | None = None
        tick_outcome: Outcome | None = None
        held = False
        for key, component in self._components.items():
            if key == event_operandum:
                if gated and event is not None:
                    outcome_for_event, held = self._step_gated(component, now, event)
                else:
                    outcome_for_event = component.step(now, event)
            else:
                out = component.step(now, None)
                if tick_outcome is None and out.reinforced:
                    tick_outcome = out
        if outcome_for_event is not None and outcome_for_event.reinforced:
            return outcome_for_event, True, held
        if tick_outcome is not None:
            return tick_outcome, False, held
        if outcome_for_event is not None:
            return outcome_for_event, True, held
        return Outcome(), False, held

    def _register_event(self, operandum: str, now: float) -> tuple[bool, str | None]:
        """Update changeover state for a response at ``operandum`` / ``now``.

        Returns ``(changeover_confirmed, from_op)`` where
        ``changeover_confirmed`` is True iff this response confirms a
        changeover (for punishment wiring), and ``from_op`` is the
        previously confirmed operandum at the moment the changeover
        occurred.

        Implements the ``frco`` semantics: a changeover is confirmed
        only after ``frco`` consecutive responses on a new operandum.
        Once confirmed, ``_switch_time`` is anchored at ``now`` so the
        COD lockout starts from the confirming response.
        """
        if self._last_operandum is None:
            # First response ever: not a changeover.
            self._last_operandum = operandum
            self._consecutive_new_count = 0
            return (False, None)

        if operandum == self._last_operandum:
            # Staying on the same operandum. Resets the streak counter.
            self._consecutive_new_count = 0
            return (False, None)

        # Operandum differs from the last response's operandum.
        if self._frco == 0:
            # Immediate changeover: arm COD now and bind to new operandum.
            from_op = self._last_operandum
            self._switch_time = now
            self._active_cod = self._resolve_cod(from_op, operandum)
            self._last_confirmed_operandum = from_op
            self._last_operandum = operandum
            self._consecutive_new_count = 0
            return (True, from_op)

        # frco > 0: count this response as part of the switching streak.
        # _last_operandum stays as the previous confirmed operandum
        # until the streak reaches `frco`.
        self._consecutive_new_count += 1
        if self._consecutive_new_count >= self._frco:
            from_op = self._last_operandum
            self._switch_time = now
            self._active_cod = self._resolve_cod(from_op, operandum)
            self._last_confirmed_operandum = from_op
            self._last_operandum = operandum
            self._consecutive_new_count = 0
            return (True, from_op)
        return (False, None)

    def _cod_active(self, operandum: str, now: float) -> bool:
        """Return True iff reinforcement on ``operandum`` is currently gated.

        COD gating applies to the operandum that the subject *has
        switched to* (i.e. the current ``_last_operandum`` after a
        confirmed changeover). Responses on other operanda are never
        gated by this implementation. The armed COD is the directional
        override in effect (if any) for the most recent switch;
        otherwise the base ``cod``.
        """
        if self._active_cod <= 0.0:
            return False
        if self._switch_time is None:
            return False
        if operandum != self._last_operandum:
            return False
        return (now - self._switch_time) < self._active_cod - _TIME_TOL

    def step(self, now: float, event: ResponseEvent | None = None) -> Outcome:
        _check_time(self._last_now, now)
        _check_event(now, event)
        self._last_now = now

        if event is None:
            # Pure tick: advance every component's clock. A time-based
            # component may fire here; its outcome is returned as-is
            # because no changeover is involved.
            outcome, _, _ = self._advance_components(now, None, None)
            # Ticks also advance punishment components' internal clocks
            # so time-based punishers keep correct time. A tick cannot
            # be the event that triggers a punisher fire (no response),
            # so any outcome from a punisher is discarded here.
            if self._punish is not None:
                for p_sched in self._punish.values():
                    p_sched.step(now, None)
            return outcome

        operandum = event.operandum
        if operandum not in self._components:
            raise ScheduleConfigError(
                f"Response event references unknown operandum {operandum!r}; "
                f"known operanda: {sorted(self._components.keys())!r}"
            )

        # Update changeover bookkeeping. Registration happens *before*
        # gating so that the response which completes a changeover is
        # itself covered by the COD window it opens: the changeover
        # response is the first "on the new operandum" response and
        # COD starts at that instant (Catania, 1966).
        self._register_event(operandum, now)
        gated = self._cod_active(operandum, now)

        # Step all components; only the matched one sees ``event``.
        # Under COD, a reinforcer the matched component would deliver
        # is held on that component until the first response after the
        # COD (Herrnstein, 1961).
        outcome, from_event, held = self._advance_components(
            now, operandum, event, gated=gated
        )
        if not gated:
            outcome, from_event = self._deliver_pending(
                operandum, now, outcome, from_event
            )

        # Advance punishment schedules for all operanda so their clocks
        # stay aligned with ``now``. Only the matched operandum's
        # punisher receives the event; other punishers receive a tick.
        punish_outcome: Outcome | None = None
        if self._punish is not None:
            for key, p_sched in self._punish.items():
                if key == operandum:
                    punish_outcome = p_sched.step(now, event)
                else:
                    p_sched.step(now, None)

        cod_suppressed = held

        base_fired = outcome.reinforced and from_event
        punisher_fired = punish_outcome is not None and punish_outcome.reinforced

        if cod_suppressed and not punisher_fired:
            # The reinforcer stays set up on the component and is not
            # delivered on this response. A reinforcer another component
            # delivered on its tick at this step is still returned, with
            # ``cod_suppressed`` recording the hold.
            if outcome.reinforced and not from_event:
                return Outcome(
                    reinforced=True,
                    reinforcer=outcome.reinforcer,
                    meta={**outcome.meta, "cod_suppressed": True},
                )
            return Outcome(
                reinforced=False,
                reinforcer=None,
                meta={"cod_suppressed": True, "operandum": operandum},
            )

        if base_fired and punisher_fired:
            # Net-magnitude compound outcome.
            assert outcome.reinforcer is not None
            assert punish_outcome is not None and punish_outcome.reinforcer is not None
            net = outcome.reinforcer.magnitude - punish_outcome.reinforcer.magnitude
            if net > 0:
                label = "SR+"
            elif net < 0:
                label = "SR-"
            else:
                label = "SR0"
            return Outcome(
                reinforced=True,
                reinforcer=Reinforcer(
                    time=outcome.reinforcer.time,
                    magnitude=net,
                    label=label,
                ),
                meta={
                    "operandum": operandum,
                    "base_fired": True,
                    "punisher_fired": True,
                    "net_magnitude": net,
                },
            )

        if punisher_fired:
            assert punish_outcome is not None and punish_outcome.reinforcer is not None
            p = punish_outcome.reinforcer
            meta: dict[str, object] = {
                "operandum": operandum,
                "punisher_fired": True,
                "base_fired": False,
            }
            if cod_suppressed:
                # The base reinforcer was earned but suppressed; the
                # punisher still fires. Report the COD suppression too.
                meta["cod_suppressed"] = True
            return Outcome(
                reinforced=True,
                reinforcer=Reinforcer(
                    time=p.time, magnitude=-p.magnitude, label="SR-"
                ),
                meta=meta,
            )

        return outcome

    def reset(self) -> None:
        for component in self._components.values():
            component.reset()
        if self._punish is not None:
            for p_sched in self._punish.values():
                p_sched.reset()
        self._last_operandum = None
        self._last_confirmed_operandum = None
        self._switch_time = None
        self._active_cod = self._cod
        self._consecutive_new_count = 0
        self._last_now = None
        self._pending = {}
