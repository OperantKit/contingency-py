"""Aversive control schedules — shock-based contingencies.

This module implements three canonical aversive-control paradigms:

* :class:`Sidman` — non-discriminated (free-operant) avoidance
  (Sidman, 1953). Shocks are scheduled at a fixed shock-shock interval
  (SSI) in the absence of responding; each response postpones the next
  shock by a response-shock interval (RSI).
* :class:`DiscriminatedAvoidance` — signalled avoidance
  (Solomon & Wynne, 1953). Each trial begins with an inter-trial
  interval, followed by a warning stimulus. A response during the
  warning window avoids the shock; absence of response yields shock at
  the end of the warning duration.
* :class:`Escape` — aversive-termination paradigm (Keller, 1941).
  Shock is on from the start of each trial and is emitted on every
  tick until a response terminates it or a maximum ``trial_duration``
  elapses.

All three deliver **negative-magnitude** :class:`Reinforcer` events
with ``label="SR-"``. The ``shock_magnitude`` constructor parameter is
supplied as a positive number and internally negated when populating
the :class:`Reinforcer.magnitude` field.

Design note — continuous emission for Escape
---------------------------------------------
Escape in this implementation emits a shock :class:`Outcome` on
**every tick** that the shock is on — not once per trial. This mirrors
the free-operant Escape paradigm where the aversive stimulus is a
continuous presence whose duration is what the subject terminates by
responding. Analysts interested in count-per-trial rather than
tick-level exposure can aggregate over ``meta["phase"] == "shock"``
ticks downstream.

References
----------
Keller, F. S. (1941). Light-aversion in the white rat. *Psychological
Record*, 4, 235-250.

Sidman, M. (1953). Avoidance conditioning with brief shock and no
exteroceptive warning signal. *Science*, 118(3058), 157-158.
https://doi.org/10.1126/science.118.3058.157

Solomon, R. L., & Wynne, L. C. (1953). Traumatic avoidance learning:
Acquisition in normal dogs. *Psychological Monographs: General and
Applied*, 67(4), 1-19. https://doi.org/10.1037/h0093649
"""

from __future__ import annotations

from contingency.entities import Outcome, Reinforcer, ResponseEvent
from contingency.errors import ScheduleConfigError, ScheduleStateError

__all__ = ["Sidman", "DiscriminatedAvoidance", "Escape"]


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


def _validate_positive(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScheduleConfigError(
            f"{name} must be numeric, got {type(value).__name__}"
        )
    if not (value > 0):
        raise ScheduleConfigError(f"{name} must be > 0, got {value!r}")
    return float(value)


def _validate_magnitude(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScheduleConfigError(
            f"shock_magnitude must be numeric, got {type(value).__name__}"
        )
    if not (value > 0):
        raise ScheduleConfigError(
            f"shock_magnitude must be > 0 (supply as positive; internally "
            f"negated), got {value!r}"
        )
    return float(value)


class Sidman:
    """Sidman (non-discriminated) avoidance schedule — Sidman (1953).

    Parameters
    ----------
    shock_shock_interval:
        Time between shocks when no responding occurs (S-S interval).
    response_shock_interval:
        Time to the next shock after a response (R-S interval).
    shock_magnitude:
        Positive number; internally negated to encode the aversive SR-.

    Semantics
    ---------
    Internal ``next_shock_time`` starts at ``shock_shock_interval``.
    Each response resets ``next_shock_time = now + response_shock_interval``.
    When ``now >= next_shock_time`` (regardless of response at this tick),
    a shock is delivered (Outcome with reinforced=True, negative-magnitude
    Reinforcer). After a shock, ``next_shock_time = now + shock_shock_interval``
    unless a response is also emitted at this tick, in which case the
    R-S override applies. A response within the same tick as a shock
    still restarts the R-S clock (a single fresh R-S interval starting
    at ``now``).
    """

    __slots__ = (
        "_ssi",
        "_rsi",
        "_magnitude",
        "_next_shock_time",
        "_last_now",
    )

    def __init__(
        self,
        shock_shock_interval: float,
        response_shock_interval: float,
        shock_magnitude: float = 1.0,
    ) -> None:
        self._ssi: float = _validate_positive(
            "shock_shock_interval", shock_shock_interval
        )
        self._rsi: float = _validate_positive(
            "response_shock_interval", response_shock_interval
        )
        self._magnitude: float = _validate_magnitude(shock_magnitude)
        self._next_shock_time: float = self._ssi
        self._last_now: float | None = None

    @property
    def shock_shock_interval(self) -> float:
        return self._ssi

    @property
    def response_shock_interval(self) -> float:
        return self._rsi

    @property
    def shock_magnitude(self) -> float:
        return self._magnitude

    @property
    def next_shock_time(self) -> float:
        return self._next_shock_time

    def step(self, now: float, event: ResponseEvent | None = None) -> Outcome:
        _check_time(self._last_now, now)
        _check_event(now, event)
        self._last_now = now

        # Response override: postpone next shock by R-S interval.
        if event is not None:
            self._next_shock_time = now + self._rsi

        # Fire shock if time has reached / exceeded next_shock_time.
        if now + _TIME_TOL >= self._next_shock_time:
            # After a shock, schedule next shock using S-S, unless the
            # response at this tick already shifted it later (R-S > S-S).
            new_next = now + self._ssi
            if event is not None:
                # The R-S clock starts fresh at the shock time.
                new_next = now + self._rsi
            self._next_shock_time = new_next
            return Outcome(
                reinforced=True,
                reinforcer=Reinforcer(
                    time=now, magnitude=-self._magnitude, label="SR-"
                ),
                meta={"aversive_event": "shock"},
            )
        return Outcome()

    def reset(self) -> None:
        self._next_shock_time = self._ssi
        self._last_now = None


class DiscriminatedAvoidance:
    """Discriminated avoidance schedule — Solomon & Wynne (1953).

    Parameters
    ----------
    warning_duration:
        Duration of warning stimulus; shock delivered at end unless
        response.
    iti:
        Inter-trial interval after warning ends (success or failure).
    shock_magnitude:
        Magnitude of the shock (positive number).

    Semantics
    ---------
    Phase-state machine: ``'iti' -> 'warning' -> (resolved: avoid|shock)
    -> 'iti' ...``. The schedule begins in the ``iti`` phase; the first
    warning stimulus begins at ``now == iti`` (the first ITI elapses
    from ``t=0``). A response during the warning window transitions
    immediately to ``iti`` with ``aversive_event="avoidance"``. Absence
    of response by ``warning_start + warning_duration`` emits a shock
    and transitions to ``iti``.
    """

    __slots__ = (
        "_warning_duration",
        "_iti",
        "_magnitude",
        "_phase",
        "_phase_end",
        "_last_now",
    )

    def __init__(
        self,
        warning_duration: float,
        iti: float,
        shock_magnitude: float = 1.0,
    ) -> None:
        self._warning_duration: float = _validate_positive(
            "warning_duration", warning_duration
        )
        self._iti: float = _validate_positive("iti", iti)
        self._magnitude: float = _validate_magnitude(shock_magnitude)
        self._phase: str = "iti"
        self._phase_end: float = self._iti
        self._last_now: float | None = None

    @property
    def warning_duration(self) -> float:
        return self._warning_duration

    @property
    def iti(self) -> float:
        return self._iti

    @property
    def shock_magnitude(self) -> float:
        return self._magnitude

    @property
    def phase(self) -> str:
        return self._phase

    def step(self, now: float, event: ResponseEvent | None = None) -> Outcome:
        _check_time(self._last_now, now)
        _check_event(now, event)
        self._last_now = now

        # ITI phase: transition to warning when elapsed.
        if self._phase == "iti":
            if now + _TIME_TOL >= self._phase_end:
                self._phase = "warning"
                self._phase_end = now + self._warning_duration
                # Fall through to warning handling below.
            else:
                return Outcome(meta={"phase": "iti"})

        # Warning phase: response avoids, timeout yields shock.
        if self._phase == "warning":
            if event is not None:
                # Avoidance: terminate warning, enter ITI.
                self._phase = "iti"
                self._phase_end = now + self._iti
                return Outcome(
                    meta={
                        "phase": "warning",
                        "aversive_event": "avoidance",
                        "trial_outcome": "avoid",
                    }
                )
            if now + _TIME_TOL >= self._phase_end:
                # Shock delivery.
                self._phase = "iti"
                self._phase_end = now + self._iti
                return Outcome(
                    reinforced=True,
                    reinforcer=Reinforcer(
                        time=now, magnitude=-self._magnitude, label="SR-"
                    ),
                    meta={
                        "phase": "warning",
                        "aversive_event": "shock",
                        "trial_outcome": "shock",
                    },
                )
            return Outcome(meta={"phase": "warning"})

        # Defensive: unknown phase.
        raise ScheduleStateError(f"unknown phase: {self._phase!r}")

    def reset(self) -> None:
        self._phase = "iti"
        self._phase_end = self._iti
        self._last_now = None


class Escape:
    """Escape schedule — aversive-termination paradigm (Keller, 1941).

    Parameters
    ----------
    trial_duration:
        Maximum shock duration if no response (defines the failed-trial
        ceiling).
    iti:
        Inter-trial interval after shock termination.
    shock_magnitude:
        Magnitude of the shock (positive number).

    Semantics
    ---------
    Trial begins with shock ON. Every tick during shock emits a
    continuous-negative :class:`Outcome` with
    ``meta["aversive_event"]="shock"`` and ``meta["phase"]="shock"``.
    The first such tick also carries ``meta["aversive_event"]="shock_onset"``.
    A response terminates shock immediately: the terminating tick emits
    ``meta["aversive_event"]="shock_termination"`` (no shock SR- on
    termination tick), then the schedule enters ITI. If no response
    arrives within ``trial_duration`` of shock onset, the trial ends
    naturally with ``meta["aversive_event"]="trial_end"`` and enters
    ITI.
    """

    __slots__ = (
        "_trial_duration",
        "_iti",
        "_magnitude",
        "_phase",
        "_phase_end",
        "_shock_onset_emitted",
        "_last_now",
    )

    def __init__(
        self,
        trial_duration: float,
        iti: float,
        shock_magnitude: float = 1.0,
    ) -> None:
        self._trial_duration: float = _validate_positive(
            "trial_duration", trial_duration
        )
        self._iti: float = _validate_positive("iti", iti)
        self._magnitude: float = _validate_magnitude(shock_magnitude)
        # Start in shock phase at construction: trial begins immediately.
        self._phase: str = "shock"
        self._phase_end: float = self._trial_duration
        self._shock_onset_emitted: bool = False
        self._last_now: float | None = None

    @property
    def trial_duration(self) -> float:
        return self._trial_duration

    @property
    def iti(self) -> float:
        return self._iti

    @property
    def shock_magnitude(self) -> float:
        return self._magnitude

    @property
    def phase(self) -> str:
        return self._phase

    def step(self, now: float, event: ResponseEvent | None = None) -> Outcome:
        _check_time(self._last_now, now)
        _check_event(now, event)
        self._last_now = now

        if self._phase == "iti":
            if now + _TIME_TOL >= self._phase_end:
                # Start new trial with shock on.
                self._phase = "shock"
                self._phase_end = now + self._trial_duration
                self._shock_onset_emitted = False
                # Fall through to shock handling.
            else:
                return Outcome(meta={"phase": "iti"})

        if self._phase == "shock":
            # Response terminates shock.
            if event is not None:
                self._phase = "iti"
                self._phase_end = now + self._iti
                self._shock_onset_emitted = False
                return Outcome(
                    meta={
                        "phase": "shock",
                        "aversive_event": "shock_termination",
                        "trial_outcome": "escape",
                    }
                )
            # Trial timeout: shock ends without escape.
            if now + _TIME_TOL >= self._phase_end:
                self._phase = "iti"
                self._phase_end = now + self._iti
                self._shock_onset_emitted = False
                return Outcome(
                    meta={
                        "phase": "shock",
                        "aversive_event": "trial_end",
                        "trial_outcome": "failed",
                    }
                )
            # Continuous shock tick.
            is_onset = not self._shock_onset_emitted
            self._shock_onset_emitted = True
            event_label = "shock_onset" if is_onset else "shock"
            return Outcome(
                reinforced=True,
                reinforcer=Reinforcer(
                    time=now, magnitude=-self._magnitude, label="SR-"
                ),
                meta={
                    "phase": "shock",
                    "aversive_event": event_label,
                },
            )

        raise ScheduleStateError(f"unknown phase: {self._phase!r}")

    def reset(self) -> None:
        self._phase = "shock"
        self._phase_end = self._trial_duration
        self._shock_onset_emitted = False
        self._last_now = None
