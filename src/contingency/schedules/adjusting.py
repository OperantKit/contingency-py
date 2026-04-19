"""Adjusting schedules.

Under an adjusting schedule a single numeric parameter — the ratio
requirement, the interval length, the delay to reinforcement, or the
reinforcer magnitude — shifts by a fixed ``step`` after every earned
reinforcer. The procedure is central to self-control and
delay-discounting research, where the adjustment converges on an
indifference point between two alternatives.

This module implements the simplest variant: the parameter always
moves *up* after a reinforcement, bounded by optional
``minimum`` / ``maximum`` clamps. Choice-contingent adjustments (in
which the sign of the step depends on the subject's choice between
concurrent alternatives) are deferred to experiment-level logic.

The runtime supports two targets:

* ``target="ratio"`` — the schedule behaves like an ``FR`` whose
  ratio requirement is updated to ``current_value`` (rounded to the
  nearest positive integer) each time a reinforcer is earned.
* ``target="interval"`` — the schedule behaves like an ``FI`` whose
  interval length is set to ``current_value`` each time a reinforcer
  is earned.

The remaining AST targets (``"delay"``, ``"amount"``) are carried
through the DSL for downstream consumers (e.g. an apparatus that
applies a reinforcement delay) but are not yet realised at runtime;
constructing an :class:`AdjustingSchedule` with these targets raises
:class:`~contingency.errors.ScheduleConfigError`.

References
----------
Mazur, J. E. (1987). An adjusting procedure for studying delayed
reinforcement. In M. L. Commons, J. E. Mazur, J. A. Nevin, & H.
Rachlin (Eds.), *Quantitative analyses of behavior, Vol. 5: The
effect of delay and of intervening events on reinforcement value*
(pp. 55-73). Lawrence Erlbaum Associates.
"""

from __future__ import annotations

from contingency.entities import Outcome, Reinforcer, ResponseEvent
from contingency.errors import ScheduleConfigError, ScheduleStateError

__all__ = ["AdjustingSchedule"]

_TIME_TOL = 1e-9

_SUPPORTED_TARGETS: frozenset[str] = frozenset({"ratio", "interval"})
_KNOWN_TARGETS: frozenset[str] = frozenset(
    {"ratio", "interval", "delay", "amount"}
)


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


class AdjustingSchedule:
    """Adjusting schedule with a single numeric parameter under update.

    The adjusted parameter starts at ``start`` and shifts by ``step``
    after every earned reinforcer, clamped into ``[minimum, maximum]``
    if either bound is supplied. ``step`` may be negative for
    decrementing variants.

    Parameters
    ----------
    target:
        One of ``"ratio"``, ``"interval"``, ``"delay"``, ``"amount"``.
        Runtime only supports ``"ratio"`` and ``"interval"``; other
        targets raise :class:`~contingency.errors.ScheduleConfigError`.
    start:
        Initial value of the adjusted parameter. Must be strictly
        positive; for ``target="ratio"`` it must round to ``>= 1``.
    step:
        Increment applied after each reinforcement (can be negative).
    minimum:
        Optional lower clamp. For ``target="ratio"`` the effective
        ratio never drops below ``max(1, round(minimum))``.
    maximum:
        Optional upper clamp.

    Notes
    -----
    * For ``target="ratio"`` the schedule wraps an ``FR``-like counter
      whose requirement is ``max(1, round(current_value))``. On the
      response that meets the requirement a reinforcer is emitted,
      the counter resets to zero, and ``current_value`` moves by
      ``step`` (clamped).
    * For ``target="interval"`` the schedule wraps an ``FI``-like
      timer. The first response at or after ``now - anchor >=
      current_value`` is reinforced; the anchor moves to ``now`` and
      ``current_value`` advances.

    The schedule is time-unit-agnostic: ``start``, ``step``,
    ``minimum``, ``maximum`` and the emitted times all live on the
    caller's monotonic clock for ``target="interval"``. For
    ``target="ratio"`` those parameters are response counts and
    ``now`` participates only in monotonicity checks.

    References
    ----------
    Mazur, J. E. (1987). An adjusting procedure for studying delayed
    reinforcement. In M. L. Commons, J. E. Mazur, J. A. Nevin, & H.
    Rachlin (Eds.), *Quantitative analyses of behavior, Vol. 5*
    (pp. 55-73). Lawrence Erlbaum Associates.
    """

    __slots__ = (
        "_target",
        "_start",
        "_step",
        "_min",
        "_max",
        "_current",
        "_count",
        "_anchor",
        "_last_now",
    )

    def __init__(
        self,
        target: str,
        start: float,
        step: float,
        minimum: float | None = None,
        maximum: float | None = None,
    ) -> None:
        if not isinstance(target, str):
            raise ScheduleConfigError(
                f"AdjustingSchedule requires a str target, got {type(target).__name__}"
            )
        if target not in _KNOWN_TARGETS:
            raise ScheduleConfigError(
                f"AdjustingSchedule target must be one of "
                f"{sorted(_KNOWN_TARGETS)}, got {target!r}"
            )
        if target not in _SUPPORTED_TARGETS:
            raise ScheduleConfigError(
                f"target={target!r} runtime not yet implemented"
            )
        if isinstance(start, bool) or not isinstance(start, (int, float)):
            raise ScheduleConfigError(
                f"AdjustingSchedule requires a numeric start, got "
                f"{type(start).__name__}"
            )
        if isinstance(step, bool) or not isinstance(step, (int, float)):
            raise ScheduleConfigError(
                f"AdjustingSchedule requires a numeric step, got "
                f"{type(step).__name__}"
            )
        if minimum is not None and (
            isinstance(minimum, bool) or not isinstance(minimum, (int, float))
        ):
            raise ScheduleConfigError(
                f"AdjustingSchedule minimum must be numeric or None, got "
                f"{type(minimum).__name__}"
            )
        if maximum is not None and (
            isinstance(maximum, bool) or not isinstance(maximum, (int, float))
        ):
            raise ScheduleConfigError(
                f"AdjustingSchedule maximum must be numeric or None, got "
                f"{type(maximum).__name__}"
            )
        if (
            minimum is not None
            and maximum is not None
            and float(minimum) > float(maximum)
        ):
            raise ScheduleConfigError(
                f"AdjustingSchedule requires minimum <= maximum, "
                f"got minimum={minimum!r}, maximum={maximum!r}"
            )
        start_f = float(start)
        if start_f <= 0:
            raise ScheduleConfigError(
                f"AdjustingSchedule start must be > 0, got {start!r}"
            )
        # Ratio-family: enforce integer-like start that rounds to >= 1.
        if target == "ratio" and round(start_f) < 1:
            raise ScheduleConfigError(
                f"AdjustingSchedule ratio start must round to >= 1, "
                f"got {start!r}"
            )

        self._target: str = target
        self._start: float = start_f
        self._step: float = float(step)
        self._min: float | None = float(minimum) if minimum is not None else None
        self._max: float | None = float(maximum) if maximum is not None else None
        # Ensure the initial current value satisfies the clamp bounds
        # even if the caller's start sits just outside them. This makes
        # the reset contract ``current_value == clamp(start)``.
        self._current: float = self._clamp(start_f)
        self._count: int = 0
        self._anchor: float | None = None
        self._last_now: float | None = None

    # --- Properties -------------------------------------------------------

    @property
    def target(self) -> str:
        """The adjusted parameter: ``"ratio"`` or ``"interval"``."""
        return self._target

    @property
    def start(self) -> float:
        """The initial value of the adjusted parameter."""
        return self._start

    @property
    def step_size(self) -> float:
        """The increment applied after each reinforcement."""
        return self._step

    @property
    def minimum(self) -> float | None:
        """The lower clamp, or ``None`` if unbounded below."""
        return self._min

    @property
    def maximum(self) -> float | None:
        """The upper clamp, or ``None`` if unbounded above."""
        return self._max

    @property
    def current_value(self) -> float:
        """The current (clamped) value of the adjusted parameter."""
        return self._current

    # --- Schedule protocol -----------------------------------------------

    def step(self, now: float, event: ResponseEvent | None = None) -> Outcome:
        _check_time(self._last_now, now)
        _check_event(now, event)
        self._last_now = now

        if self._target == "ratio":
            return self._step_ratio(now, event)
        # target == "interval"
        return self._step_interval(now, event)

    def reset(self) -> None:
        self._current = self._clamp(self._start)
        self._count = 0
        self._anchor = None
        self._last_now = None

    # --- Internals --------------------------------------------------------

    def _clamp(self, value: float) -> float:
        if self._min is not None and value < self._min:
            value = self._min
        if self._max is not None and value > self._max:
            value = self._max
        return value

    def _advance(self) -> None:
        self._current = self._clamp(self._current + self._step)

    def _step_ratio(
        self, now: float, event: ResponseEvent | None
    ) -> Outcome:
        if event is None:
            return Outcome()
        requirement = max(1, int(round(self._current)))
        self._count += 1
        if self._count >= requirement:
            self._count = 0
            self._advance()
            return Outcome(reinforced=True, reinforcer=Reinforcer(time=now))
        return Outcome()

    def _step_interval(
        self, now: float, event: ResponseEvent | None
    ) -> Outcome:
        # Anchor on the first step so the first interval is measured
        # from the caller's initial clock, not from a hard-coded t=0.
        if self._anchor is None:
            self._anchor = now
        if event is None:
            return Outcome()
        elapsed = now - self._anchor
        if elapsed + _TIME_TOL >= self._current:
            self._anchor = now
            self._advance()
            return Outcome(reinforced=True, reinforcer=Reinforcer(time=now))
        return Outcome()
