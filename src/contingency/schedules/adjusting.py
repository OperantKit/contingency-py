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

``step`` and ``reset`` run on the Rust engine (``contingency_core``).
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

from typing import Any

from contingency.entities import Outcome, ResponseEvent
from contingency.errors import ScheduleConfigError
from contingency.schedules._rust import rust_state_schedule

__all__ = ["AdjustingSchedule"]

_SUPPORTED_TARGETS: frozenset[str] = frozenset({"ratio", "interval"})
_KNOWN_TARGETS: frozenset[str] = frozenset(
    {"ratio", "interval", "delay", "amount"}
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
        "_rust",
        "_target",
        "_start",
        "_step",
        "_min",
        "_max",
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
        # The engine starts ``current_value`` at ``start`` clamped into
        # the bounds, and returns to it on reset.
        self._rust: Any = rust_state_schedule(self._rust_spec(), "current_value")

    def _rust_spec(self) -> dict[str, Any]:
        params: dict[str, Any] = {
            "target": self._target,
            "start": self._start,
            "step": self._step,
        }
        if self._min is not None:
            params["minimum"] = self._min
        if self._max is not None:
            params["maximum"] = self._max
        return {"type": "AdjustingSchedule", "params": params}

    @property
    def engine(self) -> str:
        """Always ``"rust"``: the Rust engine runs this schedule."""
        return "rust"

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
        return float(self._rust.state()["current_value"])

    # --- Schedule protocol -----------------------------------------------

    def step(self, now: float, event: ResponseEvent | None = None) -> Outcome:
        return self._rust.step(now, event)  # type: ignore[no-any-return]

    def reset(self) -> None:
        self._rust.reset()
