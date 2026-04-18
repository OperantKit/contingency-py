"""Core value objects for the contingency engine.

All types are frozen dataclasses — schedules hold state internally but
their inputs and outputs are immutable.

Time is unit-agnostic: every ``time`` field is a float on a monotonic
clock. The unit (seconds, milliseconds, arbitrary) is declared by the
caller (typically ``contingency-dsl`` meta or ``experiment-io``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "ResponseEvent",
    "Observation",
    "Reinforcer",
    "Outcome",
]


@dataclass(frozen=True)
class ResponseEvent:
    """A single response emitted by the subject at ``time``.

    Attributes
    ----------
    time:
        Monotonic timestamp of the response. Unit-agnostic.
    operandum:
        Identifier of the operandum (lever, key, button). Used by
        compound schedules to route the event.
    """

    time: float
    operandum: str = "main"


@dataclass(frozen=True)
class Observation:
    """A discrete-time snapshot of the subject's cumulative state."""

    time: float
    response_count: int = 0


@dataclass(frozen=True)
class Reinforcer:
    """A scheduled reinforcement event.

    ``magnitude`` encodes the amount (pellets, ml, points). Valence is
    positive for reinforcers, negative for punishers — aversive control
    is modelled by negative-magnitude Reinforcers with ``label="SR-"``.
    """

    time: float
    magnitude: float = 1.0
    label: str = "SR+"


@dataclass(frozen=True)
class Outcome:
    """Result returned by :meth:`Schedule.step`.

    Attributes
    ----------
    reinforced:
        True if the step produced a reinforcement delivery.
    reinforcer:
        The scheduled :class:`Reinforcer`, or ``None`` when
        ``reinforced`` is ``False``.
    meta:
        Optional backend-specific payload (e.g. which component of a
        compound schedule fired). Kept empty by default.
    """

    reinforced: bool = False
    reinforcer: Reinforcer | None = None
    meta: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.reinforced and self.reinforcer is None:
            raise ValueError("reinforced=True requires a Reinforcer")
        if not self.reinforced and self.reinforcer is not None:
            raise ValueError("reinforcer must be None when reinforced=False")
