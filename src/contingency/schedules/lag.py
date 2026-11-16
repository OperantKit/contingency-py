"""Lag schedule — differential reinforcement of response variability.

A *unit* is ``length`` consecutive responses, read as the sequence of
the operanda they were emitted on. Units do not overlap: every
``length``-th response closes a unit. At the end of a unit the schedule
reinforces iff the unit differs from each of the previous ``n`` units.
While fewer than ``n`` units have been completed, the unit is compared
with the ones there are (the first unit is always reinforced). Every
completed unit enters the comparison list, reinforced or not.
``Lag 0`` reinforces every unit. Responses inside a unit are never
reinforced; ticks without a response do nothing.

On a single operandum every unit is the same, so ``Lag n`` with
``n >= 1`` reinforces only the first unit. ``step`` and ``reset`` run
on the Rust engine (``contingency_core``).

References
----------
Page, S., & Neuringer, A. (1985). Variability is an operant. *Journal
of Experimental Psychology: Animal Behavior Processes*, 11(3), 429-452.
https://doi.org/10.1037/0097-7403.11.3.429
"""

from __future__ import annotations

from typing import Any

from contingency.entities import Outcome, ResponseEvent
from contingency.errors import ScheduleConfigError
from contingency.schedules._rust import rust_schedule

__all__ = ["Lag"]

#: Largest ``n`` and ``length`` accepted (the Rust engine's bound).
LAG_MAX = 2**32 - 1


def _check_int(name: str, value: object, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ScheduleConfigError(f"Lag {name} must be an int, got {type(value).__name__}")
    if not minimum <= value <= LAG_MAX:
        raise ScheduleConfigError(
            f"Lag {name} must be in [{minimum}, {LAG_MAX}], got {value}"
        )
    return value


class Lag:
    """Lag ``n`` schedule over units of ``length`` responses.

    Parameters
    ----------
    n:
        Number of previous units the current unit must differ from
        (``0 <= n <= LAG_MAX``).
    length:
        Number of responses in one unit (``1 <= length <= LAG_MAX``).
        Defaults to 1.
    """

    __slots__ = ("_rust", "_n", "_length")

    def __init__(self, n: int, length: int = 1) -> None:
        self._n: int = _check_int("n", n, 0)
        self._length: int = _check_int("length", length, 1)
        self._rust: Any = rust_schedule(self._rust_spec())

    def _rust_spec(self) -> dict[str, Any]:
        return {"type": "Lag", "params": {"n": self._n, "length": self._length}}

    @property
    def engine(self) -> str:
        """Always ``"rust"``: the Rust engine runs this schedule."""
        return "rust"

    @property
    def n(self) -> int:
        """Number of previous units the current unit must differ from."""
        return self._n

    @property
    def length(self) -> int:
        """Number of responses in one unit."""
        return self._length

    def step(self, now: float, event: ResponseEvent | None = None) -> Outcome:
        return self._rust.step(now, event)  # type: ignore[no-any-return]

    def reset(self) -> None:
        """Clear the current unit, the comparison list and the clock."""
        self._rust.reset()
