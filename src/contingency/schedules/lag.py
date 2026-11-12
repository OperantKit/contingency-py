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
``n >= 1`` reinforces only the first unit.

References
----------
Page, S., & Neuringer, A. (1985). Variability is an operant. *Journal
of Experimental Psychology: Animal Behavior Processes*, 11(3), 429-452.
https://doi.org/10.1037/0097-7403.11.3.429
"""

from __future__ import annotations

from collections import deque
from typing import Any

from contingency.entities import Outcome, Reinforcer, ResponseEvent
from contingency.errors import ScheduleConfigError, ScheduleStateError
from contingency.schedules._rust import engine_name, rust_or_none

__all__ = ["Lag"]

_TIME_TOL = 1e-9

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

    __slots__ = ("_rust", "_n", "_length", "_current", "_history", "_last_now")

    def __init__(self, n: int, length: int = 1) -> None:
        self._n: int = _check_int("n", n, 0)
        self._length: int = _check_int("length", length, 1)
        self._current: list[str] = []
        self._history: deque[tuple[str, ...]] = deque(maxlen=self._n)
        self._last_now: float | None = None
        self._rust: Any = rust_or_none(self._rust_spec())

    def _rust_spec(self) -> dict[str, Any]:
        return {"type": "Lag", "params": {"n": self._n, "length": self._length}}

    @property
    def engine(self) -> str:
        """``"rust"`` when the Rust engine runs this schedule, else ``"python"``."""
        return engine_name(self._rust)

    @property
    def n(self) -> int:
        """Number of previous units the current unit must differ from."""
        return self._n

    @property
    def length(self) -> int:
        """Number of responses in one unit."""
        return self._length

    def step(self, now: float, event: ResponseEvent | None = None) -> Outcome:
        if self._rust is not None:
            return self._rust.step(now, event)  # type: ignore[no-any-return]
        if self._last_now is not None and now < self._last_now - _TIME_TOL:
            raise ScheduleStateError(
                f"Non-monotonic time: new now={now!r} precedes previous now={self._last_now!r}"
            )
        if event is not None and abs(event.time - now) > _TIME_TOL:
            raise ScheduleStateError(
                f"event.time ({event.time!r}) must equal now ({now!r}) within {_TIME_TOL}"
            )
        self._last_now = now
        if event is None:
            return Outcome()
        self._current.append(event.operandum)
        if len(self._current) < self._length:
            return Outcome()
        unit = tuple(self._current)
        self._current = []
        met = unit not in self._history
        if self._n > 0:
            self._history.append(unit)
        if met:
            return Outcome(reinforced=True, reinforcer=Reinforcer(time=now))
        return Outcome()

    def reset(self) -> None:
        """Clear the current unit, the comparison list and the clock."""
        if self._rust is not None:
            self._rust.reset()
            return
        self._current = []
        self._history.clear()
        self._last_now = None
