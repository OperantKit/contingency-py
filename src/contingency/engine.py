"""Rust engine adapter — run a schedule spec on ``contingency_core``.

``contingency_core`` is the PyO3 build of contingency-rs (the ``core/``
directory of this repository builds it with maturin). This module
wraps one of its schedules behind the :class:`~contingency.interfaces.Schedule`
protocol so that callers keep using this package's value types
(:class:`~contingency.entities.ResponseEvent`,
:class:`~contingency.entities.Outcome`,
:class:`~contingency.entities.Reinforcer`) and error classes.

Entry points
------------
* :func:`from_spec` — build a :class:`RustSchedule` from a schedule
  spec ``{"type": ..., "params": ...}`` (a dict or a JSON string; the
  same form :func:`contingency.to_spec` produces).
* ``from_dsl(ast, engine="rust")`` — ``to_spec`` followed by
  :func:`from_spec`.

Known differences from the Python engine
----------------------------------------
* ``Outcome.meta`` omits keys whose value would be ``None`` (the Rust
  meta value type has no null).
* Errors raised by the Rust engine carry the Rust message text.
"""

from __future__ import annotations

from typing import Any

from contingency.entities import Outcome, Reinforcer, ResponseEvent
from contingency.errors import ScheduleConfigError, ScheduleStateError

__all__ = ["RustSchedule", "from_spec", "rust_engine_available"]

_INSTALL_HINT = (
    "the Rust engine needs the contingency_core extension module; build "
    "and install it from the core/ directory of contingency-py with "
    "maturin (see the Development section of the README)"
)


def _core() -> Any:
    try:
        import contingency_core  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(_INSTALL_HINT) from exc
    return contingency_core


def rust_engine_available() -> bool:
    """Return ``True`` when ``contingency_core`` can be imported."""
    try:
        import contingency_core  # noqa: F401
    except ImportError:
        return False
    return True


def _translate(exc: Exception) -> Exception:
    """Map a ``contingency_core`` error onto this package's error classes."""
    if isinstance(exc, ValueError):
        return ScheduleConfigError(str(exc))
    if isinstance(exc, RuntimeError):
        return ScheduleStateError(str(exc))
    return exc


class RustSchedule:
    """A :class:`~contingency.interfaces.Schedule` run by contingency-rs.

    Build instances with :func:`from_spec` (or
    ``from_dsl(..., engine="rust")``) rather than directly.
    """

    __slots__ = ("_inner", "_core")

    def __init__(self, inner: Any, core: Any) -> None:
        self._inner = inner
        self._core = core

    def step(self, now: float, event: ResponseEvent | None = None) -> Outcome:
        core_event = (
            None
            if event is None
            else self._core.ResponseEvent(float(event.time), str(event.operandum))
        )
        try:
            raw = self._inner.step(float(now), core_event)
        except (ValueError, RuntimeError) as exc:
            raise _translate(exc) from exc
        r = raw.reinforcer
        reinforcer = (
            None if r is None else Reinforcer(time=r.time, magnitude=r.magnitude, label=r.label)
        )
        return Outcome(
            reinforced=bool(raw.reinforced),
            reinforcer=reinforcer,
            meta=dict(raw.meta),
        )

    def reset(self) -> None:
        try:
            self._inner.reset()
        except (ValueError, RuntimeError) as exc:
            raise _translate(exc) from exc

    def __repr__(self) -> str:
        return f"RustSchedule({self._inner!r})"


def from_spec(spec: dict[str, Any] | str) -> RustSchedule:
    """Build a schedule from a spec, run by the Rust engine.

    Parameters
    ----------
    spec:
        ``{"type": ..., "params": ...}`` as a dict or a JSON string.

    Raises
    ------
    ScheduleConfigError
        When the spec is malformed, names an unknown type, or has
        invalid parameters.
    ImportError
        When ``contingency_core`` is not installed.
    """
    core = _core()
    try:
        inner = core.from_spec(spec)
    except (ValueError, RuntimeError) as exc:
        raise _translate(exc) from exc
    return RustSchedule(inner, core)
