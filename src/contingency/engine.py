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
* ``copy.deepcopy`` of a :class:`RustSchedule` builds a new schedule
  from the same spec and replays the steps since the last
  :meth:`~RustSchedule.reset`, so it costs time and memory in proportion
  to those steps.
* A compound schedule class run on the Rust engine (``Multiple``,
  ``Concurrent``, ``Timeout`` and the rest) steps Rust copies of its
  components built from their specs, not the component objects. Stepping
  a component object separately, or sharing one component object
  between two compounds, does not affect the compound, and attributes
  that return the component objects (``components``, ``first``,
  ``base``, ``unit`` and the like) show them as built.
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

    __slots__ = ("_inner", "_core", "_spec", "_history")

    def __init__(self, inner: Any, core: Any, spec: dict[str, Any] | str) -> None:
        self._inner = inner
        self._core = core
        self._spec = spec
        self._history: list[tuple[float, ResponseEvent | None]] = []

    def __deepcopy__(self, memo: dict[int, Any]) -> RustSchedule:
        copy = from_spec(self._spec)
        for now, event in self._history:
            copy.step(now, event)
        memo[id(self)] = copy
        return copy

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
        self._history.append((now, event))
        r = raw.reinforcer
        reinforcer = (
            None if r is None else Reinforcer(time=r.time, magnitude=r.magnitude, label=r.label)
        )
        return Outcome(
            reinforced=bool(raw.reinforced),
            reinforcer=reinforcer,
            meta=dict(raw.meta),
        )

    def probe_response(self, now: float, event: ResponseEvent) -> bool | None:
        """Whether a response ``event`` at ``now`` would reinforce, without stepping.

        ``None`` when the schedule offers no such probe (only FR, VR, RR,
        FI, VI and RI do), or when the installed ``contingency_core`` is
        too old to have it.
        """
        probe = getattr(self._inner, "probe_response", None)
        if probe is None:
            return None
        core_event = self._core.ResponseEvent(float(event.time), str(event.operandum))
        try:
            probed = probe(float(now), core_event)
        except (ValueError, RuntimeError) as exc:
            raise _translate(exc) from exc
        return None if probed is None else bool(probed)

    def state(self) -> dict[str, Any]:
        """The observable state after the last step (``Schedule::state``).

        Empty for schedules without one. Compound schedules with a
        position report ``active_index``, ``unit_completions``,
        ``in_timeout`` or ``balance``.
        """
        return dict(self._inner.state())

    def has_state(self) -> bool:
        """Whether the installed ``contingency_core`` offers :meth:`state`."""
        return hasattr(self._inner, "state")

    def reset(self) -> None:
        try:
            self._inner.reset()
        except (ValueError, RuntimeError) as exc:
            raise _translate(exc) from exc
        self._history.clear()

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
    return RustSchedule(inner, core, spec)
