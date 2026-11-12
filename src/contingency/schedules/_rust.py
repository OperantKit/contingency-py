"""Run a schedule class on the Rust engine (``contingency_core``).

The schedule classes of this package validate their arguments and keep
their public attributes in Python. Their ``step`` / ``reset`` are
delegated to a :class:`contingency.engine.RustSchedule` built from the
spec of the instance. Building a schedule class needs the
``contingency_core`` extension module, at a version whose schedules
report ``state()``; without it the constructor raises
:class:`~contingency.errors.ScheduleConfigError` that says how to
install it. ``import contingency`` and the DSL parsing do not need it.

A few arguments still run in this package's Python implementation
(a caller-supplied ``rng``, a custom ProgressiveRatio step function,
compounds over components that run in Python). ``engine`` on each
instance tells which one runs (``"rust"`` or ``"python"``).
"""

from __future__ import annotations

import random
from typing import Any

from contingency.errors import ScheduleConfigError

__all__ = [
    "core_too_old",
    "require_core",
    "rust_or_none",
    "engine_name",
    "random_source",
    "fresh_seed",
    "rust_reset",
    "spec_of",
    "specs_of",
    "rust_compound",
]


# Private source of fresh seeds, so that drawing one does not advance the
# caller's module-level ``random`` sequence.
_SEEDS = random.Random()

# Tests set this to run the Python implementation that remains in the
# schedule classes, to compare it with the Rust engine. Not public API.
_FORCE_PYTHON = False

_MISSING_CORE = (
    "the schedule classes of contingency need the contingency_core extension "
    "module (the contingency-rs engine), which is not installed; build and "
    "install its wheel from the core/ directory of contingency-py with "
    "maturin (see the Development section of the README)"
)

_OLD_CORE = (
    "the installed contingency_core extension module is too old for this "
    "version of contingency ({what}); rebuild and reinstall its wheel from "
    "the core/ directory of contingency-py with maturin (see the "
    "Development section of the README)"
)

_HAS_STATE: bool | None = None


def _core_has_state() -> bool:
    """Whether the schedules of the installed ``contingency_core`` report ``state()``."""
    global _HAS_STATE
    if _HAS_STATE:
        return True
    from contingency.engine import from_spec

    # Only a positive answer is kept: the installed module does not lose
    # state() while the process runs.
    _HAS_STATE = from_spec({"type": "EXT", "params": {}}).has_state()
    return _HAS_STATE


def core_too_old(what: str) -> ScheduleConfigError:
    """The error for a ``contingency_core`` that lacks ``what``."""
    return ScheduleConfigError(_OLD_CORE.format(what=what))


def require_core() -> None:
    """Raise unless a recent enough ``contingency_core`` is installed.

    Raises
    ------
    ScheduleConfigError
        When ``contingency_core`` cannot be imported, or when its
        schedules do not report ``state()``.
    """
    if _FORCE_PYTHON:
        return
    from contingency.engine import rust_engine_available

    if not rust_engine_available():
        raise ScheduleConfigError(_MISSING_CORE)
    if not _core_has_state():
        raise core_too_old("its schedules do not report state()")


def rust_or_none(spec: dict[str, Any]) -> Any:
    """Return a ``RustSchedule`` for ``spec`` (``None`` only in tests).

    Raises
    ------
    ScheduleConfigError
        Without a recent enough ``contingency_core`` (see
        :func:`require_core`), and when the Rust engine rejects the spec
        (integers beyond its integer width, infinite times and the like).
    """
    if _FORCE_PYTHON:
        return None
    require_core()
    from contingency.engine import from_spec

    kind = spec.get("type", "?")
    try:
        return from_spec(spec)
    except (ScheduleConfigError, ValueError, TypeError, OverflowError) as exc:
        raise ScheduleConfigError(f"the Rust engine rejected the {kind} spec: {exc}") from exc


def fresh_seed() -> int:
    """A seed for the Rust engine when the caller gave none.

    The Rust schedule is built with an explicit seed so that a copy
    rebuilt from the same spec (``copy.deepcopy``) draws the same values.
    """
    return _SEEDS.getrandbits(63)


def engine_name(rust: Any) -> str:
    """Return ``"rust"`` when ``rust`` is set, else ``"python"``."""
    return "python" if rust is None else "rust"


def random_source(
    kind: str, rng: random.Random | None, seed: int | None
) -> tuple[random.Random, bool]:
    """Resolve the ``rng`` / ``seed`` arguments of RR, RI and RT.

    Returns the Python RNG of the fallback implementation and whether
    the Rust engine may run the schedule. A caller-supplied ``rng`` keeps
    the schedule in Python: its state cannot be handed to the Rust
    engine. ``seed`` draws the same sequence as ``rng=random.Random(seed)``.
    Raises :class:`ScheduleConfigError` without ``contingency_core``.
    """
    require_core()
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ScheduleConfigError(f"{kind} requires an int seed, got {type(seed).__name__}")
        if rng is not None:
            raise ScheduleConfigError(f"{kind} takes rng or seed, not both")
        return random.Random(seed), True
    if rng is not None:
        return rng, False
    return random.Random(), True


def rust_reset(rust: Any, seed: int | None) -> Any:
    """Reset a Rust-run VR / VI / VT; return the schedule to keep.

    With a constructor ``seed`` the schedule restarts its sequence. Without
    one, the Python classes draw a new sequence on every reset; the Rust
    schedule is rebuilt with a fresh seed to do the same.
    """
    if seed is not None:
        rust.reset()
        return rust
    spec = rust._spec
    return rust_or_none({"type": spec["type"], "params": {**spec["params"], "seed": fresh_seed()}})


def spec_of(schedule: Any) -> dict[str, Any] | None:
    """The spec a schedule of this package runs on Rust with, else ``None``.

    ``None`` for a schedule that runs in Python, including any object
    outside this package that implements the ``Schedule`` protocol.
    ``None`` too for a schedule already stepped since its last reset:
    its spec describes the schedule as built, not where it stands now.
    """
    rust = getattr(schedule, "_rust", None)
    if rust is None or getattr(rust, "_history", None):
        return None
    spec = getattr(rust, "_spec", None)
    return spec if isinstance(spec, dict) else None


def specs_of(schedules: Any) -> list[dict[str, Any]] | None:
    """The specs of ``schedules``, or ``None`` when any runs in Python.

    ``None`` also when the same object appears twice: the Rust engine
    would run two independent copies of it, where Python steps the one
    object twice.
    """
    schedules = list(schedules)
    if len({id(s) for s in schedules}) != len(schedules):
        return None
    specs = [spec_of(s) for s in schedules]
    if any(s is None for s in specs):
        return None
    return specs  # type: ignore[return-value]


def rust_compound(
    spec: dict[str, Any] | None,
    *,
    needs_state: bool = False,
    state_key: str | None = None,
) -> Any:
    """Build a compound schedule on Rust from its spec, or ``None``.

    ``spec`` is ``None`` when a component runs in Python: then the
    compound runs in Python too. With ``needs_state`` the compound reads
    its position (``active_index`` and the like) from
    ``Schedule.state()``. With ``state_key`` the schedule also needs
    that key in its ``state()`` right after construction. A
    ``contingency_core`` that lacks either raises
    :class:`ScheduleConfigError`, as does a missing one.
    """
    if _FORCE_PYTHON:
        return None
    require_core()
    if spec is None:
        return None
    rust = rust_or_none(spec)
    if (needs_state or state_key is not None) and not rust.has_state():
        raise core_too_old("its schedules do not report state()")
    if state_key is not None and state_key not in rust.state():
        raise core_too_old(f"the state() of {spec.get('type', '?')} has no {state_key!r}")
    return rust
