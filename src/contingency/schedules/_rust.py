"""Run a schedule class on the Rust engine when ``contingency_core`` is present.

The schedule classes of this package validate their arguments and keep
their public attributes in Python. Their ``step`` / ``reset`` are
delegated to a :class:`contingency.engine.RustSchedule` built from the
spec of the instance, when the ``contingency_core`` extension module can
be imported. Without it, the class runs its own Python implementation.
``engine`` on each instance tells which one runs (``"rust"`` or
``"python"``).
"""

from __future__ import annotations

import random
from typing import Any

from contingency.errors import ScheduleConfigError

__all__ = [
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


def rust_or_none(spec: dict[str, Any]) -> Any:
    """Return a ``RustSchedule`` for ``spec``, or ``None`` to run in Python.

    ``None`` without the extension, and for arguments the Python class
    accepts but the Rust engine cannot take (integers beyond its integer
    width, infinite times): those keep running in Python.
    """
    from contingency.engine import from_spec, rust_engine_available
    from contingency.errors import ScheduleConfigError

    if not rust_engine_available():
        return None
    try:
        return from_spec(spec)
    except (ScheduleConfigError, ValueError, TypeError, OverflowError):
        return None


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
    """
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


def rust_compound(spec: dict[str, Any] | None, *, needs_state: bool = False) -> Any:
    """Build a compound schedule on Rust from its spec, or ``None``.

    ``spec`` is ``None`` when a component runs in Python: then the
    compound runs in Python too. With ``needs_state`` the compound reads
    its position (``active_index`` and the like) from
    ``Schedule.state()``; a ``contingency_core`` too old to have it keeps
    the compound in Python.
    """
    if spec is None:
        return None
    rust = rust_or_none(spec)
    if rust is not None and needs_state and not rust.has_state():
        return None
    return rust
