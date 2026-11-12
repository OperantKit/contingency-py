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

__all__ = ["rust_or_none", "engine_name", "random_source", "fresh_seed", "rust_reset"]


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
