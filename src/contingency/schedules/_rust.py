"""Run a schedule class on the Rust engine (``contingency_core``).

The schedule classes of this package validate their arguments and keep
their public attributes in Python. Their ``step`` / ``reset`` are
delegated to a :class:`contingency.engine.RustSchedule` built from the
spec of the instance. Building a schedule class needs the
``contingency_core`` extension module, at a version whose schedules
report ``state()``; without it the constructor raises
:class:`~contingency.errors.ScheduleConfigError` that says how to
install it. ``import contingency`` and the DSL parsing do not need it.

The single schedule classes (ratio, interval, time-based,
differential, Lag, adjusting, interlocking, aversive), the compound
schedule classes and the trial-based classes run only on the Rust
engine. A compound or trial-based schedule takes only components that
run there (:func:`component_specs`). Two cases still run in this
package's Python implementation: Percentile on a non-IRT target, and a
ProgressiveRatio over a custom step function. ``engine``
on each instance tells which one runs (``"rust"`` or ``"python"``).
"""

from __future__ import annotations

import random
from typing import Any

from contingency.errors import ScheduleConfigError

__all__ = [
    "core_too_old",
    "require_core",
    "rust_or_none",
    "rust_schedule",
    "rust_state_schedule",
    "engine_name",
    "seed_of",
    "fresh_seed",
    "rust_reset",
    "rust_compound",
    "component_specs",
    "compound_schedule",
]


# Private source of fresh seeds, so that drawing one does not advance the
# caller's module-level ``random`` sequence.
_SEEDS = random.Random()

# Tests set this to run the Python implementation that remains in the
# percentile schedule class, to compare it with the Rust engine. The
# other schedule classes ignore it. Not public API.
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
    _require_core_now()


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


def rust_schedule(spec: dict[str, Any]) -> Any:
    """Return the ``RustSchedule`` for ``spec``; the schedule has no other engine.

    Unlike :func:`rust_or_none`, this ignores the test switch that runs
    the Python implementation of the percentile schedule.

    Raises
    ------
    ScheduleConfigError
        Without a recent enough ``contingency_core`` (see
        :func:`require_core`), and when the Rust engine rejects the spec.
    """
    _require_core_now()
    from contingency.engine import from_spec

    kind = spec.get("type", "?")
    try:
        return from_spec(spec)
    except (ScheduleConfigError, ValueError, TypeError, OverflowError) as exc:
        raise ScheduleConfigError(f"the Rust engine rejected the {kind} spec: {exc}") from exc


def rust_state_schedule(spec: dict[str, Any], *state_keys: str) -> Any:
    """:func:`rust_schedule` for a schedule whose public attributes read ``state()``.

    Raises :class:`ScheduleConfigError` when the installed
    ``contingency_core`` does not report ``state()`` or lacks one of
    ``state_keys`` in it right after construction.
    """
    rust = rust_schedule(spec)
    if not rust.has_state():
        raise core_too_old("its schedules do not report state()")
    state = rust.state()
    for key in state_keys:
        if key not in state:
            raise core_too_old(f"the state() of {spec.get('type', '?')} has no {key!r}")
    return rust


def seed_of(kind: str, rng: random.Random | None, seed: int | None) -> int:
    """Resolve the ``rng`` / ``seed`` arguments of RR, RI and RT to a Rust seed.

    ``seed`` is used as given. A caller-supplied ``rng`` gives a 63-bit
    seed drawn from it (one draw, which advances ``rng``). Neither gives
    a fresh seed. Raises :class:`ScheduleConfigError` without
    ``contingency_core``, for a non-int ``seed`` and for both together.
    """
    _require_core_now()
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ScheduleConfigError(f"{kind} requires an int seed, got {type(seed).__name__}")
        if rng is not None:
            raise ScheduleConfigError(f"{kind} takes rng or seed, not both")
        return seed
    if rng is not None:
        draw = getattr(rng, "getrandbits", None)
        if not callable(draw):
            raise ScheduleConfigError(
                f"{kind} requires rng to be a random.Random, got {type(rng).__name__}"
            )
        return int(draw(63))
    return fresh_seed()


def _require_core_now() -> None:
    """:func:`require_core` without the test switch."""
    from contingency.engine import rust_engine_available

    if not rust_engine_available():
        raise ScheduleConfigError(_MISSING_CORE)
    if not _core_has_state():
        raise core_too_old("its schedules do not report state()")


def fresh_seed() -> int:
    """A seed for the Rust engine when the caller gave none.

    The Rust schedule is built with an explicit seed so that a copy
    rebuilt from the same spec (``copy.deepcopy``) draws the same values.
    """
    return _SEEDS.getrandbits(63)


def engine_name(rust: Any) -> str:
    """Return ``"rust"`` when ``rust`` is set, else ``"python"``."""
    return "python" if rust is None else "rust"


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
    return rust_schedule({"type": spec["type"], "params": {**spec["params"], "seed": fresh_seed()}})


def component_specs(kind: str, schedules: Any) -> list[dict[str, Any]]:
    """The specs of the components of a compound that runs only on Rust.

    Raises
    ------
    ScheduleConfigError
        Without a recent enough ``contingency_core``; when a component
        does not run on the Rust engine (a ``Schedule`` written in
        Python, or a schedule of this package that still runs in
        Python); when a component has been stepped since its last
        reset; and when the same object appears twice.
    """
    _require_core_now()
    schedules = list(schedules)
    seen: set[int] = set()
    specs: list[dict[str, Any]] = []
    for schedule in schedules:
        if id(schedule) in seen:
            raise ScheduleConfigError(
                f"{kind} got the same {type(schedule).__name__} object twice; "
                f"build a separate object for each place it is used"
            )
        seen.add(id(schedule))
        rust = getattr(schedule, "_rust", None)
        spec = getattr(rust, "_spec", None)
        if rust is None or not isinstance(spec, dict):
            raise ScheduleConfigError(
                f"{kind} takes only components that run on the Rust engine; "
                f"{type(schedule).__name__} does not (a Schedule written in Python "
                f"cannot be a component of a compound schedule)"
            )
        if getattr(rust, "_history", None):
            raise ScheduleConfigError(
                f"{kind} got a {type(schedule).__name__} that has already been "
                f"stepped; call its reset() before wrapping it"
            )
        specs.append(spec)
    return specs


def compound_schedule(
    spec: dict[str, Any],
    *,
    needs_state: bool = False,
) -> Any:
    """The ``RustSchedule`` of a compound that has no other engine.

    With ``needs_state`` the compound reads its position (``active_index``
    and the like) from ``state()``; a ``contingency_core`` without it
    raises :class:`ScheduleConfigError`, as does a missing one.
    """
    rust = rust_schedule(spec)
    if needs_state and not rust.has_state():
        raise core_too_old("its schedules do not report state()")
    return rust


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
