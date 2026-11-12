"""The compound schedule classes run on the Rust engine when it is installed.

Each compound is built twice from the same arguments: once with the Rust
engine and once with the extension hidden (the Python implementation).
Both must give the same outcomes and the same public state on the same
trajectory.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import pytest

import contingency.engine as engine
import contingency.schedules._rust as _rust
from contingency import ResponseEvent
from contingency.engine import rust_engine_available
from contingency.errors import ScheduleConfigError
from contingency.schedules import (
    EXT,
    FI,
    FR,
    FT,
    VI,
    Alternative,
    Chained,
    Concurrent,
    Conjunctive,
    Interpolate,
    Mixed,
    Multiple,
    Overlay,
    ResponseCost,
    SecondOrder,
    Tandem,
    Timeout,
)

pytestmark = pytest.mark.skipif(
    not rust_engine_available(), reason="contingency_core is not installed"
)

Factory = Callable[[], Any]

# (name, factory, operanda the trajectory responds on, public state attributes)
CASES: list[tuple[str, Factory, tuple[str, ...], tuple[str, ...]]] = [
    (
        "Multiple",
        lambda: Multiple([FR(2), FI(1.5)], stimuli=["red", "green"]),
        ("main",),
        ("active_index", "current_stimulus"),
    ),
    (
        "Chained",
        lambda: Chained([FR(2), FR(3)]),
        ("main",),
        ("active_index", "current_stimulus", "is_terminal"),
    ),
    ("Tandem", lambda: Tandem([FR(2), FI(1.0)]), ("main",), ("active_index", "is_terminal")),
    ("Mixed", lambda: Mixed([FR(2), FR(3)]), ("main",), ("active_index",)),
    (
        "Concurrent-COD",
        lambda: Concurrent({"left": VI(3.0, seed=5), "right": FR(4)}, cod=0.5),
        ("left", "left", "right"),
        (),
    ),
    (
        "Concurrent-FRCO-directional-punish",
        lambda: Concurrent(
            {"left": FR(2), "right": FI(1.0)},
            cod=0.5,
            frco=2,
            cod_directional={("left", "right"): 1.5},
            punish={"left": FR(3)},
        ),
        ("left", "right", "right", "left"),
        (),
    ),
    ("Alternative", lambda: Alternative(FR(5), FI(2.0)), ("main",), ()),
    ("Conjunctive", lambda: Conjunctive(FR(3), FI(2.0)), ("main",), ()),
    ("Overlay", lambda: Overlay(FR(2), FR(3)), ("main",), ()),
    (
        "Interpolate",
        lambda: Interpolate(FR(2), EXT(), interval=5.0, probe_duration=2.0),
        ("main",),
        (),
    ),
    (
        "Interpolate-first-probe",
        lambda: Interpolate(FR(2), FT(0.5), interval=4.0, probe_duration=1.0, first_probe_at=1.0),
        ("main",),
        (),
    ),
    ("SecondOrder", lambda: SecondOrder(FR(3), FR(2)), ("main",), ("unit_completions",)),
    ("Timeout", lambda: Timeout(FR(3), 2.0), ("main",), ("in_timeout",)),
    (
        "Timeout-reset-on-response",
        lambda: Timeout(FR(3), 2.0, reset_on_response=True),
        ("main",),
        ("in_timeout",),
    ),
    (
        "ResponseCost-limited",
        lambda: ResponseCost(FR(5), 1.0, initial_balance=10.0),
        ("main",),
        ("balance",),
    ),
    ("ResponseCost-unlimited", lambda: ResponseCost(FR(5), 1.0), ("main",), ("balance",)),
    (
        "nested",
        lambda: Multiple([Tandem([FR(1), FR(2)]), Timeout(FR(2), 1.0)]),
        ("main",),
        ("active_index",),
    ),
]

IDS = [c[0] for c in CASES]


def _python(factory: Factory, monkeypatch: pytest.MonkeyPatch) -> Any:
    with monkeypatch.context() as m:
        m.setattr(_rust, "_FORCE_PYTHON", True)
        return factory()


def _trace(schedule: Any, operanda: tuple[str, ...], attrs: tuple[str, ...]) -> list[Any]:
    out: list[Any] = []
    for i in range(1, 161):
        now = i * 0.25
        event = None
        if i % 3 != 0:
            event = ResponseEvent(time=now, operandum=operanda[(i // 3) % len(operanda)])
        o = schedule.step(now, event)
        r = o.reinforcer
        out.append(
            (
                o.reinforced,
                None if r is None else (r.time, round(r.magnitude, 9), r.label),
                tuple(getattr(schedule, a) for a in attrs),
            )
        )
    return out


@pytest.mark.parametrize(("name", "factory", "operanda", "attrs"), CASES, ids=IDS)
def test_compound_runs_on_rust(
    name: str, factory: Factory, operanda: tuple[str, ...], attrs: tuple[str, ...]
) -> None:
    assert factory().engine == "rust", name


@pytest.mark.parametrize(("name", "factory", "operanda", "attrs"), CASES, ids=IDS)
def test_rust_and_python_agree(
    name: str,
    factory: Factory,
    operanda: tuple[str, ...],
    attrs: tuple[str, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    python = _python(factory, monkeypatch)
    assert python.engine == "python"
    assert _trace(factory(), operanda, attrs) == _trace(python, operanda, attrs), name


@pytest.mark.parametrize(("name", "factory", "operanda", "attrs"), CASES, ids=IDS)
def test_reset_replays_the_same_trace(
    name: str, factory: Factory, operanda: tuple[str, ...], attrs: tuple[str, ...]
) -> None:
    schedule = factory()
    first = _trace(schedule, operanda, attrs)
    schedule.reset()
    assert _trace(schedule, operanda, attrs) == first, name


@pytest.mark.parametrize(("name", "factory", "operanda", "attrs"), CASES, ids=IDS)
def test_deepcopy_continues_from_the_same_point(
    name: str, factory: Factory, operanda: tuple[str, ...], attrs: tuple[str, ...]
) -> None:
    schedule = factory()
    _trace(schedule, operanda, attrs)
    clone = copy.deepcopy(schedule)
    for i in range(161, 181):
        now = i * 0.25
        event = ResponseEvent(time=now, operandum=operanda[0])
        assert schedule.step(now, event) == clone.step(now, event), name


def test_a_python_component_keeps_the_compound_in_python() -> None:
    class Custom:
        def step(self, now: float, event: ResponseEvent | None = None) -> Any:
            return FR(1).step(now, event)

        def reset(self) -> None:
            pass

    assert Alternative(Custom(), FR(2)).engine == "python"  # type: ignore[arg-type]
    assert Multiple([FR(2), Custom()]).engine == "python"  # type: ignore[list-item]


def test_without_the_extension_compounds_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    fr2, fr3 = FR(2), FR(3)
    monkeypatch.setattr(engine, "rust_engine_available", lambda: False)
    with pytest.raises(ScheduleConfigError, match="contingency_core"):
        Multiple([fr2, fr3])
    with pytest.raises(ScheduleConfigError, match="contingency_core"):
        Timeout(fr2, 1.0)


def test_an_old_extension_without_state_raises_for_stateful_compounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fr2, fr3 = FR(2), FR(3)
    monkeypatch.setattr(engine.RustSchedule, "has_state", lambda self: False)
    with pytest.raises(ScheduleConfigError, match="too old"):
        Multiple([fr2, fr3])
    with pytest.raises(ScheduleConfigError, match="too old"):
        SecondOrder(fr3, fr2)
    # Compounds without a position do not read state().
    assert Alternative(fr2, fr3).engine == "rust"


def test_reset_of_a_seedless_component_draws_a_new_sequence() -> None:
    # As in Python: resetting the compound resets the components, and a
    # VI without a seed draws a new sequence of intervals.
    schedule = Mixed([VI(2.0), VI(2.0)])
    first = _trace(schedule, ("main",), ())
    schedule.reset()
    second = _trace(schedule, ("main",), ())
    assert first != second


def test_a_component_already_stepped_keeps_the_compound_in_python() -> None:
    # The spec of a component describes it as built; a component stepped
    # before it is wrapped has moved on, so the compound runs in Python.
    fr = FR(3)
    fr.step(1.0, ResponseEvent(time=1.0))
    fr.step(2.0, ResponseEvent(time=2.0))
    alt = Alternative(fr, EXT())
    assert alt.engine == "python"
    assert alt.step(3.0, ResponseEvent(time=3.0)).reinforced


def test_a_component_reset_after_stepping_runs_on_rust_again() -> None:
    fr = FR(3)
    fr.step(1.0, ResponseEvent(time=1.0))
    fr.reset()
    assert Alternative(fr, EXT()).engine == "rust"


def test_the_same_object_twice_in_one_compound_keeps_it_in_python() -> None:
    x = FR(2)
    assert Tandem([x, x]).engine == "python"
    assert Alternative(x, x).engine == "python"
    y = FR(3)
    assert Concurrent({"a": x, "b": y}, punish={"a": y}).engine == "python"
