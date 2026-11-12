"""Lag schedule: the Python class, its Rust engine, and the DSL bridge.

A unit is ``length`` consecutive responses read as their operanda; units
do not overlap. At the end of a unit the schedule reinforces iff the unit
differs from each of the previous ``n`` units (fewer while fewer have
been emitted). Unreinforced units stay in the comparison list.
"""

from __future__ import annotations

from typing import Any

import pytest
from contingency_dsl import parse

import contingency.engine as engine
from contingency import ResponseEvent
from contingency.bridge import from_dsl, to_spec
from contingency.engine import RustSchedule, from_spec, rust_engine_available
from contingency.errors import ScheduleConfigError, ScheduleStateError
from contingency.schedules import Lag

requires_rust = pytest.mark.skipif(
    not rust_engine_available(), reason="contingency_core is not installed"
)


def _run(schedule: Any, ops: list[str]) -> list[bool]:
    out = []
    for i, op in enumerate(ops):
        now = float(i + 1)
        out.append(schedule.step(now, ResponseEvent(now, op)).reinforced)
    return out


def _python_lag(monkeypatch: pytest.MonkeyPatch, n: int, length: int = 1) -> Lag:
    with monkeypatch.context() as m:
        m.setattr(engine, "rust_engine_available", lambda: False)
        lag = Lag(n, length=length)
    assert lag.engine == "python"
    return lag


_CASES = [
    (1, 1, ["L", "R", "R", "L"], [True, True, False, True]),
    (2, 1, ["L", "L", "R", "L"], [True, False, True, False]),
    (
        2,
        2,
        ["L", "L", "L", "R", "L", "L", "R", "R", "L", "L"],
        [False, True, False, True, False, False, False, True, False, False],
    ),
    (0, 1, ["L", "L", "L"], [True, True, True]),
    (0, 2, ["L", "L", "L", "L"], [False, True, False, True]),
    (1, 1, ["main"] * 4, [True, False, False, False]),
]


@pytest.mark.unit
@pytest.mark.parametrize(("n", "length", "ops", "expected"), _CASES)
def test_python_implementation(
    monkeypatch: pytest.MonkeyPatch, n: int, length: int, ops: list[str], expected: list[bool]
) -> None:
    assert _run(_python_lag(monkeypatch, n, length), ops) == expected


@requires_rust
@pytest.mark.unit
@pytest.mark.parametrize(("n", "length", "ops", "expected"), _CASES)
def test_rust_engine(n: int, length: int, ops: list[str], expected: list[bool]) -> None:
    lag = Lag(n, length=length)
    assert lag.engine == "rust"
    assert _run(lag, ops) == expected
    assert _run(from_spec({"type": "Lag", "params": {"n": n, "length": length}}), ops) == expected


@pytest.mark.unit
def test_python_reset_ticks_and_time(monkeypatch: pytest.MonkeyPatch) -> None:
    lag = _python_lag(monkeypatch, 1)
    assert not lag.step(0.5).reinforced
    assert lag.step(1.0, ResponseEvent(1.0, "L")).reinforced
    assert not lag.step(2.0, ResponseEvent(2.0, "L")).reinforced
    lag.reset()
    assert lag.step(0.1, ResponseEvent(0.1, "L")).reinforced
    with pytest.raises(ScheduleStateError):
        lag.step(0.05, ResponseEvent(0.05, "L"))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("args", "kwargs"),
    [((-1,), {}), ((1.0,), {}), ((True,), {}), ((1,), {"length": 0}), ((1,), {"length": 2.0})],
)
def test_constructor_rejects(args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
    with pytest.raises(ScheduleConfigError):
        Lag(*args, **kwargs)


@pytest.mark.unit
def test_attributes() -> None:
    lag = Lag(5, length=8)
    assert (lag.n, lag.length) == (5, 8)


# --- DSL bridge --------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("source", "spec"),
    [
        ("Lag 1", {"type": "Lag", "params": {"n": 1, "length": 1}}),
        ("Lag5", {"type": "Lag", "params": {"n": 5, "length": 1}}),
        ("Lag(50, length=8)", {"type": "Lag", "params": {"n": 50, "length": 8}}),
        ("Lag 0", {"type": "Lag", "params": {"n": 0, "length": 1}}),
    ],
)
def test_from_dsl_and_to_spec(source: str, spec: dict[str, Any]) -> None:
    program = parse(source)
    lag = from_dsl(program)
    assert isinstance(lag, Lag)
    assert (lag.n, lag.length) == (spec["params"]["n"], spec["params"]["length"])
    assert to_spec(program) == spec


@pytest.mark.unit
def test_to_spec_inside_compound() -> None:
    spec = to_spec(parse("Mult(FR5, Lag5)"))
    assert spec["params"]["components"][1] == {"type": "Lag", "params": {"n": 5, "length": 1}}


@requires_rust
@pytest.mark.unit
def test_from_dsl_matches_rust_from_spec() -> None:
    program = parse("Lag(2, length=2)")
    ops = ["L", "L", "L", "R", "L", "L", "R", "R", "L", "L"]
    rs = from_dsl(program, engine="rust")
    assert isinstance(rs, RustSchedule)
    assert _run(from_dsl(program), ops) == _run(rs, ops) == _CASES[2][3]


@pytest.mark.unit
@pytest.mark.parametrize(("n", "length"), [(2**32, 1), (1, 2**32), (2**64, 1)])
def test_constructor_rejects_values_over_the_bound(n: int, length: int) -> None:
    with pytest.raises(ScheduleConfigError):
        Lag(n, length=length)


@pytest.mark.unit
@pytest.mark.parametrize("value", [True, "3", 1.5, -1.0, 1e300, float(2**32)])
def test_bridge_rejects_hand_built_values(value: Any) -> None:
    import dataclasses

    node = parse("Lag 1").schedule
    bad = dataclasses.replace(node, value=value)
    with pytest.raises(ScheduleConfigError):
        from_dsl(bad)
    with pytest.raises(ScheduleConfigError):
        to_spec(bad)


@pytest.mark.unit
def test_bridge_rejects_bad_length() -> None:
    import dataclasses

    node = parse("Lag 1").schedule
    for length in (0, True, 2.0, 2**32):
        bad = dataclasses.replace(node, length=length)
        with pytest.raises(ScheduleConfigError):
            from_dsl(bad)
