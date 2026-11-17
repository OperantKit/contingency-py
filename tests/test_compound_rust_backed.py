"""The compound schedule classes run only on the Rust engine.

They have no Python implementation of their own: their trajectories are
checked against traces recorded from the Python implementation they used
to have (``fixtures/compound_traces/compound.json``; when it was
recorded, the Python implementation and the Rust engine gave the same
traces). A component that cannot run on the Rust engine, one already
stepped, and one object used twice are rejected at construction.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import contingency.engine as engine
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


_RECORDED: dict[str, Any] = json.loads(
    (Path(__file__).parent / "fixtures" / "compound_traces" / "compound.json").read_text(
        encoding="utf-8"
    )
)


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
                dict(o.meta),
                tuple(getattr(schedule, a) for a in attrs),
            )
        )
    return out


def _as_recorded(trace: list[Any]) -> list[Any]:
    """``trace`` in the JSON shape of the recorded traces."""
    return json.loads(
        json.dumps(
            [
                [reinforced, None if r is None else list(r), meta, list(state)]
                for reinforced, r, meta, state in trace
            ]
        )
    )


@pytest.mark.parametrize(("name", "factory", "operanda", "attrs"), CASES, ids=IDS)
def test_compound_runs_on_rust(
    name: str, factory: Factory, operanda: tuple[str, ...], attrs: tuple[str, ...]
) -> None:
    assert factory().engine == "rust", name


def test_every_recorded_trace_has_a_case() -> None:
    assert sorted(_RECORDED) == sorted(IDS)


@pytest.mark.parametrize(("name", "factory", "operanda", "attrs"), CASES, ids=IDS)
def test_matches_the_recorded_trace(
    name: str, factory: Factory, operanda: tuple[str, ...], attrs: tuple[str, ...]
) -> None:
    trace = _trace(factory(), operanda, attrs)
    assert any(step[0] for step in trace), name
    assert _as_recorded(trace) == _RECORDED[name], name


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


class _Custom:
    """A Schedule written in Python."""

    def step(self, now: float, event: ResponseEvent | None = None) -> Any:
        return FR(1).step(now, event)

    def reset(self) -> None:
        pass


@pytest.mark.parametrize(
    "build",
    [
        lambda c: Alternative(c, FR(2)),
        lambda c: Multiple([FR(2), c]),
        lambda c: Timeout(c, 1.0),
        lambda c: Concurrent({"a": FR(1), "b": FR(2)}, punish={"a": c}),
        lambda c: Concurrent({"a": FR(1), "b": FR(2)}, punish_changeover=c),
        lambda c: Multiple([FR(2), Tandem([FR(1), c])]),
    ],
    ids=["Alternative", "Multiple", "Timeout", "Concurrent-punish", "Concurrent-changeover",
         "nested"],
)
def test_a_component_written_in_python_is_rejected(build: Callable[[Any], Any]) -> None:
    with pytest.raises(ScheduleConfigError, match="run on the Rust engine"):
        build(_Custom())


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


def test_a_component_already_stepped_is_rejected() -> None:
    # The spec of a component describes it as built; a component stepped
    # before it is wrapped has moved on from there.
    fr = FR(3)
    fr.step(1.0, ResponseEvent(time=1.0))
    fr.step(2.0, ResponseEvent(time=2.0))
    with pytest.raises(ScheduleConfigError, match="reset"):
        Alternative(fr, EXT())


def test_a_component_reset_after_stepping_is_accepted() -> None:
    fr = FR(3)
    fr.step(1.0, ResponseEvent(time=1.0))
    fr.reset()
    assert Alternative(fr, EXT()).engine == "rust"


@pytest.mark.parametrize(
    "build",
    [
        lambda x, y: Tandem([x, x]),
        lambda x, y: Alternative(x, x),
        lambda x, y: Concurrent({"a": x, "b": y}, punish={"a": y}),
        lambda x, y: Concurrent({"a": x, "b": y}, punish_changeover=x),
    ],
    ids=["Tandem", "Alternative", "Concurrent-punish", "Concurrent-changeover"],
)
def test_the_same_object_twice_is_rejected(build: Callable[[Any, Any], Any]) -> None:
    with pytest.raises(ScheduleConfigError, match="separate object"):
        build(FR(2), FR(3))


def test_stepping_a_component_after_wrapping_does_not_affect_the_compound() -> None:
    # The engine runs its own copy of each component.
    fr = FR(2)
    alt = Alternative(fr, EXT())
    fr.step(1.0, ResponseEvent(time=1.0))
    assert not alt.step(1.0, ResponseEvent(time=1.0)).reinforced
    assert alt.step(2.0, ResponseEvent(time=2.0)).reinforced
