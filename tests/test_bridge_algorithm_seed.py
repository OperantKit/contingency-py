"""Tests for ``@algorithm(seed=...)`` reaching Fleshler-Hoffman schedules.

A program-level ``@algorithm("fleshler-hoffman", seed=S)`` annotation
seeds every VI / VR / VT schedule the bridge builds. Schedules receive
``S``, ``S + 1``, ... in left-to-right source order, so a single
``VI30s`` matches ``ScheduleBuilder.vi(30.0, seed=S)`` and the
components of a compound schedule draw different interval sequences.
"""

from __future__ import annotations

import pytest
from contingency_dsl import parse
from contingency_dsl.annotations.temporal.extension import TemporalExtension
from contingency_dsl.extensions import ExtensionRegistry

from contingency import ScheduleBuilder, from_dsl
from contingency.entities import ResponseEvent
from contingency.errors import ScheduleConfigError
from contingency.interfaces import Schedule

_REGISTRY = ExtensionRegistry.build(TemporalExtension())


def _bridge(source: str) -> Schedule:
    return from_dsl(parse(source, extensions=_REGISTRY))


def _reinforced_times(
    schedule: Schedule,
    operanda: tuple[str, ...] = ("main",),
    duration: float = 900.0,
    irt: float = 0.25,
) -> list[tuple[float, str]]:
    """Respond every ``irt`` seconds, cycling through ``operanda``."""
    times: list[tuple[float, str]] = []
    n_steps = int(duration / irt)
    for i in range(1, n_steps + 1):
        now = i * irt
        operandum = operanda[i % len(operanda)]
        outcome = schedule.step(now, ResponseEvent(time=now, operandum=operandum))
        if outcome.reinforced:
            times.append((now, operandum))
    return times


@pytest.mark.unit
def test_algorithm_seed_matches_builder_seed_for_vi() -> None:
    seeded = _bridge('@algorithm("fleshler-hoffman", n=12, seed=42)\nVI30s')
    reference = ScheduleBuilder.vi(30.0, seed=42)
    assert _reinforced_times(seeded) == _reinforced_times(reference)


@pytest.mark.unit
def test_algorithm_seed_matches_builder_seed_for_vr() -> None:
    seeded = _bridge('@algorithm("fleshler-hoffman", seed=7)\nVR10')
    reference = ScheduleBuilder.vr(10.0, seed=7)
    assert _reinforced_times(seeded) == _reinforced_times(reference)


@pytest.mark.unit
def test_algorithm_seed_makes_concurrent_reproducible() -> None:
    source = '@algorithm("fleshler-hoffman", seed=1)\nConc(VI30s, VI60s)'
    ops = ("c0", "c1")
    assert _reinforced_times(_bridge(source), ops) == _reinforced_times(_bridge(source), ops)


@pytest.mark.unit
def test_algorithm_seed_offsets_each_component() -> None:
    conc = _bridge('@algorithm("fleshler-hoffman", seed=5)\nConc(VI30s, VI30s)')
    first = ScheduleBuilder.concurrent(
        {"c0": ScheduleBuilder.vi(30.0, seed=5), "c1": ScheduleBuilder.vi(30.0, seed=6)}
    )
    ops = ("c0", "c1")
    times = _reinforced_times(conc, ops)
    assert times == _reinforced_times(first, ops)
    by_op = {op: [t for t, o in times if o == op] for op in ops}
    assert by_op["c0"] != by_op["c1"]


@pytest.mark.unit
def test_seed_scope_is_one_program() -> None:
    """A seeded bridge call does not leak its seed into a later unseeded one."""
    _bridge('@algorithm("fleshler-hoffman", seed=3)\nVI30s')
    unseeded = from_dsl(parse("VI30s"))
    assert unseeded._seed is None


# --- n, name and value validation -------------------------------------------


@pytest.mark.unit
def test_algorithm_n_sets_fleshler_hoffman_intervals() -> None:
    seeded = _bridge('@algorithm("fleshler-hoffman", n=20, seed=3)\nVI30s')
    reference = ScheduleBuilder.vi(30.0, n_intervals=20, seed=3)
    assert _reinforced_times(seeded) == _reinforced_times(reference)


@pytest.mark.unit
def test_arithmetic_algorithm_on_vi_is_rejected() -> None:
    with pytest.raises(ScheduleConfigError, match="fleshler-hoffman"):
        _bridge('@algorithm("arithmetic", n=20, seed=3)\nVI30s')


@pytest.mark.unit
@pytest.mark.parametrize(
    "annotation",
    [
        '@algorithm("foo", seed=1)',
        '@algorithm("fleshler-hoffman", seed=true)',
        '@algorithm("fleshler-hoffman", seed="7")',
        '@algorithm("fleshler-hoffman", seed=1.5)',
        '@algorithm("fleshler-hoffman", n=0)',
        '@algorithm("fleshler-hoffman", seed=1)\n@algorithm("fleshler-hoffman", seed=2)',
    ],
)
def test_invalid_algorithm_annotation_is_rejected(annotation: str) -> None:
    with pytest.raises(ScheduleConfigError, match="@algorithm"):
        _bridge(f"{annotation}\nVI30s")


# --- other stochastic schedules ---------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("expr", ["RI30s", "RR10", "Conc(RI30s, VI60s)"])
def test_algorithm_seed_makes_random_schedules_reproducible(expr: str) -> None:
    source = f'@algorithm("fleshler-hoffman", seed=4)\n{expr}'
    ops = ("c0", "c1") if expr.startswith("Conc") else ("main",)
    assert _reinforced_times(_bridge(source), ops) == _reinforced_times(_bridge(source), ops)


@pytest.mark.unit
@pytest.mark.parametrize(
    "expr",
    [
        "MTS(comparisons=3, consequence=CRF, ITI=5s)",
        "GoNoGo(responseWindow=5s, consequence=CRF, ITI=10s)",
    ],
)
def test_algorithm_seed_reaches_trial_based_schedules(expr: str) -> None:
    schedule = _bridge(f'@algorithm("fleshler-hoffman", seed=8)\n{expr}')
    assert schedule._seed == 8


# --- schedule-level annotation ----------------------------------------------


@pytest.mark.unit
def test_schedule_level_algorithm_seeds_its_component() -> None:
    source = 'Conc(VI30s @algorithm("fleshler-hoffman", seed=9), VI60s)'
    reference = ScheduleBuilder.concurrent(
        {"c0": ScheduleBuilder.vi(30.0, seed=9), "c1": ScheduleBuilder.vi(60.0)}
    )

    def c0_times(schedule: Schedule) -> list[float]:
        return [t for t, op in _reinforced_times(schedule, ("c0", "c1")) if op == "c0"]

    assert c0_times(_bridge(source)) == c0_times(reference)


@pytest.mark.unit
def test_inner_algorithm_overrides_only_given_fields() -> None:
    """An inner ``@algorithm`` without ``seed`` keeps drawing the outer seeds."""
    source = (
        '@algorithm("fleshler-hoffman", seed=20)\n'
        'Conc(VI30s, VI60s @algorithm("fleshler-hoffman", n=6))'
    )
    reference = ScheduleBuilder.concurrent(
        {
            "c0": ScheduleBuilder.vi(30.0, seed=20),
            "c1": ScheduleBuilder.vi(60.0, n_intervals=6, seed=21),
        }
    )
    ops = ("c0", "c1")
    assert _reinforced_times(_bridge(source), ops) == _reinforced_times(reference, ops)
