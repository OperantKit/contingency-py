"""The aversive, adjusting, interlocking, percentile and trial-based
schedule classes run on the Rust engine when it is installed.

Each schedule is built twice from the same arguments: once with the Rust
engine and once with the extension hidden (the Python implementation).
Both must give the same outcomes, the same meta and the same public
state on the same trajectory.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import pytest

import contingency.engine as engine
from contingency import ResponseEvent
from contingency.engine import rust_engine_available
from contingency.schedules import FR, VR
from contingency.schedules.adjusting import AdjustingSchedule
from contingency.schedules.aversive import DiscriminatedAvoidance, Escape, Sidman
from contingency.schedules.interlocking import InterlockingSchedule
from contingency.schedules.percentile import Percentile
from contingency.schedules.trial_based import GoNoGo, MatchingToSample, TrialType

pytestmark = pytest.mark.skipif(
    not rust_engine_available(), reason="contingency_core is not installed"
)

Factory = Callable[[], Any]

_TYPES = [
    TrialType(sample="A1", correct="B1", comparisons=("B1", "B2", "B3")),
    TrialType(sample="A2", correct="B2", comparisons=("B1", "B2", "B3")),
]
_PROBES = [TrialType(sample="B1", correct="A1", comparisons=("A1", "A2", "A3"))]
_CHOICES = ("choice_0", "choice_1", "choice_2")

# (name, factory, operanda the trajectory responds on, public state attributes)
CASES: list[tuple[str, Factory, tuple[str, ...], tuple[str, ...]]] = [
    ("Sidman", lambda: Sidman(2.0, 0.3), ("main",), ("next_shock_time",)),
    (
        "DiscriminatedAvoidance",
        lambda: DiscriminatedAvoidance(0.2, 1.0, shock_magnitude=2.0),
        ("main",),
        ("phase",),
    ),
    ("Escape", lambda: Escape(2.0, 3.0), ("main",), ("phase",)),
    (
        "Adjusting-ratio",
        lambda: AdjustingSchedule("ratio", 2, 1, minimum=1, maximum=6),
        ("main",),
        ("current_value",),
    ),
    (
        # Passes through 2.5 and 3.5: the requirement rounds half to even.
        "Adjusting-ratio-halves",
        lambda: AdjustingSchedule("ratio", 2.0, 0.5),
        ("main",),
        ("current_value",),
    ),
    (
        "Adjusting-interval",
        lambda: AdjustingSchedule("interval", 1.0, 0.5),
        ("main",),
        ("current_value",),
    ),
    ("Interlocking", lambda: InterlockingSchedule(6, 10.0), ("main",), ("response_count",)),
    ("Percentile-above", lambda: Percentile("irt", 50, 4), ("main",), ("samples",)),
    ("Percentile-IRT-spelling", lambda: Percentile("IRT", 75, 3), ("main",), ("samples",)),
    (
        "Percentile-below",
        lambda: Percentile("irt", 25, 3, direction="below"),
        ("main",),
        ("samples",),
    ),
    (
        "MTS",
        lambda: MatchingToSample(3, sample_duration=0.5, choice_timeout=2.0, seed=7),
        _CHOICES,
        ("trial_index", "phase", "correct_operandum"),
    ),
    (
        "MTS-trial-types-consequences",
        lambda: MatchingToSample(
            3,
            choice_timeout=1.5,
            consequence=VR(2, seed=3),
            incorrect=FR(1),
            seed=11,
            trial_types=_TYPES,
            probe_trial_types=_PROBES,
        ),
        _CHOICES,
        (
            "trial_index",
            "phase",
            "correct_operandum",
            "current_sample",
            "current_comparisons",
            "current_trial_type",
            "is_probe_trial",
        ),
    ),
    (
        "GoNoGo",
        lambda: GoNoGo(0.6, response_window=1.0, iti=0.5, seed=4),
        ("main",),
        ("trial_index", "phase", "current_is_go"),
    ),
    (
        "GoNoGo-schedules",
        lambda: GoNoGo(
            0.5,
            response_window=1.5,
            iti=0.5,
            correct_go_schedule=FR(2),
            correct_nogo_schedule=FR(1),
            false_alarm=FR(1),
            seed=9,
        ),
        ("main",),
        ("trial_index", "phase", "current_is_go"),
    ),
]

IDS = [c[0] for c in CASES]


def _python(factory: Factory, monkeypatch: pytest.MonkeyPatch) -> Any:
    with monkeypatch.context() as m:
        m.setattr(engine, "rust_engine_available", lambda: False)
        return factory()


def _trace(schedule: Any, operanda: tuple[str, ...], attrs: tuple[str, ...]) -> list[Any]:
    out: list[Any] = []
    for i in range(1, 201):
        now = i * 0.25
        event = None
        if i % 3 != 0 and i % 7 != 0:
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


@pytest.mark.parametrize(("name", "factory", "operanda", "attrs"), CASES, ids=IDS)
def test_schedule_runs_on_rust(
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
    rust_trace = _trace(factory(), operanda, attrs)
    assert any(step[0] for step in rust_trace), name
    assert rust_trace == _trace(python, operanda, attrs), name


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
    for i in range(201, 221):
        now = i * 0.25
        event = ResponseEvent(time=now, operandum=operanda[0])
        assert schedule.step(now, event) == clone.step(now, event), name
        assert tuple(getattr(schedule, a) for a in attrs) == tuple(
            getattr(clone, a) for a in attrs
        ), name


def test_interlocking_current_requirement_reads_the_rust_anchor() -> None:
    s = InterlockingSchedule(10, 10.0)
    assert s.engine == "rust"
    assert s.current_requirement(3.0) == 10  # not anchored yet
    s.step(2.0, ResponseEvent(time=2.0))
    assert s.current_requirement(7.0) == 5


def test_non_irt_percentile_stays_in_python() -> None:
    assert Percentile("latency", 50, 3).engine == "python"


def test_a_python_consequence_keeps_the_trial_schedule_in_python() -> None:
    class Custom:
        def step(self, now: float, event: ResponseEvent | None = None) -> Any:
            return FR(1).step(now, event)

        def reset(self) -> None:
            pass

    assert MatchingToSample(3, consequence=Custom(), seed=1).engine == "python"  # type: ignore[arg-type]
    assert GoNoGo(false_alarm=Custom(), seed=1).engine == "python"  # type: ignore[arg-type]


def test_without_the_extension_they_run_in_python(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine, "rust_engine_available", lambda: False)
    assert Sidman(2.0, 5.0).engine == "python"
    assert MatchingToSample(3, seed=1).engine == "python"


def test_an_old_extension_without_state_keeps_them_in_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(engine.RustSchedule, "has_state", lambda self: False)
    assert Sidman(2.0, 5.0).engine == "python"
    assert GoNoGo(seed=1).engine == "python"


def test_reset_without_a_seed_draws_a_new_sequence() -> None:
    # As in Python: a trial schedule without a seed reseeds on reset.
    schedule = MatchingToSample(3, choice_timeout=1.0, iti=0.25)
    first = _trace(schedule, _CHOICES, ("correct_operandum",))
    schedule.reset()
    second = _trace(schedule, _CHOICES, ("correct_operandum",))
    assert first != second


def test_an_extension_whose_state_lacks_the_keys_keeps_them_in_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A contingency_core built before these schedule types reported
    # state has state() but returns an empty dict for them.
    monkeypatch.setattr(engine.RustSchedule, "state", lambda self: {})
    for schedule in (
        Sidman(2.0, 5.0),
        Escape(2.0, 3.0),
        AdjustingSchedule("ratio", 2, 1),
        InterlockingSchedule(5, 10.0),
        Percentile("irt", 50, 3),
        MatchingToSample(3, seed=1),
        GoNoGo(seed=1),
    ):
        assert schedule.engine == "python", type(schedule).__name__
        assert schedule.step(1.0, ResponseEvent(time=1.0)) is not None


def test_mts_without_a_sample_phase_starts_in_choice(monkeypatch: pytest.MonkeyPatch) -> None:
    # As after reset, and as on the Rust engine.
    python = _python(lambda: MatchingToSample(3, seed=1), monkeypatch)
    assert python.phase == "choice"
    python.reset()
    assert python.phase == "choice"
    assert MatchingToSample(3, seed=1).phase == "choice"
    assert _python(lambda: MatchingToSample(3, sample_duration=1.0, seed=1), monkeypatch).phase == (
        "sample"
    )
