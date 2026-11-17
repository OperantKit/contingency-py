"""The aversive, adjusting, interlocking, percentile and trial-based
schedule classes run on the Rust engine.

Their trajectories (outcomes, meta and public state) are checked against
traces recorded from the Python implementation they used to have, when
it agreed with the Rust engine step for step
(``fixtures/single_schedule_traces/remaining.json``).
"""

from __future__ import annotations

import copy
import dataclasses
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import contingency.engine as engine
from contingency import ResponseEvent
from contingency.engine import rust_engine_available
from contingency.errors import ScheduleConfigError
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

_RECORDED: dict[str, list[Any]] = json.loads(
    (Path(__file__).parent / "fixtures" / "single_schedule_traces" / "remaining.json").read_text(
        encoding="utf-8"
    )
)



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


def _json_default(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)  # a TrialType
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def _as_recorded(trace: list[Any]) -> list[Any]:
    """``trace`` in the JSON shape of the recorded traces."""
    return json.loads(
        json.dumps(
            [
                [reinforced, None if r is None else list(r), meta, list(state)]
                for reinforced, r, meta, state in trace
            ],
            default=_json_default,
        )
    )


def test_every_case_has_a_recorded_trace() -> None:
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


def test_a_python_consequence_is_rejected() -> None:
    class Custom:
        def step(self, now: float, event: ResponseEvent | None = None) -> Any:
            return FR(1).step(now, event)

        def reset(self) -> None:
            pass

    with pytest.raises(ScheduleConfigError, match="run on the Rust engine"):
        MatchingToSample(3, consequence=Custom(), seed=1)  # type: ignore[arg-type]
    with pytest.raises(ScheduleConfigError, match="run on the Rust engine"):
        GoNoGo(false_alarm=Custom(), seed=1)  # type: ignore[arg-type]


def test_a_stepped_or_shared_consequence_is_rejected() -> None:
    stepped = FR(2)
    stepped.step(1.0, ResponseEvent(time=1.0))
    with pytest.raises(ScheduleConfigError, match="reset"):
        MatchingToSample(3, consequence=stepped, seed=1)
    shared = FR(1)
    with pytest.raises(ScheduleConfigError, match="same FR object twice"):
        GoNoGo(correct_go_schedule=shared, false_alarm=shared, seed=1)
    stepped.reset()
    assert MatchingToSample(3, consequence=stepped, seed=1).engine == "rust"


def test_the_consequence_object_is_not_stepped() -> None:
    consequence = FR(2)
    mts = MatchingToSample(2, consequence=consequence, iti=0.0, seed=1)
    for t in range(1, 6):
        mts.step(float(t), ResponseEvent(time=float(t), operandum=mts.correct_operandum))
    assert consequence.engine == "rust"
    assert not consequence._rust._history


def test_without_the_extension_they_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine, "rust_engine_available", lambda: False)
    with pytest.raises(ScheduleConfigError, match="contingency_core"):
        Sidman(2.0, 5.0)
    with pytest.raises(ScheduleConfigError, match="contingency_core"):
        MatchingToSample(3, seed=1)


def test_an_old_extension_without_state_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(engine.RustSchedule, "has_state", lambda self: False)
    with pytest.raises(ScheduleConfigError, match="too old"):
        Sidman(2.0, 5.0)
    with pytest.raises(ScheduleConfigError, match="too old"):
        GoNoGo(seed=1)


def test_reset_without_a_seed_draws_a_new_sequence() -> None:
    # As in Python: a trial schedule without a seed reseeds on reset.
    schedule = MatchingToSample(3, choice_timeout=1.0, iti=0.25)
    first = _trace(schedule, _CHOICES, ("correct_operandum",))
    schedule.reset()
    second = _trace(schedule, _CHOICES, ("correct_operandum",))
    assert first != second


def test_an_extension_whose_state_lacks_the_keys_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A contingency_core built before these schedule types reported
    # state has state() but returns an empty dict for them.
    monkeypatch.setattr(engine.RustSchedule, "state", lambda self: {})
    for build in (
        lambda: Sidman(2.0, 5.0),
        lambda: Escape(2.0, 3.0),
        lambda: AdjustingSchedule("ratio", 2, 1),
        lambda: InterlockingSchedule(5, 10.0),
        lambda: Percentile("irt", 50, 3),
        lambda: MatchingToSample(3, seed=1),
        lambda: GoNoGo(seed=1),
    ):
        with pytest.raises(ScheduleConfigError, match="too old"):
            build()


def test_mts_without_a_sample_phase_starts_in_choice() -> None:
    # As after reset.
    schedule = MatchingToSample(3, seed=1)
    assert schedule.phase == "choice"
    schedule.reset()
    assert schedule.phase == "choice"
    assert MatchingToSample(3, sample_duration=1.0, seed=1).phase == "sample"
