"""The single-schedule classes run on the Rust engine when it is installed."""

from __future__ import annotations

from typing import Any

import pytest

from contingency import ResponseEvent
from contingency.engine import rust_engine_available
from contingency.errors import ScheduleConfigError
from contingency.schedules import (
    CRF,
    DRH,
    DRL,
    DRO,
    EXT,
    FI,
    FR,
    FT,
    RI,
    RR,
    RT,
    VI,
    VR,
    VT,
    LimitedHold,
)

pytestmark = pytest.mark.skipif(
    not rust_engine_available(), reason="contingency_core is not installed"
)


def _factories() -> list[tuple[str, object]]:
    return [
        ("FR", lambda: FR(3)),
        ("CRF", lambda: CRF()),
        ("EXT", lambda: EXT()),
        ("FI", lambda: FI(2.0)),
        ("VI", lambda: VI(3.0, seed=5)),
        ("VR", lambda: VR(4.0, seed=5)),
        ("FT", lambda: FT(2.0)),
        ("VT", lambda: VT(3.0, seed=5)),
        ("DRO-resetting", lambda: DRO(2.0)),
        ("DRO-momentary", lambda: DRO(2.0, dro_type="momentary")),
        ("DRL", lambda: DRL(2.0)),
        ("DRH", lambda: DRH(3, 2.0)),
        ("LH-FI", lambda: LimitedHold(FI(2.0), 1.0)),
        ("LH-VI", lambda: LimitedHold(VI(3.0, seed=5), 1.0)),
    ]


@pytest.mark.parametrize(("name", "factory"), _factories(), ids=[n for n, _ in _factories()])
def test_class_runs_on_rust(name: str, factory: object) -> None:
    schedule = factory()  # type: ignore[operator]
    assert schedule.engine == "rust", name


def _trace(schedule: object) -> list[tuple[bool, float | None]]:
    out = []
    for i in range(1, 121):
        now = i * 0.25
        event = ResponseEvent(time=now) if i % 3 == 0 else None
        o = schedule.step(now, event)  # type: ignore[attr-defined]
        out.append((o.reinforced, None if o.reinforcer is None else o.reinforcer.time))
    return out


@pytest.mark.parametrize(("name", "factory"), _factories(), ids=[n for n, _ in _factories()])
def test_reset_replays_the_same_trace(name: str, factory: object) -> None:
    schedule = factory()  # type: ignore[operator]
    first = _trace(schedule)
    schedule.reset()
    assert _trace(schedule) == first, name


def test_without_the_extension_the_class_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    import contingency.engine as engine

    fi = FI(2.0)
    monkeypatch.setattr(engine, "rust_engine_available", lambda: False)
    with pytest.raises(ScheduleConfigError, match="contingency_core"):
        FR(3)
    with pytest.raises(ScheduleConfigError, match="contingency_core"):
        LimitedHold(fi, 1.0)


def test_deepcopy_keeps_the_state() -> None:
    import copy

    schedule = FR(3)
    schedule.step(1.0, ResponseEvent(time=1.0))
    schedule.step(2.0, ResponseEvent(time=2.0))
    clone = copy.deepcopy(schedule)
    assert clone.step(3.0, ResponseEvent(time=3.0)).reinforced
    assert schedule.step(3.0, ResponseEvent(time=3.0)).reinforced


def test_random_schedules_with_an_rng_run_on_rust_with_a_seed_drawn_from_it() -> None:
    import random

    for make in (
        lambda **kw: RR(0.3, **kw),
        lambda **kw: RI(2.0, **kw),
        lambda **kw: RT(2.0, **kw),
    ):
        rng = random.Random(7)
        from_rng = make(rng=rng)
        assert from_rng.engine == "rust"
        # One 63-bit draw, which advances the caller's rng once.
        expected = random.Random(7)
        seed = expected.getrandbits(63)
        assert rng.getstate() == expected.getstate()
        first = _trace(from_rng)
        assert first == _trace(make(seed=seed))
        from_rng.reset()
        assert _trace(from_rng) == first


def test_random_schedules_reject_an_rng_without_getrandbits() -> None:
    from contingency.errors import ScheduleConfigError

    with pytest.raises(ScheduleConfigError, match="random.Random"):
        RR(0.5, rng=object())  # type: ignore[arg-type]


def test_random_schedules_reject_rng_with_seed() -> None:
    import random

    from contingency.errors import ScheduleConfigError

    with pytest.raises(ScheduleConfigError):
        RR(0.5, rng=random.Random(1), seed=1)
    # A seed the Rust engine cannot take raises.
    with pytest.raises(ScheduleConfigError, match="Rust engine"):
        RR(0.5, seed=-1)


def test_progressive_ratio_runs_on_rust_for_the_factory_steps() -> None:
    from contingency.schedules import ProgressiveRatio, arithmetic, geometric, richardson_roberts

    for step_fn in (arithmetic(2, 3), richardson_roberts()):
        schedule = ProgressiveRatio(step_fn)
        assert schedule.engine == "rust"
        reinforced = 0
        for i in range(1, 60):
            if schedule.step(float(i), ResponseEvent(time=float(i))).reinforced:
                reinforced += 1
        # The public counters follow the Rust engine's reinforcers.
        assert schedule.reinforcement_index == reinforced > 0
        assert schedule.current_requirement == step_fn(reinforced)
        schedule.reset()
        assert schedule.reinforcement_index == 0
    assert ProgressiveRatio(lambda n: n + 1).engine == "python"
    # Rust rounds halves away from zero, Python to even: geometric stays in Python.
    assert ProgressiveRatio(geometric(1, 2.5)).engine == "python"


@pytest.mark.parametrize(
    "factory",
    [
        lambda: RR(0.2),
        lambda: RI(5.0),
        lambda: RT(5.0),
        lambda: VR(5.0),
        lambda: VI(5.0),
        lambda: VT(5.0),
    ],
    ids=["RR", "RI", "RT", "VR", "VI", "VT"],
)
def test_deepcopy_of_an_unseeded_random_schedule_draws_the_same_values(factory: object) -> None:
    import copy

    schedule = factory()  # type: ignore[operator]
    assert schedule.engine == "rust"
    for i in range(1, 21):
        schedule.step(float(i), ResponseEvent(time=float(i)))
    clone = copy.deepcopy(schedule)
    for i in range(21, 220):
        now = float(i)
        assert schedule.step(now, ResponseEvent(time=now)) == clone.step(
            now, ResponseEvent(time=now)
        )


def test_concurrent_cod_holds_a_rust_component_reinforcer() -> None:
    from contingency.schedules import Concurrent

    conc = Concurrent({"L": RR(0.5), "R": FR(1000)}, cod=5.0)
    now = 0.0
    inside_cod = 0
    for _ in range(300):
        now += 1.0
        conc.step(now, ResponseEvent(time=now, operandum="R"))
        now += 1.1
        if conc.step(now, ResponseEvent(time=now, operandum="L")).reinforced:
            inside_cod += 1
    assert inside_cod == 0


@pytest.mark.parametrize(
    "build",
    [lambda: FR(2**64), lambda: DRH(2**32, 1.0), lambda: FI(float("inf"))],
    ids=["FR", "DRH", "FI"],
)
def test_arguments_beyond_the_rust_integer_width_raise(build: Any) -> None:
    with pytest.raises(ScheduleConfigError, match="Rust engine"):
        build()


def test_unseeded_schedules_leave_the_module_random_alone() -> None:
    import random

    random.seed(1)
    expected = random.random()
    random.seed(1)
    RR(0.5)
    VI(2.0)
    assert random.random() == expected


def test_limited_hold_over_an_unseeded_vi_draws_anew_on_reset() -> None:
    schedule = LimitedHold(VI(3.0), 0.4)
    assert schedule.engine == "rust"
    traces = []
    for _ in range(4):
        traces.append(_trace(schedule))
        schedule.reset()
    assert len({tuple(t) for t in traces}) > 1
