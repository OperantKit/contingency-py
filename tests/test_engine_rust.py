"""Tests for the Rust engine adapter (:mod:`contingency.engine`).

The adapter wraps a ``contingency_core`` schedule (the PyO3 build of
contingency-rs) behind the :class:`~contingency.interfaces.Schedule`
protocol and this package's value types. These tests check that:

* the adapter speaks this package's types and errors;
* every conformance fixture replays bit-for-bit through it;
* ``from_dsl(..., engine="rust")`` behaves exactly like the Python
  engine for DSL programs that have a spec form.

They are skipped when ``contingency_core`` is not installed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

core = pytest.importorskip("contingency_core")

from contingency_dsl import parse  # noqa: E402
from contingency_dsl.annotations.temporal.extension import (  # noqa: E402
    TemporalExtension,
)
from contingency_dsl.extensions import ExtensionRegistry  # noqa: E402

from contingency import from_dsl  # noqa: E402
from contingency.engine import RustSchedule, from_spec  # noqa: E402
from contingency.entities import Outcome, Reinforcer, ResponseEvent  # noqa: E402
from contingency.errors import (  # noqa: E402
    ScheduleConfigError,
    ScheduleStateError,
)
from contingency.interfaces import Schedule  # noqa: E402

CONFORMANCE_DIR = Path(__file__).resolve().parents[1] / "conformance"
FIXTURES = sorted(CONFORMANCE_DIR.rglob("*.json"))
FLOAT_TOL = 1e-9

_REGISTRY = ExtensionRegistry.build(TemporalExtension())


# --- Adapter types and errors ------------------------------------------------


@pytest.mark.unit
def test_from_spec_returns_a_schedule() -> None:
    s = from_spec({"type": "FR", "params": {"n": 2}})
    assert isinstance(s, RustSchedule)
    assert isinstance(s, Schedule)


@pytest.mark.unit
def test_from_spec_accepts_a_json_string() -> None:
    s = from_spec(json.dumps({"type": "CRF", "params": {}}))
    assert s.step(1.0, ResponseEvent(1.0)).reinforced


@pytest.mark.unit
def test_step_returns_this_packages_outcome() -> None:
    s = from_spec({"type": "FR", "params": {"n": 2}})
    first = s.step(1.0, ResponseEvent(1.0))
    second = s.step(2.0, ResponseEvent(2.0))
    assert type(first) is Outcome
    assert first == Outcome()
    assert type(second) is Outcome
    assert second.reinforced
    assert type(second.reinforcer) is Reinforcer
    assert second.reinforcer == Reinforcer(time=2.0)
    assert isinstance(second.meta, dict)


@pytest.mark.unit
def test_step_without_event_and_custom_operandum() -> None:
    s = from_spec(
        {
            "type": "Concurrent",
            "params": {
                "components": {
                    "left": {"type": "FR", "params": {"n": 1}},
                    "right": {"type": "EXT", "params": {}},
                },
                "cod": 0.0,
                "frco": 0,
            },
        }
    )
    assert not s.step(0.5).reinforced
    assert not s.step(1.0, ResponseEvent(1.0, "right")).reinforced
    assert s.step(2.0, ResponseEvent(2.0, "left")).reinforced


@pytest.mark.unit
def test_reset_returns_to_post_construction_state() -> None:
    s = from_spec({"type": "FR", "params": {"n": 2}})
    s.step(1.0, ResponseEvent(1.0))
    s.reset()
    assert not s.step(0.5, ResponseEvent(0.5)).reinforced
    assert s.step(0.6, ResponseEvent(0.6)).reinforced


@pytest.mark.unit
def test_bad_spec_raises_schedule_config_error() -> None:
    with pytest.raises(ScheduleConfigError):
        from_spec({"type": "FR", "params": {"n": 0}})
    with pytest.raises(ScheduleConfigError):
        from_spec({"type": "NoSuchSchedule", "params": {}})


@pytest.mark.unit
def test_non_monotonic_time_raises_schedule_state_error() -> None:
    s = from_spec({"type": "FR", "params": {"n": 5}})
    s.step(2.0, ResponseEvent(2.0))
    with pytest.raises(ScheduleStateError):
        s.step(1.0, ResponseEvent(1.0))


@pytest.mark.unit
def test_unknown_operandum_raises_schedule_config_error() -> None:
    s = from_spec(
        {
            "type": "Concurrent",
            "params": {
                "components": {
                    "a": {"type": "FR", "params": {"n": 1}},
                    "b": {"type": "FR", "params": {"n": 1}},
                },
                "cod": 0.0,
                "frco": 0,
            },
        }
    )
    with pytest.raises(ScheduleConfigError):
        s.step(1.0, ResponseEvent(1.0, "main"))


# --- Conformance corpus through the adapter ----------------------------------


def _fixture_id(path: Path) -> str:
    return str(path.relative_to(CONFORMANCE_DIR))


@pytest.mark.parametrize("fixture_path", FIXTURES, ids=[_fixture_id(p) for p in FIXTURES])
def test_conformance_fixture_through_rust_engine(fixture_path: Path) -> None:
    data: dict[str, Any] = json.loads(fixture_path.read_text())
    schedule = from_spec(data["schedule"])
    for i, step in enumerate(data["steps"]):
        ev = step["event"]
        event = None if ev is None else ResponseEvent(ev["time"], ev.get("operandum", "main"))
        outcome = schedule.step(step["now"], event)
        expected = step["expect"]
        where = f"{fixture_path.name} step {i}"
        assert outcome.reinforced == expected["reinforced"], where
        if expected["reinforced"]:
            exp_r = expected["reinforcer"]
            assert outcome.reinforcer is not None, where
            assert abs(outcome.reinforcer.time - exp_r["time"]) <= FLOAT_TOL, where
            assert abs(outcome.reinforcer.magnitude - exp_r["magnitude"]) <= FLOAT_TOL, where
            assert outcome.reinforcer.label == exp_r["label"], where
        else:
            assert outcome.reinforcer is None, where
        for k, v in expected.get("meta", {}).items():
            assert k in outcome.meta, f"{where}: meta key {k!r} missing"
            if isinstance(v, float):
                assert abs(float(outcome.meta[k]) - v) <= FLOAT_TOL, where  # type: ignore[arg-type]
            else:
                assert outcome.meta[k] == v, where


# --- from_dsl: Python engine vs Rust engine ----------------------------------

_SEED = '@algorithm("fleshler-hoffman", seed=11)\n'

#: DSL programs with a spec form, one or more per schedule family.
DSL_PROGRAMS = [
    "FR5",
    "CRF",
    "EXT",
    "FI30s",
    "FT15s",
    "VR10",
    "VI30s",
    "VT45s",
    "RR20",
    "RI30s",
    "RT20s",
    "FI2s LH500ms",
    "DRL10s",
    "DRO5s",
    "DRH2s",
    "Mult(FR5, EXT)",
    "Chain(FR3, FI5s)",
    "Tand(FR3, FI5s)",
    "Alt(FR10, FI30s)",
    "Conj(FR5, FI10s)",
    "Mix(FR5, FR10)",
    "Conc(FR5, VI30s)",
    "Conc(VI30s, VI60s, COD=2s, FRCO=5)",
    "FR5(FI30s)",
    "Overlay(FR5, FR10)",
    "Interpolate(FR5, FR10, count=5, onset=60s)",
    "PR(linear, start=1, increment=2)",
    "PR(geometric, start=1, ratio=2)",
    "PR(hodos)",
    "Pctl(IRT, 50, window=20, dir=above)",
    "Adj(ratio, start=5, step=2, min=1, max=20)",
    "Adj(interval, start=2s, step=1s)",
    "Interlock(R0=4, T=8s)",
    "Escape(SafeDuration=2s, MaxShock=3s)",
    "MTS(comparisons=3, consequence=VR2, incorrect=RT5s, ITI=1s)",
    "GoNoGo(responseWindow=5s, consequence=CRF, ITI=10s)",
]


#: Programs that never reinforce by design.
_NEVER_REINFORCED = {"EXT"}


def _replay(schedule: Schedule, operanda: list[str], sparse: bool) -> list[tuple[Any, ...]]:
    """Drive ``schedule`` for 400 steps and record each outcome.

    The dense stream responds on four of every five steps (IRTs of
    about 0.4-1.0 s); the sparse stream leaves pauses of up to 12 s so
    that DRO and DRL also reach reinforcement. Meta values are recorded
    with their type so that ``1`` and ``1.0`` or ``True`` and ``1``
    count as different.
    """
    rows: list[tuple[Any, ...]] = []
    t = 0.0
    for i in range(400):
        gap = 0.37 + (i % 7) * 0.11
        if sparse and i % 4 == 3:
            gap += 6.0 + (i % 3) * 3.0
        t = round(t + gap, 6)
        respond = (i % 5 != 4) if not sparse else (i % 2 == 0)
        event = ResponseEvent(t, operanda[i % len(operanda)]) if respond else None
        o = schedule.step(t, event)
        r = o.reinforcer
        meta = sorted((k, type(v).__name__, v) for k, v in o.meta.items() if v is not None)
        rows.append((o.reinforced, None if r is None else (r.time, r.magnitude, r.label), meta))
    return rows


@pytest.mark.parametrize("source", DSL_PROGRAMS)
def test_from_dsl_rust_engine_matches_python_engine(source: str) -> None:
    program = parse(_SEED + source, extensions=_REGISTRY)
    spec_ops = ["c0", "c1"] if source.startswith("Conc") else ["main"]
    reinforced = 0
    for sparse in (False, True):
        py = from_dsl(program)
        rs = from_dsl(program, engine="rust")
        assert isinstance(rs, RustSchedule)
        rs_rows = _replay(rs, spec_ops, sparse)
        assert rs_rows == _replay(py, spec_ops, sparse)
        reinforced += sum(row[0] for row in rs_rows)
    # Guard against a vacuous comparison where neither engine reinforces.
    assert (reinforced == 0) == (source in _NEVER_REINFORCED)
