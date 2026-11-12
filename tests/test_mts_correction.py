"""MTS correction trials: the Python class, its Rust engine, and the DSL bridge.

With correction active, a baseline trial that ends incorrect (a wrong
choice or a timeout) is followed, after the ITI, by the same trial: the
same correct operandum, sample and comparison positions, with no draw
from the RNG. ``True`` repeats until correct; an int ``N`` repeats at
most ``N`` times. Probe trials are not repeated.
"""

from __future__ import annotations

from typing import Any

import pytest
from contingency_dsl import parse

import contingency.schedules._rust as _rust
from contingency import ResponseEvent
from contingency.bridge import from_dsl, to_spec
from contingency.engine import from_spec, rust_engine_available
from contingency.errors import ScheduleConfigError
from contingency.schedules import MatchingToSample, TrialType

requires_rust = pytest.mark.skipif(
    not rust_engine_available(), reason="contingency_core is not installed"
)

_TYPES = [
    TrialType(sample=f"A{i}", correct=f"B{i}", comparisons=("B1", "B2", "B3"))
    for i in (1, 2, 3)
]
_PROBES = [TrialType(sample="C1", correct="B1", comparisons=("B1", "B2", "B3"))]


def _make(
    monkeypatch: pytest.MonkeyPatch,
    *,
    python: bool,
    correction: bool | int | None,
    trial_types: bool = True,
    probes: bool = False,
) -> MatchingToSample:
    kwargs: dict[str, Any] = {
        "n_comparisons": 3,
        "choice_timeout": 5.0,
        "iti": 1.0,
        "seed": 11,
        "correction": correction,
    }
    if trial_types:
        kwargs["trial_types"] = _TYPES
        kwargs["probe_trial_types"] = _PROBES if probes else None
    if python:
        with monkeypatch.context() as m:
            m.setattr(_rust, "_FORCE_PYTHON", True)
            mts = MatchingToSample(**kwargs)
        assert mts.engine == "python"
        return mts
    return MatchingToSample(**kwargs)


def _wrong(correct: str) -> str:
    return next(f"choice_{i}" for i in range(3) if f"choice_{i}" != correct)


# answers: True = correct, False = wrong, None = let the choice time out
_SCRIPT: list[bool | None] = [False, False, None, True, False, True, False, False, False, False]


def _drive(schedule: Any, script: list[bool | None] = _SCRIPT) -> list[tuple[Any, ...]]:
    """Run ``script``; record (trial, correct operandum, positions, outcome) per trial."""
    rows: list[tuple[Any, ...]] = []
    t = 0.0
    schedule.step(t, None)
    for answer in script:
        state = (schedule.trial_index, schedule.correct_operandum, schedule.current_comparisons)
        if answer is None:
            t += 5.5
            out = schedule.step(t, None)
        else:
            t += 0.5
            op = schedule.correct_operandum if answer else _wrong(schedule.correct_operandum)
            out = schedule.step(t, ResponseEvent(t, op))
        rows.append((*state, out.reinforced, dict(out.meta)))
        t += 1.0
        schedule.step(t, None)
    return rows


@pytest.mark.unit
def test_python_repeats_the_trial_until_correct(monkeypatch: pytest.MonkeyPatch) -> None:
    mts = _make(monkeypatch, python=True, correction=True)
    rows = _drive(mts)
    # trials 0-3 are the same trial (wrong, wrong, timeout, correct)
    assert len({(r[1], r[2]) for r in rows[:4]}) == 1
    assert [r[4]["correction_trial"] for r in rows[:4]] == [False, True, True, True]
    assert rows[3][3] is True
    assert rows[4][4]["correction_trial"] is False


@pytest.mark.unit
def test_python_bounded_draws_a_new_trial_after_the_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plain = _make(monkeypatch, python=True, correction=None)
    plain.step(0.0, None)
    plain.step(0.5, ResponseEvent(0.5, _wrong(plain.correct_operandum)))
    plain.step(1.5, None)
    drawn_second = (plain.correct_operandum, plain.current_comparisons)

    mts = _make(monkeypatch, python=True, correction=2)
    rows = _drive(mts, [False, False, False, True])
    assert len({(r[1], r[2]) for r in rows[:3]}) == 1
    assert (rows[3][1], rows[3][2]) == drawn_second
    assert [r[4]["correction_trial"] for r in rows] == [False, True, True, False]


@pytest.mark.unit
def test_probe_trials_are_not_repeated(monkeypatch: pytest.MonkeyPatch) -> None:
    # bounded, so the wrong answers do not hold the first baseline trial forever
    mts = _make(monkeypatch, python=True, correction=1, probes=True)
    rows = _drive(mts, [False] * 16)
    for prev, cur in zip(rows, rows[1:], strict=False):
        if prev[4]["probe"]:
            assert cur[4]["correction_trial"] is False
    assert any(r[4]["probe"] for r in rows)


@pytest.mark.unit
@pytest.mark.parametrize("correction", [None, False])
def test_disabled_adds_no_meta(monkeypatch: pytest.MonkeyPatch, correction: bool | None) -> None:
    rows = _drive(_make(monkeypatch, python=True, correction=correction))
    assert all("correction_trial" not in r[4] for r in rows)


@pytest.mark.unit
@pytest.mark.parametrize("bad", [0, -1, 2**32, 1.5, "2"])
def test_bad_correction_is_rejected(bad: Any) -> None:
    with pytest.raises(ScheduleConfigError):
        MatchingToSample(n_comparisons=2, correction=bad)


@pytest.mark.unit
def test_reset_clears_a_pending_repeat(monkeypatch: pytest.MonkeyPatch) -> None:
    mts = _make(monkeypatch, python=True, correction=True)
    first = _drive(mts, [True])
    mts.reset()
    _drive(mts, [False])
    mts.reset()
    assert _drive(mts, [True]) == first


@requires_rust
@pytest.mark.unit
@pytest.mark.parametrize("correction", [None, True, 1, 2])
@pytest.mark.parametrize("trial_types", [True, False])
@pytest.mark.parametrize("probes", [False, True])
def test_python_and_rust_agree(
    monkeypatch: pytest.MonkeyPatch,
    correction: bool | int | None,
    trial_types: bool,
    probes: bool,
) -> None:
    if probes and not trial_types:
        pytest.skip("probe trial types need trial types")
    py = _make(
        monkeypatch, python=True, correction=correction, trial_types=trial_types, probes=probes
    )
    rs = _make(
        monkeypatch, python=False, correction=correction, trial_types=trial_types, probes=probes
    )
    assert rs.engine == "rust"
    script = _SCRIPT * 3
    assert _drive(rs, script) == _drive(py, script)


_DSL = "MTS(comparisons=3, consequence=CRF, incorrect=EXT, ITI=1s, correction={c})"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("c", "expected"),
    [
        ("true", {"mode": "unlimited"}),
        ('"repeat_until_correct"', {"mode": "unlimited"}),
        ("2", {"mode": "bounded", "limit": 2}),
    ],
)
def test_to_spec_writes_correction(c: str, expected: dict[str, Any]) -> None:
    spec = to_spec(parse(_DSL.format(c=c)))
    assert spec["type"] == "MatchingToSample"
    assert spec["params"]["correction"] == expected


@pytest.mark.unit
def test_correction_false_is_the_same_spec_as_omitted() -> None:
    off = to_spec(parse(_DSL.format(c="false")))
    omitted = to_spec(parse("MTS(comparisons=3, consequence=CRF, incorrect=EXT, ITI=1s)"))
    off["params"].pop("seed", None)
    omitted["params"].pop("seed", None)
    assert off == omitted
    assert "correction" not in off["params"]


@pytest.mark.unit
@pytest.mark.parametrize(("c", "expected"), [("true", True), ("2", 2), ("false", False)])
def test_from_dsl_passes_correction(c: str, expected: bool | int) -> None:
    mts = from_dsl(parse(_DSL.format(c=c)))
    assert isinstance(mts, MatchingToSample)
    assert mts.correction == expected


@requires_rust
@pytest.mark.unit
@pytest.mark.parametrize("c", ["true", "1", "2"])
def test_from_dsl_and_rust_from_spec_agree(c: str) -> None:
    """The same DSL through from_dsl and through to_spec -> Rust from_spec.

    The two draw different seeds, so the answers are given relative to
    each schedule's own correct operandum.
    """
    src = parse(_DSL.format(c=c))
    built = from_dsl(src)
    rust = from_spec(to_spec(src))
    script = _SCRIPT * 2
    got_dsl = _drive_raw(built, script, lambda: built.correct_operandum)
    got_rust = _drive_raw(rust, script, lambda: str(rust.state()["correct_operandum"]))
    assert got_rust == got_dsl


def _drive_raw(
    schedule: Any, script: list[bool | None], correct_of: Any
) -> list[tuple[bool, Any, Any]]:
    """Like ``_drive`` but reads only outcomes (works for a bare Rust schedule)."""
    rows = []
    t = 0.0
    schedule.step(t, None)
    for answer in script:
        if answer is None:
            t += 5.5
            out = schedule.step(t, None)
        else:
            t += 0.5
            correct = correct_of()
            op = correct if answer else _wrong(correct)
            out = schedule.step(t, ResponseEvent(t, op))
        meta = dict(out.meta)
        rows.append((out.reinforced, meta.get("trial_outcome"), meta.get("correction_trial")))
        t += 1.0
        schedule.step(t, None)
    return rows


@requires_rust
@pytest.mark.unit
def test_an_engine_without_correction_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A contingency_core that predates correction reads past the key."""
    import contingency.schedules.trial_based as trial_based

    real = trial_based.rust_or_none

    class _Old:
        def __init__(self, inner: Any) -> None:
            self._inner = inner

        def step(self, now: float, event: Any = None) -> Any:
            out = self._inner.step(now, event)
            meta = {k: v for k, v in out.meta.items() if k != "correction_trial"}
            return type(out)(reinforced=out.reinforced, reinforcer=out.reinforcer, meta=meta)

    def old_engine(spec: dict[str, Any]) -> Any:
        params = {k: v for k, v in spec["params"].items() if k != "correction"}
        rust = real({**spec, "params": params})
        return None if rust is None else _Old(rust)

    monkeypatch.setattr(trial_based, "rust_or_none", old_engine)
    with pytest.raises(ScheduleConfigError, match="too old"):
        MatchingToSample(n_comparisons=2, correction=True)
    assert MatchingToSample(n_comparisons=2).engine == "rust"


@pytest.mark.unit
def test_to_spec_rejects_a_limit_from_dsl_rejects() -> None:
    src = parse(_DSL.format(c=str(2**32)))
    with pytest.raises(ScheduleConfigError):
        from_dsl(src)
    with pytest.raises(ScheduleConfigError):
        to_spec(src)
