"""Changeover punishment of ``Concurrent``.

``PUNISH(changeover)=P`` gives one punishment schedule shared by every
changeover direction; ``PUNISH(a->b)=P`` gives one per direction. A
punishment schedule receives a response only when the response confirms
a changeover (the test that arms COD; with FRCO, the FRCO-th consecutive
response on the new operandum). ``Overlay(Conc(...), P,
target=changeover)`` runs the same as ``Conc(..., PUNISH(changeover)=P)``.
"""

from __future__ import annotations

from typing import Any

import pytest
from contingency_dsl import parse

import contingency.engine as engine
from contingency import ResponseEvent
from contingency.bridge import from_dsl, to_spec
from contingency.engine import from_spec, rust_engine_available
from contingency.errors import ScheduleConfigError
from contingency.schedules import EXT, FR, Concurrent

requires_rust = pytest.mark.skipif(
    not rust_engine_available(), reason="contingency_core is not installed"
)

_OPS = ["c0", "c0", "c1", "c1", "c0"]


def _drive(schedule: Any, ops: list[str]) -> list[tuple[Any, ...]]:
    rows = []
    for i, op in enumerate(ops):
        t = float(i + 1)
        out = schedule.step(t, ResponseEvent(time=t, operandum=op))
        r = out.reinforcer
        rows.append(
            (
                out.reinforced,
                None if r is None else (r.magnitude, r.label),
                tuple(sorted(out.meta.items())),
            )
        )
    return rows


def _punished(rows: list[tuple[Any, ...]]) -> list[bool]:
    return [dict(meta).get("punisher_fired") is True for _, _, meta in rows]


def _conc(monkeypatch: pytest.MonkeyPatch, *, python: bool, **kwargs: Any) -> Concurrent:
    def build() -> Concurrent:
        comps = {"c0": EXT(), "c1": EXT()}
        return Concurrent(comps, cod=2.0, **kwargs)

    if python:
        with monkeypatch.context() as m:
            m.setattr(engine, "rust_engine_available", lambda: False)
            conc = build()
        assert conc.engine == "python"
        return conc
    return build()


# --- the class, both engines ---------------------------------------------


_CASES: list[tuple[str, dict[str, Any], list[str], list[bool]]] = [
    ("changeover", {"punish_changeover": "FR1"}, _OPS, [False, False, True, False, True]),
    # FR 2 shared: every 2nd changeover overall, whatever its direction.
    (
        "shared",
        {"punish_changeover": "FR2"},
        ["c0", "c1", "c0", "c1", "c0"],
        [False, False, True, False, True],
    ),
    ("directional", {"punish_directional": {("c0", "c1"): "FR1"}}, _OPS,
     [False, False, True, False, False]),
    (
        "per-direction",
        {"punish_directional": {("c0", "c1"): "FR2", ("c1", "c0"): "FR1"}},
        ["c0", "c1", "c0", "c1"],
        [False, False, True, True],
    ),
    ("frco", {"frco": 2, "punish_changeover": "FR1"}, ["c0", "c1", "c1", "c0", "c1"],
     [False, False, True, False, False]),
]


def _materialise(kwargs: dict[str, Any]) -> dict[str, Any]:
    def fr(text: str) -> FR:
        return FR(int(text[2:]))

    out: dict[str, Any] = {}
    for k, v in kwargs.items():
        if k == "punish_changeover":
            out[k] = fr(v)
        elif k == "punish_directional":
            out[k] = {pair: fr(s) for pair, s in v.items()}
        else:
            out[k] = v
    return out


@pytest.mark.unit
@pytest.mark.parametrize("python", [True, False])
@pytest.mark.parametrize(("name", "kwargs", "ops", "expected"), _CASES)
def test_changeover_punishment(
    monkeypatch: pytest.MonkeyPatch,
    python: bool,
    name: str,
    kwargs: dict[str, Any],
    ops: list[str],
    expected: list[bool],
) -> None:
    if not python and not rust_engine_available():
        pytest.skip("contingency_core is not installed")
    conc = _conc(monkeypatch, python=python, **_materialise(kwargs))
    if not python:
        assert conc.engine == "rust"
    rows = _drive(conc, ops)
    assert _punished(rows) == expected, name
    for (_, rf, meta), hit in zip(rows, expected, strict=True):
        if hit:
            assert rf == (-1.0, "SR-")
            assert dict(meta)["changeover_punished"] is True


@pytest.mark.unit
@pytest.mark.parametrize("python", [True, False])
def test_component_and_directional_punishers_add_up(
    monkeypatch: pytest.MonkeyPatch, python: bool
) -> None:
    if not python and not rust_engine_available():
        pytest.skip("contingency_core is not installed")
    conc = _conc(
        monkeypatch,
        python=python,
        punish={"c1": FR(1)},
        punish_directional={("c0", "c1"): FR(1)},
    )
    rows = _drive(conc, ["c0", "c1", "c1"])
    assert _punished(rows) == [False, True, True]
    assert rows[1][1] == (-2.0, "SR-")
    assert rows[2][1] == (-1.0, "SR-")
    assert "changeover_punished" not in dict(rows[2][2])


@pytest.mark.unit
@pytest.mark.parametrize("python", [True, False])
def test_reset_clears_changeover_punishers(monkeypatch: pytest.MonkeyPatch, python: bool) -> None:
    if not python and not rust_engine_available():
        pytest.skip("contingency_core is not installed")
    conc = _conc(monkeypatch, python=python, punish_changeover=FR(2))
    _drive(conc, ["c0", "c1"])
    conc.reset()
    assert _punished(_drive(conc, ["c0", "c1", "c0"])) == [False, False, True]


@requires_rust
@pytest.mark.unit
@pytest.mark.parametrize(("name", "kwargs", "ops", "expected"), _CASES)
def test_python_and_rust_agree(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    kwargs: dict[str, Any],
    ops: list[str],
    expected: list[bool],
) -> None:
    py = _conc(monkeypatch, python=True, **_materialise(kwargs))
    rs = _conc(monkeypatch, python=False, **_materialise(kwargs))
    assert rs.engine == "rust"
    script = ops * 4
    assert _drive(rs, script) == _drive(py, script), name


@pytest.mark.unit
@pytest.mark.parametrize(
    "kwargs",
    [
        {"punish_changeover": FR(1), "punish_directional": {("c0", "c1"): FR(1)}},
        {"punish_directional": {("c0", "c0"): FR(1)}},
        {"punish_directional": {("c0", "c9"): FR(1)}},
        {"punish_directional": {"c0": FR(1)}},
        {"punish_directional": [("c0", "c1")]},
        {"punish_changeover": 3},
        {"punish_directional": {("c0", "c1"): 3}},
    ],
)
def test_bad_arguments_are_rejected(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ScheduleConfigError):
        Concurrent({"c0": EXT(), "c1": EXT()}, **kwargs)


# --- from_dsl / to_spec ----------------------------------------------------


_CONC = "Conc(VI30s, VI60s, COD=2s, {p})"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (_CONC.format(p="PUNISH(changeover)=FR1"), [False, False, True, False, True]),
        (_CONC.format(p="PUNISH(1->2)=FR1"), [False, False, True, False, False]),
        ("Overlay(Conc(VI30s, VI60s, COD=2s), FR1, target=changeover)",
         [False, False, True, False, True]),
    ],
)
@pytest.mark.parametrize("rust", [False, True])
def test_from_dsl_punishes_changeovers(source: str, expected: list[bool], rust: bool) -> None:
    if rust and not rust_engine_available():
        pytest.skip("contingency_core is not installed")
    program = parse(source)
    schedule = from_dsl(program, engine="rust") if rust else from_dsl(program)
    assert _punished(_drive(schedule, _OPS)) == expected


@requires_rust
@pytest.mark.unit
@pytest.mark.parametrize(
    "source",
    [
        # Ratio components: VI without a seed draws a different sequence
        # in each engine.
        "Conc(FR2, FR3, COD=2s, PUNISH(changeover)=FR2)",
        "Conc(FR2, FR3, COD=2s, PUNISH(1->2)=FR1, PUNISH(2->1)=FR2)",
        "Conc(FR2, FR3, COD=2s, PUNISH(1)=FR2, PUNISH(1->2)=FR1)",
        "Conc(FR2, FR3, FRCO=2, PUNISH(changeover)=FR1)",
        "Overlay(Conc(FR2, FR3, COD=1s), FR2, target=changeover)",
    ],
)
def test_from_dsl_and_rust_from_spec_agree(source: str) -> None:
    program = parse(source)
    script = ["c0", "c1", "c1", "c0", "c0", "c1", "c0", "c1", "c1", "c1"] * 3
    rows = _drive(from_dsl(program), script)
    assert rows == _drive(from_spec(to_spec(program)), script)
    assert any(_punished(rows))


@pytest.mark.unit
def test_overlay_target_changeover_is_the_spec_of_punish_changeover() -> None:
    overlay = to_spec(parse("Overlay(Conc(VI30s, VI60s, COD=2s), FR1, target=changeover)"))
    conc = to_spec(parse(_CONC.format(p="PUNISH(changeover)=FR1")))
    assert overlay == conc
    assert conc["params"]["punish_changeover"] == {"type": "FR", "params": {"n": 1}}


@pytest.mark.unit
def test_to_spec_writes_directional_in_source_order() -> None:
    spec = to_spec(parse(_CONC.format(p="PUNISH(2->1)=FR2, PUNISH(1->2)=FR1")))
    assert spec["params"]["punish_directional"] == [
        {"from": "c1", "to": "c0", "schedule": {"type": "FR", "params": {"n": 2}}},
        {"from": "c0", "to": "c1", "schedule": {"type": "FR", "params": {"n": 1}}},
    ]


@pytest.mark.unit
def test_overlay_target_all_is_unchanged() -> None:
    plain = "Overlay(Conc(VI30s, VI60s), FR1)"
    explicit = "Overlay(Conc(VI30s, VI60s), FR1, target=all)"
    assert to_spec(parse(plain)) == to_spec(parse(explicit))
    assert to_spec(parse(plain))["type"] == "Overlay"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("source", "match"),
    [
        ("Overlay(VI30s, FR1, target=changeover)", "Conc baseline"),
        (
            "Overlay(Conc(VI30s, VI60s, PUNISH(changeover)=FR2), FR1, target=changeover)",
            "ambiguous",
        ),
        (
            "Overlay(Conc(VI30s, VI60s, PUNISH(1->2)=FR2), FR1, target=changeover)",
            "ambiguous",
        ),
        ("Conc(VI30s, VI60s, PUNISH(1->3)=FR1)", "component 3"),
    ],
)
def test_from_dsl_and_to_spec_reject_the_same(source: str, match: str) -> None:
    program = parse(source)
    with pytest.raises(ScheduleConfigError, match=match) as a:
        from_dsl(program)
    with pytest.raises(ScheduleConfigError, match=match) as b:
        to_spec(program)
    assert type(a.value) is type(b.value)


# --- an older contingency_core without changeover punishment ----------------


@requires_rust
@pytest.mark.unit
def test_class_runs_in_python_on_an_older_core(monkeypatch: pytest.MonkeyPatch) -> None:
    import contingency.schedules.concurrent as conc_mod

    monkeypatch.setattr(conc_mod, "_rust_has_changeover_punish", lambda: False)
    conc = Concurrent({"c0": EXT(), "c1": EXT()}, cod=2.0, punish_changeover=FR(1))
    assert conc.engine == "python"
    assert _punished(_drive(conc, _OPS)) == [False, False, True, False, True]
    plain = Concurrent({"c0": EXT(), "c1": EXT()}, cod=2.0, punish={"c1": FR(1)})
    assert plain.engine == "rust"


@requires_rust
@pytest.mark.unit
@pytest.mark.parametrize(
    "source",
    [
        _CONC.format(p="PUNISH(changeover)=FR1"),
        "Mult(FR5, Overlay(Conc(VI30s, VI60s), FR1, target=changeover))",
    ],
)
def test_engine_rust_rejects_on_an_older_core(
    monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    import contingency.schedules.concurrent as conc_mod

    monkeypatch.setattr(conc_mod, "_rust_has_changeover_punish", lambda: False)
    with pytest.raises(ScheduleConfigError, match="changeover"):
        from_dsl(parse(source), engine="rust")
    from_dsl(parse(_CONC.format(p="PUNISH(1)=FR1")), engine="rust")


@pytest.mark.unit
def test_both_forms_are_rejected_even_with_an_empty_directional() -> None:
    with pytest.raises(ScheduleConfigError, match="not both"):
        Concurrent({"c0": EXT(), "c1": EXT()}, punish_changeover=FR(1), punish_directional={})


@pytest.mark.unit
def test_overlay_target_other_than_all_or_changeover_has_no_spec_form() -> None:
    from contingency.errors import SpecUnrepresentableError

    program = parse("Overlay(Conc(FR1, FR1), FR1, target=component_1)")
    with pytest.raises(SpecUnrepresentableError, match="target"):
        to_spec(program)
    with pytest.raises(ScheduleConfigError, match="target"):
        from_dsl(program)
