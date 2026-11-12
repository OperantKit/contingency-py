"""Program-level parameter declarations (``LH``, ``COD``, ``FRCO``, ``RD``, ``BO``).

The contingency-dsl grammar applies a program-level ``LH`` to every leaf
base schedule that has no LH of its own (constraint 3), and a
program-level ``COD`` / ``FRCO`` to every ``Conc`` that does not set it
(constraint 12). ``from_dsl`` and ``to_spec`` apply these themselves,
without ``contingency_dsl.analyze()`` (which drops ``@algorithm`` seeds).
``RD`` greater than zero is rejected: the runtime has no reinforcement
delay. ``RD=0s`` means immediate delivery and is accepted.
"""

from __future__ import annotations

from typing import Any

import pytest
from contingency_dsl import parse
from contingency_dsl.annotations.temporal.extension import TemporalExtension
from contingency_dsl.extensions import ExtensionRegistry

from contingency import from_dsl, to_spec
from contingency.conformance.spec import build_schedule_from_spec
from contingency.entities import ResponseEvent
from contingency.errors import ScheduleConfigError
from contingency.interfaces import Schedule

_REGISTRY = ExtensionRegistry.build(TemporalExtension())
_SEEDED = '@algorithm("fleshler-hoffman", seed=11)\n'


def _parse(source: str) -> Any:
    return parse(source, extensions=_REGISTRY)


def _trace(schedule: Schedule, operanda: list[str]) -> list[tuple[Any, ...]]:
    rows: list[tuple[Any, ...]] = []
    t = 0.0
    for i in range(300):
        t = round(t + 0.4 + (i % 9) * 0.35 + (4.0 if i % 11 == 10 else 0.0), 6)
        event = ResponseEvent(t, operanda[i % len(operanda)]) if i % 3 != 2 else None
        o = schedule.step(t, event)
        r = o.reinforcer
        rows.append((o.reinforced, None if r is None else (r.time, r.magnitude, r.label)))
    return rows


# --- Declarations applied the same way as the expression-level forms --------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("declared", "written", "operanda"),
    [
        ("LH = 2s\nFI5s", "FI5s LH2s", ["main"]),
        ("LH = 1500ms\nVI10s", "VI10s LH1500ms", ["main"]),
        ("LH = 2s\nConc(FI5s LH1s, FI4s)", "Conc(FI5s LH1s, FI4s LH2s)", ["c0", "c1"]),
        ("LH = 2s\nMult(FI5s, FI3s)", "Mult(FI5s LH2s, FI3s LH2s)", ["main"]),
        # An LH on a ratio schedule is vacuous and is not applied.
        ("LH = 2s\nMult(FI5s, FR3, VR4, CRF)", "Mult(FI5s LH2s, FR3, VR4, CRF)", ["main"]),
        ("LH = 2s\nFR5", "FR5", ["main"]),
        ("LH = 2s\nMult(FI5s, EXT, PR(hodos))", "Mult(FI5s LH2s, EXT, PR(hodos))", ["main"]),
        (
            "LH = 2s\nFI5s TO(duration=5s, reset_on_response=false)",
            "FI5s LH2s TO(duration=5s, reset_on_response=false)",
            ["main"],
        ),
        ("LH = 2s\nFI5s ResponseCost(amount=1)", "FI5s LH2s ResponseCost(amount=1)", ["main"]),
        (
            "COD = 1s\nConc(VI30s, VI60s, COD(1->2)=3s)",
            "Conc(VI30s, VI60s, COD=1s, COD(1->2)=3s)",
            ["c0", "c1"],
        ),
        ("COD = 2s\nConc(VI30s, VI60s)", "Conc(VI30s, VI60s, COD=2s)", ["c0", "c1"]),
        ("COD = 2s\nConc(VI30s, VI60s, COD=1s)", "Conc(VI30s, VI60s, COD=1s)", ["c0", "c1"]),
        ("FRCO = 3\nConc(VI10s, VI20s)", "Conc(VI10s, VI20s, FRCO=3)", ["c0", "c1"]),
        (
            "COD = 1s\nFRCO = 2\nConc(VI10s, VI20s, FRCO=4)",
            "Conc(VI10s, VI20s, COD=1s, FRCO=4)",
            ["c0", "c1"],
        ),
        ("RD = 0s\nFR5", "FR5", ["main"]),
        ("BO = 0s\nMult(FR2, FR3)", "Mult(FR2, FR3)", ["main"]),
    ],
)
def test_declaration_matches_written_form(declared: str, written: str, operanda: list[str]) -> None:
    declared_ast = _parse(_SEEDED + declared)
    written_ast = _parse(_SEEDED + written)
    assert to_spec(declared_ast) == to_spec(written_ast)
    expected = _trace(from_dsl(written_ast), operanda)
    assert _trace(from_dsl(declared_ast), operanda) == expected
    assert _trace(build_schedule_from_spec(to_spec(declared_ast)), operanda) == expected


@pytest.mark.unit
def test_program_cod_reaches_a_nested_conc() -> None:
    spec = to_spec(_parse("COD = 2s\nMult(Conc(FR1, FR2), FR3)"))
    assert spec["params"]["components"][0]["params"]["cod"] == 2.0


@pytest.mark.unit
def test_program_lh_keeps_the_algorithm_seed() -> None:
    spec = to_spec(_parse(_SEEDED + "LH = 5s\nVI30s"))
    assert spec["type"] == "LimitedHold"
    assert spec["params"]["inner"]["params"]["seed"] == 11


@pytest.mark.unit
def test_program_lh_reaches_each_copy_of_repeat() -> None:
    spec = to_spec(_parse("LH = 2s\nRepeat(3, FI5s)"))
    assert [c["type"] for c in spec["params"]["components"]] == ["LimitedHold"] * 3


@pytest.mark.unit
def test_program_lh_on_a_time_unit_rescale() -> None:
    spec = to_spec(_parse("LH = 500ms\nFI2s"), time_unit_seconds=0.5)
    assert spec["params"]["hold"] == pytest.approx(0.25)


@pytest.mark.unit
def test_program_bo_without_mult_or_mix_is_accepted() -> None:
    ast = _parse("BO = 5s\nFR5")
    assert to_spec(ast) == {"type": "FR", "params": {"n": 5}}
    from_dsl(ast)


# --- Rejected by both, because the runtime has no such behaviour ------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("source", "match"),
    [
        ("RD = 1s\nFR5", "RD"),
        ("RD = 500ms\nConc(VI30s, VI60s)", "RD"),
        ("BO = 5s\nMult(FR2, FR3)", "BO"),
        ("MTS(comparisons=3, consequence=CRF, ITI=5s, correction=true)", "correction"),
        ("GoNoGo(responseWindow=2s, consequence=CRF, ITI=1s, LH=1s)", "LH"),
    ],
)
def test_rejected_by_from_dsl_and_to_spec(source: str, match: str) -> None:
    ast = _parse(source)
    with pytest.raises(ScheduleConfigError, match=match) as from_dsl_error:
        from_dsl(ast)
    with pytest.raises(ScheduleConfigError, match=match) as to_spec_error:
        to_spec(ast)
    assert type(to_spec_error.value) is type(from_dsl_error.value)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("declared", "written"), [("LH = 2s\nDRL5s", "DRL5s LH2s"), ("LH = 2s\nFT5s", "FT5s LH2s")]
)
def test_program_lh_on_a_leaf_the_runtime_cannot_hold(declared: str, written: str) -> None:
    """The runtime holds interval schedules only; the declared LH fails as the written one."""
    declared_ast, written_ast = _parse(declared), _parse(written)
    assert to_spec(declared_ast) == to_spec(written_ast)
    for ast in (declared_ast, written_ast):
        with pytest.raises(ScheduleConfigError, match="LimitedHold"):
            from_dsl(ast)
        with pytest.raises(ScheduleConfigError, match="LimitedHold"):
            build_schedule_from_spec(to_spec(ast))
