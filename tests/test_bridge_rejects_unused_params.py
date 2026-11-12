"""from_dsl refuses combinator parameters the runtime does not read.

Silently dropping a parameter would run a different contingency from the
one written (e.g. ``Mult(FR2, FR3, BO=5s)`` would run without the
blackout), so the bridge raises instead.
"""

from __future__ import annotations

import pytest

pytest.importorskip("contingency_dsl")

from contingency_dsl import parse  # noqa: E402

from contingency.bridge import from_dsl, to_spec  # noqa: E402
from contingency.errors import ScheduleConfigError  # noqa: E402


def _trace(schedule: object, steps: int = 12) -> list[tuple[bool, object]]:
    from contingency import ResponseEvent

    out = []
    for i in range(1, steps + 1):
        now = float(i)
        outcome = schedule.step(now, ResponseEvent(time=now))  # type: ignore[attr-defined]
        out.append((outcome.reinforced, outcome.meta.get("current_component")))
    return out


@pytest.mark.unit
@pytest.mark.parametrize(
    "source",
    [
        "Mult(FR2, FR3, BO=5s)",
        "Mix(FR2, FR3, BO=5s)",
        "Mult(FR2, FR3, Blackout=5s)",
        "Conc(FR1, Mult(FR2, FR3, BO=5s))",
    ],
)
def test_blackout_on_mult_mix_is_rejected(source: str) -> None:
    with pytest.raises(ScheduleConfigError, match=r"BO.*no blackout"):
        from_dsl(parse(source))


@pytest.mark.unit
@pytest.mark.parametrize(
    "source",
    [
        "BO = 5s\nMult(FR2, FR3)",
        "BO = 5s\nMix(FR2, FR3)",
        "BO = 5s\nConc(FR1, Mult(FR2, FR3))",
        "BO = 5s\nConc(FR1, FR2, PUNISH(1)=Mult(FR1, FR2))",
    ],
)
def test_program_level_blackout_is_rejected(source: str) -> None:
    with pytest.raises(ScheduleConfigError, match=r"BO.*no blackout"):
        from_dsl(parse(source))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("BO = 5s\nFR3", "FR"),
        # An expression-level BO=0 overrides the program-level BO.
        ("BO = 5s\nMult(FR2, FR3, BO=0s)", "Multiple"),
        # An unused binding does not receive the program-level BO.
        ("BO = 5s\nlet m = Mix(FR2, FR3)\nFR3", "FR"),
    ],
)
def test_program_level_blackout_that_applies_to_nothing_is_accepted(
    source: str, expected: str
) -> None:
    assert type(from_dsl(parse(source))).__name__ == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    ("source", "name"),
    [
        ("Conc(VI30s, VI60s, COD=2s, BO=5s)", "BO"),
        ("Chain(FR5, FI30s, BO=5s)", "BO"),
        ("Tand(FR5, FI30s, BO=5s)", "BO"),
        ("Alt(FR5, FI30s, BO=5s)", "BO"),
        ("Conj(FR5, FI30s, BO=5s)", "BO"),
        ("Mult(FR2, FR3, COD=2s)", "COD"),
        ("Mix(FR2, FR3, FRCO=3)", "FRCO"),
        ("Chain(FR5, FI30s, COD=2s)", "COD"),
        ("Overlay(FR5, FR10, target=changeover)", "target"),
    ],
)
def test_unused_combinator_parameters_are_rejected(source: str, name: str) -> None:
    with pytest.raises(ScheduleConfigError, match=name):
        from_dsl(parse(source))


@pytest.mark.unit
@pytest.mark.parametrize("combinator", ["Mult", "Mix"])
def test_zero_blackout_is_the_same_as_omitting_it(combinator: str) -> None:
    with_zero = from_dsl(parse(f"{combinator}(FR2, FR3, BO=0s)"))
    without = from_dsl(parse(f"{combinator}(FR2, FR3)"))
    assert _trace(with_zero) == _trace(without)
    assert to_spec(parse(f"{combinator}(FR2, FR3, BO=0s)")) == to_spec(
        parse(f"{combinator}(FR2, FR3)")
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "source",
    [
        "Conc(VI30s, VI60s, COD=2s, FRCO=3)",
        "Interpolate(FR2, FR3, count=3, onset=5s)",
    ],
)
def test_read_parameters_still_pass(source: str) -> None:
    from_dsl(parse(source))
