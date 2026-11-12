"""``from_dsl`` engine selection.

The engine name is validated, and ``to_spec`` runs, before the Rust
extension module is imported, so the first checks run without it. The
checks that need ``contingency_core`` are skipped when it is missing.
"""

from __future__ import annotations

import pytest
from contingency_dsl import parse

from contingency import engine as _engine
from contingency import from_dsl
from contingency import schedules as _schedules
from contingency.errors import ScheduleConfigError, SpecUnrepresentableError


@pytest.mark.unit
def test_from_dsl_rejects_unknown_engine() -> None:
    with pytest.raises(ScheduleConfigError, match="unknown engine"):
        from_dsl(parse("FR5"), engine="swift")  # type: ignore[arg-type]


@pytest.mark.unit
def test_from_dsl_rust_engine_reports_unrepresentable_programs() -> None:
    program = parse("Pctl(latency, 50)")
    with pytest.raises(SpecUnrepresentableError):
        from_dsl(program, engine="rust")


# --- The default engine -----------------------------------------------------
#
# ``from_dsl`` keeps ``engine="python"`` as its default: it returns the
# schedule classes of :mod:`contingency.schedules`, whose ``step`` / ``reset``
# run on contingency-rs when ``contingency_core`` is installed, and on the
# Python implementation otherwise. ``engine="rust"`` returns the bare
# :class:`contingency.engine.RustSchedule` built from ``to_spec``.

_DEFAULT_PROGRAMS = [
    ("FR5", _schedules.FR),
    ("VI30s", _schedules.VI),
    ("DRL5s", _schedules.DRL),
    ("Conc(VI30s, VI60s, COD=2s)", _schedules.Concurrent),
    ("Mult(FR5, EXT)", _schedules.Multiple),
    ("Chain(FR5, FI30s)", _schedules.Chained),
    ("Pctl(IRT, 50)", _schedules.Percentile),
]

_needs_core = pytest.mark.skipif(
    not _engine.rust_engine_available(), reason="contingency_core is not installed"
)


@pytest.mark.unit
@pytest.mark.parametrize(("source", "cls"), _DEFAULT_PROGRAMS)
def test_default_engine_raises_without_the_extension(
    source: str, cls: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    program = parse(source)
    monkeypatch.setattr(_engine, "rust_engine_available", lambda: False)
    with pytest.raises(ScheduleConfigError, match="contingency_core"):
        from_dsl(program)


@_needs_core
@pytest.mark.integration
@pytest.mark.parametrize(("source", "cls"), _DEFAULT_PROGRAMS)
def test_default_engine_runs_the_class_on_rust_with_the_extension(
    source: str, cls: type
) -> None:
    schedule = from_dsl(parse(source))
    assert isinstance(schedule, cls)
    assert schedule.engine == "rust"


@_needs_core
@pytest.mark.integration
@pytest.mark.parametrize(("source", "cls"), _DEFAULT_PROGRAMS)
def test_rust_engine_returns_the_bare_rust_schedule(source: str, cls: type) -> None:
    schedule = from_dsl(parse(source), engine="rust")
    assert type(schedule) is _engine.RustSchedule


@_needs_core
@pytest.mark.integration
def test_default_engine_keeps_a_class_rust_cannot_run_in_python() -> None:
    # The geometric progressive ratio rounds differently in contingency-rs,
    # so the class keeps its Python implementation even with the module.
    schedule = from_dsl(parse("PR(geometric, start=1, ratio=2)"))
    assert isinstance(schedule, _schedules.ProgressiveRatio)
    assert schedule.engine == "python"
