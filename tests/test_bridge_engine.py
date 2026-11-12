"""``from_dsl`` engine selection checks that do not need ``contingency_core``.

The engine name is validated, and ``to_spec`` runs, before the Rust
extension module is imported, so these tests run without it.
"""

from __future__ import annotations

import pytest
from contingency_dsl import parse

from contingency import from_dsl
from contingency.errors import ScheduleConfigError, SpecUnrepresentableError


@pytest.mark.unit
def test_from_dsl_rejects_unknown_engine() -> None:
    with pytest.raises(ScheduleConfigError, match="unknown engine"):
        from_dsl(parse("FR5"), engine="swift")  # type: ignore[arg-type]


@pytest.mark.unit
def test_from_dsl_rust_engine_reports_unrepresentable_programs() -> None:
    program = parse("Mult(FR3, FR5, BO=5s)")
    with pytest.raises(SpecUnrepresentableError):
        from_dsl(program, engine="rust")
