"""The schedule classes need ``contingency_core``; importing and parsing do not.

Without the extension module, or with one too old to report
``Schedule.state()``, building a schedule class raises
``ScheduleConfigError`` that says how to install it. ``import contingency``,
the DSL bridge's parsing and ``to_spec`` keep working. A spec the Rust
engine rejects raises instead of running in Python.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

import pytest

import contingency.engine as engine
import contingency.schedules._rust as _rust
from contingency.errors import ScheduleConfigError
from contingency.schedules import (
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
    Alternative,
    Concurrent,
    LimitedHold,
    MatchingToSample,
    Multiple,
    ProgressiveRatio,
    Tandem,
    Timeout,
    arithmetic,
)

pytestmark = pytest.mark.unit

_needs_core = pytest.mark.skipif(
    not engine.rust_engine_available(), reason="contingency_core is not installed"
)

FACTORIES: list[tuple[str, Callable[[], Any]]] = [
    ("FR", lambda: FR(5)),
    ("VR", lambda: VR(5, seed=1)),
    ("RR", lambda: RR(0.2, seed=1)),
    ("RR rng", lambda: RR(0.2, rng=random.Random(1))),
    ("FI", lambda: FI(5.0)),
    ("VI", lambda: VI(5.0, seed=1)),
    ("RI", lambda: RI(5.0, rng=random.Random(1))),
    ("FT", lambda: FT(5.0)),
    ("VT", lambda: VT(5.0, seed=1)),
    ("RT", lambda: RT(5.0, rng=random.Random(1))),
    ("EXT", EXT),
    ("DRL", lambda: DRL(5.0)),
    ("DRO", lambda: DRO(5.0)),
    ("PR custom step", lambda: ProgressiveRatio(lambda i: i + 1)),
    ("PR arithmetic", lambda: ProgressiveRatio(arithmetic(1, 2))),
    ("MTS", lambda: MatchingToSample(n_comparisons=2, seed=1)),
]

IDS = [f[0] for f in FACTORIES]


def _without_core(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine, "rust_engine_available", lambda: False)


@pytest.mark.parametrize("factory", [f[1] for f in FACTORIES], ids=IDS)
def test_building_a_schedule_without_the_extension_raises(
    factory: Callable[[], Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _without_core(monkeypatch)
    with pytest.raises(ScheduleConfigError, match="contingency_core"):
        factory()


def test_the_message_says_how_to_install_it(monkeypatch: pytest.MonkeyPatch) -> None:
    _without_core(monkeypatch)
    with pytest.raises(ScheduleConfigError) as info:
        FR(5)
    message = str(info.value)
    assert "maturin" in message
    assert "core/" in message


def test_import_and_parsing_work_without_the_extension(monkeypatch: pytest.MonkeyPatch) -> None:
    _without_core(monkeypatch)
    from contingency_dsl import parse

    import contingency

    ast = parse("Conc(VI30s, FR5, COD=2s)")
    assert ast is not None
    assert contingency.ScheduleConfigError is ScheduleConfigError
    with pytest.raises(ScheduleConfigError, match="contingency_core"):
        contingency.from_dsl(ast)


@_needs_core
def test_compounds_raise_when_their_components_were_built_before(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Components built with the extension, compound built without it.
    a, b = FR(2), FR(3)
    _without_core(monkeypatch)
    for build in (
        lambda: Alternative(a, b),
        lambda: Multiple([a, b]),
        lambda: Tandem([a, b]),
        lambda: Timeout(a, 1.0),
        lambda: Concurrent({"left": a, "right": b}),
    ):
        with pytest.raises(ScheduleConfigError, match="contingency_core"):
            build()


@_needs_core
def test_limited_hold_raises_without_the_extension(monkeypatch: pytest.MonkeyPatch) -> None:
    inner = FI(5.0)
    _without_core(monkeypatch)
    with pytest.raises(ScheduleConfigError, match="contingency_core"):
        LimitedHold(inner, 1.0)


@_needs_core
@pytest.mark.parametrize("factory", [f[1] for f in FACTORIES], ids=IDS)
def test_an_extension_without_state_raises(
    factory: Callable[[], Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_rust, "_core_has_state", lambda: False)
    with pytest.raises(ScheduleConfigError, match="too old"):
        factory()


@_needs_core
def test_a_spec_the_rust_engine_rejects_raises_instead_of_running_in_python() -> None:
    # Beyond the Rust integer width: the Python class used to take it.
    with pytest.raises(ScheduleConfigError, match="Rust engine"):
        FR(2**70)


@_needs_core
def test_with_the_extension_every_factory_runs_on_rust_or_in_python_as_before() -> None:
    # The Python branches stay for now: a custom PR step function and a
    # caller-supplied rng still run in Python when the extension is present.
    engines = {name: factory().engine for name, factory in FACTORIES}
    assert engines.pop("PR custom step") == "python"
    for name in ("RR rng", "RI", "RT"):
        assert engines.pop(name) == "python"
    assert set(engines.values()) == {"rust"}


def test_a_process_where_the_import_fails_raises_on_construction() -> None:
    # No monkeypatch: ``import contingency_core`` itself raises ImportError.
    import subprocess
    import sys

    code = (
        "import sys\n"
        "sys.modules['contingency_core'] = None\n"
        "import contingency\n"
        "from contingency.schedules import FR\n"
        "from contingency.errors import ScheduleConfigError\n"
        "from contingency_dsl import parse\n"
        "parse('FR5')\n"
        "try:\n"
        "    FR(5)\n"
        "except ScheduleConfigError as exc:\n"
        "    assert 'contingency_core' in str(exc), exc\n"
        "    print('raised')\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "raised"
