"""Differential test: the Python implementation against the Rust engine.

Hypothesis draws a spec (``tests/strategies_spec.py``) and a response
sequence, builds the schedule twice — once with the Python classes
(:func:`contingency.conformance.spec.build_schedule_from_spec`, with
``_rust._FORCE_PYTHON = True`` while it builds a node outside every
compound) and once with :func:`contingency.engine.from_spec` — steps both
through the same sequence and compares every step without rounding:

* ``reinforced``, the reinforcer's ``time``/``magnitude`` (as
  ``float.hex()``) and ``label``;
* ``meta`` with ``None`` values dropped, floats as ``float.hex()``, ``bool``
  and ``int`` kept apart;
* exceptions by class name (comparison stops after the first exception);
* a spec both builders reject with the same class counts as agreement.

Known differences are listed in :data:`KNOWN_DIFFERENCES`; a listed entry
that no longer produces a mismatch fails, so the list cannot go stale.

The single, compound and trial-based schedule classes (ratio, interval,
time-based, differential, Lag, adjusting, interlocking, aversive,
ProgressiveRatio over a factory step, every compound, MatchingToSample
and GoNoGo) have no Python implementation and ignore ``_FORCE_PYTHON``:
a spec built only from them runs on the Rust engine on both sides, so it
always agrees. A compound or trial-based schedule takes only components
that run on the Rust engine, so every node inside one is built without
the flag. The comparison checks the class that still has a Python
implementation (Percentile) where it is not inside a compound or a
trial-based schedule. The former Python trajectories of the other
classes are pinned by the recorded traces under
``tests/fixtures/single_schedule_traces/`` and
``tests/fixtures/compound_traces/`` instead.

The test runs once per outermost type (every type in
``contingency_core.SPEC_TYPES``) with ``DIFFERENTIAL_EXAMPLES / 37``
examples each (at least 5; ``DIFFERENTIAL_EXAMPLES`` defaults to 200).
``scripts/differential_report.py`` runs the same comparison without
failing and writes every mismatch as a fixture-shaped record.
"""

from __future__ import annotations

import itertools
import json
import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from unittest import mock

import pytest

pytest.importorskip("contingency_core")

import contingency_core  # type: ignore[import-untyped]  # noqa: E402
from hypothesis import HealthCheck, event, given, settings  # noqa: E402

import contingency.conformance.spec as _spec_module  # noqa: E402
from contingency.engine import from_spec  # noqa: E402
from contingency.entities import ResponseEvent  # noqa: E402
from contingency.schedules import _rust  # noqa: E402

from .strategies_spec import (  # noqa: E402
    SEEDED_TYPES,
    SPEC_TYPES_COVERED,
    Spec,
    Step,
    cases_of_type,
    spec_types_in,
)

pytestmark = pytest.mark.differential

EXAMPLES = int(os.environ.get("DIFFERENTIAL_EXAMPLES", "200"))

Observation = tuple[Any, ...]


# -- running one side ---------------------------------------------------------


def _norm(value: Any) -> Any:
    """JSON-safe form that keeps bool, int and float apart and floats bit-exact."""
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return ["NaN", math.copysign(1.0, value) < 0]
        return ["float", value.hex()]
    if isinstance(value, int):
        return ["int", value]
    if isinstance(value, dict):
        return [
            "dict",
            [[_norm(k), _norm(v)] for k, v in sorted(value.items(), key=lambda kv: repr(kv[0]))],
        ]
    if isinstance(value, list | tuple):
        return [_norm(v) for v in value]
    return ["repr", type(value).__name__, repr(value)]


def observe(outcome: Any) -> Observation:
    """One step's output in a form that compares bit-for-bit."""
    r = outcome.reinforcer
    reinforcer = None if r is None else (_norm(r.time), _norm(r.magnitude), r.label)
    meta = {k: _norm(v) for k, v in dict(outcome.meta).items() if v is not None}
    return ("ok", bool(outcome.reinforced), reinforcer, json.dumps(meta, sort_keys=True))


def run_side(make: Callable[[], Any], steps: list[Step]) -> list[Observation]:
    """Build with ``make`` and step through ``steps``; stop at the first exception."""
    try:
        schedule = make()
    except Exception as exc:  # noqa: BLE001 - the class name is the observation
        return [("build_error", type(exc).__name__)]
    out: list[Observation] = []
    for now, operandum, reset in steps:
        try:
            if reset:
                schedule.reset()
            event_ = None if operandum is None else ResponseEvent(now, operandum)
            out.append(observe(schedule.step(now, event_)))
        except Exception as exc:  # noqa: BLE001
            out.append(("step_error", type(exc).__name__))
            break
    return out


# Types whose class runs only on the Rust engine and takes only components
# that run there (the compounds and the trial-based schedules).
COMPOUND_TYPES = frozenset(
    {
        "Alternative",
        "Chained",
        "Concurrent",
        "Conjunctive",
        "GoNoGo",
        "Interpolate",
        "LimitedHold",
        "MatchingToSample",
        "Mixed",
        "Multiple",
        "Overlay",
        "ResponseCost",
        "SecondOrder",
        "Tandem",
        "Timeout",
    }
)

_BUILD = _spec_module.build_schedule_from_spec


def build_python(spec: Spec) -> Any:
    """Build ``spec`` with the Python implementations where they still run.

    ``_FORCE_PYTHON`` is set while a node outside every compound is built
    and cleared inside a compound, whose components must run on the Rust
    engine. The flag only matters while building.
    """
    in_compound = False

    def build(node: Spec) -> Any:
        nonlocal in_compound
        outer, previous = in_compound, _rust._FORCE_PYTHON
        in_compound = in_compound or node.get("type") in COMPOUND_TYPES
        _rust._FORCE_PYTHON = not in_compound
        try:
            return _BUILD(node)
        finally:
            in_compound, _rust._FORCE_PYTHON = outer, previous

    with mock.patch.object(_spec_module, "build_schedule_from_spec", build):
        return build(spec)


def run_python(spec: Spec, steps: list[Step]) -> list[Observation]:
    """Steps the Python classes where they still run (see :func:`build_python`)."""
    return run_side(lambda: build_python(spec), steps)


def run_rust(spec: Spec, steps: list[Step]) -> list[Observation]:
    return run_side(lambda: from_spec(spec), steps)


# -- comparing ------------------------------------------------------------------


@dataclass(frozen=True)
class Difference:
    index: int
    item: str
    python: Observation | None
    rust: Observation | None


def first_difference(a: list[Observation], b: list[Observation]) -> Difference | None:
    """The first step where the two sides differ and which item differs.

    Both sides get the same steps and stop at their first exception, so a
    length difference always shows up first as a differing element.
    """
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x == y:
            continue
        if x[0] != "ok" or y[0] != "ok":
            return Difference(i, "error", x, y)
        if x[1] != y[1]:
            return Difference(i, "reinforced", x, y)
        if x[2] != y[2]:
            return Difference(i, "reinforcer", x, y)
        mx, my = json.loads(x[3]), json.loads(y[3])
        keys = sorted(k for k in set(mx) | set(my) if mx.get(k) != my.get(k))
        return Difference(i, "meta:" + ",".join(keys), x, y)
    assert len(a) == len(b), "unreachable: equal prefixes of unequal length"
    return None


def has_nan(observations: list[Observation]) -> bool:
    """True if any reinforcer value or meta value is NaN (at any depth)."""

    def walk(node: Any) -> bool:
        if isinstance(node, list | tuple):
            if len(node) == 2 and node[0] == "NaN":
                return True
            return any(walk(v) for v in node)
        if isinstance(node, dict):
            return any(walk(v) for v in node.values())
        return False

    return any(o[0] == "ok" and (walk(o[2]) or walk(json.loads(o[3]))) for o in observations)


# -- known differences ----------------------------------------------------------
#
# Each entry names a correction that turns the Python output into the Rust
# output for that one cause: rerun Python with the spec rewritten or a
# helper patched, or rewrite its output. A mismatch counts as known only
# if some combination of the corrections that apply to the spec makes the
# two sides agree, so an unrelated difference in the same case still fails.


@dataclass(frozen=True)
class Known:
    key: str
    reason: str
    fix: str
    applies: Callable[[Spec], bool]
    probe: tuple[Spec, list[Step]]


KNOWN_DIFFERENCES: tuple[Known, ...] = ()


def known_for(
    spec: Spec, steps: list[Step], py: list[Observation], rs: list[Observation]
) -> tuple[Known, ...] | None:
    """The smallest set of known entries that accounts for the whole mismatch, if any."""
    applicable = [k for k in KNOWN_DIFFERENCES if k.applies(spec)]
    for size in range(1, len(applicable) + 1):
        for combo in itertools.combinations(applicable, size):
            if first_difference(run_python(spec, steps), rs) is None:
                return combo
    return None


# -- the tests ------------------------------------------------------------------


def test_strategies_cover_every_spec_type() -> None:
    assert frozenset(contingency_core.SPEC_TYPES) == SPEC_TYPES_COVERED


def test_seeded_types_are_spec_types() -> None:
    assert SEEDED_TYPES <= SPEC_TYPES_COVERED


TYPES = sorted(SPEC_TYPES_COVERED)
PER_TYPE = max(5, EXAMPLES // len(TYPES))


def check_case(spec: Spec, steps: list[Step]) -> None:
    """Fail unless both sides agree (or the spec is on the known list)."""
    py = run_python(spec, steps)
    rs = run_rust(spec, steps)

    event(f"type={spec['type']}")
    event("reinforced" if any(o[0] == "ok" and o[1] for o in py) else "no reinforcement")
    if any(o[0] != "ok" for o in py + rs):
        event("exception")

    assert not has_nan(py + rs), f"NaN in output for {json.dumps(spec)}"
    diff = first_difference(py, rs)
    if diff is None:
        return
    if known_for(spec, steps, py, rs) is not None:
        return
    types = ",".join(spec_types_in(spec))
    at = "(build)" if diff.python and diff.python[0] == "build_error" else steps[diff.index]
    raise AssertionError(
        f"Python and Rust differ at step {diff.index} ({diff.item}); types {types}\n"
        f"spec={json.dumps(spec)}\n"
        f"step={at}\n"
        f"python={diff.python}\nrust={diff.rust}"
    )


@pytest.mark.parametrize("outer", TYPES)
def test_python_and_rust_agree(outer: str) -> None:
    @settings(
        max_examples=PER_TYPE,
        deadline=None,
        database=None,
        derandomize=bool(os.environ.get("CI")),
        suppress_health_check=[
            HealthCheck.too_slow,
            HealthCheck.filter_too_much,
            HealthCheck.data_too_large,
        ],
    )
    @given(cases_of_type(outer))
    def run(case: tuple[Spec, list[Step]]) -> None:
        check_case(*case)

    run()


def test_known_differences_still_occur() -> None:
    """Each entry's probe still mismatches and is explained by that entry (or remove it)."""
    stale = []
    for known in KNOWN_DIFFERENCES:
        spec, steps = known.probe
        py, rs = run_python(spec, steps), run_rust(spec, steps)
        explained = known_for(spec, steps, py, rs)
        if first_difference(py, rs) is None or explained is None or known not in explained:
            stale.append(known.key)
    assert not stale, f"known differences no longer occur, remove them: {stale}"


def test_nan_is_detected_in_nested_meta() -> None:
    obs = observe(
        type("O", (), {"reinforcer": None, "reinforced": False, "meta": {"x": {"y": math.nan}}})()
    )
    assert has_nan([obs])


def test_known_list_does_not_hide_an_unrelated_difference() -> None:
    """A geometric PR next to an unrelated mismatch is not explained by the PR entry."""
    spec = {"type": "ProgressiveRatio", "params": {"step": "geometric", "start": 2, "ratio": 1.5}}
    steps: list[Step] = [(float(i + 1), "main", False) for i in range(40)]
    py, rs = run_python(spec, steps), run_rust(spec, steps)
    tampered = rs[:-1] + [("ok", True, None, "{}")]
    assert known_for(spec, steps, py, tampered) is None


def test_both_reject_a_spec_with_the_same_error() -> None:
    spec = {
        "type": "LimitedHold",
        "params": {"inner": {"type": "FR", "params": {"n": 2}}, "hold": 1.0},
    }
    steps: list[Step] = [(1.0, "main", False)]
    py, rs = run_python(spec, steps), run_rust(spec, steps)
    assert py[0][0] in ("build_error", "step_error")
    assert first_difference(py, rs) is None


def test_backwards_time_is_rejected_by_both() -> None:
    spec = {"type": "FR", "params": {"n": 3}}
    steps: list[Step] = [(2.0, "main", False), (1.0, "main", False)]
    py, rs = run_python(spec, steps), run_rust(spec, steps)
    assert py[-1] == ("step_error", "ScheduleStateError")
    assert first_difference(py, rs) is None
