"""Differential test: the Python implementation against the Rust engine.

Hypothesis draws a spec (``tests/strategies_spec.py``) and a response
sequence, builds the schedule twice — once with the Python classes
(``_rust._FORCE_PYTHON = True`` around
:func:`contingency.conformance.spec.build_schedule_from_spec`) and once
with :func:`contingency.engine.from_spec` — steps both through the same
sequence and compares every step without rounding:

* ``reinforced``, the reinforcer's ``time``/``magnitude`` (as
  ``float.hex()``) and ``label``;
* ``meta`` with ``None`` values dropped, floats as ``float.hex()``, ``bool``
  and ``int`` kept apart;
* exceptions by class name (comparison stops after the first exception);
* a spec both builders reject with the same class counts as agreement.

Known differences are listed in :data:`KNOWN_DIFFERENCES`; a listed entry
that no longer produces a mismatch fails, so the list cannot go stale.

The single schedule classes (ratio, interval, time-based, differential,
Lag, adjusting, interlocking, aversive) have no Python implementation and
ignore ``_FORCE_PYTHON``: a spec built only from them runs on the Rust
engine on both sides, so it always agrees. The comparison checks the
classes that still have a Python implementation (compounds over them,
trial-based schedules, ProgressiveRatio and Percentile). Their former
Python trajectories are pinned by the recorded traces under
``tests/fixtures/single_schedule_traces/`` instead.

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

from contingency.conformance.spec import build_schedule_from_spec  # noqa: E402
from contingency.engine import from_spec  # noqa: E402
from contingency.entities import ResponseEvent  # noqa: E402
from contingency.schedules import _rust  # noqa: E402
from contingency.schedules import progressive as _progressive  # noqa: E402

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


def run_python(spec: Spec, steps: list[Step]) -> list[Observation]:
    """Steps the Python classes. The flag only matters while building."""
    previous = _rust._FORCE_PYTHON
    _rust._FORCE_PYTHON = True
    try:
        return run_side(lambda: build_schedule_from_spec(spec), steps)
    finally:
        _rust._FORCE_PYTHON = previous


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


def _contains(spec: Spec, pred: Callable[[dict[str, Any]], bool]) -> bool:
    def walk(node: Any) -> bool:
        if isinstance(node, dict):
            if "type" in node and "params" in node and pred(node):
                return True
            return any(walk(v) for v in node.values())
        if isinstance(node, list):
            return any(walk(v) for v in node)
        return False

    return walk(spec)


def _map_specs(node: Any, fn: Callable[[dict[str, Any]], dict[str, Any]]) -> Any:
    """Apply ``fn`` to every spec node, innermost first; returns a new structure."""
    if isinstance(node, dict):
        out = {k: _map_specs(v, fn) for k, v in node.items()}
        return fn(out) if "type" in out and "params" in out else out
    if isinstance(node, list):
        return [_map_specs(v, fn) for v in node]
    return node


def _sort_concurrent(node: dict[str, Any]) -> dict[str, Any]:
    comps = node["params"].get("components")
    if node["type"] != "Concurrent" or not isinstance(comps, dict):
        return node
    return {
        **node,
        "params": {**node["params"], "components": {k: comps[k] for k in sorted(comps)}},
    }


def _rank_zero_to_one(node: dict[str, Any]) -> dict[str, Any]:
    if node["type"] == "Percentile" and node["params"].get("rank") == 0:
        return {**node, "params": {**node["params"], "rank": 1}}
    return node


def _round_half_away(x: float) -> int:
    return int(math.copysign(math.floor(abs(x) + 0.5), x))


def _drop_inner_meta(observations: list[Observation]) -> list[Observation]:
    out: list[Observation] = []
    for o in observations:
        if o[0] == "ok":
            meta = json.loads(o[3])
            meta.pop("inner_meta", None)
            o = (o[0], o[1], o[2], json.dumps(meta, sort_keys=True))
        out.append(o)
    return out


def _has_geometric(spec: Spec) -> bool:
    return _contains(
        spec, lambda n: n["type"] == "ProgressiveRatio" and n["params"].get("step") == "geometric"
    )


def _keys_unsorted(spec: Spec) -> bool:
    return json.dumps(_map_specs(spec, _sort_concurrent)) != json.dumps(spec)


def _has_trial(spec: Spec) -> bool:
    return _contains(spec, lambda n: n["type"] in ("MatchingToSample", "GoNoGo"))


def _python_corrected(spec: Spec, steps: list[Step], keys: frozenset[str]) -> list[Observation]:
    if "concurrent-key-order" in keys:
        spec = _map_specs(spec, _sort_concurrent)
    if "pr-geometric-rounding" in keys:
        # ``geometric`` calls the builtin ``round`` through the module globals.
        with mock.patch.object(_progressive, "round", _round_half_away, create=True):
            py = run_python(spec, steps)
    else:
        py = run_python(spec, steps)
    if "trial-inner-meta" in keys:
        py = _drop_inner_meta(py)
    return py


def _rank_zero_explains(spec: Spec, rs: list[Observation]) -> bool:
    """Rust refused to build only because of a Percentile with rank 0."""
    if rs[:1] != [("build_error", "ScheduleConfigError")]:
        return False
    if not _contains(spec, lambda n: n["type"] == "Percentile" and n["params"].get("rank") == 0):
        return False
    return run_rust(_map_specs(spec, _rank_zero_to_one), [])[:1] != rs[:1]


KNOWN_DIFFERENCES: tuple[Known, ...] = (
    Known(
        key="pr-geometric-rounding",
        reason="geometric PR: Rust rounds .5 away from zero, Python rounds half to even",
        fix="contingency-rs: round geometric PR requirements half to even",
        applies=_has_geometric,
        probe=(
            {"type": "ProgressiveRatio", "params": {"step": "geometric", "start": 2, "ratio": 1.5}},
            [(float(i + 1), "main", False) for i in range(40)],
        ),
    ),
    Known(
        key="percentile-rank-zero",
        reason=(
            "Percentile rank 0: Python accepts it, the Rust engine rejects it (rank must be 1-100)"
        ),
        fix="contingency-rs: accept rank 0 (the DSL allows 0-100)",
        applies=lambda spec: False,  # checked by _rank_zero_explains (a build refusal)
        probe=(
            {"type": "Percentile", "params": {"target": "irt", "rank": 0, "window": 2}},
            [(1.0, "main", False)],
        ),
    ),
    Known(
        key="concurrent-key-order",
        reason=(
            "Concurrent: the Rust engine's from_spec visits components in sorted key order "
            "(its JSON object does not keep insertion order), so when two components fire on the "
            "same tick a different one wins; Python and the Rust docs say insertion order"
        ),
        fix="contingency-rs: keep the key order of Concurrent components in from_spec",
        applies=_keys_unsorted,
        probe=(
            {
                "type": "Concurrent",
                "params": {
                    "components": {
                        "right": {"type": "FT", "params": {"interval": 0.5}},
                        "center": {
                            "type": "Sidman",
                            "params": {"shock_shock_interval": 0.5, "response_shock_interval": 0.5},
                        },
                    }
                },
            },
            [(0.5, None, False), (1.0, None, False)],
        ),
    ),
    Known(
        key="trial-inner-meta",
        reason=(
            "MatchingToSample/GoNoGo: Python copies the consequence schedule's meta "
            "to meta['inner_meta'], the Rust engine drops it (its meta values cannot nest)"
        ),
        fix="undecided: nest meta in the Rust engine, or drop inner_meta from Python",
        applies=_has_trial,
        probe=(
            {
                "type": "MatchingToSample",
                "params": {
                    "n_comparisons": 2,
                    "seed": 0,
                    "consequence": {
                        "type": "DiscriminatedAvoidance",
                        "params": {"warning_duration": 0.5, "iti": 0.5},
                    },
                },
            },
            [(1.0, "choice_1", False)],
        ),
    ),
)

_BY_KEY = {k.key: k for k in KNOWN_DIFFERENCES}


def known_for(
    spec: Spec, steps: list[Step], py: list[Observation], rs: list[Observation]
) -> tuple[Known, ...] | None:
    """The smallest set of known entries that accounts for the whole mismatch, if any."""
    if _rank_zero_explains(spec, rs):
        return (_BY_KEY["percentile-rank-zero"],)
    applicable = [k for k in KNOWN_DIFFERENCES if k.applies(spec)]
    for size in range(1, len(applicable) + 1):
        for combo in itertools.combinations(applicable, size):
            keys = frozenset(k.key for k in combo)
            if first_difference(_python_corrected(spec, steps, keys), rs) is None:
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
