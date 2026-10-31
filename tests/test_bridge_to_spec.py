"""Tests for ``to_spec``: DSL AST -> schedule spec.

For every AST that ``to_spec`` accepts, the schedule built from the spec
by ``build_schedule_from_spec`` must behave like ``from_dsl(ast)``. The
equivalence tests drive both schedules with the same pseudo-random
stream of ticks and responses and compare every outcome. Constructs
that the spec cannot express must raise ``SpecUnrepresentableError``.
"""

from __future__ import annotations

import json
import random
from typing import Any

import pytest
from contingency_dsl import analyze, parse
from contingency_dsl.annotations.temporal.extension import TemporalExtension
from contingency_dsl.extensions import ExtensionRegistry

from contingency import from_dsl, to_spec
from contingency.conformance import compare, is_stochastic, iter_fixtures
from contingency.conformance.replay import observe, replay
from contingency.conformance.spec import build_schedule_from_spec
from contingency.entities import ResponseEvent
from contingency.errors import ScheduleConfigError, SpecUnrepresentableError
from contingency.interfaces import Schedule

_REGISTRY = ExtensionRegistry.build(TemporalExtension())
_SEEDED = '@algorithm("fleshler-hoffman", seed=11)\n'


def _parse(source: str) -> Any:
    return parse(source, extensions=_REGISTRY)


def _operanda(spec: Any) -> list[str]:
    """Operandum names a spec responds to (``main`` outside Concurrent)."""
    found: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            stype = node.get("type")
            params = node.get("params", {})
            if stype == "Concurrent":
                found.extend(params["components"])
            if stype == "MatchingToSample":
                found.extend(f"choice_{i}" for i in range(params["n_comparisons"]))
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(spec)
    if not found or spec["type"] != "Concurrent":
        found.insert(0, "main")
    return list(dict.fromkeys(found))


def _trace(
    schedule: Schedule,
    operanda: list[str],
    *,
    n_steps: int = 600,
    dt: float = 0.25,
    seed: int = 1234,
) -> list[Any]:
    """Step ``schedule`` through a fixed stream; record each outcome.

    The stream alternates 10 s of frequent responding with 10 s of
    almost none, so that pausing schedules (DRO, Sidman) also act.
    """
    rng = random.Random(seed)
    out: list[Any] = []
    for i in range(n_steps):
        now = i * dt
        p_response = 0.7 if int(now // 10) % 2 == 0 else 0.02
        event = (
            ResponseEvent(time=now, operandum=rng.choice(operanda))
            if rng.random() < p_response
            else None
        )
        try:
            out.append(observe(schedule.step(now, event)))
        except Exception as exc:  # the same failure must occur in both
            out.append(("error", type(exc).__name__, str(exc)))
            break
    return out


def _assert_whole_numbers_are_ints(spec: Any) -> None:
    """JSON integers for the whole-number params (Rust reads them as u64)."""
    int_keys = {
        "n",
        "n_intervals",
        "seed",
        "frco",
        "rank",
        "window",
        "initial_ratio",
        "n_comparisons",
        "response_count",
    }

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            params = node.get("params", {})
            for key in int_keys & set(params):
                assert type(params[key]) is int, (node["type"], key, params[key])
            if node.get("type") == "ProgressiveRatio":
                for key in ("start", "step_size"):
                    if key in params:
                        assert type(params[key]) is int, (key, params[key])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(spec)


_STOCHASTIC_TYPES = frozenset({"VR", "VI", "VT", "RR", "RI", "RT", "MatchingToSample", "GoNoGo"})


def _stochastic_types(spec: Any) -> set[str]:
    if isinstance(spec, list):
        return set().union(*(_stochastic_types(v) for v in spec))
    if not isinstance(spec, dict):
        return set()
    found = {spec["type"]} & _STOCHASTIC_TYPES if "type" in spec else set()
    return found.union(*(_stochastic_types(v) for v in spec.values()))


def _assert_equivalent(ast: Any, **kwargs: Any) -> dict[str, Any]:
    spec = to_spec(ast, **kwargs)
    # The spec is JSON: round-trip it the way a fixture or the Rust
    # engine would receive it.
    spec = json.loads(json.dumps(spec))
    _assert_whole_numbers_are_ints(spec)
    if _stochastic_types(spec) and not is_stochastic(spec):
        # Unseeded: both runtimes draw fresh random values, so only the
        # seeded variant of the source can be compared step by step.
        return spec
    operanda = _operanda(spec)
    expected = _trace(from_dsl(ast, **kwargs), operanda)
    observed = _trace(build_schedule_from_spec(spec), operanda)
    assert observed == expected, json.dumps(spec)
    assert any(step != {"reinforced": False} for step in expected if isinstance(step, dict)) or (
        spec["type"] == "EXT"
    ), f"stream never exercised {json.dumps(spec)}"
    return spec


# --- Equivalence with from_dsl ----------------------------------------------

# Every DSL source used by test_bridge.py, plus forms that it builds from
# hand-made AST nodes.
_EQUIVALENT_SOURCES = [
    # atomic
    "FR5",
    "FR1",
    "FI30s",
    "FI500ms",
    "FI2min",
    "FT15s",
    "VR10",
    "VI30s",
    "VI60s",
    "VT45s",
    "RR20",
    "RI30s",
    "RT20s",
    "EXT",
    "CRF",
    # wrappers
    "FI30s LH10s",
    "FI5s LH2s",
    "FR5 TO(duration=10s, reset_on_response=false)",
    "FR5 TO(duration=3s, reset_on_response=true)",
    "FR5 TO(duration=500ms, reset_on_response=false)",
    "FR1 TO(duration=5s, reset_on_response=false)",
    "FR5 ResponseCost(amount=2)",
    "FR1 ResponseCost(amount=1)",
    # compound
    "Conc(FR1, FR1)",
    "Conc(FR5, VI30s)",
    "Conc(VI30s, VI60s, COD=1s)",
    "Conc(VI30s, VI60s, COD=2s, FRCO=5)",
    "Conc(VI10s, VI20s, VI30s, COD=500ms)",
    "Alt(FR3, FT5s)",
    "Alt(FR3, FR5, FR7)",
    "Conj(FR3, FI5s)",
    "Conj(FR3, FI5s, FR2)",
    "Mult(FR3, FR5)",
    "Mix(FR5, FR10)",
    "Mix(FR3, FR5, FR7)",
    "Chain(FR3, FR5)",
    "Chain(FR2, FI5s)",
    "Tand(FR3, FR5)",
    "Overlay(FR5, FR10)",
    "Interp(FR5, FR10)",
    "Interpolate(FR5, FR10, count=5, onset=60s)",
    "Interpolate(FR2, EXT, count=3, onset=10s)",
    "FR5(FI30s)",
    "FR2(FR3)",
    # modifiers
    "DRL10s",
    "DRL2s",
    "DRH2s",
    "DRO5s",
    "Repeat(3, FR5)",
    "PR(linear, start=1, increment=2)",
    "PR(geometric, start=1, ratio=2)",
    "PR(hodos)",
    "PR(exponential)",
    "PR 5",
    "Pctl(IRT, 50, window=20, dir=above)",
    "Pctl(IRT, 50)",
    "Pctl(IRT, 75, window=3, dir=above)",
    # stateful
    "Adj(ratio, start=2, step=1, max=3)",
    "Adj(ratio, start=5, step=2, min=1, max=20)",
    "Interlock(R0=4, T=8s)",
    # aversive
    "Sidman(SSI=5s, RSI=3s)",
    "Sidman(SSI=500ms, RSI=300ms)",
    "DiscrimAv(CSUSInterval=3s, ITI=4s, mode=escape)",
    "Escape(SafeDuration=2s)",
    "Escape(SafeDuration=2s, MaxShock=3s)",
    # trial-based
    "MTS(comparisons=3, consequence=CRF, ITI=5s)",
    "MTS(comparisons=3, consequence=CRF, ITI=500ms)",
    "MTS(comparisons=4, consequence=CRF, incorrect=FT10s, ITI=5s)",
    "MTS(comparisons=2, consequence=VR3, ITI=1s, delay=1s, LH=3s)",
    "GoNoGo(responseWindow=2s, consequence=CRF, ITI=1s)",
    "GoNoGo(responseWindow=2s, consequence=CRF, ITI=1s, falseAlarm=FT10s)",
]


@pytest.mark.unit
@pytest.mark.parametrize("source", _EQUIVALENT_SOURCES)
def test_to_spec_matches_from_dsl(source: str) -> None:
    spec = _assert_equivalent(_parse(source))
    assert not is_stochastic(spec), "no @algorithm(seed=...) means no seed in the spec"


@pytest.mark.unit
@pytest.mark.parametrize("source", _EQUIVALENT_SOURCES)
def test_to_spec_matches_from_dsl_with_algorithm_seed(source: str) -> None:
    spec = _assert_equivalent(_parse(_SEEDED + source))
    assert is_stochastic(spec) == bool(_stochastic_types(spec))


@pytest.mark.unit
@pytest.mark.parametrize(
    "source",
    [
        '@algorithm("fleshler-hoffman", n=20, seed=3)\nVI30s',
        '@algorithm("fleshler-hoffman", seed=5)\nConc(VI30s, RI30s, VR5, COD=1s)',
        '@algorithm("fleshler-hoffman", seed=8)\n'
        "MTS(comparisons=3, consequence=VR2, incorrect=RT5s, ITI=1s)",
        '@algorithm("fleshler-hoffman", seed=2)\n'
        "GoNoGo(responseWindow=2s, consequence=RR2, ITI=1s, falseAlarm=VT3s)",
    ],
)
def test_to_spec_assigns_seeds_in_from_dsl_order(source: str) -> None:
    _assert_equivalent(_parse(source))


@pytest.mark.unit
def test_to_spec_matches_from_dsl_for_hand_built_nodes() -> None:
    """Defaults the parser always fills in, left unset by hand."""
    from contingency_dsl.ast import Modifier, Special, TrialBased

    pctl = Modifier(modifier="Pctl", pctl_target="IRT", pctl_rank=50)
    assert to_spec(pctl)["params"]["direction"] == "below"
    _assert_equivalent(pctl)

    mts = TrialBased(
        trial_type="MTS", params={"comparisons": 2, "consequence": Special(kind="CRF")}
    )
    assert to_spec(mts)["params"]["iti"] == 0.0
    gonogo = TrialBased(
        trial_type="GoNoGo",
        params={"responseWindow": 2.0, "consequence": Special(kind="CRF")},
    )
    assert to_spec(gonogo)["params"]["iti"] == 0.0
    for node in (mts, gonogo):
        program = _parse(_SEEDED + "EXT")
        _assert_equivalent(type(program)(program.program_annotations, schedule=node))


@pytest.mark.unit
def test_to_spec_after_analyze() -> None:
    program = analyze(_parse("LH = 2s\nRepeat(3, FI5s)")).program
    spec = _assert_equivalent(program)
    assert spec["type"] == "Tandem"
    assert spec["params"]["components"][0] == {
        "type": "LimitedHold",
        "params": {"inner": {"type": "FI", "params": {"interval": 5.0}}, "hold": 2.0},
    }


@pytest.mark.unit
def test_to_spec_time_unit_seconds_remaps_like_from_dsl() -> None:
    ast = _parse("Conc(FI2s LH500ms, FT1s, COD=250ms)")
    spec = to_spec(ast, time_unit_seconds=0.5)
    assert spec["params"]["cod"] == pytest.approx(0.125)
    assert spec["params"]["components"]["c0"]["params"]["hold"] == pytest.approx(0.25)
    _assert_equivalent(ast, time_unit_seconds=0.5)


# --- Spec shape --------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("FR5", {"type": "FR", "params": {"n": 5}}),
        ("CRF", {"type": "CRF", "params": {}}),
        ("RR5", {"type": "RR", "params": {"probability": 0.2}}),
        ("VR5", {"type": "VR", "params": {"mean": 5.0}}),
        (
            _SEEDED + "VI10s",
            {"type": "VI", "params": {"mean_interval": 10.0, "seed": 11}},
        ),
        (
            '@algorithm("fleshler-hoffman", n=6, seed=4)\nVT10s',
            {"type": "VT", "params": {"mean_interval": 10.0, "n_intervals": 6, "seed": 4}},
        ),
        ("DRO5s", {"type": "DRO", "params": {"interval": 5.0, "dro_type": "resetting"}}),
        ("DRH2s", {"type": "DRH", "params": {"response_count": 2, "time_window": 2.0}}),
        (
            "Pctl(IRT, 50)",
            {
                "type": "Percentile",
                "params": {"target": "irt", "rank": 50, "window": 20, "direction": "below"},
            },
        ),
        (
            "PR 5",
            {
                "type": "ProgressiveRatio",
                "params": {"step": "arithmetic", "start": 5, "step_size": 5},
            },
        ),
        (
            "Conc(FR3, FI5s, COD=1s)",
            {
                "type": "Concurrent",
                "params": {
                    "components": {
                        "c0": {"type": "FR", "params": {"n": 3}},
                        "c1": {"type": "FI", "params": {"interval": 5.0}},
                    },
                    "cod": 1.0,
                    "frco": 0,
                },
            },
        ),
        (
            "Mult(FR2, FR3)",
            {
                "type": "Multiple",
                "params": {
                    "components": [
                        {"type": "FR", "params": {"n": 2}},
                        {"type": "FR", "params": {"n": 3}},
                    ],
                    "stimuli": ["s0", "s1"],
                },
            },
        ),
        (
            "FR2 ResponseCost(amount=1)",
            {
                "type": "ResponseCost",
                "params": {
                    "inner": {"type": "FR", "params": {"n": 2}},
                    "amount": 1.0,
                    "unit": "token",
                },
            },
        ),
        (
            "GoNoGo(responseWindow=2s, consequence=CRF, ITI=1s)",
            {
                "type": "GoNoGo",
                "params": {
                    "go_probability": 0.5,
                    "response_window": 2.0,
                    "iti": 1.0,
                    "correct_go_schedule": {"type": "CRF", "params": {}},
                    "correct_nogo_schedule": {"type": "EXT", "params": {}},
                    "false_alarm": {"type": "EXT", "params": {}},
                },
            },
        ),
    ],
)
def test_to_spec_shape(source: str, expected: dict[str, Any]) -> None:
    assert to_spec(_parse(source)) == expected


# --- Constructs without a spec form ------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("source", "match"),
    [
        ("COD = 2s\nConc(FR1, FR1)", "COD"),
        ("LH = 2s\nFI5s", "analyze"),
        ("Conc(VR5, VR5, COD(1->2)=2s, COD(2->1)=1s)", "directional COD"),
        ("Conc(VI30s, VI60s, COD=2s, PUNISH(1)=VI30s)", "PUNISH"),
        ("Mult(FR3, FR5, BO=5s)", "BO"),
        ("Mix(FR3, FR5, BO=5s)", "BO"),
        ("Overlay(FR5, FR10, target=changeover)", "target"),
        ("Lag(5, length=8)", "Lag"),
        ("Lag5", "Lag"),
        ("Pctl(latency, 50)", "latency"),
        ("Adj(delay, start=2s, step=1s)", "delay"),
        ("Adj(amount, start=2, step=1)", "amount"),
        ("MTS(comparisons=3, consequence=CRF, ITI=5s, correction=true)", "correction"),
        ("GoNoGo(responseWindow=2s, consequence=CRF, ITI=1s, LH=1s)", "LH"),
    ],
)
def test_to_spec_rejects_unrepresentable(source: str, match: str) -> None:
    with pytest.raises(SpecUnrepresentableError, match=match):
        to_spec(_parse(source))


@pytest.mark.unit
def test_to_spec_rejects_unrepresentable_inside_compound() -> None:
    with pytest.raises(SpecUnrepresentableError, match="Lag"):
        to_spec(_parse("Mult(FR5, Lag5)"))


@pytest.mark.unit
def test_to_spec_rejects_rd_after_analyze() -> None:
    program = analyze(_parse("RD = 1s\nFR5")).program
    with pytest.raises(SpecUnrepresentableError, match="RD"):
        to_spec(program)


@pytest.mark.unit
def test_to_spec_rejects_declaration_form_mts() -> None:
    from contingency_dsl.ast import TrialBased

    node = TrialBased(trial_type="MTS", params={"samples": 3, "comparisons": 3})
    with pytest.raises(SpecUnrepresentableError, match="samples"):
        to_spec(node)


@pytest.mark.unit
def test_unrepresentable_is_a_schedule_config_error() -> None:
    assert issubclass(SpecUnrepresentableError, ScheduleConfigError)


# --- Errors shared with from_dsl ---------------------------------------------


def _shared_error_nodes() -> list[tuple[Any, str]]:
    from contingency_dsl.ast import (
        Atomic,
        AversiveSchedule,
        Compound,
        IdentifierRef,
        Modifier,
        ScheduleType,
        Special,
        TrialBased,
    )
    from contingency_dsl.enums import Combinator, Distribution, Domain

    fr5 = Atomic(schedule_type=ScheduleType(Distribution.FIXED, Domain.RATIO), value=5.0)
    return [
        (IdentifierRef(name="x"), "IdentifierRef"),
        (Atomic(schedule_type=ScheduleType(Distribution.FIXED, Domain.RATIO), value=2.5), "FR"),
        (Atomic(schedule_type=ScheduleType(Distribution.RANDOM, Domain.RATIO), value=0.0), "RR"),
        (Special(kind="XYZ"), "Special"),
        (Compound(combinator=Combinator.ALT, components=(fr5,)), "Alt requires"),
        (Compound(combinator=Combinator.CONJ, components=(fr5,)), "Conj requires"),
        (Compound(combinator=Combinator.OVERLAY, components=(fr5,)), "Overlay requires"),
        (Modifier(modifier="DRL"), "DRL requires"),
        (Modifier(modifier="Repeat", value=1, inner=fr5), "count >= 2"),
        (Modifier(modifier="PR", pr_step="bogus"), "pr_step"),
        (Modifier(modifier="Pctl", pctl_rank=50), "target"),
        (Modifier(modifier="XYZ"), "Modifier"),
        (AversiveSchedule(kind="Sidman", params={}), "SSI"),
        (AversiveSchedule(kind="XYZ", params={}), "aversive"),
        (TrialBased(trial_type="XYZ", params={}), "trial_type"),
        (object(), "unrecognised"),
    ]


@pytest.mark.unit
@pytest.mark.parametrize(("node", "match"), _shared_error_nodes(), ids=lambda v: type(v).__name__)
def test_to_spec_raises_where_from_dsl_raises(node: Any, match: str) -> None:
    with pytest.raises(ScheduleConfigError, match=match):
        from_dsl(node)
    with pytest.raises(ScheduleConfigError, match=match) as info:
        to_spec(node)
    assert not isinstance(info.value, SpecUnrepresentableError)


@pytest.mark.unit
def test_to_spec_rejects_arithmetic_algorithm_like_from_dsl() -> None:
    ast = _parse('@algorithm("arithmetic", seed=3)\nVI30s')
    with pytest.raises(ScheduleConfigError, match="fleshler-hoffman"):
        from_dsl(ast)
    with pytest.raises(ScheduleConfigError, match="fleshler-hoffman"):
        to_spec(ast)


# --- Conformance corpus ------------------------------------------------------

# Fixtures whose schedule the DSL can write. The value is the DSL source
# and, for Concurrent, the fixture's operandum names in component order
# (the DSL names components c0, c1, ...). The other fixtures use spec
# values the DSL cannot write (DRH response_count != 2, DRO momentary,
# ResponseCost initial_balance, Interpolate geometry, MTS / GoNoGo
# without a consequence schedule).
_FIXTURE_DSL: dict[str, tuple[str, tuple[str, ...]]] = {
    "atomic/crf_basic.json": ("CRF", ()),
    "atomic/ext_basic.json": ("EXT", ()),
    "atomic/fi_basic.json": ("FI10s", ()),
    "atomic/fr_basic.json": ("FR5", ()),
    "atomic/ft_basic.json": ("FT5s", ()),
    "atomic/limited_hold_fi.json": ("FI5s LH2s", ()),
    "atomic/ri_seeded_5.json": ('@algorithm("fleshler-hoffman", seed=5)\nRI5s', ()),
    "atomic/rr_seeded_99.json": ('@algorithm("fleshler-hoffman", seed=99)\nRR5', ()),
    "atomic/rt_seeded_11.json": ('@algorithm("fleshler-hoffman", seed=11)\nRT3s', ()),
    "atomic/vi_seeded_7.json": ('@algorithm("fleshler-hoffman", n=12, seed=7)\nVI10s', ()),
    "atomic/vr_seeded_42.json": ('@algorithm("fleshler-hoffman", n=12, seed=42)\nVR5', ()),
    "atomic/vt_seeded_3.json": ('@algorithm("fleshler-hoffman", n=12, seed=3)\nVT10s', ()),
    "aversive/discriminated_avoidance_w3_iti4.json": (
        "DiscrimAv(CSUSInterval=3s, ITI=4s, mode=escape)",
        (),
    ),
    "aversive/escape_trial3_iti2.json": ("Escape(SafeDuration=2s, MaxShock=3s)", ()),
    "aversive/sidman_ssi5_rsi3.json": ("Sidman(SSI=5s, RSI=3s)", ()),
    "compound/alternative_fr_ft.json": ("Alt(FR5, FT10s)", ()),
    "compound/chained_fr2_fi5.json": ("Chain(FR2, FI5s)", ()),
    "compound/chained_fr2_fr3.json": ("Chain(FR2, FR3)", ()),
    "compound/concurrent_cod.json": ("Conc(FR3, FR3, COD=1s)", ("left", "right")),
    "compound/concurrent_cod_tick.json": ("Conc(FT5s, FI3s, COD=2s)", ("a", "b")),
    "compound/conjunctive_fr3_fi5.json": ("Conj(FR3, FI5s)", ()),
    "compound/mixed_fr2_fr3.json": ("Mix(FR2, FR3)", ()),
    "compound/multiple_fr_fr.json": ("Mult(FR2, FR3)", ()),
    "compound/overlay_fr2_fr3.json": ("Overlay(FR2, FR3)", ()),
    "compound/second_order_fr2_fr3.json": ("FR2(FR3)", ()),
    "compound/tandem_fr_fr.json": ("Tand(FR2, FR3)", ()),
    "differential/drl_basic.json": ("DRL3s", ()),
    "differential/dro_resetting.json": ("DRO5s", ()),
    "differential/percentile_irt_r75_w3.json": ("Pctl(IRT, 75, window=3, dir=above)", ()),
    "progressive/adjusting_ratio_s2_max3.json": ("Adj(ratio, start=2, step=1, max=3)", ()),
    "progressive/interlocking_r4_t8.json": ("Interlock(R0=4, T=8s)", ()),
    "progressive/pr_arithmetic.json": ("PR(linear, start=1, increment=1)", ()),
    "wrapper/timeout_fr2_to3.json": ("FR2 TO(duration=3s, reset_on_response=false)", ()),
}

# Spec defaults, so that a value written explicitly equals an omitted one.
_SPEC_DEFAULTS: dict[str, dict[str, Any]] = {
    "Sidman": {"shock_magnitude": 1.0},
    "DiscriminatedAvoidance": {"shock_magnitude": 1.0},
    "Escape": {"shock_magnitude": 1.0},
    "Timeout": {"reset_on_response": False},
    "Concurrent": {"cod": 0.0, "frco": 0},
}

_FIXTURES = {f.rel: f for f in iter_fixtures()}


def _canonical(spec: Any, rename: dict[str, str]) -> Any:
    """Fill spec defaults, drop nulls, and rename Concurrent keys."""
    if isinstance(spec, list):
        return [_canonical(v, rename) for v in spec]
    if not isinstance(spec, dict):
        return float(spec) if isinstance(spec, int) and not isinstance(spec, bool) else spec
    if "type" not in spec:
        return {rename.get(k, k): _canonical(v, rename) for k, v in spec.items()}
    params = {**_SPEC_DEFAULTS.get(spec["type"], {}), **spec.get("params", {})}
    return {
        "type": spec["type"],
        "params": {k: _canonical(v, rename) for k, v in params.items() if v is not None},
    }


@pytest.mark.integration
def test_fixture_dsl_table_names_real_fixtures() -> None:
    assert set(_FIXTURE_DSL) <= set(_FIXTURES)


@pytest.mark.integration
@pytest.mark.parametrize("rel", sorted(_FIXTURE_DSL))
def test_to_spec_reproduces_conformance_fixture(rel: str) -> None:
    source, operanda = _FIXTURE_DSL[rel]
    fixture = _FIXTURES[rel]
    spec = to_spec(_parse(source))
    rename = {f"c{i}": name for i, name in enumerate(operanda)}

    assert _canonical(spec, rename) == _canonical(fixture.data["schedule"], {})

    # Replay the fixture's own event stream on the schedule built from the
    # DSL-derived spec, with the fixture's operanda mapped to c0, c1, ...
    back = {name: key for key, name in rename.items()}
    data = json.loads(json.dumps(fixture.data))
    data["schedule"] = spec
    for step in data["steps"]:
        if step["event"] is not None and "operandum" in step["event"]:
            op = step["event"]["operandum"]
            step["event"]["operandum"] = back.get(op, op)
    observed = replay(data)
    for step in observed["steps"]:
        meta = step.get("meta", {})
        if "operandum" in meta:
            meta["operandum"] = rename.get(meta["operandum"], meta["operandum"])
    result = compare(fixture, observed, stochastic="strict")
    assert result.status == "PASS", result
