"""Hypothesis strategies for schedule specs and response sequences.

The specs use the conformance form ``{"type": <name>, "params": {...}}``
read by :func:`contingency.conformance.spec.build_schedule_from_spec` and
by :func:`contingency.engine.from_spec`. The differential test feeds the
same spec and the same steps to both and compares every step.

* :data:`SPEC_TYPES_COVERED` lists every type the strategies can emit.
  The test checks it against ``contingency_core.SPEC_TYPES``.
* :func:`specs_of_type` / :func:`cases_of_type` fix the outermost type
  (so a run can give every type the same number of cases) and nest at
  most two levels deep.
  Randomised types always carry a ``seed``.
* :func:`steps_for` builds ``(now, operandum or None, reset)`` tuples with
  strictly increasing ``now``: one of three rates, bursts of fast
  responses, runs of response-free ticks and at most one ``reset()``.
"""

from __future__ import annotations

from typing import Any

from hypothesis import strategies as st

Spec = dict[str, Any]
Step = tuple[float, str | None, bool]

SEEDED_TYPES = frozenset({"VR", "RR", "VI", "RI", "VT", "RT", "MatchingToSample", "GoNoGo"})

seeds = st.integers(0, 2**32 - 1)
seconds = st.one_of(
    st.sampled_from([0.5, 1.0, 2.0, 5.0, 10.0, 30.0]),
    st.floats(0.05, 60.0, allow_nan=False, allow_infinity=False),
)
counts = st.integers(1, 12)
probabilities = st.floats(0.02, 1.0, allow_nan=False)
n_intervals = st.one_of(st.none(), st.integers(1, 20))
magnitudes = st.sampled_from([None, 1.0, 0.5, 2.0])


def _spec(name: str, **params: st.SearchStrategy[Any]) -> st.SearchStrategy[Spec]:
    """``{"type": name, "params": {...}}``; a param drawn as ``None`` is omitted."""

    def build(drawn: dict[str, Any]) -> Spec:
        return {"type": name, "params": {k: v for k, v in drawn.items() if v is not None}}

    return st.fixed_dictionaries(params).map(build)


def _optional_nested(inner: st.SearchStrategy[Spec]) -> st.SearchStrategy[Any]:
    """Absent, explicit ``null`` or a spec (the three forms of an optional nested spec)."""
    return st.one_of(st.just("__absent__"), st.just(None), inner)


def _spec_with_optional(
    name: str,
    required: dict[str, st.SearchStrategy[Any]],
    optional: dict[str, st.SearchStrategy[Any]],
) -> st.SearchStrategy[Spec]:
    """Like :func:`_spec`, but ``optional`` values may be absent (``"__absent__"``) or ``null``."""

    def build(drawn: tuple[dict[str, Any], dict[str, Any]]) -> Spec:
        req, opt = drawn
        params = {k: v for k, v in req.items() if v is not None}
        params.update({k: v for k, v in opt.items() if v != "__absent__"})
        return {"type": name, "params": params}

    return st.tuples(st.fixed_dictionaries(required), st.fixed_dictionaries(optional)).map(build)


# -- leaves (types without an inner spec) ------------------------------------


def _percentile() -> st.SearchStrategy[Spec]:
    @st.composite
    def build(draw: st.DrawFn) -> Spec:
        window = draw(st.integers(1, 10))
        params: dict[str, Any] = {
            "target": "irt",
            "rank": draw(st.integers(0, 100)),
            "window": window,
        }
        direction = draw(st.sampled_from([None, "above", "below"]))
        if direction is not None:
            params["direction"] = direction
        return {"type": "Percentile", "params": params}

    return build()


def _adjusting() -> st.SearchStrategy[Spec]:
    @st.composite
    def build(draw: st.DrawFn) -> Spec:
        target = draw(st.sampled_from(["ratio", "interval"]))
        if target == "ratio":
            start = float(draw(st.integers(1, 10)))
            step = float(draw(st.integers(-2, 3)))
            lo, hi = 1.0, 20.0
        else:
            start = draw(seconds)
            step = draw(st.sampled_from([-1.0, 0.0, 0.5, 1.0, 2.0]))
            lo, hi = 0.1, 60.0
        params: dict[str, Any] = {"target": target, "start": start, "step": step}
        if draw(st.booleans()):
            params["minimum"] = lo
        if draw(st.booleans()):
            params["maximum"] = hi
        return {"type": "AdjustingSchedule", "params": params}

    return build()


def _leaf_strategies() -> dict[str, st.SearchStrategy[Spec]]:
    return {
        "FR": _spec("FR", n=counts),
        "CRF": _spec("CRF"),
        "EXT": _spec("EXT"),
        "VR": _spec("VR", mean=st.integers(1, 20).map(float), n_intervals=n_intervals, seed=seeds),
        "RR": _spec("RR", probability=probabilities, seed=seeds),
        "FI": _spec("FI", interval=seconds),
        "VI": _spec("VI", mean_interval=seconds, n_intervals=n_intervals, seed=seeds),
        "RI": _spec("RI", mean_interval=seconds, seed=seeds),
        "FT": _spec("FT", interval=seconds),
        "VT": _spec("VT", mean_interval=seconds, n_intervals=n_intervals, seed=seeds),
        "RT": _spec("RT", mean_interval=seconds, seed=seeds),
        "DRO": _spec(
            "DRO", interval=seconds, dro_type=st.sampled_from([None, "resetting", "momentary"])
        ),
        "DRL": _spec("DRL", interval=seconds),
        "DRH": _spec("DRH", response_count=st.integers(1, 6), time_window=seconds),
        "ProgressiveRatio": st.one_of(
            _spec(
                "ProgressiveRatio",
                step=st.just("arithmetic"),
                start=st.one_of(st.none(), st.integers(1, 4)),
                step_size=st.one_of(st.none(), st.integers(1, 4)),
            ),
            _spec(
                "ProgressiveRatio",
                step=st.just("geometric"),
                start=st.one_of(st.none(), st.integers(1, 3)),
                ratio=st.one_of(st.none(), st.sampled_from([1.1, 1.5, 2.0])),
            ),
            _spec("ProgressiveRatio", step=st.just("richardson_roberts")),
        ),
        "InterlockingSchedule": _spec(
            "InterlockingSchedule", initial_ratio=st.integers(1, 30), decay_time=seconds
        ),
        "Lag": _spec("Lag", n=st.integers(0, 4), length=st.one_of(st.none(), st.integers(1, 3))),
        "Sidman": _spec(
            "Sidman",
            shock_shock_interval=seconds,
            response_shock_interval=seconds,
            shock_magnitude=magnitudes,
        ),
        "DiscriminatedAvoidance": _spec(
            "DiscriminatedAvoidance",
            warning_duration=seconds,
            iti=seconds,
            shock_magnitude=magnitudes,
        ),
        "Escape": _spec("Escape", trial_duration=seconds, iti=seconds, shock_magnitude=magnitudes),
        "Percentile": _percentile(),
        "AdjustingSchedule": _adjusting(),
    }


# -- trial-based (leaves whose optional consequences are specs) --------------


def _mts(inner: st.SearchStrategy[Spec]) -> st.SearchStrategy[Spec]:
    return _spec_with_optional(
        "MatchingToSample",
        {"n_comparisons": st.integers(2, 4), "seed": seeds},
        {
            "sample_duration": st.one_of(st.just("__absent__"), st.sampled_from([0.0, 0.5, 1.0])),
            "choice_timeout": st.one_of(st.just("__absent__"), seconds),
            "iti": st.one_of(st.just("__absent__"), st.sampled_from([0.5, 1.0, 2.0])),
            "consequence": _optional_nested(inner),
            "incorrect": _optional_nested(inner),
            "correction": st.one_of(
                st.just("__absent__"),
                st.just({"mode": "disabled"}),
                st.just({"mode": "unlimited"}),
                st.integers(1, 3).map(lambda n: {"mode": "bounded", "limit": n}),
            ),
        },
    )


def _gonogo(inner: st.SearchStrategy[Spec]) -> st.SearchStrategy[Spec]:
    return _spec_with_optional(
        "GoNoGo",
        {"seed": seeds},
        {
            # 0 and 1 are rejected by both engines (0 < p < 1).
            "go_probability": st.one_of(
                st.just("__absent__"), st.floats(0.02, 0.98, allow_nan=False)
            ),
            "response_window": st.one_of(st.just("__absent__"), seconds),
            "iti": st.one_of(st.just("__absent__"), st.sampled_from([0.5, 1.0, 5.0])),
            "correct_go_schedule": _optional_nested(inner),
            "correct_nogo_schedule": _optional_nested(inner),
            "false_alarm": _optional_nested(inner),
        },
    )


# -- compounds ---------------------------------------------------------------

_INTERVAL_TYPES = ("FI", "VI", "RI")


def _concurrent(inner: st.SearchStrategy[Spec]) -> st.SearchStrategy[Spec]:
    @st.composite
    def build(draw: st.DrawFn) -> Spec:
        keys = ["left", "right"] + (["center"] if draw(st.booleans()) else [])
        params: dict[str, Any] = {"components": {k: draw(inner) for k in keys}}
        cod = draw(st.sampled_from([None, 0.0, 0.5, 2.0]))
        if cod is not None:
            params["cod"] = cod
        frco = draw(st.sampled_from([None, 0, 2]))
        if frco is not None:
            params["frco"] = frco
        if draw(st.integers(0, 4)) == 0:
            a, b = draw(st.permutations(keys))[:2]
            params["cod_directional"] = [
                {"from": a, "to": b, "cod": draw(st.sampled_from([0.0, 1.0, 3.0]))}
            ]
        punish_form = draw(st.integers(0, 5))
        if punish_form == 1:
            params["punish"] = {draw(st.sampled_from(keys)): draw(inner)}
        elif punish_form == 2:
            params["punish_changeover"] = draw(inner)
        elif punish_form == 3:
            params["punish_changeover"] = None
        elif punish_form == 4:
            a, b = draw(st.permutations(keys))[:2]
            params["punish_directional"] = [{"from": a, "to": b, "schedule": draw(inner)}]
        return {"type": "Concurrent", "params": params}

    return build()


def _interpolate(inner: st.SearchStrategy[Spec]) -> st.SearchStrategy[Spec]:
    """``probe_duration`` is a fraction of ``interval`` (both engines reject ``>=``)."""

    @st.composite
    def build(draw: st.DrawFn) -> Spec:
        interval = draw(seconds)
        params: dict[str, Any] = {
            "base": draw(inner),
            "probe": draw(inner),
            "interval": interval,
            "probe_duration": interval * draw(st.sampled_from([0.1, 0.25, 0.5, 0.9])),
        }
        first = draw(st.one_of(st.none(), seconds))
        if first is not None:
            params["first_probe_at"] = first
        return {"type": "Interpolate", "params": params}

    return build()


def _compound_strategies(
    inner: st.SearchStrategy[Spec], interval_leaf: st.SearchStrategy[Spec]
) -> dict[str, st.SearchStrategy[Spec]]:
    components = st.lists(inner, min_size=2, max_size=3)
    return {
        "Alternative": _spec("Alternative", first=inner, second=inner),
        "Conjunctive": _spec("Conjunctive", first=inner, second=inner),
        "Tandem": _spec("Tandem", components=components),
        "Chained": _spec("Chained", components=components),
        "Mixed": _spec("Mixed", components=components),
        "Multiple": _spec("Multiple", components=components),
        "Timeout": _spec(
            "Timeout",
            inner=inner,
            duration=seconds,
            reset_on_response=st.one_of(st.none(), st.booleans()),
        ),
        "LimitedHold": _spec("LimitedHold", inner=interval_leaf, hold=seconds),
        "ResponseCost": _spec(
            "ResponseCost",
            inner=inner,
            amount=st.sampled_from([0.5, 1.0, 2.0]),
            unit=st.sampled_from([None, "token", "point"]),
            initial_balance=st.one_of(st.none(), st.sampled_from([0.0, 1.0, 3.0, 10.0])),
        ),
        "SecondOrder": _spec("SecondOrder", overall=inner, unit=inner),
        "Interpolate": _interpolate(inner),
        "Overlay": _spec("Overlay", base=inner, punisher=inner),
        "Concurrent": _concurrent(inner),
        "MatchingToSample": _mts(inner),
        "GoNoGo": _gonogo(inner),
    }


_LEAVES = _leaf_strategies()
_ANY_LEAF = st.one_of(list(_LEAVES.values()))
_INTERVAL_LEAF = st.one_of([_LEAVES[t] for t in _INTERVAL_TYPES])
_DEPTH1 = _compound_strategies(_ANY_LEAF, _INTERVAL_LEAF)
_ANY_DEPTH1 = st.one_of(_ANY_LEAF, *_DEPTH1.values())
_DEPTH2 = _compound_strategies(_ANY_DEPTH1, _INTERVAL_LEAF)

SPEC_TYPES_COVERED: frozenset[str] = frozenset(_LEAVES) | frozenset(_DEPTH2)


def _by_type(name: str) -> st.SearchStrategy[Spec]:
    if name in _DEPTH2:
        return st.one_of(_DEPTH1[name], _DEPTH2[name])
    return _LEAVES[name]


def specs_of_type(name: str) -> st.SearchStrategy[Spec]:
    """A spec whose outermost type is ``name`` (depth 1 or 2 for compounds).

    Runs that need every type call this once per type: Hypothesis does not
    draw ``sampled_from`` uniformly over one run (over 1,000 cases one
    type came out once).
    """
    return _by_type(name)


# -- response sequences ------------------------------------------------------


def operanda_of(spec: Spec) -> list[str]:
    """Operandum names that mean something somewhere inside ``spec``.

    ``main`` always; the keys of every ``Concurrent``; ``choice_0`` …
    for every ``MatchingToSample``. The order is stable (first seen).
    """
    names: list[str] = ["main"]

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "Concurrent":
                for key in node.get("params", {}).get("components", {}):
                    if key not in names:
                        names.append(key)
            if node.get("type") == "MatchingToSample":
                n = int(node.get("params", {}).get("n_comparisons", 0))
                for i in range(n):
                    if f"choice_{i}" not in names:
                        names.append(f"choice_{i}")
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(spec)
    return names


def _concurrent_key_sets(spec: Spec) -> list[list[str]]:
    found: list[list[str]] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "Concurrent":
                found.append(list(node.get("params", {}).get("components", {})))
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(spec)
    return found


def preferred_operanda(spec: Spec) -> list[str]:
    """Names most responses use, so that the schedule keeps running.

    A ``Concurrent`` anywhere inside raises on a name that is not one of
    its keys once a response reaches it, so with a ``Concurrent`` inside
    the names are the keys every ``Concurrent`` in the spec has. An
    outermost ``MatchingToSample`` uses its ``choice_*`` names.
    """
    names = operanda_of(spec)
    if spec["type"] == "MatchingToSample":
        return [n for n in names if n.startswith("choice_")] + ["main"]
    key_sets = _concurrent_key_sets(spec)
    if key_sets:
        shared = [k for k in key_sets[0] if all(k in ks for ks in key_sets[1:])]
        return shared or key_sets[0]
    return names


RATES = (0.2, 1.0, 5.0)


def steps_for(spec: Spec) -> st.SearchStrategy[list[Step]]:
    """Strictly increasing steps for ``spec`` (1–120 of them)."""
    preferred = preferred_operanda(spec)
    others = [n for n in operanda_of(spec) if n not in preferred]

    @st.composite
    def build(draw: st.DrawFn) -> list[Step]:
        mean_gap = 1.0 / draw(st.sampled_from(RATES))
        n = draw(st.integers(1, 120))
        name = st.sampled_from(preferred)
        # About 1 case in 10 sends one response under another name from the spec,
        # which a Concurrent rejects (both engines should raise the same error).
        stray_at = draw(st.one_of(*([st.none()] * 9), st.integers(0, max(0, n - 1))))
        reset_at = draw(st.one_of(*([st.none()] * 3), st.integers(0, max(0, n - 1))))
        out: list[Step] = []
        now = 0.0
        while len(out) < n:
            kind = draw(st.sampled_from(["single"] * 6 + ["tick"] * 2 + ["burst", "pause"]))
            if kind == "burst":
                op = draw(name)
                for _ in range(draw(st.integers(2, 10))):
                    now += draw(st.floats(0.001, 0.05))
                    out.append((now, op, False))
            elif kind == "pause":
                for _ in range(draw(st.integers(1, 5))):
                    now += draw(st.floats(2.0 * mean_gap, 20.0 * mean_gap))
                    out.append((now, None, False))
            else:
                now += draw(st.floats(0.001, 4.0 * mean_gap))
                out.append((now, draw(name) if kind == "single" else None, False))
        out = out[:n]
        if others and stray_at is not None and stray_at < len(out):
            t, _, r = out[stray_at]
            out[stray_at] = (t, draw(st.sampled_from(others)), r)
        if reset_at is not None and reset_at < len(out):  # 0: reset before the first step
            t, op, _ = out[reset_at]
            out[reset_at] = (t, op, True)
        return out

    return build()


def cases_of_type(name: str) -> st.SearchStrategy[tuple[Spec, list[Step]]]:
    """``(spec, steps)`` pairs whose outermost type is ``name``."""
    return specs_of_type(name).flatmap(lambda spec: st.tuples(st.just(spec), steps_for(spec)))


def spec_types_in(spec: Spec) -> list[str]:
    """Every ``"type"`` value inside ``spec`` (outermost first, duplicates removed)."""
    found: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            t = node.get("type")
            if isinstance(t, str) and "params" in node and t not in found:
                found.append(t)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(spec)
    return found


__all__ = [
    "RATES",
    "SEEDED_TYPES",
    "SPEC_TYPES_COVERED",
    "Spec",
    "Step",
    "cases_of_type",
    "operanda_of",
    "preferred_operanda",
    "spec_types_in",
    "specs_of_type",
    "steps_for",
]
