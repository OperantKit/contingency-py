"""Write the respondent Tier A conformance fixtures.

Each fixture pins the trial list that one Tier A respondent expression
expands to under the execution contract in
``docs/en/respondent-contract.md``. This script is the reference reading
of that contract: it uses only the standard library, and every random
draw goes through ``random.Random(seed).random()`` so that another
implementation can reproduce the same stream from the 53-bit Mersenne
Twister output alone.

Run from the package root::

    python scripts/generate_respondent_conformance.py          # write
    python scripts/generate_respondent_conformance.py --check  # compare only
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

OUT_DIR = Path(__file__).resolve().parents[1] / "conformance" / "respondent"

_UNIT_SECONDS = {"s": 1.0, "ms": 0.001, "min": 60.0, "h": 3600.0}


class UnresolvedError(ValueError):
    """A value the expression leaves to the program is not supplied."""


def _seconds(value: Mapping[str, Any] | None) -> float | None:
    if value is None:
        return None
    unit = value.get("unit", value.get("time_unit"))
    if unit not in _UNIT_SECONDS:
        raise ValueError(f"duration unit not understood: {unit!r}")
    return float(value["value"]) * _UNIT_SECONDS[str(unit)]


def _time(
    *,
    cs_on: float | None = None,
    cs_off: float | None = None,
    us_on: float | None = None,
    gap: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {"cs_on": cs_on, "cs_off": cs_off, "us_on": us_on, "us_off": None, "gap": gap}


def _trial(
    cs: list[str],
    us: str | None,
    time: dict[str, Any],
    *,
    order: list[str] | None = None,
    iti_s: float | None = None,
) -> dict[str, Any]:
    return {
        "stage": "main",
        "cs": sorted(cs),
        "us": us,
        "context": None,
        "order": order,
        "time": time,
        "iti_s": iti_s,
    }


def _cs_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping) and value.get("type") == "Compound":
        if value.get("mode", "Simultaneous") != "Simultaneous":
            raise ValueError(f"Compound mode is not Tier A: {value.get('mode')!r}")
        return [str(e) for e in value["cs_list"]]
    raise ValueError(f"CS argument not understood: {value!r}")


def _label(options: Mapping[str, Any], key: str) -> str:
    labels = options.get("labels") or {}
    if key not in labels:
        raise UnresolvedError(f"labels.{key}")
    return str(labels[key])


def _draws(seed: int | None, n: int) -> list[float]:
    if seed is None:
        raise UnresolvedError("seed")
    rng = random.Random(seed)
    return [rng.random() for _ in range(n)]


def _periods(
    cs: str, us: str, p_cs: float, p_no_cs: float, n: int, seed: int | None
) -> list[dict[str, Any]]:
    """``n`` CS periods alternating with ``n`` no-CS periods, CS first."""
    u = _draws(seed, 2 * n)
    out: list[dict[str, Any]] = []
    for k in range(n):
        out.append(_trial([cs], us if u[2 * k] < p_cs else None, _time(cs_on=0.0)))
        out.append(_trial([], us if u[2 * k + 1] < p_no_cs else None, _time()))
    return out


def _iti_values(spec: Mapping[str, Any], n: int, seed: int | None) -> list[float]:
    mean = _seconds(spec["mean"])
    assert mean is not None
    dist = spec["distribution"]
    if dist == "fixed":
        return [mean] * n
    u = _draws(seed, n)
    if dist == "uniform":
        return [2.0 * mean * x for x in u]
    if dist == "exponential":
        return [-mean * math.log(1.0 - x) for x in u]
    raise ValueError(f"ITI distribution not understood: {dist!r}")


def expand(spec: Mapping[str, Any], options: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Trial list of one Tier A expression (the contract's reference reading)."""
    kind = spec["type"]
    if "trials" in spec:
        n = int(spec["trials"])
    elif "trials" in options:
        n = int(options["trials"])
    else:
        raise UnresolvedError("trials")
    seed = options.get("seed")
    if kind == "PairForwardDelay":
        t = _time(cs_on=0.0, cs_off=_seconds(spec["cs_duration"]), us_on=_seconds(spec["isi"]))
        return [_trial(_cs_list(spec["cs"]), spec["us"], t) for _ in range(n)]
    if kind == "PairForwardTrace":
        cs_off = _seconds(spec.get("cs_duration"))
        trace = _seconds(spec["trace_interval"])
        assert trace is not None
        us_on = None if cs_off is None else cs_off + trace
        gap = {"from": "cs_off", "to": "us_on", "s": trace, "at_least": False}
        t = _time(cs_on=0.0, cs_off=cs_off, us_on=us_on, gap=gap)
        return [_trial(_cs_list(spec["cs"]), spec["us"], t) for _ in range(n)]
    if kind == "PairSimultaneous":
        t = _time(cs_on=0.0, us_on=0.0)
        return [_trial(_cs_list(spec["cs"]), spec["us"], t) for _ in range(n)]
    if kind == "PairBackward":
        gap = {"from": "us_off", "to": "cs_on", "s": _seconds(spec["isi"]), "at_least": False}
        t = {"cs_on": None, "cs_off": None, "us_on": 0.0, "us_off": None, "gap": gap}
        return [_trial(_cs_list(spec["cs"]), spec["us"], t) for _ in range(n)]
    if kind in ("Extinction", "CSOnly"):
        return [_trial(_cs_list(spec["cs"]), None, _time(cs_on=0.0)) for _ in range(n)]
    if kind == "USOnly":
        return [_trial([], spec["us"], _time(us_on=0.0)) for _ in range(n)]
    if kind == "Compound":
        return [_trial(_cs_list(spec), None, _time(cs_on=0.0)) for _ in range(n)]
    if kind == "Serial":
        order = [str(e) for e in spec["cs_list"]]
        gap = {"from": "cs_off", "to": "cs_on", "s": _seconds(spec["isi"]), "at_least": False}
        t = _time(cs_on=0.0, gap=gap)
        return [_trial(order, None, t, order=order) for _ in range(n)]
    if kind == "Contingency":
        cs, us = _label(options, "cs"), _label(options, "us")
        return _periods(cs, us, spec["p_us_given_cs"], spec["p_us_given_no_cs"], n, seed)
    if kind == "TrulyRandom":
        if "p" not in spec:
            raise UnresolvedError("p")
        return _periods(spec["cs"], spec["us"], spec["p"], spec["p"], n, seed)
    if kind == "ExplicitlyUnpaired":
        sep = _seconds(spec.get("min_separation"))
        sep_gap: dict[str, Any] | None = (
            None if sep is None else {"from": "any_cs", "to": "us_on", "s": sep, "at_least": True}
        )
        out: list[dict[str, Any]] = []
        for _ in range(n):
            out.append(_trial([spec["cs"]], None, _time(cs_on=0.0)))
            out.append(_trial([], spec["us"], _time(us_on=0.0, gap=sep_gap)))
        return out
    if kind == "Differential":
        diff_us = spec.get("us") or (options.get("labels") or {}).get("us")
        if diff_us is None:
            raise UnresolvedError("labels.us")
        out = []
        for _ in range(n):
            out.append(_trial([spec["cs_positive"]], str(diff_us), _time(cs_on=0.0)))
            out.append(_trial([spec["cs_negative"]], None, _time(cs_on=0.0)))
        return out
    if kind == "ITI":
        return [_trial([], None, _time(), iti_s=v) for v in _iti_values(spec, n, seed)]
    raise ValueError(f"not a Tier A respondent expression: {kind!r}")


def _d(value: float, unit: str = "s") -> dict[str, Any]:
    return {"value": value, "unit": unit}


# (name, element, description, spec, options)
CASES: list[tuple[str, str, str, dict[str, Any], dict[str, Any]]] = [
    (
        "pair_forward_delay",
        "Pair.ForwardDelay",
        "R1. CS onset precedes US onset and the two overlap: tone on 0-15 s, shock onset at "
        "10 s (Pavlov, 1927).",
        {
            "type": "PairForwardDelay",
            "cs": "tone",
            "us": "shock",
            "isi": _d(10.0),
            "cs_duration": _d(15.0),
        },
        {"trials": 3},
    ),
    (
        "pair_forward_delay_compound_cs",
        "Pair.ForwardDelay",
        "R1 with R11. A simultaneous compound CS paired with the US: tone and light share "
        "onset and offset (Rescorla & Wagner, 1972).",
        {
            "type": "PairForwardDelay",
            "cs": {"type": "Compound", "cs_list": ["tone", "light"], "mode": "Simultaneous"},
            "us": "food",
            "isi": _d(5.0),
            "cs_duration": _d(5.0),
        },
        {"trials": 2},
    ),
    (
        "pair_forward_trace",
        "Pair.ForwardTrace",
        "R2. CS offset is followed by an empty 5-s trace before the US; the CS duration is "
        "left to the apparatus, so CS offset and US onset stay unresolved (Ellison, 1964).",
        {"type": "PairForwardTrace", "cs": "tone", "us": "food", "trace_interval": _d(5.0)},
        {"trials": 2},
    ),
    (
        "pair_forward_trace_with_cs_duration",
        "Pair.ForwardTrace",
        "R2 with an explicit 10-s CS: US onset = CS offset + trace = 15 s.",
        {
            "type": "PairForwardTrace",
            "cs": "tone",
            "us": "food",
            "trace_interval": _d(5.0),
            "cs_duration": _d(10.0),
        },
        {"trials": 2},
    ),
    (
        "pair_simultaneous",
        "Pair.Simultaneous",
        "R3. CS onset coincides with US onset (Rescorla, 1980).",
        {"type": "PairSimultaneous", "cs": "light", "us": "airpuff"},
        {"trials": 2},
    ),
    (
        "pair_backward",
        "Pair.Backward",
        "R4. The US comes first; the CS starts 2 s after US offset (Spetch, Wilkie, & Pinel, "
        "1981). US duration is not given, so CS onset is fixed only relative to US offset.",
        {"type": "PairBackward", "us": "shock", "cs": "tone", "isi": _d(2.0)},
        {"trials": 2},
    ),
    (
        "extinction",
        "Extinction",
        "R5. CS presented alone (Pavlov, 1927); the trial count comes from the program.",
        {"type": "Extinction", "cs": "tone"},
        {"trials": 3},
    ),
    (
        "cs_only",
        "CSOnly",
        "R6. CS-alone exposure with its own trial count, e.g. a pre-exposure phase "
        "(Lubow & Moore, 1959).",
        {"type": "CSOnly", "cs": "tone", "trials": 4},
        {"trials": 1},
    ),
    (
        "us_only",
        "USOnly",
        "R7. US-alone presentations with no CS (Randich & LoLordo, 1979).",
        {"type": "USOnly", "us": "shock", "trials": 3},
        {"trials": 1},
    ),
    (
        "contingency_positive",
        "Contingency",
        "R8. P(US|CS) = 0.9, P(US|no CS) = 0.1 (Rescorla, 1967): CS periods alternate with "
        "no-CS periods; one draw per period. Labels come from the program.",
        {"type": "Contingency", "p_us_given_cs": 0.9, "p_us_given_no_cs": 0.1},
        {"trials": 5, "seed": 1, "labels": {"cs": "tone", "us": "shock"}},
    ),
    (
        "contingency_negative",
        "Contingency",
        "R8. Negative contingency, P(US|CS) = 0, P(US|no CS) = 0.3 (Rescorla, 1968).",
        {"type": "Contingency", "p_us_given_cs": 0.0, "p_us_given_no_cs": 0.3},
        {"trials": 5, "seed": 7, "labels": {"cs": "tone", "us": "shock"}},
    ),
    (
        "truly_random",
        "TrulyRandom",
        "R9. Contingency(p, p) with p = 0.3: the CS carries no information about the US "
        "(Rescorla, 1967).",
        {"type": "TrulyRandom", "cs": "tone", "us": "shock", "p": 0.3},
        {"trials": 5, "seed": 3},
    ),
    (
        "truly_random_without_p",
        "TrulyRandom",
        "R9 without p: the probability is left to the program, so expansion must fail.",
        {"type": "TrulyRandom", "cs": "tone", "us": "shock"},
        {"trials": 5, "seed": 3},
    ),
    (
        "explicitly_unpaired",
        "ExplicitlyUnpaired",
        "R10. CS-alone trials alternate with US-alone periods; every US is at least 30 s "
        "from every CS (Ayres, Benedict, & Witcher, 1975).",
        {"type": "ExplicitlyUnpaired", "cs": "tone", "us": "shock", "min_separation": _d(30.0)},
        {"trials": 2},
    ),
    (
        "compound",
        "Compound",
        "R11. Tone and light presented together without a US.",
        {"type": "Compound", "cs_list": ["tone", "light"], "mode": "Simultaneous"},
        {"trials": 2},
    ),
    (
        "serial",
        "Serial",
        "R12. Light then tone, 3 s from light offset to tone onset (Rescorla, 1972).",
        {"type": "Serial", "cs_list": ["light", "tone"], "isi": _d(3.0)},
        {"trials": 2},
    ),
    (
        "iti_fixed",
        "ITI",
        "R13. Fixed 30-s intervals (Gibbon & Balsam, 1981).",
        {"type": "ITI", "distribution": "fixed", "mean": _d(30.0)},
        {"trials": 3},
    ),
    (
        "iti_uniform_minutes",
        "ITI",
        "R13. Uniform intervals on [0, 2 x mean] with the mean given in minutes.",
        {"type": "ITI", "distribution": "uniform", "mean": _d(2.0, "min")},
        {"trials": 3, "seed": 5},
    ),
    (
        "iti_exponential",
        "ITI",
        "R13. Exponential intervals with a 60-s mean (Gallistel & Gibbon, 2000).",
        {"type": "ITI", "distribution": "exponential", "mean": _d(60.0)},
        {"trials": 3, "seed": 11},
    ),
    (
        "differential",
        "Differential",
        "R14. CS+ with the US alternates with CS- without it (Pavlov, 1927).",
        {
            "type": "Differential",
            "cs_positive": "tone_plus",
            "cs_negative": "tone_minus",
            "us": "shock",
        },
        {"trials": 2},
    ),
    (
        "differential_short_form_without_us",
        "Differential",
        "R14 short form with no US from the program: expansion must fail.",
        {"type": "Differential", "cs_positive": "tone_plus", "cs_negative": "tone_minus"},
        {"trials": 2},
    ),
]


def build(case: tuple[str, str, str, dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    name, element, description, spec, options = case
    try:
        expected: dict[str, Any] = {"trials": expand(spec, options)}
    except UnresolvedError as exc:
        expected = {"error": "unresolved", "missing": str(exc)}
    return {
        "name": name,
        "element": element,
        "description": description,
        "spec": spec,
        "options": options,
        "expected": expected,
    }


def _text(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="compare without writing")
    args = parser.parse_args(argv)
    stale: list[str] = []
    for case in CASES:
        path = OUT_DIR / f"{case[0]}.json"
        text = _text(build(case))
        if args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != text:
                stale.append(path.name)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
    if stale:
        print("stale respondent fixtures: " + ", ".join(stale), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
