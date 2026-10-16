"""Spec-to-schedule builder for the conformance corpus.

``build_schedule_from_spec`` is the authoritative mapping from a
fixture's ``"schedule"`` object to a runtime :class:`Schedule`. Both the
fixture generator (``scripts/generate_conformance.py``) and the
built-in ``replay-python`` adapter use it, so there is one mapping.
Ports in other languages should reproduce it.
"""

from __future__ import annotations

import random
from typing import Any

from contingency.interfaces import Schedule
from contingency.schedules import (
    DRH,
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
    Chained,
    Concurrent,
    LimitedHold,
    Multiple,
    ProgressiveRatio,
    Tandem,
    arithmetic,
    geometric,
    richardson_roberts,
)

__all__ = ["build_schedule_from_spec"]


def build_schedule_from_spec(spec: dict[str, Any]) -> Schedule:
    """Instantiate a Schedule from a JSON-serialisable spec.

    This function is the *authoritative* spec-to-schedule mapping for
    the conformance corpus. Rust ports should reproduce it.
    """
    stype = spec["type"]
    params = spec.get("params", {})

    if stype == "FR":
        return FR(int(params["n"]))
    if stype == "CRF":
        return FR(1)
    if stype == "EXT":
        return EXT()
    if stype == "VR":
        return VR(
            float(params["mean"]),
            n_intervals=int(params.get("n_intervals", 12)),
            seed=params.get("seed"),
        )
    if stype == "RR":
        rng = random.Random(params["seed"]) if "seed" in params else None
        return RR(float(params["probability"]), rng=rng)
    if stype == "FI":
        return FI(float(params["interval"]))
    if stype == "VI":
        return VI(
            float(params["mean_interval"]),
            n_intervals=int(params.get("n_intervals", 12)),
            seed=params.get("seed"),
        )
    if stype == "RI":
        rng = random.Random(params["seed"]) if "seed" in params else None
        return RI(float(params["mean_interval"]), rng=rng)
    if stype == "FT":
        return FT(float(params["interval"]))
    if stype == "VT":
        return VT(
            float(params["mean_interval"]),
            n_intervals=int(params.get("n_intervals", 12)),
            seed=params.get("seed"),
        )
    if stype == "RT":
        rng = random.Random(params["seed"]) if "seed" in params else None
        return RT(float(params["mean_interval"]), rng=rng)
    if stype == "LimitedHold":
        inner = build_schedule_from_spec(params["inner"])
        return LimitedHold(inner, float(params["hold"]))  # type: ignore[arg-type]
    if stype == "DRO":
        return DRO(
            float(params["interval"]),
            dro_type=params.get("dro_type", "resetting"),
        )
    if stype == "DRL":
        return DRL(float(params["interval"]))
    if stype == "DRH":
        return DRH(int(params["response_count"]), float(params["time_window"]))
    if stype == "Concurrent":
        comp_map: dict[str, Schedule] = {
            key: build_schedule_from_spec(cspec)
            for key, cspec in params["components"].items()
        }
        return Concurrent(
            comp_map,
            cod=float(params.get("cod", 0.0)),
            frco=int(params.get("frco", 0)),
        )
    if stype == "Alternative":
        first = build_schedule_from_spec(params["first"])
        second = build_schedule_from_spec(params["second"])
        return Alternative(first, second)
    if stype == "Multiple":
        comp_list: list[Schedule] = [
            build_schedule_from_spec(c) for c in params["components"]
        ]
        stimuli = params.get("stimuli")
        return Multiple(comp_list, stimuli=stimuli)
    if stype == "Chained":
        comp_list = [build_schedule_from_spec(c) for c in params["components"]]
        stimuli = params.get("stimuli")
        return Chained(comp_list, stimuli=stimuli)
    if stype == "Tandem":
        comp_list = [build_schedule_from_spec(c) for c in params["components"]]
        return Tandem(comp_list)
    if stype == "ProgressiveRatio":
        step_name = params["step"]
        if step_name == "arithmetic":
            step_fn = arithmetic(
                start=int(params.get("start", 1)),
                step=int(params.get("step_size", 1)),
            )
        elif step_name == "geometric":
            step_fn = geometric(
                start=int(params.get("start", 1)),
                ratio=float(params.get("ratio", 2.0)),
            )
        elif step_name == "richardson_roberts":
            step_fn = richardson_roberts()
        else:
            raise ValueError(f"unknown PR step: {step_name!r}")
        return ProgressiveRatio(step_fn)

    raise ValueError(f"unknown schedule type: {stype!r}")
