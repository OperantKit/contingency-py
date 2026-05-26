"""Generate the cross-language conformance corpus.

Runs each fixture's event sequence against the Python runtime and
captures the produced Outcomes as the authoritative expectation.
Re-running this script must produce byte-identical JSON, so:

- Use ``json.dump(..., indent=2, sort_keys=True)``.
- Round floats to a fixed number of digits when writing to JSON so
  platform float formatting does not cause diffs.
- Use only JSON-primitive meta values (bool/int/str).

Usage::

    .venv/bin/python scripts/generate_conformance.py

Writes ``conformance/**/*.json``. The generated files are the
authoritative conformance corpus and are committed to the repository.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from contingency.entities import Outcome, ResponseEvent
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

ROOT = Path(__file__).resolve().parents[1]
CONFORMANCE_DIR = ROOT / "conformance"

# Float precision when serialising observed numeric outcomes. Chosen
# to be tight enough for determinism checks while loose enough that
# trivial float-format differences across platforms cannot cause diffs.
FLOAT_NDIGITS = 9


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Fixture execution
# ---------------------------------------------------------------------------


@dataclass
class FixtureSpec:
    """In-memory description of a fixture before Outcomes are filled in."""

    name: str
    schedule: dict[str, Any]
    steps: list[dict[str, Any]] = field(default_factory=list)
    description: str | None = None

    def add(
        self,
        now: float,
        event: dict[str, Any] | None = None,
    ) -> FixtureSpec:
        self.steps.append({"now": now, "event": event})
        return self


def _round(x: float) -> float:
    return round(x, FLOAT_NDIGITS)


def _serialize_outcome(o: Outcome) -> dict[str, Any]:
    """Convert an Outcome into the JSON ``expect`` dict.

    Only JSON-primitive meta values (bool/int/str/float) are preserved.
    """
    out: dict[str, Any] = {"reinforced": o.reinforced}
    if o.reinforcer is not None:
        out["reinforcer"] = {
            "time": _round(o.reinforcer.time),
            "magnitude": _round(o.reinforcer.magnitude),
            "label": o.reinforcer.label,
        }
    if o.meta:
        meta_out: dict[str, Any] = {}
        for k, v in o.meta.items():
            if isinstance(v, (bool, int, str)):
                meta_out[k] = v
            elif isinstance(v, float):
                meta_out[k] = _round(v)
            else:
                # Skip non-primitive meta to keep the corpus
                # language-neutral. Document any such meta in the
                # schedule's algorithm doc.
                continue
        if meta_out:
            out["meta"] = meta_out
    return out


def _run_fixture(fx: FixtureSpec) -> dict[str, Any]:
    schedule = build_schedule_from_spec(fx.schedule)
    serialized_steps: list[dict[str, Any]] = []
    for step in fx.steps:
        event_dict = step["event"]
        event = (
            ResponseEvent(
                time=event_dict["time"],
                operandum=event_dict.get("operandum", "main"),
            )
            if event_dict is not None
            else None
        )
        outcome = schedule.step(step["now"], event)
        serialized_steps.append(
            {
                "now": _round(step["now"]),
                "event": (
                    {
                        "time": _round(event_dict["time"]),
                        "operandum": event_dict.get("operandum", "main"),
                    }
                    if event_dict is not None
                    else None
                ),
                "expect": _serialize_outcome(outcome),
            }
        )
    payload: dict[str, Any] = {
        "name": fx.name,
        "schedule": fx.schedule,
        "steps": serialized_steps,
    }
    if fx.description is not None:
        payload["description"] = fx.description
    return payload


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Fixture definitions
# ---------------------------------------------------------------------------


def _response_seq(times: Iterable[float]) -> list[dict[str, Any]]:
    return [{"time": t, "operandum": "main"} for t in times]


def define_fixtures() -> list[tuple[Path, FixtureSpec]]:
    """Define every fixture in the corpus.

    Returns a list of ``(relative_path, FixtureSpec)`` pairs.
    """
    fixtures: list[tuple[Path, FixtureSpec]] = []

    # --- Atomic / deterministic ---------------------------------------

    fr5 = FixtureSpec(
        name="fr5-basic",
        description="FR 5: every 5th response produces a reinforcer.",
        schedule={"type": "FR", "params": {"n": 5}},
    )
    for i, t in enumerate([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]):
        event = {"time": t, "operandum": "main"} if i > 0 else None
        fr5.add(now=t, event=event)
    fixtures.append((CONFORMANCE_DIR / "atomic" / "fr_basic.json", fr5))

    crf = FixtureSpec(
        name="crf-basic",
        description="CRF (FR 1): every response reinforces.",
        schedule={"type": "CRF", "params": {}},
    )
    for t in [0.0, 1.0, 2.0, 3.0]:
        crf.add(now=t, event={"time": t, "operandum": "main"})
    fixtures.append((CONFORMANCE_DIR / "atomic" / "crf_basic.json", crf))

    ext = FixtureSpec(
        name="ext-basic",
        description="EXT: no response is ever reinforced.",
        schedule={"type": "EXT", "params": {}},
    )
    for t in [0.0, 1.0, 2.0, 3.0]:
        ext.add(now=t, event={"time": t, "operandum": "main"})
    ext.add(now=4.0, event=None)
    fixtures.append((CONFORMANCE_DIR / "atomic" / "ext_basic.json", ext))

    fi10 = FixtureSpec(
        name="fi10-basic",
        description=(
            "FI 10: response after 10s elapsed reinforces; responses "
            "before the interval do not reset the clock."
        ),
        schedule={"type": "FI", "params": {"interval": 10.0}},
    )
    for t in [0.0, 5.0, 9.0, 10.0, 15.0, 20.0, 21.0]:
        fi10.add(now=t, event={"time": t, "operandum": "main"})
    fixtures.append((CONFORMANCE_DIR / "atomic" / "fi_basic.json", fi10))

    ft5 = FixtureSpec(
        name="ft5-basic",
        description=(
            "FT 5: response-independent, reinforces every 5s from the "
            "first step anchor."
        ),
        schedule={"type": "FT", "params": {"interval": 5.0}},
    )
    for t in [0.0, 2.5, 5.0, 7.5, 10.0, 12.5]:
        ft5.add(now=t, event=None)
    fixtures.append((CONFORMANCE_DIR / "atomic" / "ft_basic.json", ft5))

    # --- Stochastic / seeded ------------------------------------------

    vr5_seeded = FixtureSpec(
        name="vr5-seeded-42",
        description=(
            "VR 5, n_intervals=12, seed=42. Deterministic trajectory "
            "pinned by the Python Random(seed) stream."
        ),
        schedule={
            "type": "VR",
            "params": {"mean": 5, "n_intervals": 12, "seed": 42},
        },
    )
    # 100 responses at t = 1..100; covers >= one full cycle of 12.
    for t in [float(i) for i in range(1, 101)]:
        vr5_seeded.add(now=t, event={"time": t, "operandum": "main"})
    fixtures.append((CONFORMANCE_DIR / "atomic" / "vr_seeded_42.json", vr5_seeded))

    vi10_seeded = FixtureSpec(
        name="vi10-seeded-7",
        description="VI 10, n_intervals=12, seed=7. Deterministic trajectory.",
        schedule={
            "type": "VI",
            "params": {"mean_interval": 10.0, "n_intervals": 12, "seed": 7},
        },
    )
    # Step every 1.0 time units with a response each step for 80 s.
    for t in [float(i) for i in range(0, 81)]:
        vi10_seeded.add(now=t, event={"time": t, "operandum": "main"})
    fixtures.append((CONFORMANCE_DIR / "atomic" / "vi_seeded_7.json", vi10_seeded))

    vt10_seeded = FixtureSpec(
        name="vt10-seeded-3",
        description="VT 10, n_intervals=12, seed=3. Response-independent.",
        schedule={
            "type": "VT",
            "params": {"mean_interval": 10.0, "n_intervals": 12, "seed": 3},
        },
    )
    for t in [float(i) for i in range(0, 81)]:
        vt10_seeded.add(now=t, event=None)
    fixtures.append((CONFORMANCE_DIR / "atomic" / "vt_seeded_3.json", vt10_seeded))

    rr_seeded = FixtureSpec(
        name="rr-seeded-99",
        description="RR p=0.2 seeded via Random(99) for deterministic draws.",
        schedule={
            "type": "RR",
            "params": {"probability": 0.2, "seed": 99},
        },
    )
    for t in [float(i) for i in range(1, 61)]:
        rr_seeded.add(now=t, event={"time": t, "operandum": "main"})
    fixtures.append((CONFORMANCE_DIR / "atomic" / "rr_seeded_99.json", rr_seeded))

    ri_seeded = FixtureSpec(
        name="ri-seeded-5",
        description="RI mean=5s, Random(5). Responses every 1s.",
        schedule={
            "type": "RI",
            "params": {"mean_interval": 5.0, "seed": 5},
        },
    )
    for t in [float(i) for i in range(0, 61)]:
        ri_seeded.add(now=t, event={"time": t, "operandum": "main"})
    fixtures.append((CONFORMANCE_DIR / "atomic" / "ri_seeded_5.json", ri_seeded))

    rt_seeded = FixtureSpec(
        name="rt-seeded-11",
        description="RT mean=3s, Random(11). Pure ticks.",
        schedule={
            "type": "RT",
            "params": {"mean_interval": 3.0, "seed": 11},
        },
    )
    for t in [float(i) * 0.5 for i in range(0, 81)]:  # 0..40 s, 0.5s ticks
        rt_seeded.add(now=t, event=None)
    fixtures.append((CONFORMANCE_DIR / "atomic" / "rt_seeded_11.json", rt_seeded))

    # --- LimitedHold --------------------------------------------------

    lh_fi = FixtureSpec(
        name="limited-hold-fi5-hold2",
        description=(
            "LimitedHold wrapping FI 5 with a 2s availability window. "
            "A response at t=6 reinforces; a response at t=8 (outside "
            "the window opened at t=10) does not; ticks at later times "
            "withdraw expired opportunities."
        ),
        schedule={
            "type": "LimitedHold",
            "params": {
                "hold": 2.0,
                "inner": {"type": "FI", "params": {"interval": 5.0}},
            },
        },
    )
    # Initial arm_time = 5.0.
    for t in [0.0, 1.0, 4.0]:
        lh_fi.add(now=t, event={"time": t, "operandum": "main"})
    lh_fi.add(now=6.0, event={"time": 6.0, "operandum": "main"})   # reinforce
    # Next arm_time = 6.0 + 5.0 = 11.0. Hold window = [11, 13].
    lh_fi.add(now=10.0, event={"time": 10.0, "operandum": "main"})  # no
    lh_fi.add(now=11.5, event={"time": 11.5, "operandum": "main"})  # reinforce
    # New arm_time = 16.5. Hold = [16.5, 18.5].
    lh_fi.add(now=20.0, event=None)     # expiry on tick: inner rearmed at 20
    # New arm_time after expiry = 20 + 5 = 25. No response before 30 -> expiry
    lh_fi.add(now=30.0, event={"time": 30.0, "operandum": "main"})  # no (outside)
    fixtures.append(
        (CONFORMANCE_DIR / "atomic" / "limited_hold_fi.json", lh_fi)
    )

    # --- Compound: Concurrent + COD -----------------------------------

    conc_cod = FixtureSpec(
        name="concurrent-fr3-fr3-cod1",
        description=(
            "Concurrent FR(3)/FR(3) with COD=1.0: after switching, "
            "reinforcement on the new operandum is suppressed for 1s."
        ),
        schedule={
            "type": "Concurrent",
            "params": {
                "components": {
                    "left": {"type": "FR", "params": {"n": 3}},
                    "right": {"type": "FR", "params": {"n": 3}},
                },
                "cod": 1.0,
                "frco": 0,
            },
        },
    )
    # Left 3 responses -> reinforcer on left.
    for t in [1.0, 2.0, 3.0]:
        conc_cod.add(now=t, event={"time": t, "operandum": "left"})
    # Switch to right; 3 responses within 1s of switch -> suppressed by COD
    # when the 3rd response completes because the switch anchored the COD
    # at the first right-press.
    conc_cod.add(now=4.0, event={"time": 4.0, "operandum": "right"})
    conc_cod.add(now=4.3, event={"time": 4.3, "operandum": "right"})
    conc_cod.add(now=4.6, event={"time": 4.6, "operandum": "right"})  # suppressed
    # Another FR3 on right, this time well past the COD window.
    conc_cod.add(now=10.0, event={"time": 10.0, "operandum": "right"})
    conc_cod.add(now=10.2, event={"time": 10.2, "operandum": "right"})
    conc_cod.add(now=10.4, event={"time": 10.4, "operandum": "right"})  # reinforced
    fixtures.append(
        (CONFORMANCE_DIR / "compound" / "concurrent_cod.json", conc_cod)
    )

    # --- Compound: Chained --------------------------------------------

    chained = FixtureSpec(
        name="chained-fr2-fr3",
        description=(
            "Chained FR(2) -> FR(3): 2 responses complete the first link "
            "(conditioned reinforcement, no primary), 3 more responses "
            "complete the terminal and deliver primary SR+."
        ),
        schedule={
            "type": "Chained",
            "params": {
                "components": [
                    {"type": "FR", "params": {"n": 2}},
                    {"type": "FR", "params": {"n": 3}},
                ],
                "stimuli": ["s0", "s1"],
            },
        },
    )
    # Link 0: 2 responses -> transition, no primary.
    chained.add(now=1.0, event={"time": 1.0, "operandum": "main"})
    chained.add(now=2.0, event={"time": 2.0, "operandum": "main"})
    # Link 1: 3 responses -> primary SR+.
    chained.add(now=3.0, event={"time": 3.0, "operandum": "main"})
    chained.add(now=4.0, event={"time": 4.0, "operandum": "main"})
    chained.add(now=5.0, event={"time": 5.0, "operandum": "main"})
    # Cycle back: 2 responses -> transition again.
    chained.add(now=6.0, event={"time": 6.0, "operandum": "main"})
    chained.add(now=7.0, event={"time": 7.0, "operandum": "main"})
    fixtures.append(
        (CONFORMANCE_DIR / "compound" / "chained_fr2_fr3.json", chained)
    )

    # --- Compound: Multiple -------------------------------------------

    multiple = FixtureSpec(
        name="multiple-fr2-fr3",
        description=(
            "Multiple FR(2) / FR(3): each component has its own S^D and "
            "produces primary reinforcement when its ratio is met. On "
            "reinforcement the active index advances (wraps after the "
            "terminal component)."
        ),
        schedule={
            "type": "Multiple",
            "params": {
                "components": [
                    {"type": "FR", "params": {"n": 2}},
                    {"type": "FR", "params": {"n": 3}},
                ],
                "stimuli": ["s0", "s1"],
            },
        },
    )
    # Component 0 (FR 2) active: responses 1, 2 -> 2nd reinforces; advance to
    # component 1.
    multiple.add(now=1.0, event={"time": 1.0, "operandum": "main"})
    multiple.add(now=2.0, event={"time": 2.0, "operandum": "main"})
    # Component 1 (FR 3) active: responses 3, 4, 5 -> 5th reinforces; wrap
    # back to component 0.
    multiple.add(now=3.0, event={"time": 3.0, "operandum": "main"})
    multiple.add(now=4.0, event={"time": 4.0, "operandum": "main"})
    multiple.add(now=5.0, event={"time": 5.0, "operandum": "main"})
    # Cycle back: first response of the new cycle on component 0.
    multiple.add(now=6.0, event={"time": 6.0, "operandum": "main"})
    fixtures.append(
        (CONFORMANCE_DIR / "compound" / "multiple_fr_fr.json", multiple)
    )

    # --- Compound: Tandem ---------------------------------------------

    tandem = FixtureSpec(
        name="tandem-fr2-fr3",
        description=(
            "Tandem FR(2) -> FR(3): structurally a chain but with no "
            "distinctive stimuli. The non-terminal FR(2) transitions "
            "silently (no primary SR+); the terminal FR(3) delivers "
            "primary SR+ on the 5th response overall."
        ),
        schedule={
            "type": "Tandem",
            "params": {
                "components": [
                    {"type": "FR", "params": {"n": 2}},
                    {"type": "FR", "params": {"n": 3}},
                ],
            },
        },
    )
    # Link 0 (FR 2): 2 responses -> silent transition to link 1.
    tandem.add(now=1.0, event={"time": 1.0, "operandum": "main"})
    tandem.add(now=2.0, event={"time": 2.0, "operandum": "main"})
    # Link 1 (FR 3): 3 responses -> primary SR+ on the 5th overall.
    tandem.add(now=3.0, event={"time": 3.0, "operandum": "main"})
    tandem.add(now=4.0, event={"time": 4.0, "operandum": "main"})
    tandem.add(now=5.0, event={"time": 5.0, "operandum": "main"})
    # Cycle back: 2 responses -> silent transition again.
    tandem.add(now=6.0, event={"time": 6.0, "operandum": "main"})
    tandem.add(now=7.0, event={"time": 7.0, "operandum": "main"})
    fixtures.append(
        (CONFORMANCE_DIR / "compound" / "tandem_fr_fr.json", tandem)
    )

    # --- Compound: Alternative ----------------------------------------

    alt = FixtureSpec(
        name="alternative-fr5-ft10",
        description=(
            "Alternative(FR(5), FT(10)): whichever fires first wins; "
            "both are reset on every win. FR(5) wins if 5 responses "
            "arrive before 10s; FT(10) wins on ticks after 10s anchor."
        ),
        schedule={
            "type": "Alternative",
            "params": {
                "first": {"type": "FR", "params": {"n": 5}},
                "second": {"type": "FT", "params": {"interval": 10.0}},
            },
        },
    )
    # 5 responses before 10s anchor -> first wins at t=5.
    for t in [1.0, 2.0, 3.0, 4.0, 5.0]:
        alt.add(now=t, event={"time": t, "operandum": "main"})
    # After reset, FT anchors at next step. Only tick -> FT wins after
    # 10s have elapsed from first post-reset step.
    alt.add(now=6.0, event=None)    # anchors FT at 6
    alt.add(now=15.0, event=None)   # FT fires (response-independent)
    fixtures.append(
        (CONFORMANCE_DIR / "compound" / "alternative_fr_ft.json", alt)
    )

    # --- Differential: DRO resetting & momentary -----------------------

    dro_reset = FixtureSpec(
        name="dro-resetting-5",
        description=(
            "DRO 5s resetting: response resets timer; 5s without "
            "response -> reinforcer."
        ),
        schedule={
            "type": "DRO",
            "params": {"interval": 5.0, "dro_type": "resetting"},
        },
    )
    # Anchor at t=0; tick at t=3 -> no fire.
    dro_reset.add(now=0.0, event=None)
    dro_reset.add(now=3.0, event=None)
    # Response at t=4 resets timer -> no fire.
    dro_reset.add(now=4.0, event={"time": 4.0, "operandum": "main"})
    # Tick at 9.0 (5s after reset) -> fire.
    dro_reset.add(now=9.0, event=None)
    # Tick at 14.0 (5s after reinforcer) -> fire again.
    dro_reset.add(now=14.0, event=None)
    fixtures.append(
        (CONFORMANCE_DIR / "differential" / "dro_resetting.json", dro_reset)
    )

    dro_mom = FixtureSpec(
        name="dro-momentary-5",
        description=(
            "DRO 5s momentary: boundary check every 5s; reinforce iff "
            "no response in [anchor, now)."
        ),
        schedule={
            "type": "DRO",
            "params": {"interval": 5.0, "dro_type": "momentary"},
        },
    )
    dro_mom.add(now=0.0, event=None)
    # Response at t=2 -> blocks window 1.
    dro_mom.add(now=2.0, event={"time": 2.0, "operandum": "main"})
    dro_mom.add(now=5.0, event=None)   # boundary: response occurred -> no
    # Window 2: no responses before boundary 10.
    dro_mom.add(now=10.0, event=None)  # reinforce
    # Window 3: response at 12 -> blocks.
    dro_mom.add(now=12.0, event={"time": 12.0, "operandum": "main"})
    dro_mom.add(now=15.0, event=None)  # no
    fixtures.append(
        (CONFORMANCE_DIR / "differential" / "dro_momentary.json", dro_mom)
    )

    drl = FixtureSpec(
        name="drl-3",
        description="DRL 3: reinforce if IRT >= 3s.",
        schedule={"type": "DRL", "params": {"interval": 3.0}},
    )
    drl.add(now=0.0, event={"time": 0.0, "operandum": "main"})  # first -> yes
    drl.add(now=1.0, event={"time": 1.0, "operandum": "main"})  # IRT 1 -> no
    drl.add(now=5.0, event={"time": 5.0, "operandum": "main"})  # IRT 4 -> yes
    drl.add(now=6.0, event={"time": 6.0, "operandum": "main"})  # IRT 1 -> no
    drl.add(now=9.0, event={"time": 9.0, "operandum": "main"})  # IRT 3 -> yes
    fixtures.append(
        (CONFORMANCE_DIR / "differential" / "drl_basic.json", drl)
    )

    drh = FixtureSpec(
        name="drh-3-per-2",
        description="DRH 3 responses within 2s sliding window.",
        schedule={
            "type": "DRH",
            "params": {"response_count": 3, "time_window": 2.0},
        },
    )
    drh.add(now=0.0, event={"time": 0.0, "operandum": "main"})
    drh.add(now=0.5, event={"time": 0.5, "operandum": "main"})
    drh.add(now=1.0, event={"time": 1.0, "operandum": "main"})  # reinforce
    drh.add(now=1.5, event={"time": 1.5, "operandum": "main"})  # reinforce
    drh.add(now=5.0, event={"time": 5.0, "operandum": "main"})  # lone -> no
    fixtures.append(
        (CONFORMANCE_DIR / "differential" / "drh_basic.json", drh)
    )

    # --- Progressive Ratio --------------------------------------------

    pr = FixtureSpec(
        name="pr-arithmetic-1-1",
        description=(
            "ProgressiveRatio arithmetic(start=1, step=1): requirements "
            "1, 2, 3, 4, 5, ..."
        ),
        schedule={
            "type": "ProgressiveRatio",
            "params": {"step": "arithmetic", "start": 1, "step_size": 1},
        },
    )
    # 15 responses; reinforces on counts 1, 1+2=3, 3+3=6, 6+4=10, 10+5=15.
    for t in [float(i) for i in range(1, 16)]:
        pr.add(now=t, event={"time": t, "operandum": "main"})
    fixtures.append(
        (CONFORMANCE_DIR / "progressive" / "pr_arithmetic.json", pr)
    )

    return fixtures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    fixtures = define_fixtures()
    for path, fx in fixtures:
        payload = _run_fixture(fx)
        _write(path, payload)
    print(f"Wrote {len(fixtures)} fixtures under {CONFORMANCE_DIR}")


if __name__ == "__main__":
    main()
