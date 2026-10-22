"""S-126 prototype: proposed spec forms for the 15 schedule classes without a fixture.

Reuses the generator's serialisation rules (9-digit floats, primitive meta only)
but writes to the scratchpad, never to the repo.
"""
from __future__ import annotations
import json, sys, random
from pathlib import Path
from typing import Any

sys.path.insert(0, "/home/user/contingency-py/scripts")
import generate_conformance as gen  # noqa: E402
from contingency.conformance import spec as specmod  # noqa: E402
from contingency.conformance import is_stochastic  # noqa: E402
from contingency.schedules import (  # noqa: E402
    AdjustingSchedule, Sidman, DiscriminatedAvoidance, Escape, Conjunctive,
    InterlockingSchedule, Interpolate, Mixed, Overlay, Percentile, ResponseCost,
    SecondOrder, Timeout, MatchingToSample, GoNoGo,
)

_orig = specmod.build_schedule_from_spec


def _opt(params, key):
    return _orig(params[key]) if params.get(key) is not None else None


def build(spec: dict[str, Any]):
    t, p = spec["type"], spec.get("params", {})
    if t == "AdjustingSchedule":
        return AdjustingSchedule(p["target"], float(p["start"]), float(p["step"]),
                                 minimum=p.get("minimum"), maximum=p.get("maximum"))
    if t == "Sidman":
        return Sidman(float(p["shock_shock_interval"]), float(p["response_shock_interval"]),
                      shock_magnitude=float(p.get("shock_magnitude", 1.0)))
    if t == "DiscriminatedAvoidance":
        return DiscriminatedAvoidance(float(p["warning_duration"]), float(p["iti"]),
                                      shock_magnitude=float(p.get("shock_magnitude", 1.0)))
    if t == "Escape":
        return Escape(float(p["trial_duration"]), float(p["iti"]),
                      shock_magnitude=float(p.get("shock_magnitude", 1.0)))
    if t == "Conjunctive":
        return Conjunctive(build(p["first"]), build(p["second"]))
    if t == "InterlockingSchedule":
        return InterlockingSchedule(int(p["initial_ratio"]), float(p["decay_time"]))
    if t == "Interpolate":
        return Interpolate(build(p["base"]), build(p["probe"]), float(p["interval"]),
                           float(p["probe_duration"]), first_probe_at=p.get("first_probe_at"))
    if t == "Mixed":
        return Mixed([build(c) for c in p["components"]])
    if t == "Overlay":
        return Overlay(build(p["base"]), build(p["punisher"]))
    if t == "Percentile":
        return Percentile(p.get("target", "irt"), int(p["rank"]), int(p["window"]),
                          direction=p.get("direction", "above"))
    if t == "ResponseCost":
        return ResponseCost(build(p["inner"]), float(p["amount"]), unit=p.get("unit", "token"),
                            initial_balance=p.get("initial_balance"))
    if t == "SecondOrder":
        return SecondOrder(build(p["overall"]), build(p["unit"]))
    if t == "Timeout":
        return Timeout(build(p["inner"]), float(p["duration"]),
                       reset_on_response=bool(p.get("reset_on_response", False)))
    if t == "MatchingToSample":
        return MatchingToSample(int(p["n_comparisons"]),
                                sample_duration=float(p.get("sample_duration", 0.0)),
                                choice_timeout=float(p.get("choice_timeout", 5.0)),
                                consequence=_opt(p, "consequence"), incorrect=_opt(p, "incorrect"),
                                iti=float(p.get("iti", 1.0)), seed=p.get("seed"))
    if t == "GoNoGo":
        return GoNoGo(go_probability=float(p.get("go_probability", 0.5)),
                      response_window=float(p.get("response_window", 5.0)),
                      iti=float(p.get("iti", 5.0)),
                      correct_go_schedule=_opt(p, "correct_go_schedule"),
                      correct_nogo_schedule=_opt(p, "correct_nogo_schedule"),
                      false_alarm=_opt(p, "false_alarm"), seed=p.get("seed"))
    return _orig(spec)


# Make nested specs (e.g. Conjunctive(FR, FI)) resolve through the extended builder
# the same way the real builder recurses into itself.
specmod.build_schedule_from_spec = build
gen.build_schedule_from_spec = build

OUT = Path(__file__).resolve().parent / "fixtures"
F = gen.FixtureSpec


def R(fx, t, op="main"):
    fx.add(now=float(t), event={"time": float(t), "operandum": op})


def T(fx, t):
    fx.add(now=float(t), event=None)


def define() -> list[tuple[str, F]]:
    out = []
    # --- group 1: deterministic wrappers / compounds ---------------------
    fx = F("mixed-fr2-fr3", {"type": "Mixed", "params": {"components": [
        {"type": "FR", "params": {"n": 2}}, {"type": "FR", "params": {"n": 3}}]}},
        description="Mixed FR(2)/FR(3): alternation on reinforcement without S^D; meta current_component is an int.")
    for t in range(1, 8):
        R(fx, t)
    out.append(("compound/mixed_fr_fr.json", fx))

    fx = F("timeout-fr2-to3", {"type": "Timeout", "params": {
        "inner": {"type": "FR", "params": {"n": 2}}, "duration": 3.0, "reset_on_response": False}},
        description="Timeout 3s wrapping FR 2: responses during timeout are dropped (meta timeout_active).")
    for t in [1, 2, 3, 4]:
        R(fx, t)
    R(fx, 5); R(fx, 6); T(fx, 7); R(fx, 10); R(fx, 11)
    out.append(("wrapper/timeout_fr.json", fx))

    fx = F("conjunctive-fr3-fi5", {"type": "Conjunctive", "params": {
        "first": {"type": "FR", "params": {"n": 3}}, "second": {"type": "FI", "params": {"interval": 5.0}}}},
        description="Conjunctive FR(3) AND FI(5): reinforced only on a step where both fire (5, 8, 11); both reset. FI arms at the absolute time 5 and, after reset, is immediately satisfiable.")
    for t in [1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13]:
        R(fx, t)
    out.append(("compound/conjunctive_fr_fi.json", fx))

    fx = F("interlocking-r10-t20", {"type": "InterlockingSchedule", "params": {"initial_ratio": 10, "decay_time": 20.0}},
        description="Interlocking R0=10 decaying to 0 over 20s: requirement int(10*(1-t/20)).")
    T(fx, 0)
    for t in [2, 4, 6, 8, 10, 12, 13, 20, 30]:
        R(fx, t)
    T(fx, 50); R(fx, 52)
    out.append(("atomic/interlocking_basic.json", fx))

    fx = F("response-cost-fr2-balance3", {"type": "ResponseCost", "params": {
        "inner": {"type": "FR", "params": {"n": 2}}, "amount": 1.0, "unit": "token", "initial_balance": 3.0}},
        description="ResponseCost 1 token per response, balance 3, wrapping FR 2; 4th response is unaffordable (meta cost_insufficient).")
    for t in [1, 2, 3, 4]:
        R(fx, t)
    T(fx, 5); R(fx, 6)
    out.append(("wrapper/response_cost_fr.json", fx))

    # --- group 2: deterministic, state-rich ------------------------------
    fx = F("adjusting-ratio-2-step1-max4", {"type": "AdjustingSchedule", "params": {
        "target": "ratio", "start": 2, "step": 1, "maximum": 4}},
        description="Adjusting ratio start=2 step=+1 max=4: requirements 2, 3, 4, 4, ...")
    for t in range(1, 16):
        R(fx, t)
    out.append(("progressive/adjusting_ratio.json", fx))

    fx = F("interpolate-fr2-ext-10-3", {"type": "Interpolate", "params": {
        "base": {"type": "FR", "params": {"n": 2}}, "probe": {"type": "EXT", "params": {}},
        "interval": 10.0, "probe_duration": 3.0, "first_probe_at": 5.0}},
        description="Interpolate base FR(2), probe EXT, windows [5,8) and [15,18) from the anchor; meta in_probe.")
    for t in range(0, 21):
        R(fx, t)
    out.append(("compound/interpolate_fr_ext.json", fx))

    fx = F("overlay-fr3-fr2", {"type": "Overlay", "params": {
        "base": {"type": "FR", "params": {"n": 3}}, "punisher": {"type": "FR", "params": {"n": 2}}}},
        description="Overlay base FR(3) with punisher FR(2): SR+ on 3, SR- on 2 and 4, SR0 (net 0) on 6.")
    for t in range(1, 10):
        R(fx, t)
    out.append(("compound/overlay_fr_fr.json", fx))

    fx = F("percentile-irt-50-w4-above", {"type": "Percentile", "params": {
        "target": "irt", "rank": 50, "window": 4, "direction": "above"}},
        description="Percentile IRT, rank 50, window 4, above: first 5 responses build the baseline; later IRTs >= median reinforce.")
    for t in [0, 1, 3, 4, 7, 8, 12, 13, 19, 20, 28, 29, 30]:
        R(fx, t)
    out.append(("differential/percentile_irt.json", fx))

    fx = F("second-order-fr2-of-fr3", {"type": "SecondOrder", "params": {
        "overall": {"type": "FR", "params": {"n": 2}}, "unit": {"type": "FR", "params": {"n": 3}}}},
        description="Second-order FR2(FR3): unit completions on responses 3 and 6; the 2nd completion reinforces (meta second_order).")
    for t in range(1, 10):
        R(fx, t)
    out.append(("compound/second_order_fr_fr.json", fx))

    # --- group 3: aversive control and trial-based -----------------------
    fx = F("sidman-ss5-rs8", {"type": "Sidman", "params": {
        "shock_shock_interval": 5.0, "response_shock_interval": 8.0, "shock_magnitude": 1.0}},
        description="Sidman avoidance SSI=5 RSI=8: shocks (SR-, magnitude -1) at 5 and 15; the response at 7 postpones the next shock to 15.")
    T(fx, 0); T(fx, 3); T(fx, 5); R(fx, 7); T(fx, 10); T(fx, 15); T(fx, 18); T(fx, 20)
    out.append(("aversive/sidman_basic.json", fx))

    fx = F("discriminated-avoidance-w3-iti5", {"type": "DiscriminatedAvoidance", "params": {
        "warning_duration": 3.0, "iti": 5.0, "shock_magnitude": 1.0}},
        description="Discriminated avoidance: first warning at 5; response at 6 avoids; second warning at 11 ends in shock at 14.")
    T(fx, 0); T(fx, 5); R(fx, 6); T(fx, 11); T(fx, 12); T(fx, 14); T(fx, 19); R(fx, 20)
    out.append(("aversive/discriminated_avoidance_basic.json", fx))

    fx = F("escape-td3-iti4", {"type": "Escape", "params": {
        "trial_duration": 3.0, "iti": 4.0, "shock_magnitude": 1.0}},
        description="Escape: shock on every tick until the response at 2 terminates it; the second trial (from 6) times out at 9.")
    T(fx, 0); T(fx, 1); R(fx, 2); T(fx, 6); T(fx, 7); T(fx, 9); T(fx, 13)
    out.append(("aversive/escape_basic.json", fx))

    fx = F("mts-2-seeded-1", {"type": "MatchingToSample", "params": {
        "n_comparisons": 2, "sample_duration": 1.0, "choice_timeout": 3.0, "iti": 2.0, "seed": 1}},
        description="MTS with 2 comparisons, seed=1 (stochastic: the correct comparison is drawn per trial). The subject always chooses choice_0; trial 3 times out.")
    t = 0.0
    T(fx, 0); T(fx, 1); R(fx, 2, "choice_0")      # trial 0: sample [0,1), choice, response at 2, iti [2,4)
    T(fx, 4); T(fx, 5); R(fx, 6, "choice_0")      # trial 1
    T(fx, 8); T(fx, 9); R(fx, 10, "choice_0")     # trial 2
    T(fx, 12); T(fx, 13); T(fx, 17)               # trial 3: choice window [13,16] times out
    T(fx, 19); T(fx, 20); R(fx, 21, "choice_0")   # trial 4
    out.append(("trial/mts_seeded_1.json", fx))

    fx = F("gonogo-p05-seeded-3", {"type": "GoNoGo", "params": {
        "go_probability": 0.5, "response_window": 2.0, "iti": 1.0, "seed": 3}},
        description="Go/NoGo p=0.5, window 2, ITI 1, seed=3 (stochastic: trial type drawn per trial). Responses in trials 0, 2, 4; withholding in 1, 3.")
    # trial k occupies [3k, 3k+2) window then [3k+2, 3k+3) iti
    for k in range(5):
        base = 3 * k
        T(fx, base)
        if k % 2 == 0:
            R(fx, base + 1)
        else:
            T(fx, base + 1)
        T(fx, base + 2)
    T(fx, 15)
    out.append(("trial/gonogo_seeded_3.json", fx))
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for rel, fx in define():
        payload = gen._run_fixture(fx)
        again = gen._run_fixture(fx)  # re-run from a fresh schedule -> must be identical
        # Re-run via the generic replay path (same as tests/test_conformance.py)
        path = OUT / rel
        gen._write(path, payload)
        n_r = sum(1 for s in payload["steps"] if s["expect"]["reinforced"])
        metas = sorted({k for s in payload["steps"] for k in s["expect"].get("meta", {})})
        rows.append((rel, payload["schedule"]["type"], len(payload["steps"]), n_r,
                     "stochastic" if is_stochastic(payload["schedule"]) else "deterministic",
                     payload == again, ",".join(metas)))
    print(f"{'fixture':44} {'type':24} steps reinf kind          same meta")
    for r in rows:
        print(f"{r[0]:44} {r[1]:24} {r[2]:5} {r[3]:5} {r[4]:13} {r[5]!s:5} {r[6]}")


if __name__ == "__main__":
    main()
