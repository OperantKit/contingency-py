"""Replay responses recorded by a MED-PC apparatus through the engine.

Each fixture under ``fixtures/apparatus_replay/`` holds the response times of one
published session (integer ticks) and the indices of the responses the apparatus
reinforced. The test feeds the same responses to :class:`DRL` or
:class:`ProgressiveRatio` and compares the reinforced responses.

The apparatus programs differ from the engine in two documented ways, which the
small adapters below reproduce instead of changing the engine:

- DRL 10 s: the IRT clock starts at the onset of the DRL phase (one anchor
  response at that tick), and after each reinforcer the clock stops for 4 s;
  responses in those 4 s are not passed on and the clock restarts at the end.
- PR (1, 4, 7, ...): responses within 5.1 s after a reinforcer (a post-reinforcer
  timeout the engine does not have) are not passed on.

Replaying the full datasets this way (928 DRL sessions, 30,578 reinforcers;
273 PR sessions, 3,979 reinforcers) matched 99.30% and 99.75% of the
reinforcers. Every disagreement was explained by the apparatus's timing
resolution at the 10 s boundary or at the timeout boundary, and the responses
that follow it. The expected counts in each fixture are that replay's counts
for the session.

Data
----
Sosa, R., Mata-Luévanos, J., Tapia-de-Jesús, A., Saavedra, P., &
Buenrostro-Jáuregui, M. (2026). *Longitudinal association between different
indices of behavioral inhibition* [Data set]. Zenodo.
https://doi.org/10.5281/zenodo.19164959 (CC BY 4.0)

McCutcheon, J. (2022). *Progressive ratio in protein-restricted vs.
non-restricted rats* [Data set]. Zenodo.
https://doi.org/10.5281/zenodo.5409201 (CC BY 4.0)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from contingency.entities import ResponseEvent
from contingency.schedules import DRL, ProgressiveRatio, arithmetic

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "apparatus_replay"
DRL_FIXTURES = sorted(FIXTURE_DIR.glob("drl_*.json"))
PR_FIXTURES = sorted(FIXTURE_DIR.glob("pr_*.json"))


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def replay_drl(fx: dict[str, Any]) -> set[int]:
    """Indices of the responses the engine reinforces, with the apparatus clock rules."""
    drl = DRL(fx["interval_ticks"])
    onset = fx["phase_onset_tick"]
    lock = fx["post_reinforcement_lock_ticks"]
    drl.step(onset, ResponseEvent(time=onset, operandum=0))
    reinforced: set[int] = set()
    lock_until: int | None = None
    for i, t in enumerate(fx["response_ticks"]):
        if lock_until is not None and t <= lock_until:
            continue
        if drl.step(t, ResponseEvent(time=t, operandum=0)).reinforced:
            reinforced.add(i)
            lock_until = t + lock
            drl.step(lock_until, ResponseEvent(time=lock_until, operandum=0))
    return reinforced


def replay_pr(fx: dict[str, Any]) -> tuple[set[int], int]:
    """Indices the engine reinforces, and the ratio it would require next."""
    pr = ProgressiveRatio(arithmetic(start=fx["ratio_start"], step=fx["ratio_step"]))
    timeout = fx["post_reinforcement_timeout_ticks"]
    reinforced: set[int] = set()
    last_rf: int | None = None
    for i, t in enumerate(fx["response_ticks"]):
        if last_rf is not None and last_rf < t < last_rf + timeout:
            continue
        if pr.step(t, ResponseEvent(time=t, operandum=0)).reinforced:
            reinforced.add(i)
            last_rf = t + fx["reinforcer_lag_ticks"]
    return reinforced, fx["ratio_start"] + fx["ratio_step"] * pr.reinforcement_index


def _counts(recorded: set[int], engine: set[int]) -> dict[str, int]:
    return {
        "recorded": len(recorded),
        "matched": len(recorded & engine),
        "missed": len(recorded - engine),
        "extra": len(engine - recorded),
    }


def test_fixtures_present() -> None:
    assert len(DRL_FIXTURES) == 2
    assert len(PR_FIXTURES) == 2


@pytest.mark.parametrize("path", DRL_FIXTURES, ids=lambda p: p.stem)
def test_drl_replay_matches_apparatus(path: Path) -> None:
    fx = _load(path)
    recorded = set(fx["recorded_reinforced_responses"])
    assert _counts(recorded, replay_drl(fx)) == fx["expected"]


@pytest.mark.parametrize("path", PR_FIXTURES, ids=lambda p: p.stem)
def test_pr_replay_matches_apparatus(path: Path) -> None:
    fx = _load(path)
    recorded = set(fx["recorded_reinforced_responses"])
    engine, next_ratio = replay_pr(fx)
    assert _counts(recorded, engine) == fx["expected"]
    assert next_ratio == fx["recorded_next_ratio"]


def test_drl_without_adapter_disagrees() -> None:
    """Without the onset anchor and the 4 s lock the engine reinforces extra responses."""
    fx = _load(FIXTURE_DIR / "drl_inh_r1_s1.json")
    drl = DRL(fx["interval_ticks"])
    engine = {
        i
        for i, t in enumerate(fx["response_ticks"])
        if drl.step(t, ResponseEvent(time=t, operandum=0)).reinforced
    }
    assert len(engine - set(fx["recorded_reinforced_responses"])) > 0
