"""Verify the conformance corpus against the current Python runtime.

Loads every JSON fixture under ``conformance/``, builds the schedule
described by ``schedule``, replays the event sequence, and asserts
that each step's observed outcome matches the fixture's ``expect``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

# Reuse the authoritative builder so there is one spec-to-schedule
# mapping for both fixture generation and consumption.
from contingency.conformance.spec import build_schedule_from_spec
from contingency.entities import ResponseEvent

CONFORMANCE_DIR = Path(__file__).resolve().parents[1] / "conformance"
FLOAT_TOL = 1e-9

FIXTURES = sorted(CONFORMANCE_DIR.rglob("*.json"))


def _fixture_id(path: Path) -> str:
    return str(path.relative_to(CONFORMANCE_DIR))


@pytest.mark.parametrize(
    "fixture_path", FIXTURES, ids=[_fixture_id(p) for p in FIXTURES]
)
def test_conformance_fixture(fixture_path: Path) -> None:
    data: dict[str, Any] = json.loads(fixture_path.read_text())
    schedule = build_schedule_from_spec(data["schedule"])

    for i, step in enumerate(data["steps"]):
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
        expected: dict[str, Any] = step["expect"]

        assert outcome.reinforced == expected["reinforced"], (
            f"{fixture_path.name} step {i}: "
            f"expected reinforced={expected['reinforced']!r} "
            f"got {outcome.reinforced!r}"
        )

        if expected["reinforced"]:
            exp_r = expected["reinforcer"]
            assert outcome.reinforcer is not None
            assert abs(outcome.reinforcer.time - exp_r["time"]) <= FLOAT_TOL, (
                f"{fixture_path.name} step {i}: reinforcer.time "
                f"expected {exp_r['time']} got {outcome.reinforcer.time}"
            )
            assert (
                abs(outcome.reinforcer.magnitude - exp_r["magnitude"])
                <= FLOAT_TOL
            )
            assert outcome.reinforcer.label == exp_r["label"]
        else:
            assert outcome.reinforcer is None

        # Every key present in expect.meta must match outcome.meta.
        exp_meta = expected.get("meta", {})
        for k, v in exp_meta.items():
            assert k in outcome.meta, (
                f"{fixture_path.name} step {i}: expected meta key "
                f"{k!r} missing from outcome.meta={outcome.meta!r}"
            )
            got = outcome.meta[k]
            if isinstance(v, float):
                assert isinstance(got, (int, float))
                assert abs(float(got) - v) <= FLOAT_TOL
            else:
                assert got == v, (
                    f"{fixture_path.name} step {i}: meta[{k!r}] "
                    f"expected {v!r} got {got!r}"
                )


_GOOD_TT = {"sample": "A1", "correct": "B1", "comparisons": ["B1", "B2", "B3"]}


@pytest.mark.parametrize(
    ("trial_types", "message"),
    [
        ("A1", "trial_types must be a list"),
        ([["A1", "B1"]], r"trial_types\[0\] must be an object"),
        ([{"sample": "A1", "correct": "B1"}], r"trial_types\[0\] is missing comparisons"),
        ([{**_GOOD_TT, "comparisons": "B1B"}], "must be a list of strings"),
        ([{**_GOOD_TT, "sample": None}], "sample and correct must be strings"),
    ],
    ids=["not-list", "item-not-object", "missing-key", "comparisons-str", "sample-none"],
)
def test_mts_trial_types_spec_rejects_bad_shape(trial_types: Any, message: str) -> None:
    from contingency.errors import ScheduleConfigError

    spec = {
        "type": "MatchingToSample",
        "params": {"n_comparisons": 3, "seed": 3, "trial_types": trial_types},
    }
    with pytest.raises(ScheduleConfigError, match=message):
        build_schedule_from_spec(spec)
