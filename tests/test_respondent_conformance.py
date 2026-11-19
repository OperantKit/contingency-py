"""Checks on the respondent Tier A trial-list corpus (``conformance/respondent/``).

The corpus is the execution contract in ``docs/en/respondent-contract.md``.
These tests check that the files are what the generator writes, that every
Tier A element is covered, and that the schedule runner does not pick the
files up as schedule fixtures.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from contingency.conformance import iter_fixtures

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "conformance" / "respondent"

TIER_A_ELEMENTS = frozenset(
    {
        "Pair.ForwardDelay",
        "Pair.ForwardTrace",
        "Pair.Simultaneous",
        "Pair.Backward",
        "Extinction",
        "CSOnly",
        "USOnly",
        "Contingency",
        "TrulyRandom",
        "ExplicitlyUnpaired",
        "Compound",
        "Serial",
        "ITI",
        "Differential",
    }
)
TRIAL_KEYS = frozenset({"stage", "cs", "us", "context", "order", "time", "iti_s"})
TIME_KEYS = frozenset({"cs_on", "cs_off", "us_on", "us_off", "gap"})


def _generator() -> ModuleType:
    path = ROOT / "scripts" / "generate_respondent_conformance.py"
    spec = importlib.util.spec_from_file_location("generate_respondent_conformance", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fixtures() -> list[dict[str, Any]]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(CORPUS.glob("*.json"))]


@pytest.mark.unit
def test_corpus_matches_generator() -> None:
    assert _generator().main(["--check"]) == 0


@pytest.mark.unit
def test_corpus_covers_every_tier_a_element() -> None:
    assert {f["element"] for f in _fixtures()} == TIER_A_ELEMENTS


@pytest.mark.unit
def test_trial_shape() -> None:
    for f in _fixtures():
        expected = f["expected"]
        if "error" in expected:
            assert expected["error"] == "unresolved", f["name"]
            continue
        assert expected["trials"], f["name"]
        for trial in expected["trials"]:
            assert set(trial) == TRIAL_KEYS, f["name"]
            assert set(trial["time"]) == TIME_KEYS, f["name"]
            assert trial["cs"] == sorted(trial["cs"]), f["name"]


@pytest.mark.unit
def test_schedule_runner_skips_respondent_corpus() -> None:
    rels = [fx.rel for fx in iter_fixtures(ROOT / "conformance")]
    assert rels
    assert not any(rel.startswith("respondent/") for rel in rels)
