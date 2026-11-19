"""Replay the respondent Tier A corpus through ``contingency.respondent``.

``conformance/respondent/`` pins the trial list of each Tier A element
(``docs/en/respondent-contract.md``). These tests run every case through
:func:`contingency.respondent.expand`, which hands the expression to
contingency-rs, and check the per-program registry for Tier B expansions.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

from contingency.engine import rust_engine_available

pytestmark = pytest.mark.skipif(
    not rust_engine_available(), reason="contingency_core is not installed"
)

from contingency.respondent import (  # noqa: E402
    TIER_A_TYPES,
    RespondentRegistry,
    RespondentSpecError,
    RespondentUnresolvedError,
    UnknownRespondentElementError,
    expand,
)

CORPUS = Path(__file__).resolve().parents[1] / "conformance" / "respondent"
CASES = sorted(CORPUS.glob("*.json"))


def _load(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


@pytest.mark.parametrize("path", CASES, ids=[p.stem for p in CASES])
def test_tier_a_case(path: Path) -> None:
    case = _load(path)
    expected = case["expected"]
    if "error" in expected:
        with pytest.raises(RespondentUnresolvedError) as info:
            expand(case["spec"], case["options"])
        assert info.value.missing == expected["missing"]
        return
    assert expand(case["spec"], case["options"]) == expected["trials"]


def test_tier_a_types_match_the_corpus() -> None:
    assert {_load(p)["spec"]["type"] for p in CASES} == TIER_A_TYPES


def test_unresolved_is_a_value_error() -> None:
    with pytest.raises(ValueError):
        expand({"type": "CSOnly", "cs": "tone"}, {})


def test_options_may_be_omitted_when_the_element_has_its_own_trials() -> None:
    trials = expand({"type": "CSOnly", "cs": "tone", "trials": 2})
    assert [t["cs"] for t in trials] == [["tone"], ["tone"]]


def test_spec_as_json_string() -> None:
    spec = json.dumps({"type": "USOnly", "us": "food", "trials": 1})
    assert expand(spec)[0]["us"] == "food"


@pytest.mark.parametrize(
    "spec",
    [{"cs": "tone"}, {"type": "Blocking"}, "not json", 3],
    ids=["no-type", "not-tier-a", "bad-json", "not-a-mapping"],
)
def test_bare_expand_rejects_anything_but_tier_a(spec: Any) -> None:
    with pytest.raises(RespondentSpecError):
        expand(spec, {"trials": 1})


def test_invalid_tier_a_arguments_raise_spec_error() -> None:
    with pytest.raises(RespondentSpecError):
        expand({"type": "CSOnly", "cs": "tone"}, {"trials": -1})


# ---------------------------------------------------------------------------
# Tier B registry
# ---------------------------------------------------------------------------


class _TwoStage:
    """A toy Tier B procedure: a Tier A ``first`` stage, then ``second``."""

    name = "two-stage"
    types = frozenset({"TwoStage"})

    def expand(
        self,
        spec: Mapping[str, Any],
        options: Mapping[str, Any],
        expand_nested: Callable[..., list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for stage in ("first", "second"):
            for trial in expand_nested(spec[stage], options):
                out.append({**trial, "stage": stage})
        return out


class _Wrapper:
    """Nests another registered Tier B procedure."""

    name = "wrapper"
    types = frozenset({"Wrap"})

    def expand(
        self,
        spec: Mapping[str, Any],
        options: Mapping[str, Any],
        expand_nested: Callable[..., list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        return expand_nested(spec["inner"], options)


def _two_stage_spec() -> dict[str, Any]:
    return {
        "type": "TwoStage",
        "first": {"type": "PairSimultaneous", "cs": "tone", "us": "food"},
        "second": {"type": "Extinction", "cs": "tone"},
    }


def test_registry_expands_tier_b_through_tier_a() -> None:
    registry = RespondentRegistry.build(_TwoStage())
    trials = registry.expand(_two_stage_spec(), {"trials": 2})
    assert [(t["stage"], t["cs"], t["us"]) for t in trials] == [
        ("first", ["tone"], "food"),
        ("first", ["tone"], "food"),
        ("second", ["tone"], None),
        ("second", ["tone"], None),
    ]


def test_registry_passes_nested_tier_b_back_through_itself() -> None:
    registry = RespondentRegistry.build(_TwoStage(), _Wrapper())
    spec = {"type": "Wrap", "inner": _two_stage_spec()}
    assert len(registry.expand(spec, {"trials": 1})) == 2


def test_registry_expands_tier_a_like_the_bare_function() -> None:
    registry = RespondentRegistry.build()
    spec = {"type": "USOnly", "us": "food"}
    assert registry.expand(spec, {"trials": 3}) == expand(spec, {"trials": 3})


def test_registry_is_scoped_to_the_program() -> None:
    with_ext = RespondentRegistry.build(_TwoStage())
    without = RespondentRegistry.build()
    assert "TwoStage" in with_ext.types
    with pytest.raises(UnknownRespondentElementError):
        without.expand(_two_stage_spec(), {"trials": 1})
    with pytest.raises(UnknownRespondentElementError):
        expand(_two_stage_spec(), {"trials": 1})


def test_unknown_type_is_an_error_not_a_no_op() -> None:
    registry = RespondentRegistry.build(_TwoStage())
    with pytest.raises(UnknownRespondentElementError) as info:
        registry.expand({"type": "Renewal"}, {"trials": 1})
    assert info.value.type_name == "Renewal"
    assert isinstance(info.value, RespondentSpecError)


def test_registry_rejects_a_claim_on_a_tier_a_type() -> None:
    class _Override:
        name = "override"
        types = frozenset({"Extinction"})

        def expand(self, spec: Any, options: Any, expand_nested: Any) -> Any:
            return []

    with pytest.raises(ValueError, match="Tier A"):
        RespondentRegistry.build(_Override())


def test_registry_rejects_two_owners_of_one_type() -> None:
    class _Other(_TwoStage):
        name = "other"

    with pytest.raises(ValueError, match="TwoStage"):
        RespondentRegistry.build(_TwoStage(), _Other())


def test_registry_rejects_duplicate_names() -> None:
    class _Same(_Wrapper):
        types = frozenset({"Other"})
        name = "two-stage"

    with pytest.raises(ValueError, match="two-stage"):
        RespondentRegistry.build(_TwoStage(), _Same())


def test_registry_rejects_an_expansion_without_types() -> None:
    class _Empty(_Wrapper):
        types: frozenset[str] = frozenset()

    with pytest.raises(ValueError, match="wrapper"):
        RespondentRegistry.build(_Empty())


def test_expansion_returning_a_non_list_is_rejected() -> None:
    class _Bad(_Wrapper):
        def expand(self, spec: Any, options: Any, expand_nested: Any) -> Any:
            return "trials"

    registry = RespondentRegistry.build(_Bad())
    with pytest.raises(RespondentSpecError, match="wrapper"):
        registry.expand({"type": "Wrap"}, {})


def test_nested_mappings_of_any_kind_are_accepted() -> None:
    from types import MappingProxyType

    spec = {
        "type": "PairForwardDelay",
        "cs": "tone",
        "us": "shock",
        "isi": MappingProxyType({"value": 10, "unit": "s"}),
        "cs_duration": MappingProxyType({"value": 15, "unit": "s"}),
    }
    trials = expand(MappingProxyType(spec), MappingProxyType({"trials": 1}))
    assert trials[0]["time"]["us_on"] == 10.0


def test_registry_rejects_types_given_as_a_string() -> None:
    class _Str(_Wrapper):
        types = "Blocking"  # type: ignore[assignment]

    with pytest.raises(ValueError, match="set of strings"):
        RespondentRegistry.build(_Str())


def test_registry_rejects_an_expansion_without_expand() -> None:
    class _NoExpand:
        name = "no-expand"
        types = frozenset({"Thing"})

    with pytest.raises(ValueError, match="callable 'expand'"):
        RespondentRegistry.build(_NoExpand())  # type: ignore[arg-type]


def test_registry_owners_cannot_be_changed_after_build() -> None:
    registry = RespondentRegistry.build(_TwoStage())
    with pytest.raises(TypeError):
        registry._owners["Other"] = _Wrapper()  # type: ignore[index]


def test_trial_without_the_contract_keys_is_rejected() -> None:
    class _Partial(_Wrapper):
        def expand(self, spec: Any, options: Any, expand_nested: Any) -> Any:
            return [{"stage": "x", "cs": [], "us": None}]

    registry = RespondentRegistry.build(_Partial())
    with pytest.raises(RespondentSpecError, match="trial 0"):
        registry.expand({"type": "Wrap"}, {})


def test_expansion_nesting_itself_without_end_is_a_spec_error() -> None:
    class _Loop(_Wrapper):
        def expand(self, spec: Any, options: Any, expand_nested: Any) -> Any:
            return expand_nested(spec, options)

    registry = RespondentRegistry.build(_Loop())
    with pytest.raises(RespondentSpecError, match="without end"):
        registry.expand({"type": "Wrap"}, {})
