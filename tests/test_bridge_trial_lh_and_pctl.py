"""Construction-time checks for ``Pctl`` targets and postfix LH on trial-based schedules.

* ``Pctl`` with a target other than ``IRT`` (``latency``, ``duration``,
  ``force``, ``rate``) is rejected by ``from_dsl`` when it builds the
  schedule. ``to_spec`` already rejects it; the response events carry
  only a time and an operandum, so the runtime has nothing to rank.
* A postfix ``LH`` on ``MTS`` (``MTS(...) LH10s``, the form the grammar
  writes) means the same as the keyword form ``MTS(..., LH=10s)``: the
  choice window. ``from_dsl`` and ``to_spec`` give both forms the same
  result. A postfix ``LH`` on ``GoNoGo`` is rejected like the keyword
  form. Writing both the keyword and the postfix ``LH`` is rejected.
"""

from __future__ import annotations

import pytest
from contingency_dsl import parse

from contingency import from_dsl, to_spec
from contingency.errors import ScheduleConfigError
from contingency.schedules.trial_based import MatchingToSample

# MatchingToSample exposes no public getter for the choice window or the
# sample duration; the constructor values are read from its attributes.

_MTS_KEYWORD = "MTS(comparisons=3, consequence=CRF, ITI=5s, LH=10s)"
_MTS_POSTFIX = "MTS(comparisons=3, consequence=CRF, ITI=5s) LH10s"
_GONOGO_KEYWORD = "GoNoGo(responseWindow=3s, consequence=CRF, ITI=5s, LH=2s)"
_GONOGO_POSTFIX = "GoNoGo(responseWindow=3s, consequence=CRF, ITI=5s) LH2s"


@pytest.mark.unit
@pytest.mark.parametrize(
    "target", ["latency", "duration", "force", "rate", "location", "amplitude"]
)
def test_from_dsl_rejects_non_irt_pctl_at_construction(target: str) -> None:
    with pytest.raises(ScheduleConfigError, match="IRT"):
        from_dsl(parse(f"Pctl({target}, 50)"))


@pytest.mark.unit
def test_from_dsl_accepts_irt_pctl() -> None:
    from_dsl(parse("Pctl(IRT, 50)"))


@pytest.mark.unit
def test_postfix_mts_lh_sets_choice_timeout() -> None:
    schedule = from_dsl(parse(_MTS_POSTFIX))
    assert isinstance(schedule, MatchingToSample)
    assert schedule._choice_timeout == pytest.approx(10.0)


@pytest.mark.unit
def test_postfix_mts_lh_uses_time_unit() -> None:
    schedule = from_dsl(
        parse("MTS(comparisons=2, consequence=CRF, ITI=1s) LH500ms")
    )
    assert isinstance(schedule, MatchingToSample)
    assert schedule._choice_timeout == pytest.approx(0.5)


@pytest.mark.unit
def test_postfix_mts_lh_scales_with_time_unit_seconds() -> None:
    keyword = to_spec(parse(_MTS_KEYWORD), time_unit_seconds=1000.0)
    postfix = to_spec(parse(_MTS_POSTFIX), time_unit_seconds=1000.0)
    assert postfix == keyword
    schedule = from_dsl(parse(_MTS_POSTFIX), time_unit_seconds=1000.0)
    assert isinstance(schedule, MatchingToSample)
    assert schedule._choice_timeout == pytest.approx(10000.0)


@pytest.mark.unit
def test_program_lh_declaration_does_not_count_as_a_second_mts_lh() -> None:
    # A program-level LH does not reach trial-based schedules, so the
    # postfix LH is the only one and sets the choice window.
    source = "LH = 2s\n" + _MTS_POSTFIX
    schedule = from_dsl(parse(source))
    assert isinstance(schedule, MatchingToSample)
    assert schedule._choice_timeout == pytest.approx(10.0)
    assert to_spec(parse(source)) == to_spec(parse(_MTS_KEYWORD))


@pytest.mark.unit
def test_postfix_and_keyword_mts_lh_give_the_same_runtime_values() -> None:
    keyword = from_dsl(parse(_MTS_KEYWORD))
    postfix = from_dsl(parse(_MTS_POSTFIX))
    assert isinstance(keyword, MatchingToSample)
    assert isinstance(postfix, MatchingToSample)
    for attr in ("n_comparisons", "_sample_duration", "_choice_timeout", "iti"):
        assert getattr(postfix, attr) == getattr(keyword, attr)


@pytest.mark.unit
def test_postfix_and_keyword_mts_lh_give_the_same_spec() -> None:
    assert to_spec(parse(_MTS_POSTFIX)) == to_spec(parse(_MTS_KEYWORD))


@pytest.mark.unit
@pytest.mark.parametrize("source", [_GONOGO_KEYWORD, _GONOGO_POSTFIX])
def test_gonogo_lh_rejected_by_from_dsl(source: str) -> None:
    with pytest.raises(ScheduleConfigError, match="GoNoGo LH"):
        from_dsl(parse(source))


@pytest.mark.unit
@pytest.mark.parametrize("source", [_GONOGO_KEYWORD, _GONOGO_POSTFIX])
def test_gonogo_lh_rejected_by_to_spec(source: str) -> None:
    with pytest.raises(ScheduleConfigError, match="GoNoGo LH"):
        to_spec(parse(source))


_DOUBLE_LH = [
    "MTS(comparisons=3, consequence=CRF, ITI=5s, LH=10s) LH4s",
    "(MTS(comparisons=3, consequence=CRF, ITI=5s) LH10s) LH4s",
]


@pytest.mark.unit
@pytest.mark.parametrize("source", _DOUBLE_LH)
def test_keyword_and_postfix_lh_together_rejected_by_from_dsl(source: str) -> None:
    with pytest.raises(ScheduleConfigError, match="more than one LH"):
        from_dsl(parse(source))


@pytest.mark.unit
@pytest.mark.parametrize("source", _DOUBLE_LH)
def test_keyword_and_postfix_lh_together_rejected_by_to_spec(source: str) -> None:
    with pytest.raises(ScheduleConfigError, match="more than one LH"):
        to_spec(parse(source))
