"""Tests for :func:`contingency.bridge.from_dsl`.

These tests parse source text with ``contingency_dsl.parse`` (and
occasionally ``contingency_dsl.analyze``) and verify the bridge
produces the correct runtime schedule with the correct parameters.
An end-to-end test actually drives a few :meth:`Schedule.step`
invocations and asserts reinforcement semantics.
"""

from __future__ import annotations

import pytest

from contingency import from_dsl
from contingency.entities import ResponseEvent
from contingency.errors import ScheduleConfigError
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
)

# contingency-dsl-py is a required dependency of this package; if it is
# missing, the bridge tests cannot run.
contingency_dsl = pytest.importorskip("contingency_dsl")


def parse_expr(source: str):
    """Parse DSL source text and return its schedule expression."""
    return contingency_dsl.parse(source).schedule


def analyzed_expr(source: str):
    """Parse + analyze to get the post-expansion schedule expression."""
    return contingency_dsl.analyze(contingency_dsl.parse(source)).program.schedule


# --- Atomic: 9 keyword combinations ----------------------------------------


@pytest.mark.unit
def test_bridge_fr() -> None:
    s = from_dsl(parse_expr("FR5"))
    assert isinstance(s, FR)
    assert s.n == 5


@pytest.mark.unit
def test_bridge_fi_seconds() -> None:
    s = from_dsl(parse_expr("FI30s"))
    assert isinstance(s, FI)
    assert s._interval == pytest.approx(30.0)


@pytest.mark.unit
def test_bridge_ft_seconds() -> None:
    s = from_dsl(parse_expr("FT15s"))
    assert isinstance(s, FT)
    assert s.interval == pytest.approx(15.0)


@pytest.mark.unit
def test_bridge_vr() -> None:
    s = from_dsl(parse_expr("VR10"))
    assert isinstance(s, VR)
    assert s.mean == pytest.approx(10.0)


@pytest.mark.unit
def test_bridge_vi_seconds() -> None:
    s = from_dsl(parse_expr("VI60s"))
    assert isinstance(s, VI)
    assert s._mean_interval == pytest.approx(60.0)


@pytest.mark.unit
def test_bridge_vt_seconds() -> None:
    s = from_dsl(parse_expr("VT45s"))
    assert isinstance(s, VT)
    assert s.mean_interval == pytest.approx(45.0)


@pytest.mark.unit
def test_bridge_rr_uses_mean_to_probability() -> None:
    # RR20 -> mean 20 responses per reinforcer -> p = 1/20 = 0.05.
    s = from_dsl(parse_expr("RR20"))
    assert isinstance(s, RR)
    assert s.probability == pytest.approx(0.05)


@pytest.mark.unit
def test_bridge_ri_seconds() -> None:
    s = from_dsl(parse_expr("RI30s"))
    assert isinstance(s, RI)


@pytest.mark.unit
def test_bridge_rt_seconds() -> None:
    s = from_dsl(parse_expr("RT20s"))
    assert isinstance(s, RT)


# --- Specials --------------------------------------------------------------


@pytest.mark.unit
def test_bridge_ext() -> None:
    s = from_dsl(parse_expr("EXT"))
    assert isinstance(s, EXT)


@pytest.mark.unit
def test_bridge_crf_maps_to_fr_1() -> None:
    s = from_dsl(parse_expr("CRF"))
    assert isinstance(s, FR)
    assert s.n == 1


# --- Time units -------------------------------------------------------------


@pytest.mark.unit
def test_bridge_milliseconds_scale_to_seconds() -> None:
    s = from_dsl(parse_expr("FI500ms"))
    assert isinstance(s, FI)
    assert s._interval == pytest.approx(0.5)


@pytest.mark.unit
def test_bridge_minutes_scale_to_seconds() -> None:
    s = from_dsl(parse_expr("FI2min"))
    assert isinstance(s, FI)
    assert s._interval == pytest.approx(120.0)


@pytest.mark.unit
def test_bridge_time_unit_none_passes_value_through() -> None:
    # Modifier with time_unit=None: DRO with a bare number is accepted
    # through the AST layer; the bridge should treat it as seconds
    # (factor = 1.0).
    from contingency_dsl.ast import Modifier

    node = Modifier(modifier="DRO", value=4.0, time_unit=None)
    s = from_dsl(node)
    assert isinstance(s, DRO)
    assert s.interval == pytest.approx(4.0)


@pytest.mark.unit
def test_bridge_time_unit_seconds_kwarg_remaps() -> None:
    # If the caller's runtime clock is in milliseconds they can rescale
    # via ``time_unit_seconds``; a 30s DSL value becomes 30_000 ms.
    s = from_dsl(parse_expr("FI30s"), time_unit_seconds=1000.0)
    assert isinstance(s, FI)
    assert s._interval == pytest.approx(30_000.0)


@pytest.mark.unit
def test_bridge_unknown_time_unit_raises() -> None:
    # Construct the Atomic node directly — the parser only accepts
    # recognised time units but the bridge must still defend against
    # bad AST inputs from other producers.
    from contingency_dsl.ast import Atomic, ScheduleType
    from contingency_dsl.enums import Distribution, Domain

    node = Atomic(
        schedule_type=ScheduleType(Distribution.FIXED, Domain.INTERVAL),
        value=30.0,
        time_unit="fortnights",
    )
    with pytest.raises(ScheduleConfigError, match="unsupported time_unit"):
        from_dsl(node)


# --- Compounds -------------------------------------------------------------


@pytest.mark.unit
def test_bridge_concurrent() -> None:
    s = from_dsl(parse_expr("Conc(FR5, VI30s)"))
    assert isinstance(s, Concurrent)
    assert set(s.components.keys()) == {"c0", "c1"}
    # Default COD/COR are zero when the DSL omits them.
    assert s.cod == 0.0
    assert s.cor == 0


@pytest.mark.unit
def test_bridge_concurrent_with_cod() -> None:
    s = from_dsl(parse_expr("Conc(VI30s, VI60s, COD=1s)"))
    assert isinstance(s, Concurrent)
    assert s.cod == pytest.approx(1.0)


@pytest.mark.unit
def test_bridge_concurrent_with_frco_maps_to_cor() -> None:
    s = from_dsl(parse_expr("Conc(VI30s, VI60s, COD=2s, FRCO=5)"))
    assert isinstance(s, Concurrent)
    assert s.cod == pytest.approx(2.0)
    assert s.cor == 5


@pytest.mark.unit
def test_bridge_alternative_two_components() -> None:
    s = from_dsl(parse_expr("Alt(FR3, FT5s)"))
    assert isinstance(s, Alternative)


@pytest.mark.unit
def test_bridge_alternative_three_components_nests_left() -> None:
    s = from_dsl(parse_expr("Alt(FR3, FR5, FR7)"))
    # Left-associative nesting: Alt(Alt(a, b), c).
    assert isinstance(s, Alternative)
    assert isinstance(s.first, Alternative)
    assert isinstance(s.second, FR)
    assert s.second.n == 7


@pytest.mark.unit
def test_bridge_multiple() -> None:
    s = from_dsl(parse_expr("Mult(FR3, FR5)"))
    assert isinstance(s, Multiple)
    assert s.n_components == 2
    assert s.current_stimulus == "s0"


@pytest.mark.unit
def test_bridge_chained() -> None:
    s = from_dsl(parse_expr("Chain(FR3, FR5)"))
    assert isinstance(s, Chained)


@pytest.mark.unit
def test_bridge_tandem() -> None:
    s = from_dsl(parse_expr("Tand(FR3, FR5)"))
    assert isinstance(s, Tandem)


# --- Unsupported combinators -----------------------------------------------


@pytest.mark.unit
def test_bridge_unsupported_combinator_mix_raises() -> None:
    with pytest.raises(ScheduleConfigError, match="Mix"):
        from_dsl(parse_expr("Mix(FR5, FR10)"))


@pytest.mark.unit
def test_bridge_unsupported_combinator_interpolate_raises() -> None:
    with pytest.raises(ScheduleConfigError, match="Interpolate"):
        from_dsl(parse_expr("Interp(FR5, FR10)"))


@pytest.mark.unit
def test_bridge_unsupported_combinator_conj_raises() -> None:
    # Construct directly — DSL may not accept Conj as a top-level
    # surface keyword in every build.
    from contingency_dsl.ast import Atomic, Compound, ScheduleType
    from contingency_dsl.enums import Combinator, Distribution, Domain

    left = Atomic(
        schedule_type=ScheduleType(Distribution.FIXED, Domain.RATIO),
        value=3.0,
    )
    right = Atomic(
        schedule_type=ScheduleType(Distribution.FIXED, Domain.RATIO),
        value=5.0,
    )
    node = Compound(combinator=Combinator.CONJ, components=(left, right))
    with pytest.raises(ScheduleConfigError, match="Conj"):
        from_dsl(node)


# --- Modifiers --------------------------------------------------------------


@pytest.mark.unit
def test_bridge_drl_seconds() -> None:
    s = from_dsl(parse_expr("DRL10s"))
    assert isinstance(s, DRL)
    assert s.interval == pytest.approx(10.0)


@pytest.mark.unit
def test_bridge_dro_seconds() -> None:
    s = from_dsl(parse_expr("DRO5s"))
    assert isinstance(s, DRO)
    assert s.interval == pytest.approx(5.0)
    assert s.dro_type == "resetting"


@pytest.mark.unit
def test_bridge_drh_seconds_uses_window_with_count_two() -> None:
    # DSL shape: DRH <time>. Bridge maps to DRH(response_count=2, window).
    s = from_dsl(parse_expr("DRH2s"))
    assert isinstance(s, DRH)
    assert s.response_count == 2
    assert s.time_window == pytest.approx(2.0)


@pytest.mark.unit
def test_bridge_repeat_expands_to_chained() -> None:
    # Use the pre-expansion AST shape: parse() preserves Repeat as a
    # Modifier; analyze() flattens it to Tand. Test both paths.
    s_pre = from_dsl(parse_expr("Repeat(3, FR5)"))
    assert isinstance(s_pre, Chained)
    s_post = from_dsl(analyzed_expr("Repeat(3, FR5)"))
    assert isinstance(s_post, Tandem)


# --- Progressive Ratio ------------------------------------------------------


@pytest.mark.unit
def test_bridge_pr_linear_with_start_and_increment() -> None:
    pr = from_dsl(parse_expr("PR(linear, start=1, increment=2)"))
    assert isinstance(pr, ProgressiveRatio)
    # arithmetic(start=1, step=2) -> 1, 3, 5, ...
    assert pr.current_requirement == 1


@pytest.mark.unit
def test_bridge_pr_geometric() -> None:
    pr = from_dsl(parse_expr("PR(geometric, start=1, ratio=2)"))
    assert isinstance(pr, ProgressiveRatio)
    assert pr.current_requirement == 1


@pytest.mark.unit
def test_bridge_pr_hodos_defaults_to_arithmetic_1_1() -> None:
    pr = from_dsl(parse_expr("PR(hodos)"))
    assert isinstance(pr, ProgressiveRatio)
    # Defaults start=1, increment=1: 1, 2, 3, ...
    assert pr.current_requirement == 1


@pytest.mark.unit
def test_bridge_pr_exponential_maps_to_geometric() -> None:
    pr = from_dsl(parse_expr("PR(exponential)"))
    assert isinstance(pr, ProgressiveRatio)


# --- LimitedHold -----------------------------------------------------------


@pytest.mark.unit
def test_bridge_limited_hold_wraps_fi() -> None:
    s = from_dsl(parse_expr("FI30s LH10s"))
    assert isinstance(s, LimitedHold)


# --- Deferred features -----------------------------------------------------


@pytest.mark.unit
def test_bridge_second_order_raises() -> None:
    with pytest.raises(ScheduleConfigError, match="SecondOrder"):
        from_dsl(parse_expr("FR5(FI30s)"))


@pytest.mark.unit
def test_bridge_identifier_ref_raises() -> None:
    # A raw IdentifierRef constructed directly — the parser produces
    # these inside let-binding expressions before ``analyze`` resolves
    # them. The bridge should refuse unresolved references.
    from contingency_dsl.ast import IdentifierRef

    with pytest.raises(ScheduleConfigError, match="IdentifierRef"):
        from_dsl(IdentifierRef(name="my_schedule"))


@pytest.mark.unit
def test_bridge_pctl_raises_not_implemented() -> None:
    from contingency_dsl.ast import Modifier

    # Construct a Pctl Modifier directly — the parser validates target
    # names, so we short-circuit by building the AST directly.
    node = Modifier(
        modifier="Pctl",
        pctl_target="rate",
        pctl_rank=50,
        pctl_window=20,
    )
    with pytest.raises(ScheduleConfigError, match="percentile"):
        from_dsl(node)


# --- Program, annotations, and related wrappers ----------------------------


@pytest.mark.unit
def test_bridge_program_translates_inner_schedule() -> None:
    program = contingency_dsl.parse("FR5")
    s = from_dsl(program)
    assert isinstance(s, FR)
    assert s.n == 5


@pytest.mark.unit
def test_bridge_annotated_schedule_strips_annotations() -> None:
    from contingency_dsl.ast import (
        AnnotatedSchedule,
        Annotation,
        Atomic,
        ScheduleType,
    )
    from contingency_dsl.enums import Distribution, Domain

    inner = Atomic(
        schedule_type=ScheduleType(Distribution.FIXED, Domain.RATIO),
        value=5.0,
    )
    annotated = AnnotatedSchedule(
        expr=inner,
        annotations=(Annotation(keyword="reinforcer", positional="food"),),
    )
    s = from_dsl(annotated)
    assert isinstance(s, FR)
    assert s.n == 5


@pytest.mark.unit
def test_bridge_aversive_raises_not_implemented() -> None:
    from contingency_dsl.ast import AversiveSchedule

    node = AversiveSchedule(kind="Sidman", params={})
    with pytest.raises(ScheduleConfigError, match="AversiveSchedule"):
        from_dsl(node)


@pytest.mark.unit
def test_bridge_fr_rejects_non_integer_value() -> None:
    from contingency_dsl.ast import Atomic, ScheduleType
    from contingency_dsl.enums import Distribution, Domain

    # A non-integer ratio is out of domain for FR.
    node = Atomic(
        schedule_type=ScheduleType(Distribution.FIXED, Domain.RATIO),
        value=2.5,
    )
    with pytest.raises(ScheduleConfigError, match="positive integer ratio"):
        from_dsl(node)


@pytest.mark.unit
def test_bridge_rr_rejects_non_positive_mean() -> None:
    from contingency_dsl.ast import Atomic, ScheduleType
    from contingency_dsl.enums import Distribution, Domain

    node = Atomic(
        schedule_type=ScheduleType(Distribution.RANDOM, Domain.RATIO),
        value=0.0,
    )
    with pytest.raises(ScheduleConfigError, match="positive mean-value"):
        from_dsl(node)


@pytest.mark.unit
def test_bridge_unknown_special_kind_raises() -> None:
    from contingency_dsl.ast import Special

    with pytest.raises(ScheduleConfigError, match="unknown Special kind"):
        from_dsl(Special(kind="Bogus"))


@pytest.mark.unit
def test_bridge_alt_rejects_single_component() -> None:
    from contingency_dsl.ast import Atomic, Compound, ScheduleType
    from contingency_dsl.enums import Combinator, Distribution, Domain

    only = Atomic(
        schedule_type=ScheduleType(Distribution.FIXED, Domain.RATIO),
        value=5.0,
    )
    node = Compound(combinator=Combinator.ALT, components=(only,))
    with pytest.raises(ScheduleConfigError, match="Alt requires >= 2"):
        from_dsl(node)


@pytest.mark.unit
def test_bridge_conc_directional_cod_raises() -> None:
    from contingency_dsl.ast import (
        Atomic,
        Compound,
        DirectionalCOD,
        ScheduleType,
    )
    from contingency_dsl.enums import Combinator, Distribution, Domain

    a = Atomic(
        schedule_type=ScheduleType(Distribution.VARIABLE, Domain.INTERVAL),
        value=30.0,
        time_unit="s",
    )
    b = Atomic(
        schedule_type=ScheduleType(Distribution.VARIABLE, Domain.INTERVAL),
        value=60.0,
        time_unit="s",
    )
    node = Compound(
        combinator=Combinator.CONC,
        components=(a, b),
        params={"COD": DirectionalCOD()},
    )
    with pytest.raises(ScheduleConfigError, match="directional COD"):
        from_dsl(node)


@pytest.mark.unit
def test_bridge_conc_unsupported_cod_type_raises() -> None:
    from contingency_dsl.ast import Atomic, Compound, ScheduleType
    from contingency_dsl.enums import Combinator, Distribution, Domain

    a = Atomic(
        schedule_type=ScheduleType(Distribution.VARIABLE, Domain.INTERVAL),
        value=30.0,
        time_unit="s",
    )
    b = Atomic(
        schedule_type=ScheduleType(Distribution.VARIABLE, Domain.INTERVAL),
        value=60.0,
        time_unit="s",
    )
    # Pass a raw string as COD — neither CompoundParam nor DirectionalCOD.
    node = Compound(
        combinator=Combinator.CONC,
        components=(a, b),
        params={"COD": "bogus"},  # type: ignore[dict-item]
    )
    with pytest.raises(ScheduleConfigError, match="unsupported COD"):
        from_dsl(node)


@pytest.mark.unit
def test_bridge_conc_unsupported_frco_type_raises() -> None:
    from contingency_dsl.ast import Atomic, Compound, ScheduleType
    from contingency_dsl.enums import Combinator, Distribution, Domain

    a = Atomic(
        schedule_type=ScheduleType(Distribution.VARIABLE, Domain.INTERVAL),
        value=30.0,
        time_unit="s",
    )
    b = Atomic(
        schedule_type=ScheduleType(Distribution.VARIABLE, Domain.INTERVAL),
        value=60.0,
        time_unit="s",
    )
    node = Compound(
        combinator=Combinator.CONC,
        components=(a, b),
        params={"FRCO": "bogus"},  # type: ignore[dict-item]
    )
    with pytest.raises(ScheduleConfigError, match="unsupported FRCO"):
        from_dsl(node)


@pytest.mark.unit
def test_bridge_conc_punish_raises() -> None:
    from contingency_dsl.ast import (
        Atomic,
        Compound,
        PunishParam,
        ScheduleType,
    )
    from contingency_dsl.enums import Combinator, Distribution, Domain

    a = Atomic(
        schedule_type=ScheduleType(Distribution.VARIABLE, Domain.INTERVAL),
        value=30.0,
        time_unit="s",
    )
    b = Atomic(
        schedule_type=ScheduleType(Distribution.VARIABLE, Domain.INTERVAL),
        value=60.0,
        time_unit="s",
    )
    node = Compound(
        combinator=Combinator.CONC,
        components=(a, b),
        params={"PUNISH": PunishParam(payload=())},
    )
    with pytest.raises(ScheduleConfigError, match="PUNISH"):
        from_dsl(node)


@pytest.mark.unit
def test_bridge_drl_missing_value_raises() -> None:
    from contingency_dsl.ast import Modifier

    node = Modifier(modifier="DRL", value=None)
    with pytest.raises(ScheduleConfigError, match="DRL requires"):
        from_dsl(node)


@pytest.mark.unit
def test_bridge_drh_missing_value_raises() -> None:
    from contingency_dsl.ast import Modifier

    node = Modifier(modifier="DRH", value=None)
    with pytest.raises(ScheduleConfigError, match="DRH requires"):
        from_dsl(node)


@pytest.mark.unit
def test_bridge_dro_missing_value_raises() -> None:
    from contingency_dsl.ast import Modifier

    node = Modifier(modifier="DRO", value=None)
    with pytest.raises(ScheduleConfigError, match="DRO requires"):
        from_dsl(node)


@pytest.mark.unit
def test_bridge_repeat_missing_inner_raises() -> None:
    from contingency_dsl.ast import Modifier

    node = Modifier(modifier="Repeat", value=3, inner=None)
    with pytest.raises(ScheduleConfigError, match="Repeat requires"):
        from_dsl(node)


@pytest.mark.unit
def test_bridge_repeat_requires_count_at_least_two() -> None:
    from contingency_dsl.ast import Atomic, Modifier, ScheduleType
    from contingency_dsl.enums import Distribution, Domain

    inner = Atomic(
        schedule_type=ScheduleType(Distribution.FIXED, Domain.RATIO),
        value=5.0,
    )
    node = Modifier(modifier="Repeat", value=1, inner=inner)
    with pytest.raises(ScheduleConfigError, match="count >= 2"):
        from_dsl(node)


@pytest.mark.unit
def test_bridge_modifier_unknown_kind_raises() -> None:
    from contingency_dsl.ast import Modifier

    node = Modifier(modifier="Bogus", value=1.0)
    with pytest.raises(ScheduleConfigError, match="unknown Modifier kind"):
        from_dsl(node)


@pytest.mark.unit
def test_bridge_pr_unknown_step_raises() -> None:
    from contingency_dsl.ast import Modifier

    node = Modifier(modifier="PR", pr_step="quadratic")
    with pytest.raises(ScheduleConfigError, match="unknown PR pr_step"):
        from_dsl(node)


@pytest.mark.unit
def test_bridge_unknown_node_raises() -> None:
    # A plain object is not an AST node; the dispatch falls through.
    class NotAnAstNode:
        pass

    with pytest.raises(ScheduleConfigError, match="unrecognised AST node"):
        from_dsl(NotAnAstNode())


# --- End-to-end: bridge + run ---------------------------------------------


@pytest.mark.integration
def test_end_to_end_fr5_delivers_reinforcer_on_fifth_response() -> None:
    s = from_dsl(parse_expr("FR5"))
    outcomes = []
    for t in range(1, 6):
        outcomes.append(s.step(float(t), ResponseEvent(time=float(t))))
    assert [o.reinforced for o in outcomes] == [False, False, False, False, True]
    assert outcomes[-1].reinforcer is not None
    assert outcomes[-1].reinforcer.time == 5.0


@pytest.mark.integration
def test_end_to_end_crf_delivers_every_response() -> None:
    s = from_dsl(parse_expr("CRF"))
    for t in (1.0, 2.0, 3.0):
        out = s.step(t, ResponseEvent(time=t))
        assert out.reinforced is True


@pytest.mark.integration
def test_end_to_end_concurrent_routes_events_by_operandum() -> None:
    # Conc(FR1, FR1) bridged: a response on c0 is reinforced by the
    # first component, and similarly for c1.
    s = from_dsl(parse_expr("Conc(FR1, FR1)"))
    assert isinstance(s, Concurrent)
    out = s.step(1.0, ResponseEvent(time=1.0, operandum="c0"))
    assert out.reinforced is True
    out = s.step(2.0, ResponseEvent(time=2.0, operandum="c1"))
    assert out.reinforced is True


@pytest.mark.integration
def test_end_to_end_fi_with_time_unit() -> None:
    # FI500ms -> internal interval 0.5 "seconds" on the runtime clock.
    s = from_dsl(parse_expr("FI500ms"))
    assert isinstance(s, FI)
    # No response before the interval: no reinforcement.
    assert s.step(0.1, ResponseEvent(time=0.1)).reinforced is False
    # Response after the interval elapses: reinforced.
    assert s.step(0.5, ResponseEvent(time=0.5)).reinforced is True


@pytest.mark.integration
def test_end_to_end_drl_enforces_irt() -> None:
    s = from_dsl(parse_expr("DRL2s"))
    # First response reinforced by definition.
    assert s.step(0.5, ResponseEvent(time=0.5)).reinforced is True
    # Second response 1s later: below DRL threshold of 2s.
    assert s.step(1.5, ResponseEvent(time=1.5)).reinforced is False
    # Third response 2.5s after the second: above threshold.
    assert s.step(4.0, ResponseEvent(time=4.0)).reinforced is True
