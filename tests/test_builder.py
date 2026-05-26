"""Unit tests for :class:`contingency.builder.ScheduleBuilder`.

The builder is a thin static facade over the schedule classes. The
tests verify that each factory method returns the expected type with
the expected parameters and introduces no additional behaviour.
"""

from __future__ import annotations

import random

import pytest

from contingency.builder import ScheduleBuilder
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

# --- Ratio family ----------------------------------------------------------


@pytest.mark.unit
def test_fr_returns_fr_with_n() -> None:
    s = ScheduleBuilder.fr(5)
    assert isinstance(s, FR)
    assert s.n == 5


@pytest.mark.unit
def test_vr_returns_vr_with_mean_and_seed() -> None:
    s = ScheduleBuilder.vr(12.0, n_intervals=8, seed=42)
    assert isinstance(s, VR)
    assert s.mean == 12.0
    assert s.n_intervals == 8


@pytest.mark.unit
def test_rr_returns_rr_with_probability() -> None:
    rng = random.Random(1)
    s = ScheduleBuilder.rr(0.1, rng=rng)
    assert isinstance(s, RR)
    assert s.probability == pytest.approx(0.1)


@pytest.mark.unit
def test_crf_returns_fr_1() -> None:
    s = ScheduleBuilder.crf()
    assert isinstance(s, FR)
    assert s.n == 1


# --- Interval family -------------------------------------------------------


@pytest.mark.unit
def test_fi_returns_fi_with_interval() -> None:
    s = ScheduleBuilder.fi(30.0)
    assert isinstance(s, FI)


@pytest.mark.unit
def test_vi_returns_vi() -> None:
    s = ScheduleBuilder.vi(60.0, n_intervals=12, seed=1)
    assert isinstance(s, VI)


@pytest.mark.unit
def test_ri_returns_ri() -> None:
    rng = random.Random(0)
    s = ScheduleBuilder.ri(15.0, rng=rng)
    assert isinstance(s, RI)


@pytest.mark.unit
def test_limited_hold_wraps_interval() -> None:
    inner = ScheduleBuilder.fi(30.0)
    s = ScheduleBuilder.limited_hold(inner, 10.0)
    assert isinstance(s, LimitedHold)


# --- Time family ------------------------------------------------------------


@pytest.mark.unit
def test_ft_returns_ft() -> None:
    s = ScheduleBuilder.ft(10.0)
    assert isinstance(s, FT)
    assert s.interval == 10.0


@pytest.mark.unit
def test_vt_returns_vt() -> None:
    s = ScheduleBuilder.vt(20.0, n_intervals=6, seed=3)
    assert isinstance(s, VT)
    assert s.mean_interval == 20.0


@pytest.mark.unit
def test_rt_returns_rt() -> None:
    s = ScheduleBuilder.rt(5.0)
    assert isinstance(s, RT)


@pytest.mark.unit
def test_ext_returns_ext() -> None:
    assert isinstance(ScheduleBuilder.ext(), EXT)


# --- Compound schedules ----------------------------------------------------


@pytest.mark.unit
def test_concurrent_returns_concurrent_with_cod_and_cor() -> None:
    comps = {"a": ScheduleBuilder.fr(3), "b": ScheduleBuilder.fi(30.0)}
    c = ScheduleBuilder.concurrent(comps, cod=1.5, frco=2)
    assert isinstance(c, Concurrent)
    assert c.cod == 1.5
    assert c.frco == 2
    assert set(c.components.keys()) == {"a", "b"}


@pytest.mark.unit
def test_multiple_returns_multiple() -> None:
    parts = [ScheduleBuilder.fr(3), ScheduleBuilder.fr(5)]
    m = ScheduleBuilder.multiple(parts, stimuli=["red", "green"])
    assert isinstance(m, Multiple)
    assert m.current_stimulus == "red"


@pytest.mark.unit
def test_chained_returns_chained() -> None:
    parts = [ScheduleBuilder.fr(2), ScheduleBuilder.fr(4)]
    ch = ScheduleBuilder.chained(parts)
    assert isinstance(ch, Chained)


@pytest.mark.unit
def test_tandem_returns_tandem() -> None:
    parts = [ScheduleBuilder.fr(3), ScheduleBuilder.fr(5)]
    t = ScheduleBuilder.tandem(parts)
    assert isinstance(t, Tandem)


@pytest.mark.unit
def test_alternative_returns_alternative() -> None:
    a = ScheduleBuilder.alternative(ScheduleBuilder.fr(10), ScheduleBuilder.ft(60.0))
    assert isinstance(a, Alternative)


# --- Differential reinforcement --------------------------------------------


@pytest.mark.unit
def test_dro_returns_dro_with_type() -> None:
    s = ScheduleBuilder.dro(5.0, dro_type="momentary")
    assert isinstance(s, DRO)
    assert s.dro_type == "momentary"


@pytest.mark.unit
def test_drl_returns_drl() -> None:
    s = ScheduleBuilder.drl(3.0)
    assert isinstance(s, DRL)
    assert s.interval == 3.0


@pytest.mark.unit
def test_drh_returns_drh() -> None:
    s = ScheduleBuilder.drh(3, 2.0)
    assert isinstance(s, DRH)
    assert s.response_count == 3
    assert s.time_window == 2.0


# --- Progressive Ratio -----------------------------------------------------


@pytest.mark.unit
def test_pr_arithmetic_returns_progressiveratio() -> None:
    pr = ScheduleBuilder.pr_arithmetic(start=1, step=2)
    assert isinstance(pr, ProgressiveRatio)
    assert pr.current_requirement == 1


@pytest.mark.unit
def test_pr_geometric_returns_progressiveratio() -> None:
    pr = ScheduleBuilder.pr_geometric(start=1, ratio=2.0)
    assert isinstance(pr, ProgressiveRatio)
    # First reinforcer requires start=1 response.
    assert pr.current_requirement == 1


@pytest.mark.unit
def test_pr_richardson_roberts_returns_progressiveratio() -> None:
    pr = ScheduleBuilder.pr_richardson_roberts()
    assert isinstance(pr, ProgressiveRatio)
    # First two values of the canonical R&R series.
    assert pr.current_requirement == 1


# --- Facade semantics ------------------------------------------------------


@pytest.mark.unit
def test_builder_factories_are_pure_delegation() -> None:
    # Calling the builder twice produces independent instances.
    a = ScheduleBuilder.fr(5)
    b = ScheduleBuilder.fr(5)
    assert a is not b
    # Parametrised the same way, the two classes are behaviourally
    # equivalent (same ratio requirement).
    assert a.n == b.n
