"""Tests for :class:`contingency.schedules.ResponseCost`.

Each response deducts ``amount`` from the subject's token balance. A
response that cannot be afforded is silently dropped — it does not
advance the inner schedule. Unlimited balance (``initial_balance=None``)
leaves deductions purely nominal.

References
----------
Hackenberg, T. D. (2009). Token reinforcement: A review and analysis.
*Journal of the Experimental Analysis of Behavior*, 91(2), 257-286.
https://doi.org/10.1901/jeab.2009.91-257

Weiner, H. (1962). Some effects of response cost upon human operant
behavior. *Journal of the Experimental Analysis of Behavior*, 5(2),
201-208. https://doi.org/10.1901/jeab.1962.5-201
"""

from __future__ import annotations

import pytest

from contingency.entities import ResponseEvent
from contingency.errors import ScheduleConfigError, ScheduleStateError
from contingency.schedules import FR, ResponseCost


@pytest.mark.unit
def test_response_cost_balance_depleted_after_n_responses() -> None:
    """With cost=1 and balance=5, five responses exhaust the balance."""
    s = ResponseCost(FR(1), amount=1.0, initial_balance=5.0)
    assert s.balance == pytest.approx(5.0)
    for t in (1.0, 2.0, 3.0, 4.0, 5.0):
        out = s.step(t, ResponseEvent(time=t))
        # Each afforded response reinforces on FR(1).
        assert out.reinforced is True
    assert s.balance == pytest.approx(0.0)

    # Further responses cannot be afforded.
    out = s.step(6.0, ResponseEvent(time=6.0))
    assert out.reinforced is False
    assert out.meta.get("cost_insufficient") is True


@pytest.mark.unit
def test_response_cost_insufficient_balance_does_not_step_inner() -> None:
    """A skipped response must not advance the inner schedule's counter."""
    s = ResponseCost(FR(2), amount=2.0, initial_balance=3.0)
    # First response costs 2, balance -> 1. FR(2) not yet met.
    out = s.step(1.0, ResponseEvent(time=1.0))
    assert out.reinforced is False
    # Second response: balance 1 < 2, response is skipped entirely,
    # so the inner FR(2) count stays at 1.
    out = s.step(2.0, ResponseEvent(time=2.0))
    assert out.reinforced is False
    assert out.meta.get("cost_insufficient") is True
    assert s.balance == pytest.approx(1.0)
    # If we top up conceptually by resetting, the counter resets too —
    # so instead confirm that a pure tick advances time without cost.
    out = s.step(3.0, None)
    assert out.reinforced is False


@pytest.mark.unit
def test_response_cost_unlimited_balance_never_skips() -> None:
    """``initial_balance=None`` means cost is informational only."""
    s = ResponseCost(FR(1), amount=100.0)  # unlimited by default.
    assert s.balance is None
    for t in range(1, 11):
        out = s.step(float(t), ResponseEvent(time=float(t)))
        assert out.reinforced is True
    assert s.balance is None


@pytest.mark.unit
def test_response_cost_wrapping_fr3_reinforces_every_third() -> None:
    """Wrapping FR(3) with a cost of 1 and balance 10: three responses
    deliver one reinforcer, deducting 3 from the balance."""
    s = ResponseCost(FR(3), amount=1.0, initial_balance=10.0)
    assert s.step(1.0, ResponseEvent(time=1.0)).reinforced is False
    assert s.step(2.0, ResponseEvent(time=2.0)).reinforced is False
    out = s.step(3.0, ResponseEvent(time=3.0))
    assert out.reinforced is True
    assert s.balance == pytest.approx(7.0)


@pytest.mark.unit
def test_response_cost_tick_without_event_propagates_to_inner() -> None:
    """A pure tick is delegated to the inner schedule with no deduction."""
    s = ResponseCost(FR(1), amount=1.0, initial_balance=3.0)
    s.step(0.5, None)
    assert s.balance == pytest.approx(3.0)


@pytest.mark.unit
def test_response_cost_reset_restores_initial_balance() -> None:
    s = ResponseCost(FR(1), amount=1.0, initial_balance=2.0)
    s.step(1.0, ResponseEvent(time=1.0))
    s.step(2.0, ResponseEvent(time=2.0))
    assert s.balance == pytest.approx(0.0)
    s.reset()
    assert s.balance == pytest.approx(2.0)
    # Inner FR(1) is also reset.
    out = s.step(1.0, ResponseEvent(time=1.0))
    assert out.reinforced is True


@pytest.mark.unit
def test_response_cost_reset_with_unlimited_balance_stays_unlimited() -> None:
    s = ResponseCost(FR(1), amount=1.0)
    s.step(1.0, ResponseEvent(time=1.0))
    s.reset()
    assert s.balance is None


@pytest.mark.unit
def test_response_cost_non_monotonic_time_raises() -> None:
    s = ResponseCost(FR(1), amount=1.0)
    s.step(2.0, None)
    with pytest.raises(ScheduleStateError, match="Non-monotonic"):
        s.step(1.0, None)


@pytest.mark.unit
def test_response_cost_event_mismatch_raises() -> None:
    s = ResponseCost(FR(1), amount=1.0)
    with pytest.raises(ScheduleStateError, match="event.time"):
        s.step(1.0, ResponseEvent(time=2.0))


@pytest.mark.unit
def test_response_cost_config_error_amount_zero() -> None:
    with pytest.raises(ScheduleConfigError, match="amount > 0"):
        ResponseCost(FR(1), amount=0.0)


@pytest.mark.unit
def test_response_cost_config_error_amount_negative() -> None:
    with pytest.raises(ScheduleConfigError, match="amount > 0"):
        ResponseCost(FR(1), amount=-1.0)


@pytest.mark.unit
def test_response_cost_config_error_amount_non_numeric() -> None:
    with pytest.raises(ScheduleConfigError, match="numeric amount"):
        ResponseCost(FR(1), amount="one")  # type: ignore[arg-type]


@pytest.mark.unit
def test_response_cost_config_error_amount_bool_rejected() -> None:
    with pytest.raises(ScheduleConfigError, match="numeric amount"):
        ResponseCost(FR(1), amount=True)  # type: ignore[arg-type]


@pytest.mark.unit
def test_response_cost_config_error_initial_balance_negative() -> None:
    with pytest.raises(ScheduleConfigError, match="initial_balance >= 0"):
        ResponseCost(FR(1), amount=1.0, initial_balance=-1.0)


@pytest.mark.unit
def test_response_cost_config_error_initial_balance_non_numeric() -> None:
    with pytest.raises(ScheduleConfigError, match="numeric initial_balance"):
        ResponseCost(FR(1), amount=1.0, initial_balance="five")  # type: ignore[arg-type]


@pytest.mark.unit
def test_response_cost_config_error_initial_balance_bool_rejected() -> None:
    with pytest.raises(ScheduleConfigError, match="numeric initial_balance"):
        ResponseCost(FR(1), amount=1.0, initial_balance=True)  # type: ignore[arg-type]


@pytest.mark.unit
def test_response_cost_properties() -> None:
    s = ResponseCost(FR(2), amount=0.5, unit="credit", initial_balance=10.0)
    assert s.amount == pytest.approx(0.5)
    assert s.unit == "credit"
    assert s.balance == pytest.approx(10.0)


@pytest.mark.unit
def test_response_cost_initial_balance_zero_blocks_all_responses() -> None:
    s = ResponseCost(FR(1), amount=1.0, initial_balance=0.0)
    out = s.step(1.0, ResponseEvent(time=1.0))
    assert out.reinforced is False
    assert out.meta.get("cost_insufficient") is True
