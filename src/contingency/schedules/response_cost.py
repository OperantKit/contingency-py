"""Response-cost schedule wrapper.

A response cost is a punishment procedure in which each target response
debits a fixed amount from the subject's accumulated reinforcers (or a
symbolic token economy). Weiner (1962) showed that introducing a
response cost on a concurrent schedule of human button-pressing
produced systematic reductions in response rate; Hackenberg (2009)
reviews the broader token-reinforcement literature in which response
cost is one member of a family of token-economic contingencies.

The wrapper implemented here tracks a simple numeric ``balance``.
When ``initial_balance`` is ``None``, the balance is *unlimited*:
deductions are nominal (cost is informational and never prevents the
inner schedule from running). When ``initial_balance`` is a number, a
response that cannot be afforded (balance < amount) is silently dropped
— the inner schedule is not stepped, and the outcome carries a
``cost_insufficient`` flag in its ``meta``. Reinforcement produced by
the inner schedule is passed through unchanged and does *not*
automatically credit the balance: the accounting of reinforcer-to-token
conversion belongs to the caller.

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

from contingency.entities import Outcome, ResponseEvent
from contingency.errors import ScheduleConfigError, ScheduleStateError
from contingency.interfaces import Schedule

__all__ = ["ResponseCost"]


_TIME_TOL = 1e-9


def _check_time(last_now: float | None, now: float) -> None:
    if last_now is not None and now < last_now - _TIME_TOL:
        raise ScheduleStateError(
            f"Non-monotonic time: new now={now!r} precedes previous now={last_now!r}"
        )


def _check_event(now: float, event: ResponseEvent | None) -> None:
    if event is not None and abs(event.time - now) > _TIME_TOL:
        raise ScheduleStateError(
            f"event.time ({event.time!r}) must equal now ({now!r}) within {_TIME_TOL}"
        )


class ResponseCost:
    """Response-cost wrapper — Weiner (1962); Hackenberg (2009).

    Each response costs ``amount`` tokens. If the subject has
    insufficient tokens, the response is logged but has no effect (it
    does not advance the inner schedule). Reinforcement from the inner
    schedule is passed through unchanged.

    Token balance is a simple counter; the initial balance can be
    specified or unlimited. Responses consume before the inner schedule
    is consulted.

    Parameters
    ----------
    inner:
        The schedule to wrap.
    amount:
        Cost per response (must be > 0).
    unit:
        Name of the resource being deducted (informational only).
    initial_balance:
        Starting balance; ``None`` means unlimited (cost is purely
        nominal).

    References
    ----------
    Hackenberg, T. D. (2009). Token reinforcement: A review and
    analysis. *Journal of the Experimental Analysis of Behavior*,
    91(2), 257-286. https://doi.org/10.1901/jeab.2009.91-257

    Weiner, H. (1962). Some effects of response cost upon human
    operant behavior. *Journal of the Experimental Analysis of
    Behavior*, 5(2), 201-208. https://doi.org/10.1901/jeab.1962.5-201
    """

    __slots__ = (
        "_inner",
        "_amount",
        "_unit",
        "_initial_balance",
        "_balance",
        "_last_now",
    )

    def __init__(
        self,
        inner: Schedule,
        amount: float,
        unit: str = "token",
        initial_balance: float | None = None,
    ) -> None:
        if isinstance(amount, bool) or not isinstance(amount, (int, float)):
            raise ScheduleConfigError(
                f"ResponseCost requires a numeric amount, got {type(amount).__name__}"
            )
        if not (amount > 0):
            raise ScheduleConfigError(
                f"ResponseCost requires amount > 0, got {amount!r}"
            )
        if initial_balance is not None:
            if isinstance(initial_balance, bool) or not isinstance(
                initial_balance, (int, float)
            ):
                raise ScheduleConfigError(
                    f"ResponseCost requires a numeric initial_balance or None, "
                    f"got {type(initial_balance).__name__}"
                )
            if initial_balance < 0:
                raise ScheduleConfigError(
                    f"ResponseCost requires initial_balance >= 0, "
                    f"got {initial_balance!r}"
                )
        self._inner: Schedule = inner
        self._amount: float = float(amount)
        self._unit: str = str(unit)
        self._initial_balance: float | None = (
            float(initial_balance) if initial_balance is not None else None
        )
        self._balance: float | None = self._initial_balance
        self._last_now: float | None = None

    @property
    def amount(self) -> float:
        return self._amount

    @property
    def unit(self) -> str:
        return self._unit

    @property
    def balance(self) -> float | None:
        return self._balance

    def step(self, now: float, event: ResponseEvent | None = None) -> Outcome:
        _check_time(self._last_now, now)
        _check_event(now, event)
        self._last_now = now

        if event is None:
            # No response: propagate a pure tick to the inner schedule.
            return self._inner.step(now, None)

        # A response was emitted; check balance before stepping inner.
        if self._balance is not None and self._balance < self._amount:
            return Outcome(meta={"cost_insufficient": True, "unit": self._unit})

        if self._balance is not None:
            self._balance -= self._amount

        # Reinforcement from the inner schedule is passed through
        # unchanged; cost accounting is visible via the ``balance``
        # property rather than through the outcome's metadata.
        return self._inner.step(now, event)

    def reset(self) -> None:
        self._inner.reset()
        self._balance = self._initial_balance
        self._last_now = None
