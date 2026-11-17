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

from typing import Any

from contingency.entities import Outcome, ResponseEvent
from contingency.errors import ScheduleConfigError
from contingency.interfaces import Schedule
from contingency.schedules._rust import component_specs, compound_schedule

__all__ = ["ResponseCost"]


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

    Notes
    -----
    The compound runs on the Rust engine. Its components must be
    schedules of this package that run there, not yet stepped, and
    distinct objects; otherwise the constructor raises
    :class:`ScheduleConfigError`.

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
        "_rust",
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
        self._rust: Any = compound_schedule(self._rust_spec(), needs_state=True)

    @property
    def amount(self) -> float:
        return self._amount

    @property
    def unit(self) -> str:
        return self._unit

    @property
    def balance(self) -> float | None:
        balance = self._rust.state().get("balance")
        return None if balance is None else float(balance)

    def _rust_spec(self) -> dict[str, Any]:
        (inner,) = component_specs("ResponseCost", [self._inner])
        params: dict[str, Any] = {"inner": inner, "amount": self._amount, "unit": self._unit}
        if self._initial_balance is not None:
            params["initial_balance"] = self._initial_balance
        return {"type": "ResponseCost", "params": params}

    @property
    def engine(self) -> str:
        """Always ``"rust"``: the Rust engine runs this schedule."""
        return "rust"

    def step(self, now: float, event: ResponseEvent | None = None) -> Outcome:
        return self._rust.step(now, event)  # type: ignore[no-any-return]

    def reset(self) -> None:
        self._inner.reset()
        self._rust = compound_schedule(self._rust_spec(), needs_state=True)
