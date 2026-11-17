"""Timeout (TO) schedule wrapper.

A timeout (TO) is a period following reinforcement during which the
underlying reinforcement schedule is suspended: responses do not advance
the inner schedule, and no reinforcement is available. Classically,
time-out from positive reinforcement has been studied both as a
punishment procedure (when contingent on a target response) and as a
simple interruption of the reinforcement process (see Leitenberg, 1965).

The wrapper implemented here places the inner schedule on timeout for
``duration`` time units after *every* reinforcer the inner schedule
produces. Responses emitted during the timeout interval are silently
dropped — they are not passed through to the inner schedule's
:meth:`step`, and no reinforcement can be delivered. When
``reset_on_response`` is true, any response emitted within the timeout
window extends the window (``timeout_ends`` is pushed to ``now +
duration``); this models a punishment contingency for responding in
timeout.

References
----------
Leitenberg, H. (1965). Is time-out from positive reinforcement an
aversive event? A review of the experimental evidence. *Psychological
Bulletin*, 64(6), 428-441. https://doi.org/10.1037/h0022657
"""

from __future__ import annotations

from typing import Any

from contingency.entities import Outcome, ResponseEvent
from contingency.errors import ScheduleConfigError
from contingency.interfaces import Schedule
from contingency.schedules._rust import component_specs, compound_schedule

__all__ = ["Timeout"]


class Timeout:
    """Timeout (TO) wrapper — Leitenberg (1965).

    After a reinforcement, the wrapped schedule is placed on timeout
    for ``duration`` time units. During the timeout, responses are
    ignored (do not advance the inner schedule) and no reinforcement
    is possible.

    If ``reset_on_response`` is True, a response during the timeout
    restarts the timeout clock (punishment for responding in timeout).

    Parameters
    ----------
    inner:
        The schedule to wrap.
    duration:
        Timeout duration in unit-agnostic monotonic time. Must be > 0.
    reset_on_response:
        Whether a response during timeout restarts the timeout clock.

    Notes
    -----
    The compound runs on the Rust engine. Its components must be
    schedules of this package that run there, not yet stepped, and
    distinct objects; otherwise the constructor raises
    :class:`ScheduleConfigError`.

    References
    ----------
    Leitenberg, H. (1965). Is time-out from positive reinforcement an
    aversive event? A review of the experimental evidence. *Psychological
    Bulletin*, 64(6), 428-441. https://doi.org/10.1037/h0022657
    """

    __slots__ = (
        "_inner",
        "_duration",
        "_reset_on_response",
        "_rust",
    )

    def __init__(
        self,
        inner: Schedule,
        duration: float,
        reset_on_response: bool = False,
    ) -> None:
        if isinstance(duration, bool) or not isinstance(duration, (int, float)):
            raise ScheduleConfigError(
                f"Timeout requires a numeric duration, got {type(duration).__name__}"
            )
        if not (duration > 0):
            raise ScheduleConfigError(
                f"Timeout requires duration > 0, got {duration!r}"
            )
        self._inner: Schedule = inner
        self._duration: float = float(duration)
        self._reset_on_response: bool = bool(reset_on_response)
        self._rust: Any = compound_schedule(self._rust_spec(), needs_state=True)

    @property
    def duration(self) -> float:
        return self._duration

    @property
    def reset_on_response(self) -> bool:
        return self._reset_on_response

    @property
    def in_timeout(self) -> bool:
        """Whether the wrapper is currently in a timeout window."""
        return bool(self._rust.state()["in_timeout"])

    def _rust_spec(self) -> dict[str, Any]:
        (inner,) = component_specs("Timeout", [self._inner])
        return {
            "type": "Timeout",
            "params": {
                "inner": inner,
                "duration": self._duration,
                "reset_on_response": self._reset_on_response,
            },
        }

    @property
    def engine(self) -> str:
        """Always ``"rust"``: the Rust engine runs this schedule."""
        return "rust"

    def step(self, now: float, event: ResponseEvent | None = None) -> Outcome:
        return self._rust.step(now, event)  # type: ignore[no-any-return]

    def reset(self) -> None:
        self._inner.reset()
        self._rust = compound_schedule(self._rust_spec(), needs_state=True)
