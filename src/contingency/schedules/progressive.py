"""Progressive Ratio (PR) schedule.

Under a progressive ratio, each successive reinforcer requires more
responses than the previous one. The ratio "steps up" after every
delivery of a reinforcer, and the rule by which it steps up is called
the *step function*. PR is canonically used to measure reinforcer
efficacy by locating a *breakpoint* — the ratio at which responding
ceases in a session. Breakpoint detection is an experimental procedure
and is not enforced by the schedule itself: :class:`ProgressiveRatio`
advances indefinitely.

The module factors the step rule out of the schedule so that new step
series can be added without touching :class:`ProgressiveRatio`:

* :func:`arithmetic` — linear progression ``r_n = start + n * step``
  (Hodos, 1961, used the simplest arithmetic form).
* :func:`geometric` — multiplicative progression
  ``r_n = round(start * ratio ** n)``.
* :func:`richardson_roberts` — the 30-term Richardson-Roberts series
  popularised for drug self-administration (Richardson & Roberts,
  1996, Table 2), inspired by Hursh's (1980) demand-curve analyses.
  Values beyond index 29 are extrapolated geometrically by the last
  observed step ratio.

References
----------
Hodos, W. (1961). Progressive ratio as a measure of reward strength.
*Science*, 134(3483), 943-944.
https://doi.org/10.1126/science.134.3483.943

Hursh, S. R. (1980). Economic concepts for the analysis of behavior.
*Journal of the Experimental Analysis of Behavior*, 34(2), 219-238.
https://doi.org/10.1901/jeab.1980.34-219

Richardson, N. R., & Roberts, D. C. S. (1996). Progressive ratio
schedules in drug self-administration studies in rats: A method to
evaluate reinforcing efficacy. *Journal of Neuroscience Methods*,
66(1), 1-11. https://doi.org/10.1016/0165-0270(95)00153-0
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

from contingency.entities import Outcome, Reinforcer, ResponseEvent
from contingency.errors import ScheduleConfigError, ScheduleStateError

__all__ = [
    "StepFn",
    "ProgressiveRatio",
    "arithmetic",
    "geometric",
    "richardson_roberts",
]


StepFn: TypeAlias = Callable[[int], int]
"""A step function maps a 0-based reinforcement index to the ratio
requirement that must be met to earn *that* reinforcer. The returned
value must be a positive ``int`` (``>= 1``)."""


_TIME_TOL = 1e-9


def _check_time(last_now: float | None, now: float) -> None:
    """Raise :class:`ScheduleStateError` if ``now`` moves backwards."""
    if last_now is not None and now < last_now - _TIME_TOL:
        raise ScheduleStateError(
            f"Non-monotonic time: new now={now!r} precedes previous now={last_now!r}"
        )


def _check_event(now: float, event: ResponseEvent | None) -> None:
    """Raise :class:`ScheduleStateError` if ``event.time`` disagrees with ``now``."""
    if event is not None and abs(event.time - now) > _TIME_TOL:
        raise ScheduleStateError(
            f"event.time ({event.time!r}) must equal now ({now!r}) within {_TIME_TOL}"
        )


# ---------------------------------------------------------------------------
# Step function helpers
# ---------------------------------------------------------------------------


def arithmetic(start: int = 1, step: int = 1) -> StepFn:
    """Arithmetic progression ``r_n = start + n * step``.

    Parameters
    ----------
    start:
        First ratio requirement (``r_0``). Must be ``>= 1``.
    step:
        Increment added between successive reinforcers. Must be ``>= 1``.

    Returns
    -------
    StepFn
        A callable mapping reinforcement index ``n`` (0-based) to the
        required response count for that reinforcer.

    References
    ----------
    Hodos, W. (1961). Progressive ratio as a measure of reward strength.
    *Science*, 134(3483), 943-944.
    https://doi.org/10.1126/science.134.3483.943
    """
    if not isinstance(start, int) or isinstance(start, bool):
        raise ScheduleConfigError(
            f"arithmetic requires an int start, got {type(start).__name__}"
        )
    if not isinstance(step, int) or isinstance(step, bool):
        raise ScheduleConfigError(
            f"arithmetic requires an int step, got {type(step).__name__}"
        )
    if start < 1:
        raise ScheduleConfigError(f"arithmetic requires start >= 1, got {start}")
    if step < 1:
        raise ScheduleConfigError(f"arithmetic requires step >= 1, got {step}")

    def _arithmetic(n: int) -> int:
        return start + n * step

    return _arithmetic


def geometric(start: int = 1, ratio: float = 2.0) -> StepFn:
    """Geometric progression ``r_n = round(start * ratio ** n)``.

    Parameters
    ----------
    start:
        First ratio requirement (``r_0``). Must be ``>= 1``.
    ratio:
        Multiplicative factor. Must be ``> 1.0``; otherwise the sequence
        would be non-increasing.

    Returns
    -------
    StepFn
        A callable mapping reinforcement index to response requirement,
        with the geometric value rounded to the nearest positive integer.
    """
    if not isinstance(start, int) or isinstance(start, bool):
        raise ScheduleConfigError(
            f"geometric requires an int start, got {type(start).__name__}"
        )
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
        raise ScheduleConfigError(
            f"geometric requires a numeric ratio, got {type(ratio).__name__}"
        )
    if start < 1:
        raise ScheduleConfigError(f"geometric requires start >= 1, got {start}")
    if ratio <= 1.0:
        raise ScheduleConfigError(f"geometric requires ratio > 1.0, got {ratio}")

    ratio_f = float(ratio)

    def _geometric(n: int) -> int:
        value = round(start * ratio_f**n)
        # Guard against edge cases where rounding could collapse to 0
        # (cannot happen for start >= 1 and ratio > 1, but defensive).
        return max(1, int(value))

    return _geometric


# Richardson & Roberts (1996) series, Table 2 — 30 hardcoded values.
_RICHARDSON_ROBERTS_SERIES: tuple[int, ...] = (
    1,
    2,
    4,
    6,
    9,
    12,
    16,
    20,
    25,
    32,
    40,
    50,
    62,
    77,
    95,
    118,
    145,
    178,
    219,
    268,
    328,
    402,
    492,
    603,
    737,
    901,
    1102,
    1347,
    1647,
    2012,
)


def richardson_roberts() -> StepFn:
    """Return the standard Richardson-Roberts progressive-ratio series.

    Values 0-29 are the canonical series reported by Richardson and
    Roberts (1996), derived from Hursh's (1980) demand-curve framework::

        1, 2, 4, 6, 9, 12, 16, 20, 25, 32, 40, 50, 62, 77, 95, 118,
        145, 178, 219, 268, 328, 402, 492, 603, 737, 901, 1102, 1347,
        1647, 2012.

    Beyond index 29 the series is extrapolated geometrically using the
    ratio between the final two tabulated values
    (``2012 / 1647 ≈ 1.2217``). This keeps the series continuous for
    long sessions without fabricating a different growth rule.

    Returns
    -------
    StepFn
        A callable returning the response requirement for the n-th
        reinforcer (0-based).

    References
    ----------
    Hursh, S. R. (1980). Economic concepts for the analysis of behavior.
    *Journal of the Experimental Analysis of Behavior*, 34(2), 219-238.
    https://doi.org/10.1901/jeab.1980.34-219

    Richardson, N. R., & Roberts, D. C. S. (1996). Progressive ratio
    schedules in drug self-administration studies in rats: A method to
    evaluate reinforcing efficacy. *Journal of Neuroscience Methods*,
    66(1), 1-11. https://doi.org/10.1016/0165-0270(95)00153-0
    """
    series = _RICHARDSON_ROBERTS_SERIES
    last_ratio = series[-1] / series[-2]

    def _richardson_roberts(n: int) -> int:
        if n < 0:
            raise ScheduleConfigError(
                f"richardson_roberts index must be >= 0, got {n}"
            )
        if n < len(series):
            return series[n]
        # Geometric extrapolation from the last tabulated value.
        extra = n - (len(series) - 1)
        value = round(series[-1] * (last_ratio**extra))
        return max(1, int(value))

    return _richardson_roberts


# ---------------------------------------------------------------------------
# ProgressiveRatio schedule
# ---------------------------------------------------------------------------


class ProgressiveRatio:
    """Progressive-Ratio schedule parameterised by a step function.

    Each :class:`ResponseEvent` increments an internal counter. When
    the counter reaches the current requirement (``step_fn(index)``,
    where ``index`` is the 0-based count of reinforcers *already*
    delivered), a :class:`Reinforcer` is emitted, the counter is reset
    to zero, and ``index`` advances by one. The next requirement is
    ``step_fn(index + 1)``, and so on.

    Parameters
    ----------
    step_fn:
        Callable ``(n: int) -> int`` returning the positive ratio
        requirement for the n-th reinforcer (0-based). Typically built
        with :func:`arithmetic`, :func:`geometric`, or
        :func:`richardson_roberts`.

    Notes
    -----
    **No breakpoint termination.** Progressive-ratio procedures are
    often paired with a *breakpoint* criterion (e.g. no reinforcer
    earned within some interval) to identify the ratio at which
    responding ceases. That criterion is a *procedural* layer and does
    not belong in the schedule itself — :class:`ProgressiveRatio`
    keeps stepping up indefinitely. Break-point detection should be
    implemented by the session runner.

    Invalid step-function outputs are detected lazily: if
    ``step_fn(index)`` returns a non-``int`` or a value ``< 1`` the
    *first time it is consulted on a response*, :class:`ScheduleConfigError`
    is raised. This permits inexpensive construction with arbitrarily
    late-binding step functions.

    References
    ----------
    Hodos, W. (1961). Progressive ratio as a measure of reward strength.
    *Science*, 134(3483), 943-944.
    https://doi.org/10.1126/science.134.3483.943

    Hursh, S. R. (1980). Economic concepts for the analysis of behavior.
    *Journal of the Experimental Analysis of Behavior*, 34(2), 219-238.
    https://doi.org/10.1901/jeab.1980.34-219

    Richardson, N. R., & Roberts, D. C. S. (1996). Progressive ratio
    schedules in drug self-administration studies in rats: A method to
    evaluate reinforcing efficacy. *Journal of Neuroscience Methods*,
    66(1), 1-11. https://doi.org/10.1016/0165-0270(95)00153-0
    """

    __slots__ = ("_step_fn", "_index", "_count", "_last_now")

    def __init__(self, step_fn: StepFn) -> None:
        if not callable(step_fn):
            raise ScheduleConfigError(
                "ProgressiveRatio requires a callable step_fn, got "
                f"{type(step_fn).__name__}"
            )
        self._step_fn: StepFn = step_fn
        self._index: int = 0
        self._count: int = 0
        self._last_now: float | None = None

    @property
    def reinforcement_index(self) -> int:
        """0-based index of the *next* reinforcer to be earned."""
        return self._index

    @property
    def current_requirement(self) -> int:
        """Response count needed to earn the next reinforcer.

        Raises :class:`ScheduleConfigError` if ``step_fn`` returns an
        invalid value.
        """
        return self._resolve_requirement(self._index)

    def _resolve_requirement(self, index: int) -> int:
        value = self._step_fn(index)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ScheduleConfigError(
                f"step_fn({index}) must return a positive int, got {value!r}"
            )
        return value

    def step(self, now: float, event: ResponseEvent | None = None) -> Outcome:
        _check_time(self._last_now, now)
        _check_event(now, event)
        self._last_now = now
        if event is None:
            return Outcome()
        self._count += 1
        requirement = self._resolve_requirement(self._index)
        if self._count >= requirement:
            self._count = 0
            self._index += 1
            return Outcome(reinforced=True, reinforcer=Reinforcer(time=now))
        return Outcome()

    def reset(self) -> None:
        self._index = 0
        self._count = 0
        self._last_now = None
