"""Percentile-shaped reinforcement schedule.

A percentile schedule (Platt, 1973; Galbicka, 1994) reinforces a
response whose measured dimension — typically inter-response time
(IRT), but also force, latency, duration, etc. — falls in a specified
percentile of a recent response window. Percentile schedules are the
canonical *shaping* procedure: by setting the rank threshold to e.g.
the 50th percentile and the direction to ``"above"``, every
above-median response is reinforced, which drifts the distribution
upward over time and shapes the target topography.

Implementation scope
--------------------
This implementation supports the ``"irt"`` (inter-response time)
dimension only. Other dimensions (``"force"``, ``"latency"``,
``"duration"``, ``"rate"``, ``"location"``) are accepted at
construction for the DSL round-trip but raise
:class:`ScheduleConfigError` on :meth:`step` because the runtime
cannot measure them from :class:`ResponseEvent` alone. A full
implementation will add force/latency fields to :class:`ResponseEvent`
and is out of scope here.

Window & percentile mechanics
-----------------------------
The schedule maintains a FIFO window of the last ``window`` observed
IRTs. The first ``window`` responses establish the baseline and are
never reinforced (there is nothing to compare against yet). From the
``window+1``-th response onward, the rank-th percentile of the window
is computed and compared against the current response's IRT:

* ``direction == "above"``: reinforce iff ``irt >= percentile``.
* ``direction == "below"``: reinforce iff ``irt <= percentile``.

After the comparison, the current IRT is appended to the window and
the oldest is discarded. This ordering — **compare, then update** —
prevents the response under evaluation from being included in its own
reference distribution, which would bias the threshold.

The percentile is computed via explicit sort + linear interpolation
(matching the ``numpy.percentile`` "linear" convention) so the package
keeps zero extra runtime dependencies.

References
----------
Galbicka, G. (1994). Shaping in the 21st century: Moving percentile
schedules into applied settings. *Journal of Applied Behavior
Analysis*, 27(4), 739-760. https://doi.org/10.1901/jaba.1994.27-739

Platt, J. R. (1973). Percentile reinforcement: Paradigms for
experimental analysis of response shaping. In G. H. Bower (Ed.),
*The Psychology of Learning and Motivation* (Vol. 7, pp. 271-296).
Academic Press. https://doi.org/10.1016/S0079-7421(08)60471-7
"""

from __future__ import annotations

from collections import deque

from contingency.entities import Outcome, Reinforcer, ResponseEvent
from contingency.errors import ScheduleConfigError, ScheduleStateError

__all__ = ["Percentile"]

_TIME_TOL = 1e-9

# Targets defined by the DSL (contingency_dsl.parser._PCTL_KNOWN_TARGETS).
# Only ``"irt"`` is executable in this version; the others are accepted at
# construction so an error is raised lazily at ``step`` time with a clear
# message.
_KNOWN_TARGETS: frozenset[str] = frozenset(
    {"irt", "IRT", "force", "latency", "duration", "rate", "location", "amplitude"}
)

# The subset that this runtime can actually evaluate. The DSL's ``IRT``
# (uppercase) and a lowercase alias ``irt`` are both accepted to match the
# Python API surface described in the design spec while staying compatible
# with the DSL's token casing.
_EXECUTABLE_TARGETS: frozenset[str] = frozenset({"irt", "IRT"})

_VALID_DIRECTIONS: frozenset[str] = frozenset({"above", "below"})


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


def _percentile_linear(sorted_samples: list[float], rank: int) -> float:
    """Compute the ``rank``-th percentile of pre-sorted samples.

    Uses the "linear" convention (equivalent to
    :func:`numpy.percentile` default / Type 7 in Hyndman & Fan, 1996):

    ``q = rank / 100`` mapped to the fractional index
    ``h = q * (n - 1)`` on the zero-based sorted array. Returns the
    linear interpolation between the two bracketing samples.

    The caller must pass a non-empty sorted list; this function does
    not validate its inputs because it is only ever called from
    :meth:`Percentile.step` under a length check.

    References
    ----------
    Hyndman, R. J., & Fan, Y. (1996). Sample quantiles in statistical
    packages. *The American Statistician*, 50(4), 361-365.
    https://doi.org/10.1080/00031305.1996.10473566
    """
    n = len(sorted_samples)
    if n == 1:
        return sorted_samples[0]
    # Clamp rank to [0, 100] defensively (constructor enforces this already).
    q = max(0.0, min(100.0, float(rank))) / 100.0
    h = q * (n - 1)
    lo = int(h)
    hi = min(lo + 1, n - 1)
    frac = h - lo
    return sorted_samples[lo] + (sorted_samples[hi] - sorted_samples[lo]) * frac


class Percentile:
    """Percentile-shaped schedule.

    Reinforces a response whose measured dimension falls in a specified
    percentile of the recent response window. Used to shape variable
    response topographies without the experimenter pre-specifying a
    fixed criterion.

    Parameters
    ----------
    target:
        Which response dimension to track. Accepted values:
        ``"irt"`` / ``"IRT"``, ``"force"``, ``"latency"``,
        ``"duration"``, ``"rate"``, ``"location"``, ``"amplitude"``.
        Only ``"irt"`` / ``"IRT"`` are executable in this version;
        other targets are accepted at construction (so a DSL program
        can round-trip through the bridge) but raise
        :class:`ScheduleConfigError` on the first :meth:`step` call.
    rank:
        Percentile rank threshold, an integer in ``[0, 100]``.
    window:
        Number of recent responses retained for the percentile
        computation. Must be ``>= 1``. The first ``window`` responses
        never reinforce (no baseline).
    direction:
        Which tail triggers reinforcement. ``"above"`` reinforces
        responses whose IRT is ``>= rank``-th percentile; ``"below"``
        reinforces responses whose IRT is ``<= rank``-th percentile.
        Defaults to ``"above"``.

    Notes
    -----
    Only the ``"irt"`` / ``"IRT"`` target is implemented in this
    release. The DSL accepts the full set of Platt/Galbicka dimensions
    (``force``, ``latency``, etc.) for surface-syntax completeness, but
    executing them requires fields not yet present on
    :class:`~contingency.entities.ResponseEvent`.

    References
    ----------
    Galbicka, G. (1994). Shaping in the 21st century: Moving percentile
    schedules into applied settings. *Journal of Applied Behavior
    Analysis*, 27(4), 739-760. https://doi.org/10.1901/jaba.1994.27-739

    Platt, J. R. (1973). Percentile reinforcement: Paradigms for
    experimental analysis of response shaping. In G. H. Bower (Ed.),
    *The Psychology of Learning and Motivation* (Vol. 7, pp. 271-296).
    Academic Press. https://doi.org/10.1016/S0079-7421(08)60471-7
    """

    __slots__ = (
        "_target",
        "_rank",
        "_window_size",
        "_direction",
        "_window",
        "_last_response_time",
        "_last_now",
    )

    def __init__(
        self,
        target: str,
        rank: int,
        window: int,
        direction: str = "above",
    ) -> None:
        if not isinstance(target, str):
            raise ScheduleConfigError(
                f"Percentile requires a string target, got {type(target).__name__}"
            )
        if target not in _KNOWN_TARGETS:
            raise ScheduleConfigError(
                f"Percentile target must be one of {sorted(_KNOWN_TARGETS)}, "
                f"got {target!r}"
            )
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise ScheduleConfigError(
                f"Percentile rank must be an int, got {type(rank).__name__}"
            )
        if not (0 <= rank <= 100):
            raise ScheduleConfigError(
                f"Percentile rank must be in [0, 100], got {rank}"
            )
        if isinstance(window, bool) or not isinstance(window, int):
            raise ScheduleConfigError(
                f"Percentile window must be an int, got {type(window).__name__}"
            )
        if window < 1:
            raise ScheduleConfigError(
                f"Percentile window must be >= 1, got {window}"
            )
        if direction not in _VALID_DIRECTIONS:
            raise ScheduleConfigError(
                f"Percentile direction must be 'above' or 'below', got {direction!r}"
            )
        self._target: str = target
        self._rank: int = rank
        self._window_size: int = window
        self._direction: str = direction
        self._window: deque[float] = deque(maxlen=window)
        self._last_response_time: float | None = None
        self._last_now: float | None = None

    @property
    def target(self) -> str:
        """The configured response dimension label."""
        return self._target

    @property
    def rank(self) -> int:
        """The configured percentile rank threshold (0-100)."""
        return self._rank

    @property
    def window(self) -> int:
        """The configured window size (number of recent IRTs retained)."""
        return self._window_size

    @property
    def direction(self) -> str:
        """Which tail triggers reinforcement — ``'above'`` or ``'below'``."""
        return self._direction

    @property
    def samples(self) -> tuple[float, ...]:
        """The current window of observed measurements (oldest first)."""
        return tuple(self._window)

    def step(self, now: float, event: ResponseEvent | None = None) -> Outcome:
        _check_time(self._last_now, now)
        _check_event(now, event)
        self._last_now = now

        if event is None:
            return Outcome()

        # Non-IRT targets are accepted at construction but not executable.
        if self._target not in _EXECUTABLE_TARGETS:
            raise ScheduleConfigError(
                f"Percentile target {self._target!r} is not yet implemented; "
                "only 'irt' / 'IRT' is executable in this release"
            )

        # First response ever: establish the baseline IRT anchor. There is no
        # preceding response to measure from, so we do not reinforce and we
        # do not add anything to the window (there is no IRT to add yet).
        if self._last_response_time is None:
            self._last_response_time = now
            return Outcome()

        irt = now - self._last_response_time
        # Update the "last response time" *now*, before the early-return
        # path below, so the next response measures from this one regardless
        # of whether reinforcement fires.
        self._last_response_time = now

        # If the window is not yet full, we cannot fairly compare — append
        # the IRT to the window and return unreinforced.
        if len(self._window) < self._window_size:
            self._window.append(irt)
            return Outcome()

        # Compute the percentile on the window *before* including the current
        # IRT, so this response is not part of its own reference distribution.
        sorted_samples = sorted(self._window)
        threshold = _percentile_linear(sorted_samples, self._rank)

        if self._direction == "above":
            eligible = irt >= threshold - _TIME_TOL
        else:  # "below"
            eligible = irt <= threshold + _TIME_TOL

        # Update the window with the current IRT (FIFO — deque(maxlen=...)
        # handles eviction).
        self._window.append(irt)

        if eligible:
            return Outcome(
                reinforced=True,
                reinforcer=Reinforcer(time=now),
                meta={
                    "percentile_threshold": threshold,
                    "percentile_irt": irt,
                    "percentile_rank": self._rank,
                    "percentile_direction": self._direction,
                },
            )
        return Outcome()

    def reset(self) -> None:
        """Return to post-construction state.

        Clears the window, the last-response anchor, and the
        monotonic-time bookkeeping.
        """
        self._window.clear()
        self._last_response_time = None
        self._last_now = None
