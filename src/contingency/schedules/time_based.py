"""Time-based and extinction reinforcement schedules.

This module implements the schedules whose reinforcement rule depends
on elapsed time alone (or on nothing at all, in the case of
extinction):

* :class:`FT` — Fixed Time. A reinforcer is delivered every
  ``interval`` time units, independently of the subject's behavior.
* :class:`VT` — Variable Time. Like :class:`FT`, but the inter-reinforcer
  intervals are drawn from a Fleshler-Hoffman progression with a
  configured mean.
* :class:`RT` — Random Time. Inter-reinforcer intervals are sampled from
  an exponential distribution (memoryless).
* :class:`EXT` — Extinction. No response is ever reinforced.

All three time-based schedules are **response-independent**: they
ignore the ``event`` argument to :meth:`step` for the purposes of the
reinforcement decision. The event, if supplied, is still validated to
have the same timestamp as ``now`` (consistent contract across the
schedule family).

Single-fire per step
--------------------
If :meth:`step` is called with a ``now`` that has advanced by more
than one scheduled interval since the last call (e.g. the caller
stepped late), **only one** reinforcer is emitted. After firing, the
schedule's reference time is set to ``now`` — subsequent steps wait a
full new interval. Callers are expected to step frequently enough that
this behavior is consistent with the experimenter's intent; see the
per-class ``Notes`` sections for discussion.

References
----------
Catania, A. C., & Reynolds, G. S. (1968). A quantitative analysis of
the responding maintained by interval schedules of reinforcement.
*Journal of the Experimental Analysis of Behavior*, 11(3, Pt. 2),
327-383. https://doi.org/10.1901/jeab.1968.11-s327

Fleshler, M., & Hoffman, H. S. (1962). A progression for generating
variable-interval schedules. *Journal of the Experimental Analysis of
Behavior*, 5(4), 529-530. https://doi.org/10.1901/jeab.1962.5-529

Skinner, B. F. (1957). *Schedules of reinforcement* (with C. B.
Ferster). Appleton-Century-Crofts.

Zeiler, M. D. (1968). Fixed and variable schedules of
response-independent reinforcement. *Journal of the Experimental
Analysis of Behavior*, 11(4), 405-414.
https://doi.org/10.1901/jeab.1968.11-405
"""

from __future__ import annotations

import random

from contingency.entities import Outcome, Reinforcer, ResponseEvent
from contingency.errors import ScheduleConfigError, ScheduleStateError
from contingency.helpers.fleshler_hoffman import generate_intervals

__all__ = ["FT", "VT", "RT", "EXT"]

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


class FT:
    """Fixed-Time schedule: a reinforcer every ``interval`` time units.

    Parameters
    ----------
    interval:
        The fixed inter-reinforcement interval. Must be ``> 0``. Units
        are unspecified — they must agree with the caller's clock.

    Notes
    -----
    A reinforcer is emitted as soon as ``now - last_reinforcement_time``
    has reached ``interval``. The schedule is response-independent: the
    ``event`` argument to :meth:`step` is ignored for the purpose of
    the decision, though its timestamp is still validated.

    At construction the internal ``last_reinforcement_time`` is
    ``None``. The first reinforcer fires on the first :meth:`step`
    whose ``now`` is ``>= interval`` of the first step; equivalently,
    the first step anchors ``t = 0`` on the schedule's internal clock.

    If the caller steps late — ``now`` has advanced by more than one
    interval since the last reinforcer — **only one** reinforcer is
    emitted and ``last_reinforcement_time`` is set to ``now``. The next
    reinforcer therefore requires a full additional ``interval``. In
    other words, FT does not queue missed reinforcers; the caller is
    responsible for stepping frequently enough to honour the schedule.

    References
    ----------
    Catania, A. C., & Reynolds, G. S. (1968). A quantitative analysis
    of the responding maintained by interval schedules of
    reinforcement. *Journal of the Experimental Analysis of Behavior*,
    11(3, Pt. 2), 327-383. https://doi.org/10.1901/jeab.1968.11-s327

    Zeiler, M. D. (1968). Fixed and variable schedules of
    response-independent reinforcement. *Journal of the Experimental
    Analysis of Behavior*, 11(4), 405-414.
    https://doi.org/10.1901/jeab.1968.11-405
    """

    __slots__ = ("_interval", "_anchor", "_last_now")

    def __init__(self, interval: float) -> None:
        if isinstance(interval, bool) or not isinstance(interval, (int, float)):
            raise ScheduleConfigError(
                f"FT requires a numeric interval, got {type(interval).__name__}"
            )
        if interval <= 0:
            raise ScheduleConfigError(f"FT requires interval > 0, got {interval}")
        self._interval: float = float(interval)
        self._anchor: float | None = None
        self._last_now: float | None = None

    @property
    def interval(self) -> float:
        """The configured fixed interval."""
        return self._interval

    def step(self, now: float, event: ResponseEvent | None = None) -> Outcome:
        _check_time(self._last_now, now)
        _check_event(now, event)
        self._last_now = now
        if self._anchor is None:
            self._anchor = now
            return Outcome()
        if now - self._anchor + _TIME_TOL >= self._interval:
            self._anchor = now
            return Outcome(reinforced=True, reinforcer=Reinforcer(time=now))
        return Outcome()

    def reset(self) -> None:
        self._anchor = None
        self._last_now = None


class VT:
    """Variable-Time schedule driven by a Fleshler-Hoffman progression.

    Parameters
    ----------
    mean_interval:
        Target mean inter-reinforcer interval, ``> 0``.
    n_intervals:
        Number of intervals generated per cycle. Defaults to 12.
    seed:
        Optional RNG seed for deterministic sequences. When supplied,
        :meth:`reset` restores the original sequence bit-identically.

    Notes
    -----
    A pool of ``n_intervals`` intervals with arithmetic mean
    ``mean_interval`` is generated at construction using
    :func:`contingency.helpers.fleshler_hoffman.generate_intervals`.
    The first interval becomes the current requirement. When
    ``now - anchor`` reaches that requirement a reinforcer is emitted,
    the cursor advances, and the next interval becomes the requirement.
    When the pool is exhausted, a fresh pool is generated using a
    sub-seed drawn from the internal RNG, preserving determinism.

    The first call to :meth:`step` anchors the schedule's clock at
    ``now``; the first reinforcer therefore fires after one interval
    has elapsed from that first step.

    As with :class:`FT`, only one reinforcer is emitted per
    :meth:`step` call.

    References
    ----------
    Fleshler, M., & Hoffman, H. S. (1962). A progression for generating
    variable-interval schedules. *Journal of the Experimental Analysis
    of Behavior*, 5(4), 529-530. https://doi.org/10.1901/jeab.1962.5-529

    Zeiler, M. D. (1968). Fixed and variable schedules of
    response-independent reinforcement. *Journal of the Experimental
    Analysis of Behavior*, 11(4), 405-414.
    https://doi.org/10.1901/jeab.1968.11-405
    """

    __slots__ = (
        "_mean",
        "_n_intervals",
        "_seed",
        "_rng",
        "_sequence",
        "_cursor",
        "_requirement",
        "_anchor",
        "_last_now",
    )

    def __init__(
        self,
        mean_interval: float,
        n_intervals: int = 12,
        seed: int | None = None,
    ) -> None:
        if isinstance(mean_interval, bool) or not isinstance(mean_interval, (int, float)):
            raise ScheduleConfigError(
                f"VT requires a numeric mean_interval, got {type(mean_interval).__name__}"
            )
        if mean_interval <= 0:
            raise ScheduleConfigError(
                f"VT requires mean_interval > 0, got {mean_interval}"
            )
        if not isinstance(n_intervals, int) or isinstance(n_intervals, bool):
            raise ScheduleConfigError(
                f"VT requires an int n_intervals, got {type(n_intervals).__name__}"
            )
        if n_intervals < 1:
            raise ScheduleConfigError(
                f"VT requires n_intervals >= 1, got {n_intervals}"
            )
        self._mean: float = float(mean_interval)
        self._n_intervals: int = n_intervals
        self._seed: int | None = seed
        self._rng: random.Random = random.Random(seed)
        self._sequence: list[float] = []
        self._cursor: int = 0
        self._requirement: float = 0.0
        self._anchor: float | None = None
        self._last_now: float | None = None
        self._reload_sequence()

    @property
    def mean_interval(self) -> float:
        return self._mean

    @property
    def n_intervals(self) -> int:
        return self._n_intervals

    def _next_sub_seed(self) -> int:
        return self._rng.randrange(0, 2**31 - 1)

    def _reload_sequence(self) -> None:
        sub_seed = self._next_sub_seed()
        self._sequence = generate_intervals(
            self._mean, n=self._n_intervals, seed=sub_seed
        )
        self._cursor = 0
        self._requirement = self._sequence[0]

    def _advance_requirement(self) -> None:
        self._cursor += 1
        if self._cursor >= len(self._sequence):
            self._reload_sequence()
        else:
            self._requirement = self._sequence[self._cursor]

    def step(self, now: float, event: ResponseEvent | None = None) -> Outcome:
        _check_time(self._last_now, now)
        _check_event(now, event)
        self._last_now = now
        if self._anchor is None:
            self._anchor = now
            return Outcome()
        if now - self._anchor + _TIME_TOL >= self._requirement:
            self._anchor = now
            self._advance_requirement()
            return Outcome(reinforced=True, reinforcer=Reinforcer(time=now))
        return Outcome()

    def reset(self) -> None:
        self._rng = random.Random(self._seed)
        self._anchor = None
        self._last_now = None
        self._reload_sequence()


class RT:
    """Random-Time schedule with exponentially distributed intervals.

    Parameters
    ----------
    mean_interval:
        The mean of the exponential distribution from which
        inter-reinforcer intervals are drawn. Must be ``> 0``.
    rng:
        Optional :class:`random.Random`. If ``None``, a fresh unseeded
        RNG is used. A state snapshot is captured at construction so
        :meth:`reset` returns to the original draw sequence.

    Notes
    -----
    The exponential distribution is memoryless: at any moment, the
    probability of a reinforcer in the next infinitesimal interval
    ``dt`` is ``dt / mean_interval`` independent of elapsed time. RT is
    the time-based analogue of :class:`RR`.

    Intervals are drawn on demand — one at construction (so the first
    reinforcer has an expected wait of ``mean_interval`` from the
    initial anchor) and one after every reinforcement.

    As with :class:`FT` and :class:`VT`, only one reinforcer is emitted
    per :meth:`step` call.

    References
    ----------
    Catania, A. C., & Reynolds, G. S. (1968). A quantitative analysis
    of the responding maintained by interval schedules of
    reinforcement. *Journal of the Experimental Analysis of Behavior*,
    11(3, Pt. 2), 327-383. https://doi.org/10.1901/jeab.1968.11-s327
    """

    __slots__ = (
        "_mean",
        "_initial_state",
        "_rng",
        "_requirement",
        "_anchor",
        "_last_now",
    )

    def __init__(
        self,
        mean_interval: float,
        rng: random.Random | None = None,
    ) -> None:
        if isinstance(mean_interval, bool) or not isinstance(mean_interval, (int, float)):
            raise ScheduleConfigError(
                f"RT requires a numeric mean_interval, got {type(mean_interval).__name__}"
            )
        if mean_interval <= 0:
            raise ScheduleConfigError(
                f"RT requires mean_interval > 0, got {mean_interval}"
            )
        self._mean: float = float(mean_interval)
        self._rng: random.Random = rng if rng is not None else random.Random()
        self._initial_state = self._rng.getstate()
        self._requirement: float = self._rng.expovariate(1.0 / self._mean)
        self._anchor: float | None = None
        self._last_now: float | None = None

    @property
    def mean_interval(self) -> float:
        return self._mean

    def _draw_requirement(self) -> None:
        self._requirement = self._rng.expovariate(1.0 / self._mean)

    def step(self, now: float, event: ResponseEvent | None = None) -> Outcome:
        _check_time(self._last_now, now)
        _check_event(now, event)
        self._last_now = now
        if self._anchor is None:
            self._anchor = now
            return Outcome()
        if now - self._anchor + _TIME_TOL >= self._requirement:
            self._anchor = now
            self._draw_requirement()
            return Outcome(reinforced=True, reinforcer=Reinforcer(time=now))
        return Outcome()

    def reset(self) -> None:
        self._rng.setstate(self._initial_state)
        self._requirement = self._rng.expovariate(1.0 / self._mean)
        self._anchor = None
        self._last_now = None


class EXT:
    """Extinction: no response is ever reinforced.

    Parameters
    ----------
    (none)

    Notes
    -----
    EXT is stateless — :meth:`step` always returns an un-reinforced
    :class:`Outcome`, regardless of ``now``, ``event``, or how often it
    is called. The only dynamic state is ``_last_now``, retained solely
    to enforce the monotonic-time contract shared by all schedules.

    :meth:`reset` clears that retained timestamp; apart from that it is
    a no-op.

    References
    ----------
    Skinner, B. F. (1957). *Schedules of reinforcement* (with C. B.
    Ferster). Appleton-Century-Crofts.
    """

    __slots__ = ("_last_now",)

    def __init__(self) -> None:
        self._last_now: float | None = None

    def step(self, now: float, event: ResponseEvent | None = None) -> Outcome:
        _check_time(self._last_now, now)
        _check_event(now, event)
        self._last_now = now
        return Outcome()

    def reset(self) -> None:
        self._last_now = None
