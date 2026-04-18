"""Interval-family reinforcement schedules: FI, VI, RI, LimitedHold.

In interval schedules a reinforcer becomes available only after a
specified amount of time has elapsed since the last reinforcement.
The subject must emit at least one response after the interval
elapses to actually obtain the reinforcer — this distinguishes
interval schedules from *time* schedules (FT/VT/RT), where the
reinforcer is delivered automatically on interval expiry regardless
of behaviour.

Implementations here follow the Swift OperantKit condition
``numOfResponses > previousNumOfResponses AND elapsed >= interval``:
the step returns a reinforcer only on a response that is *both*
emitted at ``now`` *and* past the currently armed target time.

LimitedHold wrapping
--------------------
Classical FI/VI/RI have no "missed opportunity" notion: any response
after the interval qualifies, no matter how late. The
:class:`LimitedHold` decorator adds a bounded availability window:
if no qualifying response is emitted within ``hold`` time units of
the interval elapsing, the opportunity is withdrawn and the inner
schedule is re-armed from scratch (a new interval is drawn without
any reinforcer being delivered).

To make this possible without leaking into the public
:class:`~contingency.interfaces.Schedule` protocol, the concrete
interval schedules expose two protected hooks:

* ``_arm_time`` — the absolute monotonic time at which the currently
  pending interval elapses (i.e. the earliest time at which a
  qualifying response could produce reinforcement).
* ``_withdraw_and_rearm(now)`` — resample the next interval and
  anchor the new ``_arm_time`` to ``now``, *without* delivering a
  reinforcer. Used by :class:`LimitedHold` when a hold window
  expires.

LimitedHold relies on duck typing of these two attributes; any
interval-family schedule that provides them can be wrapped.

References
----------
- Skinner, B. F. (1957). *Verbal behavior*. Appleton-Century-Crofts.
- Ferster, C. B., & Skinner, B. F. (1957). *Schedules of
  reinforcement*. Appleton-Century-Crofts.
- Fleshler, M., & Hoffman, H. S. (1962). A progression for
  generating variable-interval schedules. *JEAB*, 5(4), 529-530.
  https://doi.org/10.1901/jeab.1962.5-529
- Catania, A. C., & Reynolds, G. S. (1968). A quantitative analysis
  of the responding maintained by interval schedules of
  reinforcement. *JEAB*, 11(3 Pt 2), 327-383.
  https://doi.org/10.1901/jeab.1968.11-s327
"""

from __future__ import annotations

import random

from contingency.entities import Outcome, Reinforcer, ResponseEvent
from contingency.errors import ScheduleConfigError, ScheduleStateError
from contingency.helpers.fleshler_hoffman import generate_intervals

__all__ = ["FI", "VI", "RI", "LimitedHold"]


_TIME_TOL = 1e-9


def _check_time(now: float, last_now: float | None) -> None:
    if last_now is not None and now < last_now - _TIME_TOL:
        raise ScheduleStateError(
            f"non-monotonic time: now={now!r} < previous={last_now!r}"
        )


def _check_event(now: float, event: ResponseEvent | None) -> None:
    if event is not None and abs(event.time - now) > _TIME_TOL:
        raise ScheduleStateError(
            f"event.time={event.time!r} does not match now={now!r}"
        )


class FI:
    """Fixed-Interval schedule.

    The first response emitted at or after ``interval`` time units
    since the last reinforcement (or since construction / last
    reset) produces a reinforcer. Responses during the interval do
    not reset the timer and do not produce reinforcement.

    Parameters
    ----------
    interval:
        Length of the fixed interval. Must be strictly positive.

    Notes
    -----
    The Swift original's criterion, ``numOfResponses >
    previousNumOfResponses && elapsed >= interval``, is preserved:
    the reinforcer is produced only on the *response* that first
    satisfies ``now >= arm_time``. A plain time tick with
    ``event is None`` never produces reinforcement.
    """

    def __init__(self, interval: float) -> None:
        if not (interval > 0):
            raise ScheduleConfigError(
                f"interval must be > 0, got {interval!r}"
            )
        self._interval: float = float(interval)
        self._arm_time: float = self._interval
        self._last_now: float | None = None

    def step(
        self, now: float, event: ResponseEvent | None = None
    ) -> Outcome:
        _check_time(now, self._last_now)
        _check_event(now, event)
        self._last_now = now

        if event is None:
            return Outcome()

        if now + _TIME_TOL >= self._arm_time:
            reinforcer = Reinforcer(time=now)
            self._arm_time = now + self._interval
            return Outcome(reinforced=True, reinforcer=reinforcer)

        return Outcome()

    def reset(self) -> None:
        self._arm_time = self._interval
        self._last_now = None

    # --- LimitedHold hooks -------------------------------------------------

    def _withdraw_and_rearm(self, now: float) -> None:
        """Re-arm the schedule anchored at ``now`` without reinforcing."""
        self._arm_time = now + self._interval


class VI:
    """Variable-Interval schedule with a Fleshler-Hoffman sequence.

    Intervals are drawn from a shuffled Fleshler-Hoffman progression
    with arithmetic mean equal to ``mean_interval``. When the
    sequence is exhausted a fresh sequence is generated
    deterministically from the internal RNG so that a single seed
    pins the whole trajectory of the schedule.

    Parameters
    ----------
    mean_interval:
        Arithmetic mean of the interval distribution. Must be > 0.
    n_intervals:
        Number of intervals per Fleshler-Hoffman cycle. Must be >= 1.
    seed:
        Optional RNG seed for deterministic sequences.
    """

    def __init__(
        self,
        mean_interval: float,
        n_intervals: int = 12,
        seed: int | None = None,
    ) -> None:
        if not (mean_interval > 0):
            raise ScheduleConfigError(
                f"mean_interval must be > 0, got {mean_interval!r}"
            )
        if n_intervals < 1:
            raise ScheduleConfigError(
                f"n_intervals must be >= 1, got {n_intervals!r}"
            )
        self._mean_interval: float = float(mean_interval)
        self._n_intervals: int = int(n_intervals)
        self._seed: int | None = seed
        self._rng: random.Random = random.Random(seed)
        self._sequence: list[float] = self._generate_sequence()
        self._cursor: int = 1
        self._arm_time: float = self._sequence[0]
        self._last_now: float | None = None

    def _generate_sequence(self) -> list[float]:
        # Derive a per-cycle seed from the master RNG so the full
        # trajectory is deterministic given the constructor seed.
        subseed = self._rng.getrandbits(64)
        return generate_intervals(
            self._mean_interval,
            self._n_intervals,
            seed=subseed,
        )

    def _next_interval(self) -> float:
        if self._cursor >= len(self._sequence):
            self._sequence = self._generate_sequence()
            self._cursor = 0
        value = self._sequence[self._cursor]
        self._cursor += 1
        return value

    def step(
        self, now: float, event: ResponseEvent | None = None
    ) -> Outcome:
        _check_time(now, self._last_now)
        _check_event(now, event)
        self._last_now = now

        if event is None:
            return Outcome()

        if now + _TIME_TOL >= self._arm_time:
            reinforcer = Reinforcer(time=now)
            self._arm_time = now + self._next_interval()
            return Outcome(reinforced=True, reinforcer=reinforcer)

        return Outcome()

    def reset(self) -> None:
        self._rng = random.Random(self._seed)
        self._sequence = self._generate_sequence()
        self._cursor = 1
        self._arm_time = self._sequence[0]
        self._last_now = None

    # --- LimitedHold hooks -------------------------------------------------

    def _withdraw_and_rearm(self, now: float) -> None:
        self._arm_time = now + self._next_interval()


class RI:
    """Random-Interval schedule with exponentially distributed intervals.

    Each interval is drawn independently from
    ``Exp(1 / mean_interval)``, giving a memoryless reinforcement
    process with constant hazard rate ``1 / mean_interval``.

    Parameters
    ----------
    mean_interval:
        Mean of the exponential distribution. Must be > 0.
    rng:
        Optional :class:`random.Random` used to draw intervals. When
        ``None``, a freshly-seeded private RNG is used. The RNG's
        state at construction is captured so that :meth:`reset`
        restores bit-identical behaviour.
    """

    def __init__(
        self,
        mean_interval: float,
        rng: random.Random | None = None,
    ) -> None:
        if not (mean_interval > 0):
            raise ScheduleConfigError(
                f"mean_interval must be > 0, got {mean_interval!r}"
            )
        self._mean_interval: float = float(mean_interval)
        self._rng: random.Random = rng if rng is not None else random.Random()
        self._initial_state = self._rng.getstate()
        self._arm_time: float = self._draw_interval()
        self._last_now: float | None = None

    def _draw_interval(self) -> float:
        return self._rng.expovariate(1.0 / self._mean_interval)

    def step(
        self, now: float, event: ResponseEvent | None = None
    ) -> Outcome:
        _check_time(now, self._last_now)
        _check_event(now, event)
        self._last_now = now

        if event is None:
            return Outcome()

        if now + _TIME_TOL >= self._arm_time:
            reinforcer = Reinforcer(time=now)
            self._arm_time = now + self._draw_interval()
            return Outcome(reinforced=True, reinforcer=reinforcer)

        return Outcome()

    def reset(self) -> None:
        self._rng.setstate(self._initial_state)
        self._arm_time = self._draw_interval()
        self._last_now = None

    # --- LimitedHold hooks -------------------------------------------------

    def _withdraw_and_rearm(self, now: float) -> None:
        self._arm_time = now + self._draw_interval()


class LimitedHold:
    """Limited-Hold wrapper around an interval-family schedule.

    Once the inner schedule's interval elapses (i.e. ``now >=
    inner._arm_time``) a reinforcement opportunity becomes
    available. The opportunity remains available for ``hold`` time
    units. If the subject emits a response within that window, the
    inner schedule delivers its reinforcer normally. If the window
    closes before any response, the opportunity is *withdrawn* —
    the inner schedule is re-armed at ``now`` with a freshly
    sampled interval, and no reinforcer is produced.

    Parameters
    ----------
    inner:
        An interval-family schedule providing ``_arm_time`` and
        ``_withdraw_and_rearm(now)`` (FI, VI, or RI from this
        module).
    hold:
        Availability window for each reinforcement opportunity.
        Must be > 0.

    References
    ----------
    - Ferster, C. B., & Skinner, B. F. (1957). *Schedules of
      reinforcement*. Appleton-Century-Crofts. (Ch. 5)
    - Nevin, J. A. (1974). Response strength in multiple schedules.
      *JEAB*, 21(3), 389-408.
      https://doi.org/10.1901/jeab.1974.21-389
    """

    def __init__(self, inner: object, hold: float) -> None:
        if not (hold > 0):
            raise ScheduleConfigError(
                f"hold must be > 0, got {hold!r}"
            )
        if not hasattr(inner, "_arm_time") or not hasattr(
            inner, "_withdraw_and_rearm"
        ):
            raise ScheduleConfigError(
                "LimitedHold requires an interval-family schedule "
                "exposing _arm_time and _withdraw_and_rearm"
            )
        self._inner = inner
        self._hold: float = float(hold)
        self._last_now: float | None = None

    def _expire_if_needed(self, now: float) -> None:
        """Withdraw the current opportunity if the hold window has closed."""
        arm_time: float = self._inner._arm_time  # type: ignore[attr-defined]
        if now > arm_time + self._hold + _TIME_TOL:
            self._inner._withdraw_and_rearm(now)  # type: ignore[attr-defined]

    def step(
        self, now: float, event: ResponseEvent | None = None
    ) -> Outcome:
        _check_time(now, self._last_now)
        _check_event(now, event)
        self._last_now = now

        # On any step (tick or response) first check whether the
        # previously-armed opportunity has expired; if so, withdraw
        # it and re-arm before evaluating the current event.
        self._expire_if_needed(now)

        if event is None:
            return Outcome()

        arm_time: float = self._inner._arm_time  # type: ignore[attr-defined]
        if (
            now + _TIME_TOL >= arm_time
            and now <= arm_time + self._hold + _TIME_TOL
        ):
            # Delegate to the inner schedule so it updates its own
            # state (arm_time resample, sequence cursor, etc.) and
            # emits the canonical Outcome.
            return self._inner.step(now, event)  # type: ignore[attr-defined]

        return Outcome()

    def reset(self) -> None:
        self._inner.reset()  # type: ignore[attr-defined]
        self._last_now = None
