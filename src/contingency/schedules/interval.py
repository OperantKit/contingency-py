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

``step`` and ``reset`` of FI, VI and RI run on the Rust engine
(``contingency_core``); the classes validate their arguments and keep
their parameters.

LimitedHold wrapping
--------------------
Classical FI/VI/RI have no "missed opportunity" notion: any response
after the interval qualifies, no matter how late. The
:class:`LimitedHold` decorator adds a bounded availability window:
if no qualifying response is emitted within ``hold`` time units of
the interval elapsing, the opportunity is withdrawn and the inner
schedule is re-armed from scratch (a new interval is drawn without
any reinforcer being delivered). LimitedHold takes FI, VI or RI and
runs on the Rust engine; any other inner schedule raises
:class:`~contingency.errors.ScheduleConfigError`.

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
from typing import Any

from contingency.entities import Outcome, ResponseEvent
from contingency.errors import ScheduleConfigError
from contingency.schedules._rust import (
    fresh_seed,
    rust_reset,
    rust_schedule,
    seed_of,
)

__all__ = ["FI", "VI", "RI", "LimitedHold"]


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
        self._rust_spec: dict[str, Any] = {"type": "FI", "params": {"interval": self._interval}}
        self._rust = rust_schedule(self._rust_spec)

    @property
    def engine(self) -> str:
        """Always ``"rust"``: the Rust engine runs this schedule."""
        return "rust"

    def step(
        self, now: float, event: ResponseEvent | None = None
    ) -> Outcome:
        return self._rust.step(now, event)  # type: ignore[no-any-return]

    def reset(self) -> None:
        self._rust.reset()


class VI:
    """Variable-Interval schedule with a Fleshler-Hoffman sequence.

    Intervals are drawn from a shuffled Fleshler-Hoffman progression
    with arithmetic mean equal to ``mean_interval``. When the
    sequence is exhausted a fresh sequence is generated
    deterministically from the schedule's seed so that a single seed
    pins the whole trajectory of the schedule.

    Parameters
    ----------
    mean_interval:
        Arithmetic mean of the interval distribution. Must be > 0.
    n_intervals:
        Number of intervals per Fleshler-Hoffman cycle. Must be >= 1.
    seed:
        Optional RNG seed for deterministic sequences. Without it every
        :meth:`reset` draws a new sequence.
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
        vi_params: dict[str, object] = {
            "mean_interval": self._mean_interval,
            "n_intervals": self._n_intervals,
        }
        vi_params["seed"] = seed if seed is not None else fresh_seed()
        self._rust_spec: dict[str, Any] = {"type": "VI", "params": vi_params}
        self._rust = rust_schedule(self._rust_spec)

    @property
    def engine(self) -> str:
        """Always ``"rust"``: the Rust engine runs this schedule."""
        return "rust"

    def step(
        self, now: float, event: ResponseEvent | None = None
    ) -> Outcome:
        return self._rust.step(now, event)  # type: ignore[no-any-return]

    def reset(self) -> None:
        self._rust = rust_reset(self._rust, self._seed)


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
        Optional :class:`random.Random`. The constructor draws one
        63-bit seed from it (advancing it once) and the Rust engine
        draws the intervals from that seed; the schedule does not use
        ``rng`` again.
    seed:
        Optional RNG seed, an int in ``[0, 2**64)``. Draws the same
        sequence as ``rng=random.Random(seed)`` did when the schedule
        ran in Python. Give ``rng`` or ``seed``, not both. With either,
        :meth:`reset` restores bit-identical behaviour; with neither,
        the seed is fresh.
    """

    def __init__(
        self,
        mean_interval: float,
        rng: random.Random | None = None,
        *,
        seed: int | None = None,
    ) -> None:
        if not (mean_interval > 0):
            raise ScheduleConfigError(
                f"mean_interval must be > 0, got {mean_interval!r}"
            )
        self._mean_interval: float = float(mean_interval)
        ri_params: dict[str, object] = {
            "mean_interval": self._mean_interval,
            "seed": seed_of("RI", rng, seed),
        }
        self._rust_spec: dict[str, Any] = {"type": "RI", "params": ri_params}
        self._rust = rust_schedule(self._rust_spec)

    @property
    def engine(self) -> str:
        """Always ``"rust"``: the Rust engine runs this schedule."""
        return "rust"

    def step(
        self, now: float, event: ResponseEvent | None = None
    ) -> Outcome:
        return self._rust.step(now, event)  # type: ignore[no-any-return]

    def reset(self) -> None:
        self._rust.reset()


class LimitedHold:
    """Limited-Hold wrapper around an interval-family schedule.

    Once the inner schedule's interval elapses a reinforcement
    opportunity becomes available. The opportunity remains available for ``hold`` time
    units. If the subject emits a response within that window, the
    inner schedule delivers its reinforcer normally. If the window
    closes before any response, the opportunity is *withdrawn* —
    the inner schedule is re-armed at ``now`` with a freshly
    sampled interval, and no reinforcer is produced.

    Parameters
    ----------
    inner:
        FI, VI or RI from this module. Any other schedule raises
        :class:`ScheduleConfigError`. The LimitedHold runs on the Rust
        engine with its own copy of ``inner`` built from the arguments
        of ``inner``, so stepping the ``inner`` object separately does
        not affect it.
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

    def __init__(self, inner: FI | VI | RI, hold: float) -> None:
        if not (hold > 0):
            raise ScheduleConfigError(
                f"hold must be > 0, got {hold!r}"
            )
        if not isinstance(inner, (FI, VI, RI)):
            raise ScheduleConfigError(
                f"LimitedHold requires FI, VI or RI, got {type(inner).__name__}"
            )
        self._inner: FI | VI | RI = inner
        self._hold: float = float(hold)
        self._rust = rust_schedule(
            {"type": "LimitedHold", "params": {"inner": inner._rust_spec, "hold": self._hold}}
        )

    @property
    def engine(self) -> str:
        """Always ``"rust"``: the Rust engine runs this schedule."""
        return "rust"

    def step(
        self, now: float, event: ResponseEvent | None = None
    ) -> Outcome:
        return self._rust.step(now, event)  # type: ignore[no-any-return]

    def reset(self) -> None:
        if isinstance(self._inner, VI) and self._inner._seed is None:
            # An unseeded VI draws a new sequence on every reset.
            inner_spec = dict(self._inner._rust_spec)
            inner_spec["params"] = {**inner_spec["params"], "seed": fresh_seed()}
            self._rust = rust_schedule(
                {"type": "LimitedHold", "params": {"inner": inner_spec, "hold": self._hold}}
            )
        else:
            self._rust.reset()
