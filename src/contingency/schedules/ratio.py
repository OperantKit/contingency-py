"""Ratio-family reinforcement schedules.

This module implements the four canonical ratio schedules of the
experimental analysis of behavior:

* :class:`FR` — Fixed Ratio. The subject must emit a fixed count of
  responses for each reinforcer.
* :func:`CRF` — Continuous Reinforcement, a factory that returns
  ``FR(1)``. Every response is reinforced.
* :class:`VR` — Variable Ratio. Ratio requirements are drawn from a
  Fleshler-Hoffman progression with a configured mean.
* :class:`RR` — Random Ratio. Each response is reinforced independently
  with probability ``p``.

Each schedule is unit-agnostic with respect to time (``now`` is a
monotonic float on a caller-declared clock). Ratio schedules use
``now`` only to time-stamp emitted :class:`Reinforcer` objects; the
reinforcement logic is driven by the presence of a
:class:`ResponseEvent`. ``step`` and ``reset`` run on the Rust engine
(``contingency_core``); the classes validate their arguments and keep
their parameters.

References
----------
Fleshler, M., & Hoffman, H. S. (1962). A progression for generating
variable-interval schedules. *Journal of the Experimental Analysis of
Behavior*, 5(4), 529-530. https://doi.org/10.1901/jeab.1962.5-529

Skinner, B. F. (1957). *Schedules of reinforcement* (with C. B.
Ferster). Appleton-Century-Crofts.
"""

from __future__ import annotations

import random

from contingency.entities import Outcome, ResponseEvent
from contingency.errors import ScheduleConfigError
from contingency.schedules._rust import (
    fresh_seed,
    rust_reset,
    rust_schedule,
    seed_of,
)

__all__ = ["FR", "VR", "RR", "CRF"]


class FR:
    """Fixed-Ratio schedule: reinforce every ``n``-th response.

    Parameters
    ----------
    n:
        Ratio requirement. Must be ``>= 1``. ``FR(1)`` is equivalent to
        continuous reinforcement (see :func:`CRF`).

    Notes
    -----
    The schedule counts the steps called with a non-``None`` ``event``.
    When the count reaches ``n`` it is reset to zero and a
    :class:`Reinforcer` timestamped at ``now`` is emitted.

    References
    ----------
    Skinner, B. F. (1957). *Schedules of reinforcement* (with C. B.
    Ferster). Appleton-Century-Crofts.
    """

    __slots__ = ("_n", "_rust")

    def __init__(self, n: int) -> None:
        if not isinstance(n, int) or isinstance(n, bool):
            raise ScheduleConfigError(f"FR requires an int n, got {type(n).__name__}")
        if n < 1:
            raise ScheduleConfigError(f"FR requires n >= 1, got {n}")
        self._n: int = n
        self._rust = rust_schedule({"type": "FR", "params": {"n": n}})

    @property
    def n(self) -> int:
        """The configured ratio requirement."""
        return self._n

    @property
    def engine(self) -> str:
        """Always ``"rust"``: the Rust engine runs this schedule."""
        return "rust"

    def step(self, now: float, event: ResponseEvent | None = None) -> Outcome:
        return self._rust.step(now, event)  # type: ignore[no-any-return]

    def reset(self) -> None:
        self._rust.reset()


def CRF() -> FR:  # noqa: N802 - acronym per behavior-analytic convention
    """Continuous Reinforcement — a factory returning :class:`FR(1)`.

    Every response produces a reinforcer. CRF is the limiting case of
    fixed ratio with ratio requirement one.

    References
    ----------
    Skinner, B. F. (1957). *Schedules of reinforcement* (with C. B.
    Ferster). Appleton-Century-Crofts.
    """
    return FR(1)


class VR:
    """Variable-Ratio schedule driven by a Fleshler-Hoffman progression.

    Parameters
    ----------
    mean:
        Target mean ratio requirement, ``> 0``.
    n_intervals:
        Number of ratio values generated per cycle. Defaults to 12,
        matching conventional VI/VR practice.
    seed:
        Optional RNG seed for deterministic sequences. When provided,
        :meth:`reset` reshuffles with a fresh sub-seed derived from the
        original seed so that ``reset()`` is reproducible.

    Notes
    -----
    Each step with a response increments a counter. When the counter
    meets the current requirement, reinforcement is delivered, the
    counter is zeroed, and the next requirement is taken from the
    Fleshler-Hoffman sequence. When the sequence is exhausted, a fresh
    sequence is generated from the schedule's seed, preserving
    determinism. Without ``seed`` every :meth:`reset` draws a new
    sequence.

    References
    ----------
    Fleshler, M., & Hoffman, H. S. (1962). A progression for generating
    variable-interval schedules. *Journal of the Experimental Analysis
    of Behavior*, 5(4), 529-530. https://doi.org/10.1901/jeab.1962.5-529

    Skinner, B. F. (1957). *Schedules of reinforcement* (with C. B.
    Ferster). Appleton-Century-Crofts.
    """

    __slots__ = ("_mean", "_n_intervals", "_seed", "_rust")

    def __init__(
        self,
        mean: float,
        n_intervals: int = 12,
        seed: int | None = None,
    ) -> None:
        if isinstance(mean, bool) or not isinstance(mean, (int, float)):
            raise ScheduleConfigError(
                f"VR requires a numeric mean, got {type(mean).__name__}"
            )
        if mean <= 0:
            raise ScheduleConfigError(f"VR requires mean > 0, got {mean}")
        if not isinstance(n_intervals, int) or isinstance(n_intervals, bool):
            raise ScheduleConfigError(
                f"VR requires an int n_intervals, got {type(n_intervals).__name__}"
            )
        if n_intervals < 1:
            raise ScheduleConfigError(
                f"VR requires n_intervals >= 1, got {n_intervals}"
            )
        self._mean: float = float(mean)
        self._n_intervals: int = n_intervals
        self._seed: int | None = seed
        params: dict[str, object] = {"mean": self._mean, "n_intervals": n_intervals}
        params["seed"] = seed if seed is not None else fresh_seed()
        self._rust = rust_schedule({"type": "VR", "params": params})

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def n_intervals(self) -> int:
        return self._n_intervals

    @property
    def engine(self) -> str:
        """Always ``"rust"``: the Rust engine runs this schedule."""
        return "rust"

    def step(self, now: float, event: ResponseEvent | None = None) -> Outcome:
        return self._rust.step(now, event)  # type: ignore[no-any-return]

    def reset(self) -> None:
        self._rust = rust_reset(self._rust, self._seed)


class RR:
    """Random-Ratio schedule: each response reinforced with probability ``p``.

    Parameters
    ----------
    probability:
        Reinforcement probability per response. Must satisfy
        ``0 < p <= 1``.
    rng:
        Optional :class:`random.Random`. The constructor draws one
        63-bit seed from it (advancing it once) and the Rust engine
        draws from that seed; the schedule does not use ``rng`` again.
    seed:
        Optional RNG seed, an int in ``[0, 2**64)``. Draws the same
        sequence as ``rng=random.Random(seed)`` did when the schedule
        ran in Python. Give ``rng`` or ``seed``, not both. With either,
        :meth:`reset` returns to the original draw sequence; with
        neither, the seed is fresh.

    References
    ----------
    Skinner, B. F. (1957). *Schedules of reinforcement* (with C. B.
    Ferster). Appleton-Century-Crofts.
    """

    __slots__ = ("_p", "_rust")

    def __init__(
        self,
        probability: float,
        rng: random.Random | None = None,
        *,
        seed: int | None = None,
    ) -> None:
        if isinstance(probability, bool) or not isinstance(probability, (int, float)):
            raise ScheduleConfigError(
                f"RR requires a numeric probability, got {type(probability).__name__}"
            )
        if not (0.0 < probability <= 1.0):
            raise ScheduleConfigError(
                f"RR requires 0 < probability <= 1, got {probability}"
            )
        self._p: float = float(probability)
        params: dict[str, object] = {"probability": self._p, "seed": seed_of("RR", rng, seed)}
        self._rust = rust_schedule({"type": "RR", "params": params})

    @property
    def probability(self) -> float:
        return self._p

    @property
    def engine(self) -> str:
        """Always ``"rust"``: the Rust engine runs this schedule."""
        return "rust"

    def step(self, now: float, event: ResponseEvent | None = None) -> Outcome:
        return self._rust.step(now, event)  # type: ignore[no-any-return]

    def reset(self) -> None:
        self._rust.reset()
