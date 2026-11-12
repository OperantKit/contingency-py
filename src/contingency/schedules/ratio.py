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
:class:`ResponseEvent`.

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

from contingency.entities import Outcome, Reinforcer, ResponseEvent
from contingency.errors import ScheduleConfigError, ScheduleStateError
from contingency.helpers.fleshler_hoffman import generate_ratios
from contingency.schedules._rust import (
    engine_name,
    fresh_seed,
    random_source,
    rust_or_none,
    rust_reset,
)

__all__ = ["FR", "VR", "RR", "CRF"]

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


class FR:
    """Fixed-Ratio schedule: reinforce every ``n``-th response.

    Parameters
    ----------
    n:
        Ratio requirement. Must be ``>= 1``. ``FR(1)`` is equivalent to
        continuous reinforcement (see :func:`CRF`).

    Notes
    -----
    The schedule keeps an internal response counter that is incremented
    each time :meth:`step` is called with a non-``None`` ``event``.
    When the counter reaches ``n`` it is reset to zero and a
    :class:`Reinforcer` timestamped at ``now`` is emitted.

    References
    ----------
    Skinner, B. F. (1957). *Schedules of reinforcement* (with C. B.
    Ferster). Appleton-Century-Crofts.
    """

    __slots__ = ("_n", "_count", "_last_now", "_rust")

    def __init__(self, n: int) -> None:
        if not isinstance(n, int) or isinstance(n, bool):
            raise ScheduleConfigError(f"FR requires an int n, got {type(n).__name__}")
        if n < 1:
            raise ScheduleConfigError(f"FR requires n >= 1, got {n}")
        self._n: int = n
        self._count: int = 0
        self._last_now: float | None = None
        self._rust = rust_or_none({"type": "FR", "params": {"n": n}})

    @property
    def n(self) -> int:
        """The configured ratio requirement."""
        return self._n

    @property
    def engine(self) -> str:
        """``"rust"`` when the Rust engine runs this schedule, else ``"python"``."""
        return engine_name(self._rust)

    def step(self, now: float, event: ResponseEvent | None = None) -> Outcome:
        if self._rust is not None:
            return self._rust.step(now, event)  # type: ignore[no-any-return]
        _check_time(self._last_now, now)
        _check_event(now, event)
        self._last_now = now
        if event is None:
            return Outcome()
        self._count += 1
        if self._count >= self._n:
            self._count = 0
            return Outcome(reinforced=True, reinforcer=Reinforcer(time=now))
        return Outcome()

    def reset(self) -> None:
        if self._rust is not None:
            self._rust.reset()
            return
        self._count = 0
        self._last_now = None


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
    Each step with a response increments an internal counter. When the
    counter meets the current requirement, reinforcement is delivered,
    the counter is zeroed, and the next requirement is taken from the
    pre-generated Fleshler-Hoffman sequence. When the sequence is
    exhausted, a fresh sequence is generated using a seed derived from
    the internal RNG, preserving determinism.

    References
    ----------
    Fleshler, M., & Hoffman, H. S. (1962). A progression for generating
    variable-interval schedules. *Journal of the Experimental Analysis
    of Behavior*, 5(4), 529-530. https://doi.org/10.1901/jeab.1962.5-529

    Skinner, B. F. (1957). *Schedules of reinforcement* (with C. B.
    Ferster). Appleton-Century-Crofts.
    """

    __slots__ = (
        "_mean",
        "_n_intervals",
        "_seed",
        "_rng",
        "_sequence",
        "_cursor",
        "_count",
        "_requirement",
        "_last_now",
        "_rust",
    )

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
        self._rng: random.Random = random.Random(seed)
        self._sequence: list[int] = []
        self._cursor: int = 0
        self._count: int = 0
        self._requirement: int = 0
        self._last_now: float | None = None
        self._reload_sequence()
        params: dict[str, object] = {"mean": self._mean, "n_intervals": n_intervals}
        params["seed"] = seed if seed is not None else fresh_seed()
        self._rust = rust_or_none({"type": "VR", "params": params})

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def n_intervals(self) -> int:
        return self._n_intervals

    @property
    def engine(self) -> str:
        """``"rust"`` when the Rust engine runs this schedule, else ``"python"``."""
        return engine_name(self._rust)

    def _next_sub_seed(self) -> int:
        """Derive a fresh deterministic seed from the internal RNG."""
        return self._rng.randrange(0, 2**31 - 1)

    def _reload_sequence(self) -> None:
        sub_seed = self._next_sub_seed()
        self._sequence = generate_ratios(
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
        if self._rust is not None:
            return self._rust.step(now, event)  # type: ignore[no-any-return]
        _check_time(self._last_now, now)
        _check_event(now, event)
        self._last_now = now
        if event is None:
            return Outcome()
        self._count += 1
        if self._count >= self._requirement:
            self._count = 0
            self._advance_requirement()
            return Outcome(reinforced=True, reinforcer=Reinforcer(time=now))
        return Outcome()

    def reset(self) -> None:
        if self._rust is not None:
            self._rust = rust_reset(self._rust, self._seed)
            return
        self._rng = random.Random(self._seed)
        self._count = 0
        self._last_now = None
        self._reload_sequence()


class RR:
    """Random-Ratio schedule: each response reinforced with probability ``p``.

    Parameters
    ----------
    probability:
        Reinforcement probability per response. Must satisfy
        ``0 < p <= 1``.
    rng:
        Optional :class:`random.Random` instance for deterministic
        Bernoulli draws. If ``None``, a fresh unseeded RNG is used.
        A seeded ``Random`` snapshot is captured at construction so
        :meth:`reset` returns to the original draw sequence.
    seed:
        Optional RNG seed, an int in ``[0, 2**64)``. Draws the same
        sequence as ``rng=random.Random(seed)``. Give ``rng`` or ``seed``,
        not both. A caller-supplied ``rng`` runs the schedule in Python
        (its state cannot be handed to the Rust engine); ``seed`` or
        neither runs it on the Rust engine when it is installed.

    References
    ----------
    Skinner, B. F. (1957). *Schedules of reinforcement* (with C. B.
    Ferster). Appleton-Century-Crofts.
    """

    __slots__ = ("_p", "_initial_state", "_rng", "_last_now", "_rust")

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
        self._rng, rust_ok = random_source("RR", rng, seed)
        # Snapshot the RNG state so reset() restores the initial draw
        # sequence even when the caller supplies a shared Random.
        self._initial_state = self._rng.getstate()
        self._last_now: float | None = None
        params: dict[str, object] = {"probability": self._p}
        params["seed"] = seed if seed is not None else fresh_seed()
        self._rust = rust_or_none({"type": "RR", "params": params}) if rust_ok else None

    @property
    def probability(self) -> float:
        return self._p

    @property
    def engine(self) -> str:
        """``"rust"`` when the Rust engine runs this schedule, else ``"python"``."""
        return engine_name(self._rust)

    def step(self, now: float, event: ResponseEvent | None = None) -> Outcome:
        if self._rust is not None:
            return self._rust.step(now, event)  # type: ignore[no-any-return]
        _check_time(self._last_now, now)
        _check_event(now, event)
        self._last_now = now
        if event is None:
            return Outcome()
        if self._rng.random() < self._p:
            return Outcome(reinforced=True, reinforcer=Reinforcer(time=now))
        return Outcome()

    def reset(self) -> None:
        if self._rust is not None:
            self._rust.reset()
            return
        self._rng.setstate(self._initial_state)
        self._last_now = None
