"""Fluent factory for building reinforcement schedules.

:class:`ScheduleBuilder` mirrors the Swift ``ScheduleBuilder`` facade
from the original OperantKit: a thin static-method surface that
delegates to the concrete schedule classes. It exists for ergonomic
parity with the Swift API and as a stable entry point for downstream
code that wants a single import.

Examples
--------
>>> from contingency.builder import ScheduleBuilder
>>> s = ScheduleBuilder.fr(5)
>>> s.n
5
>>> c = ScheduleBuilder.concurrent({"left": ScheduleBuilder.fr(3),
...                                 "right": ScheduleBuilder.vi(30.0, seed=0)})
>>> sorted(c.components.keys())
['left', 'right']

References
----------
Ferster, C. B., & Skinner, B. F. (1957). *Schedules of reinforcement*.
Appleton-Century-Crofts.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from contingency.schedules import (
    CRF,
    DRH,
    DRL,
    DRO,
    EXT,
    FI,
    FR,
    FT,
    RI,
    RR,
    RT,
    VI,
    VR,
    VT,
    Alternative,
    Chained,
    Concurrent,
    LimitedHold,
    Multiple,
    ProgressiveRatio,
    Tandem,
    arithmetic,
    geometric,
    richardson_roberts,
)

if TYPE_CHECKING:
    from contingency.interfaces import Schedule

__all__ = ["ScheduleBuilder"]


class ScheduleBuilder:
    """Static factory mirroring the Swift ``ScheduleBuilder`` API.

    Every method is a pure delegation to the underlying schedule class.
    The builder introduces no additional state or side effects.
    """

    # ------------------------------------------------------------------
    # Ratio family
    # ------------------------------------------------------------------

    @staticmethod
    def fr(n: int) -> FR:
        """Fixed Ratio — reinforce every ``n``-th response."""
        return FR(n)

    @staticmethod
    def vr(mean: float, n_intervals: int = 12, seed: int | None = None) -> VR:
        """Variable Ratio with Fleshler-Hoffman progression."""
        return VR(mean, n_intervals=n_intervals, seed=seed)

    @staticmethod
    def rr(probability: float, rng: random.Random | None = None) -> RR:
        """Random Ratio — Bernoulli reinforcement per response."""
        return RR(probability, rng=rng)

    @staticmethod
    def crf() -> FR:
        """Continuous Reinforcement (``FR(1)``)."""
        return CRF()

    # ------------------------------------------------------------------
    # Interval family
    # ------------------------------------------------------------------

    @staticmethod
    def fi(interval: float) -> FI:
        """Fixed Interval."""
        return FI(interval)

    @staticmethod
    def vi(
        mean_interval: float,
        n_intervals: int = 12,
        seed: int | None = None,
    ) -> VI:
        """Variable Interval with Fleshler-Hoffman progression."""
        return VI(mean_interval, n_intervals=n_intervals, seed=seed)

    @staticmethod
    def ri(mean_interval: float, rng: random.Random | None = None) -> RI:
        """Random Interval (memoryless exponential waits)."""
        return RI(mean_interval, rng=rng)

    @staticmethod
    def limited_hold(inner: Schedule, hold: float) -> LimitedHold:
        """Limited-Hold decorator around an interval schedule."""
        return LimitedHold(inner, hold)

    # ------------------------------------------------------------------
    # Time-based family
    # ------------------------------------------------------------------

    @staticmethod
    def ft(interval: float) -> FT:
        """Fixed Time — response-independent reinforcement."""
        return FT(interval)

    @staticmethod
    def vt(
        mean_interval: float,
        n_intervals: int = 12,
        seed: int | None = None,
    ) -> VT:
        """Variable Time with Fleshler-Hoffman progression."""
        return VT(mean_interval, n_intervals=n_intervals, seed=seed)

    @staticmethod
    def rt(mean_interval: float, rng: random.Random | None = None) -> RT:
        """Random Time (memoryless)."""
        return RT(mean_interval, rng=rng)

    @staticmethod
    def ext() -> EXT:
        """Extinction — never reinforce."""
        return EXT()

    # ------------------------------------------------------------------
    # Compound schedules
    # ------------------------------------------------------------------

    @staticmethod
    def concurrent(
        components: dict[str, Schedule],
        cod: float = 0.0,
        cor: int = 0,
    ) -> Concurrent:
        """Concurrent (Catania, 1966) with COD/COR gating."""
        return Concurrent(components, cod=cod, cor=cor)

    @staticmethod
    def multiple(
        components: list[Schedule],
        stimuli: list[str] | None = None,
    ) -> Multiple:
        """Multiple schedule with distinctive stimuli."""
        return Multiple(components, stimuli=stimuli)

    @staticmethod
    def chained(
        components: list[Schedule],
        stimuli: list[str] | None = None,
    ) -> Chained:
        """Chained schedule with distinctive stimuli."""
        return Chained(components, stimuli=stimuli)

    @staticmethod
    def tandem(components: list[Schedule]) -> Tandem:
        """Tandem schedule (chained without stimuli)."""
        return Tandem(components)

    @staticmethod
    def alternative(first: Schedule, second: Schedule) -> Alternative:
        """Alternative — whichever-first of two components."""
        return Alternative(first, second)

    # ------------------------------------------------------------------
    # Differential reinforcement
    # ------------------------------------------------------------------

    @staticmethod
    def dro(interval: float, dro_type: str = "resetting") -> DRO:
        """Differential Reinforcement of Other behavior."""
        return DRO(interval, dro_type=dro_type)  # type: ignore[arg-type]

    @staticmethod
    def drl(interval: float) -> DRL:
        """Differential Reinforcement of Low rate."""
        return DRL(interval)

    @staticmethod
    def drh(response_count: int, time_window: float) -> DRH:
        """Differential Reinforcement of High rate."""
        return DRH(response_count, time_window)

    # ------------------------------------------------------------------
    # Progressive Ratio
    # ------------------------------------------------------------------

    @staticmethod
    def pr_arithmetic(start: int = 1, step: int = 1) -> ProgressiveRatio:
        """Progressive Ratio with arithmetic step function (Hodos, 1961)."""
        return ProgressiveRatio(arithmetic(start=start, step=step))

    @staticmethod
    def pr_geometric(start: int = 1, ratio: float = 2.0) -> ProgressiveRatio:
        """Progressive Ratio with geometric step function."""
        return ProgressiveRatio(geometric(start=start, ratio=ratio))

    @staticmethod
    def pr_richardson_roberts() -> ProgressiveRatio:
        """Richardson-Roberts (1996) progressive-ratio series."""
        return ProgressiveRatio(richardson_roberts())
