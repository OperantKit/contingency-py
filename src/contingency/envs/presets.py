"""Schedule factories and the entry points behind the registered IDs."""

from __future__ import annotations

from typing import Any

from contingency.envs.env import ScheduleEnv
from contingency.schedules import FI, FR, VI, Concurrent

__all__ = [
    "conc_vi_vi",
    "fi",
    "fr",
    "make_conc_vi_vi_env",
    "make_fi_env",
    "make_fr_env",
]

CONC_OPERANDA = ("left", "right")


def fr(n: int) -> FR:
    """Fixed-ratio ``n``."""
    return FR(n)


def fi(interval: float) -> FI:
    """Fixed-interval ``interval``."""
    return FI(interval)


def conc_vi_vi(
    mean_left: float, mean_right: float, cod: float, *, seed: int | None = None
) -> Concurrent:
    """Concurrent VI ``mean_left`` VI ``mean_right`` with changeover delay ``cod``.

    The components sit on operanda ``"left"`` and ``"right"``. With a
    ``seed``, the left VI uses ``seed`` and the right VI ``seed + 1``, so
    equal means still give the two operanda different interval sequences.
    """
    right_seed = None if seed is None else seed + 1
    return Concurrent(
        {
            "left": VI(mean_left, seed=seed),
            "right": VI(mean_right, seed=right_seed),
        },
        cod=cod,
    )


def make_fr_env(n: int = 5, *, max_time: float = 600.0, **kwargs: Any) -> ScheduleEnv:
    """Entry point of ``contingency/FR-v0``."""
    return ScheduleEnv(fr(n), max_time=max_time, **kwargs)


def make_fi_env(
    interval: float = 30.0, *, max_time: float = 600.0, **kwargs: Any
) -> ScheduleEnv:
    """Entry point of ``contingency/FI-v0``."""
    return ScheduleEnv(fi(interval), max_time=max_time, **kwargs)


def make_conc_vi_vi_env(
    mean_left: float = 30.0,
    mean_right: float = 60.0,
    cod: float = 2.0,
    *,
    max_time: float = 600.0,
    **kwargs: Any,
) -> ScheduleEnv:
    """Entry point of ``contingency/ConcVIVI-v0``.

    Action ``1`` responds on ``"left"`` and ``2`` on ``"right"``; the
    operanda are fixed, so ``operanda`` cannot be passed. The VI sequences
    are drawn again on every ``reset()``.
    """
    if "operanda" in kwargs:
        raise TypeError("ConcVIVI uses the fixed operanda ('left', 'right'); do not pass operanda")
    # Build once so that bad parameters fail here, not at the first reset().
    conc_vi_vi(mean_left, mean_right, cod, seed=0)

    def factory(seed: int) -> Concurrent:
        return conc_vi_vi(mean_left, mean_right, cod, seed=seed)

    return ScheduleEnv(factory, max_time=max_time, operanda=CONC_OPERANDA, **kwargs)
