"""Schedule factories and the entry points behind the registered IDs."""

from __future__ import annotations

from typing import Any

from contingency.envs.env import ScheduleEnv
from contingency.schedules import FI, FR

__all__ = ["fi", "fr", "make_fi_env", "make_fr_env"]


def fr(n: int) -> FR:
    """Fixed-ratio ``n``."""
    return FR(n)


def fi(interval: float) -> FI:
    """Fixed-interval ``interval``."""
    return FI(interval)


def make_fr_env(n: int = 5, *, max_time: float = 600.0, **kwargs: Any) -> ScheduleEnv:
    """Entry point of ``contingency/FR-v0``."""
    return ScheduleEnv(fr(n), max_time=max_time, **kwargs)


def make_fi_env(
    interval: float = 30.0, *, max_time: float = 600.0, **kwargs: Any
) -> ScheduleEnv:
    """Entry point of ``contingency/FI-v0``."""
    return ScheduleEnv(fi(interval), max_time=max_time, **kwargs)
