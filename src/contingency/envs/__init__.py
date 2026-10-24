"""Gymnasium environments for reinforcement schedules (extra ``gym``).

Importing this module registers the environment IDs::

    import gymnasium as gym
    import contingency.envs  # registers contingency/FR-v0, FI-v0, ConcVIVI-v0, ConcPhases-v0

    env = gym.make("contingency/FI-v0", interval=30.0, max_time=600.0)
    env = gym.make("contingency/ConcVIVI-v0", mean_left=30.0, mean_right=60.0, cod=2.0)
    env = gym.make("contingency/ConcPhases-v0")  # three-phase resurgence procedure

Gymnasium 1.0 removed automatic registration through entry points, so
the import is what makes the IDs available to :func:`gymnasium.make`.
:mod:`contingency.envs.tabular` holds a reference tabular Q-learner.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gymnasium as gym

from contingency.envs.env import ScheduleEnv, ScheduleFactory
from contingency.envs.presets import (
    make_conc_phases_env,
    make_conc_vi_vi_env,
    make_fi_env,
    make_fr_env,
)

__all__ = ["ScheduleEnv", "ScheduleFactory"]

_ENTRY_POINTS: dict[str, Callable[..., Any]] = {
    "contingency/FR-v0": make_fr_env,
    "contingency/FI-v0": make_fi_env,
    "contingency/ConcVIVI-v0": make_conc_vi_vi_env,
    "contingency/ConcPhases-v0": make_conc_phases_env,
}

for _env_id, _entry_point in _ENTRY_POINTS.items():
    if _env_id not in gym.registry:
        gym.register(id=_env_id, entry_point=_entry_point)
