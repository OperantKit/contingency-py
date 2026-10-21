"""Gymnasium environments for reinforcement schedules (extra ``gym``).

Importing this module registers the environment IDs::

    import gymnasium as gym
    import contingency.envs  # registers contingency/FR-v0 and contingency/FI-v0

    env = gym.make("contingency/FI-v0", interval=30.0, max_time=600.0)

Gymnasium 1.0 removed automatic registration through entry points, so
the import is what makes the IDs available to :func:`gymnasium.make`.
"""

from __future__ import annotations

import gymnasium as gym

from contingency.envs.env import ScheduleEnv, ScheduleFactory
from contingency.envs.presets import make_fi_env, make_fr_env

__all__ = ["ScheduleEnv", "ScheduleFactory"]

_ENTRY_POINTS = {
    "contingency/FR-v0": make_fr_env,
    "contingency/FI-v0": make_fi_env,
}

for _env_id, _entry_point in _ENTRY_POINTS.items():
    if _env_id not in gym.registry:
        gym.register(id=_env_id, entry_point=_entry_point)
