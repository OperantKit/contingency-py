"""Tabular Q-learning for :class:`~contingency.envs.ScheduleEnv` sessions.

A small, dependency-free reference learner: an epsilon-greedy Q-learner
over a discretised observation, trained online one session per episode.
It approximates the optimal policy of a schedule so that a model
subject (an LLM, a hand-written agent) can be compared against it under
the same environment.

    import gymnasium as gym
    import contingency.envs
    from contingency.envs.tabular import run_greedy, train_q

    env = gym.make("contingency/FI-v0", interval=30.0, response_cost=0.1)
    q = train_q(env, episodes=100, seed=0)
    times, actions, last_sr = run_greedy(env, q, seed=1000)

The default state, :func:`fi_state`, is meant for one operandum (pass
your own ``state`` and ``n_states`` for concurrent schedules): the
responses since the last reinforcer (capped) and the time since the
last reinforcer in bins. Without the time bins the learner cannot tell
early from late in an interval.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

__all__ = ["FI_N_STATES", "StateFn", "fi_state", "greedy", "run_greedy", "train_q"]

StateFn = Callable[[dict[str, Any]], tuple[int, ...]]
"""Maps an observation to a tuple of table indices."""

_CAP = 4
_TIME_BIN = 5.0
_N_TIME_BINS = 12

FI_N_STATES: tuple[int, int] = (_CAP + 1, _N_TIME_BINS)
"""Table shape (without the action axis) of :func:`fi_state` with its defaults."""


def fi_state(
    obs: dict[str, Any],
    *,
    cap: int = _CAP,
    time_bin: float = _TIME_BIN,
    n_time_bins: int = _N_TIME_BINS,
) -> tuple[int, int]:
    """``(responses since reinforcer capped at cap, time-since-reinforcer bin)``.

    Uses the first operandum. The last bin collects every time at or
    beyond ``time_bin * (n_time_bins - 1)``.
    """
    responses = min(int(obs["responses_since_sr"][0]), cap)
    time = int(float(obs["time_since_sr"][0]) // time_bin)
    return responses, min(time, n_time_bins - 1)


def greedy(q_values: np.ndarray, rng: np.random.Generator) -> int:
    """Index of the largest value, ties broken at random.

    With ties going to action ``0`` ("no response"), an unvisited state
    would keep the agent silent for the rest of the session.
    """
    best = np.flatnonzero(q_values == q_values.max())
    return int(rng.choice(best))


def train_q(
    env: Any,
    *,
    episodes: int = 100,
    seed: int = 0,
    state: StateFn = fi_state,
    n_states: tuple[int, ...] = FI_N_STATES,
    alpha: float = 0.1,
    gamma: float = 0.95,
    epsilon: float = 0.1,
) -> np.ndarray:
    """Train a Q table of shape ``(*n_states, n_actions)`` on ``env``.

    Episode ``i`` starts with ``env.reset(seed=i)``. Exploration and tie
    breaking use ``numpy.random.default_rng(seed)``. The update is
    one-step Q-learning; the bootstrap term is dropped on termination
    (not on truncation at the session end).
    """
    if episodes < 0:
        raise ValueError(f"episodes must be >= 0, got {episodes!r}")
    rng = np.random.default_rng(seed)
    n_actions = int(env.action_space.n)
    q = np.zeros((*n_states, n_actions))
    for episode in range(episodes):
        obs, _ = env.reset(seed=episode)
        s = state(obs)
        while True:
            explore = rng.random() < epsilon
            action = int(rng.integers(n_actions)) if explore else greedy(q[s], rng)
            obs, reward, terminated, truncated, _ = env.step(action)
            nxt = state(obs)
            target = reward + (0.0 if terminated else gamma * q[nxt].max())
            q[(*s, action)] += alpha * (target - q[(*s, action)])
            s = nxt
            if terminated or truncated:
                break
    return q


def run_greedy(
    env: Any, q: np.ndarray, *, seed: int, state: StateFn = fi_state
) -> tuple[list[float], list[int], list[int]]:
    """Run one session under the greedy policy of ``q``.

    Returns ``(times, actions, last_sr)`` per step: the session time after
    the step (``info["time"]``), the action taken, and whether that step
    was reinforced (``obs["last_sr"]``). Ties are broken with
    ``numpy.random.default_rng(seed)``.
    """
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    times: list[float] = []
    actions: list[int] = []
    last_sr: list[int] = []
    while True:
        action = greedy(q[state(obs)], rng)
        obs, _, terminated, truncated, info = env.step(action)
        times.append(float(info["time"]))
        actions.append(action)
        last_sr.append(int(obs["last_sr"]))
        if terminated or truncated:
            return times, actions, last_sr
