"""Concurrent VI VI Gymnasium environment and a learning check (extra ``gym``)."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import pytest

gym = pytest.importorskip("gymnasium")
np = pytest.importorskip("numpy")

from gymnasium.utils.env_checker import check_env  # noqa: E402

import contingency.envs  # noqa: E402,F401  (registers the environment IDs)
from contingency.envs import ScheduleEnv  # noqa: E402
from contingency.envs.presets import conc_vi_vi, make_conc_vi_vi_env  # noqa: E402
from contingency.schedules import FR, Concurrent  # noqa: E402

CONC_ID = "contingency/ConcVIVI-v0"

Policy = Callable[[dict[str, Any]], int]


def _count_reinforcers(env: Any, seed: int, policy: Policy) -> int:
    obs, _ = env.reset(seed=seed)
    n_sr = 0
    while True:
        obs, reward, terminated, truncated, _ = env.step(policy(obs))
        n_sr += int(reward > 0)
        if terminated or truncated:
            return n_sr


# --- the environment ------------------------------------------------------


def test_registered_conc_env_passes_check_env_without_warnings() -> None:
    env = gym.make(CONC_ID)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_env(env.unwrapped, skip_render_check=True)


def test_conc_env_has_two_operanda_and_three_actions() -> None:
    env = make_conc_vi_vi_env()
    assert env.operanda == ("left", "right")
    assert env.action_space == gym.spaces.Discrete(3)
    assert env.observation_space["responses_since_sr"].shape == (2,)
    assert env.max_time == 600.0


def test_conc_vi_vi_builds_concurrent_with_cod() -> None:
    schedule = conc_vi_vi(30.0, 60.0, 2.0, seed=3)
    assert isinstance(schedule, Concurrent)
    assert schedule.cod == 2.0


def test_conc_vi_vi_components_use_different_seeds() -> None:
    # Equal means must still give the two operanda different interval sequences.
    env = ScheduleEnv(
        lambda seed: conc_vi_vi(10.0, 10.0, 0.0, seed=seed),
        max_time=400.0,
        operanda=("left", "right"),
    )
    env.reset(seed=0)
    left = [env.step(1)[1] for _ in range(400)]
    env.reset(seed=0)
    right = [env.step(2)[1] for _ in range(400)]
    assert left != right


def test_conc_env_is_reproducible_per_seed() -> None:
    env = gym.make(CONC_ID)
    rng = np.random.default_rng(0)
    actions = [int(a) for a in rng.integers(3, size=600)]

    def rewards(seed: int) -> list[float]:
        env.reset(seed=seed)
        return [env.step(a)[1] for a in actions]

    assert rewards(5) == rewards(5)
    assert rewards(5) != rewards(6)


def test_changeover_delay_is_reported_in_info() -> None:
    env = ScheduleEnv(
        Concurrent({"left": FR(1), "right": FR(1)}, cod=2.0),
        max_time=20.0,
        operanda=("left", "right"),
    )
    env.reset(seed=0)
    _, reward, _, _, info = env.step(1)
    assert reward == 1.0
    _, reward, _, _, info = env.step(2)  # changeover at t=2
    assert reward == 0.0
    assert info["outcome_meta"]["cod_suppressed"] is True
    assert info["outcome_meta"]["operandum"] == "right"


def test_kwargs_override_registered_defaults() -> None:
    env = gym.make(CONC_ID, mean_left=5.0, mean_right=5.0, cod=0.0, max_time=100.0)
    base = env.unwrapped
    assert base.max_time == 100.0
    env.reset(seed=0)
    n_sr = sum(env.step(1)[1] > 0 for _ in range(100))
    assert n_sr >= 10  # VI 5 on the left over 100 time units


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mean_left": 0.0},
        {"mean_right": -1.0},
        {"cod": -0.5},
    ],
)
def test_invalid_parameters_raise_at_construction(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        make_conc_vi_vi_env(**kwargs)
    with pytest.raises(ValueError):
        gym.make(CONC_ID, **kwargs)


def test_operanda_cannot_be_overridden() -> None:
    with pytest.raises(TypeError, match="fixed operanda"):
        make_conc_vi_vi_env(operanda=("a", "b"))


# --- a tabular Q-learner gets more reinforcers than a random policy ------

_CAP = 4  # responses per operandum since the last reinforcer, capped
_TIME_BIN = 5.0  # width of a time-since-reinforcer bin
_N_TIME_BINS = 12


def _state(obs: dict[str, Any]) -> tuple[int, int, int, int]:
    counts = obs["responses_since_sr"]
    time_bin = int(obs["time_since_sr"][0] // _TIME_BIN)
    return (
        int(obs["last_operandum"]),
        min(int(counts[0]), _CAP),
        min(int(counts[1]), _CAP),
        min(time_bin, _N_TIME_BINS - 1),
    )


def _greedy(q_values: Any, rng: Any) -> int:
    # Break ties at random: with ties going to action 0 ("no response"),
    # an unvisited state would keep the agent silent for the rest of the session.
    best = np.flatnonzero(q_values == q_values.max())
    return int(rng.choice(best))


def _train_q(
    env: Any,
    episodes: int,
    seed: int,
    *,
    alpha: float = 0.1,
    gamma: float = 0.95,
    epsilon: float = 0.1,
) -> Any:
    rng = np.random.default_rng(seed)
    n_actions = int(env.action_space.n)
    q = np.zeros((n_actions, _CAP + 1, _CAP + 1, _N_TIME_BINS, n_actions))
    for episode in range(episodes):
        obs, _ = env.reset(seed=episode)
        state = _state(obs)
        while True:
            if rng.random() < epsilon:
                action = int(rng.integers(n_actions))
            else:
                action = _greedy(q[state], rng)
            obs, reward, terminated, truncated, _ = env.step(action)
            nxt = _state(obs)
            target = reward + (0.0 if terminated else gamma * q[nxt].max())
            q[state + (action,)] += alpha * (target - q[state + (action,)])
            state = nxt
            if terminated or truncated:
                break
    return q


_EVAL_SEEDS = range(1000, 1010)  # sessions not seen in training
_MARGIN = 1.5  # reinforcers per session


def _beats_baselines(env: Any, q: Any) -> bool:
    """Whether the greedy policy of ``q`` beats both random baselines by ``_MARGIN``.

    The baselines are a uniform random policy and random choice between the
    two operanda (always responding), averaged over the same sessions.
    """
    pick_rng = np.random.default_rng(1)
    random_rng = np.random.default_rng(2)

    def learned(obs: dict[str, Any]) -> int:
        return _greedy(q[_state(obs)], pick_rng)

    def uniform(_: dict[str, Any]) -> int:
        return int(random_rng.integers(3))

    def respond_anywhere(_: dict[str, Any]) -> int:
        return int(random_rng.integers(1, 3))

    q_mean = np.mean([_count_reinforcers(env, s, learned) for s in _EVAL_SEEDS])
    random_mean = np.mean([_count_reinforcers(env, s, uniform) for s in _EVAL_SEEDS])
    respond_mean = np.mean([_count_reinforcers(env, s, respond_anywhere) for s in _EVAL_SEEDS])
    print(f"Q={q_mean:.1f} random={random_mean:.1f} respond-anywhere={respond_mean:.1f}")
    return bool(q_mean >= max(random_mean, respond_mean) + _MARGIN)


def test_q_learning_beats_random_policy_on_conc_vi_vi() -> None:
    env = gym.make(CONC_ID)  # Conc VI 30 VI 60, COD 2, 600 time units
    q = _train_q(env, episodes=100, seed=0)
    assert _beats_baselines(env, q)


def test_untrained_q_table_does_not_beat_random_policy() -> None:
    # Control: an all-zero table makes the greedy policy uniform random
    # (ties broken at random), so the check above must fail without learning.
    env = gym.make(CONC_ID)
    q = _train_q(env, episodes=0, seed=0)
    assert not _beats_baselines(env, q)
