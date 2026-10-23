"""Tabular Q-learning helpers (``contingency.envs.tabular``)."""

from __future__ import annotations

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

import contingency.envs  # noqa: E402,F401  (registers the environment IDs)
from contingency.envs.tabular import (  # noqa: E402
    FI_N_STATES,
    fi_state,
    greedy,
    run_greedy,
    train_q,
)

FI_ID = "contingency/FI-v0"


def _median_prp(times: list[float], actions: list[int], last_sr: list[int]) -> float:
    """Median time from a reinforcer to the next response (first interval excluded)."""
    sr_times = [t for t, sr in zip(times, last_sr, strict=True) if sr]
    resp_times = [t for t, a in zip(times, actions, strict=True) if a != 0]
    pauses = []
    for start in sr_times:
        later = [t for t in resp_times if t > start]
        if later:
            pauses.append(later[0] - start)
    return float(np.median(pauses))


def test_fi_state_bins_responses_and_time() -> None:
    obs = {
        "time_since_sr": np.array([12.0], dtype=np.float32),
        "responses_since_sr": np.array([7], dtype=np.int32),
        "last_sr": 0,
        "last_operandum": 1,
    }
    assert fi_state(obs) == (4, 2)
    obs["time_since_sr"] = np.array([599.0], dtype=np.float32)
    assert fi_state(obs) == (4, 11)


def test_greedy_breaks_ties_at_random() -> None:
    rng = np.random.default_rng(0)
    picks = {greedy(np.zeros(2), rng) for _ in range(50)}
    assert picks == {0, 1}
    assert greedy(np.array([0.0, 1.0]), rng) == 1


def test_q_learner_responds_right_after_reinforcement_without_cost() -> None:
    env = gym.make(FI_ID, interval=30.0, max_time=600.0)
    q = train_q(env, episodes=100, seed=0)
    assert q.shape == (*FI_N_STATES, 2)
    times, actions, last_sr = run_greedy(env, q, seed=1000)
    assert sum(last_sr) >= 18  # FI 30 in 600 steps: at most 20 reinforcers
    assert _median_prp(times, actions, last_sr) == pytest.approx(1.0)


def test_response_cost_makes_the_trained_policy_respond_less() -> None:
    # Control: before training, ties are broken at random, so the greedy
    # policy responds on about half the steps whatever the cost.
    env = gym.make(FI_ID, interval=30.0, max_time=600.0, response_cost=0.1)
    _, untrained, _ = run_greedy(env, train_q(env, episodes=0, seed=0), seed=1000)
    _, trained, last_sr = run_greedy(env, train_q(env, episodes=100, seed=0), seed=1000)
    assert sum(untrained) > 0.3 * len(untrained)
    assert sum(trained) < 0.3 * sum(untrained)
    assert sum(last_sr) >= 17  # still collects nearly every reinforcer


def test_custom_state_function_sets_the_table_shape() -> None:
    def time_only(obs: dict[str, object]) -> tuple[int]:
        return fi_state(obs)[1:]

    env = gym.make(FI_ID, interval=30.0, max_time=60.0)
    q = train_q(env, episodes=2, seed=0, state=time_only, n_states=(12,))
    assert q.shape == (12, 2)
    times, actions, last_sr = run_greedy(env, q, seed=0, state=time_only)
    assert len(times) == len(actions) == len(last_sr) == 60


def test_negative_episodes_raises() -> None:
    env = gym.make(FI_ID)
    with pytest.raises(ValueError):
        train_q(env, episodes=-1)
