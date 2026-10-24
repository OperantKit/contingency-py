"""Aversive-control environments: Sidman avoidance and punished VI."""

from __future__ import annotations

import warnings
from typing import Any

import pytest

gym = pytest.importorskip("gymnasium")

from gymnasium.utils.env_checker import check_env  # noqa: E402

import contingency.envs  # noqa: E402,F401  (registers the environment IDs)
from contingency.entities import Outcome, Reinforcer, ResponseEvent  # noqa: E402
from contingency.envs import ScheduleEnv  # noqa: E402
from contingency.envs.presets import (  # noqa: E402
    AversiveScheduleEnv,
    fi,
    fr,
    make_punished_vi_env,
    make_sidman_env,
)

Step = tuple[dict[str, Any], float, dict[str, Any]]


def _run(env: Any, policy: Any, seed: int = 0) -> list[Step]:
    """Run ``policy(step_index) -> action`` to the end; return (obs, reward, info) per step."""
    env.reset(seed=seed)
    out = []
    i = 0
    while True:
        obs, reward, terminated, truncated, info = env.step(policy(i))
        out.append((obs, float(reward), info))
        i += 1
        if terminated or truncated:
            return out


# --- Sidman avoidance -----------------------------------------------------


def test_sidman_without_responding_shocks_every_ss() -> None:
    env = make_sidman_env(rs=20.0, ss=5.0, max_time=600.0)
    steps = _run(env, lambda i: 0)
    shock_steps = [info["time"] for _, r, info in steps if r < 0]
    assert len(shock_steps) == 120
    assert shock_steps[:3] == [5.0, 10.0, 15.0]
    assert all(r == -1.0 for _, r, info in steps if r != 0)
    final = steps[-1][2]
    assert final["n_shock"] == 120
    assert final["n_sr"] == 0


def test_sidman_response_postpones_shock_by_rs() -> None:
    env = make_sidman_env(rs=8.0, ss=5.0, max_time=30.0)
    # Respond only at t = 1 (step index 0): the next shock moves to 1 + 8 = 9.
    steps = _run(env, lambda i: 1 if i == 0 else 0)
    shock_times = [info["time"] for _, r, info in steps if r < 0]
    assert shock_times[0] == 9.0
    assert shock_times[1] == 14.0  # then S-S again


def test_sidman_time_since_response_and_shock_reset() -> None:
    env = make_sidman_env(rs=20.0, ss=5.0, max_time=30.0)
    env.reset(seed=0)
    obs, *_ = env.step(0)  # t = 1
    assert obs["time_since_response"][0] == pytest.approx(1.0)
    assert obs["time_since_shock"][0] == pytest.approx(1.0)
    obs, *_ = env.step(1)  # t = 2, response
    assert obs["time_since_response"][0] == 0.0
    assert obs["responses_since_shock"][0] == 1
    for _ in range(19):  # t = 21
        obs, *_ = env.step(0)
    assert obs["time_since_response"][0] == pytest.approx(19.0)
    obs, reward, *_ , info = env.step(0)  # t = 22 = 2 + rs -> shock
    assert reward == -1.0
    assert obs["time_since_shock"][0] == 0.0
    assert obs["responses_since_shock"][0] == 0
    assert obs["last_sr"] == 0  # a shock is not an SR+
    assert info["n_shock"] == 1


def test_sidman_responding_every_step_avoids_all_shocks() -> None:
    env = make_sidman_env(rs=20.0, ss=10.0, max_time=600.0)
    steps = _run(env, lambda i: 1)
    assert steps[-1][2]["n_shock"] == 0
    assert all(r == 0.0 for _, r, _ in steps)


def test_sidman_response_cost_is_subtracted() -> None:
    env = make_sidman_env(rs=20.0, ss=10.0, max_time=10.0, response_cost=0.01)
    steps = _run(env, lambda i: 1)
    assert all(r == pytest.approx(-0.01) for _, r, _ in steps)


# --- punished VI ----------------------------------------------------------


def test_punished_vi_reward_and_counts() -> None:
    env = make_punished_vi_env(mean=10.0, punishment=0.1, ratio=1, max_time=600.0)
    steps = _run(env, lambda i: 1)
    rewards = [r for _, r, _ in steps]
    assert {round(r, 9) for r in rewards} == {-0.1, 0.9}
    final = steps[-1][2]
    n_reinforced = sum(1 for r in rewards if r > 0)
    assert final["n_shock"] == len(steps)  # every response punished
    assert final["n_sr"] == n_reinforced  # punishers are not counted as SR+
    assert n_reinforced > 0
    # the compound step sets last_sr
    for obs, r, _ in steps:
        assert obs["last_sr"] == (1 if r > 0 else 0)


def test_punished_vi_ratio_punishes_every_kth_response() -> None:
    env = make_punished_vi_env(mean=10.0, punishment=0.1, ratio=5, max_time=600.0)
    steps = _run(env, lambda i: 1)
    assert steps[-1][2]["n_shock"] == 600 // 5


def test_punished_vi_zero_punishment_is_plain_vi() -> None:
    env = make_punished_vi_env(mean=10.0, punishment=0.0, max_time=600.0)
    steps = _run(env, lambda i: 1)
    assert steps[-1][2]["n_shock"] == 0
    assert all(r >= 0.0 for _, r, _ in steps)
    assert steps[-1][2]["n_sr"] > 0


def test_punished_vi_rejects_negative_punishment() -> None:
    with pytest.raises(ValueError):
        make_punished_vi_env(punishment=-0.1)


def test_punished_vi_seed_reproduces() -> None:
    env = make_punished_vi_env(mean=10.0, punishment=0.1, max_time=300.0)
    a = [r for _, r, _ in _run(env, lambda i: 1, seed=3)]
    b = [r for _, r, _ in _run(env, lambda i: 1, seed=3)]
    assert a == b


# --- both -----------------------------------------------------------------


@pytest.mark.parametrize("schedule", [fr(3), fi(10.0)], ids=["FR3", "FI10"])
def test_plain_schedule_matches_schedule_env(schedule: object) -> None:
    """Without shocks the aversive env counts and rewards like ScheduleEnv."""
    plain = ScheduleEnv(schedule, max_time=120.0)  # type: ignore[arg-type]
    aversive = AversiveScheduleEnv(schedule, max_time=120.0)
    policy = lambda i: 1 if i % 2 == 0 else 0  # noqa: E731
    a = _run(plain, policy)
    b = _run(aversive, policy)
    assert [r for _, r, _ in a] == [r for _, r, _ in b]
    assert a[-1][2]["n_sr"] == b[-1][2]["n_sr"] > 0
    assert b[-1][2]["n_shock"] == 0


def test_zero_magnitude_reinforcer_counts_as_sr() -> None:
    class ZeroCRF:
        def step(self, now: float, event: ResponseEvent | None = None) -> Outcome:
            if event is None:
                return Outcome()
            return Outcome(reinforced=True, reinforcer=Reinforcer(time=now, magnitude=0.0))

        def reset(self) -> None:
            pass

    env = AversiveScheduleEnv(ZeroCRF(), max_time=5.0)
    steps = _run(env, lambda i: 1)
    assert steps[-1][2]["n_sr"] == 5
    assert steps[-1][2]["n_shock"] == 0


@pytest.mark.parametrize(
    "env",
    [make_sidman_env(), make_punished_vi_env(), make_punished_vi_env(punishment=0.0)],
    ids=["sidman", "punished-vi", "vi-only"],
)
def test_aversive_envs_pass_check_env_without_warnings(env: AversiveScheduleEnv) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_env(env, skip_render_check=True)


@pytest.mark.parametrize("env_id", ["contingency/SidmanAvoidance-v0", "contingency/PunishedVI-v0"])
def test_registered_aversive_ids(env_id: str) -> None:
    env = gym.make(env_id, max_time=60.0)
    assert isinstance(env.unwrapped, AversiveScheduleEnv)
    check_env(env.unwrapped, skip_render_check=True)
