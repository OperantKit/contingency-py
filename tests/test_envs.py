"""Gymnasium environments wrapping reinforcement schedules (extra ``gym``)."""

from __future__ import annotations

import warnings
from typing import Any

import pytest

gym = pytest.importorskip("gymnasium")

from gymnasium.utils.env_checker import check_env  # noqa: E402

import contingency.envs  # noqa: E402,F401  (registers the environment IDs)
from contingency.entities import Outcome, Reinforcer, ResponseEvent  # noqa: E402
from contingency.envs import ScheduleEnv  # noqa: E402
from contingency.envs.presets import fi, fr  # noqa: E402
from contingency.schedules import FI, FR, VI  # noqa: E402

ENV_IDS = ("contingency/FR-v0", "contingency/FI-v0")


def _run_random(env: Any, seed: int) -> tuple[int, float]:
    """Run a uniformly random policy to the end of the session."""
    env.reset(seed=seed)
    env.action_space.seed(seed)
    n_sr = 0
    total = 0.0
    while True:
        _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        if reward > 0:
            n_sr += 1
        total += float(reward)
        if terminated or truncated:
            return n_sr, total


@pytest.mark.parametrize("schedule", [fr(5), fi(30.0)], ids=["FR5", "FI30"])
def test_check_env_passes_without_warnings(schedule: object) -> None:
    env = ScheduleEnv(schedule, max_time=600.0)  # type: ignore[arg-type]
    # Any warning (e.g. an infinite Box bound) fails the test.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_env(env, skip_render_check=True)


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_registered_ids_make_and_pass_check_env(env_id: str) -> None:
    env = gym.make(env_id, max_time=600.0)
    check_env(env.unwrapped, skip_render_check=True)


@pytest.mark.parametrize(
    ("env_id", "kwargs"),
    [("contingency/FR-v0", {"n": 5}), ("contingency/FI-v0", {"interval": 30.0})],
)
def test_random_policy_obtains_reinforcement(env_id: str, kwargs: dict[str, float]) -> None:
    env = gym.make(env_id, max_time=600.0, **kwargs)
    n_sr, total = _run_random(env, seed=0)
    assert n_sr > 0
    assert total == pytest.approx(n_sr)  # magnitude 1.0 per reinforcer, no response cost


def test_fr_random_policy_count_matches_responses() -> None:
    """Under FR n every n-th response is reinforced, so n_sr == responses // n."""
    env = ScheduleEnv(FR(5), max_time=600.0)
    env.reset(seed=1)
    env.action_space.seed(1)
    responses = 0
    n_sr = 0
    truncated = False
    while not truncated:
        action = env.action_space.sample()
        _, reward, _, truncated, _ = env.step(action)
        responses += int(action != 0)
        n_sr += int(reward > 0)
    assert n_sr == responses // 5


def test_fi_reinforces_only_first_response_after_interval() -> None:
    env = ScheduleEnv(FI(3.0), max_time=100.0)
    env.reset(seed=0)
    rewards = [env.step(1)[1] for _ in range(7)]
    # now = 1..7; FI 3 arms at 3, re-arms at 3 + 3 = 6
    assert rewards == [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]


def test_no_response_is_never_reinforced() -> None:
    env = ScheduleEnv(FI(3.0), max_time=50.0)
    env.reset(seed=0)
    truncated = False
    while not truncated:
        _, reward, terminated, truncated, _ = env.step(0)
        assert reward == 0.0
        assert not terminated


def test_observation_tracks_time_and_responses_since_reinforcement() -> None:
    env = ScheduleEnv(FR(3), max_time=100.0, dt=0.5)
    obs, info = env.reset(seed=0)
    assert obs["time_since_sr"][0] == 0.0
    assert obs["responses_since_sr"][0] == 0
    assert obs["last_sr"] == 0
    assert info["time"] == 0.0
    obs, _, _, _, info = env.step(1)
    obs, _, _, _, info = env.step(0)
    assert obs["time_since_sr"][0] == pytest.approx(1.0)
    assert obs["responses_since_sr"][0] == 1
    assert obs["last_operandum"] == 0
    env.step(1)
    obs, reward, _, _, info = env.step(1)
    assert reward == 1.0
    assert obs["last_sr"] == 1
    assert obs["time_since_sr"][0] == 0.0
    assert obs["responses_since_sr"][0] == 0
    assert obs["last_operandum"] == 1
    assert info["n_sr"] == 1
    assert info["time"] == pytest.approx(2.0)


def test_truncates_at_session_time() -> None:
    env = ScheduleEnv(FR(5), max_time=10.0)
    env.reset(seed=0)
    steps = 0
    while True:
        _, _, terminated, truncated, _ = env.step(0)
        steps += 1
        if terminated or truncated:
            break
    assert steps == 10
    assert truncated and not terminated


def test_terminates_at_max_reinforcers() -> None:
    env = ScheduleEnv(FR(1), max_time=100.0, max_reinforcers=3)
    env.reset(seed=0)
    results = [env.step(1) for _ in range(3)]
    assert [r[2] for r in results] == [False, False, True]
    assert not results[-1][3]


def test_reset_restores_schedule_state() -> None:
    env = ScheduleEnv(FR(3), max_time=100.0)
    env.reset(seed=0)
    env.step(1)
    env.step(1)
    env.reset(seed=0)
    rewards = [env.step(1)[1] for _ in range(3)]
    assert rewards == [0.0, 0.0, 1.0]


def test_used_schedule_instance_starts_fresh_and_is_not_mutated() -> None:
    schedule = FR(3)
    schedule.step(1.0, ResponseEvent(time=1.0))
    schedule.step(2.0, ResponseEvent(time=2.0))  # one response short of SR
    env = ScheduleEnv(schedule, max_time=100.0)
    env.reset(seed=0)
    assert [env.step(1)[1] for _ in range(3)] == [0.0, 0.0, 1.0]
    # The caller's object still needs exactly one more response.
    assert schedule.step(3.0, ResponseEvent(time=3.0)).reinforced


def test_factory_seed_comes_from_reset_seed() -> None:
    seen: list[int] = []

    def factory(seed: int) -> FR:
        seen.append(seed)
        return FR(2)

    env = ScheduleEnv(factory, max_time=10.0)
    env.reset(seed=7)
    env.reset(seed=7)
    env.reset(seed=8)
    assert seen[0] == seen[1]
    assert seen[0] != seen[2]


def _vi_rewards(env: ScheduleEnv, seed: int) -> list[float]:
    env.reset(seed=seed)
    return [env.step(1)[1] for _ in range(200)]


def test_random_schedule_via_factory_is_reproducible_per_seed() -> None:
    env = ScheduleEnv(lambda seed: VI(5.0, seed=seed), max_time=200.0)
    assert _vi_rewards(env, 0) == _vi_rewards(env, 0)
    assert _vi_rewards(env, 0) != _vi_rewards(env, 1)


def test_fi_interval_restarts_at_reinforcement_not_at_arming() -> None:
    env = ScheduleEnv(FI(3.0), max_time=20.0)
    env.reset(seed=0)
    respond_at = {5, 6, 7, 8}
    reinforced_at = [t for t in range(1, 10) if env.step(int(t in respond_at))[1] > 0]
    # Armed at 3, first response at 5 -> SR; next interval runs 5..8.
    assert reinforced_at == [5, 8]


def test_reward_is_reinforcer_magnitude() -> None:
    class Big:
        def step(self, now: float, event: ResponseEvent | None = None) -> Outcome:
            if event is None:
                return Outcome()
            return Outcome(reinforced=True, reinforcer=Reinforcer(time=now, magnitude=2.5))

        def reset(self) -> None:
            pass

    env = ScheduleEnv(Big(), max_time=5.0)
    env.reset(seed=0)
    assert env.step(1)[1] == 2.5
    assert env.step(0)[1] == 0.0


def test_session_never_runs_past_max_time_when_dt_does_not_divide_it() -> None:
    env = ScheduleEnv(FR(1), max_time=10.0, dt=3.0)
    env.reset(seed=0)
    times = []
    truncated = False
    while not truncated:
        _, _, _, truncated, info = env.step(1)
        times.append(info["time"])
    assert times == [3.0, 6.0, 9.0]


def test_step_after_episode_end_raises() -> None:
    env = ScheduleEnv(FR(1), max_time=2.0)
    env.reset(seed=0)
    env.step(0)
    env.step(0)
    with pytest.raises(RuntimeError):
        env.step(0)


@pytest.mark.parametrize("action", [2, -1, 1.7, True])
def test_invalid_action_raises(action: object) -> None:
    env = ScheduleEnv(FR(1), max_time=5.0)
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(action)  # type: ignore[arg-type]


def test_two_operanda_route_actions() -> None:
    env = ScheduleEnv(FR(100), max_time=10.0, operanda=("left", "right"))
    assert env.action_space == gym.spaces.Discrete(3)
    env.reset(seed=0)
    env.step(2)
    obs, *_ = env.step(2)
    obs, *_ = env.step(1)
    assert obs["responses_since_sr"].tolist() == [1, 2]
    assert obs["last_operandum"] == 1


def test_kwargs_override_registered_defaults() -> None:
    env = gym.make("contingency/FR-v0", n=2, max_time=20.0)
    env.reset(seed=0)
    rewards = [env.step(1)[1] for _ in range(4)]
    assert rewards == [0.0, 1.0, 0.0, 1.0]


def test_render_ansi() -> None:
    env = ScheduleEnv(FR(1), max_time=10.0, render_mode="ansi")
    env.reset(seed=0)
    env.step(1)
    text = env.render()
    assert text is not None
    assert "n_sr=1" in text


def test_render_through_make_does_not_warn() -> None:
    env = gym.make("contingency/FR-v0", n=1, render_mode="ansi")
    env.reset(seed=0)
    env.step(1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert env.render() is not None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_time": 0.0},
        {"max_time": 10.0, "dt": 0.0},
        {"max_time": 1.0, "dt": 2.0},
        {"max_time": 10.0, "operanda": ()},
        {"max_time": 10.0, "max_reinforcers": 0},
        {"max_time": 10.0, "max_reinforcers": 2.5},
        {"max_time": 10.0, "max_reinforcers": True},
        {"max_time": 10.0, "render_mode": "human"},
    ],
)
def test_invalid_config_raises(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ScheduleEnv(FR(1), **kwargs)  # type: ignore[arg-type]


def test_schedule_class_instead_of_instance_raises() -> None:
    with pytest.raises(TypeError):
        ScheduleEnv(FR, max_time=10.0)
