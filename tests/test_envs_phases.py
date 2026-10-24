"""PhasedConcurrent and the ConcPhases-v0 Gymnasium environment (extra ``gym``)."""

from __future__ import annotations

import warnings

import pytest

from contingency.entities import ResponseEvent
from contingency.errors import ScheduleConfigError, ScheduleStateError
from contingency.schedules import EXT, FI, Concurrent
from contingency.schedules.phased import PhasedConcurrent

# --- PhasedConcurrent -------------------------------------------------------


def _fi_then_swap() -> PhasedConcurrent:
    """Phase 0: left FI 10, right EXT for 100 s; phase 1: the reverse for 100 s."""
    return PhasedConcurrent(
        [
            (Concurrent({"left": FI(10), "right": EXT()}), 100.0),
            (Concurrent({"left": EXT(), "right": FI(10)}), 100.0),
        ]
    )


def _respond(schedule: PhasedConcurrent, now: float, operandum: str):  # type: ignore[no-untyped-def]
    return schedule.step(now, ResponseEvent(time=now, operandum=operandum))


def test_phase_index_is_reported_in_meta_on_every_step() -> None:
    s = _fi_then_swap()
    assert s.step(1.0).meta["phase"] == 0
    assert s.step(100.0).meta["phase"] == 0
    assert s.step(100.5).meta["phase"] == 1
    assert _respond(s, 150.0, "left").meta["phase"] == 1


def test_each_phase_times_its_schedule_from_the_phase_onset() -> None:
    s = _fi_then_swap()
    # Right is EXT in phase 0 and FI 10 from t = 100 on: 5 s into phase 1
    # the interval has not elapsed, although 105 s have since t = 0.
    for t in range(1, 105):
        s.step(float(t))
    assert not _respond(s, 105.0, "right").reinforced
    out = _respond(s, 110.0, "right")
    assert out.reinforced
    assert out.reinforcer is not None
    assert out.reinforcer.time == 110.0  # session time, not phase time


def test_left_is_extinguished_after_the_boundary() -> None:
    s = _fi_then_swap()
    assert _respond(s, 10.0, "left").reinforced
    assert not _respond(s, 120.0, "left").reinforced
    assert not _respond(s, 190.0, "left").reinforced


def test_last_phase_continues_past_its_duration() -> None:
    s = _fi_then_swap()
    assert s.step(250.0).meta["phase"] == 1
    assert s.duration == 200.0


def test_components_are_the_operanda_of_the_phases() -> None:
    s = _fi_then_swap()
    assert set(s.components) == {"left", "right"}
    assert s.n_phases == 2
    assert s.boundaries == (100.0, 200.0)


def test_reset_returns_to_phase_zero() -> None:
    s = _fi_then_swap()
    _respond(s, 10.0, "left")
    s.step(150.0)
    s.reset()
    assert s.step(1.0).meta["phase"] == 0
    assert _respond(s, 10.0, "left").reinforced


def test_time_must_not_move_backwards() -> None:
    s = _fi_then_swap()
    s.step(150.0)
    with pytest.raises(ScheduleStateError):
        s.step(50.0)


@pytest.mark.parametrize(
    "phases",
    [
        [],
        [(Concurrent({"left": EXT(), "right": EXT()}), 0.0)],
        [(Concurrent({"left": EXT(), "right": EXT()}), float("inf"))],
        [(FI(10), 10.0)],
        [
            (Concurrent({"left": EXT(), "right": EXT()}), 10.0),
            (Concurrent({"left": EXT(), "centre": EXT()}), 10.0),
        ],
    ],
)
def test_invalid_phases_are_rejected(phases: list) -> None:  # type: ignore[type-arg]
    with pytest.raises(ScheduleConfigError):
        PhasedConcurrent(phases)


# --- contingency/ConcPhases-v0 ----------------------------------------------

gym = pytest.importorskip("gymnasium")
np = pytest.importorskip("numpy")

from gymnasium.utils.env_checker import check_env  # noqa: E402

import contingency.envs  # noqa: E402,F401  (registers the environment IDs)
from contingency.envs.presets import make_conc_phases_env  # noqa: E402

PHASES_ID = "contingency/ConcPhases-v0"
SHORT = ((15.0, None, 300), (None, 15.0, 300), (None, None, 100))
LEFT, RIGHT = 1, 2


def _run(env, seed: int, action: int):  # type: ignore[no-untyped-def]
    """Respond with ``action`` on every step; return (phase, reward) per step."""
    _, info = env.reset(seed=seed)
    assert info["phase"] == 0 and info["phase_step"] == 0
    trace = []
    while True:
        _, reward, terminated, truncated, info = env.step(action)
        trace.append((info["phase"], reward))
        if terminated or truncated:
            return trace


def test_registered_phases_env_passes_check_env_without_warnings() -> None:
    env = gym.make(PHASES_ID, phases=SHORT)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_env(env.unwrapped, skip_render_check=True)


@pytest.mark.parametrize("seed", range(5))
def test_reinforcement_moves_from_left_to_right_at_the_boundary(seed: int) -> None:
    env = make_conc_phases_env(phases=SHORT)
    left = _run(env, seed, LEFT)
    right = _run(env, seed, RIGHT)
    # Left is reinforced only in phase 0, right only in phase 1.
    assert {p for p, r in left if r > 0} == {0}
    assert {p for p, r in right if r > 0} == {1}


def test_info_reports_phase_and_step_within_phase() -> None:
    env = make_conc_phases_env(phases=((None, None, 3), (None, None, 2), (None, None, 2)))
    env.reset(seed=0)
    seen = []
    for _ in range(7):
        _, _, _, _, info = env.step(0)
        seen.append((info["phase"], info["phase_step"]))
    assert seen == [(0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (2, 1), (2, 2)]


def test_default_procedure_terminates_after_33000_steps() -> None:
    env = gym.make(PHASES_ID)
    env.reset(seed=0)
    steps = 0
    while True:
        _, _, terminated, truncated, info = env.step(0)
        steps += 1
        if terminated or truncated:
            break
    assert steps == 33000
    assert terminated and not truncated
    assert info["phase"] == 2


@pytest.mark.parametrize("seed", range(3))
def test_responding_every_step_in_extinction_earns_nothing(seed: int) -> None:
    env = make_conc_phases_env(phases=SHORT)
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)
    reward_in_phase_2 = 0.0
    while True:
        _, reward, terminated, truncated, info = env.step(int(rng.integers(1, 3)))
        if info["phase"] == 2:
            reward_in_phase_2 += reward
        if terminated or truncated:
            break
    assert reward_in_phase_2 == 0.0


def test_same_seed_gives_the_same_session() -> None:
    env = make_conc_phases_env(phases=SHORT)
    assert _run(env, 7, LEFT) == _run(env, 7, LEFT)


def test_max_time_and_operanda_cannot_be_passed() -> None:
    with pytest.raises(TypeError):
        make_conc_phases_env(max_time=100.0)
    with pytest.raises(TypeError):
        make_conc_phases_env(operanda=("a", "b"))


@pytest.mark.parametrize(
    "phases",
    [(), ((15.0, None, 0),), ((15.0, None, 1.5),), ((-1.0, None, 10),)],
)
def test_invalid_phase_tuples_are_rejected(phases: tuple) -> None:  # type: ignore[type-arg]
    with pytest.raises((ValueError, ScheduleConfigError)):
        make_conc_phases_env(phases=phases)
