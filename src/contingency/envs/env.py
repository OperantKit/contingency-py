"""``ScheduleEnv`` — a Gymnasium environment around any :class:`Schedule`.

The agent plays the subject in a free-operant session. Each step
advances the session clock by ``dt`` and the agent either withholds
responding (action ``0``) or emits one response on ``operanda[i - 1]``
(action ``i``). The wrapped schedule decides whether that step delivers
a reinforcer; the reward is the reinforcer's magnitude, minus an
optional ``response_cost`` on every step with a response.

The observation carries only what a subject could have access to —
time and responses since the last reinforcer, whether the last step was
reinforced, and the last operandum responded on. Internal schedule
state (whether an interval has elapsed, the next ratio value) is never
exposed.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Sequence
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from contingency.entities import ResponseEvent
from contingency.interfaces import Schedule

__all__ = ["ScheduleEnv", "ScheduleFactory"]

ScheduleFactory = Callable[[int], Schedule]
"""Builds a fresh schedule from an integer seed (for VI / VR sequences)."""

_TIME_TOL = 1e-9
_SEED_HIGH = 2**31 - 1


class ScheduleEnv(gym.Env[dict[str, Any], np.int64]):
    """Free-operant session under a reinforcement schedule.

    Parameters
    ----------
    schedule:
        Either a :class:`Schedule` instance (deep-copied and reset on
        every :meth:`reset`) or a factory ``seed -> Schedule`` that is
        called on every :meth:`reset` with a seed drawn from
        ``self.np_random``. Pass schedules that draw random values
        (VI, VR, RI, RR, ...) as a factory: an instance carries its own
        random state, so ``reset(seed=...)`` would not control it.
    max_time:
        Session length in the schedule's time unit. The episode is
        truncated after the last step whose time does not exceed it
        (``floor(max_time / dt)`` steps).
    dt:
        Clock advance per step. At most one response is emitted per step.
        Schedule times are effectively rounded up to a multiple of
        ``dt`` (under FI 2.5 with ``dt=1`` the first reinforcer is at 3).
    operanda:
        Operandum identifiers; action ``i`` responds on ``operanda[i - 1]``.
    max_reinforcers:
        If given, the episode terminates once this many reinforcers
        have been delivered.
    response_cost:
        Subtracted from the reward of every step with a response
        (action ``!= 0``), reinforced or not; withholding costs nothing.
        A reinforced response earns ``magnitude - response_cost``.
        Must be a finite real ``>= 0``; the default ``0`` leaves the reward
        unchanged. This is an effort cost on the reward signal, not the
        :class:`~contingency.schedules.ResponseCost` schedule (a token
        balance that a wrapped schedule debits).
    render_mode:
        ``None`` or ``"ansi"``.

    Spaces
    ------
    action:
        ``Discrete(len(operanda) + 1)``; ``0`` = no response.
    observation:
        ``Dict`` with ``time_since_sr`` (``Box(0, max_time, (1,), float32)``),
        ``responses_since_sr`` (``Box(0, max_steps, (k,), int32)``),
        ``last_sr`` (``Discrete(2)``) and ``last_operandum``
        (``Discrete(k + 1)``; ``0`` = no response on the last step).
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(
        self,
        schedule: Schedule | ScheduleFactory,
        *,
        max_time: float,
        dt: float = 1.0,
        operanda: Sequence[str] = ("main",),
        max_reinforcers: int | None = None,
        response_cost: float = 0.0,
        render_mode: str | None = None,
    ) -> None:
        if not max_time > 0:
            raise ValueError(f"max_time must be > 0, got {max_time!r}")
        if not dt > 0:
            raise ValueError(f"dt must be > 0, got {dt!r}")
        max_steps = math.floor(max_time / dt + _TIME_TOL)
        if max_steps < 1:
            raise ValueError(f"max_time ({max_time!r}) must be >= dt ({dt!r})")
        if len(operanda) == 0:
            raise ValueError("operanda must name at least one operandum")
        if max_reinforcers is not None and (
            not isinstance(max_reinforcers, int)
            or isinstance(max_reinforcers, bool)
            or max_reinforcers < 1
        ):
            raise ValueError(f"max_reinforcers must be an int >= 1, got {max_reinforcers!r}")
        if (
            not isinstance(response_cost, int | float)
            or isinstance(response_cost, bool)
            or not math.isfinite(response_cost)
            or response_cost < 0
        ):
            raise ValueError(f"response_cost must be a finite real >= 0, got {response_cost!r}")
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode {render_mode!r}")
        if isinstance(schedule, type):
            raise TypeError(
                f"pass a Schedule instance or a seed -> Schedule factory, "
                f"not the class {schedule.__name__}"
            )

        self._source = schedule
        self.max_time = float(max_time)
        self.dt = float(dt)
        self.operanda = tuple(operanda)
        self.max_reinforcers = max_reinforcers
        self.response_cost = float(response_cost) + 0.0  # -0.0 -> 0.0
        self.render_mode = render_mode
        self.max_steps = max_steps

        k = len(self.operanda)
        self.action_space = spaces.Discrete(k + 1)
        self.observation_space = spaces.Dict(
            {
                "time_since_sr": spaces.Box(0.0, self.max_time, shape=(1,), dtype=np.float32),
                "responses_since_sr": spaces.Box(0, self.max_steps, shape=(k,), dtype=np.int32),
                "last_sr": spaces.Discrete(2),
                "last_operandum": spaces.Discrete(k + 1),
            }
        )

        self._schedule: Schedule | None = None
        self._steps = 0
        self._last_sr_time = 0.0
        self._responses = np.zeros(k, dtype=np.int32)
        self._last_sr = 0
        self._last_operandum = 0
        self._n_sr = 0
        self._done = False

    # --- Gymnasium API -----------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed)
        self._schedule = self._build_schedule()
        self._steps = 0
        self._last_sr_time = 0.0
        self._responses = np.zeros(len(self.operanda), dtype=np.int32)
        self._last_sr = 0
        self._last_operandum = 0
        self._n_sr = 0
        self._done = False
        return self._observation(), self._info({})

    def step(
        self, action: np.int64 | int
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if self._schedule is None:
            raise RuntimeError("call reset() before step()")
        if self._done:
            raise RuntimeError("the episode has ended; call reset() before step()")
        if (
            not isinstance(action, int | np.integer)
            or isinstance(action, bool)
            or not self.action_space.contains(action)
        ):
            raise ValueError(f"invalid action {action!r}")
        a = int(action)

        self._steps += 1
        now = self._now
        event = ResponseEvent(time=now, operandum=self.operanda[a - 1]) if a else None
        outcome = self._schedule.step(now, event)

        if a:
            self._responses[a - 1] += 1
        self._last_operandum = a

        reward = 0.0
        if outcome.reinforced and outcome.reinforcer is not None:
            reward = float(outcome.reinforcer.magnitude)
            self._n_sr += 1
            self._last_sr = 1
            self._last_sr_time = now
            self._responses[:] = 0
        else:
            self._last_sr = 0
        if a:
            reward -= self.response_cost

        terminated = self.max_reinforcers is not None and self._n_sr >= self.max_reinforcers
        truncated = not terminated and self._steps >= self.max_steps
        self._done = terminated or truncated
        return self._observation(), reward, terminated, truncated, self._info(outcome.meta)

    def render(self) -> str | None:
        if self.render_mode == "ansi":
            return f"t={self._now:g} n_sr={self._n_sr}"
        return None

    # --- internals ---------------------------------------------------------

    @property
    def _now(self) -> float:
        return self._steps * self.dt

    def _build_schedule(self) -> Schedule:
        if isinstance(self._source, Schedule):
            schedule = copy.deepcopy(self._source)
            schedule.reset()
            return schedule
        return self._source(int(self.np_random.integers(_SEED_HIGH)))

    def _observation(self) -> dict[str, Any]:
        elapsed = min(self._now - self._last_sr_time, self.max_time)
        return {
            "time_since_sr": np.array([elapsed], dtype=np.float32),
            "responses_since_sr": self._responses.copy(),
            "last_sr": self._last_sr,
            "last_operandum": self._last_operandum,
        }

    def _info(self, meta: dict[str, object]) -> dict[str, Any]:
        return {"time": self._now, "n_sr": self._n_sr, "outcome_meta": dict(meta)}
