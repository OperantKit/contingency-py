"""Schedule factories and the entry points behind the registered IDs."""

from __future__ import annotations

import math
import numbers
from collections.abc import Sequence
from typing import Any

from contingency.envs.env import ScheduleEnv
from contingency.interfaces import Schedule
from contingency.schedules import EXT, FI, FR, VI, Concurrent
from contingency.schedules.phased import PhasedConcurrent

__all__ = [
    "DEFAULT_PHASES",
    "PhaseSpec",
    "PhasedScheduleEnv",
    "conc_phases",
    "conc_vi_vi",
    "fi",
    "fr",
    "make_conc_phases_env",
    "make_conc_vi_vi_env",
    "make_fi_env",
    "make_fr_env",
]

CONC_OPERANDA = ("left", "right")


def fr(n: int) -> FR:
    """Fixed-ratio ``n``."""
    return FR(n)


def fi(interval: float) -> FI:
    """Fixed-interval ``interval``."""
    return FI(interval)


def conc_vi_vi(
    mean_left: float, mean_right: float, cod: float, *, seed: int | None = None
) -> Concurrent:
    """Concurrent VI ``mean_left`` VI ``mean_right`` with changeover delay ``cod``.

    The components sit on operanda ``"left"`` and ``"right"``. With a
    ``seed``, the left VI uses ``seed`` and the right VI ``seed + 1``, so
    equal means still give the two operanda different interval sequences.
    """
    right_seed = None if seed is None else seed + 1
    return Concurrent(
        {
            "left": VI(mean_left, seed=seed),
            "right": VI(mean_right, seed=right_seed),
        },
        cod=cod,
    )


def make_fr_env(n: int = 5, *, max_time: float = 600.0, **kwargs: Any) -> ScheduleEnv:
    """Entry point of ``contingency/FR-v0``."""
    return ScheduleEnv(fr(n), max_time=max_time, **kwargs)


def make_fi_env(interval: float = 30.0, *, max_time: float = 600.0, **kwargs: Any) -> ScheduleEnv:
    """Entry point of ``contingency/FI-v0``."""
    return ScheduleEnv(fi(interval), max_time=max_time, **kwargs)


def make_conc_vi_vi_env(
    mean_left: float = 30.0,
    mean_right: float = 60.0,
    cod: float = 2.0,
    *,
    max_time: float = 600.0,
    **kwargs: Any,
) -> ScheduleEnv:
    """Entry point of ``contingency/ConcVIVI-v0``.

    Action ``1`` responds on ``"left"`` and ``2`` on ``"right"``; the
    operanda are fixed, so ``operanda`` cannot be passed. The VI sequences
    are drawn again on every ``reset()``.
    """
    if "operanda" in kwargs:
        raise TypeError("ConcVIVI uses the fixed operanda ('left', 'right'); do not pass operanda")
    # Build once so that bad parameters fail here, not at the first reset().
    conc_vi_vi(mean_left, mean_right, cod, seed=0)

    def factory(seed: int) -> Concurrent:
        return conc_vi_vi(mean_left, mean_right, cod, seed=seed)

    return ScheduleEnv(factory, max_time=max_time, operanda=CONC_OPERANDA, **kwargs)


PhaseSpec = tuple[float | None, float | None, int]
"""``(left VI mean or None for EXT, right VI mean or None for EXT, steps)``."""

DEFAULT_PHASES: tuple[PhaseSpec, ...] = (
    (15.0, None, 18000),
    (None, 15.0, 12000),
    (None, None, 3000),
)
"""Resurgence procedure: target VI 15 s; then target EXT with alternative VI 15 s; then both EXT.

The phase lengths keep the 30 : 20 : 5 session ratio of Craig & Shahan
(2016) at 600 steps (10 min at ``dt = 1``) per session.
"""


def _phase_steps(n: Any, where: str) -> int:
    if isinstance(n, bool) or not isinstance(n, numbers.Integral) or n < 1:
        raise ValueError(f"{where}: steps must be an int >= 1, got {n!r}")
    return int(n)


def _normalize_phase_specs(phases: Sequence[Sequence[Any]]) -> tuple[PhaseSpec, ...]:
    """Check ``phases`` and return them as tuples of ``float | None`` and ``int``.

    Lists (as read from JSON) and numpy scalars are accepted.
    """
    if len(phases) == 0:
        raise ValueError("phases must contain at least one phase")
    out: list[PhaseSpec] = []
    for i, spec in enumerate(phases):
        if isinstance(spec, str | bytes) or not isinstance(spec, Sequence) or len(spec) != 3:
            raise ValueError(f"phase {i}: expected (left, right, steps), got {spec!r}")
        left, right, steps = spec
        means: list[float | None] = []
        for mean in (left, right):
            if mean is None:
                means.append(None)
                continue
            if (
                isinstance(mean, bool)
                or not isinstance(mean, numbers.Real)
                or not math.isfinite(mean)
                or mean <= 0
            ):
                raise ValueError(
                    f"phase {i}: a VI mean must be a finite real > 0 or None, got {mean!r}"
                )
            means.append(float(mean))
        out.append((means[0], means[1], _phase_steps(steps, f"phase {i}")))
    return tuple(out)


def _vi_or_ext(mean: float | None, seed: int | None) -> Schedule:
    return EXT() if mean is None else VI(mean, seed=seed)


def conc_phases(
    phases: Sequence[Sequence[Any]], cod: float, *, dt: float = 1.0, seed: int | None = None
) -> PhasedConcurrent:
    """Concurrent VI / EXT phases on operanda ``"left"`` and ``"right"``.

    Each phase lasts ``steps * dt``. With a ``seed``, phase ``i`` draws its
    left VI from ``seed + 2 * i`` and its right VI from ``seed + 2 * i + 1``.
    """
    built = []
    for i, (left, right, steps) in enumerate(_normalize_phase_specs(phases)):
        left_seed = None if seed is None else seed + 2 * i
        right_seed = None if seed is None else seed + 2 * i + 1
        conc = Concurrent(
            {"left": _vi_or_ext(left, left_seed), "right": _vi_or_ext(right, right_seed)},
            cod=cod,
        )
        built.append((conc, steps * dt))
    return PhasedConcurrent(built)


class PhasedScheduleEnv(ScheduleEnv):
    """:class:`ScheduleEnv` whose session is a fixed sequence of phases.

    ``info`` also carries ``phase`` (0-based index of the phase in force)
    and ``phase_step`` (steps taken within that phase). The phase is not
    in the observation. The episode *terminates* when the last phase
    ends: the procedure is over, not cut short.
    """

    def __init__(self, schedule: Any, *, phase_steps: Sequence[int], **kwargs: Any) -> None:
        if len(phase_steps) == 0:
            raise ValueError("phase_steps must contain at least one phase")
        ends: list[int] = []
        total = 0
        for i, n in enumerate(phase_steps):
            total += _phase_steps(n, f"phase_steps[{i}]")
            ends.append(total)
        self._phase_ends = tuple(ends)
        super().__init__(schedule, **kwargs)

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)
        if truncated and self._steps >= self._phase_ends[-1]:
            terminated, truncated = True, False
        return obs, reward, terminated, truncated, info

    def _info(self, meta: dict[str, object]) -> dict[str, Any]:
        info = super()._info(meta)
        phase = 0
        while phase < len(self._phase_ends) - 1 and self._steps > self._phase_ends[phase]:
            phase += 1
        start = 0 if phase == 0 else self._phase_ends[phase - 1]
        info["phase"] = phase
        info["phase_step"] = self._steps - start
        return info


def make_conc_phases_env(
    phases: Sequence[Sequence[Any]] = DEFAULT_PHASES,
    cod: float = 2.0,
    **kwargs: Any,
) -> ScheduleEnv:
    """Entry point of ``contingency/ConcPhases-v0``.

    ``phases`` lists ``(left, right, steps)``: the VI mean on each operandum
    (``None`` = extinction) and the number of steps the phase lasts. The
    session length is the sum of the steps, so ``max_time`` cannot be
    passed; the operanda are fixed to ``("left", "right")``. Action ``1``
    responds on ``"left"`` and ``2`` on ``"right"``.
    """
    for fixed in ("max_time", "operanda"):
        if fixed in kwargs:
            raise TypeError(f"ConcPhases fixes {fixed} from phases; do not pass {fixed}")
    spec = _normalize_phase_specs(phases)
    dt = float(kwargs.get("dt", 1.0))
    # Build once so that bad parameters fail here, not at the first reset().
    conc_phases(spec, cod, dt=dt, seed=0)
    phase_steps = [steps for _, _, steps in spec]

    def factory(seed: int) -> PhasedConcurrent:
        return conc_phases(spec, cod, dt=dt, seed=seed)

    return PhasedScheduleEnv(
        factory,
        phase_steps=phase_steps,
        max_time=sum(phase_steps) * dt,
        operanda=CONC_OPERANDA,
        **kwargs,
    )
