"""A sequence of concurrent schedules, each in force for a fixed time.

:class:`PhasedConcurrent` runs the phases of a multi-phase procedure in
one session: phase 0 is in force from time 0 to its duration, phase 1
for its own duration after that, and so on. The subject receives no
stimulus change at a boundary; only the contingencies change. This is
the arrangement of a resurgence procedure (Leitenberg et al., 1970;
Craig & Shahan, 2016): phase 1 reinforces the target response, phase 2
extinguishes it while reinforcing an alternative response, and phase 3
extinguishes both.

Each phase's :class:`~contingency.schedules.concurrent.Concurrent` is
timed from the onset of that phase. Interval schedules count from time
0 of their own clock, so a VI 15 s that takes effect 18000 s into the
session sets up its first reinforcer about 15 s after the boundary,
not at once. Reinforcers are reported at session time.

References
----------
Craig, A. R., & Shahan, T. A. (2016). Behavioral momentum theory fails
to account for the effects of reinforcement rate on resurgence.
*Journal of the Experimental Analysis of Behavior*, 105(3), 375-392.
https://doi.org/10.1002/jeab.207

Leitenberg, H., Rawson, R. A., & Bath, K. (1970). Reinforcement of
competing behavior during extinction. *Science*, 169(3942), 301-303.
https://doi.org/10.1126/science.169.3942.301
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace

from contingency.entities import Outcome, ResponseEvent
from contingency.errors import ScheduleConfigError, ScheduleStateError
from contingency.interfaces import Schedule
from contingency.schedules.concurrent import Concurrent

__all__ = ["PhasedConcurrent"]

_TIME_TOL = 1e-9


class PhasedConcurrent:
    """Concurrent schedules in force one after another.

    Parameters
    ----------
    phases:
        ``(schedule, duration)`` pairs in order. Every schedule is a
        :class:`Concurrent` over the same operanda; ``EXT()`` components
        extinguish an operandum. Each ``duration`` is a finite real
        ``> 0`` in the caller's time unit.

    Notes
    -----
    A step at time ``now`` belongs to phase ``k`` while
    ``now <= boundaries[k]``; the first step after a boundary belongs to
    the next phase. The last phase stays in force after its duration
    has elapsed. Every outcome carries ``meta["phase"]``, the 0-based
    index of the phase in force at that step. Changeover state does not
    carry over a boundary: each phase's ``Concurrent`` sees only the
    responses made during that phase.
    """

    __slots__ = ("_phases", "_starts", "_boundaries", "_index", "_last_now")

    def __init__(self, phases: Sequence[tuple[Concurrent, float]]) -> None:
        if len(phases) == 0:
            raise ScheduleConfigError("PhasedConcurrent requires at least one phase")
        operanda: list[str] | None = None
        schedules: list[Concurrent] = []
        starts: list[float] = []
        boundaries: list[float] = []
        elapsed = 0.0
        for i, (schedule, duration) in enumerate(phases):
            if not isinstance(schedule, Concurrent):
                raise ScheduleConfigError(
                    f"phase {i}: expected a Concurrent schedule, got {type(schedule).__name__}"
                )
            if (
                isinstance(duration, bool)
                or not isinstance(duration, int | float)
                or not math.isfinite(duration)
                or duration <= 0
            ):
                raise ScheduleConfigError(
                    f"phase {i}: duration must be a finite real > 0, got {duration!r}"
                )
            keys = list(schedule.components)
            if operanda is None:
                operanda = keys
            elif set(keys) != set(operanda):
                raise ScheduleConfigError(
                    f"phase {i}: operanda {sorted(keys)!r} differ from phase 0 {sorted(operanda)!r}"
                )
            schedules.append(schedule)
            starts.append(elapsed)
            elapsed += float(duration)
            boundaries.append(elapsed)

        self._phases: tuple[Concurrent, ...] = tuple(schedules)
        self._starts: tuple[float, ...] = tuple(starts)
        self._boundaries: tuple[float, ...] = tuple(boundaries)
        self._index = 0
        self._last_now: float | None = None

    @property
    def components(self) -> dict[str, Schedule]:
        """The components of the phase in force (same operanda in every phase)."""
        return self._phases[self._index].components

    @property
    def phases(self) -> tuple[Concurrent, ...]:
        """The schedules of the phases, in order."""
        return self._phases

    @property
    def n_phases(self) -> int:
        """Number of phases."""
        return len(self._phases)

    @property
    def boundaries(self) -> tuple[float, ...]:
        """Session time at which each phase ends."""
        return self._boundaries

    @property
    def duration(self) -> float:
        """Total duration of all phases."""
        return self._boundaries[-1]

    @property
    def phase(self) -> int:
        """0-based index of the phase in force at the last step."""
        return self._index

    def step(self, now: float, event: ResponseEvent | None = None) -> Outcome:
        if self._last_now is not None and now < self._last_now - _TIME_TOL:
            raise ScheduleStateError(
                f"Non-monotonic time: new now={now!r} precedes previous now={self._last_now!r}"
            )
        if event is not None and abs(event.time - now) > _TIME_TOL:
            raise ScheduleStateError(
                f"event.time ({event.time!r}) must equal now ({now!r}) within {_TIME_TOL}"
            )
        self._last_now = now

        last = len(self._phases) - 1
        while self._index < last and now > self._boundaries[self._index] + _TIME_TOL:
            self._index += 1

        start = self._starts[self._index]
        local = max(now - start, 0.0)
        local_event = None if event is None else replace(event, time=local)
        out = self._phases[self._index].step(local, local_event)

        reinforcer = out.reinforcer
        if reinforcer is not None:
            reinforcer = replace(reinforcer, time=reinforcer.time + start)
        return Outcome(
            reinforced=out.reinforced,
            reinforcer=reinforcer,
            meta={**out.meta, "phase": self._index},
        )

    def reset(self) -> None:
        for schedule in self._phases:
            schedule.reset()
        self._index = 0
        self._last_now = None
