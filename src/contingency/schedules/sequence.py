"""Sequence-family compound schedules: Multiple, Chained, and Tandem.

All three schedules compose a list of component schedules and expose the
same :class:`~contingency.interfaces.Schedule` protocol as their primitive
constituents. They differ in *what happens when the active component
reinforces*:

* :class:`Multiple` — components alternate cyclically. Each carries its
  own discriminative stimulus (S^D). When the active component produces
  a reinforcer, the primary reinforcer is delivered and the next
  component becomes active.
* :class:`Chained` — components form a chain with distinct S^Ds. Only the
  terminal component produces primary reinforcement. Completing a
  non-terminal component is a *conditioned* reinforcement event: the S^D
  changes but no :class:`~contingency.entities.Reinforcer` is delivered.
* :class:`Tandem` — the chained topology but **without** distinctive
  stimuli. The subject receives no external cue that a link has
  advanced; only the experimenter's record (the ``current_component``
  index in ``meta``) tracks link transitions.

Time-stepping inactive components
---------------------------------
Inactive components are *not* stepped. Only the active component receives
:meth:`Schedule.step` calls. This matches the operant-chamber convention
that inactive components are "off": their clocks do not advance while
another S^D is in effect. The consequence is that a newly-activated
component is implicitly "cold" — time-based schedules (FI, FT, VI, VT,
RI, RT) will anchor on their first step *after* the transition, not on
the moment the transition occurred. In practice this matters only for
time-based terminal components in a Chained/Tandem: the interval begins
when the first step after the transition is taken, exactly as in a real
operant chamber where the S^D and the clock both restart on link entry.

References
----------
Ferster, C. B., & Skinner, B. F. (1957). *Schedules of reinforcement*.
Appleton-Century-Crofts.

Kelleher, R. T., & Gollub, L. R. (1962). A review of positive
conditioned reinforcement. *Journal of the Experimental Analysis of
Behavior*, 5(4 Suppl), 543-597.
https://doi.org/10.1901/jeab.1962.5-s543

Reynolds, G. S. (1961). Behavioral contrast. *Journal of the
Experimental Analysis of Behavior*, 4(1), 57-71.
https://doi.org/10.1901/jeab.1961.4-57
"""

from __future__ import annotations

from collections.abc import Sequence

from contingency.entities import Outcome, ResponseEvent
from contingency.errors import ScheduleConfigError, ScheduleStateError
from contingency.interfaces import Schedule

__all__ = ["Multiple", "Chained", "Tandem"]

_TIME_TOL = 1e-9


def _check_time(last_now: float | None, now: float) -> None:
    """Raise :class:`ScheduleStateError` if ``now`` moves backwards."""
    if last_now is not None and now < last_now - _TIME_TOL:
        raise ScheduleStateError(
            f"Non-monotonic time: new now={now!r} precedes previous now={last_now!r}"
        )


def _check_event(now: float, event: ResponseEvent | None) -> None:
    """Raise :class:`ScheduleStateError` if ``event.time`` disagrees with ``now``."""
    if event is not None and abs(event.time - now) > _TIME_TOL:
        raise ScheduleStateError(
            f"event.time ({event.time!r}) must equal now ({now!r}) within {_TIME_TOL}"
        )


def _validate_components(components: Sequence[Schedule]) -> list[Schedule]:
    if not isinstance(components, Sequence) or isinstance(components, (str, bytes)):
        raise ScheduleConfigError(
            f"components must be a sequence of schedules, got {type(components).__name__}"
        )
    comp_list = list(components)
    if len(comp_list) < 2:
        raise ScheduleConfigError(
            f"compound schedule requires >= 2 components, got {len(comp_list)}"
        )
    for i, c in enumerate(comp_list):
        if not isinstance(c, Schedule):
            raise ScheduleConfigError(
                f"component {i} does not satisfy the Schedule protocol: {type(c).__name__}"
            )
    return comp_list


def _validate_stimuli(
    stimuli: Sequence[str] | None, n_components: int
) -> list[str]:
    if stimuli is None:
        return [f"comp_{i}" for i in range(n_components)]
    if isinstance(stimuli, (str, bytes)) or not isinstance(stimuli, Sequence):
        raise ScheduleConfigError(
            f"stimuli must be a sequence of strings, got {type(stimuli).__name__}"
        )
    names = list(stimuli)
    if len(names) != n_components:
        raise ScheduleConfigError(
            f"stimuli length ({len(names)}) must equal components length ({n_components})"
        )
    for i, s in enumerate(names):
        if not isinstance(s, str):
            raise ScheduleConfigError(
                f"stimulus {i} must be a string, got {type(s).__name__}"
            )
    if len(set(names)) != len(names):
        raise ScheduleConfigError(
            f"stimulus names must be unique, got {names!r}"
        )
    return names


class Multiple:
    """Multiple (``mult``) compound schedule.

    Two or more independent component schedules alternate in the presence
    of distinctive discriminative stimuli (S^D). Exactly one component is
    active at a time; responses are routed only to that component. When
    the active component produces a reinforcer, the primary reinforcer
    is delivered and the active index advances (wrapping to 0 after the
    last component).

    Parameters
    ----------
    components:
        Two or more schedules, each satisfying the
        :class:`~contingency.interfaces.Schedule` protocol.
    stimuli:
        Optional sequence of unique string labels naming the S^D that
        accompanies each component. Defaults to ``["comp_0", "comp_1", ...]``.

    Notes
    -----
    Inactive components are not stepped; see the module docstring.

    References
    ----------
    Ferster, C. B., & Skinner, B. F. (1957). *Schedules of reinforcement*.
    Appleton-Century-Crofts.

    Reynolds, G. S. (1961). Behavioral contrast. *Journal of the
    Experimental Analysis of Behavior*, 4(1), 57-71.
    https://doi.org/10.1901/jeab.1961.4-57
    """

    __slots__ = ("_components", "_stimuli", "_active", "_last_now")

    def __init__(
        self,
        components: Sequence[Schedule],
        stimuli: Sequence[str] | None = None,
    ) -> None:
        self._components: list[Schedule] = _validate_components(components)
        self._stimuli: list[str] = _validate_stimuli(
            stimuli, len(self._components)
        )
        self._active: int = 0
        self._last_now: float | None = None

    @property
    def n_components(self) -> int:
        return len(self._components)

    @property
    def active_index(self) -> int:
        return self._active

    @property
    def current_stimulus(self) -> str:
        return self._stimuli[self._active]

    def step(self, now: float, event: ResponseEvent | None = None) -> Outcome:
        _check_time(self._last_now, now)
        _check_event(now, event)
        self._last_now = now

        inner = self._components[self._active].step(now, event)
        stim = self._stimuli[self._active]

        if inner.reinforced:
            # Primary reinforcement; then advance (wrap) for the next step.
            self._active = (self._active + 1) % len(self._components)
            meta: dict[str, object] = dict(inner.meta)
            meta["current_component"] = stim
            return Outcome(
                reinforced=True, reinforcer=inner.reinforcer, meta=meta
            )

        meta = dict(inner.meta)
        meta["current_component"] = stim
        return Outcome(meta=meta)

    def reset(self) -> None:
        self._active = 0
        self._last_now = None
        for c in self._components:
            c.reset()


class Chained:
    """Chained (``chain``) compound schedule.

    A sequence of N components (links). Each non-terminal link carries a
    distinctive S^D; completing a non-terminal link produces a *conditioned*
    reinforcement event (the S^D changes, signalling access to the next
    link) but delivers no primary :class:`~contingency.entities.Reinforcer`.
    Only the terminal (last) link produces primary reinforcement. After
    primary reinforcement the active link resets to 0.

    Parameters
    ----------
    components:
        Two or more schedules.
    stimuli:
        Optional sequence of unique string labels. Defaults to
        ``["comp_0", "comp_1", ...]``.

    Notes
    -----
    Inactive components are not stepped; see the module docstring.

    References
    ----------
    Ferster, C. B., & Skinner, B. F. (1957). *Schedules of reinforcement*.
    Appleton-Century-Crofts.

    Kelleher, R. T., & Gollub, L. R. (1962). A review of positive
    conditioned reinforcement. *Journal of the Experimental Analysis of
    Behavior*, 5(4 Suppl), 543-597.
    https://doi.org/10.1901/jeab.1962.5-s543
    """

    __slots__ = ("_components", "_stimuli", "_active", "_last_now")

    def __init__(
        self,
        components: Sequence[Schedule],
        stimuli: Sequence[str] | None = None,
    ) -> None:
        self._components: list[Schedule] = _validate_components(components)
        self._stimuli: list[str] = _validate_stimuli(
            stimuli, len(self._components)
        )
        self._active: int = 0
        self._last_now: float | None = None

    @property
    def n_components(self) -> int:
        return len(self._components)

    @property
    def active_index(self) -> int:
        return self._active

    @property
    def current_stimulus(self) -> str:
        return self._stimuli[self._active]

    @property
    def is_terminal(self) -> bool:
        return self._active == len(self._components) - 1

    def step(self, now: float, event: ResponseEvent | None = None) -> Outcome:
        _check_time(self._last_now, now)
        _check_event(now, event)
        self._last_now = now

        inner = self._components[self._active].step(now, event)

        if inner.reinforced:
            if self.is_terminal:
                # Primary reinforcement; cycle back to link 0.
                self._active = 0
                meta: dict[str, object] = dict(inner.meta)
                meta["current_component"] = self._stimuli[self._active]
                return Outcome(
                    reinforced=True,
                    reinforcer=inner.reinforcer,
                    meta=meta,
                )
            # Non-terminal completion: conditioned reinforcement =
            # transition to next link; no primary reinforcer.
            self._active += 1
            meta = dict(inner.meta)
            meta["current_component"] = self._stimuli[self._active]
            meta["chain_transition"] = True
            return Outcome(meta=meta)

        meta = dict(inner.meta)
        meta["current_component"] = self._stimuli[self._active]
        return Outcome(meta=meta)

    def reset(self) -> None:
        self._active = 0
        self._last_now = None
        for c in self._components:
            c.reset()


class Tandem:
    """Tandem (``tand``) compound schedule.

    Structurally identical to :class:`Chained`, but without distinctive
    stimuli. Link transitions are invisible to the subject — only the
    experimenter's record (the integer ``current_component`` index in
    ``meta``) tracks the active link. Primary reinforcement occurs only
    on the terminal link.

    Parameters
    ----------
    components:
        Two or more schedules.

    Notes
    -----
    Inactive components are not stepped; see the module docstring.

    References
    ----------
    Ferster, C. B., & Skinner, B. F. (1957). *Schedules of reinforcement*.
    Appleton-Century-Crofts.

    Kelleher, R. T., & Gollub, L. R. (1962). A review of positive
    conditioned reinforcement. *Journal of the Experimental Analysis of
    Behavior*, 5(4 Suppl), 543-597.
    https://doi.org/10.1901/jeab.1962.5-s543
    """

    __slots__ = ("_components", "_active", "_last_now")

    def __init__(self, components: Sequence[Schedule]) -> None:
        self._components: list[Schedule] = _validate_components(components)
        self._active: int = 0
        self._last_now: float | None = None

    @property
    def n_components(self) -> int:
        return len(self._components)

    @property
    def active_index(self) -> int:
        return self._active

    @property
    def is_terminal(self) -> bool:
        return self._active == len(self._components) - 1

    def step(self, now: float, event: ResponseEvent | None = None) -> Outcome:
        _check_time(self._last_now, now)
        _check_event(now, event)
        self._last_now = now

        inner = self._components[self._active].step(now, event)

        if inner.reinforced:
            if self.is_terminal:
                self._active = 0
                meta: dict[str, object] = dict(inner.meta)
                meta["current_component"] = self._active
                return Outcome(
                    reinforced=True,
                    reinforcer=inner.reinforcer,
                    meta=meta,
                )
            self._active += 1
            meta = dict(inner.meta)
            meta["current_component"] = self._active
            meta["chain_transition"] = True
            return Outcome(meta=meta)

        meta = dict(inner.meta)
        meta["current_component"] = self._active
        return Outcome(meta=meta)

    def reset(self) -> None:
        self._active = 0
        self._last_now = None
        for c in self._components:
            c.reset()
