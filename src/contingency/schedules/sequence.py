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
another S^D is in effect.

Each component therefore runs on its own clock, which advances only
while that component is active. The compound translates ``now`` into
the component's clock before stepping it and translates the reinforcer
time back. A newly-activated component's clock resumes on its first
step *after* the transition, at the time it had accumulated while
active (0 on its first activation). Time-based schedules (FI, FT, VI,
VT, RI, RT, DRO) therefore count from their first step after
activation — not from session start or from the component's last
reinforcer — exactly as in a real operant chamber where the S^D and the
clock both restart on link entry. The first component runs from time 0,
as a standalone schedule does. The gap between the transition and that
first step is counted by no component, so drive time-based links with a
step grid fine enough for the intended timing.

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
from typing import Any

from contingency.entities import Outcome, ResponseEvent
from contingency.errors import ScheduleConfigError
from contingency.interfaces import Schedule
from contingency.schedules._rust import component_specs, compound_schedule

__all__ = ["Multiple", "Chained", "Tandem"]

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

    The compound runs on the Rust engine. Its components must be
    schedules of this package that run there, not yet stepped, and
    distinct objects; otherwise the constructor raises
    :class:`ScheduleConfigError`.

    References
    ----------
    Ferster, C. B., & Skinner, B. F. (1957). *Schedules of reinforcement*.
    Appleton-Century-Crofts.

    Reynolds, G. S. (1961). Behavioral contrast. *Journal of the
    Experimental Analysis of Behavior*, 4(1), 57-71.
    https://doi.org/10.1901/jeab.1961.4-57
    """

    __slots__ = ("_components", "_stimuli", "_rust")

    def __init__(
        self,
        components: Sequence[Schedule],
        stimuli: Sequence[str] | None = None,
    ) -> None:
        self._components: list[Schedule] = _validate_components(components)
        self._stimuli: list[str] = _validate_stimuli(
            stimuli, len(self._components)
        )
        self._rust: Any = compound_schedule(self._rust_spec(), needs_state=True)

    @property
    def n_components(self) -> int:
        return len(self._components)

    @property
    def active_index(self) -> int:
        return int(self._rust.state()["active_index"])

    @property
    def current_stimulus(self) -> str:
        return self._stimuli[self.active_index]

    def _rust_spec(self) -> dict[str, Any]:
        specs = component_specs("Multiple", self._components)
        return {"type": "Multiple", "params": {"components": specs, "stimuli": list(self._stimuli)}}

    @property
    def engine(self) -> str:
        """Always ``"rust"``: the Rust engine runs this schedule."""
        return "rust"

    def step(self, now: float, event: ResponseEvent | None = None) -> Outcome:
        return self._rust.step(now, event)  # type: ignore[no-any-return]

    def reset(self) -> None:
        for component in self._components:
            component.reset()
        self._rust = compound_schedule(self._rust_spec(), needs_state=True)


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

    The compound runs on the Rust engine. Its components must be
    schedules of this package that run there, not yet stepped, and
    distinct objects; otherwise the constructor raises
    :class:`ScheduleConfigError`.

    References
    ----------
    Ferster, C. B., & Skinner, B. F. (1957). *Schedules of reinforcement*.
    Appleton-Century-Crofts.

    Kelleher, R. T., & Gollub, L. R. (1962). A review of positive
    conditioned reinforcement. *Journal of the Experimental Analysis of
    Behavior*, 5(4 Suppl), 543-597.
    https://doi.org/10.1901/jeab.1962.5-s543
    """

    __slots__ = ("_components", "_stimuli", "_rust")

    def __init__(
        self,
        components: Sequence[Schedule],
        stimuli: Sequence[str] | None = None,
    ) -> None:
        self._components: list[Schedule] = _validate_components(components)
        self._stimuli: list[str] = _validate_stimuli(
            stimuli, len(self._components)
        )
        self._rust: Any = compound_schedule(self._rust_spec(), needs_state=True)

    @property
    def n_components(self) -> int:
        return len(self._components)

    @property
    def active_index(self) -> int:
        return int(self._rust.state()["active_index"])

    @property
    def current_stimulus(self) -> str:
        return self._stimuli[self.active_index]

    @property
    def is_terminal(self) -> bool:
        return self.active_index == len(self._components) - 1

    def _rust_spec(self) -> dict[str, Any]:
        specs = component_specs("Chained", self._components)
        return {"type": "Chained", "params": {"components": specs, "stimuli": list(self._stimuli)}}

    @property
    def engine(self) -> str:
        """Always ``"rust"``: the Rust engine runs this schedule."""
        return "rust"

    def step(self, now: float, event: ResponseEvent | None = None) -> Outcome:
        return self._rust.step(now, event)  # type: ignore[no-any-return]

    def reset(self) -> None:
        for component in self._components:
            component.reset()
        self._rust = compound_schedule(self._rust_spec(), needs_state=True)


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

    The compound runs on the Rust engine. Its components must be
    schedules of this package that run there, not yet stepped, and
    distinct objects; otherwise the constructor raises
    :class:`ScheduleConfigError`.

    References
    ----------
    Ferster, C. B., & Skinner, B. F. (1957). *Schedules of reinforcement*.
    Appleton-Century-Crofts.

    Kelleher, R. T., & Gollub, L. R. (1962). A review of positive
    conditioned reinforcement. *Journal of the Experimental Analysis of
    Behavior*, 5(4 Suppl), 543-597.
    https://doi.org/10.1901/jeab.1962.5-s543
    """

    __slots__ = ("_components", "_rust")

    def __init__(self, components: Sequence[Schedule]) -> None:
        self._components: list[Schedule] = _validate_components(components)
        self._rust: Any = compound_schedule(self._rust_spec(), needs_state=True)

    @property
    def n_components(self) -> int:
        return len(self._components)

    @property
    def active_index(self) -> int:
        return int(self._rust.state()["active_index"])

    @property
    def is_terminal(self) -> bool:
        return self.active_index == len(self._components) - 1

    def _rust_spec(self) -> dict[str, Any]:
        specs = component_specs("Tandem", self._components)
        return {"type": "Tandem", "params": {"components": specs}}

    @property
    def engine(self) -> str:
        """Always ``"rust"``: the Rust engine runs this schedule."""
        return "rust"

    def step(self, now: float, event: ResponseEvent | None = None) -> Outcome:
        return self._rust.step(now, event)  # type: ignore[no-any-return]

    def reset(self) -> None:
        for component in self._components:
            component.reset()
        self._rust = compound_schedule(self._rust_spec(), needs_state=True)
