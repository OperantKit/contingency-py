"""Concurrent compound schedule with Changeover Delay (COD).

A concurrent schedule presents two or more component schedules on
distinct *operanda* (keys, levers, buttons). Each component is a
standalone :class:`~contingency.interfaces.Schedule`; responses on
operandum ``k`` are routed exclusively to ``components[k]``. The
composite returns the :class:`~contingency.entities.Outcome` produced by
the component that received the event, subject to changeover gating.

Changeover Delay (Catania, 1963)
--------------------------------
When the subject switches from operandum A to operandum B, no response
on B is reinforced for ``cod`` time units after the switch. This
discourages rapid alternation and superstitious changeover responding.
The timer resets on every subsequent changeover: "since the last
changeover".

A reinforcer that B's component sets up during the COD is *held*, not
lost: it is delivered by the first response on B after the COD ends,
and B's component does not start its next interval (or ratio) until
that delivery. This is the procedure of Herrnstein (1961): "no
response was reinforced within 1.5 seconds of a changeover", with
the programmed reinforcer kept until collected.

If no changeover has yet occurred (e.g. the very first response of a
session), COD is not active. This module implements the standard
"since-last-changeover" semantics of Catania (1963): each fresh
switch re-anchors the COD clock.

Fixed-Ratio Changeover (FRCO)
----------------------
With ``frco > 0``, a switch is not counted as a changeover until the
subject has emitted ``frco`` *consecutive* responses on the new
operandum. The COD timer arms only on the ``frco``-th such response
(and at that response's timestamp). Responses 1..frco-1 on the new
operandum are not yet "a changeover" and are therefore not gated;
reinforcement on those early responses is allowed. This mirrors the
``frco`` semantics of ``contingency-dsl``.

Choice analytics NOT in scope
-----------------------------
By design this class handles only reinforcement *gating*. Choice
allocation, exclusive-choice detection, and other analytics live in
``session-analyzer`` and consume the event log, not the schedule
itself (see Issue #40).

Asymmetric COD — not implemented
--------------------------------
``cod_asymmetric: dict[tuple[str, str], float]`` (per-direction COD)
is deferred. When added, it will be a drop-in extension of the
scalar ``cod`` parameter; the single ``_cod_active`` helper below is
the extension point.

References
----------
Catania, A. C. (1963). Concurrent performances: Reinforcement
interaction and response independence. *Journal of the Experimental
Analysis of Behavior*, 6(2), 253-263.
https://doi.org/10.1901/jeab.1963.6-253

Herrnstein, R. J. (1961). Relative and absolute strength of response
as a function of frequency of reinforcement. *Journal of the
Experimental Analysis of Behavior*, 4(3), 267-272.
https://doi.org/10.1901/jeab.1961.4-267
"""

from __future__ import annotations

from typing import Any

from contingency.entities import Outcome, ResponseEvent
from contingency.errors import ScheduleConfigError
from contingency.interfaces import Schedule
from contingency.schedules._rust import component_specs, compound_schedule, core_too_old

__all__ = ["Concurrent"]

class Concurrent:
    """Concurrent schedule with changeover gating.

    Parameters
    ----------
    components:
        Mapping from operandum identifier (e.g. ``"left"``) to a
        :class:`Schedule`. Must contain at least two entries. Insertion
        order is preserved.
    cod:
        Changeover Delay in the caller's time units. After a changeover,
        no response on the new operandum is reinforced for ``cod`` time
        units; a reinforcer set up meanwhile is held until the first
        response after the COD. Default ``0.0`` disables COD. Must be
        ``>= 0``.
    frco:
        fixed-ratio changeover: number of consecutive responses on a new
        operandum required for the switch to count as a changeover.
        Default ``0`` treats every switch as immediate. Must be ``>= 0``.

    Notes
    -----
    Time advancement. On every :meth:`step` each component receives a
    :meth:`Schedule.step` call so that time-based components (FT, VT,
    RT) can advance their internal clocks. Only the component keyed by
    ``event.operandum`` receives the :class:`ResponseEvent`; all other
    components receive ``None`` at the same ``now``.

    Gating. While COD is active on the responded operandum, a
    reinforcer that the response would produce is held and
    :class:`Concurrent` returns an un-reinforced outcome carrying
    ``meta["cod_suppressed"] = True`` alongside
    ``meta["operandum"] = <key>``. The first response on that operandum
    after the COD collects the held reinforcer. If another component
    delivers a reinforcer on its tick at that step and no punisher
    fires, that reinforcer is returned instead, with
    ``meta["cod_suppressed"] = True`` and no ``"operandum"`` key. How it
    is held depends on the component:

    - FR, VR, RR, FI, VI, RI: the response is first tried on a copy of
      the component. If the copy would reinforce, the real component
      receives only a time tick, so the reinforcer stays set up and its
      next interval or ratio does not start until delivery. A response
      that would not reinforce is applied as usual, so it counts toward
      a ratio.
    - Any other component (time-based, differential, nested compound):
      the response is applied as usual, so the component sees every
      response, and the reinforcer it produces is kept by
      :class:`Concurrent` until delivery. The component's own next
      cycle starts at once.

    The compound runs on the Rust engine. Its components must be
    schedules of this package that run there, not yet stepped, and
    distinct objects; otherwise the constructor raises
    :class:`ScheduleConfigError`.
    Components and punishment schedules count alike: each object may
    appear only once among them. When two components deliver a
    reinforcer on the same tick, the first one in the insertion order
    of ``components`` wins (the Rust engine keeps that order).
    """

    __slots__ = (
        "_components",
        "_cod",
        "_cod_directional",
        "_frco",
        "_punish",
        "_punish_changeover",
        "_punish_directional",
        "_rust",
    )

    def __init__(
        self,
        components: dict[str, Schedule],
        cod: float = 0.0,
        frco: int = 0,
        cod_directional: dict[tuple[str, str], float] | None = None,
        punish: dict[str, Schedule] | None = None,
        punish_changeover: Schedule | None = None,
        punish_directional: dict[tuple[str, str], Schedule] | None = None,
    ) -> None:
        if not isinstance(components, dict):
            raise ScheduleConfigError(
                f"Concurrent requires a dict of components, got "
                f"{type(components).__name__}"
            )
        if len(components) < 2:
            raise ScheduleConfigError(
                f"Concurrent requires at least 2 components, got {len(components)}"
            )
        if isinstance(cod, bool) or not isinstance(cod, (int, float)):
            raise ScheduleConfigError(
                f"Concurrent requires a numeric cod, got {type(cod).__name__}"
            )
        if cod < 0:
            raise ScheduleConfigError(f"Concurrent requires cod >= 0, got {cod}")
        if not isinstance(frco, int) or isinstance(frco, bool):
            raise ScheduleConfigError(
                f"Concurrent requires an int frco, got {type(frco).__name__}"
            )
        if frco < 0:
            raise ScheduleConfigError(f"Concurrent requires frco >= 0, got {frco}")

        # dict(...) to take a shallow copy; preserves insertion order.
        self._components: dict[str, Schedule] = dict(components)
        self._cod: float = float(cod)
        self._frco: int = frco

        # Validate directional COD: keys must be (str, str), no self-
        # transitions, no duplicates (duplicates are intrinsic to dict
        # keys; we also reject the (A, A) form).
        self._cod_directional: dict[tuple[str, str], float] | None = None
        if cod_directional is not None:
            if not isinstance(cod_directional, dict):
                raise ScheduleConfigError(
                    f"Concurrent cod_directional must be a dict, got "
                    f"{type(cod_directional).__name__}"
                )
            validated: dict[tuple[str, str], float] = {}
            for key, value in cod_directional.items():
                if (
                    not isinstance(key, tuple)
                    or len(key) != 2
                    or not all(isinstance(k, str) for k in key)
                ):
                    raise ScheduleConfigError(
                        f"Concurrent cod_directional keys must be "
                        f"(from, to) tuples of str, got {key!r}"
                    )
                from_op, to_op = key
                if from_op == to_op:
                    raise ScheduleConfigError(
                        f"Concurrent cod_directional rejects self-"
                        f"transition {key!r}"
                    )
                if isinstance(value, bool) or not isinstance(
                    value, (int, float)
                ):
                    raise ScheduleConfigError(
                        f"Concurrent cod_directional values must be "
                        f"numeric, got {type(value).__name__} for {key!r}"
                    )
                if value < 0:
                    raise ScheduleConfigError(
                        f"Concurrent cod_directional values must be "
                        f">= 0, got {value} for {key!r}"
                    )
                validated[(from_op, to_op)] = float(value)
            if validated:
                self._cod_directional = validated

        # Validate punish dict.
        self._punish: dict[str, Schedule] | None = None
        if punish is not None:
            if not isinstance(punish, dict):
                raise ScheduleConfigError(
                    f"Concurrent punish must be a dict, got "
                    f"{type(punish).__name__}"
                )
            punish_validated: dict[str, Schedule] = {}
            for op_key, sched in punish.items():
                if not isinstance(op_key, str):
                    raise ScheduleConfigError(
                        f"Concurrent punish keys must be operandum "
                        f"strings, got {type(op_key).__name__}"
                    )
                if op_key not in self._components:
                    raise ScheduleConfigError(
                        f"Concurrent punish references unknown operandum "
                        f"{op_key!r}; known operanda: "
                        f"{sorted(self._components.keys())!r}"
                    )
                if not isinstance(sched, Schedule):
                    raise ScheduleConfigError(
                        f"Concurrent punish[{op_key!r}] must implement "
                        f"Schedule, got {type(sched).__name__}"
                    )
                punish_validated[op_key] = sched
            if punish_validated:
                self._punish = punish_validated

        self._punish_changeover, self._punish_directional = _validate_changeover_punish(
            self._components, punish_changeover, punish_directional
        )

        self._rust: Any = compound_schedule(self._rust_spec())

    @property
    def components(self) -> dict[str, Schedule]:
        """The component mapping (read-only view; modifying has no effect)."""
        return dict(self._components)

    @property
    def cod(self) -> float:
        """The configured changeover delay."""
        return self._cod

    @property
    def frco(self) -> int:
        """The configured fixed-ratio changeover."""
        return self._frco

    @property
    def cod_directional(self) -> dict[tuple[str, str], float] | None:
        """The configured per-direction COD overrides, if any."""
        if self._cod_directional is None:
            return None
        return dict(self._cod_directional)

    @property
    def punish(self) -> dict[str, Schedule] | None:
        """The configured per-operandum punishment schedules, if any."""
        if self._punish is None:
            return None
        return dict(self._punish)

    @property
    def punish_changeover(self) -> Schedule | None:
        """The punishment schedule shared by every changeover direction, if any."""
        return self._punish_changeover

    @property
    def punish_directional(self) -> dict[tuple[str, str], Schedule] | None:
        """The per-``(from, to)`` changeover punishment schedules, if any."""
        if self._punish_directional is None:
            return None
        return dict(self._punish_directional)

    def _changeover_punishers(self) -> list[Schedule]:
        out: list[Schedule] = []
        if self._punish_changeover is not None:
            out.append(self._punish_changeover)
        if self._punish_directional is not None:
            out.extend(self._punish_directional.values())
        return out

    def _rust_spec(self) -> dict[str, Any]:
        keys = list(self._components)
        pkeys = list(self._punish or {})
        dkeys = list(self._punish_directional or {})
        changeover = [] if self._punish_changeover is None else [self._punish_changeover]
        specs = component_specs(
            "Concurrent",
            [
                *(self._components[k] for k in keys),
                *((self._punish or {})[k] for k in pkeys),
                *changeover,
                *((self._punish_directional or {})[k] for k in dkeys),
            ],
        )
        comp_specs = specs[: len(keys)]
        rest = specs[len(keys) :]
        punish_specs, rest = rest[: len(pkeys)], rest[len(pkeys) :]
        changeover_specs, directional_specs = rest[: len(changeover)], rest[len(changeover) :]
        params: dict[str, Any] = {
            "components": dict(zip(keys, comp_specs, strict=True)),
            "cod": self._cod,
            "frco": self._frco,
        }
        if self._cod_directional is not None:
            params["cod_directional"] = [
                {"from": a, "to": b, "cod": cod} for (a, b), cod in self._cod_directional.items()
            ]
        if pkeys:
            params["punish"] = dict(zip(pkeys, punish_specs, strict=True))
        if changeover_specs:
            params["punish_changeover"] = changeover_specs[0]
        if dkeys:
            params["punish_directional"] = [
                {"from": a, "to": b, "schedule": spec}
                for (a, b), spec in zip(dkeys, directional_specs, strict=True)
            ]
        if (changeover_specs or dkeys) and not _rust_has_changeover_punish():
            raise core_too_old("Concurrent runs without changeover punishment")
        return {"type": "Concurrent", "params": params}

    @property
    def engine(self) -> str:
        """Always ``"rust"``: the Rust engine runs this schedule."""
        return "rust"

    def step(self, now: float, event: ResponseEvent | None = None) -> Outcome:
        return self._rust.step(now, event)  # type: ignore[no-any-return]

    def reset(self) -> None:
        for component in [
            *self._components.values(),
            *(self._punish or {}).values(),
            *self._changeover_punishers(),
        ]:
            component.reset()
        self._rust = compound_schedule(self._rust_spec())


def _validate_changeover_punish(
    components: dict[str, Schedule],
    punish_changeover: Schedule | None,
    punish_directional: dict[tuple[str, str], Schedule] | None,
) -> tuple[Schedule | None, dict[tuple[str, str], Schedule] | None]:
    """Check the changeover punishment arguments of :class:`Concurrent`."""
    if punish_changeover is not None and punish_directional is not None:
        raise ScheduleConfigError(
            "Concurrent takes punish_changeover or punish_directional, not both"
        )
    if punish_changeover is not None and not isinstance(punish_changeover, Schedule):
        raise ScheduleConfigError(
            f"Concurrent punish_changeover must implement Schedule, "
            f"got {type(punish_changeover).__name__}"
        )
    if punish_directional is None:
        return punish_changeover, None
    if not isinstance(punish_directional, dict):
        raise ScheduleConfigError(
            f"Concurrent punish_directional must be a dict, got "
            f"{type(punish_directional).__name__}"
        )
    validated: dict[tuple[str, str], Schedule] = {}
    for key, sched in punish_directional.items():
        if not isinstance(key, tuple) or len(key) != 2 or not all(isinstance(k, str) for k in key):
            raise ScheduleConfigError(
                f"Concurrent punish_directional keys must be (from, to) tuples of str, got {key!r}"
            )
        if key[0] == key[1]:
            raise ScheduleConfigError(
                f"Concurrent punish_directional rejects self-transition {key!r}"
            )
        for op in key:
            if op not in components:
                raise ScheduleConfigError(
                    f"Concurrent punish_directional references unknown operandum "
                    f"{op!r}; known operanda: {sorted(components)!r}"
                )
        if not isinstance(sched, Schedule):
            raise ScheduleConfigError(
                f"Concurrent punish_directional[{key!r}] must implement Schedule, "
                f"got {type(sched).__name__}"
            )
        validated[key] = sched
    return punish_changeover, (validated or None)


_RUST_CHANGEOVER_PUNISH: bool | None = None


def _rust_has_changeover_punish() -> bool:
    """Whether the installed ``contingency_core`` runs changeover punishment.

    An older build reads the Concurrent spec without the
    ``punish_changeover`` / ``punish_directional`` keys and would run the
    schedule without the punishment; such a build accepts a spec that
    gives both keys, which a current build rejects.
    """
    global _RUST_CHANGEOVER_PUNISH
    if _RUST_CHANGEOVER_PUNISH is None:
        from contingency.engine import from_spec, rust_engine_available

        if not rust_engine_available():
            return False
        fr1 = {"type": "FR", "params": {"n": 1}}
        probe = {
            "type": "Concurrent",
            "params": {
                "components": {"a": {"type": "EXT"}, "b": {"type": "EXT"}},
                "punish_changeover": fr1,
                "punish_directional": [{"from": "a", "to": "b", "schedule": fr1}],
            },
        }
        try:
            from_spec(probe)
        except (ScheduleConfigError, ValueError, TypeError):
            _RUST_CHANGEOVER_PUNISH = True
        else:
            _RUST_CHANGEOVER_PUNISH = False
    return _RUST_CHANGEOVER_PUNISH
