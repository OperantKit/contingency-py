"""Apply program-level parameter declarations to the schedule expression.

The contingency-dsl grammar lets a program declare defaults before its
schedule (``LH = 5s``, ``COD = 2s``, ``FRCO = 3``, ``BO = 5s``,
``RD = 1s``). :func:`apply_program_decls` returns the schedule with
those defaults written into it, so that :func:`contingency.from_dsl`
and :func:`contingency.to_spec` run the program as declared without
``contingency_dsl.analyze()`` (which drops ``@algorithm`` annotations):

* ``LH`` wraps every leaf base schedule that has no LH of its own
  (grammar constraint 3), following ``contingency_dsl.analyze()``: the
  leaves are atomic schedules, ``EXT`` / ``CRF``, modifiers and
  second-order schedules reached through combinators and schedule
  annotations, and inside ``TO`` / ``ResponseCost`` wrappers; each copy of
  ``Repeat`` receives it. Ratio schedules (``FR`` / ``VR`` / ``RR`` /
  ``CRF`` / ``PR``) and ``EXT`` are left without it: an LH on them is
  vacuous (the analyzer's ``VACUOUS_LH_RATIO``), and the runtime holds
  only interval schedules. A leaf the runtime cannot
  hold (``FT``, ``DRL``, ...) raises, as the written ``LH`` does.
* ``COD`` and ``FRCO`` are set on every ``Conc`` that does not set them
  itself (constraint 12), however deeply it is nested. On a ``Conc``
  with directional COD entries and no symmetric base, the program
  ``COD`` becomes the base (constraint 59).
* ``BO`` is checked by the caller (the runtime has no blackout) and is
  otherwise not applied.
* ``RD`` greater than zero is rejected: the runtime has no
  reinforcement delay. ``RD=0s`` means immediate delivery and is
  dropped.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from contingency.errors import ScheduleConfigError

__all__ = ["apply_program_decls"]

_CONC_DECLS = ("COD", "FRCO")


def apply_program_decls(program: Any) -> Any:
    """Return ``program.schedule`` with the program's declarations applied."""
    schedule = program.schedule
    for decl in program.param_decls or ():
        if decl.name == "LH":
            if not decl.value > 0:
                raise ScheduleConfigError(
                    f"program-level LH must be greater than zero, got {decl.value!r}"
                )
            schedule = _apply_lh(schedule, float(decl.value), decl.time_unit)
        elif decl.name in _CONC_DECLS:
            schedule = _apply_conc_param(schedule, decl.name, decl.value, decl.time_unit)
        elif decl.name == "RD":
            if decl.value != 0:
                raise ScheduleConfigError(
                    "program-level RD declaration: the runtime has no reinforcement "
                    "delay; running it would silently drop the delay (write the delay "
                    "into the schedule, e.g. Tand(S, FT d))"
                )
        elif decl.name != "BO":
            raise ScheduleConfigError(f"unknown program-level parameter declaration: {decl.name!r}")
    return schedule


def _apply_lh(expr: Any, hold: float, time_unit: str | None) -> Any:
    from contingency_dsl.ast import (
        AnnotatedSchedule,
        Atomic,
        Compound,
        LimitedHold,
        Modifier,
        ResponseCostWrapped,
        SecondOrder,
        Special,
        TimeoutWrapped,
    )
    from contingency_dsl.enums import Domain

    if isinstance(expr, LimitedHold):
        return expr
    if (
        (isinstance(expr, Atomic) and expr.domain is Domain.RATIO)
        or isinstance(expr, Special)
        or (isinstance(expr, Modifier) and expr.modifier == "PR")
    ):
        # Vacuous: a ratio schedule is satisfied on the response itself,
        # and EXT never reinforces.
        return expr
    if isinstance(expr, (TimeoutWrapped, ResponseCostWrapped)):
        # The written form ``S LH x TO(...)`` puts the LH inside the wrapper.
        return dataclasses.replace(expr, inner=_apply_lh(expr.inner, hold, time_unit))
    if isinstance(expr, Compound):
        return dataclasses.replace(
            expr, components=tuple(_apply_lh(c, hold, time_unit) for c in expr.components)
        )
    if isinstance(expr, AnnotatedSchedule):
        return dataclasses.replace(expr, expr=_apply_lh(expr.expr, hold, time_unit))
    if isinstance(expr, Modifier) and expr.modifier == "Repeat" and expr.inner is not None:
        # analyze() expands Repeat before applying LH, so each copy gets it.
        return dataclasses.replace(expr, inner=_apply_lh(expr.inner, hold, time_unit))
    if isinstance(expr, (Atomic, Special, Modifier, SecondOrder)):
        return LimitedHold(hold_duration=hold, inner=expr, time_unit=time_unit)
    return expr


def _apply_conc_param(node: Any, name: str, value: float, time_unit: str | None) -> Any:
    from contingency_dsl.ast import Compound, CompoundParam, DirectionalCOD
    from contingency_dsl.enums import Combinator

    def walk(item: Any) -> Any:
        if isinstance(item, tuple):
            return tuple(walk(x) for x in item)
        if isinstance(item, list):
            return [walk(x) for x in item]
        if isinstance(item, dict):
            return {k: walk(v) for k, v in item.items()}
        if not dataclasses.is_dataclass(item) or isinstance(item, type):
            return item
        changes = {f.name: walk(getattr(item, f.name)) for f in dataclasses.fields(item) if f.init}
        if isinstance(item, Compound) and item.combinator is Combinator.CONC:
            params = dict(changes.get("params") or {})
            current = params.get(name)
            if current is None:
                params[name] = CompoundParam(value=value, time_unit=time_unit)
            elif isinstance(current, DirectionalCOD) and current.base is None:
                # The program COD is the symmetric base that the directional
                # entries override (grammar constraint 59).
                params[name] = dataclasses.replace(
                    current, base=CompoundParam(value=value, time_unit=time_unit)
                )
            changes["params"] = params
        return dataclasses.replace(item, **changes)

    return walk(node)
