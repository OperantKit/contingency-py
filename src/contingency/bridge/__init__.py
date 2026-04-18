"""Bridge from contingency-dsl AST nodes to runtime :class:`Schedule` objects.

The DSL ``contingency-dsl`` defines a declarative language for
reinforcement contingencies. Its Python reference parser
(``contingency-dsl-py``) produces an immutable AST. This module
translates that AST into executable schedules from
:mod:`contingency.schedules`.

Translation table
-----------------

Atomic schedules — ``(Distribution, Domain)`` selects the class:

========  =========  ===============================================
Dist      Domain     Runtime class (with value handling)
========  =========  ===============================================
FIXED     RATIO      ``FR(int(value))``
FIXED     INTERVAL   ``FI(value * time_unit_factor)``
FIXED     TIME       ``FT(value * time_unit_factor)``
VARIABLE  RATIO      ``VR(value)``
VARIABLE  INTERVAL   ``VI(value * time_unit_factor)``
VARIABLE  TIME       ``VT(value * time_unit_factor)``
RANDOM    RATIO      ``RR(1.0 / value)``  *(mean-value to probability)*
RANDOM    INTERVAL   ``RI(value * time_unit_factor)``
RANDOM    TIME       ``RT(value * time_unit_factor)``
========  =========  ===============================================

Time units resolve via ``time_unit_factor``:

* ``None`` -> ``1.0`` (pass through; caller provides the base)
* ``"s"``  -> ``1.0``
* ``"ms"`` -> ``0.001``
* ``"min"`` -> ``60.0``
* any other string -> :class:`ScheduleConfigError`

The kwarg ``time_unit_seconds`` multiplies the final seconds value —
it is a remap knob for callers whose runtime clock is not in seconds.

Specials:
  * ``EXT`` -> :class:`EXT`
  * ``CRF`` -> ``FR(1)``

Compound (by ``Combinator``):
  * ``CONC`` -> :class:`Concurrent` with numeric keys ``c0``, ``c1``,
    ...; optional ``cod`` and ``cor`` extracted from ``params``.
  * ``ALT`` -> :class:`Alternative`; >2 components are nested left-
    associative: ``Alt(Alt(a, b), c)``.
  * ``MULT`` -> :class:`Multiple` with default stimuli ``s0, s1, ...``.
  * ``CHAIN`` -> :class:`Chained` with default stimuli ``s0, s1, ...``.
  * ``TAND`` -> :class:`Tandem`.
  * All other combinators (``CONJ``, ``MIX``, ``OVERLAY``,
    ``INTERPOLATE``, ``INTERP``) raise :class:`ScheduleConfigError`.

Modifier:
  * ``DRL`` -> ``DRL(value * time_unit_factor)``.
  * ``DRH`` -> ``DRH(2, value * time_unit_factor)``. The DSL carries a
    single time value for DRH (an IRT threshold, e.g. ``DRH2s``).
    Because the runtime :class:`DRH` is the more general "N responses
    per window" variant, we supply ``response_count=2`` to match the
    IRT-threshold intent: at least two responses within the window
    (equivalent to the last inter-response time being ``<= window``).
  * ``DRO`` -> ``DRO(value * time_unit_factor, "resetting")``.
  * ``PR`` — step function chosen by ``pr_step``:

    - ``"linear"`` / ``"hodos"``: :func:`arithmetic`.
    - ``"geometric"`` / ``"exponential"``: :func:`geometric`.
    - Missing ``pr_start`` defaults to ``1``; missing ``pr_increment``
      defaults to ``1``; missing ``pr_ratio`` defaults to ``2.0``.
  * ``Repeat`` -> :class:`Chained` over ``int(value)`` copies of
    ``from_dsl(inner)``. (The DSL analyzer expands ``Repeat`` into
    ``Tand``; this bridge only sees ``Repeat`` on the pre-expansion
    path, but still handles it for robustness. If the caller passes
    an analyzed program the ``Tand`` path is taken instead.)
  * ``Pctl`` -> :class:`ScheduleConfigError` (deferred feature).

``LimitedHold`` -> :class:`LimitedHold` wrapping ``from_dsl(inner)``.

``SecondOrder``, ``IdentifierRef``, and the aversive / timeout /
response-cost / adjusting / interlocking / trial-based wrappers raise
:class:`ScheduleConfigError` — they are not yet implemented.

``Program`` nodes translate via ``program.schedule``; program-level
annotations and param declarations are ignored for this pass.

References
----------
Catania, A. C. (1966). Concurrent performances: Reinforcement
interaction and response independence. *Journal of the Experimental
Analysis of Behavior*, 9(3), 253-263.
https://doi.org/10.1901/jeab.1966.9-253

Ferster, C. B., & Skinner, B. F. (1957). *Schedules of reinforcement*.
Appleton-Century-Crofts.

Fleshler, M., & Hoffman, H. S. (1962). A progression for generating
variable-interval schedules. *Journal of the Experimental Analysis of
Behavior*, 5(4), 529-530. https://doi.org/10.1901/jeab.1962.5-529
"""

from __future__ import annotations

from typing import Any

from contingency.errors import ScheduleConfigError
from contingency.interfaces import Schedule
from contingency.schedules import (
    DRH,
    DRL,
    DRO,
    EXT,
    FI,
    FR,
    FT,
    RI,
    RR,
    RT,
    VI,
    VR,
    VT,
    Alternative,
    Chained,
    Concurrent,
    LimitedHold,
    Multiple,
    ProgressiveRatio,
    Tandem,
    arithmetic,
    geometric,
)

__all__ = ["from_dsl"]


# --- Time unit resolution --------------------------------------------------


_TIME_UNIT_TABLE: dict[str, float] = {
    "s": 1.0,
    "ms": 0.001,
    "min": 60.0,
}


def _time_unit_factor(time_unit: str | None) -> float:
    """Return the seconds-multiplier for ``time_unit`` or raise."""
    if time_unit is None:
        return 1.0
    if time_unit in _TIME_UNIT_TABLE:
        return _TIME_UNIT_TABLE[time_unit]
    raise ScheduleConfigError(
        f"unsupported time_unit {time_unit!r}; expected one of "
        f"{sorted(_TIME_UNIT_TABLE) + ['None']}"
    )


def _scaled_time(value: float, time_unit: str | None, time_unit_seconds: float) -> float:
    """Translate a DSL time value to runtime seconds.

    ``value`` is the AST's numeric value (e.g. ``30`` from ``FI30s``).
    ``time_unit`` is the AST's unit string.
    ``time_unit_seconds`` is the caller's remap factor: a caller whose
    runtime clock is already in ms can pass ``time_unit_seconds=1000``.
    """
    return value * _time_unit_factor(time_unit) * time_unit_seconds


# --- Dispatch --------------------------------------------------------------


def from_dsl(ast_node: Any, *, time_unit_seconds: float = 1.0) -> Schedule:
    """Translate a contingency-dsl AST node to a :class:`Schedule`.

    Parameters
    ----------
    ast_node:
        A ``contingency_dsl`` AST node: ``Atomic``, ``Special``,
        ``Compound``, ``Modifier``, ``LimitedHold``, or ``Program``.
        When passed a ``Program``, translates ``Program.schedule``.
    time_unit_seconds:
        Conversion factor from DSL seconds to the caller's runtime
        clock unit. Defaults to ``1.0``, i.e. the DSL's seconds pass
        through directly.

    Returns
    -------
    Schedule
        An executable reinforcement schedule.

    Raises
    ------
    ScheduleConfigError
        When the AST contains constructs not yet supported (see
        module docstring for the supported subset).
    """
    # Imported lazily so that ``contingency`` remains importable even
    # when ``contingency_dsl`` is not installed; the ImportError is
    # surfaced only at bridge time.
    from contingency_dsl.ast import (
        AdjustingSchedule,
        AnnotatedSchedule,
        Atomic,
        AversiveSchedule,
        Compound,
        IdentifierRef,
        InterlockingSchedule,
        Modifier,
        Program,
        ResponseCostWrapped,
        SecondOrder,
        Special,
        TimeoutWrapped,
        TrialBased,
    )
    from contingency_dsl.ast import LimitedHold as DSLLimitedHold

    if isinstance(ast_node, Program):
        return from_dsl(ast_node.schedule, time_unit_seconds=time_unit_seconds)

    if isinstance(ast_node, AnnotatedSchedule):
        # Annotations are metadata; for this pass they are ignored.
        return from_dsl(ast_node.expr, time_unit_seconds=time_unit_seconds)

    if isinstance(ast_node, Atomic):
        return _from_atomic(ast_node, time_unit_seconds)

    if isinstance(ast_node, Special):
        return _from_special(ast_node)

    if isinstance(ast_node, Compound):
        return _from_compound(ast_node, time_unit_seconds)

    if isinstance(ast_node, Modifier):
        return _from_modifier(ast_node, time_unit_seconds)

    if isinstance(ast_node, DSLLimitedHold):
        inner = from_dsl(ast_node.inner, time_unit_seconds=time_unit_seconds)
        hold = _scaled_time(
            ast_node.hold_duration, ast_node.time_unit, time_unit_seconds
        )
        return LimitedHold(inner, hold)

    if isinstance(ast_node, SecondOrder):
        raise ScheduleConfigError(
            "SecondOrder schedules are not yet supported by the bridge; "
            f"got overall={ast_node.overall!r} unit={ast_node.unit!r}"
        )

    if isinstance(ast_node, IdentifierRef):
        raise ScheduleConfigError(
            f"unresolved IdentifierRef({ast_node.name!r}); "
            "call contingency_dsl.analyze() on the Program first to "
            "substitute let-bindings before bridging"
        )

    if isinstance(
        ast_node,
        (
            AdjustingSchedule,
            AversiveSchedule,
            InterlockingSchedule,
            ResponseCostWrapped,
            TimeoutWrapped,
            TrialBased,
        ),
    ):
        raise ScheduleConfigError(
            f"AST node type {type(ast_node).__name__} is not yet supported "
            "by the bridge"
        )

    raise ScheduleConfigError(
        f"unrecognised AST node: {type(ast_node).__name__}"
    )


# --- Per-node translators --------------------------------------------------


def _from_atomic(node: Any, time_unit_seconds: float) -> Schedule:
    from contingency_dsl.enums import Distribution, Domain

    dist = node.dist
    domain = node.domain
    value = node.value

    if dist is Distribution.FIXED and domain is Domain.RATIO:
        if value != int(value) or value < 1:
            raise ScheduleConfigError(
                f"FR requires a positive integer ratio, got {value!r}"
            )
        return FR(int(value))
    if dist is Distribution.FIXED and domain is Domain.INTERVAL:
        return FI(_scaled_time(value, node.time_unit, time_unit_seconds))
    if dist is Distribution.FIXED and domain is Domain.TIME:
        return FT(_scaled_time(value, node.time_unit, time_unit_seconds))
    if dist is Distribution.VARIABLE and domain is Domain.RATIO:
        return VR(float(value))
    if dist is Distribution.VARIABLE and domain is Domain.INTERVAL:
        return VI(_scaled_time(value, node.time_unit, time_unit_seconds))
    if dist is Distribution.VARIABLE and domain is Domain.TIME:
        return VT(_scaled_time(value, node.time_unit, time_unit_seconds))
    if dist is Distribution.RANDOM and domain is Domain.RATIO:
        if not (value > 0):
            raise ScheduleConfigError(
                f"RR requires a positive mean-value to convert to "
                f"probability, got {value!r}"
            )
        return RR(1.0 / float(value))
    if dist is Distribution.RANDOM and domain is Domain.INTERVAL:
        return RI(_scaled_time(value, node.time_unit, time_unit_seconds))
    if dist is Distribution.RANDOM and domain is Domain.TIME:
        return RT(_scaled_time(value, node.time_unit, time_unit_seconds))

    raise ScheduleConfigError(
        f"unhandled atomic combination: dist={dist!r} domain={domain!r}"
    )


def _from_special(node: Any) -> Schedule:
    if node.kind == "EXT":
        return EXT()
    if node.kind == "CRF":
        return FR(1)
    raise ScheduleConfigError(f"unknown Special kind: {node.kind!r}")


def _from_compound(node: Any, time_unit_seconds: float) -> Schedule:
    from contingency_dsl.enums import Combinator

    combinator = node.combinator
    components = [
        from_dsl(c, time_unit_seconds=time_unit_seconds)
        for c in node.components
    ]
    params = node.params or {}

    if combinator is Combinator.CONC:
        return _build_concurrent(components, params, time_unit_seconds)

    if combinator is Combinator.ALT:
        if len(components) < 2:
            raise ScheduleConfigError(
                f"Alt requires >= 2 components, got {len(components)}"
            )
        # Left-associative nesting for >2 components.
        acc = Alternative(components[0], components[1])
        for extra in components[2:]:
            acc = Alternative(acc, extra)
        return acc

    if combinator is Combinator.MULT:
        stimuli = [f"s{i}" for i in range(len(components))]
        return Multiple(components, stimuli=stimuli)

    if combinator is Combinator.CHAIN:
        stimuli = [f"s{i}" for i in range(len(components))]
        return Chained(components, stimuli=stimuli)

    if combinator is Combinator.TAND:
        return Tandem(components)

    raise ScheduleConfigError(
        f"combinator {combinator.value!r} is not yet supported by the "
        "bridge (CONJ, MIX, OVERLAY, INTERPOLATE, INTERP deferred)"
    )


def _build_concurrent(
    components: list[Schedule],
    params: dict[str, Any],
    time_unit_seconds: float,
) -> Concurrent:
    # Import lazily to avoid a hard dep at module import time.
    from contingency_dsl.ast import CompoundParam, DirectionalCOD, PunishParam

    comp_dict: dict[str, Schedule] = {
        f"c{i}": c for i, c in enumerate(components)
    }

    cod = 0.0
    cor = 0
    if "COD" in params:
        cod_value = params["COD"]
        if isinstance(cod_value, DirectionalCOD):
            raise ScheduleConfigError(
                "directional COD (per-direction overrides) is not yet "
                "supported by the bridge"
            )
        if isinstance(cod_value, CompoundParam):
            cod = _scaled_time(
                cod_value.value, cod_value.time_unit, time_unit_seconds
            )
        else:
            raise ScheduleConfigError(
                f"unsupported COD parameter value: {type(cod_value).__name__}"
            )
    if "FRCO" in params:
        frco_value = params["FRCO"]
        if isinstance(frco_value, CompoundParam):
            cor = int(frco_value.value)
        else:
            raise ScheduleConfigError(
                f"unsupported FRCO parameter value: {type(frco_value).__name__}"
            )
    if "PUNISH" in params and isinstance(params["PUNISH"], PunishParam):
        raise ScheduleConfigError(
            "PUNISH parameter on Concurrent is not yet supported by the bridge"
        )
    return Concurrent(comp_dict, cod=cod, cor=cor)


def _from_modifier(node: Any, time_unit_seconds: float) -> Schedule:
    mod = node.modifier

    if mod == "DRL":
        if node.value is None:
            raise ScheduleConfigError("DRL requires a numeric value")
        return DRL(_scaled_time(node.value, node.time_unit, time_unit_seconds))

    if mod == "DRH":
        if node.value is None:
            raise ScheduleConfigError("DRH requires a numeric value")
        # The DSL's DRH carries a single time value interpreted as an
        # IRT threshold (e.g. "DRH2s" means "at least two responses
        # within 2 seconds"). We materialise it as the runtime's
        # 2-response sliding-window DRH with the DSL value as the
        # window length.
        window = _scaled_time(node.value, node.time_unit, time_unit_seconds)
        return DRH(2, window)

    if mod == "DRO":
        if node.value is None:
            raise ScheduleConfigError("DRO requires a numeric value")
        return DRO(
            _scaled_time(node.value, node.time_unit, time_unit_seconds),
            dro_type="resetting",
        )

    if mod == "PR":
        return _from_pr(node)

    if mod == "Repeat":
        if node.value is None or node.inner is None:
            raise ScheduleConfigError(
                "Repeat requires an integer value and an inner schedule"
            )
        count = int(node.value)
        if count < 2:
            raise ScheduleConfigError(
                f"Repeat requires a count >= 2, got {count}"
            )
        # Each copy is a freshly bridged inner; sharing one instance
        # across links would cause state entanglement.
        components = [
            from_dsl(node.inner, time_unit_seconds=time_unit_seconds)
            for _ in range(count)
        ]
        return Chained(
            components, stimuli=[f"s{i}" for i in range(count)]
        )

    if mod == "Pctl":
        raise ScheduleConfigError(
            "percentile (Pctl) schedules are not yet supported by the bridge"
        )

    raise ScheduleConfigError(f"unknown Modifier kind: {mod!r}")


def _from_pr(node: Any) -> ProgressiveRatio:
    step = node.pr_step
    start = int(node.pr_start) if node.pr_start is not None else 1
    if step in ("linear", "hodos", None):
        # ``hodos`` is a named linear preset; we also accept a bare PR
        # as linear(1, 1) for symmetry.
        increment = (
            int(node.pr_increment) if node.pr_increment is not None else 1
        )
        return ProgressiveRatio(arithmetic(start=start, step=increment))
    if step in ("geometric", "exponential"):
        ratio = (
            float(node.pr_ratio) if node.pr_ratio is not None else 2.0
        )
        return ProgressiveRatio(geometric(start=start, ratio=ratio))
    raise ScheduleConfigError(
        f"unknown PR pr_step: {step!r}"
    )
