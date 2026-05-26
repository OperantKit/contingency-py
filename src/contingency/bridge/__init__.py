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
    ...; optional ``cod`` and ``frco`` extracted from ``params``.
  * ``ALT`` -> :class:`Alternative`; >2 components are nested left-
    associative: ``Alt(Alt(a, b), c)``.
  * ``CONJ`` -> :class:`Conjunctive`; >2 components are nested left-
    associative like :class:`Alternative`.
  * ``MULT`` -> :class:`Multiple` with default stimuli ``s0, s1, ...``.
  * ``MIX`` -> :class:`Mixed` (no stimuli; integer ``current_component``
    reported in ``Outcome.meta``).
  * ``CHAIN`` -> :class:`Chained` with default stimuli ``s0, s1, ...``.
  * ``TAND`` -> :class:`Tandem`.
  * Remaining combinators (``OVERLAY``, ``INTERPOLATE``, ``INTERP``)
    raise :class:`ScheduleConfigError`.

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

``TimeoutWrapped`` -> :class:`Timeout` wrapping ``from_dsl(inner)`` with
``duration`` scaled to runtime seconds via the DSL's ``duration_unit``
and the caller's ``time_unit_seconds``; ``reset_on_response`` is
forwarded unchanged.

``ResponseCostWrapped`` -> :class:`ResponseCost` wrapping
``from_dsl(inner)`` with ``amount`` and ``unit`` forwarded from the
DSL's ``ResponseCost`` parameters.

``AversiveSchedule`` nodes (``kind`` in ``{"Sidman", "DiscrimAv",
"Escape"}``) translate to :class:`Sidman`, :class:`DiscriminatedAvoidance`,
and :class:`Escape` runtime schedules. DSL parameter names (``SSI``,
``RSI``, ``CSUSInterval``, ``ITI``, ``SafeDuration``, ``MaxShock``) are
mapped to runtime constructor kwargs inside :func:`_from_aversive`.

``IdentifierRef`` and the trial-based wrapper raise
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
    AdjustingSchedule,
    Alternative,
    Chained,
    Concurrent,
    Conjunctive,
    DiscriminatedAvoidance,
    Escape,
    GoNoGo,
    InterlockingSchedule,
    Interpolate,
    LimitedHold,
    MatchingToSample,
    Mixed,
    Multiple,
    Overlay,
    Percentile,
    ProgressiveRatio,
    ResponseCost,
    SecondOrder,
    Sidman,
    Tandem,
    Timeout,
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
    # surfaced only at bridge time. Rename DSL node classes that
    # collide with runtime schedule classes of the same name.
    from contingency_dsl.ast import (
        AdjustingSchedule as DSLAdjusting,
    )
    from contingency_dsl.ast import (
        AnnotatedSchedule,
        Atomic,
        AversiveSchedule,
        Compound,
        IdentifierRef,
        Modifier,
        Program,
        ResponseCostWrapped,
        Special,
        TimeoutWrapped,
        TrialBased,
    )
    from contingency_dsl.ast import InterlockingSchedule as DSLInterlocking
    from contingency_dsl.ast import LimitedHold as DSLLimitedHold
    from contingency_dsl.ast import SecondOrder as DSLSecondOrder

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
        return LimitedHold(inner, hold)  # type: ignore[arg-type]

    if isinstance(ast_node, DSLSecondOrder):
        overall_rt = from_dsl(
            ast_node.overall, time_unit_seconds=time_unit_seconds
        )
        unit_rt = from_dsl(
            ast_node.unit, time_unit_seconds=time_unit_seconds
        )
        return SecondOrder(overall_rt, unit_rt)

    if isinstance(ast_node, DSLAdjusting):
        return _from_adjusting(ast_node, time_unit_seconds)

    if isinstance(ast_node, DSLInterlocking):
        return _from_interlocking(ast_node, time_unit_seconds)

    if isinstance(ast_node, IdentifierRef):
        raise ScheduleConfigError(
            f"unresolved IdentifierRef({ast_node.name!r}); "
            "call contingency_dsl.analyze() on the Program first to "
            "substitute let-bindings before bridging"
        )

    if isinstance(ast_node, TimeoutWrapped):
        inner = from_dsl(ast_node.inner, time_unit_seconds=time_unit_seconds)
        duration = _scaled_time(
            ast_node.timeout.duration,
            ast_node.timeout.duration_unit,
            time_unit_seconds,
        )
        return Timeout(
            inner,
            duration,
            reset_on_response=ast_node.timeout.reset_on_response,
        )

    if isinstance(ast_node, ResponseCostWrapped):
        inner = from_dsl(ast_node.inner, time_unit_seconds=time_unit_seconds)
        return ResponseCost(
            inner,
            ast_node.response_cost.amount,
            unit=ast_node.response_cost.unit,
        )

    if isinstance(ast_node, AversiveSchedule):
        return _from_aversive(ast_node, time_unit_seconds)

    if isinstance(ast_node, TrialBased):
        return _from_trial_based(ast_node, time_unit_seconds)

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

    if combinator is Combinator.CONJ:
        if len(components) < 2:
            raise ScheduleConfigError(
                f"Conj requires >= 2 components, got {len(components)}"
            )
        # Left-associative nesting for >2 components, mirroring Alt.
        conj_acc = Conjunctive(components[0], components[1])
        for extra in components[2:]:
            conj_acc = Conjunctive(conj_acc, extra)
        return conj_acc

    if combinator is Combinator.MIX:
        return Mixed(components)

    if combinator is Combinator.OVERLAY:
        if len(components) != 2:
            raise ScheduleConfigError(
                f"Overlay requires exactly 2 components (base, punisher), "
                f"got {len(components)}"
            )
        return Overlay(components[0], components[1])

    if combinator in (Combinator.INTERPOLATE, Combinator.INTERP):
        return _build_interpolate(components, node.params or {}, time_unit_seconds)

    raise ScheduleConfigError(
        f"combinator {combinator.value!r} is not yet supported by the "
        "bridge"
    )


def _build_interpolate(
    components: list[Schedule],
    params: dict[str, Any],
    time_unit_seconds: float,
) -> Interpolate:
    """Construct :class:`Interpolate` from DSL params.

    The DSL exposes two keyword parameters relevant to Interpolate:

    * ``count`` — the number of probe windows planned for the session;
    * ``onset`` — the time from session start at which the first probe
      window opens.

    The runtime :class:`Interpolate` requires ``interval`` (period
    between probe starts) and ``probe_duration`` (length of each probe
    window), which are not first-class DSL concepts. When the DSL does
    not provide them, we fall back on reasonable defaults:

    * ``first_probe_at`` defaults to ``onset`` when present and
      ``interval`` otherwise;
    * ``interval`` defaults to ``onset`` when ``onset`` is provided,
      else the unit second;
    * ``probe_duration`` defaults to ``interval / 2``.

    These defaults let the bridge translate canonical DSL Interpolate
    forms to an executable runtime schedule; applications requiring
    precise window geometry should construct :class:`Interpolate`
    directly.
    """
    from contingency_dsl.ast import CompoundParam

    if len(components) != 2:
        raise ScheduleConfigError(
            f"Interpolate requires exactly 2 components, got "
            f"{len(components)}"
        )
    base, probe = components

    onset_seconds: float | None = None
    if "onset" in params:
        onset_val = params["onset"]
        if not isinstance(onset_val, CompoundParam):
            raise ScheduleConfigError(
                f"unsupported onset parameter value: "
                f"{type(onset_val).__name__}"
            )
        onset_seconds = _scaled_time(
            onset_val.value, onset_val.time_unit, time_unit_seconds
        )

    # "count" is currently informational; the runtime does not cap the
    # number of probe windows per session — it simply repeats at
    # ``interval`` forever.
    interval_seconds = onset_seconds if onset_seconds is not None else 1.0
    if interval_seconds <= 0:
        interval_seconds = 1.0
    probe_duration = interval_seconds / 2.0
    first_probe_at = onset_seconds

    return Interpolate(
        base,
        probe,
        interval=interval_seconds,
        probe_duration=probe_duration,
        first_probe_at=first_probe_at,
    )


def _build_concurrent(
    components: list[Schedule],
    params: dict[str, Any],
    time_unit_seconds: float,
) -> Concurrent:
    # Import lazily to avoid a hard dep at module import time.
    from contingency_dsl.ast import (
        CompoundParam,
        DirectionalCOD,
        DirectionalEntry,
        PunishParam,
    )

    comp_dict: dict[str, Schedule] = {
        f"c{i}": c for i, c in enumerate(components)
    }

    def _resolve_op(ref: int | str) -> str:
        """Resolve a 1-indexed int or identifier to the ``c<k>`` key."""
        if isinstance(ref, int):
            return f"c{ref - 1}"
        return str(ref)

    cod = 0.0
    frco = 0
    cod_directional: dict[tuple[str, str], float] | None = None
    punish: dict[str, Schedule] | None = None

    if "COD" in params:
        cod_value = params["COD"]
        if isinstance(cod_value, DirectionalCOD):
            if cod_value.base is not None:
                cod = _scaled_time(
                    cod_value.base.value,
                    cod_value.base.time_unit,
                    time_unit_seconds,
                )
            cod_directional = {}
            for entry in cod_value.directional:
                assert isinstance(entry, DirectionalEntry)
                from_op = _resolve_op(entry.from_ref)
                to_op = _resolve_op(entry.to_ref)
                cod_directional[(from_op, to_op)] = _scaled_time(
                    entry.value, entry.time_unit, time_unit_seconds
                )
        elif isinstance(cod_value, CompoundParam):
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
            frco = int(frco_value.value)
        else:
            raise ScheduleConfigError(
                f"unsupported FRCO parameter value: {type(frco_value).__name__}"
            )

    if "PUNISH" in params:
        punish_value = params["PUNISH"]
        # The parser may emit either a PunishParam (with a tuple payload
        # of key-value pairs) or a raw dict. Normalise to a dict.
        if isinstance(punish_value, PunishParam):
            punish_dict = dict(punish_value.payload)
        elif isinstance(punish_value, dict):
            punish_dict = punish_value
        else:
            raise ScheduleConfigError(
                f"unsupported PUNISH parameter value: "
                f"{type(punish_value).__name__}"
            )
        # Implemented: "component" form (punishment schedule per
        # operandum). Directional / changeover forms are deferred.
        unsupported = [k for k in punish_dict if k not in ("component",)]
        if unsupported:
            raise ScheduleConfigError(
                f"PUNISH form(s) {unsupported!r} are not yet supported "
                "by the bridge (only the component form is wired); "
                "directional and changeover punishment are deferred"
            )
        component_entries = punish_dict.get("component", [])
        if component_entries:
            punish = {}
            for entry in component_entries:
                ref = entry["component"]
                op = _resolve_op(ref)
                sched_spec = entry["schedule"]
                punish[op] = from_dsl(
                    sched_spec, time_unit_seconds=time_unit_seconds
                )

    return Concurrent(
        comp_dict,
        cod=cod,
        frco=frco,
        cod_directional=cod_directional,
        punish=punish,
    )


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
        return _from_pctl(node)

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


def _from_pctl(node: Any) -> Percentile:
    """Translate a ``Pctl`` :class:`Modifier` to a :class:`Percentile`.

    The DSL parser enforces rank ∈ ``[0, 100]`` and a closed set of
    target dimensions. The bridge mirrors these checks defensively —
    callers may construct :class:`Modifier` nodes directly, bypassing
    the parser.
    """
    if node.pctl_target is None:
        raise ScheduleConfigError("Pctl requires a target dimension")
    if node.pctl_rank is None:
        raise ScheduleConfigError("Pctl requires a rank")
    # The DSL parser defaults window to 20 and dir to "below" when either
    # is omitted; be tolerant if a hand-built Modifier leaves them as None.
    window = 20 if node.pctl_window is None else int(node.pctl_window)
    direction = "below" if node.pctl_dir is None else str(node.pctl_dir)
    return Percentile(
        target=str(node.pctl_target),
        rank=int(node.pctl_rank),
        window=window,
        direction=direction,
    )


def _adj_numeric(
    payload: dict[str, Any] | None,
    time_unit_seconds: float,
    *,
    temporal: bool,
) -> float | None:
    """Resolve an ``{"value", "time_unit"}`` DSL payload to a runtime scalar.

    ``temporal=True`` applies the time-unit and runtime-clock factors;
    otherwise the raw numeric value is returned verbatim.
    """
    if payload is None:
        return None
    value = payload["value"]
    time_unit = payload.get("time_unit")
    if temporal:
        return _scaled_time(float(value), time_unit, time_unit_seconds)
    # Dimensionless targets ("ratio", "amount") — the AST layer rejects
    # a non-None time_unit, but we assert here for defence in depth.
    if time_unit is not None:
        raise ScheduleConfigError(
            f"AdjustingSchedule dimensionless payload carries time_unit="
            f"{time_unit!r}; expected None"
        )
    return float(value)


def _from_adjusting(node: Any, time_unit_seconds: float) -> AdjustingSchedule:
    target = node.adj_target
    temporal = target in ("interval", "delay")
    start = _adj_numeric(node.adj_start, time_unit_seconds, temporal=temporal)
    step = _adj_numeric(node.adj_step, time_unit_seconds, temporal=temporal)
    minimum = _adj_numeric(node.adj_min, time_unit_seconds, temporal=temporal)
    maximum = _adj_numeric(node.adj_max, time_unit_seconds, temporal=temporal)
    # The DSL grammar enforces start and step; guard defensively against
    # hand-built AST nodes that omit them.
    if start is None:
        raise ScheduleConfigError("AdjustingSchedule requires adj_start")
    if step is None:
        raise ScheduleConfigError("AdjustingSchedule requires adj_step")
    return AdjustingSchedule(
        target=target,
        start=start,
        step=step,
        minimum=minimum,
        maximum=maximum,
    )


def _from_interlocking(
    node: Any, time_unit_seconds: float
) -> InterlockingSchedule:
    t_payload = node.interlock_T
    decay = _scaled_time(
        float(t_payload["value"]),
        t_payload.get("time_unit"),
        time_unit_seconds,
    )
    return InterlockingSchedule(
        initial_ratio=int(node.interlock_R0),
        decay_time=decay,
    )


def _aversive_time_param(
    params: dict[str, Any], key: str, time_unit_seconds: float
) -> float:
    """Extract and scale a ``{"value", "time_unit"}`` aversive param."""
    if key not in params:
        raise ScheduleConfigError(
            f"AversiveSchedule requires parameter {key!r}"
        )
    payload = params[key]
    if not isinstance(payload, dict):
        raise ScheduleConfigError(
            f"AversiveSchedule parameter {key!r} must be a dict, got "
            f"{type(payload).__name__}"
        )
    return _scaled_time(
        float(payload["value"]),
        payload.get("time_unit"),
        time_unit_seconds,
    )


def _aversive_magnitude(params: dict[str, Any]) -> float:
    """Extract ``magnitude`` from aversive params, defaulting to 1.0."""
    payload = params.get("magnitude")
    if payload is None:
        return 1.0
    if isinstance(payload, dict):
        return float(payload["value"])
    if isinstance(payload, (int, float)) and not isinstance(payload, bool):
        return float(payload)
    raise ScheduleConfigError(
        f"AversiveSchedule 'magnitude' has unsupported form: "
        f"{type(payload).__name__}"
    )


def _from_aversive(node: Any, time_unit_seconds: float) -> Schedule:
    """Translate an ``AversiveSchedule`` AST node to a runtime schedule.

    Parameter-name mapping from DSL to runtime:

    * Sidman:    ``SSI`` -> ``shock_shock_interval``,
                 ``RSI`` -> ``response_shock_interval``.
    * DiscrimAv: ``CSUSInterval`` -> ``warning_duration``,
                 ``ITI`` -> ``iti``. ``mode`` and ``ShockDuration`` /
                 ``MaxShock`` are accepted but not currently modelled
                 by the runtime (the runtime always terminates the
                 warning on response and delivers a punctate shock on
                 timeout).
    * Escape:    ``SafeDuration`` -> ``iti``. ``MaxShock`` (if given)
                 maps to ``trial_duration``; otherwise defaults to a
                 large value (effectively no ceiling).
    """
    params = node.params or {}
    kind = node.kind

    if kind == "Sidman":
        return Sidman(
            shock_shock_interval=_aversive_time_param(
                params, "SSI", time_unit_seconds
            ),
            response_shock_interval=_aversive_time_param(
                params, "RSI", time_unit_seconds
            ),
            shock_magnitude=_aversive_magnitude(params),
        )

    if kind == "DiscrimAv":
        return DiscriminatedAvoidance(
            warning_duration=_aversive_time_param(
                params, "CSUSInterval", time_unit_seconds
            ),
            iti=_aversive_time_param(params, "ITI", time_unit_seconds),
            shock_magnitude=_aversive_magnitude(params),
        )

    if kind == "Escape":
        iti_seconds = _aversive_time_param(
            params, "SafeDuration", time_unit_seconds
        )
        # ``MaxShock`` optionally bounds the failed-trial duration; if
        # absent, fall back to a very large value (treated as "no
        # ceiling" for practical purposes).
        if "MaxShock" in params:
            trial_duration = _aversive_time_param(
                params, "MaxShock", time_unit_seconds
            )
        else:
            trial_duration = 1e9
        return Escape(
            trial_duration=trial_duration,
            iti=iti_seconds,
            shock_magnitude=_aversive_magnitude(params),
        )

    raise ScheduleConfigError(f"unknown aversive kind: {kind!r}")


# --- Trial-based schedules -------------------------------------------------


def _trial_based_time(
    params: dict[str, Any],
    value_key: str,
    unit_key: str,
    time_unit_seconds: float,
    *,
    default: float | None = None,
) -> float | None:
    """Resolve ``(value, unit)`` pair from a ``TrialBased.params`` dict."""
    if value_key not in params:
        return default
    value = params[value_key]
    unit = params.get(unit_key)
    return _scaled_time(float(value), unit, time_unit_seconds)


def _from_trial_based(node: Any, time_unit_seconds: float) -> Schedule:
    """Translate a ``TrialBased`` AST node to a runtime schedule."""
    trial_type = node.trial_type
    params = dict(node.params)

    if trial_type == "MTS":
        return _build_mts(params, time_unit_seconds)
    if trial_type == "GoNoGo":
        return _build_go_nogo(params, time_unit_seconds)
    raise ScheduleConfigError(f"unknown TrialBased trial_type: {trial_type!r}")


def _bridge_nested_schedule(
    raw: Any, time_unit_seconds: float
) -> Schedule | None:
    """Bridge a nested consequence/incorrect slot, preserving ``None``."""
    if raw is None:
        return None
    return from_dsl(raw, time_unit_seconds=time_unit_seconds)


def _build_mts(
    params: dict[str, Any], time_unit_seconds: float
) -> MatchingToSample:
    comparisons = params.get("comparisons")
    if comparisons is None:
        raise ScheduleConfigError(
            "MTS requires a 'comparisons' parameter"
        )
    iti_seconds = _trial_based_time(
        params, "ITI", "ITI_unit", time_unit_seconds, default=0.0
    )
    # DMTS delay: sample_duration is treated as the pre-choice interval.
    sample_duration = _trial_based_time(
        params, "delay", "delay_unit", time_unit_seconds, default=0.0
    )
    # Optional LH on MTS scopes the choice window.
    choice_timeout = _trial_based_time(
        params, "limitedHold", "limitedHoldUnit", time_unit_seconds
    )
    if choice_timeout is None:
        choice_timeout = 5.0
    consequence = _bridge_nested_schedule(
        params.get("consequence"), time_unit_seconds
    )
    incorrect = _bridge_nested_schedule(
        params.get("incorrect"), time_unit_seconds
    )
    assert iti_seconds is not None
    return MatchingToSample(
        n_comparisons=int(comparisons),
        sample_duration=sample_duration or 0.0,
        choice_timeout=choice_timeout,
        consequence=consequence,
        incorrect=incorrect,
        iti=iti_seconds,
    )


def _build_go_nogo(
    params: dict[str, Any], time_unit_seconds: float
) -> GoNoGo:
    response_window = _trial_based_time(
        params,
        "responseWindow",
        "responseWindowUnit",
        time_unit_seconds,
    )
    if response_window is None:
        raise ScheduleConfigError(
            "GoNoGo requires a 'responseWindow' parameter"
        )
    iti_seconds = _trial_based_time(
        params, "ITI", "ITI_unit", time_unit_seconds, default=0.0
    )
    assert iti_seconds is not None
    consequence = _bridge_nested_schedule(
        params.get("consequence"), time_unit_seconds
    )
    incorrect = _bridge_nested_schedule(
        params.get("incorrect"), time_unit_seconds
    )
    false_alarm = _bridge_nested_schedule(
        params.get("falseAlarm"), time_unit_seconds
    )
    # ``consequence`` = reinforcement for correct Go;
    # ``incorrect``   = reinforcement for correct NoGo withhold (miss/
    #                   withhold consequence slot); when EXT this means
    #                   "no reinforcement for withholding" which maps to
    #                   correct_nogo_schedule=EXT (always non-reinforcing).
    # ``falseAlarm``  = NoGo response punishment.
    return GoNoGo(
        response_window=response_window,
        iti=iti_seconds,
        correct_go_schedule=consequence,
        correct_nogo_schedule=incorrect,
        false_alarm=false_alarm,
    )
