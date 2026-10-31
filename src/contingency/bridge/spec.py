"""Translate contingency-dsl AST nodes to schedule specs.

:func:`to_spec` returns the JSON-serialisable ``{"type", "params"}``
object that the conformance corpus uses (see
:func:`contingency.conformance.spec.build_schedule_from_spec`). For
every AST that :func:`to_spec` accepts,
``build_schedule_from_spec(to_spec(ast))`` behaves like
``from_dsl(ast)``: the same values, defaults, seeds and component names.

Spec conventions
----------------

* Times are plain numbers in the caller's clock unit (seconds by
  default); ``time_unit`` and ``time_unit_seconds`` are applied here,
  as in :func:`from_dsl`.
* Whole-number parameters (``n``, ``n_intervals``, ``seed``, ``frco``,
  ``rank``, ``window``, ``start`` / ``step_size`` of PR,
  ``initial_ratio``, ``n_comparisons``, ``response_count``) are JSON
  integers.
* An optional value that is absent is left out rather than written as
  ``null``. ``seed`` is written only when ``@algorithm(seed=...)`` is in
  scope.
* Values the DSL fixes but the spec defaults differently are written
  explicitly: ``Percentile.direction`` (DSL default ``below``, spec
  default ``above``) and the ``iti`` of ``MatchingToSample`` /
  ``GoNoGo`` (bridge default ``0.0``).
* ``Percentile.target`` is lower-case ``"irt"``.
* ``Concurrent`` keys are ``c0``, ``c1``, ...; ``Multiple`` / ``Chained``
  stimuli are ``s0``, ``s1``, ...

Lossy mappings kept from :func:`from_dsl`
-----------------------------------------

* ``Interpolate(count=, onset=)``: ``count`` is dropped and the window
  geometry is fitted from ``onset`` (``interval = onset``,
  ``probe_duration = interval / 2``, ``first_probe_at = onset``).
* ``DiscrimAv``: ``mode``, ``ShockDuration`` and ``MaxShock`` are
  dropped; the runtime terminates the warning stimulus on a response and
  delivers a punctate shock on timeout.
* ``MTS``: ``mts_type`` is dropped; the runtime does not distinguish
  identity from arbitrary matching.

Constructs without a spec form
------------------------------

These raise :class:`SpecUnrepresentableError` (a
:class:`ScheduleConfigError`) instead of being dropped:

* program-level parameter declarations (``LH``, ``COD``, ``FRCO``,
  ``BO``, ``RD``) — run ``contingency_dsl.analyze()`` to apply ``LH``,
  and write ``COD`` / ``FRCO`` on the ``Conc`` itself;
* ``Conc`` with a directional ``COD(a->b)`` or any ``PUNISH`` form;
* ``Mult`` / ``Mix`` with ``BO``; ``Overlay`` with ``target=``;
* any other combinator parameter the spec has no key for;
* ``Lag``;
* ``Pctl`` with a target other than ``IRT``;
* ``Adj`` with target ``delay`` or ``amount``;
* ``MTS`` with ``correction`` enabled, and the declaration form
  ``MTS(samples=...)``;
* ``GoNoGo`` with ``LH``.

Errors that :func:`from_dsl` raises (unresolved ``IdentifierRef``,
malformed values, ``@algorithm`` misuse) are raised here too.
"""

from __future__ import annotations

from typing import Any

from contingency.bridge import (
    _adj_numeric,
    _algorithm,
    _algorithm_scope,
    _aversive_magnitude,
    _aversive_time_param,
    _fleshler_hoffman_kwargs,
    _next_seed,
    _scaled_time,
    _trial_based_time,
)
from contingency.errors import ScheduleConfigError, SpecUnrepresentableError

__all__ = ["to_spec"]

Spec = dict[str, Any]


def _spec(stype: str, **params: Any) -> Spec:
    return {"type": stype, "params": {k: v for k, v in params.items() if v is not None}}


def to_spec(ast_node: Any, *, time_unit_seconds: float = 1.0) -> Spec:
    """Translate a contingency-dsl AST node to a schedule spec.

    Parameters
    ----------
    ast_node:
        A ``contingency_dsl`` AST node or ``Program``, as accepted by
        :func:`from_dsl`.
    time_unit_seconds:
        Conversion factor from DSL seconds to the caller's runtime
        clock unit, as in :func:`from_dsl`.

    Returns
    -------
    dict
        A JSON-serialisable ``{"type": ..., "params": {...}}`` spec.

    Raises
    ------
    SpecUnrepresentableError
        When the AST contains a construct with no spec form (see the
        module docstring).
    ScheduleConfigError
        When :func:`from_dsl` would reject the AST.
    """
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

    def recurse(node: Any) -> Spec:
        return to_spec(node, time_unit_seconds=time_unit_seconds)

    def t(value: float, time_unit: str | None) -> float:
        return float(_scaled_time(value, time_unit, time_unit_seconds))

    if isinstance(ast_node, Program):
        if ast_node.param_decls:
            names = [d.name for d in ast_node.param_decls]
            raise SpecUnrepresentableError(
                f"program-level parameter declaration(s) {names!r} have no spec "
                "form; call contingency_dsl.analyze() to apply LH, and write "
                "COD / FRCO on the Conc itself"
            )
        with _algorithm(_algorithm_scope(ast_node.program_annotations)):
            return recurse(ast_node.schedule)

    if isinstance(ast_node, AnnotatedSchedule):
        with _algorithm(_algorithm_scope(ast_node.annotations)):
            return recurse(ast_node.expr)

    if isinstance(ast_node, Atomic):
        return _atomic_spec(ast_node, t)

    if isinstance(ast_node, Special):
        if ast_node.kind in ("EXT", "CRF"):
            return _spec(ast_node.kind)
        raise ScheduleConfigError(f"unknown Special kind: {ast_node.kind!r}")

    if isinstance(ast_node, Compound):
        return _compound_spec(ast_node, recurse, t)

    if isinstance(ast_node, Modifier):
        return _modifier_spec(ast_node, recurse, t)

    if isinstance(ast_node, DSLLimitedHold):
        return _spec(
            "LimitedHold",
            inner=recurse(ast_node.inner),
            hold=t(ast_node.hold_duration, ast_node.time_unit),
        )

    if isinstance(ast_node, DSLSecondOrder):
        return _spec("SecondOrder", overall=recurse(ast_node.overall), unit=recurse(ast_node.unit))

    if isinstance(ast_node, DSLAdjusting):
        return _adjusting_spec(ast_node, time_unit_seconds)

    if isinstance(ast_node, DSLInterlocking):
        t_payload = ast_node.interlock_T
        return _spec(
            "InterlockingSchedule",
            initial_ratio=int(ast_node.interlock_R0),
            decay_time=t(float(t_payload["value"]), t_payload.get("time_unit")),
        )

    if isinstance(ast_node, IdentifierRef):
        raise ScheduleConfigError(
            f"unresolved IdentifierRef({ast_node.name!r}); "
            "call contingency_dsl.analyze() on the Program first to "
            "substitute let-bindings before bridging"
        )

    if isinstance(ast_node, TimeoutWrapped):
        return _spec(
            "Timeout",
            inner=recurse(ast_node.inner),
            duration=t(ast_node.timeout.duration, ast_node.timeout.duration_unit),
            reset_on_response=bool(ast_node.timeout.reset_on_response),
        )

    if isinstance(ast_node, ResponseCostWrapped):
        return _spec(
            "ResponseCost",
            inner=recurse(ast_node.inner),
            amount=float(ast_node.response_cost.amount),
            unit=str(ast_node.response_cost.unit),
        )

    if isinstance(ast_node, AversiveSchedule):
        return _aversive_spec(ast_node, time_unit_seconds)

    if isinstance(ast_node, TrialBased):
        return _trial_based_spec(ast_node, recurse, time_unit_seconds)

    raise ScheduleConfigError(f"unrecognised AST node: {type(ast_node).__name__}")


# --- Per-node translators --------------------------------------------------


def _atomic_spec(node: Any, t: Any) -> Spec:
    from contingency_dsl.enums import Distribution, Domain

    dist = node.dist
    domain = node.domain
    value = node.value

    if dist is Distribution.FIXED and domain is Domain.RATIO:
        if value != int(value) or value < 1:
            raise ScheduleConfigError(f"FR requires a positive integer ratio, got {value!r}")
        return _spec("FR", n=int(value))
    if dist is Distribution.FIXED and domain is Domain.INTERVAL:
        return _spec("FI", interval=t(value, node.time_unit))
    if dist is Distribution.FIXED and domain is Domain.TIME:
        return _spec("FT", interval=t(value, node.time_unit))
    if dist is Distribution.VARIABLE and domain is Domain.RATIO:
        return _spec("VR", mean=float(value), **_fleshler_hoffman_kwargs("VR"))
    if dist is Distribution.VARIABLE and domain is Domain.INTERVAL:
        return _spec("VI", mean_interval=t(value, node.time_unit), **_fleshler_hoffman_kwargs("VI"))
    if dist is Distribution.VARIABLE and domain is Domain.TIME:
        return _spec("VT", mean_interval=t(value, node.time_unit), **_fleshler_hoffman_kwargs("VT"))
    if dist is Distribution.RANDOM and domain is Domain.RATIO:
        if not (value > 0):
            raise ScheduleConfigError(
                f"RR requires a positive mean-value to convert to probability, got {value!r}"
            )
        return _spec("RR", probability=1.0 / float(value), seed=_next_seed())
    if dist is Distribution.RANDOM and domain is Domain.INTERVAL:
        return _spec("RI", mean_interval=t(value, node.time_unit), seed=_next_seed())
    if dist is Distribution.RANDOM and domain is Domain.TIME:
        return _spec("RT", mean_interval=t(value, node.time_unit), seed=_next_seed())

    raise ScheduleConfigError(f"unhandled atomic combination: dist={dist!r} domain={domain!r}")


_COMPOUND_PARAMS: dict[str, frozenset[str]] = {
    "Conc": frozenset({"COD", "FRCO"}),
    "Interpolate": frozenset({"count", "onset"}),
    "Interp": frozenset({"count", "onset"}),
}


def _compound_spec(node: Any, recurse: Any, t: Any) -> Spec:
    from contingency_dsl.ast import CompoundParam, DirectionalCOD
    from contingency_dsl.enums import Combinator

    combinator = node.combinator
    components = [recurse(c) for c in node.components]
    params = node.params or {}

    unsupported = sorted(set(params) - _COMPOUND_PARAMS.get(combinator.value, frozenset()))
    if unsupported:
        raise SpecUnrepresentableError(
            f"{combinator.value} parameter(s) {unsupported!r} have no spec form"
        )

    if combinator is Combinator.CONC:
        cod = 0.0
        frco = 0
        if "COD" in params:
            cod_value = params["COD"]
            if isinstance(cod_value, DirectionalCOD):
                if cod_value.directional:
                    raise SpecUnrepresentableError("directional COD (COD(a->b)) has no spec form")
                if cod_value.base is not None:
                    cod = t(cod_value.base.value, cod_value.base.time_unit)
            elif isinstance(cod_value, CompoundParam):
                cod = t(cod_value.value, cod_value.time_unit)
            else:
                raise ScheduleConfigError(
                    f"unsupported COD parameter value: {type(cod_value).__name__}"
                )
        if "FRCO" in params:
            frco_value = params["FRCO"]
            if not isinstance(frco_value, CompoundParam):
                raise ScheduleConfigError(
                    f"unsupported FRCO parameter value: {type(frco_value).__name__}"
                )
            frco = int(frco_value.value)
        return _spec(
            "Concurrent",
            components={f"c{i}": c for i, c in enumerate(components)},
            cod=cod,
            frco=frco,
        )

    if combinator in (Combinator.ALT, Combinator.CONJ):
        name = "Alt" if combinator is Combinator.ALT else "Conj"
        stype = "Alternative" if combinator is Combinator.ALT else "Conjunctive"
        if len(components) < 2:
            raise ScheduleConfigError(f"{name} requires >= 2 components, got {len(components)}")
        acc = _spec(stype, first=components[0], second=components[1])
        for extra in components[2:]:
            acc = _spec(stype, first=acc, second=extra)
        return acc

    if combinator in (Combinator.MULT, Combinator.CHAIN):
        stype = "Multiple" if combinator is Combinator.MULT else "Chained"
        return _spec(
            stype,
            components=components,
            stimuli=[f"s{i}" for i in range(len(components))],
        )

    if combinator is Combinator.TAND:
        return _spec("Tandem", components=components)

    if combinator is Combinator.MIX:
        return _spec("Mixed", components=components)

    if combinator is Combinator.OVERLAY:
        if len(components) != 2:
            raise ScheduleConfigError(
                f"Overlay requires exactly 2 components (base, punisher), got {len(components)}"
            )
        return _spec("Overlay", base=components[0], punisher=components[1])

    if combinator in (Combinator.INTERPOLATE, Combinator.INTERP):
        if len(components) != 2:
            raise ScheduleConfigError(
                f"Interpolate requires exactly 2 components, got {len(components)}"
            )
        onset: float | None = None
        if "onset" in params:
            onset_val = params["onset"]
            if not isinstance(onset_val, CompoundParam):
                raise ScheduleConfigError(
                    f"unsupported onset parameter value: {type(onset_val).__name__}"
                )
            onset = t(onset_val.value, onset_val.time_unit)
        interval = onset if onset is not None else 1.0
        if interval <= 0:
            interval = 1.0
        return _spec(
            "Interpolate",
            base=components[0],
            probe=components[1],
            interval=interval,
            probe_duration=interval / 2.0,
            first_probe_at=onset,
        )

    raise ScheduleConfigError(f"combinator {combinator.value!r} is not yet supported by the bridge")


def _modifier_spec(node: Any, recurse: Any, t: Any) -> Spec:
    mod = node.modifier

    if mod in ("DRL", "DRH", "DRO"):
        if node.value is None:
            raise ScheduleConfigError(f"{mod} requires a numeric value")
        time = t(node.value, node.time_unit)
        if mod == "DRL":
            return _spec("DRL", interval=time)
        if mod == "DRH":
            # Same reading as from_dsl: the DSL value is an IRT threshold,
            # i.e. two responses within the window.
            return _spec("DRH", response_count=2, time_window=time)
        return _spec("DRO", interval=time, dro_type="resetting")

    if mod == "PR":
        step = node.pr_step
        start = int(node.pr_start) if node.pr_start is not None else 1
        if step in ("linear", "hodos", None):
            increment = int(node.pr_increment) if node.pr_increment is not None else 1
            return _spec("ProgressiveRatio", step="arithmetic", start=start, step_size=increment)
        if step in ("geometric", "exponential"):
            ratio = float(node.pr_ratio) if node.pr_ratio is not None else 2.0
            return _spec("ProgressiveRatio", step="geometric", start=start, ratio=ratio)
        raise ScheduleConfigError(f"unknown PR pr_step: {step!r}")

    if mod == "Repeat":
        if node.value is None or node.inner is None:
            raise ScheduleConfigError("Repeat requires an integer value and an inner schedule")
        count = int(node.value)
        if count < 2:
            raise ScheduleConfigError(f"Repeat requires a count >= 2, got {count}")
        return _spec(
            "Chained",
            components=[recurse(node.inner) for _ in range(count)],
            stimuli=[f"s{i}" for i in range(count)],
        )

    if mod == "Pctl":
        if node.pctl_target is None:
            raise ScheduleConfigError("Pctl requires a target dimension")
        if node.pctl_rank is None:
            raise ScheduleConfigError("Pctl requires a rank")
        if str(node.pctl_target).lower() != "irt":
            raise SpecUnrepresentableError(
                f"Pctl target {node.pctl_target!r} has no spec form; only IRT is supported"
            )
        return _spec(
            "Percentile",
            target="irt",
            rank=int(node.pctl_rank),
            window=20 if node.pctl_window is None else int(node.pctl_window),
            direction="below" if node.pctl_dir is None else str(node.pctl_dir),
        )

    if mod == "Lag":
        raise SpecUnrepresentableError("Lag has no spec form (no runtime schedule)")

    raise ScheduleConfigError(f"unknown Modifier kind: {mod!r}")


def _adjusting_spec(node: Any, time_unit_seconds: float) -> Spec:
    target = node.adj_target
    if target not in ("ratio", "interval"):
        raise SpecUnrepresentableError(
            f"Adj target {target!r} has no spec form; the spec supports 'ratio' and 'interval'"
        )
    temporal = target == "interval"
    start = _adj_numeric(node.adj_start, time_unit_seconds, temporal=temporal)
    step = _adj_numeric(node.adj_step, time_unit_seconds, temporal=temporal)
    if start is None:
        raise ScheduleConfigError("AdjustingSchedule requires adj_start")
    if step is None:
        raise ScheduleConfigError("AdjustingSchedule requires adj_step")
    return _spec(
        "AdjustingSchedule",
        target=target,
        start=start,
        step=step,
        minimum=_adj_numeric(node.adj_min, time_unit_seconds, temporal=temporal),
        maximum=_adj_numeric(node.adj_max, time_unit_seconds, temporal=temporal),
    )


def _aversive_spec(node: Any, time_unit_seconds: float) -> Spec:
    params = node.params or {}
    kind = node.kind

    def time(key: str) -> float:
        return float(_aversive_time_param(params, key, time_unit_seconds))

    if kind == "Sidman":
        return _spec(
            "Sidman",
            shock_shock_interval=time("SSI"),
            response_shock_interval=time("RSI"),
            shock_magnitude=_aversive_magnitude(params),
        )
    if kind == "DiscrimAv":
        return _spec(
            "DiscriminatedAvoidance",
            warning_duration=time("CSUSInterval"),
            iti=time("ITI"),
            shock_magnitude=_aversive_magnitude(params),
        )
    if kind == "Escape":
        iti = time("SafeDuration")
        trial_duration = time("MaxShock") if "MaxShock" in params else 1e9
        return _spec(
            "Escape",
            trial_duration=trial_duration,
            iti=iti,
            shock_magnitude=_aversive_magnitude(params),
        )
    raise ScheduleConfigError(f"unknown aversive kind: {kind!r}")


def _trial_based_spec(node: Any, recurse: Any, time_unit_seconds: float) -> Spec:
    trial_type = node.trial_type
    params = dict(node.params)

    def time(value_key: str, unit_key: str, default: float | None = None) -> float | None:
        value = _trial_based_time(params, value_key, unit_key, time_unit_seconds, default=default)
        return float(value) if value is not None else None

    def nested(key: str) -> Spec | None:
        raw = params.get(key)
        return recurse(raw) if raw is not None else None

    if trial_type == "MTS":
        if "samples" in params:
            raise SpecUnrepresentableError(
                "declaration-form MTS(samples=...) describes trial structure, "
                "not a running schedule, and has no spec form"
            )
        correction = params.get("correction")
        if correction is not None and correction.get("mode") != "disabled":
            raise SpecUnrepresentableError("MTS correction trials have no spec form")
        comparisons = params.get("comparisons")
        if comparisons is None:
            raise ScheduleConfigError("MTS requires a 'comparisons' parameter")
        iti = time("ITI", "ITI_unit", 0.0)
        sample_duration = time("delay", "delay_unit", 0.0)
        choice_timeout = time("limitedHold", "limitedHoldUnit")
        consequence = nested("consequence")
        incorrect = nested("incorrect")
        return _spec(
            "MatchingToSample",
            n_comparisons=int(comparisons),
            sample_duration=sample_duration or 0.0,
            choice_timeout=choice_timeout if choice_timeout is not None else 5.0,
            consequence=consequence,
            incorrect=incorrect,
            iti=iti,
            seed=_next_seed(),
        )

    if trial_type == "GoNoGo":
        if "limitedHold" in params:
            raise SpecUnrepresentableError(
                "GoNoGo LH has no spec form; responseWindow bounds the response"
            )
        response_window = time("responseWindow", "responseWindowUnit")
        if response_window is None:
            raise ScheduleConfigError("GoNoGo requires a 'responseWindow' parameter")
        iti = time("ITI", "ITI_unit", 0.0)
        consequence = nested("consequence")
        incorrect = nested("incorrect")
        false_alarm = nested("falseAlarm")
        return _spec(
            "GoNoGo",
            go_probability=0.5,
            response_window=response_window,
            iti=iti,
            correct_go_schedule=consequence,
            correct_nogo_schedule=incorrect,
            false_alarm=false_alarm,
            seed=_next_seed(),
        )

    raise ScheduleConfigError(f"unknown TrialBased trial_type: {trial_type!r}")
