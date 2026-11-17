"""Trial-based (discrete-trial) schedules.

Free-operant schedules (FR, VI, VR, ...) treat responses as a continuous
stream. Trial-based schedules impose a discrete structure in which
behavior is organised into trials, each with an explicit begin, response
opportunity, consequence, and inter-trial interval.

Two canonical trial-based paradigms are implemented:

* :class:`MatchingToSample` (MTS) — after a sample stimulus is presented,
  the subject chooses among comparison stimuli; the choice that matches
  the sample is correct.
* :class:`GoNoGo` — on each trial a Go or NoGo stimulus is presented;
  responding on Go is correct, withholding on NoGo is correct.

Both conform to the ``Schedule`` Protocol (``step(now, event) -> Outcome``)
so they compose with existing free-operant wrappers. Time is
unit-agnostic; the caller declares the monotonic clock unit.

Design notes
------------
* Both classes run on the Rust engine (``contingency_core``); they keep
  their arguments and read their public state (``phase``,
  ``trial_index`` and the like) from the engine's ``state()``.
* The state machines are driven by two kinds of input: ticks (calls to
  ``step`` with ``event=None``) that advance the internal clock, and
  response events that are evaluated during response-eligible phases.
* During inter-trial intervals (ITI) and sample presentation, responses
  are ignored but ticks drive phase transitions.
* Consequence delivery is delegated to a nested schedule: a correct
  choice fires a synthetic response event into the consequence
  schedule, returning whatever outcome it yields (typically an immediate
  reinforcer when the consequence is ``FR(1)``/CRF). A nested schedule
  must be a schedule of this package that runs on the Rust engine; the
  engine runs its own copy, built from the nested schedule's spec, so
  the object passed in is not stepped.
* The ``meta`` payload on the returned ``Outcome`` always reports the
  current phase, trial index, and (when applicable) the outcome label of
  the completed trial. The meta of the nested schedule is not copied.

References
----------
Cook, L., & Cloak, N. L. (1956). Reinforcement schedules and some
behavioral effects. *Journal of the Experimental Analysis of Behavior*
(precursor literature; see also Nevin, 1969).

Cumming, W. W., & Berryman, R. (1965). The complex discriminated
operant: Studies of matching-to-sample and related problems. In D. I.
Mostofsky (Ed.), *Stimulus Generalization* (pp. 284-330). Stanford
University Press.

Nevin, J. A. (1969). Signal detection theory and operant behavior: A
review of D. M. Green and J. A. Swets' Signal Detection Theory and
Psychophysics. *Journal of the Experimental Analysis of Behavior*,
12(3), 475-480. https://doi.org/10.1901/jeab.1969.12-475
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from contingency.entities import Outcome, ResponseEvent
from contingency.errors import ScheduleConfigError
from contingency.interfaces import Schedule
from contingency.schedules._rust import (
    component_specs,
    compound_schedule,
    core_too_old,
    fresh_seed,
    rust_schedule,
)

__all__ = ["MatchingToSample", "GoNoGo", "TrialType", "trial_types_from_relations"]

def _require_nonnegative(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScheduleConfigError(
            f"{name} must be a non-negative number, got {type(value).__name__}"
        )
    if value < 0:
        raise ScheduleConfigError(f"{name} must be >= 0, got {value!r}")
    return float(value)


def _require_positive(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScheduleConfigError(
            f"{name} must be a positive number, got {type(value).__name__}"
        )
    if not (value > 0):
        raise ScheduleConfigError(f"{name} must be > 0, got {value!r}")
    return float(value)


@dataclass(frozen=True)
class TrialType:
    """One matching-to-sample trial type.

    Parameters
    ----------
    sample:
        Name of the sample stimulus.
    correct:
        Name of the comparison stimulus whose selection is correct.
    comparisons:
        Names of all comparison stimuli presented on the trial; it must
        contain ``correct``. Their positions (``choice_0`` ...) are
        shuffled at the start of each trial.
    """

    sample: str
    correct: str
    comparisons: tuple[str, ...]


def trial_types_from_relations(relations: Mapping[str, str]) -> tuple[TrialType, ...]:
    """Build one :class:`TrialType` per ``sample -> correct`` relation.

    Every trial type presents the same comparisons: the values of
    ``relations`` in insertion order. ``{"A1": "B1", "A2": "B2"}`` gives
    ``TrialType("A1", "B1", ("B1", "B2"))`` and
    ``TrialType("A2", "B2", ("B1", "B2"))``.

    Raises
    ------
    ScheduleConfigError
        If two samples share a correct comparison.
    """
    comparisons = tuple(relations.values())
    if len(set(comparisons)) != len(comparisons):
        raise ScheduleConfigError(
            f"relations must map each sample to a distinct comparison, got {dict(relations)!r}"
        )
    return tuple(
        TrialType(sample=sample, correct=correct, comparisons=comparisons)
        for sample, correct in relations.items()
    )


def _validate_trial_types(
    name: str, trial_types: Sequence[TrialType], n_comparisons: int
) -> tuple[TrialType, ...]:
    result = tuple(trial_types)
    for tt in result:
        if not isinstance(tt, TrialType):
            raise ScheduleConfigError(
                f"{name} must contain TrialType values, got {type(tt).__name__}"
            )
        comparisons = tuple(tt.comparisons)
        if len(comparisons) != n_comparisons:
            raise ScheduleConfigError(
                f"{name}: {tt!r} has {len(comparisons)} comparisons, "
                f"n_comparisons is {n_comparisons}"
            )
        if len(set(comparisons)) != len(comparisons):
            raise ScheduleConfigError(f"{name}: {tt!r} repeats a comparison")
        if tt.correct not in comparisons:
            raise ScheduleConfigError(
                f"{name}: correct {tt.correct!r} is not among the comparisons of {tt!r}"
            )
    return result


def _python_meta(outcome: Outcome) -> Outcome:
    """Give an outcome of the Rust engine the meta shape of these classes.

    The Rust engine leaves ``trial_outcome`` out when there is none,
    leaves ``choice`` out when a trial with trial types ends without a
    named comparison (a timeout or an unknown operandum), and returns
    ``comparisons`` as a list. These classes always set
    ``trial_outcome`` (``None`` when no trial ended), set ``choice`` to
    ``None`` in those cases, and use a tuple.
    """
    meta = dict(outcome.meta)
    meta.setdefault("trial_outcome", None)
    if "sample" in meta and meta["trial_outcome"] in ("correct", "incorrect"):
        meta.setdefault("choice", None)
    comparisons = meta.get("comparisons")
    if isinstance(comparisons, list):
        meta["comparisons"] = tuple(comparisons)
    return Outcome(reinforced=outcome.reinforced, reinforcer=outcome.reinforcer, meta=meta)


_CORRECTION_LIMIT_MAX = 2**32 - 1


def _check_correction(correction: bool | int | None) -> bool | int:
    """Normalise ``MatchingToSample(correction=...)``."""
    if correction is None or isinstance(correction, bool):
        return bool(correction)
    if not isinstance(correction, int):
        raise ScheduleConfigError(
            "correction must be None, a bool or an int >= 1, "
            f"got {type(correction).__name__}"
        )
    if not 1 <= correction <= _CORRECTION_LIMIT_MAX:
        raise ScheduleConfigError(
            f"correction limit must be in [1, {_CORRECTION_LIMIT_MAX}], got {correction}"
        )
    return correction


def _correction_spec(correction: bool | int) -> dict[str, Any] | None:
    """Spec form of a normalised correction; ``None`` when disabled."""
    if correction is False:
        return None
    if correction is True:
        return {"mode": "unlimited"}
    return {"mode": "bounded", "limit": correction}


def _trial_types_spec(types: Sequence[TrialType]) -> list[dict[str, Any]]:
    return [
        {"sample": t.sample, "correct": t.correct, "comparisons": list(t.comparisons)}
        for t in types
    ]


def _optional_specs(kind: str, params: dict[str, Any], named: dict[str, Any]) -> None:
    """Put the specs of the non-``None`` schedules in ``named`` into ``params``.

    Raises :class:`ScheduleConfigError` (see
    :func:`~contingency.schedules._rust.component_specs`) when one of
    them does not run on the Rust engine, has been stepped since its
    last reset, or is the same object as another.
    """
    present = {k: v for k, v in named.items() if v is not None}
    params.update(zip(present, component_specs(kind, present.values()), strict=True))


def _trial_schedule(spec: dict[str, Any]) -> Any:
    """The ``RustSchedule`` of a trial-based spec, checked to report ``phase``."""
    rust = compound_schedule(spec, needs_state=True)
    if "phase" not in rust.state():
        raise core_too_old(f"the state() of {spec['type']} has no 'phase'")
    return rust


class MatchingToSample:
    """Matching-to-sample (MTS) discrete-trial schedule.

    A trial unfolds as ``sample -> choice -> iti``:

    1. **Sample phase** (``sample_duration`` seconds): a sample stimulus
       is presented; response events are ignored.
    2. **Choice phase** (up to ``choice_timeout`` seconds): ``n_comparisons``
       comparison stimuli are available as operanda ``choice_0`` ...
       ``choice_{N-1}``. Exactly one is designated correct (sampled
       uniformly at random at trial start, or set by the trial type when
       ``trial_types`` is given). Choosing the correct
       comparison fires the nested ``consequence`` schedule; any other
       choice fires ``incorrect``. If no response occurs before the
       timeout, the trial is scored as incorrect.
    3. **ITI** (``iti`` seconds): response events are ignored; next trial
       begins when the ITI elapses.

    Parameters
    ----------
    n_comparisons:
        Number of comparison stimuli per trial. Must be ``>= 2``.
    sample_duration:
        Duration of the sample presentation; ``0`` presents choice
        immediately.
    choice_timeout:
        Maximum response latency in the choice phase. Must be ``> 0``.
    consequence:
        Schedule driving reinforcement on correct choice. If ``None``
        the correct choice yields continuous reinforcement (``FR(1)``
        semantics: an immediate reinforcer timestamped at ``now``).
    incorrect:
        Schedule driven on incorrect choice (e.g. a ``FT`` timeout).
        If ``None`` no consequence is delivered — the trial simply
        ends.
    iti:
        Inter-trial interval. Must be ``>= 0``.
    seed:
        Optional RNG seed for reproducible sample/correct-comparison
        selection.
    trial_types:
        Optional trial types (see :func:`trial_types_from_relations`).
        When given, each block presents every trial type in
        ``trial_types + probe_trial_types`` once, in an order shuffled at
        the start of the block, and the comparisons of each trial are
        shuffled over the positions ``choice_0`` ... ``choice_{N-1}``.
        The RNG is used only for these two shuffles. Outcome ``meta``
        then also reports ``sample``, ``comparisons`` (in position
        order), ``correct`` and ``probe``, and the outcome of a settled
        trial reports ``choice`` (the chosen comparison, ``None`` on
        timeout).
    probe_trial_types:
        Optional trial types presented without consequences: neither
        ``consequence`` nor ``incorrect`` is stepped and the outcome is
        never reinforced, while ``trial_outcome`` is still scored as
        ``"correct"`` or ``"incorrect"``. Requires ``trial_types``.
    correction:
        Correction procedure. ``None`` or ``False`` (default): none.
        ``True``: a baseline trial that ends incorrect (a wrong choice or
        a timeout) is presented again, after the ITI, until it is
        answered correctly. An ``int`` ``N >= 1``: at most ``N`` repeats,
        then the next trial is drawn. A repeated trial keeps the correct
        operandum, the sample and the comparison positions, and draws
        nothing from the RNG. Probe trials are not repeated. With
        correction active, outcome ``meta`` also reports
        ``correction_trial`` (whether the current trial is a repeat).

    References
    ----------
    Cumming, W. W., & Berryman, R. (1965). The complex discriminated
    operant: Studies of matching-to-sample and related problems. In
    D. I. Mostofsky (Ed.), *Stimulus Generalization* (pp. 284-330).
    Stanford University Press.
    """

    __slots__ = (
        "_rust",
        "_n_comparisons",
        "_sample_duration",
        "_choice_timeout",
        "_consequence",
        "_incorrect",
        "_iti",
        "_seed",
        "_trial_types",
        "_n_baseline_types",
        "_correction",
    )

    def __init__(
        self,
        n_comparisons: int,
        sample_duration: float = 0.0,
        choice_timeout: float = 5.0,
        consequence: Schedule | None = None,
        incorrect: Schedule | None = None,
        iti: float = 1.0,
        seed: int | None = None,
        trial_types: Sequence[TrialType] | None = None,
        probe_trial_types: Sequence[TrialType] | None = None,
        correction: bool | int | None = None,
    ) -> None:
        if isinstance(n_comparisons, bool) or not isinstance(n_comparisons, int):
            raise ScheduleConfigError(
                f"n_comparisons must be an int, got {type(n_comparisons).__name__}"
            )
        if n_comparisons < 2:
            raise ScheduleConfigError(
                f"n_comparisons must be >= 2, got {n_comparisons}"
            )
        self._n_comparisons: int = n_comparisons
        self._sample_duration: float = _require_nonnegative(
            "sample_duration", sample_duration
        )
        self._choice_timeout: float = _require_positive("choice_timeout", choice_timeout)
        self._iti: float = _require_nonnegative("iti", iti)
        self._consequence: Schedule | None = consequence
        self._incorrect: Schedule | None = incorrect
        if trial_types is None:
            if probe_trial_types is not None:
                raise ScheduleConfigError("probe_trial_types requires trial_types")
            self._trial_types: tuple[TrialType, ...] | None = None
            self._n_baseline_types: int = 0
        else:
            baseline = _validate_trial_types("trial_types", trial_types, n_comparisons)
            if not baseline:
                raise ScheduleConfigError("trial_types must not be empty")
            probes = _validate_trial_types(
                "probe_trial_types", probe_trial_types or (), n_comparisons
            )
            self._trial_types = baseline + probes
            self._n_baseline_types = len(baseline)
        self._seed: int | None = seed
        self._correction: bool | int = _check_correction(correction)
        self._rust: Any = self._build_rust()

    def _build_rust(self) -> Any:
        """The Rust engine for this schedule.

        A ``contingency_core`` that predates correction trials reads
        past the unknown ``correction`` key and would run the schedule
        without them; with correction active the engine must report
        ``correction_trial``, or the constructor raises
        :class:`ScheduleConfigError`.
        """
        spec = self._rust_spec()
        rust = _trial_schedule(spec)
        if self._correction is False:
            return rust
        probe = rust_schedule(spec)
        if "correction_trial" not in probe.step(0.0, None).meta:
            raise core_too_old("MatchingToSample runs without correction trials")
        return rust

    def _rust_spec(self) -> dict[str, Any]:
        params: dict[str, Any] = {
            "n_comparisons": self._n_comparisons,
            "sample_duration": self._sample_duration,
            "choice_timeout": self._choice_timeout,
            "iti": self._iti,
            "seed": self._seed if self._seed is not None else fresh_seed(),
        }
        _optional_specs(
            "MatchingToSample",
            params,
            {"consequence": self._consequence, "incorrect": self._incorrect},
        )
        if self._trial_types is not None:
            n = self._n_baseline_types
            params["trial_types"] = _trial_types_spec(self._trial_types[:n])
            if len(self._trial_types) > n:
                params["probe_trial_types"] = _trial_types_spec(self._trial_types[n:])
        correction_spec = _correction_spec(self._correction)
        if correction_spec is not None:
            params["correction"] = correction_spec
        return {"type": "MatchingToSample", "params": params}

    def _type_index_now(self) -> int | None:
        index = self._rust.state().get("current_type_index")
        return None if index is None else int(index)

    @property
    def engine(self) -> str:
        """Always ``"rust"``: the Rust engine runs this schedule."""
        return "rust"

    # -- introspection ------------------------------------------------------

    @property
    def n_comparisons(self) -> int:
        return self._n_comparisons

    @property
    def trial_index(self) -> int:
        return int(self._rust.state()["trial_index"])

    @property
    def phase(self) -> str:
        return str(self._rust.state()["phase"])

    @property
    def correct_operandum(self) -> str:
        return str(self._rust.state()["correct_operandum"])

    @property
    def iti(self) -> float:
        return self._iti

    @property
    def correction(self) -> bool | int:
        """``False`` (none), ``True`` (until correct) or the repeat limit."""
        return self._correction

    @property
    def choice_timeout(self) -> float:
        return self._choice_timeout

    @property
    def current_trial_type(self) -> TrialType | None:
        """Trial type of the current trial; ``None`` without ``trial_types``."""
        index = self._type_index_now()
        if self._trial_types is None or index is None:
            return None
        return self._trial_types[index]

    @property
    def current_sample(self) -> str | None:
        tt = self.current_trial_type
        return None if tt is None else tt.sample

    @property
    def current_comparisons(self) -> tuple[str, ...]:
        """Comparisons of the current trial in position order (``choice_0`` first)."""
        return tuple(self._rust.state()["current_positions"])

    @property
    def is_probe_trial(self) -> bool:
        index = self._type_index_now()
        return index is not None and index >= self._n_baseline_types

    # -- core step ----------------------------------------------------------

    def step(self, now: float, event: ResponseEvent | None = None) -> Outcome:
        return _python_meta(self._rust.step(now, event))

    def reset(self) -> None:
        for component in (self._consequence, self._incorrect):
            if component is not None:
                component.reset()
        self._rust = self._build_rust()


class GoNoGo:
    """Go/NoGo discrimination schedule.

    Each trial is sampled as a Go trial (probability ``go_probability``)
    or a NoGo trial. During the response window:

    * **Go trial + response** → ``correct_go_schedule`` is fired (or
      default CRF if ``None``).
    * **Go trial + no response** → miss; no reinforcement.
    * **NoGo trial + response** → ``false_alarm`` is fired (if any);
      meta outcome label is ``"false_alarm"``.
    * **NoGo trial + no response** (window elapses without a response)
      → ``correct_nogo_schedule`` is fired on the window boundary
      (reinforce withholding). If that schedule is ``None`` no
      reinforcement is produced — the trial is silently scored
      ``"correct_withhold"``.

    Parameters
    ----------
    go_probability:
        Probability a trial is a Go trial. Must satisfy
        ``0 < p < 1``.
    response_window:
        Response-eligibility window in unit-agnostic time. Must be
        ``> 0``.
    iti:
        Inter-trial interval, ``>= 0``.
    correct_go_schedule:
        Schedule fired on correct Go responses. Defaults to CRF (one
        immediate reinforcer) when ``None``.
    correct_nogo_schedule:
        Schedule fired on correct NoGo withholdings. ``None`` means a
        traditional discrimination procedure with no reinforcement for
        withholding.
    false_alarm:
        Schedule fired on NoGo responding (the "false alarm"). ``None``
        leaves the response unconsequated.
    seed:
        Optional RNG seed for deterministic trial-type sampling.

    References
    ----------
    Cook, L., & Cloak, N. L. (1956). Reinforcement schedules and some
    behavioral effects.

    Nevin, J. A. (1969). Signal detection theory and operant behavior.
    *Journal of the Experimental Analysis of Behavior*, 12(3), 475-480.
    https://doi.org/10.1901/jeab.1969.12-475
    """

    __slots__ = (
        "_rust",
        "_go_probability",
        "_response_window",
        "_iti",
        "_correct_go_schedule",
        "_correct_nogo_schedule",
        "_false_alarm",
        "_seed",
    )

    def __init__(
        self,
        go_probability: float = 0.5,
        response_window: float = 5.0,
        iti: float = 5.0,
        correct_go_schedule: Schedule | None = None,
        correct_nogo_schedule: Schedule | None = None,
        false_alarm: Schedule | None = None,
        seed: int | None = None,
    ) -> None:
        if isinstance(go_probability, bool) or not isinstance(
            go_probability, (int, float)
        ):
            raise ScheduleConfigError(
                "go_probability must be numeric, "
                f"got {type(go_probability).__name__}"
            )
        if not (0.0 < go_probability < 1.0):
            raise ScheduleConfigError(
                f"go_probability must satisfy 0 < p < 1, got {go_probability}"
            )
        self._go_probability: float = float(go_probability)
        self._response_window: float = _require_positive(
            "response_window", response_window
        )
        self._iti: float = _require_nonnegative("iti", iti)
        self._correct_go_schedule: Schedule | None = correct_go_schedule
        self._correct_nogo_schedule: Schedule | None = correct_nogo_schedule
        self._false_alarm: Schedule | None = false_alarm
        self._seed: int | None = seed
        self._rust: Any = _trial_schedule(self._rust_spec())

    def _rust_spec(self) -> dict[str, Any]:
        params: dict[str, Any] = {
            "go_probability": self._go_probability,
            "response_window": self._response_window,
            "iti": self._iti,
            "seed": self._seed if self._seed is not None else fresh_seed(),
        }
        _optional_specs(
            "GoNoGo",
            params,
            {
                "correct_go_schedule": self._correct_go_schedule,
                "correct_nogo_schedule": self._correct_nogo_schedule,
                "false_alarm": self._false_alarm,
            },
        )
        return {"type": "GoNoGo", "params": params}

    @property
    def engine(self) -> str:
        """Always ``"rust"``: the Rust engine runs this schedule."""
        return "rust"

    # -- introspection ------------------------------------------------------

    @property
    def trial_index(self) -> int:
        return int(self._rust.state()["trial_index"])

    @property
    def phase(self) -> str:
        return str(self._rust.state()["phase"])

    @property
    def current_is_go(self) -> bool:
        return bool(self._rust.state()["current_is_go"])

    @property
    def response_window(self) -> float:
        return self._response_window

    @property
    def iti(self) -> float:
        return self._iti

    @property
    def go_probability(self) -> float:
        return self._go_probability

    # -- core step ----------------------------------------------------------

    def step(self, now: float, event: ResponseEvent | None = None) -> Outcome:
        return _python_meta(self._rust.step(now, event))

    def reset(self) -> None:
        for component in (
            self._correct_go_schedule,
            self._correct_nogo_schedule,
            self._false_alarm,
        ):
            if component is not None:
                component.reset()
        self._rust = _trial_schedule(self._rust_spec())
