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
* The state machines are driven by two kinds of input: ticks (calls to
  ``step`` with ``event=None``) that advance the internal clock, and
  response events that are evaluated during response-eligible phases.
* During inter-trial intervals (ITI) and sample presentation, responses
  are ignored but ticks drive phase transitions.
* Consequence delivery is delegated to a nested ``Schedule`` object; a
  correct choice fires a synthetic response event into the consequence
  schedule, returning whatever outcome it yields (typically an immediate
  reinforcer when the consequence is ``FR(1)``/CRF).
* The ``meta`` payload on the returned ``Outcome`` always reports the
  current phase, trial index, and (when applicable) the outcome label of
  the completed trial.

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

import random
from typing import Literal

from contingency.entities import Outcome, Reinforcer, ResponseEvent
from contingency.errors import ScheduleConfigError, ScheduleStateError
from contingency.interfaces import Schedule

__all__ = ["MatchingToSample", "GoNoGo"]

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


class MatchingToSample:
    """Matching-to-sample (MTS) discrete-trial schedule.

    A trial unfolds as ``sample -> choice -> iti``:

    1. **Sample phase** (``sample_duration`` seconds): a sample stimulus
       is presented; response events are ignored.
    2. **Choice phase** (up to ``choice_timeout`` seconds): ``n_comparisons``
       comparison stimuli are available as operanda ``choice_0`` ...
       ``choice_{N-1}``. Exactly one is designated correct (sampled
       uniformly at random at trial start). Choosing the correct
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

    References
    ----------
    Cumming, W. W., & Berryman, R. (1965). The complex discriminated
    operant: Studies of matching-to-sample and related problems. In
    D. I. Mostofsky (Ed.), *Stimulus Generalization* (pp. 284-330).
    Stanford University Press.
    """

    __slots__ = (
        "_n_comparisons",
        "_sample_duration",
        "_choice_timeout",
        "_consequence",
        "_incorrect",
        "_iti",
        "_seed",
        "_rng",
        "_phase",
        "_phase_started",
        "_trial_index",
        "_correct_operandum",
        "_last_now",
        "_pending_outcome_label",
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
        self._seed: int | None = seed
        self._rng: random.Random = random.Random(seed)
        self._phase: Literal["sample", "choice", "iti"] = "sample"
        self._phase_started: float | None = None
        self._trial_index: int = 0
        self._correct_operandum: str = self._pick_correct_operandum()
        self._last_now: float | None = None
        self._pending_outcome_label: str | None = None

    # -- introspection ------------------------------------------------------

    @property
    def n_comparisons(self) -> int:
        return self._n_comparisons

    @property
    def trial_index(self) -> int:
        return self._trial_index

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def correct_operandum(self) -> str:
        return self._correct_operandum

    @property
    def iti(self) -> float:
        return self._iti

    # -- state helpers ------------------------------------------------------

    def _pick_correct_operandum(self) -> str:
        idx = self._rng.randrange(self._n_comparisons)
        return f"choice_{idx}"

    def _begin_trial(self, now: float) -> None:
        self._phase = "sample" if self._sample_duration > 0 else "choice"
        self._phase_started = now
        self._correct_operandum = self._pick_correct_operandum()

    def _enter_choice(self, now: float) -> None:
        self._phase = "choice"
        self._phase_started = now

    def _enter_iti(self, now: float, outcome_label: str) -> None:
        self._phase = "iti"
        self._phase_started = now
        self._pending_outcome_label = outcome_label

    def _finish_iti(self, now: float) -> None:
        self._trial_index += 1
        self._begin_trial(now)

    # -- core step ----------------------------------------------------------

    def step(self, now: float, event: ResponseEvent | None = None) -> Outcome:
        _check_time(self._last_now, now)
        _check_event(now, event)
        self._last_now = now

        # First step ever: anchor phase start at the first observed time.
        if self._phase_started is None:
            self._phase_started = now

        # Consume any phase transitions that should have already happened
        # by ``now`` before evaluating the current event. We iterate
        # because, e.g., a long tick might cross both the sample->choice
        # and choice->iti boundaries.
        while True:
            progressed = False
            if self._phase == "sample":
                assert self._phase_started is not None
                if now - self._phase_started >= self._sample_duration - _TIME_TOL:
                    self._enter_choice(self._phase_started + self._sample_duration)
                    progressed = True
            elif self._phase == "choice":
                assert self._phase_started is not None
                # Choice phase ends (as incorrect) only on timeout. The
                # timeout check is performed inline below so that a
                # response event at exactly the boundary still counts.
                if now - self._phase_started > self._choice_timeout + _TIME_TOL:
                    # Choice timed out without any response.
                    self._enter_iti(
                        self._phase_started + self._choice_timeout,
                        outcome_label="incorrect",
                    )
                    return self._incorrect_outcome(now)
            elif self._phase == "iti":
                assert self._phase_started is not None
                if now - self._phase_started >= self._iti - _TIME_TOL:
                    self._finish_iti(self._phase_started + self._iti)
                    progressed = True
            if not progressed:
                break

        # Evaluate response event according to the (possibly updated) phase.
        if event is None:
            return self._phase_only_outcome()

        if self._phase == "choice":
            return self._handle_choice(now, event)

        # During sample / ITI, responses are ignored but still return
        # phase metadata.
        return self._phase_only_outcome()

    # -- outcome helpers ----------------------------------------------------

    def _phase_only_outcome(self) -> Outcome:
        meta: dict[str, object] = {
            "phase": self._phase,
            "trial_index": self._trial_index,
            "trial_outcome": self._pending_outcome_label,
        }
        # The pending label only reports once per transition.
        self._pending_outcome_label = None
        return Outcome(meta=meta)

    def _handle_choice(self, now: float, event: ResponseEvent) -> Outcome:
        op = event.operandum
        correct = op == self._correct_operandum
        label = "correct" if correct else "incorrect"
        # Move to ITI immediately after the choice resolves. The label
        # is reported in this outcome directly, so clear the pending
        # label to avoid a duplicate report on the next tick.
        self._enter_iti(now, outcome_label=label)
        self._pending_outcome_label = None
        # Deliver consequence through the nested schedule (or default CRF).
        consequence_schedule = (
            self._consequence if correct else self._incorrect
        )
        if correct and consequence_schedule is None:
            reinforcer = Reinforcer(time=now)
            return Outcome(
                reinforced=True,
                reinforcer=reinforcer,
                meta={
                    "phase": self._phase,
                    "trial_index": self._trial_index,
                    "trial_outcome": label,
                },
            )
        if consequence_schedule is None:
            return Outcome(
                meta={
                    "phase": self._phase,
                    "trial_index": self._trial_index,
                    "trial_outcome": label,
                }
            )
        sub_outcome = consequence_schedule.step(now, event)
        meta: dict[str, object] = {
            "phase": self._phase,
            "trial_index": self._trial_index,
            "trial_outcome": label,
        }
        # Preserve nested meta non-destructively under a reserved key.
        if sub_outcome.meta:
            meta["inner_meta"] = dict(sub_outcome.meta)
        return Outcome(
            reinforced=sub_outcome.reinforced,
            reinforcer=sub_outcome.reinforcer,
            meta=meta,
        )

    def _incorrect_outcome(self, now: float) -> Outcome:
        # Timed-out choice: fire the ``incorrect`` consequence (if any)
        # with no response event — the subject did not emit one.
        if self._incorrect is None:
            return self._phase_only_outcome()
        sub_outcome = self._incorrect.step(now, None)
        meta: dict[str, object] = {
            "phase": self._phase,
            "trial_index": self._trial_index,
            "trial_outcome": "incorrect",
        }
        self._pending_outcome_label = None
        if sub_outcome.meta:
            meta["inner_meta"] = dict(sub_outcome.meta)
        return Outcome(
            reinforced=sub_outcome.reinforced,
            reinforcer=sub_outcome.reinforcer,
            meta=meta,
        )

    # -- reset --------------------------------------------------------------

    def reset(self) -> None:
        self._rng = random.Random(self._seed)
        self._phase = "sample" if self._sample_duration > 0 else "choice"
        self._phase_started = None
        self._trial_index = 0
        self._correct_operandum = self._pick_correct_operandum()
        self._last_now = None
        self._pending_outcome_label = None
        if self._consequence is not None:
            self._consequence.reset()
        if self._incorrect is not None:
            self._incorrect.reset()


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
        "_go_probability",
        "_response_window",
        "_iti",
        "_correct_go_schedule",
        "_correct_nogo_schedule",
        "_false_alarm",
        "_seed",
        "_rng",
        "_phase",
        "_phase_started",
        "_trial_index",
        "_current_is_go",
        "_responded_this_trial",
        "_last_now",
        "_pending_outcome_label",
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
        self._rng: random.Random = random.Random(seed)
        self._phase: Literal["go", "nogo", "iti"] = "go"
        self._phase_started: float | None = None
        self._trial_index: int = 0
        self._current_is_go: bool = self._draw_trial_type()
        self._phase = "go" if self._current_is_go else "nogo"
        self._responded_this_trial: bool = False
        self._last_now: float | None = None
        self._pending_outcome_label: str | None = None

    # -- introspection ------------------------------------------------------

    @property
    def trial_index(self) -> int:
        return self._trial_index

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def current_is_go(self) -> bool:
        return self._current_is_go

    @property
    def response_window(self) -> float:
        return self._response_window

    @property
    def iti(self) -> float:
        return self._iti

    @property
    def go_probability(self) -> float:
        return self._go_probability

    # -- state helpers ------------------------------------------------------

    def _draw_trial_type(self) -> bool:
        return self._rng.random() < self._go_probability

    def _begin_trial(self, now: float) -> None:
        self._current_is_go = self._draw_trial_type()
        self._phase = "go" if self._current_is_go else "nogo"
        self._phase_started = now
        self._responded_this_trial = False

    def _enter_iti(self, now: float, outcome_label: str) -> None:
        self._phase = "iti"
        self._phase_started = now
        self._pending_outcome_label = outcome_label

    def _finish_iti(self, now: float) -> None:
        self._trial_index += 1
        self._begin_trial(now)

    # -- core step ----------------------------------------------------------

    def step(self, now: float, event: ResponseEvent | None = None) -> Outcome:
        _check_time(self._last_now, now)
        _check_event(now, event)
        self._last_now = now

        if self._phase_started is None:
            self._phase_started = now

        # Handle response-window timeout(s) first. A long tick may cross
        # multiple phase boundaries (response_window and subsequent ITI).
        resolution_outcome: Outcome | None = None
        while True:
            progressed = False
            if self._phase in ("go", "nogo"):
                assert self._phase_started is not None
                if now - self._phase_started >= self._response_window - _TIME_TOL:
                    # Window closed without a response in this trial.
                    boundary = self._phase_started + self._response_window
                    if self._current_is_go:
                        # Go + no response = miss, no reinforcement.
                        self._enter_iti(boundary, outcome_label="incorrect")
                    else:
                        # NoGo + withhold = correct_withhold.
                        if self._correct_nogo_schedule is not None:
                            # Fire a synthetic "withhold" event.
                            sub = self._correct_nogo_schedule.step(boundary, None)
                            resolution_outcome = Outcome(
                                reinforced=sub.reinforced,
                                reinforcer=sub.reinforcer,
                                meta={
                                    "phase": "nogo",
                                    "trial_index": self._trial_index,
                                    "trial_outcome": "correct_withhold",
                                    **(
                                        {"inner_meta": dict(sub.meta)}
                                        if sub.meta
                                        else {}
                                    ),
                                },
                            )
                        self._enter_iti(
                            boundary, outcome_label="correct_withhold"
                        )
                    progressed = True
            elif self._phase == "iti":
                assert self._phase_started is not None
                if now - self._phase_started >= self._iti - _TIME_TOL:
                    self._finish_iti(self._phase_started + self._iti)
                    progressed = True
            if not progressed:
                break

        if resolution_outcome is not None and event is None:
            # The outcome label has already been included in
            # ``resolution_outcome``; drop the pending duplicate.
            self._pending_outcome_label = None
            return resolution_outcome

        if event is None:
            return self._phase_only_outcome()

        if self._phase in ("go", "nogo"):
            return self._handle_response(now, event)

        # Response during ITI — ignored.
        return self._phase_only_outcome()

    # -- outcome helpers ----------------------------------------------------

    def _phase_only_outcome(self) -> Outcome:
        meta: dict[str, object] = {
            "phase": self._phase,
            "trial_index": self._trial_index,
            "trial_outcome": self._pending_outcome_label,
        }
        self._pending_outcome_label = None
        return Outcome(meta=meta)

    def _handle_response(self, now: float, event: ResponseEvent) -> Outcome:
        self._responded_this_trial = True
        is_go = self._current_is_go
        label = "correct" if is_go else "false_alarm"
        target_schedule = (
            self._correct_go_schedule if is_go else self._false_alarm
        )
        # Close the trial: enter ITI starting now. The outcome label is
        # reported in this outcome directly; clear the pending label so
        # it is not re-reported on the next tick.
        self._enter_iti(now, outcome_label=label)
        self._pending_outcome_label = None
        if is_go and target_schedule is None:
            # Default CRF on correct Go.
            return Outcome(
                reinforced=True,
                reinforcer=Reinforcer(time=now),
                meta={
                    "phase": self._phase,
                    "trial_index": self._trial_index,
                    "trial_outcome": label,
                },
            )
        if target_schedule is None:
            return Outcome(
                meta={
                    "phase": self._phase,
                    "trial_index": self._trial_index,
                    "trial_outcome": label,
                }
            )
        sub = target_schedule.step(now, event)
        meta: dict[str, object] = {
            "phase": self._phase,
            "trial_index": self._trial_index,
            "trial_outcome": label,
        }
        if sub.meta:
            meta["inner_meta"] = dict(sub.meta)
        return Outcome(
            reinforced=sub.reinforced,
            reinforcer=sub.reinforcer,
            meta=meta,
        )

    # -- reset --------------------------------------------------------------

    def reset(self) -> None:
        self._rng = random.Random(self._seed)
        self._phase_started = None
        self._trial_index = 0
        self._current_is_go = self._draw_trial_type()
        self._phase = "go" if self._current_is_go else "nogo"
        self._responded_this_trial = False
        self._last_now = None
        self._pending_outcome_label = None
        for sched in (
            self._correct_go_schedule,
            self._correct_nogo_schedule,
            self._false_alarm,
        ):
            if sched is not None:
                sched.reset()
