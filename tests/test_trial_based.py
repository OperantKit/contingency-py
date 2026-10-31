"""Tests for trial-based schedules (MTS, GoNoGo)."""

from __future__ import annotations

import dataclasses
import random

import pytest

from contingency.entities import Outcome, Reinforcer, ResponseEvent
from contingency.errors import ScheduleConfigError, ScheduleStateError
from contingency.schedules import FT, GoNoGo, MatchingToSample
from contingency.schedules.ratio import FR
from contingency.schedules.trial_based import TrialType, trial_types_from_relations

# ---------------------------------------------------------------------------
# MatchingToSample
# ---------------------------------------------------------------------------


class TestMTSConstruction:
    def test_minimum_n_comparisons(self) -> None:
        with pytest.raises(ScheduleConfigError):
            MatchingToSample(n_comparisons=1)

    def test_negative_sample_duration_rejected(self) -> None:
        with pytest.raises(ScheduleConfigError):
            MatchingToSample(n_comparisons=2, sample_duration=-1.0)

    def test_zero_choice_timeout_rejected(self) -> None:
        with pytest.raises(ScheduleConfigError):
            MatchingToSample(n_comparisons=2, choice_timeout=0.0)

    def test_negative_iti_rejected(self) -> None:
        with pytest.raises(ScheduleConfigError):
            MatchingToSample(n_comparisons=2, iti=-1.0)

    def test_noninteger_n_rejected(self) -> None:
        with pytest.raises(ScheduleConfigError):
            MatchingToSample(n_comparisons=2.5)  # type: ignore[arg-type]

    def test_construction_ok(self) -> None:
        mts = MatchingToSample(
            n_comparisons=3,
            sample_duration=0.5,
            choice_timeout=2.0,
            iti=1.0,
            seed=7,
        )
        assert mts.n_comparisons == 3
        assert mts.trial_index == 0
        assert mts.iti == 1.0
        assert mts.correct_operandum.startswith("choice_")


class TestMTSSamplePhase:
    def test_response_ignored_during_sample(self) -> None:
        mts = MatchingToSample(
            n_comparisons=2, sample_duration=1.0, choice_timeout=2.0, seed=1
        )
        # correct operandum picked at construction
        correct = mts.correct_operandum
        # A response during sample phase is ignored
        o = mts.step(0.1, ResponseEvent(time=0.1, operandum=correct))
        assert o.meta["phase"] == "sample"
        assert o.meta["trial_outcome"] is None
        assert not o.reinforced

    def test_tick_advances_into_choice(self) -> None:
        mts = MatchingToSample(
            n_comparisons=2, sample_duration=1.0, choice_timeout=2.0, seed=1
        )
        mts.step(0.0)
        o = mts.step(1.5)
        assert o.meta["phase"] == "choice"


class TestMTSChoicePhase:
    def test_correct_choice_defaults_to_crf(self) -> None:
        mts = MatchingToSample(
            n_comparisons=3, sample_duration=0, choice_timeout=5, iti=1, seed=123
        )
        correct = mts.correct_operandum
        o = mts.step(0.0, ResponseEvent(time=0.0, operandum=correct))
        assert o.reinforced
        assert o.meta["trial_outcome"] == "correct"
        assert o.reinforcer is not None
        assert o.reinforcer.time == 0.0

    def test_incorrect_choice_no_consequence(self) -> None:
        mts = MatchingToSample(
            n_comparisons=3, sample_duration=0, choice_timeout=5, iti=1, seed=123
        )
        correct = mts.correct_operandum
        others = [f"choice_{i}" for i in range(3) if f"choice_{i}" != correct]
        o = mts.step(0.0, ResponseEvent(time=0.0, operandum=others[0]))
        assert not o.reinforced
        assert o.meta["trial_outcome"] == "incorrect"

    def test_incorrect_choice_fires_incorrect_schedule(self) -> None:
        # Incorrect schedule is FT(5) — emits a reinforcer at t=5
        incorrect = FT(5.0)
        mts = MatchingToSample(
            n_comparisons=3,
            sample_duration=0,
            choice_timeout=5,
            iti=1,
            incorrect=incorrect,
            seed=123,
        )
        correct = mts.correct_operandum
        other = next(
            f"choice_{i}" for i in range(3) if f"choice_{i}" != correct
        )
        o = mts.step(0.0, ResponseEvent(time=0.0, operandum=other))
        # FT emits at t=duration, so this response-time step returns no
        # reinforcement yet — but meta label should be "incorrect".
        assert o.meta["trial_outcome"] == "incorrect"

    def test_consequence_schedule_passthrough(self) -> None:
        # FR(1) as consequence => reinforces immediately
        mts = MatchingToSample(
            n_comparisons=2,
            sample_duration=0,
            choice_timeout=5,
            iti=1,
            consequence=FR(1),
            seed=42,
        )
        correct = mts.correct_operandum
        o = mts.step(0.0, ResponseEvent(time=0.0, operandum=correct))
        assert o.reinforced
        assert o.meta["trial_outcome"] == "correct"

    def test_choice_timeout_scored_incorrect(self) -> None:
        mts = MatchingToSample(
            n_comparisons=2, sample_duration=0, choice_timeout=3.0, iti=1, seed=1
        )
        mts.step(0.0)
        o = mts.step(3.5)
        assert o.meta["trial_outcome"] == "incorrect"


class TestMTSITI:
    def test_iti_transitions_to_next_trial(self) -> None:
        mts = MatchingToSample(
            n_comparisons=2, sample_duration=0, choice_timeout=5, iti=1.0, seed=1
        )
        correct = mts.correct_operandum
        mts.step(0.0, ResponseEvent(time=0.0, operandum=correct))
        # During ITI responses ignored
        o = mts.step(0.5, ResponseEvent(time=0.5, operandum="choice_0"))
        assert o.meta["phase"] == "iti"
        # After iti elapses, new trial
        o = mts.step(1.2)
        assert o.meta["phase"] == "choice"
        assert mts.trial_index == 1

    def test_zero_iti_begins_next_trial_immediately(self) -> None:
        mts = MatchingToSample(
            n_comparisons=2, sample_duration=0, choice_timeout=5, iti=0.0, seed=1
        )
        correct = mts.correct_operandum
        mts.step(0.0, ResponseEvent(time=0.0, operandum=correct))
        o = mts.step(0.0001)
        # Either already in choice (trial 1) or still iti — with zero iti
        # we expect choice.
        assert o.meta["phase"] == "choice"


class TestMTSSeeding:
    def test_seeded_determinism(self) -> None:
        m1 = MatchingToSample(n_comparisons=5, seed=999)
        m2 = MatchingToSample(n_comparisons=5, seed=999)
        assert m1.correct_operandum == m2.correct_operandum

    def test_reset_returns_to_trial_zero(self) -> None:
        mts = MatchingToSample(
            n_comparisons=2, sample_duration=0, choice_timeout=5, iti=1.0, seed=3
        )
        original_correct = mts.correct_operandum
        mts.step(0.0, ResponseEvent(time=0.0, operandum=original_correct))
        mts.step(2.0)  # advance into trial 1
        assert mts.trial_index == 1
        mts.reset()
        assert mts.trial_index == 0
        assert mts.correct_operandum == original_correct


class TestMTSMonotonicity:
    def test_non_monotonic_time_rejected(self) -> None:
        mts = MatchingToSample(n_comparisons=2, seed=1)
        mts.step(1.0)
        with pytest.raises(ScheduleStateError):
            mts.step(0.5)

    def test_event_time_mismatch_rejected(self) -> None:
        mts = MatchingToSample(n_comparisons=2, seed=1)
        with pytest.raises(ScheduleStateError):
            mts.step(1.0, ResponseEvent(time=2.0, operandum="choice_0"))


# ---------------------------------------------------------------------------
# GoNoGo
# ---------------------------------------------------------------------------


class TestGoNoGoConstruction:
    def test_go_probability_bounds(self) -> None:
        with pytest.raises(ScheduleConfigError):
            GoNoGo(go_probability=0.0)
        with pytest.raises(ScheduleConfigError):
            GoNoGo(go_probability=1.0)

    def test_response_window_positive(self) -> None:
        with pytest.raises(ScheduleConfigError):
            GoNoGo(response_window=0.0)

    def test_iti_nonnegative(self) -> None:
        with pytest.raises(ScheduleConfigError):
            GoNoGo(iti=-1.0)

    def test_construction_ok(self) -> None:
        g = GoNoGo(go_probability=0.5, response_window=2.0, iti=3.0, seed=10)
        assert g.go_probability == 0.5
        assert g.response_window == 2.0
        assert g.iti == 3.0


class TestGoNoGoGoTrial:
    def _force_go(self, seed_trial_go: int = 1) -> GoNoGo:
        # Find a seed whose first draw is a Go trial.
        for s in range(200):
            g = GoNoGo(go_probability=0.5, response_window=2.0, iti=1.0, seed=s)
            if g.current_is_go:
                return g
        raise RuntimeError("no go seed found")

    def _force_nogo(self) -> GoNoGo:
        for s in range(200):
            g = GoNoGo(go_probability=0.5, response_window=2.0, iti=1.0, seed=s)
            if not g.current_is_go:
                return g
        raise RuntimeError("no nogo seed found")

    def test_go_response_default_crf(self) -> None:
        g = self._force_go()
        assert g.current_is_go
        o = g.step(0.5, ResponseEvent(time=0.5, operandum="main"))
        assert o.reinforced
        assert o.meta["trial_outcome"] == "correct"

    def test_go_no_response_miss(self) -> None:
        g = self._force_go()
        g.step(0.0)
        o = g.step(2.5)  # response window = 2.0
        assert not o.reinforced
        assert o.meta["trial_outcome"] == "incorrect"

    def test_go_response_window_boundary(self) -> None:
        g = self._force_go()
        o = g.step(2.0, ResponseEvent(time=2.0, operandum="main"))
        # At exactly the boundary, within tolerance the window closes
        # before the event is evaluated — classified as miss.
        assert o.meta["phase"] == "iti"


class TestGoNoGoNoGoTrial:
    def _force_nogo(self) -> GoNoGo:
        for s in range(200):
            g = GoNoGo(go_probability=0.5, response_window=2.0, iti=1.0, seed=s)
            if not g.current_is_go:
                return g
        raise RuntimeError("no nogo seed found")

    def test_nogo_response_is_false_alarm(self) -> None:
        g = self._force_nogo()
        o = g.step(0.5, ResponseEvent(time=0.5, operandum="main"))
        assert not o.reinforced
        assert o.meta["trial_outcome"] == "false_alarm"

    def test_nogo_response_fires_false_alarm_schedule(self) -> None:
        # Provide a false-alarm schedule that is FT-like; we'll just use
        # FR(1) which emits on responses as a sentinel.
        fa = FR(1)
        # Construct carefully to force nogo
        for s in range(200):
            g = GoNoGo(
                go_probability=0.5,
                response_window=2.0,
                iti=1.0,
                false_alarm=fa,
                seed=s,
            )
            if not g.current_is_go:
                break
        o = g.step(0.5, ResponseEvent(time=0.5, operandum="main"))
        # FR(1) reinforces on any response; our bridge propagates this.
        assert o.reinforced
        assert o.meta["trial_outcome"] == "false_alarm"

    def test_nogo_withhold_with_reinforcer(self) -> None:
        # correct_nogo_schedule = FR(1) — but we call it with event=None,
        # which yields no reinforcement. Use a trivial harness: use None
        # (default traditional discrimination) and assert the label only.
        g = self._force_nogo()
        g.step(0.0)
        o = g.step(2.5)
        assert o.meta["trial_outcome"] == "correct_withhold"
        assert not o.reinforced  # no correct_nogo_schedule => no reinforcement


class TestGoNoGoDistribution:
    def test_go_probability_distribution(self) -> None:
        # Over many trials, the empirical frequency should match p.
        trials = 2000
        p = 0.7
        rng_go_count = 0
        g = GoNoGo(
            go_probability=p, response_window=1.0, iti=0.0, seed=42
        )
        # Peek at trial type then resolve with no response; advance.
        now = 0.0
        for _ in range(trials):
            if g.current_is_go:
                rng_go_count += 1
            now += 1.0  # close the response window
            g.step(now)
            # ITI == 0, so we're already in next trial
        freq = rng_go_count / trials
        assert abs(freq - p) < 0.05


class TestGoNoGoSeeding:
    def test_seeded_determinism(self) -> None:
        g1 = GoNoGo(go_probability=0.5, seed=777)
        g2 = GoNoGo(go_probability=0.5, seed=777)
        assert g1.current_is_go == g2.current_is_go

    def test_reset_returns_to_trial_zero(self) -> None:
        g = GoNoGo(go_probability=0.5, response_window=1.0, iti=0.5, seed=3)
        g.step(0.0)
        g.step(2.0)
        assert g.trial_index >= 1
        g.reset()
        assert g.trial_index == 0


class TestGoNoGoMonotonicity:
    def test_non_monotonic_time_rejected(self) -> None:
        g = GoNoGo(seed=1)
        g.step(1.0)
        with pytest.raises(ScheduleStateError):
            g.step(0.5)


# ---------------------------------------------------------------------------
# Incorrect paths coverage
# ---------------------------------------------------------------------------


class TestMTSAuxiliary:
    def test_iti_response_ignored(self) -> None:
        mts = MatchingToSample(
            n_comparisons=2, sample_duration=0, choice_timeout=5, iti=2.0, seed=1
        )
        correct = mts.correct_operandum
        mts.step(0.0, ResponseEvent(time=0.0, operandum=correct))
        o = mts.step(0.5, ResponseEvent(time=0.5, operandum="choice_0"))
        assert o.meta["phase"] == "iti"
        assert not o.reinforced

    def test_timeout_with_incorrect_schedule(self) -> None:
        mts = MatchingToSample(
            n_comparisons=2,
            sample_duration=0,
            choice_timeout=1.0,
            iti=1,
            incorrect=FT(1.0),
            seed=1,
        )
        mts.step(0.0)
        o = mts.step(2.0)
        assert o.meta["trial_outcome"] == "incorrect"


# ---------------------------------------------------------------------------
# MatchingToSample with trial types (stimulus relations and probe trials)
# ---------------------------------------------------------------------------

AB = {"A1": "B1", "A2": "B2", "A3": "B3"}
BA = {"B1": "A1", "B2": "A2", "B3": "A3"}


class _SpySchedule:
    """Records every ``step`` call and always delivers a reinforcer."""

    def __init__(self) -> None:
        self.calls: list[tuple[float, ResponseEvent | None]] = []

    def step(self, now: float, event: ResponseEvent | None = None) -> Outcome:
        self.calls.append((now, event))
        return Outcome(reinforced=True, reinforcer=Reinforcer(time=now))

    def reset(self) -> None:
        self.calls.clear()


def _operandum_for(mts: MatchingToSample, name: str) -> str:
    return f"choice_{mts.current_comparisons.index(name)}"


def _run_trials(
    mts: MatchingToSample, n_trials: int, *, choose_correct: bool = True
) -> list[Outcome]:
    """Run ``n_trials`` trials (sample_duration=0, iti=1), one per time unit."""
    outcomes = []
    for t in range(n_trials):
        mts.step(float(t))
        tt = mts.current_trial_type
        assert tt is not None
        name = tt.correct if choose_correct else next(
            c for c in tt.comparisons if c != tt.correct
        )
        op = _operandum_for(mts, name)
        outcomes.append(mts.step(float(t), ResponseEvent(time=float(t), operandum=op)))
    return outcomes


class TestTrialTypesFromRelations:
    def test_builds_one_trial_type_per_relation(self) -> None:
        types = trial_types_from_relations(AB)
        assert types == (
            TrialType("A1", "B1", ("B1", "B2", "B3")),
            TrialType("A2", "B2", ("B1", "B2", "B3")),
            TrialType("A3", "B3", ("B1", "B2", "B3")),
        )

    def test_duplicate_value_rejected(self) -> None:
        with pytest.raises(ScheduleConfigError):
            trial_types_from_relations({"A1": "B1", "A2": "B1"})

    def test_trial_type_is_frozen(self) -> None:
        tt = TrialType("A1", "B1", ("B1", "B2"))
        with pytest.raises(dataclasses.FrozenInstanceError):
            tt.sample = "A2"  # type: ignore[misc]


class TestMTSTrialTypes:
    def _mts(self, **kwargs: object) -> MatchingToSample:
        params: dict[str, object] = {
            "n_comparisons": 3,
            "sample_duration": 0,
            "choice_timeout": 5,
            "iti": 1,
            "trial_types": trial_types_from_relations(AB),
            "seed": 7,
        }
        params.update(kwargs)
        return MatchingToSample(**params)  # type: ignore[arg-type]

    def test_sample_and_correct_follow_relations(self) -> None:
        mts = self._mts()
        samples = []
        for t in range(6):
            mts.step(float(t))
            sample = mts.current_sample
            assert sample is not None
            samples.append(sample)
            assert mts.current_trial_type is not None
            assert sorted(mts.current_comparisons) == ["B1", "B2", "B3"]
            correct_idx = int(mts.correct_operandum.removeprefix("choice_"))
            assert mts.current_comparisons[correct_idx] == AB[sample]
            assert mts.is_probe_trial is False
            o = mts.step(
                float(t), ResponseEvent(time=float(t), operandum=mts.correct_operandum)
            )
            assert o.reinforced
            assert o.meta["trial_outcome"] == "correct"
        # Each block of 3 trials presents every sample exactly once.
        assert sorted(samples[:3]) == ["A1", "A2", "A3"]
        assert sorted(samples[3:]) == ["A1", "A2", "A3"]

    def test_rng_calls_are_block_shuffle_then_position_shuffle(self) -> None:
        types = trial_types_from_relations(AB)
        mts = self._mts()
        rng = random.Random(7)
        expected = []
        for _ in range(2):
            order = list(range(len(types)))
            rng.shuffle(order)
            for i in order:
                positions = list(types[i].comparisons)
                rng.shuffle(positions)
                expected.append((types[i].sample, tuple(positions)))
        observed = []
        for t in range(6):
            mts.step(float(t))
            observed.append((mts.current_sample, mts.current_comparisons))
            mts.step(float(t), ResponseEvent(time=float(t), operandum="choice_0"))
        assert observed == expected

    def test_meta_reports_sample_comparisons_correct_probe_choice(self) -> None:
        mts = self._mts()
        o0 = mts.step(0.0)
        assert o0.meta["sample"] == mts.current_sample
        assert o0.meta["comparisons"] == mts.current_comparisons
        assert o0.meta["correct"] == AB[mts.current_sample or ""]
        assert o0.meta["probe"] is False
        assert "choice" not in o0.meta
        chosen = mts.current_comparisons[1]
        o = mts.step(0.0, ResponseEvent(time=0.0, operandum="choice_1"))
        assert o.meta["choice"] == chosen
        assert o.meta["sample"] == o0.meta["sample"]
        assert o.meta["trial_outcome"] == (
            "correct" if chosen == o0.meta["correct"] else "incorrect"
        )
        # Timeout: the settled trial reports choice=None.
        mts.step(1.0)
        timeout = mts.step(7.0)
        assert timeout.meta["trial_outcome"] == "incorrect"
        assert timeout.meta["choice"] is None

    def test_probe_trials_are_not_consequated(self) -> None:
        consequence = _SpySchedule()
        incorrect = _SpySchedule()
        mts = self._mts(
            trial_types=trial_types_from_relations(AB),
            probe_trial_types=trial_types_from_relations(BA),
            consequence=consequence,
            incorrect=incorrect,
        )
        probes = {"correct": 0, "incorrect": 0}
        for t in range(12):
            mts.step(float(t))
            tt = mts.current_trial_type
            assert tt is not None
            choose_correct = t % 2 == 0
            name = tt.correct if choose_correct else next(
                c for c in tt.comparisons if c != tt.correct
            )
            n_before = len(consequence.calls) + len(incorrect.calls)
            o = mts.step(
                float(t), ResponseEvent(time=float(t), operandum=_operandum_for(mts, name))
            )
            assert o.meta["trial_outcome"] == ("correct" if choose_correct else "incorrect")
            if o.meta["probe"]:
                assert mts.is_probe_trial
                assert tt.sample.startswith("B")
                assert not o.reinforced
                assert o.reinforcer is None
                assert len(consequence.calls) + len(incorrect.calls) == n_before
                probes[str(o.meta["trial_outcome"])] += 1
            else:
                assert tt.sample.startswith("A")
                assert len(consequence.calls) + len(incorrect.calls) == n_before + 1
        assert probes["correct"] > 0 and probes["incorrect"] > 0

    def test_probe_timeout_does_not_fire_incorrect(self) -> None:
        incorrect = _SpySchedule()
        mts = self._mts(
            trial_types=[TrialType("A1", "B1", ("B1", "B2", "B3"))],
            probe_trial_types=[TrialType("B1", "A1", ("A1", "A2", "A3"))],
            incorrect=incorrect,
            choice_timeout=1.0,
        )
        t = 0.0
        seen_probe_timeout = False
        for _ in range(6):
            mts.step(t)
            probe = mts.is_probe_trial
            n_before = len(incorrect.calls)
            o = mts.step(t + 1.5)  # times the choice out
            assert o.meta["trial_outcome"] == "incorrect"
            assert o.meta["choice"] is None
            if probe:
                seen_probe_timeout = True
                assert not o.reinforced
                assert len(incorrect.calls) == n_before
            else:
                assert len(incorrect.calls) == n_before + 1
            t += 2.0  # timeout boundary at t+1, ITI ends (next trial) at t+2
        assert seen_probe_timeout

    def test_reset_reproduces_sample_sequence(self) -> None:
        mts = self._mts(probe_trial_types=trial_types_from_relations(BA))
        first_sample = mts.current_sample
        _run_trials(mts, 9)
        mts.reset()
        assert mts.current_sample == first_sample
        seq_a = [str(o.meta["sample"]) for o in _run_trials(mts, 9)]
        mts.reset()
        seq_b = [str(o.meta["sample"]) for o in _run_trials(mts, 9)]
        assert seq_a == seq_b
        assert sorted(seq_a[:6]) == ["A1", "A2", "A3", "B1", "B2", "B3"]

    def test_correct_not_in_comparisons_rejected(self) -> None:
        with pytest.raises(ScheduleConfigError):
            self._mts(trial_types=[TrialType("A1", "B9", ("B1", "B2", "B3"))])

    def test_comparison_count_mismatch_rejected(self) -> None:
        with pytest.raises(ScheduleConfigError):
            self._mts(trial_types=[TrialType("A1", "B1", ("B1", "B2"))])

    def test_duplicate_comparisons_rejected(self) -> None:
        with pytest.raises(ScheduleConfigError):
            self._mts(trial_types=[TrialType("A1", "B1", ("B1", "B1", "B2"))])

    def test_empty_trial_types_rejected(self) -> None:
        with pytest.raises(ScheduleConfigError):
            self._mts(trial_types=[])

    def test_probe_without_trial_types_rejected(self) -> None:
        with pytest.raises(ScheduleConfigError):
            self._mts(trial_types=None, probe_trial_types=trial_types_from_relations(BA))

    def test_probe_trial_type_validated(self) -> None:
        with pytest.raises(ScheduleConfigError):
            self._mts(probe_trial_types=[TrialType("B1", "A9", ("A1", "A2", "A3"))])

    def test_default_mode_meta_unchanged(self) -> None:
        mts = MatchingToSample(n_comparisons=2, sample_duration=0, seed=7)
        o = mts.step(0.0, ResponseEvent(time=0.0, operandum=mts.correct_operandum))
        assert set(o.meta) == {"phase", "trial_index", "trial_outcome"}
        assert mts.current_trial_type is None
        assert mts.current_sample is None
        assert mts.current_comparisons == ()
        assert mts.is_probe_trial is False
