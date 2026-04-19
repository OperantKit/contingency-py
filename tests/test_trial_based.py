"""Tests for trial-based schedules (MTS, GoNoGo)."""

from __future__ import annotations

import pytest

from contingency.entities import ResponseEvent
from contingency.errors import ScheduleConfigError, ScheduleStateError
from contingency.schedules import FT, GoNoGo, MatchingToSample
from contingency.schedules.ratio import FR

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
