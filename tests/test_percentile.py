"""Tests for the percentile-shaped schedule.

A percentile schedule (Platt, 1973; Galbicka, 1994) reinforces a
response whose measured dimension (IRT, force, etc.) falls at the
specified percentile of a recent response window.

References
----------
Galbicka, G. (1994). Shaping in the 21st century: Moving percentile
schedules into applied settings. *Journal of Applied Behavior
Analysis*, 27(4), 739-760. https://doi.org/10.1901/jaba.1994.27-739

Platt, J. R. (1973). Percentile reinforcement: Paradigms for
experimental analysis of response shaping. *Psychology of Learning
and Motivation*, 7, 271-296.
"""

from __future__ import annotations

import pytest

from contingency.entities import Outcome, ResponseEvent
from contingency.errors import ScheduleConfigError, ScheduleStateError
from contingency.interfaces import Schedule
from contingency.schedules import Percentile


def _respond(s: Schedule, now: float) -> Outcome:
    return s.step(now, ResponseEvent(time=now))


class TestProtocolCompliance:
    def test_percentile_satisfies_schedule_protocol(self) -> None:
        assert isinstance(Percentile("irt", 50, 5), Schedule)


class TestConstruction:
    def test_construct_with_defaults(self) -> None:
        p = Percentile("irt", 50, 10)
        assert p.target == "irt"
        assert p.rank == 50
        assert p.window == 10
        assert p.direction == "above"

    def test_construct_with_explicit_direction(self) -> None:
        p = Percentile("irt", 25, 4, direction="below")
        assert p.direction == "below"

    def test_accepts_uppercase_irt_target(self) -> None:
        p = Percentile("IRT", 50, 5)
        assert p.target == "IRT"

    def test_accepts_non_executable_targets(self) -> None:
        # Non-IRT targets construct fine; step() is where they raise.
        for target in ("force", "latency", "duration", "rate", "location"):
            p = Percentile(target, 50, 5)
            assert p.target == target

    def test_rejects_unknown_target(self) -> None:
        with pytest.raises(ScheduleConfigError, match="target"):
            Percentile("bogus", 50, 5)

    def test_rejects_non_string_target(self) -> None:
        with pytest.raises(ScheduleConfigError, match="string target"):
            Percentile(42, 50, 5)  # type: ignore[arg-type]

    def test_rejects_non_integer_rank(self) -> None:
        with pytest.raises(ScheduleConfigError, match="rank"):
            Percentile("irt", 50.5, 5)  # type: ignore[arg-type]

    def test_rejects_rank_below_zero(self) -> None:
        with pytest.raises(ScheduleConfigError, match="rank"):
            Percentile("irt", -1, 5)

    def test_rejects_rank_above_hundred(self) -> None:
        with pytest.raises(ScheduleConfigError, match="rank"):
            Percentile("irt", 101, 5)

    def test_rejects_non_integer_window(self) -> None:
        with pytest.raises(ScheduleConfigError, match="window"):
            Percentile("irt", 50, 5.5)  # type: ignore[arg-type]

    def test_rejects_window_zero(self) -> None:
        with pytest.raises(ScheduleConfigError, match="window"):
            Percentile("irt", 50, 0)

    def test_rejects_unknown_direction(self) -> None:
        with pytest.raises(ScheduleConfigError, match="direction"):
            Percentile("irt", 50, 5, direction="sideways")

    def test_rejects_bool_rank(self) -> None:
        # bool is-a int in Python; guard against accidental True/False leakage.
        with pytest.raises(ScheduleConfigError, match="rank"):
            Percentile("irt", True, 5)  # type: ignore[arg-type]


class TestBaselineWindow:
    def test_first_window_responses_never_reinforce(self) -> None:
        p = Percentile("irt", 50, 5)
        # Response 1 establishes the anchor (no IRT); responses 2..6 fill
        # the window with 5 IRTs. None of these can reinforce because the
        # percentile is computed only once the window is full.
        outcomes = [_respond(p, float(t)) for t in range(1, 7)]
        assert all(not o.reinforced for o in outcomes)

    def test_no_reinforcement_without_any_responses(self) -> None:
        p = Percentile("irt", 50, 3)
        out = p.step(1.0)
        assert out.reinforced is False

    def test_empty_event_time_ticks_do_not_advance_baseline(self) -> None:
        p = Percentile("irt", 50, 3)
        for t in (1.0, 2.0, 3.0):
            assert p.step(t).reinforced is False
        # No responses yet, so window is still empty.
        assert p.samples == ()


class TestAboveDirection:
    def test_above_reinforces_long_irt_after_baseline(self) -> None:
        """Responses 1–6 establish a 1-second IRT baseline; response 7 at
        a 2-second IRT exceeds the 50th percentile → reinforced."""
        p = Percentile("irt", 50, 5, direction="above")
        # Response 1 anchors; responses 2..6 fill the 5-slot window with
        # IRTs of 1.0 s each.
        for t in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
            assert _respond(p, t).reinforced is False
        # Response 7 at t=8.0 → IRT = 2.0 s > 1.0 s (median) → reinforced.
        out = _respond(p, 8.0)
        assert out.reinforced is True
        assert out.reinforcer is not None
        assert out.reinforcer.time == 8.0

    def test_above_does_not_reinforce_short_irt(self) -> None:
        p = Percentile("irt", 50, 5, direction="above")
        for t in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
            _respond(p, t)
        # Response at t=6.5 → IRT = 0.5 s < 1.0 s median → not reinforced.
        out = _respond(p, 6.5)
        assert out.reinforced is False


class TestBelowDirection:
    def test_below_reinforces_short_irt_after_baseline(self) -> None:
        p = Percentile("irt", 50, 5, direction="below")
        for t in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
            _respond(p, t)
        # IRT 0.5 s < 1.0 s median → reinforced (below).
        out = _respond(p, 6.5)
        assert out.reinforced is True

    def test_below_does_not_reinforce_long_irt(self) -> None:
        p = Percentile("irt", 50, 5, direction="below")
        for t in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
            _respond(p, t)
        out = _respond(p, 11.0)  # IRT = 5 s > median.
        assert out.reinforced is False


class TestPercentileThreshold:
    def test_meta_carries_threshold_and_irt_on_win(self) -> None:
        p = Percentile("irt", 50, 5, direction="above")
        for t in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
            _respond(p, t)
        out = _respond(p, 8.0)
        assert out.reinforced is True
        assert out.meta["percentile_rank"] == 50
        assert out.meta["percentile_direction"] == "above"
        assert out.meta["percentile_threshold"] == pytest.approx(1.0)
        assert out.meta["percentile_irt"] == pytest.approx(2.0)

    def test_rank_100_requires_longest_irt(self) -> None:
        """Rank 100 above: only IRT >= max of window reinforces."""
        p = Percentile("irt", 100, 5, direction="above")
        # Baseline IRTs 1.0, 1.0, 1.0, 1.0, 1.0 → max = 1.0.
        for t in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
            _respond(p, t)
        # IRT 1.0 s exactly meets rank-100 → reinforced.
        out = _respond(p, 7.0)
        assert out.reinforced is True

    def test_rank_0_with_above_reinforces_anything_geq_min(self) -> None:
        p = Percentile("irt", 0, 5, direction="above")
        for t in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
            _respond(p, t)
        # Any IRT >= min (1.0 s here) reinforces.
        out = _respond(p, 7.5)
        assert out.reinforced is True


class TestWindowFIFO:
    def test_window_evicts_oldest_after_capacity(self) -> None:
        p = Percentile("irt", 50, 3)
        # Fill the window with IRT=1.0 three times.
        for t in (1.0, 2.0, 3.0, 4.0):
            _respond(p, t)
        # Now replace with a different cadence.
        _respond(p, 6.0)  # IRT = 2.0
        samples = p.samples
        assert len(samples) == 3
        # Oldest baseline IRT should have been evicted.
        assert samples[-1] == pytest.approx(2.0)


class TestNonExecutableTargets:
    def test_force_target_raises_on_step(self) -> None:
        p = Percentile("force", 50, 5)
        with pytest.raises(ScheduleConfigError, match="not yet implemented"):
            _respond(p, 1.0)

    def test_latency_target_raises_on_step(self) -> None:
        p = Percentile("latency", 50, 5)
        with pytest.raises(ScheduleConfigError, match="not yet implemented"):
            _respond(p, 1.0)


class TestReset:
    def test_reset_clears_window_and_anchor(self) -> None:
        p = Percentile("irt", 50, 5)
        # Response 1 anchors; responses 2..6 fill the window.
        for t in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
            _respond(p, t)
        assert len(p.samples) == 5

        p.reset()
        assert p.samples == ()

        # After reset we need a full new baseline before reinforcement.
        for t in (10.0, 11.0, 12.0, 13.0, 14.0, 15.0):
            assert _respond(p, t).reinforced is False

    def test_reset_re_anchors_last_now(self) -> None:
        p = Percentile("irt", 50, 5)
        p.step(10.0)
        p.reset()
        # After reset, a step at t=1.0 should not raise non-monotonic.
        p.step(1.0)


class TestStateErrors:
    def test_non_monotonic_now_raises(self) -> None:
        p = Percentile("irt", 50, 5)
        p.step(2.0)
        with pytest.raises(ScheduleStateError, match="monotonic"):
            p.step(1.0)

    def test_event_time_mismatch_raises(self) -> None:
        p = Percentile("irt", 50, 5)
        with pytest.raises(ScheduleStateError, match="event.time"):
            p.step(1.0, ResponseEvent(time=2.0))


class TestSamplesSnapshot:
    def test_samples_returns_tuple_not_deque(self) -> None:
        p = Percentile("irt", 50, 5)
        _respond(p, 1.0)
        _respond(p, 2.0)
        samples = p.samples
        assert isinstance(samples, tuple)

    def test_samples_ordered_oldest_first(self) -> None:
        p = Percentile("irt", 50, 5)
        # IRTs: (2.0 - 1.0) = 1.0, (4.0 - 2.0) = 2.0, (5.0 - 4.0) = 1.0.
        _respond(p, 1.0)
        _respond(p, 2.0)
        _respond(p, 4.0)
        _respond(p, 5.0)
        assert list(p.samples) == pytest.approx([1.0, 2.0, 1.0])
