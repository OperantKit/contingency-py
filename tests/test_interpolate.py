"""Tests for the Interpolate compound schedule (mid-session probe)."""

from __future__ import annotations

import pytest

from contingency.entities import ResponseEvent
from contingency.errors import ScheduleConfigError, ScheduleStateError
from contingency.schedules import EXT, FR, Interpolate


@pytest.mark.unit
class TestInterpolateConfig:
    def test_rejects_non_schedule_base(self) -> None:
        with pytest.raises(ScheduleConfigError):
            Interpolate("nope", FR(1), interval=10.0, probe_duration=1.0)  # type: ignore[arg-type]

    def test_rejects_non_schedule_probe(self) -> None:
        with pytest.raises(ScheduleConfigError):
            Interpolate(FR(1), 42, interval=10.0, probe_duration=1.0)  # type: ignore[arg-type]

    def test_rejects_non_positive_interval(self) -> None:
        with pytest.raises(ScheduleConfigError):
            Interpolate(FR(1), FR(2), interval=0, probe_duration=1.0)

    def test_rejects_non_positive_probe_duration(self) -> None:
        with pytest.raises(ScheduleConfigError):
            Interpolate(FR(1), FR(2), interval=10.0, probe_duration=0)

    def test_rejects_probe_duration_geq_interval(self) -> None:
        with pytest.raises(ScheduleConfigError):
            Interpolate(FR(1), FR(2), interval=5.0, probe_duration=5.0)

    def test_rejects_negative_first_probe_at(self) -> None:
        with pytest.raises(ScheduleConfigError):
            Interpolate(
                FR(1), FR(2), interval=10.0, probe_duration=1.0, first_probe_at=-1.0
            )

    def test_default_first_probe_at_equals_interval(self) -> None:
        sch = Interpolate(FR(1), FR(2), interval=10.0, probe_duration=1.0)
        assert sch.first_probe_at == pytest.approx(10.0)


@pytest.mark.unit
class TestInterpolateStep:
    def test_starts_outside_probe(self) -> None:
        sch = Interpolate(
            FR(1), EXT(), interval=10.0, probe_duration=2.0, first_probe_at=5.0
        )
        out = sch.step(0.0, ResponseEvent(0.0))
        assert out.meta.get("in_probe") is False
        # base is FR(1) -> reinforces.
        assert out.reinforced

    def test_enters_probe_window_at_first_probe_at(self) -> None:
        sch = Interpolate(
            EXT(), FR(1), interval=10.0, probe_duration=2.0, first_probe_at=5.0
        )
        sch.step(0.0, None)  # anchor at t=0
        out = sch.step(5.0, ResponseEvent(5.0))
        assert out.meta.get("in_probe") is True
        # probe is FR(1) -> reinforces.
        assert out.reinforced

    def test_exits_probe_window_after_probe_duration(self) -> None:
        sch = Interpolate(
            EXT(), FR(1), interval=10.0, probe_duration=2.0, first_probe_at=5.0
        )
        sch.step(0.0, None)
        # Inside probe window at t=5.0.
        assert sch.step(5.0, ResponseEvent(5.0)).meta.get("in_probe") is True
        # After 2s window, back to base (EXT, no reinforce).
        out = sch.step(7.5, ResponseEvent(7.5))
        assert out.meta.get("in_probe") is False
        assert not out.reinforced

    def test_probe_window_repeats_every_interval(self) -> None:
        sch = Interpolate(
            EXT(), FR(1), interval=10.0, probe_duration=2.0, first_probe_at=5.0
        )
        sch.step(0.0, None)
        # First probe at t=5..7
        assert sch.step(5.0, ResponseEvent(5.0)).meta.get("in_probe") is True
        # Outside probe at t=8
        assert sch.step(8.0, ResponseEvent(8.0)).meta.get("in_probe") is False
        # Second probe at t=15..17
        assert sch.step(15.0, ResponseEvent(15.0)).meta.get("in_probe") is True
        assert sch.step(16.5, ResponseEvent(16.5)).meta.get("in_probe") is True
        # Outside probe at t=18
        assert sch.step(18.0, ResponseEvent(18.0)).meta.get("in_probe") is False

    def test_base_state_persists_across_probe(self) -> None:
        sch = Interpolate(
            FR(3), EXT(), interval=10.0, probe_duration=2.0, first_probe_at=5.0
        )
        # 2 base responses pre-probe.
        sch.step(0.0, ResponseEvent(0.0))
        sch.step(1.0, ResponseEvent(1.0))
        # Response during probe: probe is EXT; not reinforced.
        out = sch.step(5.0, ResponseEvent(5.0))
        assert out.meta.get("in_probe") is True
        assert not out.reinforced
        # Response after probe exits: base still has 1 response pending
        # for FR(3) — the single post-probe response completes it.
        out = sch.step(8.0, ResponseEvent(8.0))
        assert out.meta.get("in_probe") is False
        assert out.reinforced

    def test_reset_clears_anchor_and_components(self) -> None:
        sch = Interpolate(
            FR(1), EXT(), interval=10.0, probe_duration=2.0, first_probe_at=5.0
        )
        sch.step(0.0, ResponseEvent(0.0))
        sch.reset()
        # Anchor should re-establish at next step.
        out = sch.step(0.0, ResponseEvent(0.0))
        assert out.meta.get("in_probe") is False
        assert out.reinforced

    def test_non_monotonic_time_raises(self) -> None:
        sch = Interpolate(FR(1), FR(2), interval=10.0, probe_duration=1.0)
        sch.step(1.0, ResponseEvent(1.0))
        with pytest.raises(ScheduleStateError):
            sch.step(0.5, ResponseEvent(0.5))

    def test_event_time_mismatch_raises(self) -> None:
        sch = Interpolate(FR(1), FR(2), interval=10.0, probe_duration=1.0)
        with pytest.raises(ScheduleStateError):
            sch.step(1.0, ResponseEvent(0.9))

    def test_first_probe_at_zero_means_probe_active_from_start(self) -> None:
        sch = Interpolate(
            EXT(), FR(1), interval=10.0, probe_duration=2.0, first_probe_at=0.0
        )
        out = sch.step(0.0, ResponseEvent(0.0))
        assert out.meta.get("in_probe") is True
        assert out.reinforced

    def test_probe_window_boundary_half_open(self) -> None:
        """The window is [probe_start, probe_start + probe_duration)."""
        sch = Interpolate(
            EXT(), FR(1), interval=10.0, probe_duration=2.0, first_probe_at=5.0
        )
        sch.step(0.0, None)
        # Exactly at the boundary t=7 -> outside.
        out = sch.step(7.0, ResponseEvent(7.0))
        assert out.meta.get("in_probe") is False

    def test_probe_accessor_properties(self) -> None:
        base = FR(1)
        probe = FR(2)
        sch = Interpolate(
            base, probe, interval=10.0, probe_duration=2.0, first_probe_at=3.0
        )
        assert sch.base is base
        assert sch.probe is probe
        assert sch.interval == pytest.approx(10.0)
        assert sch.probe_duration == pytest.approx(2.0)
        assert sch.first_probe_at == pytest.approx(3.0)

    def test_meta_passthrough_from_inner_schedule(self) -> None:
        """Inner meta is preserved alongside ``in_probe``."""
        sch = Interpolate(
            FR(1), EXT(), interval=10.0, probe_duration=2.0, first_probe_at=5.0
        )
        out = sch.step(0.0, ResponseEvent(0.0))
        assert "in_probe" in out.meta
