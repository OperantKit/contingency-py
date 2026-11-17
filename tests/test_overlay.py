"""Tests for the Overlay compound schedule (Azrin & Holz, 1966)."""

from __future__ import annotations

import pytest

from contingency.entities import Outcome, Reinforcer, ResponseEvent
from contingency.errors import ScheduleConfigError, ScheduleStateError
from contingency.interfaces import Schedule
from contingency.schedules import EXT, FR, Overlay


@pytest.mark.unit
class TestOverlayConfig:
    def test_rejects_non_schedule_base(self) -> None:
        with pytest.raises(ScheduleConfigError):
            Overlay("not a schedule", FR(1))  # type: ignore[arg-type]

    def test_rejects_non_schedule_punisher(self) -> None:
        with pytest.raises(ScheduleConfigError):
            Overlay(FR(1), 42)  # type: ignore[arg-type]

    def test_accepts_two_schedules(self) -> None:
        ov = Overlay(FR(5), FR(10))
        assert isinstance(ov, Schedule)
        assert ov.base is not None
        assert ov.punisher is not None


@pytest.mark.unit
class TestOverlayStep:
    def test_neither_fires_returns_empty_outcome(self) -> None:
        ov = Overlay(FR(5), FR(5))
        out = ov.step(0.0, ResponseEvent(0.0))
        assert not out.reinforced
        assert out.reinforcer is None

    def test_base_only_passes_through_positive_reinforcer(self) -> None:
        ov = Overlay(FR(1), EXT())
        out = ov.step(0.0, ResponseEvent(0.0))
        assert out.reinforced
        assert out.reinforcer is not None
        assert out.reinforcer.magnitude == pytest.approx(1.0)
        assert out.reinforcer.label == "SR+"
        assert out.meta.get("base_fired") is True
        assert out.meta.get("punisher_fired") is False

    def test_punisher_only_emits_negative_labelled_sr_minus(self) -> None:
        # base never fires (EXT); punisher fires on every response (FR1).
        ov = Overlay(EXT(), FR(1))
        out = ov.step(1.0, ResponseEvent(1.0))
        assert out.reinforced
        assert out.reinforcer is not None
        assert out.reinforcer.magnitude == pytest.approx(-1.0)
        assert out.reinforcer.label == "SR-"
        assert out.meta.get("punisher_fired") is True
        assert out.meta.get("base_fired") is False

    def test_both_fire_compound_meta_and_net_magnitude(self) -> None:
        # The schedules of this package deliver magnitude 1, so base and
        # punisher firing together net 0 (label SR0). Nets of other
        # sizes are tested in the Rust engine, whose Overlay this runs.
        ov = Overlay(FR(1), FR(1))
        out = ov.step(0.0, ResponseEvent(0.0))
        assert out.reinforced
        assert out.meta.get("base_fired") is True
        assert out.meta.get("punisher_fired") is True
        assert out.meta.get("net_magnitude") == pytest.approx(0.0)
        assert out.reinforcer is not None
        assert out.reinforcer.magnitude == pytest.approx(0.0)
        assert out.reinforcer.label == "SR0"

    def test_a_punisher_written_in_python_is_rejected(self) -> None:
        class OneShot:
            def step(self, now: float, event: ResponseEvent | None = None) -> Outcome:
                return Outcome(reinforced=True, reinforcer=Reinforcer(time=now, magnitude=0.3))

            def reset(self) -> None:
                pass

        with pytest.raises(ScheduleConfigError, match="Rust engine"):
            Overlay(FR(1), OneShot())

    def test_tick_without_event_returns_no_reinforcement_for_ratio(self) -> None:
        ov = Overlay(FR(2), FR(3))
        out = ov.step(5.0, None)
        assert not out.reinforced

    def test_non_monotonic_time_raises(self) -> None:
        ov = Overlay(FR(1), FR(2))
        ov.step(1.0, ResponseEvent(1.0))
        with pytest.raises(ScheduleStateError):
            ov.step(0.5, ResponseEvent(0.5))

    def test_event_time_mismatch_raises(self) -> None:
        ov = Overlay(FR(1), FR(2))
        with pytest.raises(ScheduleStateError):
            ov.step(1.0, ResponseEvent(0.9))

    def test_reset_clears_both_components(self) -> None:
        ov = Overlay(FR(2), FR(2))
        ov.step(0.0, ResponseEvent(0.0))
        ov.reset()
        # After reset, first response should not reinforce (FR(2) fresh).
        out = ov.step(1.0, ResponseEvent(1.0))
        assert not out.reinforced

    def test_sequence_fr_base_only_pattern(self) -> None:
        ov = Overlay(FR(3), EXT())
        for t in (1.0, 2.0):
            assert not ov.step(t, ResponseEvent(t)).reinforced
        out = ov.step(3.0, ResponseEvent(3.0))
        assert out.reinforced
        assert out.reinforcer is not None
        assert out.reinforcer.label == "SR+"

    def test_sequence_fr_punisher_only_pattern(self) -> None:
        ov = Overlay(EXT(), FR(3))
        for t in (1.0, 2.0):
            assert not ov.step(t, ResponseEvent(t)).reinforced
        out = ov.step(3.0, ResponseEvent(3.0))
        assert out.reinforced
        assert out.reinforcer is not None
        assert out.reinforcer.label == "SR-"

    def test_repeated_cycles_alternate_correctly(self) -> None:
        ov = Overlay(FR(2), FR(3))
        # 2 responses -> base fires (cycle 1).
        ov.step(1.0, ResponseEvent(1.0))
        out = ov.step(2.0, ResponseEvent(2.0))
        assert out.reinforced
        assert out.meta.get("base_fired") is True
        # 3rd response ever = 1st of new cycle for base but 3rd for
        # punisher -> punisher fires on its own.
        out = ov.step(3.0, ResponseEvent(3.0))
        assert out.reinforced
        assert out.meta.get("punisher_fired") is True
        assert out.meta.get("base_fired") is False
