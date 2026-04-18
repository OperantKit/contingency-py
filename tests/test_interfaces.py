"""Tests for the Schedule protocol."""

from __future__ import annotations

from dataclasses import dataclass

from contingency.entities import Outcome, Reinforcer, ResponseEvent
from contingency.interfaces import Schedule


@dataclass
class _DummySchedule:
    """Minimal Schedule implementation for protocol conformance tests."""

    count: int = 0

    def step(self, now: float, event: ResponseEvent | None = None) -> Outcome:
        if event is None:
            return Outcome()
        self.count += 1
        if self.count >= 3:
            return Outcome(reinforced=True, reinforcer=Reinforcer(time=now))
        return Outcome()

    def reset(self) -> None:
        self.count = 0


def test_dummy_satisfies_protocol() -> None:
    d = _DummySchedule()
    assert isinstance(d, Schedule)


def test_step_accumulates_and_fires() -> None:
    d = _DummySchedule()
    assert d.step(0.0, ResponseEvent(time=0.0)).reinforced is False
    assert d.step(1.0, ResponseEvent(time=1.0)).reinforced is False
    out = d.step(2.0, ResponseEvent(time=2.0))
    assert out.reinforced is True
    assert out.reinforcer is not None
    assert out.reinforcer.time == 2.0


def test_reset_restores_state() -> None:
    d = _DummySchedule()
    d.step(0.0, ResponseEvent(time=0.0))
    d.step(1.0, ResponseEvent(time=1.0))
    d.reset()
    assert d.step(2.0, ResponseEvent(time=2.0)).reinforced is False
