"""Tests for core value objects."""

from __future__ import annotations

import pytest

from contingency.entities import Observation, Outcome, Reinforcer, ResponseEvent


class TestResponseEvent:
    def test_defaults(self) -> None:
        ev = ResponseEvent(time=1.0)
        assert ev.time == 1.0
        assert ev.operandum == "main"

    def test_custom_operandum(self) -> None:
        ev = ResponseEvent(time=2.5, operandum="left-lever")
        assert ev.operandum == "left-lever"

    def test_frozen(self) -> None:
        ev = ResponseEvent(time=1.0)
        with pytest.raises(AttributeError):
            ev.time = 2.0  # type: ignore[misc]


class TestObservation:
    def test_defaults(self) -> None:
        obs = Observation(time=1.0)
        assert obs.response_count == 0

    def test_frozen(self) -> None:
        obs = Observation(time=1.0, response_count=3)
        with pytest.raises(AttributeError):
            obs.response_count = 5  # type: ignore[misc]


class TestReinforcer:
    def test_default_label_and_magnitude(self) -> None:
        r = Reinforcer(time=1.0)
        assert r.label == "SR+"
        assert r.magnitude == 1.0

    def test_punisher_expressed_as_negative_magnitude(self) -> None:
        r = Reinforcer(time=1.0, magnitude=-1.0, label="SR-")
        assert r.magnitude == -1.0
        assert r.label == "SR-"


class TestOutcome:
    def test_non_reinforced_default(self) -> None:
        o = Outcome()
        assert o.reinforced is False
        assert o.reinforcer is None

    def test_reinforced_with_reinforcer(self) -> None:
        r = Reinforcer(time=1.0)
        o = Outcome(reinforced=True, reinforcer=r)
        assert o.reinforced is True
        assert o.reinforcer is r

    def test_reinforced_without_reinforcer_raises(self) -> None:
        with pytest.raises(ValueError, match="requires a Reinforcer"):
            Outcome(reinforced=True)

    def test_non_reinforced_with_reinforcer_raises(self) -> None:
        r = Reinforcer(time=1.0)
        with pytest.raises(ValueError, match="reinforcer must be None"):
            Outcome(reinforced=False, reinforcer=r)

    def test_meta_defaults_empty(self) -> None:
        o = Outcome()
        assert o.meta == {}
