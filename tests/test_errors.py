"""Tests for the exception hierarchy."""

from __future__ import annotations

import pytest

from contingency.errors import (
    ContingencyError,
    ScheduleConfigError,
    ScheduleStateError,
)


def test_hierarchy() -> None:
    assert issubclass(ScheduleConfigError, ContingencyError)
    assert issubclass(ScheduleStateError, ContingencyError)
    assert issubclass(ScheduleConfigError, ValueError)
    assert issubclass(ScheduleStateError, RuntimeError)


def test_raise_config_error() -> None:
    with pytest.raises(ScheduleConfigError, match="bad n"):
        raise ScheduleConfigError("bad n")


def test_raise_state_error() -> None:
    with pytest.raises(ScheduleStateError, match="time went backwards"):
        raise ScheduleStateError("time went backwards")
