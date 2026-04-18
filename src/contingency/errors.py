"""Exception hierarchy for the contingency engine."""

from __future__ import annotations

__all__ = [
    "ContingencyError",
    "ScheduleConfigError",
    "ScheduleStateError",
]


class ContingencyError(Exception):
    """Base class for all contingency-py errors."""


class ScheduleConfigError(ContingencyError, ValueError):
    """Raised when a schedule is constructed with invalid parameters."""


class ScheduleStateError(ContingencyError, RuntimeError):
    """Raised when :meth:`Schedule.step` is called with inconsistent state
    (e.g. time moving backwards, or a response event timestamped before
    the current schedule time).
    """
