"""Exception hierarchy for the contingency engine."""

from __future__ import annotations

__all__ = [
    "ContingencyError",
    "ScheduleConfigError",
    "ScheduleStateError",
    "SpecUnrepresentableError",
    "HardwareError",
    "NotConnectedError",
]


class ContingencyError(Exception):
    """Base class for all contingency-py errors."""


class ScheduleConfigError(ContingencyError, ValueError):
    """Raised when a schedule is constructed with invalid parameters."""


class SpecUnrepresentableError(ScheduleConfigError):
    """Raised when a DSL construct has no form in the schedule spec.

    :func:`contingency.bridge.to_spec` raises it instead of dropping a
    construct that would change the contingency (for example a
    directional COD, ``PUNISH``, ``BO``, or ``Lag``).
    """


class ScheduleStateError(ContingencyError, RuntimeError):
    """Raised when :meth:`Schedule.step` is called with inconsistent state
    (e.g. time moving backwards, or a response event timestamped before
    the current schedule time).
    """


class HardwareError(ContingencyError):
    """Raised on I/O failures in the HAL layer (``contingency.hw``).

    Covers I/O errors, malformed configuration, unsupported channels,
    missing optional dependencies, and transport-level failures.
    """


class NotConnectedError(HardwareError, RuntimeError):
    """Raised when HAL operations are attempted on a disconnected apparatus.

    Every HAL backend must raise this for read/write operations invoked
    before :meth:`Apparatus.connect` (or after :meth:`Apparatus.disconnect`).
    """
