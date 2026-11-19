"""Exception hierarchy for the contingency engine."""

from __future__ import annotations

__all__ = [
    "ContingencyError",
    "ScheduleConfigError",
    "ScheduleStateError",
    "SpecUnrepresentableError",
    "RespondentSpecError",
    "RespondentUnresolvedError",
    "UnknownRespondentElementError",
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


class RespondentSpecError(ContingencyError, ValueError):
    """Raised when a respondent expression or its options are malformed."""


class UnknownRespondentElementError(RespondentSpecError):
    """Raised when a respondent ``type`` is neither Tier A nor registered.

    ``type_name`` holds the type. Tier B types are recognised only through
    the program's :class:`contingency.respondent.RespondentRegistry`.
    """

    def __init__(self, type_name: str) -> None:
        super().__init__(
            f"respondent type '{type_name}' is not Tier A and is not registered "
            "in this program's RespondentRegistry"
        )
        self.type_name = type_name


class RespondentUnresolvedError(ContingencyError, ValueError):
    """Raised when a value a respondent element needs is not supplied.

    ``missing`` names it: an option (``trials``, ``seed``, ``labels.cs``,
    ``labels.us``) or the element's own argument (such as ``p``). The
    expansion never fills it with a default.
    """

    def __init__(self, missing: str) -> None:
        super().__init__(f"unresolved: {missing}")
        self.missing = missing


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
