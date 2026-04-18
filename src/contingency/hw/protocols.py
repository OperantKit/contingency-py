"""HAL contract: :class:`Apparatus` protocol plus descriptor value types.

An :class:`Apparatus` exposes three families of operations:

* **lifecycle** — :meth:`connect` / :meth:`disconnect`
* **input** — :meth:`poll_responses` drains pending
  :class:`~contingency.entities.ResponseEvent` values
* **output** — :meth:`deliver_reinforcer` fires a reinforcer,
  :meth:`set_stimulus` toggles a stimulus line

The protocol is synchronous and polling-based. Any async / event-loop
integration belongs in a higher-level session runner.

All backends raise :class:`~contingency.errors.NotConnectedError` for
input/output calls made before :meth:`connect` (or after
:meth:`disconnect`) and :class:`~contingency.errors.HardwareError` for
unknown channel names, malformed I/O, and transport-level failures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from contingency.entities import Reinforcer, ResponseEvent

__all__ = [
    "Apparatus",
    "ApparatusInfo",
    "ApparatusStatus",
]


@dataclass(frozen=True)
class ApparatusInfo:
    """Static description of an apparatus.

    Attributes
    ----------
    name:
        Human-readable identifier (appears in logs / dashboards).
    backend:
        Backend tag — one of ``"virtual"``, ``"serial"``, ``"hil_bridge"``.
        Session recorders use this for provenance.
    operanda:
        Tuple of operandum identifiers that this apparatus can emit
        response events for (e.g. ``("left", "right")``).
    reinforcers:
        Tuple of reinforcer channel names that
        :meth:`Apparatus.deliver_reinforcer` will accept.
    stimuli:
        Tuple of stimulus identifiers (lights, tones, cue signals) that
        :meth:`Apparatus.set_stimulus` will accept.
    """

    name: str
    backend: str
    operanda: tuple[str, ...]
    reinforcers: tuple[str, ...]
    stimuli: tuple[str, ...]


@dataclass(frozen=True)
class ApparatusStatus:
    """Runtime connection state for an apparatus.

    Attributes
    ----------
    connected:
        ``True`` between successful :meth:`Apparatus.connect` and
        :meth:`Apparatus.disconnect` calls.
    last_error:
        Human-readable message for the most recent transport error,
        or ``None`` if none has occurred since the last reset.
    """

    connected: bool
    last_error: str | None = None


@runtime_checkable
class Apparatus(Protocol):
    """Hardware-agnostic operant-box interface.

    Backends must implement every method below. ``now`` arguments are
    caller-supplied monotonic timestamps (unit-agnostic, same convention
    as :class:`contingency.entities.ResponseEvent`). HAL does not read
    a wall clock; the caller chooses the time source.
    """

    @property
    def info(self) -> ApparatusInfo:
        """Static description of this apparatus."""
        ...

    @property
    def status(self) -> ApparatusStatus:
        """Current connection state and last error (if any)."""
        ...

    def connect(self) -> None:
        """Open the transport and make the apparatus ready for I/O.

        Raises :class:`~contingency.errors.HardwareError` if the
        transport cannot be opened.
        """
        ...

    def disconnect(self) -> None:
        """Close the transport. Idempotent — safe to call when already
        disconnected.
        """
        ...

    def poll_responses(self, now: float) -> list[ResponseEvent]:
        """Drain pending response events.

        Returns an empty list if no responses are currently buffered.
        Non-blocking; the backend decides how long it is willing to
        wait for fresh data (typically a small read timeout).

        Raises :class:`~contingency.errors.NotConnectedError` when the
        apparatus is not connected.
        """
        ...

    def deliver_reinforcer(
        self,
        now: float,
        reinforcer: Reinforcer,
        channel: str = "food",
    ) -> None:
        """Fire a reinforcer on the named channel.

        ``reinforcer.magnitude`` and ``reinforcer.label`` are forwarded
        to the backend; interpretation is backend-specific (pellet
        count, ml, points, etc.).

        Raises :class:`~contingency.errors.HardwareError` if the
        channel is unknown, and :class:`~contingency.errors.NotConnectedError`
        if the apparatus is not connected.
        """
        ...

    def set_stimulus(self, now: float, stimulus: str, on: bool) -> None:
        """Turn a stimulus (light, tone, cue) on or off.

        Raises :class:`~contingency.errors.HardwareError` if the
        stimulus name is unknown, and
        :class:`~contingency.errors.NotConnectedError` if the apparatus
        is not connected.
        """
        ...
