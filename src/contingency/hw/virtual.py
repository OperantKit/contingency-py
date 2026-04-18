"""In-memory :class:`Apparatus` backend for tests and offline simulation.

The caller injects synthetic responses via :meth:`VirtualApparatus.press`
and drains them on the next :meth:`VirtualApparatus.poll_responses` call.
Deliveries and stimulus toggles append to :attr:`event_log`, which
exposes a full, ordered trace of every interaction for assertions.

The backend is deliberately deterministic: no random IRT generator, no
clock, no latency. Those concerns belong to higher layers — e.g. a
subject simulator that pushes presses via :meth:`press`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from contingency.entities import Reinforcer, ResponseEvent
from contingency.errors import HardwareError, NotConnectedError
from contingency.hw.protocols import ApparatusInfo, ApparatusStatus

__all__ = ["VirtualApparatus", "LoggedEvent"]


@dataclass(frozen=True)
class LoggedEvent:
    """A single entry in :attr:`VirtualApparatus.event_log`.

    Attributes
    ----------
    time:
        Monotonic timestamp of the event (unit-agnostic).
    kind:
        One of ``"response"``, ``"reinforcer"``, ``"stimulus_on"``,
        ``"stimulus_off"``.
    payload:
        Backend-specific details. For ``"response"`` events the payload
        includes ``{"operandum": <name>}``. For ``"reinforcer"`` events
        it includes ``{"channel": <name>, "magnitude": <float>,
        "label": <str>}``. For ``"stimulus_*"`` events it includes
        ``{"stimulus": <name>}``.
    """

    time: float
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)


class VirtualApparatus:
    """In-memory operant-box simulator.

    Parameters
    ----------
    operanda:
        Valid operandum identifiers for :meth:`press`. Defaults to a
        single ``"main"`` operandum.
    reinforcers:
        Valid channels accepted by :meth:`deliver_reinforcer`.
    stimuli:
        Valid stimulus identifiers accepted by :meth:`set_stimulus`.
        An empty tuple disables stimulus delivery.
    name:
        Human-readable identifier reflected in :attr:`info.name`.
    """

    __slots__ = (
        "_info",
        "_status",
        "_queue",
        "_log",
    )

    def __init__(
        self,
        operanda: tuple[str, ...] = ("main",),
        reinforcers: tuple[str, ...] = ("food",),
        stimuli: tuple[str, ...] = (),
        name: str = "virtual",
    ) -> None:
        if not operanda:
            raise HardwareError("at least one operandum must be declared")
        if not reinforcers:
            raise HardwareError("at least one reinforcer channel must be declared")
        self._info = ApparatusInfo(
            name=name,
            backend="virtual",
            operanda=tuple(operanda),
            reinforcers=tuple(reinforcers),
            stimuli=tuple(stimuli),
        )
        self._status = ApparatusStatus(connected=False)
        self._queue: deque[ResponseEvent] = deque()
        self._log: list[LoggedEvent] = []

    # ------------------------------------------------------------------
    # Apparatus protocol
    # ------------------------------------------------------------------

    @property
    def info(self) -> ApparatusInfo:
        return self._info

    @property
    def status(self) -> ApparatusStatus:
        return self._status

    def connect(self) -> None:
        """Mark the apparatus as connected.

        Virtual connect is in-memory only. Idempotent: calling twice is
        harmless. Previously queued presses survive reconnect.
        """
        self._status = ApparatusStatus(connected=True, last_error=None)

    def disconnect(self) -> None:
        """Mark the apparatus as disconnected and drop queued responses.

        Dropping the queue matches the semantics of real transports:
        once the line is closed, in-flight bytes are gone.
        """
        self._status = ApparatusStatus(connected=False, last_error=self._status.last_error)
        self._queue.clear()

    def poll_responses(self, now: float) -> list[ResponseEvent]:
        self._require_connected()
        if not self._queue:
            return []
        drained = list(self._queue)
        self._queue.clear()
        return drained

    def deliver_reinforcer(
        self,
        now: float,
        reinforcer: Reinforcer,
        channel: str = "food",
    ) -> None:
        self._require_connected()
        if channel not in self._info.reinforcers:
            raise HardwareError(
                f"unknown reinforcer channel: {channel!r}; "
                f"available: {self._info.reinforcers}"
            )
        self._log.append(
            LoggedEvent(
                time=now,
                kind="reinforcer",
                payload={
                    "channel": channel,
                    "magnitude": reinforcer.magnitude,
                    "label": reinforcer.label,
                },
            )
        )

    def set_stimulus(self, now: float, stimulus: str, on: bool) -> None:
        self._require_connected()
        if stimulus not in self._info.stimuli:
            raise HardwareError(
                f"unknown stimulus: {stimulus!r}; available: {self._info.stimuli}"
            )
        self._log.append(
            LoggedEvent(
                time=now,
                kind="stimulus_on" if on else "stimulus_off",
                payload={"stimulus": stimulus},
            )
        )

    # ------------------------------------------------------------------
    # VirtualApparatus-specific
    # ------------------------------------------------------------------

    def press(self, operandum: str = "main", *, at: float) -> None:
        """Enqueue a response event for the next :meth:`poll_responses`.

        Parameters
        ----------
        operandum:
            Must appear in :attr:`info.operanda`. Unknown names raise
            :class:`~contingency.errors.HardwareError`.
        at:
            Keyword-only; the timestamp to stamp on the ``ResponseEvent``.
        """
        if operandum not in self._info.operanda:
            raise HardwareError(
                f"unknown operandum: {operandum!r}; "
                f"available: {self._info.operanda}"
            )
        event = ResponseEvent(time=at, operandum=operandum)
        self._queue.append(event)
        self._log.append(
            LoggedEvent(
                time=at,
                kind="response",
                payload={"operandum": operandum},
            )
        )

    @property
    def event_log(self) -> tuple[LoggedEvent, ...]:
        """Immutable snapshot of every delivery, stimulus, and response
        seen by this apparatus, in insertion order."""
        return tuple(self._log)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _require_connected(self) -> None:
        if not self._status.connected:
            raise NotConnectedError("VirtualApparatus is not connected")
