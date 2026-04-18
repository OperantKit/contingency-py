"""pyserial-backed :class:`Apparatus` backend.

Wire protocol
-------------

The module assumes an ASCII, line-oriented protocol where every frame
is terminated by a single ``\\n`` (LF). Firmware is expected to emit
one event per line.

**Apparatus → host (ignored lines silently dropped)**

* ``R <operandum>`` — the subject produced a response on the named
  operandum (for example ``R left``).

**Host → apparatus**

* ``D <channel> <magnitude>`` — deliver a reinforcer on ``channel``
  with the given magnitude (rendered with ``repr``-like float
  precision).
* ``S <stimulus> <0|1>`` — set the stimulus state (``1`` = on,
  ``0`` = off).

The backend does no retries, no reconnection, and no handshake. Those
are session-level concerns and belong to higher layers.

Importing this module does **not** require ``pyserial``: the dependency
is guarded with a ``try/except ImportError`` so ``import contingency``
succeeds on a minimal install. Constructing :class:`SerialApparatus`
without ``pyserial`` raises :class:`~contingency.errors.HardwareError`.
"""

from __future__ import annotations

from typing import Any

from contingency.entities import Reinforcer, ResponseEvent
from contingency.errors import HardwareError, NotConnectedError
from contingency.hw.protocols import ApparatusInfo, ApparatusStatus

__all__ = ["SerialApparatus"]

try:
    import serial

    _HAS_PYSERIAL = True  # pragma: no cover - exercised when pyserial is installed
except ImportError:  # pragma: no cover - exercised when pyserial is not installed
    serial = None
    _HAS_PYSERIAL = False


_PYSERIAL_MISSING_MSG = (
    "pyserial is required for SerialApparatus; "
    "install with `pip install contingency-py[hw-serial]`"
)


class SerialApparatus:
    """Line-oriented serial transport for operant-box hardware.

    Parameters
    ----------
    port:
        Device path (``/dev/ttyACM0``, ``COM3``, …) passed straight to
        ``serial.Serial``.
    baudrate:
        UART baud rate. Defaults to ``115200``.
    operanda, reinforcers, stimuli:
        Allowed names for response events, reinforcer channels, and
        stimulus lines. Incoming events for unknown operanda are
        silently dropped; outbound calls for unknown channels raise
        :class:`~contingency.errors.HardwareError`.
    name:
        Human-readable identifier reflected in :attr:`info.name`.
    read_timeout:
        Timeout (in seconds) passed to ``serial.Serial`` for
        non-blocking reads.
    """

    __slots__ = (
        "_info",
        "_status",
        "_port_name",
        "_baudrate",
        "_read_timeout",
        "_serial",
        "_read_buffer",
    )

    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        operanda: tuple[str, ...] = ("left", "right"),
        reinforcers: tuple[str, ...] = ("food",),
        stimuli: tuple[str, ...] = (),
        name: str = "serial",
        read_timeout: float = 0.01,
    ) -> None:
        if not _HAS_PYSERIAL:
            raise HardwareError(_PYSERIAL_MISSING_MSG)
        if not operanda:
            raise HardwareError("at least one operandum must be declared")
        if not reinforcers:
            raise HardwareError("at least one reinforcer channel must be declared")
        self._info = ApparatusInfo(
            name=name,
            backend="serial",
            operanda=tuple(operanda),
            reinforcers=tuple(reinforcers),
            stimuli=tuple(stimuli),
        )
        self._status = ApparatusStatus(connected=False)
        self._port_name = port
        self._baudrate = baudrate
        self._read_timeout = read_timeout
        self._serial: Any = None
        self._read_buffer = b""

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
        if self._serial is not None:
            return
        try:
            self._serial = serial.Serial(
                port=self._port_name,
                baudrate=self._baudrate,
                timeout=self._read_timeout,
            )
        except Exception as exc:  # pragma: no cover - pyserial-specific
            self._status = ApparatusStatus(connected=False, last_error=str(exc))
            raise HardwareError(f"failed to open serial port {self._port_name!r}: {exc}") from exc
        self._status = ApparatusStatus(connected=True, last_error=None)
        self._read_buffer = b""

    def disconnect(self) -> None:
        if self._serial is None:
            return
        try:
            self._serial.close()
        except Exception as exc:  # pragma: no cover - pyserial-specific
            self._status = ApparatusStatus(connected=False, last_error=str(exc))
        else:
            self._status = ApparatusStatus(connected=False, last_error=self._status.last_error)
        self._serial = None
        self._read_buffer = b""

    def poll_responses(self, now: float) -> list[ResponseEvent]:
        self._require_connected()
        assert self._serial is not None  # for type-checkers
        try:
            chunk = self._serial.read(4096)
        except Exception as exc:  # pragma: no cover - pyserial-specific
            self._status = ApparatusStatus(connected=True, last_error=str(exc))
            raise HardwareError(f"serial read failed: {exc}") from exc
        if chunk:
            self._read_buffer += chunk
        events: list[ResponseEvent] = []
        while b"\n" in self._read_buffer:
            line, self._read_buffer = self._read_buffer.split(b"\n", 1)
            event = self._parse_line(line, now)
            if event is not None:
                events.append(event)
        return events

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
        payload = f"D {channel} {reinforcer.magnitude}\n".encode("ascii")
        self._write(payload)

    def set_stimulus(self, now: float, stimulus: str, on: bool) -> None:
        self._require_connected()
        if stimulus not in self._info.stimuli:
            raise HardwareError(
                f"unknown stimulus: {stimulus!r}; available: {self._info.stimuli}"
            )
        payload = f"S {stimulus} {1 if on else 0}\n".encode("ascii")
        self._write(payload)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _parse_line(self, raw: bytes, now: float) -> ResponseEvent | None:
        try:
            text = raw.decode("ascii", errors="strict").strip()
        except UnicodeDecodeError:
            return None
        if not text:
            return None
        parts = text.split()
        if len(parts) != 2 or parts[0] != "R":
            return None
        operandum = parts[1]
        if operandum not in self._info.operanda:
            return None
        return ResponseEvent(time=now, operandum=operandum)

    def _write(self, payload: bytes) -> None:
        assert self._serial is not None  # for type-checkers
        try:
            self._serial.write(payload)
            flush = getattr(self._serial, "flush", None)
            if callable(flush):
                flush()
        except Exception as exc:  # pragma: no cover - pyserial-specific
            self._status = ApparatusStatus(connected=True, last_error=str(exc))
            raise HardwareError(f"serial write failed: {exc}") from exc

    def _require_connected(self) -> None:
        if self._serial is None or not self._status.connected:
            raise NotConnectedError("SerialApparatus is not connected")
