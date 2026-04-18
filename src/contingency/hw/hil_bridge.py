"""TCP/JSONL :class:`Apparatus` backend for Hardware-In-the-Loop rigs.

Wire protocol
-------------

The backend opens a TCP connection to a bridge endpoint (typically
provided by ``contingency-rs``) and exchanges newline-delimited JSON
messages (JSONL). Each message is a single JSON object followed by a
``\\n``.

**Apparatus → host**

``{"type": "response", "time": <float>, "operandum": "<name>"}``

Any message whose ``type`` is not ``"response"`` is silently ignored —
forward-compatible with future message types.

**Host → apparatus**

* ``{"type": "reinforcer", "time": <float>, "channel": "<name>",
  "magnitude": <float>, "label": "<name>"}``
* ``{"type": "stimulus", "time": <float>, "stimulus": "<name>",
  "on": <bool>}``
* ``{"type": "disconnect"}`` sent just before the socket is closed.

The backend only uses the Python :mod:`socket` standard library — no
third-party transport dependency is required.

Forward compatibility
---------------------

* Only the declared keys are required in each message. Additional
  keys are allowed and must be ignored by both sides.
* New ``type`` values must be ignored (rather than errored) so that an
  older client can interoperate with a newer bridge.
* JSON parse errors on a single line are treated as corrupt-on-wire
  and skipped; the surrounding stream continues. A client that needs
  strict validation should layer it on top of this backend.
"""

from __future__ import annotations

import contextlib
import errno
import json
import socket
from typing import Any

from contingency.entities import Reinforcer, ResponseEvent
from contingency.errors import HardwareError, NotConnectedError
from contingency.hw.protocols import ApparatusInfo, ApparatusStatus

__all__ = ["HilBridgeApparatus"]


class HilBridgeApparatus:
    """TCP bridge client for Hardware-In-the-Loop rigs.

    Parameters
    ----------
    host:
        TCP host to connect to. Defaults to ``"127.0.0.1"``.
    port:
        TCP port. Defaults to ``7788``.
    operanda, reinforcers, stimuli:
        Allowed names. Unknown inbound operanda are dropped; unknown
        outbound channels raise :class:`~contingency.errors.HardwareError`.
    name:
        Human-readable identifier reflected in :attr:`info.name`.
    connect_timeout:
        Timeout (in seconds) for :func:`socket.create_connection`.
    read_timeout:
        Timeout (in seconds) for non-blocking polls.
    """

    __slots__ = (
        "_info",
        "_status",
        "_host",
        "_port",
        "_connect_timeout",
        "_read_timeout",
        "_sock",
        "_read_buffer",
    )

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7788,
        operanda: tuple[str, ...] = ("left", "right"),
        reinforcers: tuple[str, ...] = ("food",),
        stimuli: tuple[str, ...] = (),
        name: str = "hil_bridge",
        connect_timeout: float = 1.0,
        read_timeout: float = 0.01,
    ) -> None:
        if not operanda:
            raise HardwareError("at least one operandum must be declared")
        if not reinforcers:
            raise HardwareError("at least one reinforcer channel must be declared")
        self._info = ApparatusInfo(
            name=name,
            backend="hil_bridge",
            operanda=tuple(operanda),
            reinforcers=tuple(reinforcers),
            stimuli=tuple(stimuli),
        )
        self._status = ApparatusStatus(connected=False)
        self._host = host
        self._port = port
        self._connect_timeout = connect_timeout
        self._read_timeout = read_timeout
        self._sock: socket.socket | None = None
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
        if self._sock is not None:
            return
        try:
            sock = socket.create_connection(
                (self._host, self._port),
                timeout=self._connect_timeout,
            )
        except OSError as exc:
            self._status = ApparatusStatus(connected=False, last_error=str(exc))
            raise HardwareError(
                f"failed to connect to HIL bridge {self._host}:{self._port}: {exc}"
            ) from exc
        sock.settimeout(self._read_timeout)
        self._sock = sock
        self._read_buffer = b""
        self._status = ApparatusStatus(connected=True, last_error=None)

    def disconnect(self) -> None:
        if self._sock is None:
            return
        # Best-effort courtesy message; ignore any failures.
        with contextlib.suppress(OSError):
            self._sock.sendall(b'{"type": "disconnect"}\n')
        try:
            self._sock.close()
        except OSError as exc:  # pragma: no cover - depends on kernel state
            self._status = ApparatusStatus(connected=False, last_error=str(exc))
        else:
            self._status = ApparatusStatus(connected=False, last_error=self._status.last_error)
        self._sock = None
        self._read_buffer = b""

    def poll_responses(self, now: float) -> list[ResponseEvent]:
        self._require_connected()
        assert self._sock is not None  # for type-checkers
        try:
            chunk = self._sock.recv(4096)
        except TimeoutError:
            chunk = b""
        except BlockingIOError:  # pragma: no cover - platform-dependent
            chunk = b""
        except OSError as exc:  # pragma: no cover - platform-dependent
            if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                chunk = b""
            else:
                self._status = ApparatusStatus(connected=True, last_error=str(exc))
                raise HardwareError(f"HIL bridge recv failed: {exc}") from exc
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
        message: dict[str, Any] = {
            "type": "reinforcer",
            "time": now,
            "channel": channel,
            "magnitude": reinforcer.magnitude,
            "label": reinforcer.label,
        }
        self._send(message)

    def set_stimulus(self, now: float, stimulus: str, on: bool) -> None:
        self._require_connected()
        if stimulus not in self._info.stimuli:
            raise HardwareError(
                f"unknown stimulus: {stimulus!r}; available: {self._info.stimuli}"
            )
        message: dict[str, Any] = {
            "type": "stimulus",
            "time": now,
            "stimulus": stimulus,
            "on": bool(on),
        }
        self._send(message)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _parse_line(self, raw: bytes, now: float) -> ResponseEvent | None:
        if not raw.strip():
            return None
        try:
            decoded = raw.decode("utf-8")
        except UnicodeDecodeError:  # pragma: no cover - JSON layer usually catches garbage first
            return None
        try:
            obj = json.loads(decoded)
        except json.JSONDecodeError:
            return None
        if not isinstance(obj, dict):  # pragma: no cover - JSON primitives rarely seen
            return None
        if obj.get("type") != "response":
            return None
        operandum = obj.get("operandum")
        if not isinstance(operandum, str):
            return None
        if operandum not in self._info.operanda:
            return None
        raw_time = obj.get("time", now)
        try:
            event_time = float(raw_time)
        except (TypeError, ValueError):
            event_time = now
        return ResponseEvent(time=event_time, operandum=operandum)

    def _send(self, message: dict[str, Any]) -> None:
        assert self._sock is not None  # for type-checkers
        payload = (json.dumps(message, separators=(",", ":")) + "\n").encode("utf-8")
        try:
            self._sock.sendall(payload)
        except OSError as exc:  # pragma: no cover - depends on peer state
            self._status = ApparatusStatus(connected=True, last_error=str(exc))
            raise HardwareError(f"HIL bridge send failed: {exc}") from exc

    def _require_connected(self) -> None:
        if self._sock is None or not self._status.connected:
            raise NotConnectedError("HilBridgeApparatus is not connected")
