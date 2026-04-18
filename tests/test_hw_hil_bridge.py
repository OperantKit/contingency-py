"""Tests for :class:`HilBridgeApparatus`.

The tests stand up a minimal loopback TCP server on an ephemeral port
and drive the client against it. No external endpoint or contingency-rs
build is required.
"""

from __future__ import annotations

import contextlib
import json
import socket
import threading
import time
from collections.abc import Generator

import pytest

from contingency import Reinforcer
from contingency.errors import HardwareError, NotConnectedError
from contingency.hw import HilBridgeApparatus

# ---------------------------------------------------------------------------
# loopback TCP fixture
# ---------------------------------------------------------------------------


class _LoopbackServer:
    """Tiny accept-one-client TCP echo/record server.

    Listens on an ephemeral port. The first client socket can be
    accessed via :attr:`client_sock` once a connection lands.
    """

    def __init__(self) -> None:
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.bind(("127.0.0.1", 0))
        self._server.listen(1)
        self._server.settimeout(2.0)
        self.host, self.port = self._server.getsockname()
        self.client_sock: socket.socket | None = None
        self.received = b""
        self._accept_thread = threading.Thread(target=self._accept, daemon=True)
        self._accept_thread.start()

    def _accept(self) -> None:
        try:
            sock, _ = self._server.accept()
        except OSError:
            return
        sock.settimeout(2.0)
        self.client_sock = sock

    def wait_client(self, timeout: float = 2.0) -> socket.socket:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.client_sock is not None:
                return self.client_sock
            time.sleep(0.01)
        raise TimeoutError("no client connection within timeout")

    def send(self, data: bytes) -> None:
        sock = self.wait_client()
        sock.sendall(data)

    def drain(self, timeout: float = 0.5) -> bytes:
        sock = self.wait_client()
        sock.settimeout(timeout)
        buf = b""
        try:
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                buf += chunk
        except TimeoutError:
            pass
        return buf

    def close(self) -> None:
        if self.client_sock is not None:
            with contextlib.suppress(OSError):
                self.client_sock.close()
        with contextlib.suppress(OSError):
            self._server.close()


@pytest.fixture
def server() -> Generator[_LoopbackServer, None, None]:
    srv = _LoopbackServer()
    try:
        yield srv
    finally:
        srv.close()


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------


def test_info_backend_tag() -> None:
    app = HilBridgeApparatus()
    assert app.info.backend == "hil_bridge"
    assert app.info.operanda == ("left", "right")


def test_empty_operanda_rejected() -> None:
    with pytest.raises(HardwareError, match="operandum"):
        HilBridgeApparatus(operanda=())


def test_empty_reinforcers_rejected() -> None:
    with pytest.raises(HardwareError, match="reinforcer"):
        HilBridgeApparatus(reinforcers=())


# ---------------------------------------------------------------------------
# lifecycle / NotConnectedError
# ---------------------------------------------------------------------------


def test_poll_before_connect_raises() -> None:
    app = HilBridgeApparatus()
    with pytest.raises(NotConnectedError):
        app.poll_responses(now=0.0)


def test_connect_refused_raises() -> None:
    # Bind an ephemeral port, close immediately so connect is refused.
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    host, port = s.getsockname()
    s.close()
    app = HilBridgeApparatus(host=host, port=port, connect_timeout=0.5)
    with pytest.raises(HardwareError, match="failed to connect"):
        app.connect()
    assert app.status.connected is False
    assert app.status.last_error is not None


def test_connect_success(server: _LoopbackServer) -> None:
    app = HilBridgeApparatus(host=server.host, port=server.port)
    app.connect()
    assert app.status.connected is True
    assert app.status.last_error is None
    app.disconnect()


def test_connect_idempotent(server: _LoopbackServer) -> None:
    app = HilBridgeApparatus(host=server.host, port=server.port)
    app.connect()
    app.connect()
    assert app.status.connected is True
    app.disconnect()


def test_disconnect_without_connect_safe() -> None:
    app = HilBridgeApparatus()
    app.disconnect()
    assert app.status.connected is False


def test_disconnect_sends_disconnect_message(server: _LoopbackServer) -> None:
    app = HilBridgeApparatus(host=server.host, port=server.port)
    app.connect()
    app.disconnect()
    received = server.drain(timeout=0.5)
    lines = [line for line in received.split(b"\n") if line]
    assert any(json.loads(line) == {"type": "disconnect"} for line in lines)


def test_operations_after_disconnect_raise(server: _LoopbackServer) -> None:
    app = HilBridgeApparatus(host=server.host, port=server.port)
    app.connect()
    app.disconnect()
    with pytest.raises(NotConnectedError):
        app.poll_responses(now=0.0)


# ---------------------------------------------------------------------------
# poll_responses parsing
# ---------------------------------------------------------------------------


def test_parse_response_message(server: _LoopbackServer) -> None:
    app = HilBridgeApparatus(host=server.host, port=server.port)
    app.connect()
    server.send(b'{"type": "response", "time": 0.75, "operandum": "left"}\n')
    # poll a few times to let the kernel deliver the bytes
    events = _drain_until(app, expected=1)
    assert len(events) == 1
    assert events[0].operandum == "left"
    assert events[0].time == 0.75
    app.disconnect()


def test_parse_multiple_response_messages(server: _LoopbackServer) -> None:
    app = HilBridgeApparatus(host=server.host, port=server.port)
    app.connect()
    server.send(
        b'{"type": "response", "time": 0.1, "operandum": "left"}\n'
        b'{"type": "response", "time": 0.2, "operandum": "right"}\n'
    )
    events = _drain_until(app, expected=2)
    assert [e.operandum for e in events] == ["left", "right"]
    app.disconnect()


def test_malformed_lines_ignored(server: _LoopbackServer) -> None:
    app = HilBridgeApparatus(host=server.host, port=server.port)
    app.connect()
    server.send(
        b"not json\n"
        b'{"type": "unknown"}\n'
        b'{"type": "response"}\n'                                    # missing operandum
        b'{"type": "response", "operandum": 42}\n'                   # wrong type
        b'{"type": "response", "operandum": "unknown"}\n'            # unknown name
        b'{"type": "response", "operandum": "left", "time": "nope"}\n'  # non-numeric time
        b"\n"
        b'{"type": "response", "operandum": "left", "time": 1.0}\n'
    )
    events = _drain_until(app, expected=2)
    # both the bad-time message (falls back to `now`) and the ok message
    # produce valid events
    assert len(events) == 2
    assert all(e.operandum == "left" for e in events)
    app.disconnect()


def test_partial_line_buffered(server: _LoopbackServer) -> None:
    app = HilBridgeApparatus(host=server.host, port=server.port)
    app.connect()
    server.send(b'{"type": "response", "time": 1.0, "ope')
    assert app.poll_responses(now=0.0) == []
    server.send(b'randum": "left"}\n')
    events = _drain_until(app, expected=1)
    assert events[0].operandum == "left"
    app.disconnect()


def test_poll_with_no_data_returns_empty_list(server: _LoopbackServer) -> None:
    # With a quiet peer, the socket's read timeout fires and poll
    # must return an empty list rather than raising.
    app = HilBridgeApparatus(
        host=server.host,
        port=server.port,
        read_timeout=0.02,
    )
    app.connect()
    # wait a moment so the server-side accept has completed
    server.wait_client()
    assert app.poll_responses(now=0.0) == []
    app.disconnect()


# ---------------------------------------------------------------------------
# deliver_reinforcer / set_stimulus
# ---------------------------------------------------------------------------


def test_deliver_reinforcer_sends_json(server: _LoopbackServer) -> None:
    app = HilBridgeApparatus(host=server.host, port=server.port)
    app.connect()
    app.deliver_reinforcer(
        now=1.5,
        reinforcer=Reinforcer(time=1.5, magnitude=2.0, label="SR+"),
    )
    app.disconnect()
    received = server.drain(timeout=0.5)
    msgs = [json.loads(line) for line in received.split(b"\n") if line]
    reinforcer_msgs = [m for m in msgs if m["type"] == "reinforcer"]
    assert reinforcer_msgs == [
        {
            "type": "reinforcer",
            "time": 1.5,
            "channel": "food",
            "magnitude": 2.0,
            "label": "SR+",
        }
    ]


def test_deliver_unknown_channel_rejected(server: _LoopbackServer) -> None:
    app = HilBridgeApparatus(host=server.host, port=server.port)
    app.connect()
    with pytest.raises(HardwareError, match="reinforcer channel"):
        app.deliver_reinforcer(
            now=0.0,
            reinforcer=Reinforcer(time=0.0),
            channel="water",
        )
    app.disconnect()


def test_set_stimulus_sends_json(server: _LoopbackServer) -> None:
    app = HilBridgeApparatus(
        host=server.host,
        port=server.port,
        stimuli=("house_light",),
    )
    app.connect()
    app.set_stimulus(now=0.25, stimulus="house_light", on=True)
    app.set_stimulus(now=0.30, stimulus="house_light", on=False)
    app.disconnect()
    received = server.drain(timeout=0.5)
    msgs = [json.loads(line) for line in received.split(b"\n") if line]
    stim = [m for m in msgs if m["type"] == "stimulus"]
    assert stim == [
        {"type": "stimulus", "time": 0.25, "stimulus": "house_light", "on": True},
        {"type": "stimulus", "time": 0.30, "stimulus": "house_light", "on": False},
    ]


def test_set_stimulus_unknown_rejected(server: _LoopbackServer) -> None:
    app = HilBridgeApparatus(host=server.host, port=server.port, stimuli=("house_light",))
    app.connect()
    with pytest.raises(HardwareError, match="stimulus"):
        app.set_stimulus(now=0.0, stimulus="sd_red", on=True)
    app.disconnect()


def test_deliver_before_connect_raises() -> None:
    app = HilBridgeApparatus()
    with pytest.raises(NotConnectedError):
        app.deliver_reinforcer(now=0.0, reinforcer=Reinforcer(time=0.0))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _drain_until(
    app: HilBridgeApparatus,
    *,
    expected: int,
    timeout: float = 2.0,
) -> list:
    """Poll repeatedly until ``expected`` events have arrived or timeout."""
    events: list = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and len(events) < expected:
        events.extend(app.poll_responses(now=0.0))
        if len(events) < expected:
            time.sleep(0.01)
    return events
