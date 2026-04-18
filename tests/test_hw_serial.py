"""Tests for :class:`SerialApparatus`.

The tests use a fully mocked ``serial.Serial`` (see :class:`_FakeSerial`),
so neither pyserial nor physical hardware is required in CI. A tiny
stub ``serial`` module is installed for the duration of the test file
when pyserial is not available on the interpreter.
"""

from __future__ import annotations

import sys
import types

import pytest

from contingency import Reinforcer
from contingency.errors import HardwareError, NotConnectedError

# ---------------------------------------------------------------------------
# pyserial stub & module-level reload helper
# ---------------------------------------------------------------------------


class _FakeSerial:
    """Stand-in for ``serial.Serial`` with an injectable read buffer."""

    instances: list[_FakeSerial] = []

    def __init__(self, port: str, baudrate: int, timeout: float) -> None:
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.closed = False
        self.writes: list[bytes] = []
        self._read_queue: list[bytes] = []
        self.flush_count = 0
        _FakeSerial.instances.append(self)

    def queue_read(self, data: bytes) -> None:
        self._read_queue.append(data)

    def read(self, size: int) -> bytes:
        if not self._read_queue:
            return b""
        return self._read_queue.pop(0)

    def write(self, data: bytes) -> int:
        self.writes.append(data)
        return len(data)

    def flush(self) -> None:
        self.flush_count += 1

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_serial_backend(monkeypatch: pytest.MonkeyPatch) -> type[_FakeSerial]:
    """Install ``_FakeSerial`` as the active ``serial.Serial`` implementation.

    The fixture is compatible with installs where pyserial is actually
    present: we only patch the class attribute that
    :mod:`contingency.hw.serial_backend` references.
    """
    _FakeSerial.instances = []

    # Ensure a `serial` module object exists in sys.modules so that
    # the backend module's `serial.Serial` attribute access resolves.
    module: types.ModuleType
    if "serial" not in sys.modules:
        module = types.ModuleType("serial")
        sys.modules["serial"] = module
    else:
        module = sys.modules["serial"]
    monkeypatch.setattr(module, "Serial", _FakeSerial, raising=False)

    # Re-import the backend so that _HAS_PYSERIAL + bound `serial` name
    # reflect the fresh stub.
    from contingency.hw import serial_backend as backend

    monkeypatch.setattr(backend, "serial", module, raising=True)
    monkeypatch.setattr(backend, "_HAS_PYSERIAL", True, raising=True)

    return _FakeSerial


# ---------------------------------------------------------------------------
# missing pyserial branch
# ---------------------------------------------------------------------------


def test_construction_without_pyserial_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    import contingency.hw.serial_backend as backend

    monkeypatch.setattr(backend, "_HAS_PYSERIAL", False, raising=True)
    with pytest.raises(HardwareError, match="pyserial is required"):
        backend.SerialApparatus(port="/dev/null")


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------


def test_empty_operanda_rejected(fake_serial_backend: type[_FakeSerial]) -> None:
    from contingency.hw import SerialApparatus

    with pytest.raises(HardwareError, match="operandum"):
        SerialApparatus(port="/dev/ttyACM0", operanda=())


def test_empty_reinforcers_rejected(fake_serial_backend: type[_FakeSerial]) -> None:
    from contingency.hw import SerialApparatus

    with pytest.raises(HardwareError, match="reinforcer"):
        SerialApparatus(port="/dev/ttyACM0", reinforcers=())


def test_info_backend_tag(fake_serial_backend: type[_FakeSerial]) -> None:
    from contingency.hw import SerialApparatus

    app = SerialApparatus(port="/dev/ttyACM0")
    assert app.info.backend == "serial"
    assert app.info.operanda == ("left", "right")
    assert app.info.reinforcers == ("food",)


# ---------------------------------------------------------------------------
# lifecycle / NotConnectedError
# ---------------------------------------------------------------------------


def test_poll_before_connect_raises(fake_serial_backend: type[_FakeSerial]) -> None:
    from contingency.hw import SerialApparatus

    app = SerialApparatus(port="/dev/ttyACM0")
    with pytest.raises(NotConnectedError):
        app.poll_responses(now=0.0)


def test_connect_opens_serial_with_config(fake_serial_backend: type[_FakeSerial]) -> None:
    from contingency.hw import SerialApparatus

    app = SerialApparatus(port="/dev/ttyACM0", baudrate=9600, read_timeout=0.05)
    app.connect()
    assert app.status.connected is True
    assert len(fake_serial_backend.instances) == 1
    inst = fake_serial_backend.instances[0]
    assert inst.port == "/dev/ttyACM0"
    assert inst.baudrate == 9600
    assert inst.timeout == 0.05


def test_connect_idempotent(fake_serial_backend: type[_FakeSerial]) -> None:
    from contingency.hw import SerialApparatus

    app = SerialApparatus(port="/dev/ttyACM0")
    app.connect()
    app.connect()
    assert len(fake_serial_backend.instances) == 1


def test_disconnect_closes_serial(fake_serial_backend: type[_FakeSerial]) -> None:
    from contingency.hw import SerialApparatus

    app = SerialApparatus(port="/dev/ttyACM0")
    app.connect()
    app.disconnect()
    assert fake_serial_backend.instances[0].closed is True
    assert app.status.connected is False


def test_disconnect_without_connect_is_safe(fake_serial_backend: type[_FakeSerial]) -> None:
    from contingency.hw import SerialApparatus

    app = SerialApparatus(port="/dev/ttyACM0")
    app.disconnect()
    assert app.status.connected is False


# ---------------------------------------------------------------------------
# poll_responses parsing
# ---------------------------------------------------------------------------


def test_parse_single_response(fake_serial_backend: type[_FakeSerial]) -> None:
    from contingency.hw import SerialApparatus

    app = SerialApparatus(port="/dev/ttyACM0")
    app.connect()
    fake_serial_backend.instances[0].queue_read(b"R left\n")
    events = app.poll_responses(now=1.25)
    assert len(events) == 1
    assert events[0].operandum == "left"
    assert events[0].time == 1.25


def test_parse_multiple_lines_single_read(fake_serial_backend: type[_FakeSerial]) -> None:
    from contingency.hw import SerialApparatus

    app = SerialApparatus(port="/dev/ttyACM0")
    app.connect()
    fake_serial_backend.instances[0].queue_read(b"R left\nR right\n")
    events = app.poll_responses(now=0.0)
    assert [e.operandum for e in events] == ["left", "right"]


def test_partial_line_buffered(fake_serial_backend: type[_FakeSerial]) -> None:
    from contingency.hw import SerialApparatus

    app = SerialApparatus(port="/dev/ttyACM0")
    app.connect()
    inst = fake_serial_backend.instances[0]
    inst.queue_read(b"R le")
    # no full line yet
    assert app.poll_responses(now=0.0) == []
    inst.queue_read(b"ft\n")
    events = app.poll_responses(now=0.0)
    assert [e.operandum for e in events] == ["left"]


def test_malformed_lines_ignored(fake_serial_backend: type[_FakeSerial]) -> None:
    from contingency.hw import SerialApparatus

    app = SerialApparatus(port="/dev/ttyACM0")
    app.connect()
    fake_serial_backend.instances[0].queue_read(
        b"garbage\n"
        b"X left\n"          # wrong verb
        b"R\n"               # missing operandum
        b"R left extra\n"    # too many tokens
        b"R unknown\n"       # operandum not declared
        b"\n"                # empty line
        b"R left\n"          # ok
    )
    events = app.poll_responses(now=0.0)
    assert [e.operandum for e in events] == ["left"]


def test_empty_read_returns_empty_list(fake_serial_backend: type[_FakeSerial]) -> None:
    from contingency.hw import SerialApparatus

    app = SerialApparatus(port="/dev/ttyACM0")
    app.connect()
    assert app.poll_responses(now=0.0) == []


def test_non_ascii_line_dropped(fake_serial_backend: type[_FakeSerial]) -> None:
    from contingency.hw import SerialApparatus

    app = SerialApparatus(port="/dev/ttyACM0")
    app.connect()
    # invalid ASCII (high bit) in a full line
    fake_serial_backend.instances[0].queue_read(b"\xff\xfe\n")
    assert app.poll_responses(now=0.0) == []


# ---------------------------------------------------------------------------
# deliver_reinforcer / set_stimulus
# ---------------------------------------------------------------------------


def test_deliver_writes_wire_format(fake_serial_backend: type[_FakeSerial]) -> None:
    from contingency.hw import SerialApparatus

    app = SerialApparatus(port="/dev/ttyACM0")
    app.connect()
    app.deliver_reinforcer(now=0.0, reinforcer=Reinforcer(time=0.0, magnitude=1.0))
    inst = fake_serial_backend.instances[0]
    assert inst.writes == [b"D food 1.0\n"]
    assert inst.flush_count == 1


def test_deliver_unknown_channel_rejected(fake_serial_backend: type[_FakeSerial]) -> None:
    from contingency.hw import SerialApparatus

    app = SerialApparatus(port="/dev/ttyACM0")
    app.connect()
    with pytest.raises(HardwareError, match="reinforcer channel"):
        app.deliver_reinforcer(
            now=0.0,
            reinforcer=Reinforcer(time=0.0),
            channel="water",
        )


def test_set_stimulus_writes_wire_format(fake_serial_backend: type[_FakeSerial]) -> None:
    from contingency.hw import SerialApparatus

    app = SerialApparatus(port="/dev/ttyACM0", stimuli=("house_light",))
    app.connect()
    app.set_stimulus(now=0.0, stimulus="house_light", on=True)
    app.set_stimulus(now=0.1, stimulus="house_light", on=False)
    inst = fake_serial_backend.instances[0]
    assert inst.writes == [b"S house_light 1\n", b"S house_light 0\n"]


def test_set_stimulus_unknown_rejected(fake_serial_backend: type[_FakeSerial]) -> None:
    from contingency.hw import SerialApparatus

    app = SerialApparatus(port="/dev/ttyACM0", stimuli=("house_light",))
    app.connect()
    with pytest.raises(HardwareError, match="stimulus"):
        app.set_stimulus(now=0.0, stimulus="sd_red", on=True)


def test_deliver_before_connect_raises(fake_serial_backend: type[_FakeSerial]) -> None:
    from contingency.hw import SerialApparatus

    app = SerialApparatus(port="/dev/ttyACM0")
    with pytest.raises(NotConnectedError):
        app.deliver_reinforcer(now=0.0, reinforcer=Reinforcer(time=0.0))
