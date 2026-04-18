"""Tests for the in-memory :class:`VirtualApparatus` HAL backend."""

from __future__ import annotations

import pytest

from contingency import Reinforcer, ResponseEvent
from contingency.errors import HardwareError, NotConnectedError
from contingency.hw import Apparatus, ApparatusInfo, ApparatusStatus, VirtualApparatus
from contingency.hw.virtual import LoggedEvent

# ---------------------------------------------------------------------------
# construction + info / status
# ---------------------------------------------------------------------------


def test_defaults_match_spec() -> None:
    app = VirtualApparatus()
    assert app.info == ApparatusInfo(
        name="virtual",
        backend="virtual",
        operanda=("main",),
        reinforcers=("food",),
        stimuli=(),
    )
    assert app.status == ApparatusStatus(connected=False, last_error=None)


def test_custom_labels_propagate_to_info() -> None:
    app = VirtualApparatus(
        operanda=("left", "right"),
        reinforcers=("food", "water"),
        stimuli=("house_light",),
        name="sim-box",
    )
    assert app.info.name == "sim-box"
    assert app.info.operanda == ("left", "right")
    assert app.info.reinforcers == ("food", "water")
    assert app.info.stimuli == ("house_light",)


def test_empty_operanda_is_rejected() -> None:
    with pytest.raises(HardwareError, match="operandum"):
        VirtualApparatus(operanda=())


def test_empty_reinforcers_is_rejected() -> None:
    with pytest.raises(HardwareError, match="reinforcer"):
        VirtualApparatus(reinforcers=())


def test_conforms_to_apparatus_protocol() -> None:
    app = VirtualApparatus()
    assert isinstance(app, Apparatus)


# ---------------------------------------------------------------------------
# lifecycle / NotConnectedError
# ---------------------------------------------------------------------------


def test_poll_before_connect_raises() -> None:
    app = VirtualApparatus()
    with pytest.raises(NotConnectedError):
        app.poll_responses(now=0.0)


def test_deliver_before_connect_raises() -> None:
    app = VirtualApparatus()
    with pytest.raises(NotConnectedError):
        app.deliver_reinforcer(now=0.0, reinforcer=Reinforcer(time=0.0))


def test_set_stimulus_before_connect_raises() -> None:
    app = VirtualApparatus(stimuli=("house_light",))
    with pytest.raises(NotConnectedError):
        app.set_stimulus(now=0.0, stimulus="house_light", on=True)


def test_connect_then_disconnect_toggles_status() -> None:
    app = VirtualApparatus()
    app.connect()
    assert app.status.connected is True
    app.disconnect()
    assert app.status.connected is False


def test_connect_is_idempotent() -> None:
    app = VirtualApparatus()
    app.connect()
    app.connect()
    assert app.status.connected is True


def test_disconnect_clears_queue() -> None:
    app = VirtualApparatus()
    app.connect()
    app.press(at=1.0)
    app.press(at=2.0)
    app.disconnect()
    app.connect()
    # queue was dropped on disconnect, so nothing to drain
    assert app.poll_responses(now=3.0) == []


def test_operations_after_disconnect_raise() -> None:
    app = VirtualApparatus()
    app.connect()
    app.disconnect()
    with pytest.raises(NotConnectedError):
        app.poll_responses(now=0.0)


# ---------------------------------------------------------------------------
# press() -> poll() roundtrip
# ---------------------------------------------------------------------------


def test_press_poll_single_event() -> None:
    app = VirtualApparatus()
    app.connect()
    app.press(at=0.5)
    events = app.poll_responses(now=1.0)
    assert events == [ResponseEvent(time=0.5, operandum="main")]


def test_press_poll_preserves_order() -> None:
    app = VirtualApparatus(operanda=("left", "right"))
    app.connect()
    app.press(operandum="left", at=0.1)
    app.press(operandum="right", at=0.2)
    app.press(operandum="left", at=0.3)
    events = app.poll_responses(now=0.4)
    assert [e.operandum for e in events] == ["left", "right", "left"]
    assert [e.time for e in events] == [0.1, 0.2, 0.3]


def test_poll_drains_queue() -> None:
    app = VirtualApparatus()
    app.connect()
    app.press(at=0.1)
    assert len(app.poll_responses(now=1.0)) == 1
    # second poll sees empty queue
    assert app.poll_responses(now=2.0) == []


def test_unknown_operandum_rejected() -> None:
    app = VirtualApparatus(operanda=("left",))
    app.connect()
    with pytest.raises(HardwareError, match="operandum"):
        app.press(operandum="right", at=0.0)


# ---------------------------------------------------------------------------
# deliver_reinforcer
# ---------------------------------------------------------------------------


def test_deliver_appends_to_event_log() -> None:
    app = VirtualApparatus()
    app.connect()
    r = Reinforcer(time=1.0, magnitude=2.5, label="SR+")
    app.deliver_reinforcer(now=1.0, reinforcer=r)
    log = app.event_log
    assert len(log) == 1
    entry = log[0]
    assert isinstance(entry, LoggedEvent)
    assert entry.kind == "reinforcer"
    assert entry.time == 1.0
    assert entry.payload == {"channel": "food", "magnitude": 2.5, "label": "SR+"}


def test_deliver_on_alt_channel() -> None:
    app = VirtualApparatus(reinforcers=("food", "water"))
    app.connect()
    app.deliver_reinforcer(now=0.0, reinforcer=Reinforcer(time=0.0), channel="water")
    assert app.event_log[0].payload["channel"] == "water"


def test_deliver_unknown_channel_rejected() -> None:
    app = VirtualApparatus()
    app.connect()
    with pytest.raises(HardwareError, match="reinforcer channel"):
        app.deliver_reinforcer(
            now=0.0,
            reinforcer=Reinforcer(time=0.0),
            channel="water",
        )


# ---------------------------------------------------------------------------
# set_stimulus
# ---------------------------------------------------------------------------


def test_stimulus_toggle_logged() -> None:
    app = VirtualApparatus(stimuli=("house_light",))
    app.connect()
    app.set_stimulus(now=0.0, stimulus="house_light", on=True)
    app.set_stimulus(now=1.0, stimulus="house_light", on=False)
    kinds = [e.kind for e in app.event_log]
    assert kinds == ["stimulus_on", "stimulus_off"]


def test_stimulus_unknown_rejected() -> None:
    app = VirtualApparatus(stimuli=("house_light",))
    app.connect()
    with pytest.raises(HardwareError, match="stimulus"):
        app.set_stimulus(now=0.0, stimulus="sd_red", on=True)


def test_stimulus_empty_configuration_rejects_all() -> None:
    app = VirtualApparatus()
    app.connect()
    with pytest.raises(HardwareError, match="stimulus"):
        app.set_stimulus(now=0.0, stimulus="house_light", on=True)


# ---------------------------------------------------------------------------
# event log ordering
# ---------------------------------------------------------------------------


def test_event_log_interleaves_all_kinds() -> None:
    app = VirtualApparatus(
        operanda=("left",),
        reinforcers=("food",),
        stimuli=("sd_red",),
    )
    app.connect()
    app.set_stimulus(now=0.0, stimulus="sd_red", on=True)
    app.press(operandum="left", at=0.5)
    app.deliver_reinforcer(now=0.6, reinforcer=Reinforcer(time=0.6))
    app.set_stimulus(now=1.0, stimulus="sd_red", on=False)
    assert [e.kind for e in app.event_log] == [
        "stimulus_on",
        "response",
        "reinforcer",
        "stimulus_off",
    ]


def test_event_log_is_immutable_snapshot() -> None:
    app = VirtualApparatus()
    app.connect()
    app.press(at=0.1)
    snapshot = app.event_log
    app.press(at=0.2)
    # snapshot taken before the second press is unchanged
    assert len(snapshot) == 1
    assert len(app.event_log) == 2
