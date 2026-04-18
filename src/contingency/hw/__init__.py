"""Hardware Abstraction Layer for the contingency engine.

A thin, synchronous I/O interface for reading response events from an
operant apparatus and delivering reinforcers / stimuli. The HAL is
deliberately decoupled from :class:`contingency.Schedule`: a session
runner is responsible for feeding ``poll_responses`` output into
``Schedule.step`` and delivering ``Outcome.reinforcer`` payloads back
to the apparatus.

Backends
--------

* :class:`VirtualApparatus` — in-memory, fully deterministic; used by
  tests and offline simulation.
* :class:`SerialApparatus` — pyserial-backed transport. Requires the
  ``hw-serial`` optional dependency. Not re-exported at the top level
  of :mod:`contingency` so the base package stays importable without
  pyserial installed.
* :class:`HilBridgeApparatus` — TCP/JSONL bridge for Hardware-In-the-Loop
  test rigs (typically paired with ``contingency-rs``).

No retry / reconnection logic lives in this layer; session runners can
compose those behaviours around a backend.
"""

from __future__ import annotations

from contingency.hw.hil_bridge import HilBridgeApparatus
from contingency.hw.protocols import Apparatus, ApparatusInfo, ApparatusStatus
from contingency.hw.serial_backend import SerialApparatus
from contingency.hw.virtual import VirtualApparatus

__all__ = [
    "Apparatus",
    "ApparatusInfo",
    "ApparatusStatus",
    "VirtualApparatus",
    "SerialApparatus",
    "HilBridgeApparatus",
]
