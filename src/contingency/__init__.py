"""contingency — Python package for the contingency-rs schedule engine.

The schedule classes check their arguments in Python and run on
contingency-rs through the ``contingency_core`` extension module. The
bridge turns contingency-dsl AST nodes into those classes.
"""

from __future__ import annotations

from contingency.bridge import from_dsl, to_spec
from contingency.builder import ScheduleBuilder
from contingency.entities import Observation, Outcome, Reinforcer, ResponseEvent
from contingency.errors import (
    ContingencyError,
    HardwareError,
    NotConnectedError,
    ScheduleConfigError,
    ScheduleStateError,
    SpecUnrepresentableError,
)
from contingency.hw import (
    Apparatus,
    ApparatusInfo,
    ApparatusStatus,
    HilBridgeApparatus,
    VirtualApparatus,
)
from contingency.interfaces import Schedule

__all__ = [
    "Observation",
    "Outcome",
    "Reinforcer",
    "ResponseEvent",
    "Schedule",
    "ScheduleBuilder",
    "ContingencyError",
    "HardwareError",
    "NotConnectedError",
    "ScheduleConfigError",
    "ScheduleStateError",
    "SpecUnrepresentableError",
    "Apparatus",
    "ApparatusInfo",
    "ApparatusStatus",
    "HilBridgeApparatus",
    "VirtualApparatus",
    "from_dsl",
    "to_spec",
    "__version__",
]

try:
    from contingency._version import __version__
except ImportError:
    __version__ = "0.0.0+unknown"
