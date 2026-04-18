"""contingency — Python reinforcement schedule engine.

Bridges contingency-dsl AST nodes into executable Schedule runtime objects.
"""

from __future__ import annotations

from contingency.bridge import from_dsl
from contingency.builder import ScheduleBuilder
from contingency.entities import Observation, Outcome, Reinforcer, ResponseEvent
from contingency.errors import (
    ContingencyError,
    ScheduleConfigError,
    ScheduleStateError,
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
    "ScheduleConfigError",
    "ScheduleStateError",
    "from_dsl",
    "__version__",
]

try:
    from contingency._version import __version__
except ImportError:
    __version__ = "0.0.0+unknown"
