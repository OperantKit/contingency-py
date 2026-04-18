"""Reinforcement schedule implementations.

Schedules are added per family (ratio, interval, time, differential,
compound, progressive) and re-exported from this package.
"""

from __future__ import annotations

from contingency.schedules.alternative import Alternative
from contingency.schedules.concurrent import Concurrent
from contingency.schedules.differential import DRH, DRL, DRO
from contingency.schedules.interval import FI, RI, VI, LimitedHold
from contingency.schedules.progressive import (
    ProgressiveRatio,
    arithmetic,
    geometric,
    richardson_roberts,
)
from contingency.schedules.ratio import CRF, FR, RR, VR
from contingency.schedules.sequence import Chained, Multiple, Tandem
from contingency.schedules.time_based import EXT, FT, RT, VT

__all__ = [
    "FR",
    "VR",
    "RR",
    "CRF",
    "FI",
    "VI",
    "RI",
    "LimitedHold",
    "FT",
    "VT",
    "RT",
    "EXT",
    "DRO",
    "DRL",
    "DRH",
    "Multiple",
    "Chained",
    "Tandem",
    "Alternative",
    "Concurrent",
    "ProgressiveRatio",
    "arithmetic",
    "geometric",
    "richardson_roberts",
]
