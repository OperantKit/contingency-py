"""Reinforcement schedule implementations.

Schedules are added per family (ratio, interval, time, differential,
compound, progressive) and re-exported from this package.
"""

from __future__ import annotations

from contingency.schedules.adjusting import AdjustingSchedule
from contingency.schedules.alternative import Alternative
from contingency.schedules.concurrent import Concurrent
from contingency.schedules.conjunctive import Conjunctive
from contingency.schedules.differential import DRH, DRL, DRO
from contingency.schedules.interlocking import InterlockingSchedule
from contingency.schedules.interpolate import Interpolate
from contingency.schedules.interval import FI, RI, VI, LimitedHold
from contingency.schedules.mixed import Mixed
from contingency.schedules.overlay import Overlay
from contingency.schedules.percentile import Percentile
from contingency.schedules.progressive import (
    ProgressiveRatio,
    arithmetic,
    geometric,
    richardson_roberts,
)
from contingency.schedules.ratio import CRF, FR, RR, VR
from contingency.schedules.response_cost import ResponseCost
from contingency.schedules.second_order import SecondOrder
from contingency.schedules.sequence import Chained, Multiple, Tandem
from contingency.schedules.time_based import EXT, FT, RT, VT
from contingency.schedules.timeout import Timeout

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
    "Conjunctive",
    "Mixed",
    "Percentile",
    "ProgressiveRatio",
    "arithmetic",
    "geometric",
    "richardson_roberts",
    "Timeout",
    "ResponseCost",
    "AdjustingSchedule",
    "InterlockingSchedule",
    "SecondOrder",
    "Overlay",
    "Interpolate",
]
