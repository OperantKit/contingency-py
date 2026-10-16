"""Built-in adapter: replay a fixture against the Python runtime.

Used by ``contingency-conformance replay-python`` and as the reference
example of a ``run -- CMD`` adapter: read the fixture JSON on stdin,
step the schedule through every ``(now, event)`` pair, write the
observed document on stdout.
"""

from __future__ import annotations

from typing import Any

from contingency.conformance.spec import build_schedule_from_spec
from contingency.entities import Outcome, ResponseEvent

__all__ = ["observe", "replay"]

FLOAT_NDIGITS = 9


def _round(x: float) -> float:
    return round(x, FLOAT_NDIGITS)


def observe(outcome: Outcome) -> dict[str, Any]:
    """Serialise one Outcome as an observed step."""
    out: dict[str, Any] = {"reinforced": outcome.reinforced}
    if outcome.reinforcer is not None:
        out["reinforcer"] = {
            "time": _round(outcome.reinforcer.time),
            "magnitude": _round(outcome.reinforcer.magnitude),
            "label": outcome.reinforcer.label,
        }
    meta: dict[str, Any] = {}
    for key, value in outcome.meta.items():
        if isinstance(value, (bool, int, str)):
            meta[key] = value
        elif isinstance(value, float):
            meta[key] = _round(value)
    if meta:
        out["meta"] = meta
    return out


def replay(fixture: dict[str, Any]) -> dict[str, Any]:
    """Run a fixture document through the Python runtime."""
    schedule = build_schedule_from_spec(fixture["schedule"])
    steps: list[dict[str, Any]] = []
    for step in fixture["steps"]:
        ev = step["event"]
        event = (
            ResponseEvent(time=ev["time"], operandum=ev.get("operandum", "main"))
            if ev is not None
            else None
        )
        steps.append(observe(schedule.step(step["now"], event)))
    return {"implementation": "contingency-py (replay-python)", "steps": steps}
