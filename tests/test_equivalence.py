"""Same-experiment fixtures: the operant program must yield the expected stimulus sequence.

The fixtures live in the contingency-dsl repository under
``conformance/equivalence/``. Each one holds one arrangement written in the
respondent and the operant vocabularies, input response sequences, and the
expected stimulus sequence. Two programs are the same experiment when the same
response sequence yields the same stimulus sequence. This test runs only the
``operant`` program on this package's schedules; the respondent runner checks
the ``respondent`` program against the same expected sequence.

How the operant program becomes a stimulus sequence (the runner below):

* The program text is parsed with contingency-dsl-py (``@sd`` needs the
  stimulus annotation extension) and built with :func:`contingency.bridge.from_dsl`.
* A reinforcer is the presentation of a stimulus seen from the operant side.
  Its name and duration are the one entry of the fixture's ``presentation_s``
  (the operant program does not name it).
* ``@sd("<name>", component=k)`` names the stimulus present while link ``k``
  (1-based) of a ``Chain`` is active. Links without ``@sd`` present nothing.
  The bridge labels the links ``s0``, ``s1``, ...; the runner maps those labels
  to the ``@sd`` names.
* The schedule clock stops while the reinforcer is presented and restarts at
  its offset; responses during the presentation are not given to the schedule.
* When a chain link changes, the schedule is stepped again at the same time so
  that the new link starts timing at the change.
* At the same time, offsets come before onsets.

Set ``CONTINGENCY_DSL_CONFORMANCE`` to the ``conformance`` directory of a
contingency-dsl checkout; by default the sibling checkout is used.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from contingency.bridge import from_dsl
from contingency.entities import ResponseEvent

contingency_dsl = pytest.importorskip("contingency_dsl")

from contingency_dsl.annotations.stimulus.extension import (  # noqa: E402
    StimulusExtension,
)
from contingency_dsl.extensions import ExtensionRegistry  # noqa: E402

_DEFAULT = Path(__file__).resolve().parents[2] / "contingency-dsl" / "conformance"
CONFORMANCE = Path(os.environ.get("CONTINGENCY_DSL_CONFORMANCE", _DEFAULT))
EQUIVALENCE = CONFORMANCE / "equivalence"

TICK_S = 0.5
MAX_TICKS = 100_000

Event = tuple[float, str, str]


def _cases() -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    if not EQUIVALENCE.is_dir():
        return []
    out = []
    for path in sorted(EQUIVALENCE.glob("*.json")):
        for case in json.loads(path.read_text(encoding="utf-8")):
            for inp in case["inputs"]:
                out.append((path.name, case, inp))
    return out


CASES = _cases()


def _parse(text: str) -> Any:
    registry = ExtensionRegistry.build(StimulusExtension())
    return contingency_dsl.parse(text, extensions=registry)


def _link_names(program: Any) -> dict[str, str]:
    """Map the bridge's link labels (``s0``, ``s1``, ...) to ``@sd`` names."""
    schedule = program.schedule
    names: dict[str, str] = {}
    for ann in getattr(schedule, "annotations", ()):
        if ann.keyword != "sd":
            continue
        component = (ann.params or {}).get("component")
        if component is None or ann.positional is None:
            continue
        names[f"s{int(component) - 1}"] = str(ann.positional)
    return names


def operant_timeline(
    text: str,
    *,
    presentation_s: dict[str, float],
    end_offsets: dict[str, int],
    responses: list[tuple[float, str]],
) -> list[Event]:
    """Run the operant program and return its stimulus sequence."""
    if len(presentation_s) != 1:
        raise ValueError(f"expected one presented stimulus, got {sorted(presentation_s)}")
    ((reinforcer, duration),) = presentation_s.items()
    program = _parse(text)
    names = _link_names(program)
    schedule = from_dsl(program)

    events: list[Event] = []
    offsets = dict.fromkeys(end_offsets, 0)
    pending = sorted(responses)
    paused = 0.0
    presenting_until: float | None = None
    link_stimulus: str | None = None

    def emit(now: float, stimulus: str, kind: str) -> None:
        events.append((now, stimulus, kind))
        if kind == "off" and stimulus in offsets:
            offsets[stimulus] += 1

    def set_link(label: object, now: float) -> None:
        nonlocal link_stimulus
        new = names.get(str(label)) if label is not None else None
        if new == link_stimulus:
            return
        if link_stimulus is not None:
            emit(now, link_stimulus, "off")
        if new is not None:
            emit(now, new, "on")
        link_stimulus = new

    def step(now: float, event: ResponseEvent | None) -> bool:
        """Step the schedule at session time ``now``; return True on a reinforcer."""
        outcome = schedule.step(now - paused, event)
        if outcome.reinforced:
            set_link(None, now)
            return True
        set_link(outcome.meta.get("current_component"), now)
        if outcome.meta.get("chain_transition"):
            return step(now, None)
        return False

    def ended() -> bool:
        return all(offsets[s] >= n for s, n in end_offsets.items())

    # Session time advances from one moment to the next: the next tick, the
    # next response, or the end of the presentation, whichever comes first.
    # Time-only expiries are seen at the ticks, so their resolution is TICK_S.
    now = 0.0
    tick = 0
    for _ in range(MAX_TICKS):
        if presenting_until is not None and now == presenting_until:
            emit(now, reinforcer, "off")
            paused += duration
            presenting_until = None
        if ended():
            break
        if presenting_until is None:
            reinforced = False
            while pending and pending[0][0] == now and not reinforced:
                pending.pop(0)
                reinforced = step(now, ResponseEvent(time=now - paused))
            if not reinforced:
                reinforced = step(now, None)
            if reinforced:
                emit(now, reinforcer, "on")
                presenting_until = now + duration
        # Responses during the presentation are not given to the schedule.
        while pending and pending[0][0] <= now:
            pending.pop(0)
        while tick * TICK_S <= now:
            tick += 1
        candidates = [tick * TICK_S]
        if pending:
            candidates.append(pending[0][0])
        if presenting_until is not None:
            candidates.append(presenting_until)
        now = min(candidates)
    else:
        raise AssertionError("session did not reach its end within MAX_TICKS")

    order = {"off": 0, "on": 1}
    return sorted(events, key=lambda e: (e[0], order[e[2]], e[1]))


def test_fixtures_found() -> None:
    if not EQUIVALENCE.is_dir():
        pytest.skip(f"contingency-dsl conformance not found: {EQUIVALENCE}")
    names = {case["equivalence"] for _, case, _ in CASES}
    assert {"autoshaping", "superstition-ft"} <= names


@pytest.mark.parametrize(
    ("filename", "case", "inp"),
    CASES,
    ids=[f"{fn}::{inp['name']}" for fn, _, inp in CASES],
)
def test_operant_program_matches_expected(
    filename: str, case: dict[str, Any], inp: dict[str, Any]
) -> None:
    del filename
    expected = [(float(t), str(s), str(k)) for t, s, k in case["expected"]]
    got = operant_timeline(
        case["programs"]["operant"]["input"],
        presentation_s={k: float(v) for k, v in case["presentation_s"].items()},
        end_offsets={k: int(v) for k, v in case["end"]["stimulus_offsets"].items()},
        responses=[(float(t), str(r)) for t, r in inp["responses"]],
    )
    assert got == expected


# --- The runner itself, on arrangements the fixtures do not cover ----------


def test_runner_response_at_presentation_offset_reaches_schedule() -> None:
    got = operant_timeline(
        "FR 1",
        presentation_s={"food": 3.0},
        end_offsets={"food": 2},
        responses=[(1.0, "key_peck"), (4.0, "key_peck")],
    )
    assert got == [
        (1.0, "food", "on"),
        (4.0, "food", "off"),
        (4.0, "food", "on"),
        (7.0, "food", "off"),
    ]


def test_runner_off_grid_response_keeps_exact_times() -> None:
    got = operant_timeline(
        "FR 1",
        presentation_s={"food": 3.0},
        end_offsets={"food": 2},
        responses=[(1.3, "key_peck"), (2.0, "key_peck"), (4.4, "key_peck")],
    )
    assert got == [
        (1.3, "food", "on"),
        (4.3, "food", "off"),
        (4.4, "food", "on"),
        (7.4, "food", "off"),
    ]


def test_runner_end_counts_link_stimulus_offsets() -> None:
    got = operant_timeline(
        'Chain(FT 2-s, FT 2-s) @sd("light", component=2)',
        presentation_s={"food": 3.0},
        end_offsets={"light": 1},
        responses=[],
    )
    assert got == [(2.0, "light", "on"), (4.0, "light", "off"), (4.0, "food", "on")]
