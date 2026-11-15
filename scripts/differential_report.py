"""Count every Python/Rust mismatch over N random cases without failing.

Uses the same strategies and comparison as ``tests/test_differential_rs.py``
and writes one JSON document:

* ``summary``: cases, steps, seconds, cases with reinforcement, cases with
  an exception, mismatches (known and unknown), cases per outermost type;
* ``groups``: mismatches grouped by (types in the spec, first differing
  item), each with its count and one representative;
* ``mismatches``: every mismatch in fixture shape —
  ``{"schedule", "steps": [{"now", "event", "reset"}], "python", "rust",
  "first_difference", "known"}``.

Usage::

    .venv/bin/python scripts/differential_report.py [N] [--out PATH] [--seed S]
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hypothesis import HealthCheck, given, seed, settings  # noqa: E402

from tests.strategies_spec import (  # noqa: E402
    SPEC_TYPES_COVERED,
    Spec,
    Step,
    cases_of_type,
    spec_types_in,
)
from tests.test_differential_rs import (  # noqa: E402
    first_difference,
    has_nan,
    known_for,
    run_python,
    run_rust,
)


def _fixture(
    spec: Spec, steps: list[Step], py: list[Any], rs: list[Any], diff: Any, known: Any
) -> dict[str, Any]:
    return {
        "schedule": spec,
        "steps": [
            {"now": now, "now_hex": now.hex(), "event": op, "reset": reset}
            for now, op, reset in steps
        ],
        "python": py,
        "rust": rs,
        "first_difference": {"step": diff.index, "item": diff.item},
        "known": None
        if known is None
        else [{"key": k.key, "fix": k.fix, "reason": k.reason} for k in known],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("n", nargs="?", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    stats: collections.Counter[str] = collections.Counter()
    by_type: collections.Counter[str] = collections.Counter()
    inner_types: collections.Counter[str] = collections.Counter()
    mismatches: list[dict[str, Any]] = []

    def check(case: tuple[Spec, list[Step]]) -> None:
        spec, steps = case
        py, rs = run_python(spec, steps), run_rust(spec, steps)
        stats["cases"] += 1
        stats["steps"] += len(steps)
        by_type[spec["type"]] += 1
        for t in spec_types_in(spec):
            inner_types[t] += 1
        if any(o[0] == "ok" and o[1] for o in py):
            stats["cases_with_reinforcement"] += 1
        if any(o[0] != "ok" for o in py + rs):
            stats["cases_with_exception"] += 1
        if py and py[0][0] == "build_error" and rs and rs[0][0] == "build_error":
            stats["both_rejected_build"] += 1
        if has_nan(py + rs):
            stats["nan"] += 1
        diff = first_difference(py, rs)
        if diff is None:
            return
        known = known_for(spec, steps, py, rs)
        stats["mismatch_known" if known else "mismatch_unknown"] += 1
        mismatches.append(_fixture(spec, steps, py, rs, diff, known))

    types = sorted(SPEC_TYPES_COVERED)

    def run_type(name: str, n: int) -> None:
        @seed(args.seed)
        @settings(
            max_examples=n, deadline=None, database=None, suppress_health_check=list(HealthCheck)
        )
        @given(cases_of_type(name))
        def run(case: tuple[Spec, list[Step]]) -> None:
            check(case)

        run()

    t0 = time.perf_counter()
    for i, name in enumerate(types):
        run_type(name, args.n // len(types) + (1 if i < args.n % len(types) else 0))
    seconds = time.perf_counter() - t0

    groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    for m in mismatches:
        item = m["first_difference"]["item"]
        known_keys = "+".join(k["key"] for k in m["known"]) if m["known"] else "UNKNOWN"
        key = (",".join(spec_types_in(m["schedule"])), item, known_keys)
        g = groups.setdefault(
            key,
            {"types": key[0], "item": item, "known": known_keys, "count": 0, "example": m},
        )
        g["count"] += 1
    ordered = sorted(groups.values(), key=lambda g: (-g["count"], g["types"], g["item"]))

    doc: dict[str, Any] = {
        "summary": {
            "n_requested": args.n,
            "seed": args.seed,
            "seconds": round(seconds, 2),
            **stats,
            "outermost_type_counts": dict(sorted(by_type.items())),
            "contained_type_counts": dict(sorted(inner_types.items())),
        },
        "groups": ordered,
        "mismatches": mismatches,
    }
    text = json.dumps(doc, indent=1, ensure_ascii=False)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    s = doc["summary"]
    print(
        f"cases={s.get('cases')} steps={s.get('steps')} seconds={s['seconds']} "
        f"mismatch_unknown={s.get('mismatch_unknown', 0)} "
        f"mismatch_known={s.get('mismatch_known', 0)} "
        f"groups={len(ordered)}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
