"""Conformance corpus access and comparison.

The corpus pins the exact outcome sequence of each ``contingency-py``
schedule for a known event sequence. Any implementation — in any
language — can be checked against it: produce an *observed* document
for a fixture and pass it to :func:`compare`, or drive it through the
``contingency-conformance`` command.

Observed document shape::

    {"fixture": "atomic/fr_basic.json",          # optional for ``run``
     "implementation": "my-impl 0.1",            # optional
     "steps": [{"reinforced": false},
               {"reinforced": true,
                "reinforcer": {"time": 5.0, "magnitude": 1.0, "label": "SR+"},
                "meta": {"component": 0}}]}

Fixture lookup order (:func:`fixtures_dir`): explicit argument →
``CONTINGENCY_CONFORMANCE_DIR`` → corpus bundled in the wheel →
``conformance/`` of a development checkout.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

__all__ = [
    "ENV_VAR",
    "FLOAT_TOL",
    "CorpusError",
    "Fixture",
    "FixtureDirNotFoundError",
    "Mismatch",
    "Result",
    "StochasticMode",
    "compare",
    "fixtures_dir",
    "is_stochastic",
    "iter_fixtures",
    "load_index",
]

ENV_VAR = "CONTINGENCY_CONFORMANCE_DIR"
FLOAT_TOL = 1e-9
INDEX_NAME = "index.json"
#: Band for ``--stochastic count``: observed reinforcer count must lie in
#: ``[LOW, HIGH] × expected``. Same band as the contingency-rs harness.
COUNT_BAND = (0.5, 1.5)

StochasticMode = Literal["strict", "count", "skip"]
Status = Literal["PASS", "FAIL", "ERROR", "SKIP"]

_PACKAGE_DIR = Path(__file__).resolve().parent
_DEV_DIR = _PACKAGE_DIR.parents[2] / "conformance"
_DEV_INDEX = _PACKAGE_DIR.parents[2] / "conformance-index.json"


class FixtureDirNotFoundError(FileNotFoundError):
    """No conformance corpus could be located."""


class CorpusError(ValueError):
    """A file in the corpus directory is not a readable fixture."""


def _has_fixtures(path: Path) -> bool:
    return path.is_dir() and any(
        p.name != INDEX_NAME for p in path.rglob("*.json")
    )


def fixtures_dir(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Return the directory holding the conformance fixtures.

    Raises :class:`FixtureDirNotFoundError` if an explicit or environment
    directory does not exist, or if no corpus is found at all.
    """
    if explicit is not None:
        path = Path(explicit).expanduser().resolve()
        if not path.is_dir():
            raise FixtureDirNotFoundError(f"--dir {path} is not a directory")
        return path
    env = os.environ.get(ENV_VAR)
    if env:
        path = Path(env).expanduser().resolve()
        if not path.is_dir():
            raise FixtureDirNotFoundError(f"{ENV_VAR}={path} is not a directory")
        return path
    if _has_fixtures(_PACKAGE_DIR):
        return _PACKAGE_DIR
    if _has_fixtures(_DEV_DIR):
        return _DEV_DIR
    raise FixtureDirNotFoundError(
        f"no conformance corpus found (set {ENV_VAR} or pass --dir)"
    )


def is_stochastic(schedule: dict[str, Any]) -> bool:
    """True if any node of the schedule spec carries a ``seed`` param."""

    def walk(node: Any) -> bool:
        if isinstance(node, dict):
            params = node.get("params")
            if isinstance(params, dict) and "seed" in params:
                return True
            return any(walk(v) for v in node.values())
        if isinstance(node, list):
            return any(walk(v) for v in node)
        return False

    return walk(schedule)


@dataclass(frozen=True)
class Fixture:
    """One fixture file of the corpus."""

    rel: str
    path: Path
    data: dict[str, Any]

    @property
    def stochastic(self) -> bool:
        return is_stochastic(self.data["schedule"])

    @property
    def kind(self) -> str:
        return "stochastic" if self.stochastic else "deterministic"

    @property
    def type(self) -> str:
        return str(self.data["schedule"]["type"])

    @property
    def steps(self) -> list[dict[str, Any]]:
        steps: list[dict[str, Any]] = self.data["steps"]
        return steps


def iter_fixtures(directory: str | os.PathLike[str] | None = None) -> Iterator[Fixture]:
    """Yield every fixture under the corpus directory, sorted by path."""
    root = fixtures_dir(directory)
    for path in sorted(root.rglob("*.json")):
        if path.name == INDEX_NAME and path.parent == root:
            continue
        rel = path.relative_to(root).as_posix()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise CorpusError(f"{rel}: not readable JSON ({exc})") from exc
        if not (
            isinstance(data, dict)
            and isinstance(data.get("schedule"), dict)
            and "type" in data["schedule"]
            and isinstance(data.get("steps"), list)
        ):
            raise CorpusError(f"{rel}: not a fixture (needs \"schedule\".type and \"steps\")")
        yield Fixture(rel=rel, path=path, data=data)


def load_index(directory: str | os.PathLike[str] | None = None) -> dict[str, Any] | None:
    """Return the corpus index (``schema``, ``fixtures``) if one is present."""
    root = fixtures_dir(directory)
    for candidate in (root / INDEX_NAME, _DEV_INDEX if root == _DEV_DIR else None):
        if candidate is not None and candidate.is_file():
            index: dict[str, Any] = json.loads(candidate.read_text(encoding="utf-8"))
            return index
    return None


@dataclass(frozen=True)
class Mismatch:
    """First point where the observed trajectory departs from the fixture."""

    step: int | None
    field: str
    expected: Any
    observed: Any

    def describe(self) -> str:
        where = f"step {self.step} " if self.step is not None else ""
        return f"{where}{self.field}: expected {self.expected!r} got {self.observed!r}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "field": self.field,
            "expected": self.expected,
            "observed": self.observed,
        }


@dataclass(frozen=True)
class Result:
    """Outcome of checking one observed document against one fixture."""

    fixture: str
    status: Status
    steps: int
    first_mismatch: Mismatch | None = None
    detail: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        line = f"{self.status} {self.fixture}"
        if self.first_mismatch is not None:
            line += f" {self.first_mismatch.describe()}"
        elif self.detail:
            line += f" ({self.detail})"
        else:
            line += f" ({self.steps} steps)"
        return line

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "fixture": self.fixture,
            "status": self.status,
            "steps": self.steps,
            "first_mismatch": (
                self.first_mismatch.to_dict() if self.first_mismatch is not None else None
            ),
        }
        if self.detail:
            out["detail"] = self.detail
        out.update(self.extra)
        return out


_MISSING = "<missing>"


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _num_equal(a: Any, b: Any) -> bool:
    if not (_is_number(a) and _is_number(b)):
        return False
    try:
        return abs(float(a) - float(b)) <= FLOAT_TOL
    except OverflowError:
        return False


def _value_equal(expected: Any, observed: Any) -> bool:
    """Equality for meta values: numbers within tolerance, bools never equal numbers."""
    if _is_number(expected) or _is_number(observed):
        return _num_equal(expected, observed)
    return type(observed) is type(expected) and observed == expected


def _compare_step(i: int, expect: dict[str, Any], obs: Any) -> Mismatch | None:
    if not isinstance(obs, dict):
        return Mismatch(i, "step", "object", obs)
    exp_r = expect["reinforced"]
    got_r = obs.get("reinforced", _MISSING)
    if got_r is not exp_r:
        return Mismatch(i, "reinforced", exp_r, got_r)
    if exp_r:
        exp_sr = expect["reinforcer"]
        got_sr = obs.get("reinforcer")
        if not isinstance(got_sr, dict):
            return Mismatch(i, "reinforcer", exp_sr, got_sr if got_sr is not None else _MISSING)
        for key in ("time", "magnitude"):
            got = got_sr.get(key, _MISSING)
            if not _num_equal(exp_sr[key], got):
                return Mismatch(i, f"reinforcer.{key}", exp_sr[key], got)
        got_label = got_sr.get("label", _MISSING)
        if got_label != exp_sr["label"]:
            return Mismatch(i, "reinforcer.label", exp_sr["label"], got_label)
    exp_meta = expect.get("meta") or {}
    got_meta = obs.get("meta") or {}
    if exp_meta and not isinstance(got_meta, dict):
        return Mismatch(i, "meta", exp_meta, got_meta)
    for key, value in exp_meta.items():
        if key not in got_meta:
            return Mismatch(i, f"meta.{key}", value, _MISSING)
        got = got_meta[key]
        if not _value_equal(value, got):
            return Mismatch(i, f"meta.{key}", value, got)
    return None


def _finite(x: Any) -> bool:
    try:
        return _is_number(x) and math.isfinite(float(x))
    except OverflowError:
        return False


def _compare_count(fixture: Fixture, obs_steps: list[Any]) -> Result:
    n = len(fixture.steps)
    recorded = [s["expect"]["reinforcer"] for s in fixture.steps if s["expect"]["reinforced"]]
    allowed = {(float(r["magnitude"]), r["label"]) for r in recorded}
    t0 = float(fixture.steps[0]["now"]) if fixture.steps else 0.0
    observed_n = 0

    def fail(mismatch: Mismatch) -> Result:
        return Result(fixture.rel, "FAIL", n, mismatch)

    for i, obs in enumerate(obs_steps):
        if not isinstance(obs, dict):
            return fail(Mismatch(i, "step", "object", obs))
        reinforced = obs.get("reinforced", _MISSING)
        if not isinstance(reinforced, bool):
            return fail(Mismatch(i, "reinforced", "bool", reinforced))
        if not reinforced:
            continue
        observed_n += 1
        sr = obs.get("reinforcer")
        if not isinstance(sr, dict):
            return fail(Mismatch(i, "reinforcer", "object", sr if sr is not None else _MISSING))
        time = sr.get("time", _MISSING)
        if not _finite(time) or float(time) < t0:
            return fail(Mismatch(i, "reinforcer.time", f"finite number >= {t0}", time))
        magnitude, label = sr.get("magnitude", _MISSING), sr.get("label", _MISSING)
        if not any(_num_equal(m, magnitude) and label == lab for m, lab in allowed):
            expected = [{"magnitude": m, "label": lab} for m, lab in sorted(allowed)]
            got = {"magnitude": magnitude, "label": label}
            return fail(Mismatch(i, "reinforcer", expected, got))
    extra = {"reinforcers": {"expected": len(recorded), "observed": observed_n}}
    low, high = COUNT_BAND
    if not (low * len(recorded) <= observed_n <= high * len(recorded)):
        band = [low * len(recorded), high * len(recorded)]
        mismatch = Mismatch(None, "reinforcer count", band, observed_n)
        return Result(fixture.rel, "FAIL", n, mismatch, extra=extra)
    return Result(
        fixture.rel,
        "PASS",
        n,
        detail=f"count mode, {observed_n} reinforcers vs {len(recorded)}",
        extra=extra,
    )


def compare(
    fixture: Fixture,
    observed: dict[str, Any],
    stochastic: StochasticMode = "count",
) -> Result:
    """Check an observed document against a fixture.

    Rules (deterministic fixtures, and stochastic ones under ``strict``):
    ``reinforced`` must match exactly; when reinforced, ``reinforcer.time``
    and ``magnitude`` within :data:`FLOAT_TOL` and ``label`` equal; every
    key of the fixture's ``meta`` must be present and equal (extra
    observed keys are allowed); the number of steps must match.

    Stochastic fixtures under ``count`` only require the reinforcer count
    to lie within :data:`COUNT_BAND` of the recorded count and the first
    reinforcer not to precede the first step. ``skip`` returns ``SKIP``.
    """
    n = len(fixture.steps)
    if fixture.stochastic and stochastic == "skip":
        return Result(fixture.rel, "SKIP", n, detail="stochastic fixture skipped")
    obs_steps = observed.get("steps") if isinstance(observed, dict) else None
    if not isinstance(obs_steps, list):
        return Result(fixture.rel, "FAIL", n, Mismatch(None, "steps", "list", obs_steps))
    if len(obs_steps) != n:
        return Result(fixture.rel, "FAIL", n, Mismatch(None, "step count", n, len(obs_steps)))
    if fixture.stochastic and stochastic == "count":
        return _compare_count(fixture, obs_steps)
    for i, (step, obs) in enumerate(zip(fixture.steps, obs_steps, strict=True)):
        mismatch = _compare_step(i, step["expect"], obs)
        if mismatch is not None:
            return Result(fixture.rel, "FAIL", n, mismatch)
    return Result(fixture.rel, "PASS", n)
