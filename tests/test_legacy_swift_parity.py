"""Differential test against the original Swift OperantKit.

``contingency-py`` is a port of the Swift library OperantKit
(https://github.com/YutoMizutani/OperantKit). This module builds the
Swift original from a local checkout, drives it through the conformance
fixtures with ``contingency-conformance run -- <driver>``, and compares
the Fleshler-Hoffman generators of both implementations.

Set ``OPERANTKIT_LEGACY_DIR`` to the root of a Swift OperantKit checkout
(the directory holding ``Package.swift``). Without it, or without a
``swift`` toolchain, the module is skipped. The checkout is never
modified: it is copied to a temporary directory, patched there with
``tools/legacy-swift/operantkit-swift63.patch`` (Swift 5+ spelling of
six generic-member references; no behavioural change), and built with
the executable target ``conformance-driver`` from
``tools/legacy-swift/driver/main.swift``. ``swift build`` resolves
RxSwift 4.x from GitHub on the first build.

The Swift original has no seedable RNG (``Int.random``), so the variable
schedules are compared by injecting the value list: the test reproduces
the sequence ``contingency-py`` draws for the fixture's seed and passes
it to ``VI(_:values:)`` / ``VR(_:values:)`` / ``VT(_:values:)``.

Fixtures compared
-----------------

========================== =========================================
Fixture                    Result
========================== =========================================
atomic/fr_basic            PASS
atomic/fi_basic            PASS
atomic/ft_basic            PASS
atomic/crf_basic           PASS
atomic/ext_basic           PASS
compound/alternative_fr_ft known difference, step 6 (see below)
atomic/vi_seeded_7         known difference, step 32 (see below)
atomic/vr_seeded_42        known difference, step 4 (see below)
atomic/vt_seeded_3         known difference, step 17 (see below)
========================== =========================================

Fixtures not compared
---------------------

============================================ =================================
Fixture                                      Reason
============================================ =================================
atomic/limited_hold_fi                       Swift has no LimitedHold
differential/dro_*, drl_basic, drh_basic     Swift has no DRO / DRL / DRH
progressive/pr_arithmetic                    Swift has no progressive ratio
compound/chained_*, tandem_*, multiple_*     Swift has no Chained / Tandem /
                                             Multiple
compound/concurrent_cod                      Swift's Conc has no changeover
                                             delay
atomic/rr_seeded_99, ri_seeded_5, rt_seeded_11  Swift draws from Int.random,
                                             which cannot be seeded
============================================ =================================

Known differences
-----------------

* Variable schedules (VI / VR / VT): the Swift original never advances
  through its value list. ``Parameter.nextOrder()`` is not ``mutating``
  and increments a local copy of ``order``, so every requirement after
  the first is ``values[0]`` again. The control build applies
  ``tools/legacy-swift/nextorder-control.patch`` (that one function made
  ``mutating``) and then all three fixtures pass under
  ``--stochastic strict``.
* Alternative(FR 5, FT 10 s): after the FR component reinforces at
  t = 5 s, Swift restarts the FT clock at the reinforcement (5 s) and
  reinforces again at t = 15 s; ``contingency-py`` resets FT to anchor at
  the next step (6 s), so t = 15 s is not yet due.

Fleshler-Hoffman generators
---------------------------

* ``generatedInterval`` sums to ``v * n`` and equals the rounded
  ``_fh_progression`` except for one element that carries the remnant
  (Swift adds it to ``rd[0]`` after the random placement).
* ``generatedRatio`` equals ``generate_ratios`` as a multiset.
* ``hantula1991`` keeps a 0 where ``generate_intervals_hantula1991``
  substitutes 1; with that substitution the two agree as multisets.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from contingency.conformance import fixtures_dir
from contingency.conformance.spec import build_schedule_from_spec
from contingency.helpers.fleshler_hoffman import (
    _fh_progression,
    generate_intervals_hantula1991,
    generate_ratios,
)

ENV_VAR = "OPERANTKIT_LEGACY_DIR"
TOOLS = Path(__file__).resolve().parents[1] / "tools" / "legacy-swift"
COMPAT_PATCH = TOOLS / "operantkit-swift63.patch"
CONTROL_PATCH = TOOLS / "nextorder-control.patch"
DRIVER = TOOLS / "driver" / "main.swift"
FH_PROBE = TOOLS / "fh_probe.swift"
BUILD_TIMEOUT = 1200

DETERMINISTIC = [
    "atomic/fr_basic.json",
    "atomic/fi_basic.json",
    "atomic/ft_basic.json",
    "atomic/crf_basic.json",
    "atomic/ext_basic.json",
]
VARIABLE = [
    "atomic/vi_seeded_7.json",
    "atomic/vr_seeded_42.json",
    "atomic/vt_seeded_3.json",
]
#: fixture -> (step, field) of the first mismatch against the Swift original.
KNOWN_DIFFERENCES = {
    "compound/alternative_fr_ft.json": (6, "reinforced"),
    "atomic/vi_seeded_7.json": (32, "reinforced"),
    "atomic/vr_seeded_42.json": (4, "reinforced"),
    "atomic/vt_seeded_3.json": (17, "reinforced"),
}
#: Cycles of the Fleshler-Hoffman sequence injected into VI / VR / VT.
INJECTED_CYCLES = 6


def _legacy_dir() -> Path | None:
    value = os.environ.get(ENV_VAR)
    return Path(value) if value else None


_LEGACY = _legacy_dir()

pytestmark = [
    pytest.mark.skipif(
        _LEGACY is None,
        reason=f"{ENV_VAR} is not set (path to a Swift OperantKit checkout)",
    ),
    pytest.mark.skipif(
        shutil.which("swift") is None or shutil.which("swiftc") is None,
        reason="no Swift toolchain (swift / swiftc) on PATH",
    ),
]


def _run(cmd: list[str], cwd: Path, timeout: float = BUILD_TIMEOUT) -> str:
    proc = subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"{' '.join(cmd)} failed ({proc.returncode}):\n"
            f"{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}"
        )
    return proc.stdout


def _build_driver(root: Path, patches: list[Path]) -> Path:
    assert _LEGACY is not None
    if not (_LEGACY / "Package.swift").is_file():
        pytest.fail(f"{ENV_VAR}={_LEGACY} has no Package.swift")
    package = root / "OperantKit"
    shutil.copytree(_LEGACY, package, ignore=shutil.ignore_patterns(".git", ".build", "Carthage"))
    for patch in patches:
        _run(["patch", "-p1", "--forward", "-i", str(patch)], cwd=package)
    (package / "Driver").mkdir()
    shutil.copy(DRIVER, package / "Driver" / "main.swift")
    manifest = package / "Package.swift"
    text = manifest.read_text(encoding="utf-8")
    target = (
        '.target(name: "conformance-driver", '
        'dependencies: ["OperantKit", "RxSwift", "RxBlocking"], path: "Driver"),\n'
        "        .testTarget("
    )
    patched, count = re.subn(r"\.testTarget\(", target, text, count=1)
    assert count == 1, "Package.swift has no .testTarget( to anchor the driver"
    manifest.write_text(patched, encoding="utf-8")
    _run(["swift", "build", "--product", "conformance-driver"], cwd=package)
    binary = (
        Path(_run(["swift", "build", "--show-bin-path"], cwd=package).strip())
        / "conformance-driver"
    )
    assert binary.is_file(), binary
    return binary


@pytest.fixture(scope="module")
def swift_driver(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _build_driver(tmp_path_factory.mktemp("swift-original"), [COMPAT_PATCH])


@pytest.fixture(scope="module")
def swift_control_driver(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _build_driver(tmp_path_factory.mktemp("swift-control"), [COMPAT_PATCH, CONTROL_PATCH])


def _variable_sequence(spec: dict[str, Any]) -> list[float]:
    """Requirements ``contingency-py`` draws for the spec, in order."""
    schedule: Any = build_schedule_from_spec(spec)
    values = list(schedule._sequence)
    for _ in range(INJECTED_CYCLES - 1):
        if spec["type"] == "VI":
            values += schedule._generate_sequence()
        else:
            schedule._reload_sequence()
            values += list(schedule._sequence)
    return [float(v) for v in values]


@pytest.fixture(scope="module")
def injected_corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Copies of the VI / VR / VT fixtures with the value list in params."""
    out = tmp_path_factory.mktemp("injected-corpus")
    for rel in VARIABLE:
        doc = json.loads((fixtures_dir() / rel).read_text(encoding="utf-8"))
        spec = doc["schedule"]
        values = _variable_sequence(spec)
        reinforcers = sum(1 for s in doc["steps"] if s["expect"]["reinforced"])
        # The Swift list wraps around; it must outlast the fixture.
        assert reinforcers < len(values), (rel, reinforcers, len(values))
        if spec["type"] == "VR":
            spec["params"]["values"] = [int(v) for v in values]
        else:
            spec["params"]["values_ms"] = [round(v * 1000) for v in values]
        path = out / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc), encoding="utf-8")
    return out


def _conformance_run(
    driver: Path, only: list[str], directory: Path | None = None
) -> dict[str, dict[str, Any]]:
    cmd = [sys.executable, "-m", "contingency.conformance.cli", "run", "--json"]
    cmd += ["--stochastic", "strict"]
    if directory is not None:
        cmd += ["--dir", str(directory)]
    for rel in only:
        cmd += ["--only", rel]
    cmd += ["--", str(driver)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=False)
    assert proc.returncode in (0, 1), proc.stderr
    report = json.loads(proc.stdout)
    results = {r["fixture"]: r for r in report["results"]}
    assert sorted(results) == sorted(only), results
    return results


def test_deterministic_fixtures_match_swift(swift_driver: Path) -> None:
    results = _conformance_run(swift_driver, DETERMINISTIC)
    statuses = {rel: r["status"] for rel, r in results.items()}
    assert statuses == dict.fromkeys(DETERMINISTIC, "PASS"), results


def test_known_differences_from_swift(swift_driver: Path, injected_corpus: Path) -> None:
    results = _conformance_run(swift_driver, ["compound/alternative_fr_ft.json"])
    results |= _conformance_run(swift_driver, VARIABLE, injected_corpus)
    observed = {
        rel: (r["status"], r["first_mismatch"]["step"], r["first_mismatch"]["field"])
        for rel, r in results.items()
        if r.get("first_mismatch")
    }
    expected = {rel: ("FAIL", *where) for rel, where in KNOWN_DIFFERENCES.items()}
    assert observed == expected, results


def test_variable_schedules_match_when_swift_advances_its_list(
    swift_control_driver: Path, injected_corpus: Path
) -> None:
    results = _conformance_run(swift_control_driver, VARIABLE, injected_corpus)
    statuses = {rel: r["status"] for rel, r in results.items()}
    assert statuses == dict.fromkeys(VARIABLE, "PASS"), results


# --- Fleshler-Hoffman generators ---------------------------------------------

PROBE_RUNS = 20


@pytest.fixture(scope="module")
def fh_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    assert _LEGACY is not None
    source = _LEGACY / "Sources" / "Common" / "Helpers" / "FleshlerHoffman.swift"
    if not source.is_file():
        pytest.fail(f"{source} not found under {ENV_VAR}")
    work = tmp_path_factory.mktemp("fh-probe")
    shutil.copy(FH_PROBE, work / "main.swift")
    binary = work / "fh_probe"
    _run(["swiftc", str(source), "main.swift", "-o", str(binary)], cwd=work)
    return binary


def _probe(binary: Path, generator: str, v: int, n: int) -> list[list[int]]:
    out = _run([str(binary), generator, str(v), str(n), str(PROBE_RUNS)], cwd=binary.parent)
    lists: list[list[int]] = json.loads(out)
    assert len(lists) == PROBE_RUNS
    return lists


def _swift_round(x: float) -> int:
    """Swift ``round`` (half away from zero)."""
    return int(math.copysign(math.floor(abs(x) + 0.5), x))


@pytest.mark.parametrize(("v", "n"), [(10, 12), (60, 20), (30000, 12)])
def test_fh_interval_matches_rounded_progression(fh_probe: Path, v: int, n: int) -> None:
    rounded = [_swift_round(x) for x in _fh_progression(v, n)]
    remnant = v * n - sum(rounded)
    for swift in _probe(fh_probe, "interval", v, n):
        assert sum(swift) == v * n
        extra = Counter(swift) - Counter(rounded)
        missing = Counter(rounded) - Counter(swift)
        if remnant == 0:
            assert not extra and not missing, swift
        else:
            assert sum(extra.values()) == 1 and sum(missing.values()) == 1, swift
            (added,) = extra
            (removed,) = missing
            assert added - removed == remnant, swift


def test_fh_ratio_matches_generate_ratios(fh_probe: Path) -> None:
    python = sorted(generate_ratios(5, 12, seed=0))
    for swift in _probe(fh_probe, "ratio", 5, 12):
        assert sorted(swift) == python


def test_fh_hantula_differs_only_by_zero_to_one(fh_probe: Path) -> None:
    python = sorted(generate_intervals_hantula1991(10, 12, seed=0))
    for swift in _probe(fh_probe, "hantula", 10, 12):
        assert 0 in swift and 0 not in python
        assert sorted(1 if x == 0 else x for x in swift) == python
