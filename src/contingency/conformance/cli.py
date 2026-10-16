"""``contingency-conformance`` — check any implementation against the corpus.

Subcommands::

    contingency-conformance list
    contingency-conformance path
    contingency-conformance check OBSERVED.json [OBSERVED.json ...]
    contingency-conformance run [--only GLOB] -- CMD [ARGS ...]
    contingency-conformance replay-python < fixture.json

Exit codes: 0 = no failures and no errors; 1 = at least one FAIL or
ERROR; 2 = input error (corpus not found, unknown fixture, unreadable
observed file, no fixture selected, bad usage).
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from contingency.conformance import (
    CorpusError,
    Fixture,
    FixtureDirNotFoundError,
    Result,
    StochasticMode,
    compare,
    fixtures_dir,
    iter_fixtures,
)

__all__ = ["main"]

EXIT_OK = 0
EXIT_MISMATCH = 1
EXIT_INPUT = 2


class InputError(Exception):
    """Raised for problems that make the run meaningless (exit 2)."""


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dir", help="fixture directory (overrides lookup)")
    parser.add_argument("--json", action="store_true", help="machine-readable output")


def _checking(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--stochastic",
        choices=["strict", "count", "skip"],
        default="count",
        help=(
            "how to judge seeded fixtures: strict = bit-equal (Python-compatible "
            "PRNG), count = reinforcer count within 0.5-1.5x and first reinforcer "
            "not before the first step (default), skip = do not check"
        ),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="contingency-conformance",
        description="Check a reinforcement-schedule implementation against the "
        "contingency-py conformance corpus.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="list fixtures")
    _common(p_list)

    p_path = sub.add_parser("path", help="print the fixture directory in use")
    _common(p_path)

    p_check = sub.add_parser("check", help="compare observed JSON files with fixtures")
    _common(p_check)
    _checking(p_check)
    p_check.add_argument("observed", nargs="+", help="observed JSON file(s)")

    p_run = sub.add_parser(
        "run",
        help="pipe each fixture to CMD on stdin and compare its stdout",
    )
    _common(p_run)
    _checking(p_run)
    p_run.add_argument("--only", action="append", default=[], metavar="GLOB",
                       help="only fixtures whose relative path matches GLOB (repeatable)")
    p_run.add_argument("--timeout", type=float, default=60.0, help="seconds per fixture")
    p_run.add_argument("cmd", nargs=argparse.REMAINDER, help="-- CMD [ARGS ...]")

    sub.add_parser("replay-python", help="adapter: fixture on stdin -> observed on stdout")
    return parser


def _load_corpus(directory: str | None) -> dict[str, Fixture]:
    try:
        corpus = {fx.rel: fx for fx in iter_fixtures(directory)}
    except (FixtureDirNotFoundError, CorpusError) as exc:
        raise InputError(str(exc)) from exc
    if not corpus:
        raise InputError("fixture directory contains no fixtures")
    return corpus


def _report(results: list[Result], as_json: bool) -> int:
    counts = {s: sum(1 for r in results if r.status == s)
              for s in ("PASS", "FAIL", "ERROR", "SKIP")}
    summary = {
        "passed": counts["PASS"],
        "failed": counts["FAIL"],
        "errors": counts["ERROR"],
        "skipped": counts["SKIP"],
        "total": len(results),
    }
    if as_json:
        print(json.dumps({"summary": summary, "results": [r.to_dict() for r in results]},
                         indent=2, sort_keys=True))
    else:
        for r in results:
            print(r.describe())
        print(f"{summary['passed']} passed, {summary['failed']} failed, "
              f"{summary['errors']} errors, {summary['skipped']} skipped")
    return EXIT_OK if counts["FAIL"] == 0 and counts["ERROR"] == 0 else EXIT_MISMATCH


def _cmd_list(args: argparse.Namespace) -> int:
    corpus = _load_corpus(args.dir)
    rows = [{"path": fx.rel, "kind": fx.kind, "type": fx.type, "steps": len(fx.steps)}
            for fx in corpus.values()]
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        for row in rows:
            print(f"{row['path']}\t{row['kind']}\t{row['type']}\t{row['steps']}")
    return EXIT_OK


def _cmd_path(args: argparse.Namespace) -> int:
    try:
        print(fixtures_dir(args.dir))
    except FixtureDirNotFoundError as exc:
        raise InputError(str(exc)) from exc
    return EXIT_OK


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:  # ValueError covers JSON and UTF-8 errors
        raise InputError(f"cannot read observed file {path}: {exc}") from exc


def _cmd_check(args: argparse.Namespace) -> int:
    corpus = _load_corpus(args.dir)
    results: list[Result] = []
    for name in args.observed:
        observed = _read_json(Path(name))
        if not isinstance(observed, dict) or not isinstance(observed.get("fixture"), str):
            raise InputError(f"{name}: observed document needs a \"fixture\" string")
        rel = observed["fixture"]
        if rel not in corpus:
            raise InputError(f"{name}: unknown fixture {rel!r}")
        results.append(compare(corpus[rel], observed, args.stochastic))
    return _report(results, args.json)


def _run_one(fx: Fixture, cmd: list[str], mode: StochasticMode, timeout: float) -> Result:
    if fx.stochastic and mode == "skip":
        return compare(fx, {}, mode)
    try:
        proc = subprocess.run(
            cmd,
            input=fx.path.read_text(encoding="utf-8"),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        return Result(fx.rel, "ERROR", len(fx.steps), detail=f"adapter failed: {exc}")
    if proc.returncode != 0:
        tail = proc.stderr.strip().splitlines()[-1:] or [""]
        return Result(fx.rel, "ERROR", len(fx.steps),
                      detail=f"adapter exit {proc.returncode}: {tail[0]}")
    try:
        observed = json.loads(proc.stdout)
    except ValueError as exc:
        return Result(fx.rel, "ERROR", len(fx.steps), detail=f"adapter stdout is not JSON: {exc}")
    if not isinstance(observed, dict):
        return Result(fx.rel, "ERROR", len(fx.steps), detail="adapter stdout is not a JSON object")
    return compare(fx, observed, mode)


def _cmd_run(args: argparse.Namespace) -> int:
    cmd = list(args.cmd)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        raise InputError("run needs a command after --")
    corpus = _load_corpus(args.dir)
    selected = [fx for rel, fx in corpus.items()
                if not args.only or any(fnmatch.fnmatch(rel, g) for g in args.only)]
    if not selected:
        raise InputError(f"no fixture matches --only {args.only}")
    results = [_run_one(fx, cmd, args.stochastic, args.timeout) for fx in selected]
    return _report(results, args.json)


def _cmd_replay_python() -> int:
    from contingency.conformance.replay import replay

    try:
        observed = replay(json.loads(sys.stdin.read()))
    except (ValueError, KeyError, TypeError) as exc:
        raise InputError(f"replay-python: stdin is not a fixture: {exc!r}") from exc
    json.dump(observed, sys.stdout)
    sys.stdout.write("\n")
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "list":
            return _cmd_list(args)
        if args.command == "path":
            return _cmd_path(args)
        if args.command == "check":
            return _cmd_check(args)
        if args.command == "run":
            return _cmd_run(args)
        return _cmd_replay_python()
    except InputError as exc:
        print(f"contingency-conformance: error: {exc}", file=sys.stderr)
        return EXIT_INPUT


if __name__ == "__main__":
    sys.exit(main())
