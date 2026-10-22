"""Tests for ``contingency.conformance`` and the ``contingency-conformance`` CLI."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from contingency.conformance import (
    ENV_VAR,
    FixtureDirNotFoundError,
    compare,
    fixtures_dir,
    iter_fixtures,
    load_index,
)
from contingency.conformance.cli import main
from contingency.conformance.replay import replay

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "conformance"
N_FIXTURES = 23
ADAPTER = [sys.executable, "-m", "contingency.conformance.cli", "replay-python"]


def _fixture(rel: str) -> Any:
    return next(fx for fx in iter_fixtures(CORPUS) if fx.rel == rel)


def _observed(rel: str) -> dict[str, Any]:
    fx = _fixture(rel)
    doc = replay(fx.data)
    doc["fixture"] = rel
    return doc


def _first_reinforced(doc: dict[str, Any]) -> int:
    return next(i for i, s in enumerate(doc["steps"]) if s["reinforced"])


def _write(tmp_path: Path, doc: dict[str, Any], name: str = "observed.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


# --- lookup -----------------------------------------------------------------


def test_iter_fixtures_finds_whole_corpus() -> None:
    fixtures = list(iter_fixtures(CORPUS))
    assert len(fixtures) == N_FIXTURES
    assert sum(fx.stochastic for fx in fixtures) == 6


def test_env_var_overrides_lookup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(ENV_VAR, str(tmp_path))
    assert fixtures_dir() == tmp_path.resolve()
    monkeypatch.setenv(ENV_VAR, str(tmp_path / "missing"))
    with pytest.raises(FixtureDirNotFoundError):
        fixtures_dir()


def test_explicit_dir_wins_over_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(ENV_VAR, str(tmp_path))
    assert fixtures_dir(CORPUS) == CORPUS.resolve()


def test_index_matches_corpus() -> None:
    import hashlib

    index = load_index(CORPUS)
    assert index is not None
    assert index["schema"] == 1
    entries = {e["path"]: e for e in index["fixtures"]}
    fixtures = {fx.rel: fx for fx in iter_fixtures(CORPUS)}
    assert set(entries) == set(fixtures)
    for rel, fx in fixtures.items():
        e = entries[rel]
        assert e["kind"] == fx.kind
        assert e["type"] == fx.type
        assert e["steps"] == len(fx.steps)
        assert e["sha256"] == hashlib.sha256(fx.path.read_bytes()).hexdigest()


# --- compare ----------------------------------------------------------------


@pytest.mark.parametrize("mode", ["strict", "count"])
def test_python_replay_passes_every_fixture(mode: str) -> None:
    for fx in iter_fixtures(CORPUS):
        result = compare(fx, replay(fx.data), mode)  # type: ignore[arg-type]
        assert result.status == "PASS", result.describe()


def test_skip_mode_skips_only_stochastic() -> None:
    statuses = {fx.rel: compare(fx, replay(fx.data), "skip").status
                for fx in iter_fixtures(CORPUS)}
    assert sum(s == "SKIP" for s in statuses.values()) == 6
    assert statuses["atomic/fr_basic.json"] == "PASS"


def test_extra_observed_meta_is_tolerated() -> None:
    doc = _observed("compound/chained_fr2_fr3.json")
    for step in doc["steps"]:
        step.setdefault("meta", {})["extra"] = "ok"
    assert compare(_fixture("compound/chained_fr2_fr3.json"), doc).status == "PASS"


def test_count_mode_rejects_out_of_band_count() -> None:
    doc = _observed("atomic/vr_seeded_42.json")
    for step in doc["steps"]:
        step.clear()
        step["reinforced"] = False
    result = compare(_fixture("atomic/vr_seeded_42.json"), doc, "count")
    assert result.status == "FAIL"
    assert result.first_mismatch is not None
    assert result.first_mismatch.field == "reinforcer count"


# --- CLI: the four broken-observation cases ---------------------------------


def _break_reinforced(doc: dict[str, Any]) -> tuple[int | None, str]:
    i = _first_reinforced(doc)
    doc["steps"][i] = {"reinforced": False}
    return i, "reinforced"


def _break_time(doc: dict[str, Any]) -> tuple[int | None, str]:
    i = _first_reinforced(doc)
    doc["steps"][i]["reinforcer"]["time"] += 1e-6
    return i, "reinforcer.time"


def _break_meta(doc: dict[str, Any]) -> tuple[int | None, str]:
    i = next(i for i, s in enumerate(doc["steps"]) if s.get("meta"))
    key = sorted(doc["steps"][i]["meta"])[0]
    del doc["steps"][i]["meta"][key]
    return i, f"meta.{key}"


def _drop_step(doc: dict[str, Any]) -> tuple[int | None, str]:
    doc["steps"].pop()
    return None, "step count"


BREAKS = {
    "reinforced-flipped": ("atomic/fr_basic.json", _break_reinforced),
    "time-plus-1e-6": ("atomic/fi_basic.json", _break_time),
    "meta-key-removed": ("compound/chained_fr2_fr3.json", _break_meta),
    "step-removed": ("atomic/fr_basic.json", _drop_step),
}


@pytest.mark.parametrize("case", sorted(BREAKS))
def test_check_reports_broken_observation(
    case: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rel, breaker = BREAKS[case]
    doc = copy.deepcopy(_observed(rel))
    step, field = breaker(doc)
    path = _write(tmp_path, doc)

    code = main(["check", "--dir", str(CORPUS), "--json", str(path)])

    assert code == 1
    report = json.loads(capsys.readouterr().out)
    assert report["summary"]["failed"] == 1
    (result,) = report["results"]
    assert result["status"] == "FAIL"
    assert result["first_mismatch"]["field"] == field
    assert result["first_mismatch"]["step"] == step


def test_check_intact_observation_passes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _write(tmp_path, _observed("atomic/fr_basic.json"))
    assert main(["check", "--dir", str(CORPUS), str(path)]) == 0
    out = capsys.readouterr().out
    assert "PASS atomic/fr_basic.json" in out
    assert "1 passed, 0 failed, 0 errors, 0 skipped" in out


@pytest.mark.parametrize(
    "doc",
    [{"steps": []}, {"fixture": "atomic/nope.json", "steps": []}],
    ids=["no-fixture-key", "unknown-fixture"],
)
def test_check_input_errors_exit_2(doc: dict[str, Any], tmp_path: Path) -> None:
    assert main(["check", "--dir", str(CORPUS), str(_write(tmp_path, doc))]) == 2


def test_missing_corpus_exits_2(tmp_path: Path) -> None:
    assert main(["list", "--dir", str(tmp_path / "missing")]) == 2


# --- CLI: run / list / path -------------------------------------------------


def test_run_with_python_adapter_passes_all_strict(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["run", "--dir", str(CORPUS), "--stochastic", "strict", "--json", "--", *ADAPTER])
    report = json.loads(capsys.readouterr().out)
    assert code == 0
    assert report["summary"] == {
        "passed": N_FIXTURES, "failed": 0, "errors": 0, "skipped": 0, "total": N_FIXTURES,
    }


def test_run_only_glob_and_adapter_error(capsys: pytest.CaptureFixture[str]) -> None:
    failing = [sys.executable, "-c", "import sys; sys.exit(3)"]
    code = main(["run", "--dir", str(CORPUS), "--only", "atomic/f*", "--", *failing])
    out = capsys.readouterr().out
    assert code == 1
    assert out.count("ERROR ") == 3  # fi_basic, fr_basic, ft_basic
    assert "0 passed, 0 failed, 3 errors, 0 skipped" in out


def test_run_without_command_exits_2() -> None:
    assert main(["run", "--dir", str(CORPUS)]) == 2


def test_list_and_path(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["list", "--dir", str(CORPUS)]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == N_FIXTURES
    assert "atomic/vi_seeded_7.json\tstochastic\tVI\t81" in lines
    assert main(["path", "--dir", str(CORPUS)]) == 0
    assert capsys.readouterr().out.strip() == str(CORPUS.resolve())


# --- malformed input (review findings) ---------------------------------------


def test_non_utf8_observed_file_exits_2(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_bytes(b"\xff\xfe")
    assert main(["check", "--dir", str(CORPUS), str(path)]) == 2


@pytest.mark.parametrize("content", ['{"x": 1}', "{not json"], ids=["not-fixture", "not-json"])
def test_malformed_corpus_exits_2(content: str, tmp_path: Path) -> None:
    (tmp_path / "broken.json").write_text(content, encoding="utf-8")
    assert main(["list", "--dir", str(tmp_path)]) == 2


def test_huge_integer_is_fail_not_crash() -> None:
    doc = _observed("atomic/fr_basic.json")
    doc["steps"][_first_reinforced(doc)]["reinforcer"]["time"] = 10**400
    result = compare(_fixture("atomic/fr_basic.json"), doc)
    assert result.status == "FAIL"


@pytest.mark.parametrize("bad", [float("nan"), -1.0, 10**400], ids=["nan", "negative", "huge"])
def test_count_mode_rejects_bad_first_time(bad: float) -> None:
    doc = _observed("atomic/vi_seeded_7.json")
    doc["steps"][_first_reinforced(doc)]["reinforcer"]["time"] = bad
    result = compare(_fixture("atomic/vi_seeded_7.json"), doc, "count")
    assert result.status == "FAIL"
    assert result.first_mismatch is not None
    assert result.first_mismatch.field == "reinforcer.time"


def test_count_mode_checks_magnitude_and_label() -> None:
    doc = _observed("atomic/vr_seeded_42.json")
    doc["steps"][_first_reinforced(doc)]["reinforcer"]["label"] = "x"
    result = compare(_fixture("atomic/vr_seeded_42.json"), doc, "count")
    assert result.status == "FAIL"
    assert result.first_mismatch is not None
    assert result.first_mismatch.field == "reinforcer"


def test_meta_int_and_float_compare_numerically_but_bool_is_not_int() -> None:
    fx = _fixture("compound/tandem_fr_fr.json")
    doc = _observed("compound/tandem_fr_fr.json")
    doc["steps"][0]["meta"]["current_component"] = 0.0
    assert compare(fx, doc).status == "PASS"
    doc["steps"][0]["meta"]["current_component"] = False
    assert compare(fx, doc).status == "FAIL"


def test_replay_python_bad_stdin_exits_2(monkeypatch: pytest.MonkeyPatch) -> None:
    import io

    monkeypatch.setattr(sys, "stdin", io.StringIO("{not json"))
    assert main(["replay-python"]) == 2
