"""Execute every ```python block in the READMEs so the documented examples keep working."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
READMES = ("README.md", "README.ja.md")
_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)


def _blocks(name: str) -> list[str]:
    return _BLOCK.findall((ROOT / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", READMES)
def test_readme_has_python_examples(name: str) -> None:
    assert _blocks(name)


@pytest.mark.parametrize(
    ("name", "index"),
    [(name, i) for name in READMES for i in range(len(_blocks(name)))],
)
def test_readme_example_runs(name: str, index: int) -> None:
    exec(compile(_blocks(name)[index], f"{name}#{index}", "exec"), {})
