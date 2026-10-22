"""Execute every ```python block in the schedule-terms pages so their claims keep holding."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PAGES = ("docs/en/schedule-terms.md", "docs/ja/schedule-terms.md")
_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)


def _blocks(name: str) -> list[str]:
    return _BLOCK.findall((ROOT / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", PAGES)
def test_page_has_python_examples(name: str) -> None:
    assert _blocks(name)


@pytest.mark.parametrize(
    ("name", "index"),
    [(name, i) for name in PAGES for i in range(len(_blocks(name)))],
)
def test_page_example_runs(name: str, index: int) -> None:
    exec(compile(_blocks(name)[index], f"{name}#{index}", "exec"), {})
