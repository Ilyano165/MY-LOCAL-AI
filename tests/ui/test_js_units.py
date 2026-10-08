"""Führt die JavaScript-Unit-Tests der Oberfläche (node --test) in der Python-Suite aus."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js nicht installiert")
def test_ui_javascript_units() -> None:
    files = sorted(str(p) for p in HERE.glob("*.test.mjs"))
    result = subprocess.run(
        ["node", "--test", *files], capture_output=True, text=True, timeout=120, check=False
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-2000:]
    assert "# fail 0" in result.stdout
