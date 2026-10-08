from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from scripts import benchmark_hardware, benchmark_models

CONFIG = """
[[providers]]
name = "local"
type = "openai_compatible"
base_url = "http://127.0.0.1:9"
health_path = "/health"

[[models]]
name = "m1"
provider = "local"
parameter_count = "1B"
context_length = 4096
reasoning_capability = "basic"
coding_capability = "basic"
vision_capability = "none"
tool_calling = false
speed = "fast"
memory_requirement = 1
quantization = "Q8_0"
"""


async def test_benchmark_hardware_json(capsys: pytest.CaptureFixture[str]) -> None:
    code = await benchmark_hardware._main(argparse.Namespace(json=True, sample=0.3, pid=None))
    assert code == 0
    data = json.loads(capsys.readouterr().out)
    assert data["status"] == "DETECTED" and len(data["fingerprint"]) == 16
    assert data["sampling"]["status"] == "MEASURED" and data["sampling"]["samples"] >= 2


async def test_benchmark_models_list_marks_unmeasured(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "models.toml"
    config.write_text(CONFIG)
    args = argparse.Namespace(
        config=config, profiles=tmp_path / "p", list=True, model=None, server_cmd=None
    )
    assert await benchmark_models._main(args) == 0
    out = capsys.readouterr().out
    assert "m1" in out and "UNMEASURED" in out


async def test_benchmark_models_unreachable_runtime_fails_honestly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "models.toml"
    config.write_text(CONFIG)
    args = argparse.Namespace(
        config=config,
        profiles=tmp_path / "p",
        list=False,
        model=None,
        server_cmd=None,
        tasks="chat",
        timeout=5.0,
        no_exec=False,
        performance_only=False,
        capabilities_only=False,
        runs=1,
        tokens=16,
        measure_load=False,
        server_pid=None,
        long_context_tokens=4000,
        json=False,
    )
    assert await benchmark_models._main(args) == 1
    assert "FEHLER m1" in capsys.readouterr().out
    assert not (tmp_path / "p" / "m1.json").exists()  # kein Profil ohne Messung
