"""CLI: Wissensspeicher → versionierter Datensatz → Trainingsplan (Dry-Run)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from api.cli import main
from research.knowledge import KnowledgeStore


def run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str]:
    with pytest.raises(SystemExit) as exc:
        main(list(argv))
    return int(exc.value.code or 0), capsys.readouterr().out


def seed(data: Path, n: int = 250) -> None:
    store = KnowledgeStore(data / "knowledge.db")
    findings = [
        {
            "id": f"F{i}",
            "subquestion": f"Q{i}",
            "statement": f"Fact {i} about subject {i} is documented.",
            "classification": "supported" if i % 10 else "contested",
            "source_ids": ["S1"],
            "reasons": [],
        }
        for i in range(n)
    ]
    questions = {f"Q{i}": f"What do we know about subject {i}?" for i in range(n)}
    sources = {
        "S1": {
            "id": "S1",
            "url": "https://docs.example.org/x",
            "title": "Doc",
            "fetched_at": "2026-10-10T00:00:00+00:00",
            "domain": "example.org",
            "score": 0.9,
            "via": "search:searxng",
        }
    }
    store.add_findings("r1", "subjects", findings, questions, sources)
    store.close()


def test_dataset_and_plan_via_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data = tmp_path / "data"
    data.mkdir()
    seed(data)
    manual = tmp_path / "manual.jsonl"
    manual.write_text(
        json.dumps({"question": "What is NOVA?", "answer": "NOVA is a local AI platform."}) + "\n"
    )
    code, out = run(
        capsys,
        "--data-dir",
        str(data),
        "dataset",
        "build",
        "nova-knowledge",
        "--from-knowledge",
        "--manual",
        str(manual),
    )
    manifest = json.loads(out)
    assert code == 0 and manifest["version"] == "0.1.0"
    assert sum(manifest["counts"].values()) == 226  # 225 supported + 1 manuell
    assert manifest["excluded_before_build"] == {"classification:contested": 25}
    assert manifest["provenance"] == {"research_finding": 225, "manual": 1}
    code, out = run(capsys, "--data-dir", str(data), "dataset", "verify", "nova-knowledge@0.1.0")
    assert code == 0 and json.loads(out)["ok"] is True
    code, out = run(capsys, "--data-dir", str(data), "dataset", "list")
    assert json.loads(out)[0]["name"] == "nova-knowledge"

    config = tmp_path / "train.toml"
    config.write_text(
        'name = "nova-qwen3.5-4b"\nbase_model = "Qwen/Qwen3.5-4B-Instruct"\n'
        'base_revision = "0123abcd"\nbase_license = "apache-2.0"\n'
        'dataset = "nova-knowledge@0.1.0"\n'
    )
    hardware = tmp_path / "probe.json"
    hardware.write_text(
        json.dumps(
            {
                "problems": [],
                "vram_gb": 11.0,
                "compute_dtype": "float32",
                "packages": {"bitsandbytes": "0.48"},
            }
        )
    )
    code, out = run(
        capsys,
        "--data-dir",
        str(data),
        "train",
        "plan",
        "--config",
        str(config),
        "--hardware",
        str(hardware),
    )
    plan = json.loads(out)
    assert code == 0 and plan["ok"] is True and plan["estimated_vram_gb"] < 11
    assert plan["dataset_manifest"]["version"] == "0.1.0"


def test_dataset_build_without_input(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit, match="no input"):
        main(["--data-dir", str(tmp_path), "dataset", "build", "x"])
