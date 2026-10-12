"""Lernzyklus Ende-zu-Ende über die API – mit **echtem** PyTorch-Training auf der CPU.

Voraussetzung (sonst übersprungen): eine Trainingsumgebung mit torch/transformers/peft
(``NOVA_TEST_TRAIN_PYTHON``) und llama.cpp-Konverterquellen (``NOVA_TEST_LLAMA_SRC``).
Als Basismodell dient ein winziges, lokal erzeugtes Zufallsmodell (kein Download).
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest

from tests.api.conftest import Factory
from tests.runtime.test_runtime import FAKE_SERVER

TRAIN_PY = os.environ.get("NOVA_TEST_TRAIN_PYTHON", "")
LLAMA_SRC = os.environ.get("NOVA_TEST_LLAMA_SRC", "")
pytestmark = pytest.mark.skipif(
    not (TRAIN_PY and Path(TRAIN_PY).is_file() and LLAMA_SRC and Path(LLAMA_SRC).is_dir()),
    reason="no training environment (NOVA_TEST_TRAIN_PYTHON / NOVA_TEST_LLAMA_SRC)",
)
HERE = Path(__file__).resolve().parent


def seed_knowledge(service: Any, n: int = 120) -> None:
    findings = [
        {
            "id": f"F{i}",
            "subquestion": f"Q{i}",
            "statement": f"The capital of country {i} is city {i}.",
            "classification": "supported",
            "source_ids": ["S1"],
            "reasons": [],
        }
        for i in range(n)
    ]
    questions = {f"Q{i}": f"What is the capital of country {i}?" for i in range(n)}
    sources = {
        "S1": {
            "id": "S1",
            "url": "https://atlas.example.org/x",
            "title": "Atlas",
            "fetched_at": "2026-10-12T00:00:00+00:00",
            "domain": "example.org",
            "score": 0.9,
            "via": "seed",
        }
    }
    service.research.knowledge.add_findings("r1", "capitals", findings, questions, sources)


def install_fake_runtime(service: Any, tmp: Path) -> None:
    server = tmp / "llama-server"
    server.write_text(FAKE_SERVER)
    server.chmod(0o755)
    model = tmp / "base.gguf"
    model.write_bytes(b"GGUF" + b"\0" * 64)
    state = {
        "runtime_tag": "b9999",
        "variant": "cpu",
        "server": str(server),
        "model": {
            "id": "general-4b",
            "label": "test base",
            "repo": "x/y",
            "registry_name": "base-4b",
            "file": "base.gguf",
            "revision": "a" * 40,
            "sha256": "0" * 64,
            "size": 68,
            "license": "apache-2.0",
            "license_url": "https://example.org",
            "path": str(model),
            "settings": {
                "registry_name": "base-4b",
                "context_length": 16384,
                "parameter_count": "4B",
                "memory_requirement": 4,
            },
        },
    }
    service.runtime.state_path.write_text(json.dumps(state))


def wait_job(client: Any, job_id: str, timeout: float = 300) -> dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job: dict[str, Any] = client.get(f"/training/jobs/{job_id}").json()
        evaluation = job.get("evaluation") or {}
        training = job.get("training") or {}
        finished = training.get("status") == "failed" or evaluation.get("status") in (
            "completed",
            "failed",
        )
        if finished and not job["running"]:
            return job
        time.sleep(1)
    raise AssertionError("training job did not finish")


def test_learning_cycle_end_to_end(
    make_client: Factory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NOVA_TRAIN_PYTHON", TRAIN_PY)
    client, service = make_client()
    with client:
        # 1) Wissen → versionierter Datensatz
        seed_knowledge(service)
        r = client.post("/training/datasets", json={"name": "capitals"})
        assert r.status_code == 200, r.text
        manifest = r.json()
        assert manifest["version"] == "0.1.0" and manifest["counts"]["test"] >= 10
        # 2) Trainingsumgebung (vorhandene Python-Umgebung) + Hardware-Probe
        service.training.install_scripts()
        probe = client.post("/training/env/probe").json()
        assert probe["torch"] and probe["cuda_available"] is False
        status = client.get("/training/status").json()
        assert status["env"]["ready"] is True and status["datasets"][0]["name"] == "capitals"
        # 3) Training eines winzigen lokalen Modells (echtes PyTorch auf der CPU)
        tiny = tmp_path / "tiny"
        subprocess.run(
            [
                TRAIN_PY,
                str(HERE / "tiny_model.py"),
                str(tiny),
                str(service.training.datasets.root / "capitals" / "0.1.0"),
            ],
            check=True,
            capture_output=True,
        )
        r = client.post(
            "/training/jobs",
            json={
                "dataset": "capitals@0.1.0",
                "choice": "general-4b",
                "epochs": 2,
                "local_model_dir": str(tiny),
            },
        )
        assert r.status_code == 202, r.text
        job = r.json()
        done = wait_job(client, job["id"])
        assert done["training"]["status"] == "completed", done["training"]
        assert done["training"]["step"] > 0 and done["training"]["loss"] is not None
        assert done["evaluation"]["status"] == "completed", done["evaluation"]
        report = done["report"]
        assert report["test_examples"] >= 10
        assert set(report["base"]) >= {"f1", "cited_rate", "regression_passed"}
        # 4) Gate-Entscheidung passt zu den gemessenen Werten (keine Selbstbewertung)
        gain = report["tuned"]["f1"] - report["base"]["f1"]
        drop = report["base"]["regression_passed"] - report["tuned"]["regression_passed"]
        assert report["gate"]["eligible"] == (gain >= 0.02 and drop <= 1)
        install_fake_runtime(service, tmp_path)
        job_dir = service.training.jobs_dir / done["id"]
        rep = json.loads((job_dir / "eval" / "report.json").read_text())
        rep["gate"] = {"eligible": False, "reasons": ["forced for test"], "settings": {}}
        (job_dir / "eval" / "report.json").write_text(json.dumps(rep))
        denied = client.post(f"/training/jobs/{done['id']}/approve")
        assert denied.status_code == 409 and denied.json()["error"]["code"] == "not_eligible"
        # 5) Freigabe-Mechanik mit bestandenem Gate: Export → Aktivierung → Chat
        rep["gate"] = {"eligible": True, "reasons": ["forced for test"], "settings": {}}
        (job_dir / "eval" / "report.json").write_text(json.dumps(rep))
        # llama.cpp-Quellen dort, wo NOVA sie für den Runtime-Tag erwartet (statt Download)
        src_dir = service.training.tools / "llama.cpp-src-b9999"
        src_dir.mkdir(parents=True)
        (src_dir / "llama.cpp-b9999").symlink_to(Path(LLAMA_SRC), target_is_directory=True)
        r = client.post(f"/training/jobs/{done['id']}/approve")
        assert r.status_code == 200, r.text
        activated = r.json()
        assert activated["approval"]["adapter_gguf_sha256"]
        assert (job_dir / "adapter.gguf").read_bytes()[:4] == b"GGUF"
        assert "--lora" in service.runtime.process.args
        toml = service.layout.models_config.read_text()
        assert 'name = "nova-general-4b-' in toml and "NOVA checkpoint active" in toml
        names = [m["name"] for m in client.get("/models").json()["models"]]
        assert names and names[0].startswith("nova-general-4b-")
        events = client.post("/chat/stream", json={"message": "What is the capital of country 3?"})
        assert '"event": "done"' in events.text
        # 6) Zurück zum Basismodell
        assert client.post("/training/deactivate").status_code == 200
        assert "--lora" not in service.runtime.process.args
    service.runtime.stop()
