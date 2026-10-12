"""TrainingManager ohne Trainingsumgebung: Status, Sperren und Freigabe-Regeln (läuft in CI)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from api.training_service import trainer_source
from tests.api.conftest import Factory
from tests.training.test_training_e2e import seed_knowledge


def fake_job(service: Any, job_id: str, *, eligible: bool, choice: str = "general-4b") -> Path:
    out = service.training.jobs_dir / job_id
    (out / "eval").mkdir(parents=True)
    (out / "job.json").write_text(json.dumps({"job_id": job_id, "base_choice": choice}))
    report = {
        "base": {"f1": 0.1, "cited_rate": 0.0, "regression_passed": 9},
        "tuned": {"f1": 0.3, "cited_rate": 0.0, "regression_passed": 9},
        "gate": {"eligible": eligible, "reasons": ["test"]},
        "test_examples": 12,
    }
    (out / "eval" / "report.json").write_text(json.dumps(report))
    return out


def test_trainer_scripts_are_found() -> None:
    src = trainer_source()
    for name in ("runner.py", "evaluate.py", "export.py", "probe.py"):
        assert (src / name).is_file()


def test_status_without_environment_blocks_training(make_client: Factory, tmp_path: Path) -> None:
    client, service = make_client()
    with client:
        status = client.get("/training/status").json()
        assert status["env"]["ready"] is False and status["jobs"] == []
        assert {c["id"] for c in status["choices"]} >= {"general-4b", "general-9b"}
        seed_knowledge(service, 40)
        assert client.post("/training/datasets", json={"name": "knowledge"}).status_code == 200
        r = client.post(
            "/training/jobs", json={"dataset": "knowledge@0.1.0", "choice": "general-4b"}
        )
        assert r.status_code == 409 and "training environment" in r.text


def test_approval_requires_gate_and_matching_base(make_client: Factory) -> None:
    client, service = make_client()
    with client:
        fake_job(service, "20261012-000000-aaaa", eligible=False)
        r = client.post("/training/jobs/20261012-000000-aaaa/approve")
        assert r.status_code == 409 and "gate" in r.text
        fake_job(service, "20261012-000000-bbbb", eligible=True)
        r = client.post("/training/jobs/20261012-000000-bbbb/approve")
        assert r.status_code == 409 and "Install the matching model" in r.text
        jobs = client.get("/training/status").json()["jobs"]
        assert [j["id"] for j in jobs] == ["20261012-000000-bbbb", "20261012-000000-aaaa"]
        assert jobs[0]["report"]["gate"]["eligible"] is True and jobs[0]["approval"] is None
