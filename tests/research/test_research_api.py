"""Research über den Core (API + Manager): Konfiguration, Start, Bericht, Wissen, Neustart."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import pytest

from research.fetch import Fetcher
from research.manager import ResearchManager
from research.search import SearxngProvider, provider_from_config
from tests.api.conftest import Factory
from tests.research.fakes import FakeWeb, default_model


def wire(service: Any, web: FakeWeb, *, model: bool = True) -> None:
    research = service.research

    async def model_factory() -> Any:
        return default_model() if model else None

    def provider_factory(cfg: dict[str, Any]) -> Any:
        real = provider_from_config(cfg)  # echte Validierung, dann Fake-Netz verwenden
        if real is not None and real.name == "searxng":
            return SearxngProvider("http://search.local", client=web.client())
        return real

    research.model_factory = model_factory
    research.provider_factory = provider_factory
    research.fetcher_factory = lambda cfg: Fetcher(client=web.client(), min_interval_s=0)


def wait_done(client: Any, run_id: str, timeout: float = 15) -> dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        data: dict[str, Any] = client.get(f"/research/runs/{run_id}").json()
        if data["status"] not in ("created", "queued", "running", "paused"):
            return data
        time.sleep(0.05)
    raise AssertionError("research run did not finish")


def test_config_validation_and_no_provider(make_client: Factory) -> None:
    client, service = make_client()
    wire(service, FakeWeb())
    with client:
        status = client.get("/research/status").json()
        assert status["search_provider"] is None and status["respect_robots"] is True
        r = client.post("/research/runs", json={"objective": "NX-1 battery capacity"})
        assert r.status_code == 400 and r.json()["error"]["code"] == "no_search_provider"
        bad = client.put(
            "/research/config", json={"provider": "brave", "api_key_env": "NOVA_UNSET_X"}
        )
        assert bad.status_code == 400 and "API key" in bad.json()["error"]["message"]
        bad = client.put("/research/config", json={"provider": "searxng", "url": "ftp://x"})
        assert bad.status_code == 400
        key = client.put("/research/config", json={"provider": "brave", "api_key": "secret"})
        assert key.status_code == 422  # unbekanntes Feld → Schlüssel landet nie in der Datei
        ok = client.put(
            "/research/config", json={"provider": "searxng", "url": "http://search.local"}
        )
        assert ok.status_code == 200
        assert "secret" not in Path(service.layout.research_config).read_text()


def test_research_run_via_api(make_client: Factory) -> None:
    client, service = make_client()
    wire(service, FakeWeb())
    with client:
        client.put("/research/config", json={"provider": "searxng", "url": "http://search.local"})
        r = client.post(
            "/research/runs",
            json={
                "objective": "What battery does the NX-1 have?",
                "duration_minutes": 5,
                "max_sources": 10,
            },
        )
        assert r.status_code == 202, r.text
        run_id = r.json()["id"]
        done = wait_done(client, run_id)
        assert done["status"] == "completed" and done["model"] == "fake-model"
        assert done["counts"]["sources"] >= 3 and done["counts"]["findings"] >= 3
        assert [run["id"] for run in client.get("/research/runs").json()["runs"]] == [run_id]
        report = client.get(f"/research/runs/{run_id}/report").json()
        assert report["markdown"].startswith("# Research report")
        assert report["report"]["counts"]["claims"] == done["counts"]["claims"]
        found = client.get("/knowledge", params={"q": "charging minutes"}).json()["findings"]
        assert found and found[0]["classification"] == "supported"
        resume = client.post(f"/research/runs/{run_id}/resume")
        assert resume.status_code == 409
        assert client.get("/research/runs/../../etc").status_code == 404
        assert client.get("/research/runs/r1-nope").status_code == 404


def test_interrupted_run_is_paused_and_resumable(tmp_path: Path) -> None:
    web = FakeWeb()

    async def model_factory() -> Any:
        return default_model()

    def make() -> ResearchManager:
        return ResearchManager(
            tmp_path / "research",
            knowledge_path=tmp_path / "knowledge.db",
            config_path=tmp_path / "research.json",
            model_factory=model_factory,
            provider_factory=lambda cfg: SearxngProvider(
                "http://search.local", client=web.client()
            ),
            fetcher_factory=lambda cfg: Fetcher(client=web.client(), min_interval_s=0),
        )

    async def scenario() -> None:
        from research.engine import Budget

        manager = make()
        started = await manager.start(
            "What battery does the NX-1 have?", Budget(max_duration_s=300), []
        )
        run_id = started["id"]
        await manager.shutdown()  # Core wird beendet, während der Lauf läuft
        state = json.loads((tmp_path / "research" / run_id / "state.json").read_text())
        assert state["status"] in ("paused", "completed")
        # simulierter Absturz: Status „running“ im Checkpoint
        state["status"] = "running"
        (tmp_path / "research" / run_id / "state.json").write_text(json.dumps(state))
        restarted = make()
        assert restarted.get(run_id)["status"] == "paused"
        await restarted.resume(run_id)
        final = await restarted.wait(run_id, timeout_s=15)
        assert final["status"] == "completed"
        await restarted.shutdown()

    asyncio.run(scenario())


def test_run_without_model_reports_it(make_client: Factory) -> None:
    client, service = make_client()
    wire(service, FakeWeb(), model=False)
    with client:
        r = client.post(
            "/research/runs",
            json={"objective": "NX-1 battery", "seed_urls": ["https://docs.example.org/battery"]},
        )
        assert r.status_code == 202
        done = wait_done(client, r.json()["id"])
        assert done["status"] == "completed" and done["counts"]["claims"] == 0
        report = client.get(f"/research/runs/{done['id']}/report").json()["markdown"]
        assert "No language model was used" in report


@pytest.mark.parametrize("objective", ["", "abc"])
def test_start_validation(make_client: Factory, objective: str) -> None:
    client, _service = make_client()
    with client:
        assert client.post("/research/runs", json={"objective": objective}).status_code == 422
