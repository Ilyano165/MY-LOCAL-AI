"""Integrationstest gegen eine echte, lokal laufende Runtime.

Läuft nur mit ``pytest -m integration`` und gesetzter Umgebung:

    NOVA_IT_CONFIG=config/models.toml   # Pfad zur Modellkonfiguration
    NOVA_IT_MODEL=<registry-name>       # optional, sonst alle Modelle
"""

from __future__ import annotations

import os

import pytest

from models.health import ModelHealthChecker
from models.inference import InferenceEngine

pytestmark = pytest.mark.integration


@pytest.fixture
def engine() -> InferenceEngine:
    path = os.environ.get("NOVA_IT_CONFIG")
    if not path:
        pytest.skip("NOVA_IT_CONFIG nicht gesetzt")
    return InferenceEngine.from_toml(path)


async def test_live_health(engine: InferenceEngine) -> None:
    checker = ModelHealthChecker(engine)
    name = os.environ.get("NOVA_IT_MODEL")
    reports = [await checker.check(name)] if name else await checker.check_all()
    await engine.aclose()
    failed = [r.to_dict() for r in reports if not r.healthy]
    assert not failed, failed


async def test_live_benchmark(
    engine: InferenceEngine, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Echter Benchmark-Lauf (kurz): erzeugt ein Profil mit echten Messwerten."""
    from evaluation.benchmark_results import ProfileStore
    from evaluation.benchmark_runner import ModelBenchmarkRunner
    from evaluation.model_benchmark import BenchmarkOptions

    store = ProfileStore(tmp_path_factory.mktemp("profiles"))
    options = BenchmarkOptions(tasks=["chat", "short_analysis"], throughput_runs=1)
    name = os.environ.get("NOVA_IT_MODEL")
    outcomes = await ModelBenchmarkRunner(engine, store, options=options).run(
        [name] if name else None
    )
    await engine.aclose()
    assert all(o.ok for o in outcomes), [o.error for o in outcomes]
    for o in outcomes:
        assert o.profile is not None
        assert o.profile.performance["tokens_per_second"].is_fact
