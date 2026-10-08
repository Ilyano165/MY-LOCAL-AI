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
