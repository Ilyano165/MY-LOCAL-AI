from __future__ import annotations

from pathlib import Path

import pytest

from models.health import CheckStatus, HealthCheckConfig, ModelHealthChecker
from models.inference import InferenceEngine, ProviderRegistry
from models.local_provider import OpenAICompatibleProvider
from models.model_registry import ModelRegistry
from tests.models.fakes import FakeLlamaServer, StubProvider, make_meta

ALL = ("present", "loadable", "inference", "response_time", "context", "tool_calling")


def checker_for(
    server: FakeLlamaServer, config: HealthCheckConfig | None = None, **meta: object
) -> ModelHealthChecker:
    provider = OpenAICompatibleProvider(
        "local", "http://127.0.0.1:8080", transport=server.transport()
    )
    engine = InferenceEngine(ModelRegistry([make_meta(**meta)]), ProviderRegistry([provider]))
    return ModelHealthChecker(engine, config)


def statuses(report: object) -> dict[str, CheckStatus]:
    return {c.name: c.status for c in report.checks}  # type: ignore[attr-defined]


async def test_all_checks_pass_for_working_model(tmp_path: Path) -> None:
    weights = tmp_path / "model.gguf"
    weights.write_bytes(b"GGUF")
    report = await checker_for(FakeLlamaServer(), local_path=str(weights)).check("test-model")

    assert report.healthy, report.to_dict()
    assert statuses(report) == dict.fromkeys(ALL, CheckStatus.PASSED)
    assert "Datei vorhanden" in report.get("present").detail
    data = report.to_dict()
    assert data["healthy"] is True and len(data["checks"]) == 6


async def test_missing_file_fails_and_skips_dependents(tmp_path: Path) -> None:
    report = await checker_for(FakeLlamaServer(), local_path=str(tmp_path / "nope.gguf")).check(
        "test-model"
    )
    s = statuses(report)
    assert s["present"] is CheckStatus.FAILED
    assert all(s[name] is CheckStatus.SKIPPED for name in ALL[1:])
    assert not report.healthy


async def test_model_not_listed_by_runtime() -> None:
    report = await checker_for(FakeLlamaServer(models=["other"])).check("test-model")
    assert report.get("present").status is CheckStatus.FAILED
    assert "other" in report.get("present").detail


async def test_runtime_lists_model_by_file_path() -> None:
    server = FakeLlamaServer(models=["/srv/weights/model.gguf"])
    checker = checker_for(server, local_path=None, served_name="model.gguf")
    report = await checker.check("test-model")
    assert report.get("present").status is CheckStatus.PASSED


async def test_runtime_still_loading() -> None:
    report = await checker_for(FakeLlamaServer(health_status=503)).check("test-model")
    s = statuses(report)
    assert s["present"] is CheckStatus.PASSED
    assert s["loadable"] is CheckStatus.FAILED
    assert "Loading" in report.get("loadable").detail
    assert s["inference"] is CheckStatus.SKIPPED


async def test_wrong_answer_fails_inference() -> None:
    report = await checker_for(FakeLlamaServer(mode="dumb")).check("test-model")
    s = statuses(report)
    assert s["inference"] is CheckStatus.FAILED
    assert s["context"] is CheckStatus.SKIPPED and s["tool_calling"] is CheckStatus.SKIPPED


async def test_short_memory_fails_context_only() -> None:
    report = await checker_for(FakeLlamaServer(mode="short_memory")).check("test-model")
    s = statuses(report)
    assert s["inference"] is CheckStatus.PASSED
    assert s["context"] is CheckStatus.FAILED
    assert s["tool_calling"] is CheckStatus.PASSED


async def test_slow_response_fails_response_time() -> None:
    config = HealthCheckConfig(max_response_s=0.01, timeout_s=5)
    report = await checker_for(FakeLlamaServer(delay_s=0.05), config).check("test-model")
    assert report.get("inference").status is CheckStatus.PASSED
    assert report.get("response_time").status is CheckStatus.FAILED


async def test_hard_timeout_per_check() -> None:
    config = HealthCheckConfig(timeout_s=0.05, max_response_s=0.05)
    report = await checker_for(FakeLlamaServer(delay_s=1.0), config).check("test-model")
    assert report.get("loadable").status is CheckStatus.FAILED
    assert "Zeitlimit" in report.get("loadable").detail


async def test_tool_calling_skipped_when_not_declared() -> None:
    report = await checker_for(FakeLlamaServer(), tool_calling=False).check("test-model")
    assert report.get("tool_calling").status is CheckStatus.SKIPPED
    assert report.healthy


async def test_wrong_tool_argument_fails() -> None:
    report = await checker_for(FakeLlamaServer(wrong_tool_value=True)).check("test-model")
    assert report.get("tool_calling").status is CheckStatus.FAILED
    assert "falsches Argument" in report.get("tool_calling").detail


async def test_context_probe_respects_limits() -> None:
    server = FakeLlamaServer()
    config = HealthCheckConfig(
        context_fill_ratio=0.5, context_max_tokens=1024, checks=frozenset({"context"})
    )
    report = await checker_for(server, config, context_length=100_000).check("test-model")
    assert report.get("context").status is CheckStatus.PASSED
    assert "~1024 Tokens" in report.get("context").detail
    # Prompt-Größe ~ 1024 Tokens * ~4 Zeichen, deutlich unter dem Kontextfenster
    assert len(server.requests[-1].content) < 1024 * 8


async def test_check_subset_and_check_all() -> None:
    provider = StubProvider(served=["a", "b"])
    engine = InferenceEngine(
        ModelRegistry([make_meta(name="a"), make_meta(name="b")]), ProviderRegistry([provider])
    )
    reports = await ModelHealthChecker(
        engine, HealthCheckConfig(checks=frozenset({"present"}))
    ).check_all()
    assert [r.model for r in reports] == ["a", "b"]
    assert all(r.healthy and len(r.checks) == 1 for r in reports)


async def test_provider_without_listing_relies_on_file(tmp_path: Path) -> None:
    provider = StubProvider(served=None)
    engine = InferenceEngine(
        ModelRegistry([make_meta(local_path=str(tmp_path))]), ProviderRegistry([provider])
    )
    report = await ModelHealthChecker(
        engine, HealthCheckConfig(checks=frozenset({"present"}))
    ).check("test-model")
    assert report.get("present").status is CheckStatus.PASSED


async def test_report_without_passed_checks_is_not_healthy() -> None:
    engine = InferenceEngine(
        ModelRegistry([make_meta(tool_calling=False)]), ProviderRegistry([StubProvider()])
    )
    report = await ModelHealthChecker(
        engine, HealthCheckConfig(checks=frozenset({"tool_calling"}))
    ).check("test-model")
    assert not report.healthy  # nur "skipped" ist kein Gesundheitsnachweis
    with pytest.raises(KeyError):
        report.get("present")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"timeout_s": 0},
        {"max_response_s": -1},
        {"context_fill_ratio": 0},
        {"context_max_tokens": 10},
    ],
)
def test_config_validation(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        HealthCheckConfig(**kwargs)  # type: ignore[arg-type]
