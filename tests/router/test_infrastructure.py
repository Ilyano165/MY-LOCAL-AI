"""Verfügbarkeit, Ressourcen, Routing-Logs, Engine- und Agent-Integration."""

from __future__ import annotations

from pathlib import Path

import pytest

from models.base import ChatRequest, Message, ProviderUnavailableError
from models.capabilities import TaskRequirements, TaskType
from models.inference import InferenceEngine, ProviderRegistry
from models.model_registry import ModelRegistry
from router import (
    JsonlRoutingLog,
    MemoryRoutingLog,
    ProviderAvailability,
    ResourceBudget,
    RoutingRequest,
    RuleBasedRouter,
    StaticAvailability,
    detect_resources,
    estimate_tokens,
    format_decision,
)
from router.resources import _meminfo_available_gb
from tests.models.fakes import StubProvider
from tests.router.conftest import fleet, model

# --------------------------------------------------------------------------- Verfügbarkeit


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


class FlakyProvider(StubProvider):
    def __init__(self, *, ready: bool = True, served: list[str] | None = None) -> None:
        super().__init__("local", served=served)
        self.ready = ready
        self.health_calls = 0

    async def health(self):  # type: ignore[no-untyped-def]
        from models.base import ProviderHealth

        self.health_calls += 1
        return ProviderHealth(True, self.ready, "lädt Modell" if not self.ready else "ok")


async def test_provider_availability_checks_runtime_listing() -> None:
    provider = FlakyProvider(served=["coder"])
    availability = ProviderAvailability(ProviderRegistry([provider]))
    assert (await availability.check(model("coder"))).available
    missing = await availability.check(model("thinker"))
    assert not missing.available and "listet" in missing.reason


async def test_runtime_not_ready_or_missing_provider() -> None:
    not_ready = ProviderAvailability(ProviderRegistry([FlakyProvider(ready=False, served=["x"])]))
    result = await not_ready.check(model("x"))
    assert not result.available and "nicht bereit" in result.reason
    no_provider = ProviderAvailability(ProviderRegistry())
    assert not (await no_provider.check(model("x"))).available


async def test_availability_without_listing_checks_local_file(tmp_path: Path) -> None:
    availability = ProviderAvailability(ProviderRegistry([FlakyProvider(served=None)]))
    assert (await availability.check(model("a"))).available
    gone = await availability.check(model("b", local_path=str(tmp_path / "fehlt.gguf")))
    assert not gone.available and "Modelldatei fehlt" in gone.reason


async def test_availability_is_cached_and_failures_are_sticky() -> None:
    clock = Clock()
    provider = FlakyProvider(served=["coder"])
    availability = ProviderAvailability(ProviderRegistry([provider]), ttl_s=30, clock=clock)
    await availability.check(model("coder"))
    await availability.check(model("coder"))
    assert provider.health_calls == 1
    availability.mark_unavailable("coder", "Timeout")
    assert not (await availability.check(model("coder"))).available
    clock.t = 31  # nach Ablauf wird neu geprüft
    assert (await availability.check(model("coder"))).available
    assert provider.health_calls == 2
    availability.invalidate()
    await availability.check(model("coder"))
    assert provider.health_calls == 3


# --------------------------------------------------------------------------- Ressourcen


@pytest.mark.parametrize(
    ("budget", "memory", "placement"),
    [
        (ResourceBudget(vram_gb=24, ram_gb=64), 20, "gpu"),
        (ResourceBudget(vram_gb=8, ram_gb=64), 20, "cpu"),
        (ResourceBudget(vram_gb=8, ram_gb=16), 20, None),
        (ResourceBudget(ram_gb=32), 20, "ram"),  # Apple Silicon / nur CPU
        (ResourceBudget(ram_gb=32), 30, None),  # 30 > 32 × 0.8
    ],
)
def test_placement(budget: ResourceBudget, memory: float, placement: str | None) -> None:
    assert budget.placement(memory) == placement


def test_meminfo_parsing(tmp_path: Path) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal: 16000000 kB\nMemAvailable: 8388608 kB\n")
    assert _meminfo_available_gb(meminfo) == pytest.approx(8.0)
    assert _meminfo_available_gb(tmp_path / "fehlt") is None


async def test_detect_resources_on_this_machine() -> None:
    budget = await detect_resources()
    assert budget.ram_gb is not None and budget.ram_gb > 0 and budget.known


def test_estimate_tokens_is_conservative() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("x" * 3500) >= 1000  # eher über- als unterschätzt


# --------------------------------------------------------------------------- Logs


async def test_format_matches_routing_log_example() -> None:
    reg = ModelRegistry(fleet())
    decision = await RuleBasedRouter(
        reg, StaticAvailability([m.name for m in fleet()]), resources=ResourceBudget(vram_gb=48)
    ).route(RoutingRequest("Debug this Python project"))
    text = format_decision(decision)
    lines = text.splitlines()
    assert lines[:3] == ["TASK:", '"Debug this Python project"', "ROUTING:"]
    assert "category = CODING" in lines and "complexity = HIGH" in lines
    assert "selected_model = coder" in lines
    assert any(line.startswith('reason = "Erfordert Codeverständnis') for line in lines)


async def test_jsonl_log_joins_decisions_with_outcomes(tmp_path: Path) -> None:
    log = JsonlRoutingLog(tmp_path / "routing.jsonl")
    r = RuleBasedRouter(ModelRegistry(fleet()), StaticAvailability(["small-fast"]), log=log)
    first = await r.route(RoutingRequest("Hallo"))
    await r.route(RoutingRequest("Danke"))
    r.record_outcome(first.id, success=True, verdict="success", quality=0.9)
    joined = log.joined()
    assert len(joined) == 2
    assert joined[0]["outcome"]["quality"] == 0.9 and joined[1]["outcome"] is None


def test_memory_log_without_router_log_is_noop() -> None:
    r = RuleBasedRouter(ModelRegistry(), StaticAvailability([]))
    r.record_outcome("x", success=False)  # kein Log konfiguriert → kein Fehler
    assert MemoryRoutingLog().decisions() == []


# --------------------------------------------------------------------------- Engine


def engine_with_router(
    provider: StubProvider, available: list[str], log: MemoryRoutingLog | None = None
) -> InferenceEngine:
    reg = ModelRegistry(fleet())
    router = RuleBasedRouter(
        reg, StaticAvailability(available), resources=ResourceBudget(vram_gb=48), log=log
    )
    return InferenceEngine(reg, ProviderRegistry([provider]), router=router)


async def test_engine_uses_router_and_returns_routing_id() -> None:
    log = MemoryRoutingLog()
    provider = StubProvider()
    engine = engine_with_router(provider, ["small-fast", "coder"], log)
    result = await engine.chat(
        ChatRequest(messages=(Message.user("Debug this Python project"),)), task=TaskType.CODING
    )
    assert result.model.name == "coder" and provider.calls == ["coder"]
    assert result.routing_id == log.decisions()[-1]["id"]


async def test_engine_never_calls_unavailable_model() -> None:
    provider = StubProvider()
    engine = engine_with_router(provider, ["small-fast"])
    await engine.chat(ChatRequest(messages=(Message.user("Debug this Python project"),)))
    assert provider.calls == ["small-fast"]


async def test_engine_falls_back_and_reports_failures() -> None:
    provider = StubProvider(behaviors={"coder": ProviderUnavailableError("weg")})
    engine = engine_with_router(provider, ["coder", "thinker", "small-fast"])
    request = ChatRequest(messages=(Message.user("Debug this Python project"),))
    result = await engine.chat(request, fallback=True)
    assert result.attempts[0] == "coder" and result.model.name != "coder"
    # Der Ausfall wurde gemeldet: die nächste Anfrage geht gar nicht erst an den Coder
    provider.calls.clear()
    await engine.chat(request)
    assert "coder" not in provider.calls


async def test_engine_without_fallback_raises() -> None:
    provider = StubProvider(behaviors={"coder": ProviderUnavailableError("weg")})
    engine = engine_with_router(provider, ["coder", "thinker"])
    with pytest.raises(ProviderUnavailableError):
        await engine.chat(ChatRequest(messages=(Message.user("Debug this Python project"),)))


async def test_explicit_model_bypasses_router() -> None:
    provider = StubProvider()
    engine = engine_with_router(provider, ["small-fast"])
    result = await engine.chat(ChatRequest(messages=(Message.user("x"),)), model="thinker")
    assert result.model.name == "thinker" and result.routing_id is None


def test_routing_request_from_chat() -> None:
    from models.base import ToolSpec

    request = ChatRequest(
        messages=(Message.system("System " * 100), Message.user("Fix den Bug in api.py")),
        tools=(ToolSpec("read_file", "lesen"),),
    )
    routing = RoutingRequest.from_chat(
        request, TaskRequirements(task_type=TaskType.CODING, min_context_length=8000)
    )
    assert routing.task == "Fix den Bug in api.py"
    assert routing.required_tools == ("read_file",)
    assert routing.conversation_tokens > 100 and routing.min_context_tokens == 8000
    assert routing.category_hint is not None and routing.category_hint.value == "coding"


# --------------------------------------------------------------------------- Agent


async def test_agent_reports_routing_outcome(tmp_path: Path) -> None:
    from agents.agent import Agent
    from agents.state import InMemoryTaskStore
    from tests.agents.fakes import ScriptedProvider
    from tools.registry import ToolRegistry

    log = MemoryRoutingLog()
    reg = ModelRegistry([model("local-model")])
    provider = ScriptedProvider(execute=["17 * 23 = 391"])
    router = RuleBasedRouter(reg, StaticAvailability(["local-model"]), log=log)
    engine = InferenceEngine(reg, ProviderRegistry([provider]), router=router)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    task = await Agent(engine, ToolRegistry(), InMemoryTaskStore(), workspace).run(
        "Was ist 17 * 23?"
    )

    assert task.routing_ids and task.routing_ids[0] == log.decisions()[0]["id"]
    outcomes = [e for e in log.entries if e["type"] == "outcome"]
    assert outcomes and outcomes[0]["verdict"] == "success" and outcomes[0]["success"] is True
