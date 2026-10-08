"""Modellauswahl: harte Filter, Rangfolge, Begründung – und nie ein nicht verfügbares Modell."""

from __future__ import annotations

import pytest

from models.model_registry import ModelRegistry
from router import (
    Complexity,
    Latency,
    MemoryRoutingLog,
    NoModelAvailableError,
    ResourceBudget,
    RoutingCategory,
    RoutingRequest,
    RuleBasedRouter,
    StaticAvailability,
)
from tests.router.conftest import fleet, model

ALL = [m.name for m in fleet()]
BIG = ResourceBudget(vram_gb=48, ram_gb=64)


def router(
    registry: ModelRegistry,
    available: list[str] | None = None,
    *,
    resources: ResourceBudget = BIG,
    log: MemoryRoutingLog | None = None,
) -> RuleBasedRouter:
    return RuleBasedRouter(
        registry,
        StaticAvailability(ALL if available is None else available),
        resources=resources,
        log=log,
    )


@pytest.mark.parametrize(
    ("task", "kwargs", "expected", "category"),
    [
        ("Debug this Python project", {}, "coder", RoutingCategory.CODING),
        ("Hallo!", {}, "small-fast", RoutingCategory.FAST),
        (
            "Beweise, dass es unendlich viele Primzahlen gibt, und analysiere den Beweis",
            {},
            "thinker",
            RoutingCategory.REASONING,
        ),
        ("Was zeigt dieses Bild?", {"needs_vision": True}, "seer", RoutingCategory.VISION),
        (
            "Fasse das Protokoll zusammen",
            {"conversation_tokens": 100_000},
            "longctx",
            RoutingCategory.LONG_CONTEXT,
        ),
        (
            "Schreibe eine Funktion, die zwei Zahlen addiert",
            {},
            "small-fast",
            RoutingCategory.CODING,
        ),  # LOW: „fähig genug“ + schnell schlägt „maximal stark“
    ],
)
async def test_routes_task_types(
    registry: ModelRegistry,
    task: str,
    kwargs: dict[str, object],
    expected: str,
    category: RoutingCategory,
) -> None:
    decision = await router(registry).route(RoutingRequest(task, **kwargs))  # type: ignore[arg-type]
    assert (decision.model.name, decision.category) == (expected, category), decision.reason
    assert decision.reason and decision.model.name not in [f.name for f in decision.fallbacks]


async def test_never_selects_unavailable_model(registry: ModelRegistry) -> None:
    available = ["small-fast", "thinker"]  # der Coder (beste Wahl) ist nicht geladen
    decision = await router(registry, available).route(RoutingRequest("Debug this Python project"))
    assert decision.model.name in available
    assert all(f.name in available for f in decision.fallbacks)
    rejected = {r.model: r.reasons for r in decision.rejected}
    assert any("nicht verfügbar" in r for r in rejected["coder"])


async def test_no_available_model_raises_with_reasons(registry: ModelRegistry) -> None:
    log = MemoryRoutingLog()
    with pytest.raises(NoModelAvailableError) as info:
        await router(registry, [], log=log).route(RoutingRequest("Hallo"))
    assert set(info.value.rejected) == set(ALL)
    assert log.entries[-1]["type"] == "failure"


async def test_vision_without_vision_model_is_an_error(registry: ModelRegistry) -> None:
    with pytest.raises(NoModelAvailableError, match="keine Vision-Fähigkeit"):
        await router(registry, ["coder", "thinker"]).route(
            RoutingRequest("Beschreibe das Bild", needs_vision=True)
        )


async def test_memory_limits_filter_and_place_models(registry: ModelRegistry) -> None:
    small_gpu = ResourceBudget(vram_gb=8, ram_gb=24)  # 24 GB RAM × 0.8 = 19.2 GB nutzbar
    decision = await router(registry, resources=small_gpu).route(
        RoutingRequest("Debug this Python project")
    )
    rejected = {r.model: " ".join(r.reasons) for r in decision.rejected}
    assert "passt nicht in den Speicher" in rejected["longctx"]  # 20 GB > 19.2 GB
    assert decision.model.name == "coder"  # 18 GB passt nur in den RAM …
    assert any("CPU/RAM" in n for n in decision.notes)  # … und wird als langsamer markiert


async def test_cpu_placement_costs_speed(registry: ModelRegistry) -> None:
    gpu_small = ResourceBudget(vram_gb=8, ram_gb=64)
    reg = ModelRegistry(
        [
            model("gpu-model", memory_requirement=6, coding_capability="good"),
            model("ram-model", memory_requirement=20, coding_capability="good"),
        ]
    )
    decision = await RuleBasedRouter(
        reg, StaticAvailability(["gpu-model", "ram-model"]), resources=gpu_small
    ).route(RoutingRequest("Fix the bug in utils.py", complexity=Complexity.MEDIUM))
    assert decision.model.name == "gpu-model"


async def test_unknown_resources_do_not_filter_but_are_noted(registry: ModelRegistry) -> None:
    decision = await router(registry, resources=ResourceBudget()).route(RoutingRequest("Hallo"))
    assert any("Ressourcen unbekannt" in n for n in decision.notes)
    assert not any("Speicher" in " ".join(r.reasons) for r in decision.rejected)


async def test_context_and_tool_requirements(registry: ModelRegistry) -> None:
    decision = await router(registry).route(
        RoutingRequest("Lies notes.md und fasse zusammen", required_tools=("read_file",))
    )
    assert decision.model.tool_calling
    assert "seer" in {r.model for r in decision.rejected}  # kein Tool-Calling
    huge = await router(registry).route(
        RoutingRequest("Zusammenfassen", conversation_tokens=200_000)
    )
    assert huge.model.name == "longctx"
    assert {"small-fast", "thinker", "seer", "coder"} <= {r.model for r in huge.rejected}


async def test_vision_with_tools_needs_both(registry: ModelRegistry) -> None:
    with pytest.raises(NoModelAvailableError, match="kein Tool-Calling"):
        await router(registry).route(
            RoutingRequest(
                "Beschreibe das Bild und speichere es",
                needs_vision=True,
                required_tools=("write_file",),
            )
        )


async def test_latency_requirements(registry: ModelRegistry) -> None:
    task = "Vergleiche zwei Sortieralgorithmen"
    realtime = await router(registry).route(RoutingRequest(task, latency=Latency.REALTIME))
    batch = await router(registry).route(RoutingRequest(task, latency=Latency.BATCH))
    assert realtime.model.name == "small-fast"
    assert batch.model.name == "thinker"


async def test_complexity_shifts_choice_within_category(registry: ModelRegistry) -> None:
    low = await router(registry).route(
        RoutingRequest("Fix bug", complexity=Complexity.LOW, category_hint=RoutingCategory.CODING)
    )
    high = await router(registry).route(
        RoutingRequest("Fix bug", complexity=Complexity.HIGH, category_hint=RoutingCategory.CODING)
    )
    assert low.model.name == "small-fast" and high.model.name == "coder"
    assert "schnellstes Modell mit ausreichender Coding-Fähigkeit" in low.reason
    assert "stärkste Coding-Fähigkeit" in high.reason


async def test_models_without_category_capability_are_excluded(registry: ModelRegistry) -> None:
    decision = await router(registry).route(RoutingRequest("Debug this Python project"))
    rejected = {r.model: r.reasons for r in decision.rejected}
    assert "keine Coding-Fähigkeit" in rejected["seer"]


async def test_deterministic_tie_break() -> None:
    reg = ModelRegistry([model("zeta"), model("alpha"), model("mid")])
    names = {
        (
            await RuleBasedRouter(reg, StaticAvailability(["zeta", "alpha", "mid"])).route(
                RoutingRequest("Schreibe einen Text über Bäume")
            )
        ).model.name
        for _ in range(5)
    }
    assert names == {"alpha"}


async def test_decision_is_logged_with_reason(registry: ModelRegistry) -> None:
    log = MemoryRoutingLog()
    decision = await router(registry, log=log).route(RoutingRequest("Debug this Python project"))
    entry = log.decisions()[-1]
    assert entry["id"] == decision.id and entry["selected_model"] == "coder"
    assert entry["category"] == "coding" and entry["complexity"] == "HIGH"
    assert "mehrstufiges Vorgehen" in entry["reason"]


async def test_max_fallbacks(registry: ModelRegistry) -> None:
    r = RuleBasedRouter(registry, StaticAvailability(ALL), resources=BIG, max_fallbacks=1)
    decision = await r.route(RoutingRequest("Hallo"))
    assert len(decision.fallbacks) == 1


async def test_report_failure_marks_model_unavailable(registry: ModelRegistry) -> None:
    r = router(registry)
    first = await r.route(RoutingRequest("Debug this Python project"))
    r.report_failure(first.model.name, "Timeout")
    second = await r.route(RoutingRequest("Debug this Python project"))
    assert second.model.name != first.model.name
    assert any("Timeout" in reason for rej in second.rejected for reason in rej.reasons)
