"""Router mit gemessenen Benchmark-Daten: Vorrang vor Konfiguration, ehrliche Kennzeichnung."""

from __future__ import annotations

from models.capabilities import CapabilityLevel, ModelMetadata, Speed
from models.measured import DataStatus, MeasuredOverrides
from models.model_registry import ModelRegistry
from router import RoutingRequest, RuleBasedRouter, StaticAvailability, format_decision
from tests.router.conftest import model


class DictSource:
    def __init__(self, data: dict[str, MeasuredOverrides]) -> None:
        self.data = data

    def overrides_for(self, m: ModelMetadata) -> MeasuredOverrides | None:
        return self.data.get(m.name)


def router(models: list[ModelMetadata], source: DictSource | None = None) -> RuleBasedRouter:
    registry = ModelRegistry(models, measurements=source)
    return RuleBasedRouter(registry, StaticAvailability([m.name for m in models]))


TASK = "Refactor the parser module and fix the failing unit tests in the project"


async def test_without_profiles_reason_says_unmeasured() -> None:
    decision = await router([model("a", coding_capability="strong")]).route(RoutingRequest(TASK))
    assert "konfiguriert – UNMEASURED" in decision.reason
    assert "gemessen)" not in decision.reason
    assert any("UNMEASURED" in n for n in decision.notes)
    assert decision.data_status == {"a": "UNMEASURED"}
    assert "data = UNMEASURED" in format_decision(decision)
    assert decision.to_dict()["data_status"] == {"a": "UNMEASURED"}


async def test_measured_capability_overrides_configuration() -> None:
    """Konfiguriert „strong“, gemessen nur „basic“ → das andere Modell gewinnt."""
    fleet = [
        model("claims-strong", coding_capability="strong"),
        model("proven-good", coding_capability="good"),
    ]
    source = DictSource(
        {
            "claims-strong": MeasuredOverrides(
                DataStatus.MEASURED, source="benchmark t", coding_capability=CapabilityLevel.BASIC
            ),
            "proven-good": MeasuredOverrides(
                DataStatus.MEASURED, source="benchmark t", coding_capability=CapabilityLevel.STRONG
            ),
        }
    )
    assert (await router(fleet).route(RoutingRequest(TASK))).model.name == "claims-strong"
    decision = await router(fleet, source).route(RoutingRequest(TASK))
    assert decision.model.name == "proven-good"
    assert "strong, gemessen" in decision.reason
    assert any("MEASURED (benchmark t" in n for n in decision.notes)
    assert "data = MEASURED" in format_decision(decision)


async def test_measured_speed_changes_ranking_for_fast_tasks() -> None:
    fleet = [model("a", speed="fast"), model("b", speed="medium")]
    source = DictSource(
        {
            "a": MeasuredOverrides(
                DataStatus.MEASURED, source="t", speed=Speed.SLOW, tokens_per_second=6.0
            ),
            "b": MeasuredOverrides(
                DataStatus.MEASURED, source="t", speed=Speed.FAST, tokens_per_second=70.0
            ),
        }
    )
    decision = await router(fleet, source).route(RoutingRequest("hi"))
    assert decision.model.name == "b"


async def test_partial_measurement_is_labelled() -> None:
    source = DictSource({"a": MeasuredOverrides(DataStatus.PARTIAL, source="t", speed=Speed.FAST)})
    decision = await router([model("a")], source).route(RoutingRequest(TASK))
    assert "konfiguriert – UNMEASURED" in decision.reason  # Coding-Fähigkeit nicht gemessen
    assert decision.data_status == {"a": "PARTIAL"}


async def test_measured_context_limit_is_a_hard_filter() -> None:
    """Runtime meldet nur 4096 Tokens Kontext, obwohl 32768 konfiguriert sind."""
    source = DictSource(
        {"a": MeasuredOverrides(DataStatus.MEASURED, source="t", context_length=4096)}
    )
    r = router([model("a"), model("b")], source)
    decision = await r.route(RoutingRequest("Summarize this", conversation_tokens=10_000))
    assert decision.model.name == "b"
    assert any(
        rej.model == "a" and "Kontext zu klein" in rej.reasons[0] for rej in decision.rejected
    )
