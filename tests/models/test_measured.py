from __future__ import annotations

from models.capabilities import CapabilityLevel, ModelMetadata, Speed, TaskType
from models.measured import DataStatus, MeasuredOverrides, MeasurementSource, apply_overrides
from models.model_registry import ModelRegistry
from tests.models.fakes import make_meta


class DictSource:
    def __init__(self, data: dict[str, MeasuredOverrides]) -> None:
        self.data = data

    def overrides_for(self, model: ModelMetadata) -> MeasuredOverrides | None:
        return self.data.get(model.name)


def measured(**values: object) -> MeasuredOverrides:
    return MeasuredOverrides(DataStatus.MEASURED, source="benchmark test", **values)  # type: ignore[arg-type]


def test_protocol() -> None:
    assert isinstance(DictSource({}), MeasurementSource)


def test_without_source_everything_is_unmeasured() -> None:
    registry = ModelRegistry([make_meta()])
    status = registry.data_status("test-model")
    assert status.status is DataStatus.UNMEASURED
    assert "UNMEASURED" in status.describe()
    effective = registry.effective("test-model")
    assert effective.extra["data_status"] == "UNMEASURED"
    assert effective.coding_capability is CapabilityLevel.GOOD  # Konfiguration bleibt


def test_model_without_profile_is_explicitly_unmeasured() -> None:
    registry = ModelRegistry([make_meta()], measurements=DictSource({}))
    assert registry.data_status("test-model").status is DataStatus.UNMEASURED
    assert "kein Benchmark-Profil" in registry.data_status("test-model").describe()


def test_measured_values_replace_configured_and_keep_history() -> None:
    o = measured(coding_capability=CapabilityLevel.BASIC, speed=Speed.FAST, tokens_per_second=55.0)
    registry = ModelRegistry([make_meta()], measurements=DictSource({"test-model": o}))
    eff = registry.effective("test-model")
    assert eff.coding_capability is CapabilityLevel.BASIC
    assert eff.speed is Speed.FAST
    assert eff.reasoning_capability is CapabilityLevel.GOOD  # nicht gemessen → Konfiguration
    assert eff.extra["configured"] == {
        "coding_capability": CapabilityLevel.GOOD,
        "speed": Speed.MEDIUM,
    }
    assert eff.extra["measured_tokens_per_second"] == 55.0
    assert registry.get("test-model").coding_capability is CapabilityLevel.GOOD  # roh unverändert
    assert "MEASURED" in o.describe() and "coding_capability" in o.describe()


def test_stale_profile_is_not_applied() -> None:
    o = MeasuredOverrides(
        DataStatus.STALE, source="benchmark 0.9", coding_capability=CapabilityLevel.NONE
    )
    eff = apply_overrides(make_meta(), o)
    assert eff.coding_capability is CapabilityLevel.GOOD
    assert "veraltet" in o.describe()


def test_find_best_for_prefers_measured_data() -> None:
    a = make_meta(name="a", coding_capability="strong")
    b = make_meta(name="b", coding_capability="good")
    source = DictSource({"a": measured(coding_capability=CapabilityLevel.BASIC)})
    registry = ModelRegistry([a, b])
    assert registry.find_best_for(TaskType.CODING).name == "a"
    registry.set_measurements(source)
    assert registry.find_best_for(TaskType.CODING).name == "b"


def test_measured_no_tool_calling_excludes_model() -> None:
    registry = ModelRegistry(
        [make_meta(name="a")], measurements=DictSource({"a": measured(tool_calling=False)})
    )
    from models.capabilities import TaskRequirements

    assert registry.rank_for(TaskRequirements(needs_tool_calling=True)) == []
