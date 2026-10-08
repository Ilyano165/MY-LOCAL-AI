from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.benchmark_results import (
    Measurement,
    MeasurementStatus,
    ModelProfile,
    ProfileStore,
    TaskResult,
    TaskStatus,
    score_to_level,
    tokens_per_second_to_speed,
)
from models.capabilities import CapabilityLevel, Speed
from models.measured import DataStatus
from tests.models.fakes import make_meta

M = MeasurementStatus


def profile(**overrides: object) -> ModelProfile:
    data: dict[str, object] = {
        "model": "test-model",
        "benchmark_version": "1.0+abc",
        "runtime": {"runtime": "llama.cpp"},
        "hardware": {"cpu_model": "X"},
        "hardware_fingerprint": "hw1",
        "capabilities": {
            "reasoning": Measurement.measured(0.9, "score", "bench"),
            "coding": Measurement.measured(0.5, "score", "bench"),
            "vision": Measurement.missing(M.NOT_SUPPORTED, "image input is not supported"),
            "tool_calling": Measurement.measured(1.0, "score", "bench"),
        },
        "performance": {
            "tokens_per_second": Measurement.measured(12.0, "tok/s", "diff"),
            "memory_footprint": Measurement.reported(5.5, "GB", "ollama size"),
            "load_time": Measurement.missing(M.NOT_MEASURABLE, "preloaded", "s"),
        },
        "model_info": {"context_length": Measurement.reported(8192, "Tokens", "/props n_ctx")},
        "tasks": [TaskResult("chat", "general", TaskStatus.COMPLETED, 1.0, {"a": True})],
    }
    data.update(overrides)
    return ModelProfile(**data)  # type: ignore[arg-type]


class TestMeasurement:
    def test_fact_requires_value(self) -> None:
        with pytest.raises(ValueError):
            Measurement(None, M.MEASURED, method="x")
        with pytest.raises(ValueError):
            Measurement(None, M.REPORTED)

    def test_measured_requires_method(self) -> None:
        with pytest.raises(ValueError, match="Messmethode"):
            Measurement(1.0, M.MEASURED)

    def test_missing_rejects_fact_status(self) -> None:
        with pytest.raises(ValueError):
            Measurement.missing(M.MEASURED, "x")

    def test_configured_is_never_a_fact(self) -> None:
        m = Measurement.configured(42.0, "tok/s")
        assert not m.is_fact
        assert m.number() is None  # konfigurierte Zahlen zählen nie als Messwert

    def test_round_trip_and_str(self) -> None:
        m = Measurement.measured(12.345, "tok/s", "diff", "n=3")
        assert Measurement.from_dict(m.to_dict()) == m
        assert str(m) == "12.35 tok/s [MEASURED]"
        assert "NOT_MEASURABLE" in str(Measurement.missing(M.NOT_MEASURABLE, "why"))

    def test_bool_is_not_a_number(self) -> None:
        assert Measurement.detected(True).number() is None


def test_task_result_validation() -> None:
    with pytest.raises(ValueError):
        TaskResult("x", "general", TaskStatus.COMPLETED, None)
    with pytest.raises(ValueError):
        TaskResult("x", "general", TaskStatus.COMPLETED, 1.5)
    ok = TaskResult("x", "general", TaskStatus.COMPLETED, 1.0)
    assert ok.passed
    assert TaskResult.from_dict(ok.to_dict()) == ok


@pytest.mark.parametrize(
    ("score", "level"),
    [
        (1.0, CapabilityLevel.STRONG),
        (0.85, CapabilityLevel.STRONG),
        (0.84, CapabilityLevel.GOOD),
        (0.6, CapabilityLevel.GOOD),
        (0.3, CapabilityLevel.BASIC),
        (0.29, CapabilityLevel.NONE),
    ],
)
def test_score_to_level(score: float, level: CapabilityLevel) -> None:
    assert score_to_level(score) is level


def test_speed_mapping() -> None:
    assert tokens_per_second_to_speed(80) is Speed.FAST
    assert tokens_per_second_to_speed(20) is Speed.MEDIUM
    assert tokens_per_second_to_speed(3) is Speed.SLOW


class TestProfile:
    def test_json_has_required_top_level_fields(self) -> None:
        data = profile().to_dict()
        for key in (
            "model",
            "runtime",
            "hardware",
            "load_time",
            "tokens_per_second",
            "peak_vram",
            "peak_ram",
            "capabilities",
            "benchmark_version",
        ):
            assert key in data
        assert data["peak_vram"]["status"] == "UNMEASURED"  # nicht gemessen ≠ 0
        assert data["load_time"]["status"] == "NOT_MEASURABLE"

    def test_round_trip(self) -> None:
        p = profile()
        again = ModelProfile.from_dict(json.loads(json.dumps(p.to_dict())))
        assert again.to_dict() == p.to_dict()

    def test_unknown_schema_rejected(self) -> None:
        data = profile().to_dict()
        data["schema"] = 99
        with pytest.raises(ValueError):
            ModelProfile.from_dict(data)

    def test_overrides_same_hardware(self) -> None:
        o = profile().to_overrides(hardware_fingerprint="hw1", benchmark_version="1.0+abc")
        assert o.status is DataStatus.MEASURED
        assert o.reasoning_capability is CapabilityLevel.STRONG
        assert o.coding_capability is CapabilityLevel.BASIC
        assert o.vision_capability is CapabilityLevel.NONE  # NOT_SUPPORTED
        assert o.tool_calling is True
        assert o.speed is Speed.SLOW  # 12 tok/s
        assert o.memory_requirement == 5.5
        assert o.context_length == 8192
        assert "Score 0.90" in o.field_sources["reasoning_capability"]

    def test_other_hardware_keeps_only_capabilities(self) -> None:
        """Fähigkeit ist hardwareunabhängig, Leistung nicht."""
        o = profile().to_overrides(hardware_fingerprint="other", benchmark_version="1.0+abc")
        assert o.status is DataStatus.PARTIAL
        assert o.reasoning_capability is CapabilityLevel.STRONG
        assert o.speed is None and o.memory_requirement is None and o.context_length is None
        assert any("anderer Hardware" in n for n in o.notes)

    def test_other_benchmark_version_is_stale(self) -> None:
        o = profile().to_overrides(hardware_fingerprint="hw1", benchmark_version="2.0+zzz")
        assert o.status is DataStatus.STALE
        assert o.applied_fields() == []

    def test_configured_values_never_become_overrides(self) -> None:
        p = profile(
            capabilities={"coding": Measurement.configured(0.9)},
            performance={"tokens_per_second": Measurement.configured(99.0, "tok/s")},
            model_info={"context_length": Measurement.configured(4096, "Tokens")},
        )
        o = p.to_overrides(hardware_fingerprint="hw1", benchmark_version=None)
        assert o.status is DataStatus.UNMEASURED
        assert o.applied_fields() == []

    def test_failed_measurements_are_ignored(self) -> None:
        p = profile(
            capabilities={"coding": Measurement.missing(M.FAILED, "timeout")},
            performance={},
            model_info={},
        )
        o = p.to_overrides(hardware_fingerprint=None, benchmark_version=None)
        assert o.coding_capability is None


class TestStore:
    def test_save_load_list_history(self, tmp_path: Path) -> None:
        store = ProfileStore(tmp_path)
        path = store.save(profile())
        assert path == tmp_path / "test-model.json"
        assert store.load("test-model") is not None
        assert [p.model for p in store.list()] == ["test-model"]
        assert len(list((tmp_path / "history").glob("*.json"))) == 1
        assert not list(tmp_path.glob(".tmp-*"))

    def test_slug_prevents_path_escape(self, tmp_path: Path) -> None:
        store = ProfileStore(tmp_path)
        assert store.path_for("../../etc/passwd").parent == tmp_path
        assert store.slug("org/model:7b") == "org_model_7b"

    def test_overrides_for(self, tmp_path: Path) -> None:
        store = ProfileStore(tmp_path, hardware_fingerprint="hw1", benchmark_version="1.0+abc")
        meta = make_meta()
        assert store.overrides_for(meta) is None
        store.save(profile())
        o = store.overrides_for(meta)
        assert o is not None and o.status is DataStatus.MEASURED

    def test_corrupt_profile_is_unmeasured(self, tmp_path: Path) -> None:
        store = ProfileStore(tmp_path)
        (tmp_path / "test-model.json").write_text("{not json")
        o = store.overrides_for(make_meta())
        assert o is not None and o.status is DataStatus.UNMEASURED
        assert store.list() == []

    def test_env_default_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NOVA_BENCHMARK_DIR", str(tmp_path / "b"))
        assert ProfileStore().root == tmp_path / "b"


def test_store_cache_follows_file_changes(tmp_path: Path) -> None:
    import os

    store = ProfileStore(tmp_path)
    store.save(profile())
    first = store.load("test-model")
    assert store.load("test-model") is first  # unverändert → aus dem Cache
    path = store.path_for("test-model")
    data = json.loads(path.read_text())
    data["notes"] = ["neu"]
    path.write_text(json.dumps(data))
    os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 1_000_000))
    reloaded = store.load("test-model")
    assert reloaded is not None and reloaded.notes == ["neu"]
    path.unlink()
    assert store.load("test-model") is None
