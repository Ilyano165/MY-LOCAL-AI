from __future__ import annotations

from pathlib import Path

import pytest

from models.capabilities import (
    CapabilityLevel,
    ModelMetadata,
    Speed,
    TaskRequirements,
    TaskType,
    parse_memory_gb,
    parse_parameter_count,
)
from tests.models.fakes import make_meta


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("8B", 8_000_000_000),
        ("0.6B", 600_000_000),
        ("500M", 500_000_000),
        ("80b", 80 * 10**9),
        (7_000_000_000, 7_000_000_000),
        ("1.5T", 1_500_000_000_000),
    ],
)
def test_parse_parameter_count(raw: object, expected: int) -> None:
    assert parse_parameter_count(raw) == expected  # type: ignore[arg-type]


@pytest.mark.parametrize("raw", ["", "acht", "8X", "-1B", 0, -5])
def test_parse_parameter_count_rejects_invalid(raw: object) -> None:
    with pytest.raises(ValueError):
        parse_parameter_count(raw)  # type: ignore[arg-type]


def test_parse_parameter_count_rejects_bool() -> None:
    with pytest.raises(TypeError):
        parse_parameter_count(True)


@pytest.mark.parametrize(
    ("raw", "expected"), [(16, 16.0), ("16GB", 16.0), ("4.5 GiB", 4.5), ("2g", 2.0)]
)
def test_parse_memory(raw: object, expected: float) -> None:
    assert parse_memory_gb(raw) == expected  # type: ignore[arg-type]


@pytest.mark.parametrize("raw", ["16TB", "viel", 0, -1])
def test_parse_memory_rejects_invalid(raw: object) -> None:
    with pytest.raises(ValueError):
        parse_memory_gb(raw)  # type: ignore[arg-type]


def test_capability_and_speed_parse() -> None:
    assert CapabilityLevel.parse("Strong") is CapabilityLevel.STRONG
    assert CapabilityLevel.parse(1) is CapabilityLevel.BASIC
    assert CapabilityLevel.parse(False) is CapabilityLevel.NONE
    assert Speed.parse("fast") is Speed.FAST
    with pytest.raises(ValueError):
        CapabilityLevel.parse("genius")
    with pytest.raises(ValueError):
        Speed.parse("warp")


def test_metadata_normalizes_fields() -> None:
    meta = make_meta(local_path="~/models/x.gguf", served_name="alias")
    assert meta.parameter_count == 8_000_000_000
    assert meta.reasoning_capability is CapabilityLevel.GOOD
    assert meta.speed is Speed.MEDIUM
    assert isinstance(meta.local_path, Path)
    assert "~" not in str(meta.local_path)
    assert meta.runtime_name == "alias"
    assert make_meta().runtime_name == "test-model"


def test_metadata_is_hashable_and_immutable() -> None:
    meta = make_meta()
    assert hash(meta) == hash(make_meta())
    with pytest.raises(AttributeError):
        meta.name = "other"  # type: ignore[misc]


@pytest.mark.parametrize(
    "override",
    [
        {"name": ""},
        {"provider": " "},
        {"context_length": 0},
        {"quantization": ""},
        {"memory_requirement": 0},
    ],
)
def test_metadata_validation(override: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        make_meta(**override)


def test_metadata_tool_calling_must_be_bool() -> None:
    with pytest.raises(TypeError):
        make_meta(tool_calling="yes")


def test_from_dict_reports_missing_fields() -> None:
    with pytest.raises(ValueError, match="quantization"):
        ModelMetadata.from_dict(
            {
                "name": "x",
                "provider": "p",
                "parameter_count": 1,
                "context_length": 1,
                "reasoning_capability": 0,
                "coding_capability": 0,
                "vision_capability": 0,
                "tool_calling": False,
                "speed": 1,
                "memory_requirement": 1,
            }
        )


def test_to_dict_roundtrip_keeps_extra_fields() -> None:
    meta = make_meta(license="apache-2.0", local_path="/m/x.gguf")
    data = meta.to_dict()
    assert data["license"] == "apache-2.0"
    assert data["coding_capability"] == "good"
    assert ModelMetadata.from_dict(data) == meta


def test_unmet_requirements() -> None:
    meta = make_meta(context_length=8192, tool_calling=False, memory_requirement=20)
    req = TaskRequirements(
        task_type=TaskType.VISION,
        min_context_length=16384,
        needs_tool_calling=True,
        min_reasoning="strong",
        min_coding="strong",
        max_memory_gb=16,
    )
    reasons = meta.unmet_requirements(req)
    assert len(reasons) == 6
    assert make_meta().unmet_requirements(TaskRequirements()) == []


def test_task_requirements_coerce_and_validation() -> None:
    assert TaskRequirements.coerce("coding").task_type is TaskType.CODING
    assert TaskRequirements.coerce(TaskType.VISION).requires_vision
    with pytest.raises(ValueError):
        TaskRequirements.coerce("dance")
    with pytest.raises(ValueError):
        TaskRequirements(min_context_length=-1)
    with pytest.raises(ValueError):
        TaskRequirements(max_memory_gb=0)
