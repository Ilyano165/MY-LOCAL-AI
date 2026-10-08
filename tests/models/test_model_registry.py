from __future__ import annotations

from pathlib import Path

import pytest

from models.base import ModelNotFoundError, NoSuitableModelError
from models.capabilities import TaskRequirements, TaskType
from models.model_registry import ModelRegistry
from tests.models.fakes import make_meta


@pytest.fixture
def registry() -> ModelRegistry:
    return ModelRegistry(
        [
            make_meta(
                name="small",
                speed="fast",
                memory_requirement=2,
                context_length=8192,
                reasoning_capability="basic",
                coding_capability="basic",
            ),
            make_meta(
                name="coder",
                coding_capability="strong",
                reasoning_capability="good",
                context_length=131072,
                memory_requirement=18,
            ),
            make_meta(
                name="thinker",
                reasoning_capability="strong",
                coding_capability="good",
                speed="slow",
                memory_requirement=14,
            ),
            make_meta(
                name="seer",
                vision_capability="good",
                tool_calling=False,
                reasoning_capability="basic",
                coding_capability="none",
            ),
        ]
    )


def test_register_get_list(registry: ModelRegistry) -> None:
    assert len(registry) == 4
    assert registry.get("coder").name == "coder"
    assert [m.name for m in registry.list()] == ["coder", "seer", "small", "thinker"]
    assert "coder" in registry and "nope" not in registry
    assert [m.name for m in registry] == ["coder", "seer", "small", "thinker"]


def test_list_filters_by_provider() -> None:
    reg = ModelRegistry([make_meta(name="a", provider="p1"), make_meta(name="b", provider="p2")])
    assert [m.name for m in reg.list(provider="p2")] == ["b"]


def test_duplicate_registration_rejected_unless_replace(registry: ModelRegistry) -> None:
    with pytest.raises(ValueError, match="bereits registriert"):
        registry.register(make_meta(name="coder"))
    registry.register(make_meta(name="coder", context_length=4096), replace=True)
    assert registry.get("coder").context_length == 4096


def test_register_rejects_wrong_type(registry: ModelRegistry) -> None:
    with pytest.raises(TypeError):
        registry.register({"name": "x"})  # type: ignore[arg-type]


def test_get_unknown_lists_known(registry: ModelRegistry) -> None:
    with pytest.raises(ModelNotFoundError, match="coder"):
        registry.get("missing")


def test_unregister(registry: ModelRegistry) -> None:
    registry.unregister("seer")
    assert "seer" not in registry
    with pytest.raises(ModelNotFoundError):
        registry.unregister("seer")


@pytest.mark.parametrize(
    ("task", "expected"),
    [
        (TaskType.CODING, "coder"),
        (TaskType.REASONING, "thinker"),
        (TaskType.FAST, "small"),
        (TaskType.VISION, "seer"),
        ("general", "coder"),
    ],
)
def test_find_best_for_task_types(registry: ModelRegistry, task: TaskType, expected: str) -> None:
    assert registry.find_best_for(task).name == expected


def test_hard_requirements_filter(registry: ModelRegistry) -> None:
    req = TaskRequirements(task_type=TaskType.CODING, max_memory_gb=15)
    assert registry.find_best_for(req).name == "thinker"
    req = TaskRequirements(task_type=TaskType.FAST, min_context_length=100_000)
    assert registry.find_best_for(req).name == "coder"
    req = TaskRequirements(task_type=TaskType.VISION, needs_tool_calling=True)
    with pytest.raises(NoSuitableModelError, match="seer: kein Tool-Calling"):
        registry.find_best_for(req)


def test_exclude_and_rank(registry: ModelRegistry) -> None:
    assert registry.find_best_for(TaskType.CODING, exclude=["coder"]).name == "thinker"
    ranked = [m.name for m in registry.rank_for(TaskType.CODING)]
    assert ranked[0] == "coder" and "seer" in ranked
    assert ranked.index("seer") > ranked.index("small")  # seer kann nicht coden


def test_tie_break_is_deterministic() -> None:
    reg = ModelRegistry([make_meta(name="zeta"), make_meta(name="alpha"), make_meta(name="mid")])
    assert reg.find_best_for(TaskType.GENERAL).name == "alpha"


def test_empty_registry_raises() -> None:
    with pytest.raises(NoSuitableModelError, match="Registry leer"):
        ModelRegistry().find_best_for(TaskType.GENERAL)


def test_from_toml(tmp_path: Path) -> None:
    cfg = tmp_path / "models.toml"
    cfg.write_text(
        """
[[models]]
name = "m1"
provider = "local"
parameter_count = "4B"
context_length = 32768
reasoning_capability = "basic"
coding_capability = "basic"
vision_capability = "none"
tool_calling = true
speed = "fast"
memory_requirement = "3.5GB"
quantization = "Q4_K_M"
license = "apache-2.0"
""",
        encoding="utf-8",
    )
    reg = ModelRegistry.from_toml(cfg)
    meta = reg.get("m1")
    assert meta.parameter_count == 4_000_000_000
    assert meta.memory_requirement == 3.5
    assert meta.extra["license"] == "apache-2.0"


def test_from_mapping_rejects_non_list() -> None:
    with pytest.raises(ValueError):
        ModelRegistry.from_mapping({"models": {"name": "x"}})


def test_example_config_is_valid() -> None:
    path = Path(__file__).resolve().parents[2] / "config" / "models.example.toml"
    reg = ModelRegistry.from_toml(path)
    assert len(reg) >= 1
