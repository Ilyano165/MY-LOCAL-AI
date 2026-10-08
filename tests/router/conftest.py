from __future__ import annotations

from typing import Any

import pytest

from models.capabilities import ModelMetadata
from models.model_registry import ModelRegistry


def model(name: str, **overrides: Any) -> ModelMetadata:
    data: dict[str, Any] = {
        "name": name,
        "provider": "local",
        "parameter_count": "8B",
        "context_length": 32768,
        "reasoning_capability": "good",
        "coding_capability": "good",
        "vision_capability": "none",
        "tool_calling": True,
        "speed": "medium",
        "memory_requirement": 6.0,
        "quantization": "Q4_K_M",
    }
    data.update(overrides)
    return ModelMetadata.from_dict(data)


def fleet() -> list[ModelMetadata]:
    """Typischer lokaler Bestand: klein/schnell, Coder, Denker, Vision, Langkontext."""
    return [
        model(
            "small-fast",
            parameter_count="4B",
            speed="fast",
            memory_requirement=3,
            reasoning_capability="basic",
            coding_capability="basic",
        ),
        model(
            "coder",
            parameter_count="27B",
            coding_capability="strong",
            memory_requirement=18,
            context_length=131072,
        ),
        model(
            "thinker",
            parameter_count="20B",
            reasoning_capability="strong",
            speed="slow",
            memory_requirement=14,
        ),
        model(
            "seer",
            vision_capability="good",
            tool_calling=False,
            coding_capability="none",
            reasoning_capability="basic",
        ),
        model(
            "longctx",
            parameter_count="30B",
            context_length=262144,
            memory_requirement=20,
            speed="slow",
        ),
    ]


@pytest.fixture
def registry() -> ModelRegistry:
    return ModelRegistry(fleet())
