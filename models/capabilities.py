"""Modell-Metadaten, Fähigkeitsstufen und Aufgabenanforderungen.

Dieses Modul kennt keine konkreten Modelle. Welche Modelle existieren, steht
ausschließlich in der Konfiguration bzw. im :class:`~models.model_registry.ModelRegistry`.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from pathlib import Path
from typing import Any


class CapabilityLevel(IntEnum):
    """Grobe, vergleichbare Fähigkeitsstufe eines Modells in einem Bereich."""

    NONE = 0
    BASIC = 1
    GOOD = 2
    STRONG = 3

    @classmethod
    def parse(cls, value: CapabilityLevel | int | str | bool) -> CapabilityLevel:
        if isinstance(value, CapabilityLevel):
            return value
        if isinstance(value, bool):
            return cls.GOOD if value else cls.NONE
        if isinstance(value, int):
            return cls(value)
        if isinstance(value, str):
            try:
                return cls[value.strip().upper()]
            except KeyError:
                raise ValueError(f"Unbekannte Fähigkeitsstufe: {value!r}") from None
        raise TypeError(f"Fähigkeitsstufe muss int/str sein, nicht {type(value).__name__}")


class Speed(IntEnum):
    """Relative Generierungsgeschwindigkeit auf der Zielhardware."""

    SLOW = 1
    MEDIUM = 2
    FAST = 3

    @classmethod
    def parse(cls, value: Speed | int | str) -> Speed:
        if isinstance(value, Speed):
            return value
        if isinstance(value, int):
            return cls(value)
        if isinstance(value, str):
            try:
                return cls[value.strip().upper()]
            except KeyError:
                raise ValueError(f"Unbekannte Geschwindigkeit: {value!r}") from None
        raise TypeError(f"Geschwindigkeit muss int/str sein, nicht {type(value).__name__}")


class TaskType(StrEnum):
    """Aufgabenklassen, für die der Registry ein passendes Modell sucht."""

    FAST = "fast"
    CODING = "coding"
    REASONING = "reasoning"
    VISION = "vision"
    GENERAL = "general"


_PARAM_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([KMBT]?)\s*$", re.IGNORECASE)
_PARAM_FACTORS = {"": 1, "K": 10**3, "M": 10**6, "B": 10**9, "T": 10**12}
_MEM_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(GB|GIB|G)?\s*$", re.IGNORECASE)


def parse_parameter_count(value: int | float | str) -> int:
    """``"8B"`` → 8_000_000_000, ``"0.6B"`` → 600_000_000, ``500_000_000`` → unverändert."""
    if isinstance(value, bool):
        raise TypeError("parameter_count darf kein bool sein")
    if isinstance(value, int | float):
        count = int(value)
    elif isinstance(value, str):
        match = _PARAM_RE.match(value)
        if not match:
            raise ValueError(f"Ungültige Parameteranzahl: {value!r}")
        count = round(float(match.group(1)) * _PARAM_FACTORS[match.group(2).upper()])
    else:
        raise TypeError(f"parameter_count: unerwarteter Typ {type(value).__name__}")
    if count <= 0:
        raise ValueError("parameter_count muss > 0 sein")
    return count


def parse_memory_gb(value: int | float | str) -> float:
    """Speicherbedarf in GB; akzeptiert Zahl oder ``"16GB"``."""
    if isinstance(value, bool):
        raise TypeError("memory_requirement darf kein bool sein")
    if isinstance(value, int | float):
        gb = float(value)
    elif isinstance(value, str):
        match = _MEM_RE.match(value)
        if not match:
            raise ValueError(
                f"Ungültiger Speicherbedarf: {value!r} (erwartet z. B. 16 oder '16GB')"
            )
        gb = float(match.group(1))
    else:
        raise TypeError(f"memory_requirement: unerwarteter Typ {type(value).__name__}")
    if gb <= 0:
        raise ValueError("memory_requirement muss > 0 sein")
    return gb


@dataclass(frozen=True, slots=True)
class TaskRequirements:
    """Harte Anforderungen und Zielrichtung für die Modellauswahl."""

    task_type: TaskType = TaskType.GENERAL
    min_context_length: int = 0
    needs_tool_calling: bool = False
    needs_vision: bool = False
    min_reasoning: CapabilityLevel = CapabilityLevel.NONE
    min_coding: CapabilityLevel = CapabilityLevel.NONE
    max_memory_gb: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_type", TaskType(self.task_type))
        object.__setattr__(self, "min_reasoning", CapabilityLevel.parse(self.min_reasoning))
        object.__setattr__(self, "min_coding", CapabilityLevel.parse(self.min_coding))
        if self.min_context_length < 0:
            raise ValueError("min_context_length darf nicht negativ sein")
        if self.max_memory_gb is not None and self.max_memory_gb <= 0:
            raise ValueError("max_memory_gb muss > 0 sein")

    @property
    def requires_vision(self) -> bool:
        return self.needs_vision or self.task_type is TaskType.VISION

    @classmethod
    def coerce(cls, task: TaskRequirements | TaskType | str) -> TaskRequirements:
        if isinstance(task, TaskRequirements):
            return task
        return cls(task_type=TaskType(task))


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    """Beschreibung eines lokal verfügbaren Modells.

    ``provider`` verweist auf den Namen eines registrierten
    :class:`~models.base.ModelProvider` (z. B. eine laufende llama.cpp-Instanz).
    ``served_name`` ist der Name, unter dem die Runtime das Modell kennt;
    standardmäßig identisch mit ``name``.
    ``memory_requirement`` ist in GB angegeben (Gewichte + typischer KV-Cache).
    """

    name: str
    provider: str
    parameter_count: int
    context_length: int
    reasoning_capability: CapabilityLevel
    coding_capability: CapabilityLevel
    vision_capability: CapabilityLevel
    tool_calling: bool
    speed: Speed
    memory_requirement: float
    quantization: str
    local_path: Path | None = None
    served_name: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name darf nicht leer sein")
        if not isinstance(self.provider, str) or not self.provider.strip():
            raise ValueError(f"{self.name}: provider darf nicht leer sein")
        if not isinstance(self.context_length, int) or self.context_length <= 0:
            raise ValueError(f"{self.name}: context_length muss eine positive Ganzzahl sein")
        if not isinstance(self.tool_calling, bool):
            raise TypeError(f"{self.name}: tool_calling muss bool sein")
        if not isinstance(self.quantization, str) or not self.quantization.strip():
            raise ValueError(
                f"{self.name}: quantization darf nicht leer sein (z. B. 'Q4_K_M', 'F16')"
            )
        set_ = object.__setattr__
        set_(self, "parameter_count", parse_parameter_count(self.parameter_count))
        set_(self, "memory_requirement", parse_memory_gb(self.memory_requirement))
        set_(self, "reasoning_capability", CapabilityLevel.parse(self.reasoning_capability))
        set_(self, "coding_capability", CapabilityLevel.parse(self.coding_capability))
        set_(self, "vision_capability", CapabilityLevel.parse(self.vision_capability))
        set_(self, "speed", Speed.parse(self.speed))
        if self.local_path is not None:
            set_(self, "local_path", Path(self.local_path).expanduser())
        set_(self, "extra", dict(self.extra))

    @property
    def runtime_name(self) -> str:
        return self.served_name or self.name

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ModelMetadata:
        known = {f for f in cls.__dataclass_fields__ if f != "extra"}
        missing = {
            "name",
            "provider",
            "parameter_count",
            "context_length",
            "reasoning_capability",
            "coding_capability",
            "vision_capability",
            "tool_calling",
            "speed",
            "memory_requirement",
            "quantization",
        } - set(data)
        if missing:
            raise ValueError(f"Modell-Metadaten unvollständig, fehlend: {sorted(missing)}")
        kwargs = {k: v for k, v in data.items() if k in known}
        extra = {k: v for k, v in data.items() if k not in known}
        return cls(**kwargs, extra=extra)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "provider": self.provider,
            "parameter_count": self.parameter_count,
            "context_length": self.context_length,
            "reasoning_capability": self.reasoning_capability.name.lower(),
            "coding_capability": self.coding_capability.name.lower(),
            "vision_capability": self.vision_capability.name.lower(),
            "tool_calling": self.tool_calling,
            "speed": self.speed.name.lower(),
            "memory_requirement": self.memory_requirement,
            "quantization": self.quantization,
            "local_path": str(self.local_path) if self.local_path else None,
            "served_name": self.served_name,
            **dict(self.extra),
        }

    def unmet_requirements(self, req: TaskRequirements) -> list[str]:
        """Liste der verletzten harten Anforderungen (leer = geeignet)."""
        reasons: list[str] = []
        if self.context_length < req.min_context_length:
            reasons.append(f"context_length {self.context_length} < {req.min_context_length}")
        if req.needs_tool_calling and not self.tool_calling:
            reasons.append("kein Tool-Calling")
        if req.requires_vision and self.vision_capability is CapabilityLevel.NONE:
            reasons.append("keine Vision-Fähigkeit")
        if self.reasoning_capability < req.min_reasoning:
            reasons.append(f"reasoning {self.reasoning_capability.name} < {req.min_reasoning.name}")
        if self.coding_capability < req.min_coding:
            reasons.append(f"coding {self.coding_capability.name} < {req.min_coding.name}")
        if req.max_memory_gb is not None and self.memory_requirement > req.max_memory_gb:
            reasons.append(f"memory {self.memory_requirement} GB > {req.max_memory_gb} GB")
        return reasons

    def ranking_key(self, task_type: TaskType) -> tuple[float, ...]:
        """Sortierschlüssel (größer = besser) für eine Aufgabenklasse.

        Lexikografische Tupel statt gewichteter Summen: die Priorität der Kriterien
        ist explizit und nachvollziehbar.
        """
        r, c, v = self.reasoning_capability, self.coding_capability, self.vision_capability
        tools = 1 if self.tool_calling else 0
        lighter = -self.memory_requirement
        match task_type:
            case TaskType.FAST:
                return (self.speed, lighter, (r + c) / 2)
            case TaskType.CODING:
                return (c, tools, r, self.speed, lighter)
            case TaskType.REASONING:
                return (r, c, tools, self.speed, lighter)
            case TaskType.VISION:
                return (v, r, self.speed, lighter)
            case TaskType.GENERAL:
                return ((r + c) / 2, tools, self.speed, lighter)
