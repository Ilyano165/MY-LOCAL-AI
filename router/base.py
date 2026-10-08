"""Datentypen und Schnittstellen des Model Routers.

Ablauf: :class:`RoutingRequest` → :class:`TaskClassifier` (Kategorie, Komplexität, Bedarf) →
:class:`ModelRouter` (harte Filter: Verfügbarkeit, Speicher, Kontext, Tools, Vision; danach
Rangfolge) → :class:`RoutingDecision` mit Begründung, abgelehnten Kandidaten und Fallbacks.

Klassifikator und Router sind getrennt austauschbar: regelbasiert, LLM-basiert oder später
gelernt (Trainingsdaten liefern die Routing-Logs inklusive Ergebnis-Rückmeldung).
"""

from __future__ import annotations

import abc
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import IntEnum, StrEnum
from typing import Any

from models.base import ChatRequest, Role
from models.capabilities import ModelMetadata, TaskRequirements, TaskType
from models.inference import RoutedChoice


class RoutingCategory(StrEnum):
    FAST = "fast"
    GENERAL = "general"
    CODING = "coding"
    REASONING = "reasoning"
    VISION = "vision"
    LONG_CONTEXT = "long_context"

    @property
    def task_type(self) -> TaskType:
        """Fähigkeit, nach der innerhalb der Kategorie gerankt wird."""
        return {
            RoutingCategory.FAST: TaskType.FAST,
            RoutingCategory.GENERAL: TaskType.GENERAL,
            RoutingCategory.CODING: TaskType.CODING,
            RoutingCategory.REASONING: TaskType.REASONING,
            RoutingCategory.VISION: TaskType.VISION,
            RoutingCategory.LONG_CONTEXT: TaskType.GENERAL,
        }[self]

    @classmethod
    def from_task_type(cls, task_type: TaskType) -> RoutingCategory:
        return cls(task_type.value)


class Complexity(IntEnum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3

    @classmethod
    def parse(cls, value: Complexity | str | int) -> Complexity:
        if isinstance(value, Complexity):
            return value
        if isinstance(value, int):
            return cls(value)
        return cls[str(value).strip().upper()]


class Latency(StrEnum):
    REALTIME = "realtime"  # Antwort sofort (Chat, Klassifikation)
    NORMAL = "normal"
    BATCH = "batch"  # Qualität vor Geschwindigkeit (Hintergrundaufgaben)


class NoModelAvailableError(Exception):
    """Kein verfügbares Modell erfüllt die harten Anforderungen."""

    def __init__(self, message: str, rejected: dict[str, list[str]] | None = None) -> None:
        super().__init__(message)
        self.rejected = rejected or {}


def estimate_tokens(text: str) -> int:
    """Grobe Schätzung (≈ 4 Zeichen/Token für EN, etwas weniger für DE/Code).

    Bewusst konservativ (3.5 Zeichen/Token): lieber ein Modell mit zu großem als mit zu
    kleinem Kontextfenster wählen. Exakte Zählung braucht den Tokenizer des Zielmodells.
    """
    return int(len(text) / 3.5) + 1 if text else 0


@dataclass
class RoutingRequest:
    task: str
    conversation_tokens: int = 0
    """Geschätzte Tokens des bisherigen Gesprächs/Kontexts (ohne ``task``)."""
    required_tools: tuple[str, ...] = ()
    needs_vision: bool = False
    complexity: Complexity | None = None
    """Komplexitätsschätzung von außen (z. B. Task Analyzer); sonst schätzt der Klassifikator."""
    category_hint: RoutingCategory | None = None
    latency: Latency = Latency.NORMAL
    expected_output_tokens: int = 2048
    min_context_tokens: int = 0

    @property
    def task_tokens(self) -> int:
        return estimate_tokens(self.task)

    @property
    def total_input_tokens(self) -> int:
        return self.conversation_tokens + self.task_tokens

    @classmethod
    def from_chat(
        cls, request: ChatRequest, requirements: TaskRequirements | None = None, **overrides: Any
    ) -> RoutingRequest:
        """Leitet die Routing-Anfrage aus einem Chat-Request ab."""
        users = [m for m in request.messages if m.role is Role.USER]
        task = users[-1].content if users else request.messages[-1].content
        context = sum(estimate_tokens(m.content) for m in request.messages) - estimate_tokens(task)
        context += sum(estimate_tokens(str(t.parameters)) + 20 for t in request.tools)
        hint = None
        min_context = 0
        needs_vision = False
        if requirements is not None:
            if requirements.task_type is not TaskType.GENERAL:
                hint = RoutingCategory.from_task_type(requirements.task_type)
            min_context = requirements.min_context_length
            needs_vision = requirements.requires_vision
        tools = tuple(t.name for t in request.tools)
        if requirements is not None and requirements.needs_tool_calling and not tools:
            tools = ("*",)
        data: dict[str, Any] = {
            "task": task,
            "conversation_tokens": max(context, 0),
            "required_tools": tools,
            "category_hint": hint,
            "min_context_tokens": min_context,
            "needs_vision": needs_vision,
            "expected_output_tokens": request.params.max_tokens or 2048,
        }
        data.update(overrides)
        return cls(**data)


@dataclass
class TaskClassification:
    category: RoutingCategory
    complexity: Complexity
    needs_tools: bool
    context_tokens: int
    """Benötigtes Kontextfenster (Eingabe + Antwortreserve), geschätzt."""
    confidence: float
    """Sicherheit des Klassifikators in [0, 1] – Heuristik, keine Wahrscheinlichkeit."""
    secondary: RoutingCategory | None = None
    """Bei LONG_CONTEXT: die inhaltliche Kategorie (z. B. CODING)."""
    signals: list[str] = field(default_factory=list)
    reason: str = ""
    classifier: str = "rules"


@dataclass(frozen=True)
class Rejection:
    model: str
    reasons: tuple[str, ...]


@dataclass
class RoutingDecision:
    model: ModelMetadata
    classification: TaskClassification
    reason: str
    fallbacks: list[ModelMetadata] = field(default_factory=list)
    rejected: list[Rejection] = field(default_factory=list)
    considered: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    request: RoutingRequest | None = None
    data_status: dict[str, str] = field(default_factory=dict)
    """Modell → MEASURED/PARTIAL/STALE/UNMEASURED (Herkunft der Routing-Daten)."""
    ranker: str = "rules"
    """Welche Komponente die Rangfolge bestimmt hat (``rules``, ``learned``, …)."""
    scores: dict[str, float] = field(default_factory=dict)
    """Scores des Rankers je Kandidat (leer bei Regeln)."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))

    @property
    def category(self) -> RoutingCategory:
        return self.classification.category

    @property
    def complexity(self) -> Complexity:
        return self.classification.complexity

    def to_dict(self) -> dict[str, Any]:
        c = self.classification
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "task": (self.request.task[:500] if self.request else None),
            "category": c.category.value,
            "secondary": c.secondary.value if c.secondary else None,
            "complexity": c.complexity.name,
            "needs_tools": c.needs_tools,
            "context_tokens": c.context_tokens,
            "confidence": round(c.confidence, 2),
            "classifier": c.classifier,
            "signals": c.signals,
            "selected_model": self.model.name,
            "reason": self.reason,
            "fallbacks": [m.name for m in self.fallbacks],
            "considered": self.considered,
            "rejected": {r.model: list(r.reasons) for r in self.rejected},
            "notes": self.notes,
            "data_status": self.data_status,
            "ranker": self.ranker,
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
            "needs_vision": self.request.needs_vision if self.request else None,
            "conversation_tokens": self.request.conversation_tokens if self.request else None,
            "latency": self.request.latency.value if self.request else None,
        }


class TaskClassifier(abc.ABC):
    name: str = "classifier"

    @abc.abstractmethod
    async def classify(self, request: RoutingRequest) -> TaskClassification: ...


class ModelRouter(abc.ABC):
    @abc.abstractmethod
    async def route(self, request: RoutingRequest) -> RoutingDecision:
        """Wählt ein *verfügbares* Modell; wirft :class:`NoModelAvailableError` sonst."""

    async def route_chat(
        self, request: ChatRequest, requirements: TaskRequirements | None
    ) -> RoutedChoice:
        """Schnittstelle für :class:`models.inference.InferenceEngine`."""
        decision = await self.route(RoutingRequest.from_chat(request, requirements))
        return RoutedChoice(decision.model, list(decision.fallbacks), decision.id)

    def report_failure(self, model_name: str, reason: str) -> None:  # noqa: B027 – optional
        """Akuter Ausfall eines Modells (Default: ignoriert)."""

    def record_outcome(
        self,
        decision_id: str,
        *,
        success: bool,
        verdict: str | None = None,
        quality: float | None = None,
        latency_ms: float | None = None,
    ) -> None:
        """Ergebnis einer gerouteten Aufgabe ins Routing-Log (falls vorhanden) nachtragen."""
        log = getattr(self, "log", None)
        if log is not None:
            log.record_outcome(
                decision_id,
                success=success,
                verdict=verdict,
                quality=quality,
                latency_ms=latency_ms,
            )
