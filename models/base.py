"""Backend-unabhängige Datentypen, Fehler und die ``ModelProvider``-Abstraktion.

Alles oberhalb der Modellschicht spricht nur mit diesen Typen. Ein neues Backend
(andere Runtime, In-Process-Inference, …) implementiert ``ModelProvider`` – der Rest
der Anwendung bleibt unverändert.
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from models.capabilities import ModelMetadata

# --------------------------------------------------------------------------- Fehler


class ModelError(Exception):
    """Basisklasse aller Fehler der Modellschicht."""


class ModelNotFoundError(ModelError):
    """Modell ist weder im Registry noch in der Runtime bekannt."""


class ProviderNotRegisteredError(ModelError):
    """Ein Modell verweist auf einen Provider, der nicht registriert ist."""


class ProviderUnavailableError(ModelError):
    """Runtime nicht erreichbar oder (noch) nicht bereit, z. B. während des Ladens."""


class ModelTimeoutError(ModelError):
    """Antwort kam nicht innerhalb des erlaubten Zeitlimits."""


class ContextLengthExceededError(ModelError):
    """Die Anfrage passt nicht in das Kontextfenster des Modells."""


class InvalidResponseError(ModelError):
    """Die Runtime hat eine unerwartete oder nicht parsebare Antwort geliefert."""


class NoSuitableModelError(ModelError):
    """Kein registriertes Modell erfüllt die Anforderungen der Aufgabe."""


# --------------------------------------------------------------------------- Datentypen


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict, hash=False)


@dataclass(frozen=True, slots=True)
class ImageInput:
    """Bild für Vision-Modelle (Rohdaten; die Runtime-Adapter kodieren selbst)."""

    data: bytes = field(repr=False)
    mime_type: str = "image/png"

    def __post_init__(self) -> None:
        if not self.data:
            raise ValueError("Leeres Bild")
        if not self.mime_type.startswith("image/"):
            raise ValueError(f"Kein Bild-MIME-Typ: {self.mime_type}")


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    images: tuple[ImageInput, ...] = ()

    @classmethod
    def system(cls, content: str) -> Message:
        return cls(Role.SYSTEM, content)

    @classmethod
    def user(cls, content: str, images: Sequence[ImageInput] = ()) -> Message:
        return cls(Role.USER, content, images=tuple(images))

    @classmethod
    def assistant(cls, content: str = "", tool_calls: Sequence[ToolCall] = ()) -> Message:
        return cls(Role.ASSISTANT, content, tuple(tool_calls))

    @classmethod
    def tool(cls, tool_call_id: str, content: str) -> Message:
        return cls(Role.TOOL, content, tool_call_id=tool_call_id)


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Werkzeugbeschreibung, wie sie das Modell sieht (``parameters`` = JSON Schema)."""

    name: str
    description: str
    parameters: Mapping[str, Any] = field(
        default_factory=lambda: {"type": "object", "properties": {}}, hash=False
    )


@dataclass(frozen=True, slots=True)
class GenerationParams:
    temperature: float = 0.2
    max_tokens: int | None = None
    top_p: float | None = None
    stop: tuple[str, ...] = ()
    seed: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature muss in [0, 2] liegen")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("max_tokens muss > 0 sein")
        if self.top_p is not None and not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p muss in (0, 1] liegen")


@dataclass(frozen=True, slots=True)
class ChatRequest:
    messages: tuple[Message, ...]
    tools: tuple[ToolSpec, ...] = ()
    params: GenerationParams = field(default_factory=GenerationParams)
    timeout_s: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "messages", tuple(self.messages))
        object.__setattr__(self, "tools", tuple(self.tools))
        if not self.messages:
            raise ValueError("ChatRequest braucht mindestens eine Nachricht")
        if self.timeout_s is not None and self.timeout_s <= 0:
            raise ValueError("timeout_s muss > 0 sein")


class FinishReason(StrEnum):
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True, slots=True)
class ChatResponse:
    message: Message
    finish_reason: FinishReason
    usage: TokenUsage
    model: str
    latency_ms: float
    runtime_stats: Mapping[str, float] = field(default_factory=dict, hash=False)
    """Von der Runtime *gemeldete* Kennzahlen dieses Aufrufs (z. B. llama.cpp ``timings``:
    ``prompt_per_second``, ``predicted_per_second``). Leer, wenn die Runtime nichts meldet."""


@dataclass(frozen=True, slots=True)
class StreamChunk:
    """Teil einer gestreamten Antwort.

    ``delta`` ist neuer Text. Der letzte Chunk trägt ``finish_reason`` und – falls die Runtime
    sie meldet – ``usage``/``runtime_stats``. ``streamed=False`` heißt: die Runtime/der Provider
    kann nicht streamen, die ganze Antwort kam in einem Stück (ehrlich gekennzeichnet).
    """

    delta: str = ""
    finish_reason: FinishReason | None = None
    usage: TokenUsage | None = None
    runtime_stats: Mapping[str, float] = field(default_factory=dict, hash=False)
    model: str = ""
    streamed: bool = True


@dataclass(frozen=True, slots=True)
class ProviderHealth:
    """Zustand der Runtime selbst (nicht eines einzelnen Modells)."""

    reachable: bool
    ready: bool
    detail: str = ""


# --------------------------------------------------------------------------- Provider


class ModelProvider(abc.ABC):
    """Ein Inference-Backend (z. B. eine laufende lokale Runtime).

    Ein Provider kann mehrere Modelle bedienen; welches Modell gemeint ist, steht in
    den übergebenen :class:`ModelMetadata`.
    """

    def __init__(self, name: str) -> None:
        if not name.strip():
            raise ValueError("Provider-Name darf nicht leer sein")
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @abc.abstractmethod
    async def chat(self, model: ModelMetadata, request: ChatRequest) -> ChatResponse:
        """Führt eine Chat-Completion aus."""

    @abc.abstractmethod
    async def list_models(self) -> list[str]:
        """Namen der Modelle, die die Runtime aktuell bereitstellen kann."""

    @abc.abstractmethod
    async def health(self) -> ProviderHealth:
        """Prüft Erreichbarkeit und Bereitschaft der Runtime."""

    async def ensure_loaded(self, model: ModelMetadata) -> None:
        """Stellt sicher, dass das Modell geladen ist.

        Standard: minimale Generierung (1 Token). Runtimes mit expliziter Lade-API
        überschreiben diese Methode.
        """
        await self.chat(
            model,
            ChatRequest(
                messages=(Message.user("ping"),),
                params=GenerationParams(temperature=0.0, max_tokens=1),
            ),
        )

    async def stream(
        self, model: ModelMetadata, request: ChatRequest
    ) -> AsyncIterator[StreamChunk]:
        """Gestreamte Antwort. Standard: kein echtes Streaming – eine Antwort als ein Chunk."""
        response = await self.chat(model, request)
        yield StreamChunk(
            delta=response.message.content,
            finish_reason=response.finish_reason,
            usage=response.usage,
            runtime_stats=response.runtime_stats,
            model=response.model,
            streamed=False,
        )

    async def runtime_info(self, model: ModelMetadata | None = None) -> dict[str, Any]:
        """Von der Runtime gemeldete Fakten (Typ, Version, Kontext, Modellgröße, Modalitäten).

        Nur, was die Runtime tatsächlich liefert – fehlende Angaben bleiben weg.
        """
        return {}

    async def unload(self, model: ModelMetadata) -> None:
        """Entlädt das Modell aus dem Speicher (nur Runtimes mit Lade-API, z. B. Ollama)."""
        raise NotImplementedError(f"{self.name}: Runtime unterstützt kein Entladen")

    async def load(self, model: ModelMetadata) -> dict[str, float]:
        """Lädt das Modell explizit; liefert von der Runtime gemeldete Zeiten (Sekunden).

        Nur für Runtimes mit Lade-API. Die Wanduhrzeit misst der Aufrufer selbst.
        """
        raise NotImplementedError(f"{self.name}: Runtime hat keine Lade-API")

    async def aclose(self) -> None:  # noqa: B027 – optionaler Hook
        """Gibt Ressourcen (Verbindungen) frei."""
