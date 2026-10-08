"""Einheitlicher Einstiegspunkt für Inference.

``InferenceEngine`` verbindet ``ModelRegistry`` (was gibt es?) mit ``ProviderRegistry``
(wer führt es aus?). Aufrufer geben eine Aufgabe oder – falls bewusst gewünscht – einen
Registry-Namen an; konkrete Modellnamen stehen nur in der Konfiguration.
"""

from __future__ import annotations

import asyncio
import logging
import tomllib
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from models.base import (
    ChatRequest,
    ChatResponse,
    ModelError,
    ModelNotFoundError,
    ModelProvider,
    ModelTimeoutError,
    NoSuitableModelError,
    ProviderNotRegisteredError,
    ProviderUnavailableError,
)
from models.capabilities import ModelMetadata, TaskRequirements, TaskType
from models.local_provider import OpenAICompatibleProvider
from models.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

#: Fehler, bei denen ein Fallback auf das nächstbeste Modell sinnvoll ist.
RETRYABLE_ERRORS: tuple[type[ModelError], ...] = (
    ProviderUnavailableError,
    ModelTimeoutError,
    ModelNotFoundError,
)


class ProviderRegistry:
    """Benannte Provider-Instanzen (z. B. ``"local"`` → laufende llama.cpp-Instanz)."""

    def __init__(self, providers: Iterable[ModelProvider] = ()) -> None:
        self._providers: dict[str, ModelProvider] = {}
        for provider in providers:
            self.register(provider)

    def register(self, provider: ModelProvider, *, replace: bool = False) -> None:
        if provider.name in self._providers and not replace:
            raise ValueError(f"Provider {provider.name!r} ist bereits registriert")
        self._providers[provider.name] = provider

    def get(self, name: str) -> ModelProvider:
        try:
            return self._providers[name]
        except KeyError:
            raise ProviderNotRegisteredError(f"Provider {name!r} ist nicht registriert") from None

    def names(self) -> list[str]:
        return sorted(self._providers)

    async def aclose(self) -> None:
        for provider in self._providers.values():
            await provider.aclose()


ProviderFactory = Callable[[Mapping[str, Any]], ModelProvider]


def _openai_compatible(cfg: Mapping[str, Any]) -> ModelProvider:
    options = {k: v for k, v in cfg.items() if k not in ("name", "type")}
    return OpenAICompatibleProvider(str(cfg["name"]), **options)


#: Erweiterungspunkt: neue Backend-Typen hier eintragen (oder per ``register_provider_type``).
PROVIDER_TYPES: dict[str, ProviderFactory] = {
    "openai_compatible": _openai_compatible,
}


def register_provider_type(type_name: str, factory: ProviderFactory) -> None:
    if type_name in PROVIDER_TYPES:
        raise ValueError(f"Provider-Typ {type_name!r} existiert bereits")
    PROVIDER_TYPES[type_name] = factory


def create_provider(cfg: Mapping[str, Any]) -> ModelProvider:
    if "name" not in cfg or "type" not in cfg:
        raise ValueError("Provider-Konfiguration braucht 'name' und 'type'")
    try:
        factory = PROVIDER_TYPES[str(cfg["type"])]
    except KeyError:
        known = ", ".join(sorted(PROVIDER_TYPES))
        raise ValueError(f"Unbekannter Provider-Typ {cfg['type']!r} (bekannt: {known})") from None
    return factory(cfg)


@dataclass(frozen=True, slots=True)
class InferenceResult:
    response: ChatResponse
    model: ModelMetadata
    attempts: tuple[str, ...] = field(default=())  # versuchte Modelle in Reihenfolge
    routing_id: str | None = None
    """ID der Routing-Entscheidung (wenn ein Router gewählt hat) – für Ergebnis-Rückmeldung."""


@dataclass(frozen=True, slots=True)
class RoutedChoice:
    model: ModelMetadata
    fallbacks: list[ModelMetadata]
    decision_id: str


class Router(Protocol):
    """Schnittstelle, über die ein Model Router (siehe ``router/``) die Auswahl übernimmt.

    Als Protocol hier definiert, damit ``models`` nicht von ``router`` abhängt."""

    async def route_chat(
        self, request: ChatRequest, requirements: TaskRequirements | None
    ) -> RoutedChoice: ...

    def report_failure(self, model_name: str, reason: str) -> None: ...

    def record_outcome(
        self,
        decision_id: str,
        *,
        success: bool,
        verdict: str | None = None,
        quality: float | None = None,
    ) -> None: ...


class InferenceEngine:
    def __init__(
        self,
        models: ModelRegistry,
        providers: ProviderRegistry,
        *,
        default_timeout_s: float = 120.0,
        router: Router | None = None,
    ) -> None:
        if default_timeout_s <= 0:
            raise ValueError("default_timeout_s muss > 0 sein")
        self.models = models
        self.providers = providers
        self.default_timeout_s = default_timeout_s
        self.router = router

    def validate(self) -> list[str]:
        """Konsistenzprüfung: verweist jedes Modell auf einen registrierten Provider?"""
        known = set(self.providers.names())
        return [
            f"{m.name}: Provider {m.provider!r} nicht registriert"
            for m in self.models.list()
            if m.provider not in known
        ]

    def resolve(
        self,
        *,
        model: str | None = None,
        task: TaskRequirements | TaskType | str | None = None,
        exclude: Iterable[str] = (),
    ) -> ModelMetadata:
        if model is not None and task is not None:
            raise ValueError("Entweder model oder task angeben, nicht beides")
        if model is not None:
            return self.models.get(model)
        return self.models.find_best_for(task or TaskType.GENERAL, exclude=exclude)

    async def run(self, model: ModelMetadata, request: ChatRequest) -> ChatResponse:
        """Führt eine Anfrage auf genau diesem Modell aus – mit hartem Timeout."""
        provider = self.providers.get(model.provider)
        timeout = request.timeout_s or self.default_timeout_s
        try:
            return await asyncio.wait_for(provider.chat(model, request), timeout=timeout)
        except TimeoutError as exc:
            raise ModelTimeoutError(
                f"{model.name}: keine Antwort innerhalb von {timeout:.1f}s"
            ) from exc

    async def chat(
        self,
        request: ChatRequest,
        *,
        model: str | None = None,
        task: TaskRequirements | TaskType | str | None = None,
        fallback: bool = False,
    ) -> InferenceResult:
        """Wählt ein Modell (explizit oder per Aufgabe) und führt die Anfrage aus.

        Mit ``fallback=True`` (nur bei Auswahl per Aufgabe) wird bei
        Nichterreichbarkeit/Timeout das nächstbeste geeignete Modell versucht.
        """
        if model is None and self.router is not None:
            return await self._chat_routed(request, task, fallback)
        chosen = self.resolve(model=model, task=task)
        if not fallback or model is not None:
            response = await self.run(chosen, request)
            return InferenceResult(response, chosen, (chosen.name,))

        attempts: list[str] = []
        errors: list[str] = []
        while True:
            attempts.append(chosen.name)
            try:
                response = await self.run(chosen, request)
                return InferenceResult(response, chosen, tuple(attempts))
            except RETRYABLE_ERRORS as exc:
                logger.warning("Modell %s fehlgeschlagen, versuche Fallback: %s", chosen.name, exc)
                errors.append(f"{chosen.name}: {exc}")
            try:
                chosen = self.resolve(task=task, exclude=attempts)
            except NoSuitableModelError:
                raise ProviderUnavailableError(
                    "Alle geeigneten Modelle fehlgeschlagen: " + " | ".join(errors)
                ) from None

    async def _chat_routed(
        self, request: ChatRequest, task: TaskRequirements | TaskType | str | None, fallback: bool
    ) -> InferenceResult:
        """Auswahl durch den Router; Fallbacks sind ebenfalls vom Router geprüft (verfügbar,
        passend). Ausfälle werden dem Router gemeldet, damit er das Modell vorerst meidet."""
        assert self.router is not None
        requirements = TaskRequirements.coerce(task) if task is not None else None
        choice = await self.router.route_chat(request, requirements)
        candidates = [choice.model, *choice.fallbacks] if fallback else [choice.model]
        attempts: list[str] = []
        errors: list[str] = []
        for candidate in candidates:
            attempts.append(candidate.name)
            try:
                response = await self.run(candidate, request)
                return InferenceResult(response, candidate, tuple(attempts), choice.decision_id)
            except RETRYABLE_ERRORS as exc:
                logger.warning("Modell %s fehlgeschlagen: %s", candidate.name, exc)
                errors.append(f"{candidate.name}: {exc}")
                self.router.report_failure(candidate.name, f"{type(exc).__name__}: {exc}")
                if not fallback:
                    raise
        raise ProviderUnavailableError(
            "Alle vom Router gewählten Modelle fehlgeschlagen: " + " | ".join(errors)
        )

    async def aclose(self) -> None:
        await self.providers.aclose()

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> InferenceEngine:
        provider_cfgs = config.get("providers", [])
        if not isinstance(provider_cfgs, list):
            raise ValueError("'providers' muss eine Liste von Tabellen sein ([[providers]])")
        engine_cfg = config.get("inference", {})
        engine = cls(
            ModelRegistry.from_mapping(config),
            ProviderRegistry(create_provider(c) for c in provider_cfgs),
            default_timeout_s=float(engine_cfg.get("default_timeout_s", 120.0)),
        )
        problems = engine.validate()
        if problems:
            raise ValueError("Ungültige Modellkonfiguration: " + "; ".join(problems))
        return engine

    @classmethod
    def from_toml(cls, path: str | Path) -> InferenceEngine:
        with Path(path).open("rb") as fh:
            return cls.from_mapping(tomllib.load(fh))
