"""Modellverfügbarkeit: Der Router darf nur Modelle wählen, die *jetzt* bedienbar sind.

:class:`ProviderAvailability` fragt die Runtime (Health + Modellliste) und cached das Ergebnis
kurz (TTL), damit nicht jede Anfrage zusätzliche HTTP-Aufrufe auslöst. Fehlgeschlagene Aufrufe
können über :meth:`mark_unavailable` sofort gemeldet werden (z. B. nach einem Timeout).
"""

from __future__ import annotations

import abc
import asyncio
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from models.base import ModelError
from models.capabilities import ModelMetadata
from models.inference import ProviderRegistry


@dataclass(frozen=True)
class Availability:
    available: bool
    reason: str


class ModelAvailability(abc.ABC):
    @abc.abstractmethod
    async def check(self, model: ModelMetadata) -> Availability: ...

    async def check_all(self, models: Iterable[ModelMetadata]) -> dict[str, Availability]:
        items = list(models)
        results = await asyncio.gather(*(self.check(m) for m in items))
        return {m.name: r for m, r in zip(items, results, strict=True)}

    def mark_unavailable(self, model_name: str, reason: str) -> None:  # noqa: B027 – optional
        """Meldet einen akuten Ausfall (Default: ignoriert)."""


class StaticAvailability(ModelAvailability):
    """Feste Verfügbarkeit (Konfiguration, Tests)."""

    def __init__(self, available: Iterable[str], reasons: dict[str, str] | None = None) -> None:
        self.available = set(available)
        self.reasons = reasons or {}

    async def check(self, model: ModelMetadata) -> Availability:
        if model.name in self.available:
            return Availability(True, "verfügbar")
        return Availability(False, self.reasons.get(model.name, "nicht verfügbar"))

    def mark_unavailable(self, model_name: str, reason: str) -> None:
        self.available.discard(model_name)
        self.reasons[model_name] = reason


class ProviderAvailability(ModelAvailability):
    def __init__(
        self,
        providers: ProviderRegistry,
        *,
        ttl_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.providers = providers
        self.ttl_s = ttl_s
        self.clock = clock
        self._cache: dict[str, tuple[float, Availability]] = {}
        self._served: dict[str, tuple[float, list[str] | None, str]] = {}

    def mark_unavailable(self, model_name: str, reason: str) -> None:
        self._cache[model_name] = (self.clock(), Availability(False, reason))

    def invalidate(self) -> None:
        self._cache.clear()
        self._served.clear()

    async def _provider_state(self, provider_name: str) -> tuple[list[str] | None, str]:
        """(gelistete Modelle oder None = Listing nicht unterstützt, Fehlergrund oder '')."""
        cached = self._served.get(provider_name)
        if cached and self.clock() - cached[0] < self.ttl_s:
            return cached[1], cached[2]
        try:
            provider = self.providers.get(provider_name)
        except ModelError as exc:
            state: tuple[list[str] | None, str] = (None, str(exc))
        else:
            try:
                health = await provider.health()
                if not health.ready:
                    state = (None, f"Runtime nicht bereit: {health.detail}")
                else:
                    try:
                        state = (await provider.list_models(), "")
                    except NotImplementedError:
                        state = (None, "")
            except ModelError as exc:
                state = (None, f"Runtime-Fehler: {exc}")
        self._served[provider_name] = (self.clock(), state[0], state[1])
        return state

    async def check(self, model: ModelMetadata) -> Availability:
        cached = self._cache.get(model.name)
        if cached and self.clock() - cached[0] < self.ttl_s:
            return cached[1]
        served, error = await self._provider_state(model.provider)
        if error:
            result = Availability(False, error)
        elif served is not None:
            wanted = {model.runtime_name}
            if model.local_path is not None:
                wanted |= {str(model.local_path), Path(model.local_path).name}
            ok = any(s in wanted or Path(s).name in wanted for s in served)
            result = Availability(
                ok,
                "von Runtime gelistet"
                if ok
                else f"Runtime {model.provider!r} listet {model.runtime_name!r} nicht",
            )
        elif model.local_path is not None and not await asyncio.to_thread(
            Path(model.local_path).exists
        ):
            result = Availability(False, f"Modelldatei fehlt: {model.local_path}")
        else:
            result = Availability(True, "Runtime bereit (Modellliste nicht verfügbar)")
        self._cache[model.name] = (self.clock(), result)
        return result
