"""Zentrales Verzeichnis aller bekannten Modelle.

Der Registry ist die einzige Stelle, die Modellnamen kennt – und er bezieht sie
ausschließlich aus Konfiguration oder expliziten ``register``-Aufrufen. Anwendungscode
fragt nach einer *Aufgabe* (``find_best_for``), nicht nach einem Modellnamen.
"""

from __future__ import annotations

import builtins
import threading
import tomllib
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from models.base import ModelNotFoundError, NoSuitableModelError
from models.capabilities import ModelMetadata, TaskRequirements, TaskType
from models.measured import MeasuredOverrides, MeasurementSource, apply_overrides


class ModelRegistry:
    """Thread-sicherer In-Memory-Registry für :class:`ModelMetadata`.

    Bewusst instanzbasiert (kein globaler Zustand): Tests und unterschiedliche
    Konfigurationen bekommen jeweils einen eigenen Registry.
    """

    def __init__(
        self,
        models: Iterable[ModelMetadata] = (),
        *,
        measurements: MeasurementSource | None = None,
    ) -> None:
        self._models: dict[str, ModelMetadata] = {}
        self._lock = threading.RLock()
        self.measurements = measurements
        for model in models:
            self.register(model)

    # ------------------------------------------------------------------ Messdaten

    def set_measurements(self, source: MeasurementSource | None) -> None:
        """Quelle gemessener Profile (z. B. ``ProfileStore``); ``None`` = nur Konfiguration."""
        self.measurements = source

    def data_status(self, name: str) -> MeasuredOverrides:
        """Messstatus eines Modells; ohne Profil ausdrücklich ``UNMEASURED``."""
        model = self.get(name)
        if self.measurements is None:
            return MeasuredOverrides.unmeasured("keine Benchmark-Quelle konfiguriert")
        overrides = self.measurements.overrides_for(model)
        return overrides or MeasuredOverrides.unmeasured("kein Benchmark-Profil vorhanden")

    def effective(self, name: str) -> ModelMetadata:
        """Metadaten mit gemessenen Werten anstelle der konfigurierten (falls vorhanden)."""
        return apply_overrides(self.get(name), self.data_status(name))

    def list_effective(self, *, provider: str | None = None) -> builtins.list[ModelMetadata]:
        return [self.effective(m.name) for m in self.list(provider=provider)]

    # ------------------------------------------------------------------ CRUD

    def register(self, model: ModelMetadata, *, replace: bool = False) -> ModelMetadata:
        if not isinstance(model, ModelMetadata):
            raise TypeError("register erwartet ModelMetadata")
        with self._lock:
            if model.name in self._models and not replace:
                raise ValueError(f"Modell {model.name!r} ist bereits registriert")
            self._models[model.name] = model
        return model

    def unregister(self, name: str) -> ModelMetadata:
        with self._lock:
            try:
                return self._models.pop(name)
            except KeyError:
                raise ModelNotFoundError(f"Modell {name!r} ist nicht registriert") from None

    def get(self, name: str) -> ModelMetadata:
        with self._lock:
            try:
                return self._models[name]
            except KeyError:
                known = ", ".join(sorted(self._models)) or "—"
                raise ModelNotFoundError(
                    f"Modell {name!r} ist nicht registriert (bekannt: {known})"
                ) from None

    def list(self, *, provider: str | None = None) -> list[ModelMetadata]:
        with self._lock:
            models = sorted(self._models.values(), key=lambda m: m.name)
        return [m for m in models if provider is None or m.provider == provider]

    def __contains__(self, name: object) -> bool:
        with self._lock:
            return name in self._models

    def __len__(self) -> int:
        with self._lock:
            return len(self._models)

    def __iter__(self) -> Iterator[ModelMetadata]:
        return iter(self.list())

    # ------------------------------------------------------------------ Auswahl

    def rank_for(
        self,
        task: TaskRequirements | TaskType | str,
        *,
        exclude: Iterable[str] = (),
    ) -> builtins.list[ModelMetadata]:
        """Alle geeigneten Modelle, bestes zuerst (deterministisch)."""
        req = TaskRequirements.coerce(task)
        excluded = set(exclude)
        candidates = [
            m
            for m in self.list_effective()
            if m.name not in excluded and not m.unmet_requirements(req)
        ]
        # Stabil sortieren: erst Name aufsteigend, dann Ranking absteigend → Gleichstand
        # wird alphabetisch und damit reproduzierbar aufgelöst.
        candidates.sort(key=lambda m: m.name)
        candidates.sort(key=lambda m: m.ranking_key(req.task_type), reverse=True)
        return candidates

    def find_best_for(
        self,
        task: TaskRequirements | TaskType | str,
        *,
        exclude: Iterable[str] = (),
    ) -> ModelMetadata:
        req = TaskRequirements.coerce(task)
        excluded = set(exclude)
        ranked = self.rank_for(req, exclude=excluded)
        if ranked:
            return ranked[0]
        reasons = {
            m.name: m.unmet_requirements(req)
            for m in self.list_effective()
            if m.name not in excluded
        }
        detail = "; ".join(f"{n}: {', '.join(r)}" for n, r in reasons.items()) or "Registry leer"
        raise NoSuitableModelError(f"Kein Modell für {req.task_type.value} geeignet ({detail})")

    # ------------------------------------------------------------------ Konfiguration

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> ModelRegistry:
        entries = config.get("models", [])
        if not isinstance(entries, list):
            raise ValueError("'models' muss eine Liste von Tabellen sein ([[models]])")
        return cls(ModelMetadata.from_dict(entry) for entry in entries)

    @classmethod
    def from_toml(cls, path: str | Path) -> ModelRegistry:
        with Path(path).open("rb") as fh:
            return cls.from_mapping(tomllib.load(fh))
