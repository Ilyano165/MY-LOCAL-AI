"""Gemessene Modelldaten – neutrale Schnittstelle zwischen Registry und Benchmarking.

``models`` darf ``evaluation`` nicht importieren. Deshalb definiert dieses Modul nur den
Vertrag: eine :class:`MeasurementSource` liefert pro Modell :class:`MeasuredOverrides`
(z. B. :class:`evaluation.benchmark_results.ProfileStore`), der Registry wendet sie an.

Grundsatz: Konfigurierte Werte sind **Annahmen**. Nur Werte aus echten Benchmark-Läufen
dürfen als gemessen gelten. Ohne Profil ist ein Modell ausdrücklich ``UNMEASURED``.

Zwei Gruppen werden getrennt behandelt:

* **Modellfähigkeit** (Reasoning, Coding, Vision, Tool-Calling) – hardwareunabhängig.
* **Hardware-Leistung** (Geschwindigkeit, Speicherbedarf, real konfigurierter Kontext) –
  gilt nur für die Hardware/Runtime, auf der gemessen wurde.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, runtime_checkable

from models.capabilities import CapabilityLevel, ModelMetadata, Speed


class DataStatus(StrEnum):
    MEASURED = "MEASURED"
    """Aktuelles Profil, Fähigkeit und Hardware-Leistung auf dieser Hardware gemessen."""
    PARTIAL = "PARTIAL"
    """Nur ein Teil gemessen (z. B. Fähigkeiten gültig, Leistung auf anderer Hardware)."""
    STALE = "STALE"
    """Profil vorhanden, aber mit anderer Benchmark-Version – nicht angewendet."""
    UNMEASURED = "UNMEASURED"
    """Kein Benchmark: alle Werte stammen aus der Konfiguration (Annahmen)."""


CAPABILITY_FIELDS = (
    "reasoning_capability",
    "coding_capability",
    "vision_capability",
    "tool_calling",
)
PERFORMANCE_FIELDS = ("speed", "memory_requirement", "context_length")


@dataclass(frozen=True)
class MeasuredOverrides:
    """Aus einem Benchmark-Profil abgeleitete Werte; ``None`` = nicht gemessen."""

    status: DataStatus
    source: str = ""
    """Herkunft, z. B. ``"benchmark 1.0+ab12cd34 vom 2026-10-08"``."""
    reasoning_capability: CapabilityLevel | None = None
    coding_capability: CapabilityLevel | None = None
    vision_capability: CapabilityLevel | None = None
    tool_calling: bool | None = None
    speed: Speed | None = None
    memory_requirement: float | None = None
    context_length: int | None = None
    tokens_per_second: float | None = None
    notes: tuple[str, ...] = ()
    field_sources: Mapping[str, str] = field(default_factory=dict, hash=False)
    """Feld → Messmethode/Status (für Logs und Nachvollziehbarkeit)."""

    @classmethod
    def unmeasured(cls, note: str = "") -> MeasuredOverrides:
        return cls(DataStatus.UNMEASURED, notes=(note,) if note else ())

    def applied_fields(self) -> list[str]:
        return [
            name
            for name in (*CAPABILITY_FIELDS, *PERFORMANCE_FIELDS)
            if getattr(self, name) is not None
        ]

    def describe(self) -> str:
        """Kurzbeschreibung für Routing-Logs."""
        if self.status is DataStatus.UNMEASURED:
            text = "UNMEASURED – Fähigkeiten/Leistung aus Konfiguration (nicht gemessen)"
        elif self.status is DataStatus.STALE:
            text = f"UNMEASURED – Profil veraltet ({self.source}), Konfigurationswerte genutzt"
        else:
            measured = ", ".join(self.applied_fields()) or "keine Felder"
            text = f"{self.status.value} ({self.source}; gemessen: {measured})"
        if self.notes:
            text += " – " + "; ".join(self.notes)
        return text


@runtime_checkable
class MeasurementSource(Protocol):
    def overrides_for(self, model: ModelMetadata) -> MeasuredOverrides | None:
        """Gemessene Werte für ``model`` oder ``None``, wenn kein Profil existiert."""
        ...


def apply_overrides(model: ModelMetadata, overrides: MeasuredOverrides) -> ModelMetadata:
    """Effektive Metadaten: gemessene Werte ersetzen die konfigurierten.

    Die konfigurierten Werte bleiben in ``extra["configured"]`` erhalten, der Datenstatus
    steht in ``extra["data_status"]`` – so bleibt jede Entscheidung nachvollziehbar.
    """
    extra = dict(model.extra)
    extra["data_status"] = overrides.status.value
    if overrides.status in (DataStatus.UNMEASURED, DataStatus.STALE):
        return _replace(model, extra=extra)
    changes: dict[str, object] = {}
    configured: dict[str, object] = {}
    for name in overrides.applied_fields():
        configured[name] = getattr(model, name)
        changes[name] = getattr(overrides, name)
    extra["configured"] = configured
    extra["measured_source"] = overrides.source
    if overrides.tokens_per_second is not None:
        extra["measured_tokens_per_second"] = overrides.tokens_per_second
    return _replace(model, extra=extra, **changes)


def _replace(model: ModelMetadata, **changes: object) -> ModelMetadata:
    data = {name: getattr(model, name) for name in model.__dataclass_fields__}
    data.update(changes)
    return ModelMetadata(**data)
