"""Ergebnisse des Model-Benchmarkings: Messwerte mit Herkunft, Profile, Speicherung.

Jeder Wert ist eine :class:`Measurement` mit **Status** und **Methode**:

==================  ======================================================================
``MEASURED``        von NOVA in einem echten Lauf gemessen (Wanduhr, Sampling, Scoring)
``REPORTED``        von der Runtime in einem echten Lauf gemeldet (z. B. llama.cpp timings)
``DETECTED``        per Systemabfrage erkannt (Hardware, Runtime-Konfiguration)
``CONFIGURED``      aus der Konfiguration übernommen – **nie** ein Messwert
``NOT_SUPPORTED``   Modell/Runtime unterstützt die Fähigkeit nachweislich nicht
``NOT_MEASURABLE``  in dieser Konstellation nicht messbar (z. B. Ladezeit bei vorgeladenem
                    Modell) – mit Begründung
``FAILED``          Messung versucht, aber fehlgeschlagen
``UNMEASURED``      nie gemessen
==================  ======================================================================

Modellfähigkeit (``capabilities``) und Hardware-Leistung (``performance``) sind getrennte
Abschnitte. Ein langsames Modell ist nicht automatisch ein schlechtes Modell.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from models.capabilities import CapabilityLevel, ModelMetadata, Speed
from models.measured import DataStatus, MeasuredOverrides

PROFILE_SCHEMA = 1


class MeasurementStatus(StrEnum):
    MEASURED = "MEASURED"
    REPORTED = "REPORTED"
    DETECTED = "DETECTED"
    CONFIGURED = "CONFIGURED"
    NOT_SUPPORTED = "NOT_SUPPORTED"
    NOT_MEASURABLE = "NOT_MEASURABLE"
    FAILED = "FAILED"
    UNMEASURED = "UNMEASURED"

    @property
    def is_fact(self) -> bool:
        """Stammt der Wert aus einem echten Lauf bzw. einer echten Systemabfrage?"""
        return self in (
            MeasurementStatus.MEASURED,
            MeasurementStatus.REPORTED,
            MeasurementStatus.DETECTED,
        )


Value = float | int | str | bool | None


@dataclass(frozen=True)
class Measurement:
    value: Value
    status: MeasurementStatus
    unit: str = ""
    method: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        if self.status.is_fact and self.value is None:
            raise ValueError(f"{self.status.value} ohne Wert ist unzulässig ({self.method})")
        if self.status is MeasurementStatus.MEASURED and not self.method:
            raise ValueError("MEASURED erfordert eine Messmethode")

    @classmethod
    def measured(cls, value: Value, unit: str, method: str, note: str = "") -> Measurement:
        return cls(value, MeasurementStatus.MEASURED, unit, method, note)

    @classmethod
    def reported(cls, value: Value, unit: str, method: str, note: str = "") -> Measurement:
        return cls(value, MeasurementStatus.REPORTED, unit, method, note)

    @classmethod
    def detected(
        cls, value: Value, unit: str = "", method: str = "", note: str = ""
    ) -> Measurement:
        return cls(value, MeasurementStatus.DETECTED, unit, method, note)

    @classmethod
    def configured(cls, value: Value, unit: str = "", note: str = "") -> Measurement:
        return cls(value, MeasurementStatus.CONFIGURED, unit, "Konfiguration", note)

    @classmethod
    def missing(cls, status: MeasurementStatus, note: str, unit: str = "") -> Measurement:
        if status.is_fact:
            raise ValueError("missing() nur für Status ohne Messwert")
        return cls(None, status, unit, "", note)

    @property
    def is_fact(self) -> bool:
        return self.status.is_fact

    def number(self) -> float | None:
        """Zahlenwert nur, wenn er aus einem echten Lauf/einer echten Abfrage stammt."""
        if not self.is_fact or isinstance(self.value, bool):
            return None
        return float(self.value) if isinstance(self.value, int | float) else None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"value": self.value, "status": self.status.value}
        for key in ("unit", "method", "note"):
            if getattr(self, key):
                out[key] = getattr(self, key)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Measurement:
        return cls(
            value=data.get("value"),
            status=MeasurementStatus(data["status"]),
            unit=str(data.get("unit", "")),
            method=str(data.get("method", "")),
            note=str(data.get("note", "")),
        )

    def __str__(self) -> str:
        if self.value is None:
            return f"{self.status.value}" + (f" ({self.note})" if self.note else "")
        value = f"{self.value:.2f}" if isinstance(self.value, float) else str(self.value)
        unit = f" {self.unit}" if self.unit else ""
        return f"{value}{unit} [{self.status.value}]"


class TaskStatus(StrEnum):
    COMPLETED = "completed"
    """Lauf durchgeführt und bewertet (Score kann auch 0 sein)."""
    SKIPPED = "skipped"
    """Nicht anwendbar (z. B. Kontext zu klein, Tool-Calling nicht unterstützt)."""
    UNSUPPORTED = "unsupported"
    """Modell/Runtime lehnt die Fähigkeit nachweislich ab (z. B. Bild-Eingaben)."""
    ERROR = "error"
    """Lauf abgebrochen (Runtime-Fehler, Timeout) – kein Fähigkeitsurteil möglich."""


@dataclass
class TaskResult:
    task_id: str
    category: str
    status: TaskStatus
    score: float | None
    """Anteil erfüllter, unabhängig geprüfter Kriterien in [0, 1]; ``None`` ohne Bewertung."""
    checks: dict[str, bool] = field(default_factory=dict)
    detail: str = ""
    duration_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    runtime_stats: dict[str, float] = field(default_factory=dict)
    output_excerpt: str = ""
    repetitions: int = 1

    def __post_init__(self) -> None:
        if self.score is not None and not 0.0 <= self.score <= 1.0:
            raise ValueError("score muss in [0, 1] liegen")
        if self.status is TaskStatus.COMPLETED and self.score is None:
            raise ValueError("abgeschlossene Aufgabe braucht einen Score")

    @property
    def passed(self) -> bool:
        return self.status is TaskStatus.COMPLETED and self.score == 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "category": self.category,
            "status": self.status.value,
            "score": self.score,
            "checks": self.checks,
            "detail": self.detail,
            "duration_s": round(self.duration_s, 3),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "runtime_stats": self.runtime_stats,
            "output_excerpt": self.output_excerpt,
            "repetitions": self.repetitions,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TaskResult:
        return cls(
            task_id=str(data["task_id"]),
            category=str(data["category"]),
            status=TaskStatus(data["status"]),
            score=data.get("score"),
            checks=dict(data.get("checks", {})),
            detail=str(data.get("detail", "")),
            duration_s=float(data.get("duration_s", 0.0)),
            prompt_tokens=int(data.get("prompt_tokens", 0)),
            completion_tokens=int(data.get("completion_tokens", 0)),
            runtime_stats=dict(data.get("runtime_stats", {})),
            output_excerpt=str(data.get("output_excerpt", "")),
            repetitions=int(data.get("repetitions", 1)),
        )


# ---------------------------------------------------------------------- Abbildung auf Stufen

LEVEL_THRESHOLDS = (
    (0.85, CapabilityLevel.STRONG),
    (0.6, CapabilityLevel.GOOD),
    (0.3, CapabilityLevel.BASIC),
)
SPEED_THRESHOLDS = ((40.0, Speed.FAST), (15.0, Speed.MEDIUM))
"""Dekodier-Tokens/s → Geschwindigkeitsstufe (interaktiv flüssig ≈ ab 40 tok/s)."""


def score_to_level(score: float) -> CapabilityLevel:
    for threshold, level in LEVEL_THRESHOLDS:
        if score >= threshold:
            return level
    return CapabilityLevel.NONE


def tokens_per_second_to_speed(tps: float) -> Speed:
    for threshold, speed in SPEED_THRESHOLDS:
        if tps >= threshold:
            return speed
    return Speed.SLOW


# ---------------------------------------------------------------------- Profil


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class ModelProfile:
    """Gemessenes Profil eines Modells auf einer konkreten Hardware/Runtime."""

    model: str
    benchmark_version: str
    runtime: dict[str, Any]
    hardware: dict[str, Any]
    hardware_fingerprint: str
    model_info: dict[str, Measurement] = field(default_factory=dict)
    """Größe, Quantisierung, Kontextlänge, Parameter – mit Herkunft."""
    capabilities: dict[str, Measurement] = field(default_factory=dict)
    """Modellfähigkeit (hardwareunabhängig): Scores/Stufen aus bewerteten Aufgaben."""
    performance: dict[str, Measurement] = field(default_factory=dict)
    """Hardware-Leistung: load_time, tokens_per_second, peak_vram, peak_ram, …"""
    tasks: list[TaskResult] = field(default_factory=list)
    created_at: str = field(default_factory=_now)
    configured: dict[str, Any] = field(default_factory=dict)
    """Konfigurationswerte zum Vergleich – ausdrücklich keine Messwerte."""
    notes: list[str] = field(default_factory=list)

    def perf(self, key: str) -> Measurement:
        return self.performance.get(key) or Measurement.missing(
            MeasurementStatus.UNMEASURED, "nicht gemessen"
        )

    def capability(self, key: str) -> Measurement:
        return self.capabilities.get(key) or Measurement.missing(
            MeasurementStatus.UNMEASURED, "nicht gemessen"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROFILE_SCHEMA,
            "model": self.model,
            "benchmark_version": self.benchmark_version,
            "created_at": self.created_at,
            "runtime": self.runtime,
            "hardware": self.hardware,
            "hardware_fingerprint": self.hardware_fingerprint,
            "load_time": self.perf("load_time").to_dict(),
            "tokens_per_second": self.perf("tokens_per_second").to_dict(),
            "peak_vram": self.perf("peak_vram").to_dict(),
            "peak_ram": self.perf("peak_ram").to_dict(),
            "capabilities": {k: v.to_dict() for k, v in self.capabilities.items()},
            "performance": {k: v.to_dict() for k, v in self.performance.items()},
            "model_info": {k: v.to_dict() for k, v in self.model_info.items()},
            "tasks": [t.to_dict() for t in self.tasks],
            "configured": self.configured,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ModelProfile:
        if data.get("schema") != PROFILE_SCHEMA:
            raise ValueError(f"Unbekanntes Profilschema: {data.get('schema')!r}")

        def section(name: str) -> dict[str, Measurement]:
            return {k: Measurement.from_dict(v) for k, v in dict(data.get(name, {})).items()}

        return cls(
            model=str(data["model"]),
            benchmark_version=str(data["benchmark_version"]),
            runtime=dict(data.get("runtime", {})),
            hardware=dict(data.get("hardware", {})),
            hardware_fingerprint=str(data.get("hardware_fingerprint", "")),
            model_info=section("model_info"),
            capabilities=section("capabilities"),
            performance=section("performance"),
            tasks=[TaskResult.from_dict(t) for t in data.get("tasks", [])],
            created_at=str(data.get("created_at", "")),
            configured=dict(data.get("configured", {})),
            notes=list(data.get("notes", [])),
        )

    # ------------------------------------------------------------------ → Registry

    def to_overrides(
        self, *, hardware_fingerprint: str | None, benchmark_version: str | None
    ) -> MeasuredOverrides:
        """Leitet die für den Router nutzbaren Werte ab.

        * andere Benchmark-Version → ``STALE`` (nichts angewendet)
        * andere Hardware → nur Fähigkeiten (hardwareunabhängig), keine Leistungswerte
        * nur Werte mit Status MEASURED/REPORTED/DETECTED werden übernommen
        """
        source = f"benchmark {self.benchmark_version} vom {self.created_at[:10]}"
        if benchmark_version is not None and benchmark_version != self.benchmark_version:
            return MeasuredOverrides(
                DataStatus.STALE,
                source=source,
                notes=(f"aktuelle Benchmark-Version {benchmark_version}",),
            )
        values: dict[str, Any] = {}
        sources: dict[str, str] = {}

        def level(key: str, target: str) -> None:
            m = self.capabilities.get(key)
            if m is not None and m.status is MeasurementStatus.NOT_SUPPORTED:
                values[target] = CapabilityLevel.NONE
                sources[target] = f"NOT_SUPPORTED ({m.note})"
            elif m is not None and (score := m.number()) is not None:
                values[target] = score_to_level(score)
                sources[target] = f"{m.status.value}: Score {score:.2f} ({m.method})"

        level("reasoning", "reasoning_capability")
        level("coding", "coding_capability")
        level("vision", "vision_capability")
        tools = self.capabilities.get("tool_calling")
        if tools is not None and tools.status is MeasurementStatus.NOT_SUPPORTED:
            values["tool_calling"] = False
            sources["tool_calling"] = f"NOT_SUPPORTED ({tools.note})"
        elif tools is not None and (score := tools.number()) is not None:
            values["tool_calling"] = score >= 0.5
            sources["tool_calling"] = f"{tools.status.value}: Score {score:.2f}"

        notes: list[str] = []
        same_hw = hardware_fingerprint is None or hardware_fingerprint == self.hardware_fingerprint
        tps: float | None = None
        latency: float | None = None
        if same_hw:
            tps_m = self.perf("tokens_per_second")
            tps = tps_m.number()
            if tps is not None:
                values["speed"] = tokens_per_second_to_speed(tps)
                sources["speed"] = f"{tps_m.status.value}: {tps:.1f} tok/s ({tps_m.method})"
            latency = self.perf("first_token_latency").number()
            footprint = self.perf("memory_footprint")
            if (gb := footprint.number()) is not None and gb > 0:
                values["memory_requirement"] = round(gb, 2)
                sources["memory_requirement"] = f"{footprint.status.value}: {footprint.method}"
            ctx = self.model_info.get("context_length")
            if ctx is not None and (n := ctx.number()) is not None and n > 0:
                values["context_length"] = int(n)
                sources["context_length"] = f"{ctx.status.value}: {ctx.method}"
        else:
            notes.append("Leistungswerte auf anderer Hardware gemessen – nicht übernommen")
        if not values:
            return MeasuredOverrides(
                DataStatus.UNMEASURED,
                source=source,
                notes=("Profil enthält keine verwertbaren Messwerte", *notes),
            )
        has_perf = any(k in values for k in ("speed", "memory_requirement", "context_length"))
        has_caps = any(
            k in values
            for k in (
                "reasoning_capability",
                "coding_capability",
                "vision_capability",
                "tool_calling",
            )
        )
        status = DataStatus.MEASURED if has_perf and has_caps and same_hw else DataStatus.PARTIAL
        return MeasuredOverrides(
            status,
            source=source,
            tokens_per_second=tps,
            latency_s=latency,
            notes=tuple(notes),
            field_sources=sources,
            **values,
        )


def configured_snapshot(model: ModelMetadata) -> dict[str, Any]:
    """Konfigurationswerte zum Vergleich im Profil (Status CONFIGURED)."""
    return {
        "reasoning_capability": model.reasoning_capability.name.lower(),
        "coding_capability": model.coding_capability.name.lower(),
        "vision_capability": model.vision_capability.name.lower(),
        "tool_calling": model.tool_calling,
        "speed": model.speed.name.lower(),
        "memory_requirement_gb": model.memory_requirement,
        "context_length": model.context_length,
        "quantization": model.quantization,
        "parameter_count": model.parameter_count,
        "status": MeasurementStatus.CONFIGURED.value,
    }


# ---------------------------------------------------------------------- Speicherung

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def default_profile_dir() -> Path:
    env = os.environ.get("NOVA_BENCHMARK_DIR")
    return Path(env).expanduser() if env else Path.home() / ".nova" / "benchmarks"


class ProfileStore:
    """Profile als JSON: ``<root>/<modell>.json`` (aktuell) + ``<root>/history/…`` (alle).

    Implementiert :class:`models.measured.MeasurementSource` für den Registry. Gemessen
    wird nur, was ein echter Lauf geschrieben hat – manuell bearbeitete Dateien erkennt der
    Store nicht; deshalb schreibt nur der Benchmark-Runner Profile (``save``).
    """

    def __init__(
        self,
        root: Path | None = None,
        *,
        hardware_fingerprint: str | None = None,
        benchmark_version: str | None = None,
    ) -> None:
        self.root = (root or default_profile_dir()).expanduser()
        self.hardware_fingerprint = hardware_fingerprint
        self.benchmark_version = benchmark_version
        self._cache: dict[Path, tuple[int, ModelProfile]] = {}
        """Pfad → (mtime_ns, Profil): der Router fragt bei jeder Entscheidung nach."""

    @staticmethod
    def slug(model_name: str) -> str:
        return _SAFE.sub("_", model_name).strip("._") or "model"

    def path_for(self, model_name: str) -> Path:
        return self.root / f"{self.slug(model_name)}.json"

    def save(self, profile: ModelProfile) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        history = self.root / "history"
        history.mkdir(exist_ok=True)
        text = json.dumps(profile.to_dict(), indent=2, ensure_ascii=False)
        stamp = re.sub(r"[^0-9T]", "", profile.created_at)[:15]
        (history / f"{self.slug(profile.model)}-{stamp}.json").write_text(text, encoding="utf-8")
        target = self.path_for(profile.model)
        fd, tmp = tempfile.mkstemp(dir=self.root, prefix=".tmp-", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        Path(tmp).replace(target)  # atomar: nie halbe Profile
        return target

    def load(self, model_name: str) -> ModelProfile | None:
        path = self.path_for(model_name)
        try:
            mtime = path.stat().st_mtime_ns
        except FileNotFoundError:
            self._cache.pop(path, None)
            return None
        cached = self._cache.get(path)
        if cached is not None and cached[0] == mtime:
            profile = cached[1]
        else:
            profile = ModelProfile.from_dict(json.loads(path.read_text(encoding="utf-8")))
            self._cache[path] = (mtime, profile)
        if profile.model != model_name:
            return None
        return profile

    def list(self) -> list[ModelProfile]:
        if not self.root.is_dir():
            return []
        profiles = []
        for path in sorted(self.root.glob("*.json")):
            try:
                profiles.append(ModelProfile.from_dict(json.loads(path.read_text("utf-8"))))
            except (ValueError, KeyError, OSError):
                continue
        return profiles

    def overrides_for(self, model: ModelMetadata) -> MeasuredOverrides | None:
        try:
            profile = self.load(model.name)
        except (ValueError, KeyError, OSError) as exc:
            return MeasuredOverrides.unmeasured(f"Profil unlesbar: {exc}")
        if profile is None:
            return None
        return profile.to_overrides(
            hardware_fingerprint=self.hardware_fingerprint,
            benchmark_version=self.benchmark_version,
        )
