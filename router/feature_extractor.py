"""Merkmale für den Learned Router: (Aufgabe, Kandidatenmodell) → Zahlenvektor.

Grundsatz gegen Data Leakage: Merkmale entstehen **nur** aus dem, was zur Laufzeit bekannt
ist – Aufgabentext, Klassifikation des Routers, Kontextgröße, angebotene Tools, angehängte
Bilder, Modellmetadaten (gemessen oder konfiguriert) und Statistiken vergangener Ergebnisse.
Erwartete Labels eines Datensatzes kommen hier nie vor.

Der Vektor besteht aus

* **Modellmerkmalen** (Fähigkeiten, Geschwindigkeit, gemessene tok/s, Latenz, Speicher,
  Kontextreserve, historische Qualität/Erfolg/Fehlerrate – global und je Kategorie),
* **Interaktionen** Aufgabe × Modell (z. B. Komplexität × Reasoning-Stufe): Ein lineares
  Modell kann so lernen, *für welche Aufgaben* welche Modelleigenschaft zählt.

Reine Aufgabenmerkmale ohne Modellbezug sind für die Rangfolge innerhalb einer Anfrage
konstant und fließen nur über die Interaktionen ein.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from models.capabilities import CapabilityLevel, ModelMetadata, Speed
from router.base import (
    Complexity,
    RoutingCategory,
    RoutingRequest,
    TaskClassification,
    estimate_tokens,
)
from router.classifier import BROAD_SCOPE, CODING, DEEP_WORK, FORMAL, MULTI_STEP, REASONING

FEATURE_VERSION = 1

_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
_LIST_ITEM = re.compile(r"(?:^|\s)(?:\(\d+\)|\d+[.)]|[-*•])\s", re.MULTILINE)


@dataclass(frozen=True)
class TaskView:
    """Laufzeitwissen über eine Aufgabe (live aus Request + Klassifikation oder aus dem Log)."""

    text: str
    category: RoutingCategory
    complexity: Complexity
    confidence: float
    needs_tools: bool
    needs_vision: bool
    context_tokens: int
    """Angehängter Kontext (ohne Aufgabentext)."""
    secondary: RoutingCategory | None = None

    @classmethod
    def from_routing(cls, request: RoutingRequest, c: TaskClassification) -> TaskView:
        return cls(
            text=request.task,
            category=c.category,
            complexity=c.complexity,
            confidence=c.confidence,
            needs_tools=bool(request.required_tools),
            needs_vision=request.needs_vision,
            context_tokens=request.conversation_tokens,
            secondary=c.secondary,
        )

    @classmethod
    def from_log(cls, entry: Mapping[str, Any]) -> TaskView:
        """Aus einem Eintrag von :class:`~router.routing_log.JsonlRoutingLog`."""
        category = RoutingCategory(entry["category"])
        secondary = entry.get("secondary")
        return cls(
            text=str(entry.get("task") or ""),
            category=category,
            complexity=Complexity.parse(entry["complexity"]),
            confidence=float(entry.get("confidence") or 0.0),
            needs_tools=bool(entry.get("needs_tools")),
            needs_vision=bool(entry.get("needs_vision", category is RoutingCategory.VISION)),
            context_tokens=int(entry.get("conversation_tokens") or 0),
            secondary=RoutingCategory(secondary) if secondary else None,
        )

    @property
    def content(self) -> RoutingCategory:
        if self.category is RoutingCategory.LONG_CONTEXT:
            return self.secondary or RoutingCategory.GENERAL
        return self.category


# ---------------------------------------------------------------------- Historie


@dataclass
class OutcomeCounts:
    n: int = 0
    successes: int = 0
    hard_failures: int = 0
    quality_sum: float = 0.0
    quality_n: int = 0

    def add(self, success: bool, quality: float | None, hard_failure: bool, sign: int = 1) -> None:
        self.n += sign
        self.successes += sign * int(success)
        self.hard_failures += sign * int(hard_failure)
        if quality is not None:
            self.quality_sum += sign * quality
            self.quality_n += sign

    def copy(self) -> OutcomeCounts:
        return OutcomeCounts(
            self.n, self.successes, self.hard_failures, self.quality_sum, self.quality_n
        )


@dataclass
class HistoryStats:
    """Historische Ergebnisse je Modell und je (Modell, Kategorie).

    Werte werden mit einem Prior geglättet (wenige Beobachtungen → nahe am Prior), damit ein
    einzelner Erfolg kein Modell „perfekt“ macht.
    """

    per_model: dict[str, OutcomeCounts] = field(default_factory=dict)
    per_model_category: dict[tuple[str, str], OutcomeCounts] = field(default_factory=dict)
    prior_success: float = 0.5
    prior_quality: float = 0.5
    prior_weight: float = 5.0

    def add(
        self,
        model: str,
        category: RoutingCategory,
        success: bool,
        quality: float | None,
        hard_failure: bool,
        sign: int = 1,
    ) -> None:
        self.per_model.setdefault(model, OutcomeCounts()).add(success, quality, hard_failure, sign)
        self.per_model_category.setdefault((model, category.value), OutcomeCounts()).add(
            success, quality, hard_failure, sign
        )

    def _rates(self, counts: OutcomeCounts | None) -> tuple[float, float, float, float]:
        c = counts or OutcomeCounts()
        w = self.prior_weight
        success = (c.successes + w * self.prior_success) / (c.n + w)
        quality = (c.quality_sum + w * self.prior_quality) / (c.quality_n + w)
        failure = (c.hard_failures + w * (1 - self.prior_success) / 2) / (c.n + w)
        return success, quality, failure, math.log1p(max(c.n, 0)) / 5

    def features(self, model: str, category: RoutingCategory) -> dict[str, float]:
        s, q, f, n = self._rates(self.per_model.get(model))
        cs, cq, _cf, cn = self._rates(self.per_model_category.get((model, category.value)))
        return {
            "hist_success": s,
            "hist_quality": q,
            "hist_failure": f,
            "hist_count": n,
            "hist_cat_success": cs,
            "hist_cat_quality": cq,
            "hist_cat_count": cn,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "prior": [self.prior_success, self.prior_quality, self.prior_weight],
            "per_model": {k: vars(v) for k, v in self.per_model.items()},
            "per_model_category": {
                f"{m}|{c}": vars(v) for (m, c), v in self.per_model_category.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> HistoryStats:
        prior = data.get("prior", [0.5, 0.5, 5.0])
        stats = cls(prior_success=prior[0], prior_quality=prior[1], prior_weight=prior[2])
        stats.per_model = {k: OutcomeCounts(**v) for k, v in data.get("per_model", {}).items()}
        for key, v in data.get("per_model_category", {}).items():
            model, _, cat = key.rpartition("|")
            stats.per_model_category[(model, cat)] = OutcomeCounts(**v)
        return stats


# ---------------------------------------------------------------------- Extraktion

_INTERACT_MODEL = ("reasoning", "coding", "speed")


class FeatureExtractor:
    version = FEATURE_VERSION

    def task_features(self, task: TaskView) -> dict[str, float]:
        text = task.text
        words = len(text.split())
        code_hits = len({m.group(0).lower() for m in CODING.finditer(text)})
        reasoning_hits = len({m.group(0).lower() for m in REASONING.finditer(text)})
        content = task.content
        return {
            "complexity": (int(task.complexity) - 1) / 2,
            "confidence": task.confidence,
            "log_context": math.log1p(task.context_tokens) / 12,
            "words": math.log1p(words) / 6,
            "numbers": min(len(_NUMBER.findall(text)), 12) / 12,
            "list_items": min(len(_LIST_ITEM.findall(text)), 6) / 6,
            "code_signals": min(code_hits, 5) / 5,
            "reasoning_signals": min(reasoning_hits, 5) / 5,
            "multi_step": float(bool(MULTI_STEP.search(text))),
            "formal": float(bool(FORMAL.search(text))),
            "deep_work": float(bool(DEEP_WORK.search(text))),
            "broad_scope": float(bool(BROAD_SCOPE.search(text))),
            "needs_tools": float(task.needs_tools),
            "needs_vision": float(task.needs_vision),
            "cat_fast": float(task.category is RoutingCategory.FAST),
            "cat_general": float(content is RoutingCategory.GENERAL),
            "cat_coding": float(content is RoutingCategory.CODING),
            "cat_reasoning": float(content is RoutingCategory.REASONING),
            "cat_vision": float(task.category is RoutingCategory.VISION),
            "cat_long": float(task.category is RoutingCategory.LONG_CONTEXT),
        }

    def model_features(
        self, model: ModelMetadata, task: TaskView, history: HistoryStats
    ) -> dict[str, float]:
        extra = model.extra
        tps = extra.get("measured_tokens_per_second")
        latency = extra.get("measured_latency_s")
        needed = task.context_tokens + estimate_tokens(task.text) + 2048
        relevant = {
            RoutingCategory.CODING: model.coding_capability,
            RoutingCategory.REASONING: model.reasoning_capability,
            RoutingCategory.VISION: model.vision_capability,
        }.get(task.content, (model.reasoning_capability + model.coding_capability) / 2)
        if task.category is RoutingCategory.VISION:
            relevant = model.vision_capability
        features = {
            "m_reasoning": model.reasoning_capability / 3,
            "m_coding": model.coding_capability / 3,
            "m_vision": model.vision_capability / 3,
            "m_tools": float(model.tool_calling),
            "m_speed": (int(model.speed) - 1) / 2,
            "m_memory": math.log1p(model.memory_requirement) / 4,
            "m_context_headroom": max(min(math.log(model.context_length / max(needed, 1)), 4), 0)
            / 4,
            "m_tps": math.log1p(float(tps)) / 5 if isinstance(tps, int | float) else 0.0,
            "m_tps_known": float(isinstance(tps, int | float)),
            "m_latency": math.log1p(float(latency)) / 3
            if isinstance(latency, int | float)
            else 0.0,
            "m_latency_known": float(isinstance(latency, int | float)),
            "m_measured": float(extra.get("data_status") in ("MEASURED", "PARTIAL")),
            "m_relevant_cap": float(relevant) / 3,
            "m_cap_gap": (float(relevant) - int(task.complexity)) / 3,
        }
        features.update(history.features(model.name, task.content))
        return features

    def pair_features(
        self, model: ModelMetadata, task: TaskView, history: HistoryStats
    ) -> dict[str, float]:
        t = self.task_features(task)
        m = self.model_features(model, task, history)
        out = dict(m)
        model_values = {
            "reasoning": m["m_reasoning"],
            "coding": m["m_coding"],
            "speed": m["m_speed"],
        }
        for tk, tv in t.items():
            if tk == "confidence":
                continue
            for mk in _INTERACT_MODEL:
                out[f"x_{tk}*{mk}"] = tv * model_values[mk]
        out["x_complexity*relevant_cap"] = t["complexity"] * m["m_relevant_cap"]
        out["x_confidence*cap_gap"] = t["confidence"] * m["m_cap_gap"]
        out["x_log_context*headroom"] = t["log_context"] * m["m_context_headroom"]
        return out

    def names(self) -> list[str]:
        """Feste Merkmalsreihenfolge (Grundlage gespeicherter Modelle)."""
        probe_model = ModelMetadata(
            name="probe",
            provider="p",
            parameter_count=1,
            context_length=8192,
            reasoning_capability=CapabilityLevel.BASIC,
            coding_capability=CapabilityLevel.BASIC,
            vision_capability=CapabilityLevel.NONE,
            tool_calling=True,
            speed=Speed.MEDIUM,
            memory_requirement=1,
            quantization="Q",
        )
        probe_task = TaskView(
            "probe", RoutingCategory.GENERAL, Complexity.LOW, 0.5, False, False, 0
        )
        return list(self.pair_features(probe_model, probe_task, HistoryStats()))

    def vector(self, features: Mapping[str, float], names: Sequence[str]) -> list[float]:
        return [float(features.get(n, 0.0)) for n in names]
