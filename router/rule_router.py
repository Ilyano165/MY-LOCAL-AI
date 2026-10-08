"""Regelbasierter Model Router.

1. **Klassifikation** (austauschbar): Kategorie, Komplexität, Kontextbedarf, Tools.
2. **Harte Filter** – ein Modell ist nur wählbar, wenn es
   * jetzt verfügbar ist (Runtime bereit, Modell geladen/gelistet),
   * in den verfügbaren Speicher passt (VRAM, sonst RAM – sofern bekannt),
   * ein ausreichend großes Kontextfenster hat,
   * Tool-Calling kann, wenn Tools gebraucht werden,
   * Bilder versteht, wenn es eine Vision-Aufgabe ist.
3. **Rangfolge** unter den übrigen:
   * Fähigkeit in der Kategorie, gedeckelt auf das, was die Komplexität verlangt
     (LOW → basic, MEDIUM → good, HIGH → strong): „fähig genug“ zählt, nicht „maximal“;
   * dann Geschwindigkeit (Modelle, die nur im RAM laufen, gelten als langsamer);
   * dann volle Fähigkeit, dann geringerer Speicherbedarf, dann Name (deterministisch).
   * Latenz ``batch`` → Qualität vor Geschwindigkeit; ``realtime`` → schon „basic“ genügt.
   * LONG_CONTEXT → Modelle mit reichlich Kontextreserve zuerst.

Datengrundlage: gemessene Benchmark-Profile haben Vorrang vor Konfigurationswerten
(``ModelRegistry.list_effective``). Jede Begründung nennt, ob die Fähigkeit **gemessen** oder
nur **konfiguriert (UNMEASURED)** ist; ungemessene Werte werden nie als Messung ausgegeben.

Jede Entscheidung wird mit Begründung, abgelehnten Kandidaten und Fallbacks protokolliert.
"""

from __future__ import annotations

from collections.abc import Sequence

from models.capabilities import CapabilityLevel, ModelMetadata, TaskType
from models.measured import DataStatus, MeasuredOverrides
from models.model_registry import ModelRegistry
from router.availability import Availability, ModelAvailability
from router.base import (
    Complexity,
    Latency,
    ModelRouter,
    NoModelAvailableError,
    Rejection,
    RoutingCategory,
    RoutingDecision,
    RoutingRequest,
    TaskClassification,
    TaskClassifier,
)
from router.classifier import RuleBasedClassifier
from router.resources import ResourceBudget
from router.routing_log import RoutingLog

_REQUIRED = {Complexity.LOW: 1, Complexity.MEDIUM: 2, Complexity.HIGH: 3}
_LEVEL = {0: "none", 1: "basic", 2: "good", 3: "strong"}
_CAP_LABEL = {
    TaskType.CODING: "Coding",
    TaskType.REASONING: "Reasoning",
    TaskType.VISION: "Vision",
    TaskType.GENERAL: "Allround",
    TaskType.FAST: "Allround",
}


def capability(model: ModelMetadata, task_type: TaskType) -> float:
    if task_type is TaskType.CODING:
        return float(model.coding_capability)
    if task_type is TaskType.REASONING:
        return float(model.reasoning_capability)
    if task_type is TaskType.VISION:
        return float(model.vision_capability)
    return (model.reasoning_capability + model.coding_capability) / 2


class RuleBasedRouter(ModelRouter):
    def __init__(
        self,
        registry: ModelRegistry,
        availability: ModelAvailability,
        *,
        classifier: TaskClassifier | None = None,
        resources: ResourceBudget | None = None,
        log: RoutingLog | None = None,
        max_fallbacks: int = 3,
    ) -> None:
        self.registry = registry
        self.availability = availability
        self.classifier = classifier or RuleBasedClassifier()
        self.resources = resources or ResourceBudget()
        self.log = log
        self.max_fallbacks = max_fallbacks

    # ------------------------------------------------------------------ öffentliche API

    async def route(self, request: RoutingRequest) -> RoutingDecision:
        classification = await self.classifier.classify(request)
        models = self.registry.list_effective()
        data = {m.name: self.registry.data_status(m.name) for m in models}
        availability = await self.availability.check_all(models)
        candidates: list[ModelMetadata] = []
        rejected: list[Rejection] = []
        placements: dict[str, str] = {}
        for model in models:
            reasons, placement = self._hard_filter(model, classification, availability[model.name])
            if reasons:
                rejected.append(Rejection(model.name, tuple(reasons)))
            else:
                candidates.append(model)
                placements[model.name] = placement
        if not candidates:
            detail = (
                "; ".join(f"{r.model}: {', '.join(r.reasons)}" for r in rejected) or "Registry leer"
            )
            error = NoModelAvailableError(
                f"Kein verfügbares Modell für {classification.category.value} "
                f"(Komplexität {classification.complexity.name}): {detail}",
                {r.model: list(r.reasons) for r in rejected},
            )
            if self.log is not None:
                self.log.record_failure(request, classification, error)
            raise error

        ranked = sorted(candidates, key=lambda m: m.name)
        ranked.sort(
            key=lambda m: self._rank_key(m, classification, request, placements), reverse=True
        )
        selected = ranked[0]
        notes = self._notes(selected, placements[selected.name])
        notes.append(f"Daten {selected.name}: {data[selected.name].describe()}")
        decision = RoutingDecision(
            model=selected,
            classification=classification,
            reason=self._reason(selected, ranked, classification, request, data[selected.name]),
            fallbacks=ranked[1 : 1 + self.max_fallbacks],
            rejected=rejected,
            considered=[m.name for m in ranked],
            notes=notes,
            request=request,
            data_status={name: d.status.value for name, d in data.items()},
        )
        if self.log is not None:
            self.log.record(decision)
        return decision

    def report_failure(self, model_name: str, reason: str) -> None:
        """Akuter Ausfall (Timeout, Runtime weg): Modell bis zum nächsten Check meiden."""
        self.availability.mark_unavailable(model_name, reason)

    # ------------------------------------------------------------------ Filter & Rang

    def _hard_filter(
        self, model: ModelMetadata, c: TaskClassification, availability: Availability
    ) -> tuple[list[str], str]:
        reasons: list[str] = []
        if not availability.available:
            reasons.append(f"nicht verfügbar ({availability.reason})")
        placement = "unbekannt"
        if self.resources.known:
            fit = self.resources.placement(model.memory_requirement)
            if fit is None:
                reasons.append(
                    f"passt nicht in den Speicher (braucht {model.memory_requirement:g} GB; "
                    f"VRAM frei {_gb(self.resources.vram_gb)}, "
                    f"RAM frei {_gb(self.resources.ram_gb)})"
                )
            else:
                placement = fit
        if model.context_length < c.context_tokens:
            reasons.append(
                f"Kontext zu klein ({model.context_length} < ~{c.context_tokens} Tokens)"
            )
        if c.needs_tools and not model.tool_calling:
            reasons.append("kein Tool-Calling")
        if c.category is RoutingCategory.VISION and model.vision_capability is CapabilityLevel.NONE:
            reasons.append("keine Vision-Fähigkeit")
        content = self._content_type(c)
        if content in (TaskType.CODING, TaskType.REASONING) and capability(model, content) == 0:
            reasons.append(f"keine {_CAP_LABEL[content]}-Fähigkeit")
        return reasons, placement

    @staticmethod
    def _content_type(c: TaskClassification) -> TaskType:
        if c.category is RoutingCategory.LONG_CONTEXT:
            return (c.secondary or RoutingCategory.GENERAL).task_type
        return c.category.task_type

    def _rank_key(
        self,
        model: ModelMetadata,
        c: TaskClassification,
        request: RoutingRequest,
        placements: dict[str, str],
    ) -> tuple[float, ...]:
        cap = capability(model, self._content_type(c))
        speed = float(model.speed) - (1.0 if placements[model.name] == "cpu" else 0.0)
        lighter = -model.memory_requirement
        tools = 1.0 if model.tool_calling else 0.0
        if request.latency is Latency.BATCH:
            key: tuple[float, ...] = (cap, tools, speed, lighter)
        else:
            required = _REQUIRED[c.complexity]
            if request.latency is Latency.REALTIME or c.category is RoutingCategory.FAST:
                required = 1
            key = (min(cap, required), speed, cap, lighter)
        if c.category is RoutingCategory.LONG_CONTEXT:
            headroom = 1.0 if model.context_length >= 2 * c.context_tokens else 0.0
            key = (headroom, *key)
        return key

    # ------------------------------------------------------------------ Begründung

    def _reason(
        self,
        model: ModelMetadata,
        ranked: Sequence[ModelMetadata],
        c: TaskClassification,
        request: RoutingRequest,
        data: MeasuredOverrides | None = None,
    ) -> str:
        task_type = self._content_type(c)
        cap = capability(model, task_type)
        best = max(capability(m, task_type) for m in ranked)
        label = _CAP_LABEL[task_type]
        level = _LEVEL[round(cap)] if cap == int(cap) else f"{cap:.1f}"
        level += ", " + _provenance(task_type, data)
        if len(ranked) == 1:
            why = f"einziges Modell, das alle Anforderungen erfüllt ({label}: {level})"
        elif request.latency is Latency.BATCH or (cap >= best and c.complexity is Complexity.HIGH):
            why = f"stärkste {label}-Fähigkeit ({level}) unter {len(ranked)} geeigneten Modellen"
        elif cap >= best:
            why = f"beste {label}-Fähigkeit ({level}) und am schnellsten unter den fähigsten"
        else:
            need = _LEVEL[
                1
                if request.latency is Latency.REALTIME or c.category is RoutingCategory.FAST
                else _REQUIRED[c.complexity]
            ]
            why = f"schnellstes Modell mit ausreichender {label}-Fähigkeit (≥ {need})"
        extras = []
        if c.needs_tools:
            extras.append("unterstützt Tool-Calling")
        if c.category is RoutingCategory.LONG_CONTEXT:
            extras.append(f"Kontextfenster {model.context_length} Tokens")
        return f"{c.reason}. Gewählt: {why}" + (f"; {', '.join(extras)}" if extras else "")

    def _notes(self, model: ModelMetadata, placement: str) -> list[str]:
        notes = []
        if not self.resources.known:
            notes.append("Speicherbedarf nicht geprüft (Ressourcen unbekannt)")
        elif placement == "cpu":
            notes.append(
                f"{model.name} passt nicht in den VRAM – läuft voraussichtlich auf CPU/RAM"
            )
        return notes


def _provenance(task_type: TaskType, data: MeasuredOverrides | None) -> str:
    """„gemessen“ nur, wenn die verwendete Fähigkeit aus einem Benchmark-Profil stammt."""
    if data is None or data.status in (DataStatus.UNMEASURED, DataStatus.STALE):
        return "konfiguriert – UNMEASURED"
    fields = {
        TaskType.CODING: ("coding_capability",),
        TaskType.REASONING: ("reasoning_capability",),
        TaskType.VISION: ("vision_capability",),
    }.get(task_type, ("reasoning_capability", "coding_capability"))
    measured = [getattr(data, f) is not None for f in fields]
    if all(measured):
        return "gemessen"
    if any(measured):
        return "teilweise gemessen"
    return "konfiguriert – UNMEASURED"


def _gb(value: float | None) -> str:
    return "?" if value is None else f"{value:.1f} GB"
