"""Evaluation des Model Routers gegen einen gelabelten Datensatz.

Gemessen pro Szenario (Flotte + Verfügbarkeit):

* **category accuracy** / **complexity accuracy** (inkl. Konfusionsmatrix)
* **capability violations** – gewähltes Modell kann eine *erforderliche* Fähigkeit nicht
  (Vision, Coding, Reasoning, Tool-Calling, Kontextfenster) → **kritisch**
* **unavailable-model violations** – gewähltes Modell ist nicht verfügbar → kritisch
* **routing failures** – kein Modell gewählt, obwohl ein geeignetes verfügbar war (kritisch),
  bzw. ein Modell gewählt, obwohl keines geeignet war (kritisch, steckt in den Violations)
* **fallback rate** – Anteil der Aufgaben, bei denen Ausfälle eine andere Wahl erzwangen;
  dazu Fallback-Abdeckung (gibt es eine Ersatzwahl?) und Failover-Sicherheit
  (ist die erste Ersatzwahl ebenfalls geeignet?)
* **confidence calibration** – Klassifikator-Confidence gegen Kategorie-Treffer
  (ECE, Brier, Bins, überzeugte Fehler)
* **average routing latency** – Wanduhr pro ``route()``-Aufruf

Gewichtung (Strafpunkte pro Aufgabe, gedeckelt auf 10):

====================  ====  ==========================================================
kritisch              10    ungeeignetes/nicht verfügbares Modell, unberechtigter Fehlschlag
schwer (major)        3     Modell zu schwach für die Komplexität, obwohl ein stärkeres
                            geeignetes Modell verfügbar war
Kategorie             1     falsche Kategorie (ohne Folgen für die Eignung)
Komplexität           0.5   falsche Komplexitätsstufe
Überdimensioniert     0.5   einfache Aufgabe auf langsamerem Modell als nötig
====================  ====  ==========================================================

Ein Fehler mit ungeeignetem Modell wiegt damit zehnmal so schwer wie eine reine
Kategorieabweichung. ``routing_score = 1 − mittlere Strafe / 10``.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import statistics
import time
import tomllib
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from models.capabilities import ModelMetadata, Speed
from models.model_registry import ModelRegistry
from router.availability import StaticAvailability
from router.base import (
    Complexity,
    ModelRouter,
    NoModelAvailableError,
    RoutingCategory,
    RoutingDecision,
    RoutingRequest,
    estimate_tokens,
)
from router.rule_router import RuleBasedRouter

DATASET_DIR = Path(__file__).resolve().parent / "datasets"
DEFAULT_DATASET = DATASET_DIR / "routing_tasks.jsonl"
DEFAULT_FLEET = DATASET_DIR / "routing_fleet.toml"

SETS = ("FAST", "GENERAL", "CODING", "REASONING", "LONG_CONTEXT", "VISION", "MULTI_STEP")
CAPABILITIES = ("coding", "reasoning", "vision", "tool_calling", "long_context")
CONTEXT_LABELS = ("small", "medium", "large", "very_large")
DIFFICULTIES = ("easy", "medium", "hard")
OUTPUT_RESERVE = 2048
"""Antwortreserve wie in :meth:`RoutingRequest.from_chat` (``expected_output_tokens``)."""

WEIGHTS = {
    "critical": 10.0,
    "major": 3.0,
    "category": 1.0,
    "complexity": 0.5,
    "over_provisioned": 0.5,
}
_REQUIRED_LEVEL = {Complexity.LOW: 1, Complexity.MEDIUM: 2, Complexity.HIGH: 3}


# ---------------------------------------------------------------------- Datensatz


class DatasetError(ValueError):
    pass


@dataclass(frozen=True)
class RoutingTask:
    id: str
    set: str
    prompt: str
    expected_category: RoutingCategory
    expected_complexity: Complexity
    required_capabilities: tuple[str, ...]
    requires_tools: bool
    requires_vision: bool
    estimated_context: str
    context_tokens: int
    difficulty: str
    expected_secondary: RoutingCategory | None = None
    expected_routable: bool = True
    language: str = "de"
    tags: tuple[str, ...] = ()
    note: str = ""

    @property
    def needed_context(self) -> int:
        """Benötigtes Kontextfenster: angehängtes Material + Prompt + Antwortreserve."""
        return self.context_tokens + estimate_tokens(self.prompt) + OUTPUT_RESERVE

    def to_request(self) -> RoutingRequest:
        """Was das System zur Laufzeit *weiß*: Bilder, angebotene Tools, Kontextgröße.

        Kategorie und Komplexität werden **nicht** vorgegeben – die schätzt der Router.
        """
        return RoutingRequest(
            task=self.prompt,
            conversation_tokens=self.context_tokens,
            required_tools=("workspace",) if self.requires_tools else (),
            needs_vision=self.requires_vision,
            expected_output_tokens=OUTPUT_RESERVE,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any], line: int = 0) -> RoutingTask:
        where = f"Zeile {line} ({data.get('id', '?')})"
        required = (
            "id",
            "set",
            "prompt",
            "expected_category",
            "expected_complexity",
            "required_capabilities",
            "requires_tools",
            "requires_vision",
            "estimated_context",
            "difficulty",
        )
        missing = [k for k in required if k not in data]
        if missing:
            raise DatasetError(f"{where}: fehlende Felder {missing}")
        try:
            task = cls(
                id=str(data["id"]),
                set=str(data["set"]),
                prompt=str(data["prompt"]),
                expected_category=RoutingCategory(str(data["expected_category"]).lower()),
                expected_complexity=Complexity.parse(str(data["expected_complexity"])),
                required_capabilities=tuple(data["required_capabilities"]),
                requires_tools=bool(data["requires_tools"]),
                requires_vision=bool(data["requires_vision"]),
                estimated_context=str(data["estimated_context"]),
                context_tokens=int(data.get("context_tokens", 0)),
                difficulty=str(data["difficulty"]),
                expected_secondary=RoutingCategory(str(data["expected_secondary"]).lower())
                if data.get("expected_secondary")
                else None,
                expected_routable=bool(data.get("expected_routable", True)),
                language=str(data.get("language", "de")),
                tags=tuple(data.get("tags", ())),
                note=str(data.get("note", "")),
            )
        except (ValueError, KeyError) as exc:
            raise DatasetError(f"{where}: {exc}") from exc
        task.validate(where)
        return task

    def validate(self, where: str = "") -> None:
        problems = []
        if self.set not in SETS:
            problems.append(f"unbekanntes Set {self.set!r}")
        if unknown := set(self.required_capabilities) - set(CAPABILITIES):
            problems.append(f"unbekannte Fähigkeiten {sorted(unknown)}")
        if self.estimated_context not in CONTEXT_LABELS:
            problems.append(f"unbekannte Kontextgröße {self.estimated_context!r}")
        if self.difficulty not in DIFFICULTIES:
            problems.append(f"unbekannte Schwierigkeit {self.difficulty!r}")
        if self.requires_vision != ("vision" in self.required_capabilities):
            problems.append("requires_vision und required_capabilities widersprechen sich")
        if self.requires_tools != ("tool_calling" in self.required_capabilities):
            problems.append("requires_tools und required_capabilities widersprechen sich")
        if self.requires_vision and self.expected_category is not RoutingCategory.VISION:
            problems.append("Bildaufgabe muss VISION erwarten")
        if (self.expected_category is RoutingCategory.LONG_CONTEXT) != (
            self.expected_secondary is not None
        ):
            problems.append("expected_secondary genau bei LONG_CONTEXT angeben")
        if not self.prompt.strip():
            problems.append("leerer Prompt")
        if problems:
            raise DatasetError(f"{where or self.id}: " + "; ".join(problems))


def load_dataset(path: Path = DEFAULT_DATASET) -> list[RoutingTask]:
    tasks: list[RoutingTask] = []
    seen: set[str] = set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"Zeile {line_no}: kein gültiges JSON ({exc})") from exc
        task = RoutingTask.from_dict(data, line_no)
        if task.id in seen:
            raise DatasetError(f"Zeile {line_no}: doppelte ID {task.id}")
        seen.add(task.id)
        tasks.append(task)
    return tasks


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


# ---------------------------------------------------------------------- Flotte


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str = ""
    unavailable: tuple[str, ...] = ()


@dataclass
class Fleet:
    models: list[ModelMetadata]
    scenarios: list[Scenario]

    @classmethod
    def from_toml(cls, path: Path = DEFAULT_FLEET) -> Fleet:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
        models = ModelRegistry.from_mapping(data).list()
        scenarios = [
            Scenario(str(s["name"]), str(s.get("description", "")), tuple(s.get("unavailable", ())))
            for s in data.get("scenarios", [{"name": "all_available"}])
        ]
        names = {m.name for m in models}
        for s in scenarios:
            if unknown := set(s.unavailable) - names:
                raise ValueError(f"Szenario {s.name}: unbekannte Modelle {sorted(unknown)}")
        return cls(models, scenarios)

    def available(self, scenario: Scenario) -> list[ModelMetadata]:
        return [m for m in self.models if m.name not in scenario.unavailable]


RouterFactory = Callable[[ModelRegistry, StaticAvailability], ModelRouter]


def default_router_factory(
    registry: ModelRegistry, availability: StaticAvailability
) -> ModelRouter:
    return RuleBasedRouter(registry, availability)


# ---------------------------------------------------------------------- Eignung


def _capability_level(model: ModelMetadata, capability: str) -> int:
    if capability == "coding":
        return int(model.coding_capability)
    if capability == "reasoning":
        return int(model.reasoning_capability)
    if capability == "vision":
        return int(model.vision_capability)
    return 0


def missing_capabilities(model: ModelMetadata, task: RoutingTask) -> list[str]:
    """Harte Eignungsverletzungen – jede davon ist ein kritischer Fehler."""
    missing = []
    for cap in task.required_capabilities:
        if cap in ("coding", "reasoning", "vision") and _capability_level(model, cap) == 0:
            missing.append(f"keine {cap}-Fähigkeit")
        elif cap == "tool_calling" and not model.tool_calling:
            missing.append("kein Tool-Calling")
    if model.context_length < task.needed_context:
        missing.append(f"Kontext {model.context_length} < benötigt ~{task.needed_context}")
    return missing


def task_strength(model: ModelMetadata, task: RoutingTask) -> float:
    """Relevante Fähigkeitsstufe für die Aufgabe (schwächste geforderte Fähigkeit)."""
    caps = [c for c in task.required_capabilities if c in ("coding", "reasoning", "vision")]
    if not caps:
        return (int(model.reasoning_capability) + int(model.coding_capability)) / 2
    return float(min(_capability_level(model, c) for c in caps))


# ---------------------------------------------------------------------- Ergebnisse


@dataclass
class TaskOutcome:
    task: RoutingTask
    selected: str | None
    category: RoutingCategory | None
    secondary: RoutingCategory | None
    complexity: Complexity | None
    confidence: float | None
    fallbacks: list[str]
    latency_ms: float
    issues: dict[str, list[str]] = field(default_factory=dict)
    """Schwere → Liste von Befunden (critical, major, category, complexity, over_provisioned)."""
    reference_selection: str | None = None
    """Wahl bei voller Verfügbarkeit (für die Fallback-Rate)."""
    failover: str | None = None
    failover_ok: bool | None = None
    error: str | None = None

    @property
    def penalty(self) -> float:
        total = sum(WEIGHTS[kind] for kind, items in self.issues.items() if items)
        return min(total, WEIGHTS["critical"])

    @property
    def critical(self) -> bool:
        return bool(self.issues.get("critical"))

    @property
    def category_correct(self) -> bool:
        return self.category is self.task.expected_category

    @property
    def complexity_correct(self) -> bool:
        return self.complexity is self.task.expected_complexity

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.task.id,
            "set": self.task.set,
            "expected_category": self.task.expected_category.name,
            "category": self.category.name if self.category else None,
            "expected_complexity": self.task.expected_complexity.name,
            "complexity": self.complexity.name if self.complexity else None,
            "confidence": self.confidence,
            "selected": self.selected,
            "fallbacks": self.fallbacks,
            "reference_selection": self.reference_selection,
            "failover": self.failover,
            "failover_ok": self.failover_ok,
            "issues": {k: v for k, v in self.issues.items() if v},
            "penalty": self.penalty,
            "error": self.error,
        }


@dataclass
class CalibrationBin:
    low: float
    high: float
    count: int
    mean_confidence: float
    accuracy: float


@dataclass
class ScenarioReport:
    scenario: Scenario
    outcomes: list[TaskOutcome]

    # ------------------------------------------------------------------ Kennzahlen

    @property
    def n(self) -> int:
        return len(self.outcomes)

    def _classified(self) -> list[TaskOutcome]:
        return [o for o in self.outcomes if o.category is not None]

    @property
    def category_accuracy(self) -> float:
        c = self._classified()
        return sum(o.category_correct for o in c) / len(c) if c else 0.0

    @property
    def secondary_accuracy(self) -> float | None:
        c = [o for o in self._classified() if o.task.expected_secondary is not None]
        if not c:
            return None
        return sum(o.secondary is o.task.expected_secondary for o in c) / len(c)

    @property
    def complexity_accuracy(self) -> float:
        c = [o for o in self._classified() if o.complexity is not None]
        return sum(o.complexity_correct for o in c) / len(c) if c else 0.0

    def count(self, kind: str, marker: str | None = None) -> int:
        return sum(
            1
            for o in self.outcomes
            if o.issues.get(kind) and (marker is None or any(marker in i for i in o.issues[kind]))
        )

    @property
    def capability_violations(self) -> int:
        return sum(
            1
            for o in self.outcomes
            if any(i.startswith("ungeeignet") for i in o.issues.get("critical", []))
        )

    @property
    def unavailable_violations(self) -> int:
        return self.count("critical", "nicht verfügbar gewählt")

    @property
    def routing_failures(self) -> int:
        return self.count("critical", "Routing fehlgeschlagen")

    @property
    def correct_refusals(self) -> int:
        return sum(1 for o in self.outcomes if not o.task.expected_routable and o.selected is None)

    @property
    def critical_errors(self) -> int:
        return sum(o.critical for o in self.outcomes)

    @property
    def routing_score(self) -> float:
        return 1 - statistics.fmean(o.penalty for o in self.outcomes) / WEIGHTS["critical"]

    @property
    def fallback_rate(self) -> float:
        routed = [o for o in self.outcomes if o.selected is not None]
        changed = [o for o in routed if o.reference_selection not in (None, o.selected)]
        return len(changed) / len(routed) if routed else 0.0

    @property
    def fallback_coverage(self) -> float:
        routed = [o for o in self.outcomes if o.selected is not None]
        return sum(bool(o.fallbacks) for o in routed) / len(routed) if routed else 0.0

    @property
    def failover_safety(self) -> float | None:
        checked = [o for o in self.outcomes if o.failover_ok is not None]
        return sum(bool(o.failover_ok) for o in checked) / len(checked) if checked else None

    def latency(self) -> dict[str, float]:
        values = sorted(o.latency_ms for o in self.outcomes)
        return {
            "mean_ms": statistics.fmean(values),
            "p50_ms": values[len(values) // 2],
            "p95_ms": values[min(len(values) - 1, int(len(values) * 0.95))],
            "max_ms": values[-1],
        }

    def calibration(
        self, edges: Sequence[float] = (0.0, 0.6, 0.8, 0.9, 1.0001)
    ) -> tuple[list[CalibrationBin], float, float]:
        """(Bins, ECE, Brier) für Kategorie-Treffer."""
        data = [
            (o.confidence, o.category_correct)
            for o in self._classified()
            if o.confidence is not None
        ]
        bins = []
        for low, high in itertools.pairwise(edges):
            items = [(c, ok) for c, ok in data if low <= c < high]
            if items:
                bins.append(
                    CalibrationBin(
                        low,
                        min(high, 1.0),
                        len(items),
                        statistics.fmean(c for c, _ in items),
                        sum(ok for _, ok in items) / len(items),
                    )
                )
        total = len(data) or 1
        ece = sum(b.count / total * abs(b.mean_confidence - b.accuracy) for b in bins)
        brier = statistics.fmean((c - float(ok)) ** 2 for c, ok in data) if data else 0.0
        return bins, ece, brier

    def overconfident_errors(self, threshold: float = 0.85) -> list[TaskOutcome]:
        return [
            o
            for o in self._classified()
            if o.confidence is not None
            and o.confidence >= threshold
            and (not o.category_correct or o.critical)
        ]

    def confusion(self) -> Counter[tuple[str, str]]:
        return Counter(
            (o.task.expected_category.name, o.category.name)
            for o in self._classified()
            if not o.category_correct and o.category is not None
        )

    def complexity_confusion(self) -> Counter[tuple[str, str]]:
        return Counter(
            (o.task.expected_complexity.name, o.complexity.name)
            for o in self._classified()
            if not o.complexity_correct and o.complexity is not None
        )

    def per_set(self) -> dict[str, dict[str, float]]:
        out = {}
        for name in SETS:
            items = [o for o in self.outcomes if o.task.set == name]
            if not items:
                continue
            out[name] = {
                "n": len(items),
                "category_accuracy": sum(o.category_correct for o in items) / len(items),
                "complexity_accuracy": sum(o.complexity_correct for o in items) / len(items),
                "critical": sum(o.critical for o in items),
                "score": 1 - statistics.fmean(o.penalty for o in items) / WEIGHTS["critical"],
            }
        return out

    def per_difficulty(self) -> dict[str, float]:
        out = {}
        for d in DIFFICULTIES:
            items = [o for o in self.outcomes if o.task.difficulty == d]
            if items:
                out[d] = sum(o.category_correct for o in items) / len(items)
        return out

    def summary(self) -> dict[str, Any]:
        bins, ece, brier = self.calibration()
        return {
            "scenario": self.scenario.name,
            "tasks": self.n,
            "routing_score": round(self.routing_score, 4),
            "category_accuracy": round(self.category_accuracy, 4),
            "secondary_accuracy": None
            if self.secondary_accuracy is None
            else round(self.secondary_accuracy, 4),
            "complexity_accuracy": round(self.complexity_accuracy, 4),
            "critical_errors": self.critical_errors,
            "capability_violations": self.capability_violations,
            "unavailable_violations": self.unavailable_violations,
            "routing_failures": self.routing_failures,
            "correct_refusals": self.correct_refusals,
            "major_errors": self.count("major"),
            "over_provisioned": self.count("over_provisioned"),
            "fallback_rate": round(self.fallback_rate, 4),
            "fallback_coverage": round(self.fallback_coverage, 4),
            "failover_safety": None
            if self.failover_safety is None
            else round(self.failover_safety, 4),
            "ece": round(ece, 4),
            "brier": round(brier, 4),
            "overconfident_errors": len(self.overconfident_errors()),
            "latency": {k: round(v, 3) for k, v in self.latency().items()},
            "calibration_bins": [vars(b) for b in bins],
        }


# ---------------------------------------------------------------------- Evaluator


class RoutingEvaluator:
    def __init__(
        self,
        fleet: Fleet,
        *,
        router_factory: RouterFactory = default_router_factory,
        latency_repeats: int = 3,
    ) -> None:
        if latency_repeats < 1:
            raise ValueError("latency_repeats muss ≥ 1 sein")
        self.fleet = fleet
        self.router_factory = router_factory
        self.latency_repeats = latency_repeats

    def _router(self, available: Iterable[str]) -> ModelRouter:
        registry = ModelRegistry(self.fleet.models)
        return self.router_factory(registry, StaticAvailability(available))

    async def _route(
        self, router: ModelRouter, task: RoutingTask
    ) -> tuple[RoutingDecision | None, str | None, float]:
        timings = []
        decision: RoutingDecision | None = None
        error: str | None = None
        for _ in range(self.latency_repeats):
            started = time.perf_counter()
            try:
                decision = await router.route(task.to_request())
                error = None
            except NoModelAvailableError as exc:
                decision, error = None, str(exc)
            timings.append((time.perf_counter() - started) * 1000)
        return decision, error, min(timings)

    async def evaluate(self, tasks: Sequence[RoutingTask], scenario: Scenario) -> ScenarioReport:
        available = self.fleet.available(scenario)
        available_names = [m.name for m in available]
        router = self._router(available_names)
        reference = self._router(m.name for m in self.fleet.models)
        by_name = {m.name: m for m in self.fleet.models}
        outcomes = []
        for task in tasks:
            decision, error, latency = await self._route(router, task)
            ref_decision, _, _ = await self._route(reference, task)
            outcome = TaskOutcome(
                task=task,
                selected=decision.model.name if decision else None,
                category=decision.classification.category if decision else None,
                secondary=decision.classification.secondary if decision else None,
                complexity=decision.classification.complexity if decision else None,
                confidence=decision.classification.confidence if decision else None,
                fallbacks=[m.name for m in decision.fallbacks] if decision else [],
                latency_ms=latency,
                reference_selection=ref_decision.model.name if ref_decision else None,
                error=error,
            )
            if decision is None:
                # Klassifikation trotzdem bewerten (Router scheiterte erst an den Filtern)
                classification = await _classify(router, task)
                if classification is not None:
                    outcome.category, outcome.secondary, outcome.complexity, outcome.confidence = (
                        classification
                    )
            self._judge(outcome, available, by_name)
            if decision is not None:
                await self._failover(outcome, available_names, by_name)
            outcomes.append(outcome)
        return ScenarioReport(scenario, outcomes)

    async def evaluate_all(self, tasks: Sequence[RoutingTask]) -> list[ScenarioReport]:
        return [await self.evaluate(tasks, s) for s in self.fleet.scenarios]

    # ------------------------------------------------------------------ Bewertung

    def _judge(
        self,
        outcome: TaskOutcome,
        available: Sequence[ModelMetadata],
        by_name: dict[str, ModelMetadata],
    ) -> None:
        task = outcome.task
        issues: dict[str, list[str]] = {k: [] for k in WEIGHTS}
        suitable = [m for m in available if not missing_capabilities(m, task)]
        if outcome.selected is None:
            if suitable:
                issues["critical"].append(
                    "Routing fehlgeschlagen, obwohl geeignet: "
                    + ", ".join(m.name for m in suitable)
                )
        else:
            model = by_name[outcome.selected]
            if model not in available:
                issues["critical"].append(f"{model.name} nicht verfügbar gewählt")
            missing = missing_capabilities(model, task)
            if missing:
                issues["critical"].append(f"ungeeignet: {model.name} – {'; '.join(missing)}")
            elif suitable:
                need = _REQUIRED_LEVEL[task.expected_complexity]
                have = task_strength(model, task)
                best = max(task_strength(m, task) for m in suitable)
                if have < need and best > have:
                    issues["major"].append(
                        f"{model.name} zu schwach (Stufe {have:g} < {need} für "
                        f"{task.expected_complexity.name}; verfügbar bis {best:g})"
                    )
                capable = [m.speed for m in suitable if task_strength(m, task) >= need]
                if task.expected_complexity is Complexity.LOW and capable:
                    fastest = max(capable)
                    if model.speed < fastest:
                        issues["over_provisioned"].append(
                            f"{model.name} ({Speed(model.speed).name}) statt eines "
                            f"{Speed(fastest).name}-Modells"
                        )
        if outcome.category is not None and not outcome.category_correct:
            issues["category"].append(
                f"{task.expected_category.name} erwartet, {outcome.category.name} erkannt"
            )
        if outcome.complexity is not None and not outcome.complexity_correct:
            issues["complexity"].append(
                f"{task.expected_complexity.name} erwartet, {outcome.complexity.name} erkannt"
            )
        outcome.issues = issues

    async def _failover(
        self, outcome: TaskOutcome, available: Sequence[str], by_name: dict[str, ModelMetadata]
    ) -> None:
        """Simuliert den Ausfall der ersten Wahl: ist die Ersatzwahl noch geeignet?"""
        remaining = [n for n in available if n != outcome.selected]
        if not any(not missing_capabilities(by_name[n], outcome.task) for n in remaining):
            return  # kein geeigneter Ersatz existiert – nichts zu bewerten
        decision, _, _ = await self._route(self._router(remaining), outcome.task)
        outcome.failover = decision.model.name if decision else None
        outcome.failover_ok = decision is not None and not missing_capabilities(
            decision.model, outcome.task
        )


async def _classify(
    router: ModelRouter, task: RoutingTask
) -> tuple[RoutingCategory, RoutingCategory | None, Complexity, float] | None:
    classifier = getattr(router, "classifier", None)
    if classifier is None:
        return None
    c = await classifier.classify(task.to_request())
    return c.category, c.secondary, c.complexity, c.confidence


# ---------------------------------------------------------------------- Bericht


def _pct(value: float | None) -> str:
    return "–" if value is None else f"{value * 100:.1f} %"


def render_report(
    reports: Sequence[ScenarioReport],
    *,
    dataset: Path,
    fleet: Path,
    router_name: str,
    generated: str,
    environment: str = "",
    recommendations: Sequence[str] = (),
) -> str:
    tasks = reports[0].outcomes if reports else []
    sets = Counter(o.task.set for o in tasks)
    lines = [
        "# NOVA – Routing-Baseline",
        "",
        f"Erzeugt: {generated} · Router: `{router_name}` · Datensatz: "
        f"`{dataset.name}` (sha256 {file_hash(dataset)}, {len(tasks)} Aufgaben) · Flotte: "
        f"`{fleet.name}` (sha256 {file_hash(fleet)})",
        "",
        "Reproduzieren: `python -m scripts.evaluate_routing` – alle Werte außer der Latenz sind "
        "deterministisch.",
        "",
        "Verteilung: " + ", ".join(f"{k} {sets[k]}" for k in SETS if sets[k]),
        "",
        "## 1. Gesamtergebnis",
        "",
        "| Kennzahl | " + " | ".join(r.scenario.name for r in reports) + " |",
        "|---|" + "---|" * len(reports),
    ]
    rows: list[tuple[str, Callable[[ScenarioReport], str]]] = [
        ("Routing-Score (gewichtet, 1 = fehlerfrei)", lambda r: f"{r.routing_score:.3f}"),
        ("Kategorie-Genauigkeit", lambda r: _pct(r.category_accuracy)),
        ("Sekundärkategorie (LONG_CONTEXT)", lambda r: _pct(r.secondary_accuracy)),
        ("Komplexitäts-Genauigkeit", lambda r: _pct(r.complexity_accuracy)),
        ("**Kritische Fehler**", lambda r: f"**{r.critical_errors}**"),
        ("– Capability-Verletzungen", lambda r: str(r.capability_violations)),
        ("– nicht verfügbares Modell gewählt", lambda r: str(r.unavailable_violations)),
        ("– unberechtigte Routing-Fehlschläge", lambda r: str(r.routing_failures)),
        ("Korrekte Ablehnungen (kein Modell geeignet)", lambda r: str(r.correct_refusals)),
        ("Schwere Fehler (Modell zu schwach)", lambda r: str(r.count("major"))),
        (
            "Überdimensioniert (einfache Aufgabe, langsames Modell)",
            lambda r: str(r.count("over_provisioned")),
        ),
        ("Fallback-Rate (Wahl durch Ausfall geändert)", lambda r: _pct(r.fallback_rate)),
        ("Fallback-Abdeckung (Ersatzmodell vorhanden)", lambda r: _pct(r.fallback_coverage)),
        ("Failover-Sicherheit (Ersatzwahl geeignet)", lambda r: _pct(r.failover_safety)),
        ("ECE (Confidence-Kalibrierung, 0 = ideal)", lambda r: f"{r.calibration()[1]:.3f}"),
        ("Brier-Score", lambda r: f"{r.calibration()[2]:.3f}"),
        ("Überzeugte Fehler (Confidence ≥ 0.85)", lambda r: str(len(r.overconfident_errors()))),
        (
            "Routing-Latenz Ø / p95",
            lambda r: f"{r.latency()['mean_ms']:.2f} / {r.latency()['p95_ms']:.2f} ms",
        ),
    ]
    for label, fn in rows:
        lines.append(f"| {label} | " + " | ".join(fn(r) for r in reports) + " |")
    if environment:
        lines += ["", f"Latenz gemessen auf: {environment} (einziger hardwareabhängiger Wert)."]

    base = reports[0]
    lines += [
        "",
        f"## 2. Ergebnis pro Aufgabenset ({base.scenario.name})",
        "",
        "| Set | n | Kategorie | Komplexität | kritisch | Score |",
        "|---|---|---|---|---|---|",
    ]
    for name, s in base.per_set().items():
        lines.append(
            f"| {name} | {s['n']:.0f} | {_pct(s['category_accuracy'])} | "
            f"{_pct(s['complexity_accuracy'])} | {s['critical']:.0f} | {s['score']:.3f} |"
        )
    lines += [
        "",
        "Kategorie-Genauigkeit nach Schwierigkeit: "
        + ", ".join(f"{k} {_pct(v)}" for k, v in base.per_difficulty().items()),
    ]

    lines += ["", "## 3. Kritische Fehler", ""]
    any_critical = False
    for r in reports:
        crit = [o for o in r.outcomes if o.critical]
        if not crit:
            continue
        any_critical = True
        lines += [
            f"### Szenario `{r.scenario.name}` ({len(crit)})",
            "",
            "| ID | erwartet | erkannt | gewählt | Befund |",
            "|---|---|---|---|---|",
        ]
        for o in crit:
            lines.append(
                f"| {o.task.id} | {o.task.expected_category.name} | "
                f"{o.category.name if o.category else '–'} | {o.selected or '–'} | "
                f"{'; '.join(o.issues['critical'])} |"
            )
        lines.append("")
    if not any_critical:
        lines += ["Keine kritischen Fehler in den Szenarien.", ""]

    lines += [
        "## 4. Häufigste Fehlklassifikationen",
        "",
        "| erwartet → erkannt | Anzahl | Beispiele |",
        "|---|---|---|",
    ]
    for (exp, got), count in base.confusion().most_common(10):
        examples = [
            o.task.id
            for o in base.outcomes
            if o.task.expected_category.name == exp and o.category and o.category.name == got
        ][:5]
        lines.append(f"| {exp} → {got} | {count} | {', '.join(examples)} |")
    lines += [
        "",
        "Komplexität (erwartet → erkannt): "
        + ", ".join(f"{e}→{g}: {c}" for (e, g), c in base.complexity_confusion().most_common()),
    ]

    lines += [
        "",
        "## 5. Problematische Aufgaben",
        "",
        "Höchste Strafpunkte im Szenario "
        f"`{base.scenario.name}` (kritisch 10 · schwer 3 · Kategorie 1 · Komplexität 0.5 · "
        "überdimensioniert 0.5):",
        "",
        "| ID | Strafe | Befunde | Prompt (Anfang) |",
        "|---|---|---|---|",
    ]
    worst = sorted(base.outcomes, key=lambda o: (-o.penalty, o.task.id))[:20]
    for o in worst:
        if o.penalty == 0:
            break
        found = "; ".join(i for items in o.issues.values() for i in items)
        prompt = o.task.prompt.replace("\n", " ").replace("|", "/")[:70]
        lines.append(f"| {o.task.id} | {o.penalty:g} | {found} | {prompt}… |")

    lines += [
        "",
        "## 6. Confidence / Sicherheit",
        "",
        "| Confidence | n | Ø Confidence | Treffer |",
        "|---|---|---|---|",
    ]
    bins, ece, brier = base.calibration()
    for b in bins:
        lines.append(
            f"| {b.low:.2f}–{b.high:.2f} | {b.count} | {b.mean_confidence:.2f} | "
            f"{_pct(b.accuracy)} |"
        )
    over = base.overconfident_errors()
    lines += [
        "",
        f"ECE {ece:.3f}, Brier {brier:.3f}. Überzeugte Fehler (Confidence ≥ 0.85, "
        f"falsche Kategorie oder kritisch): {len(over)}",
    ]
    if over:
        lines.append("")
        for o in over[:15]:
            lines.append(
                f"* {o.task.id}: {o.confidence:.2f} → "
                f"{o.category.name if o.category else '–'} "
                f"(erwartet {o.task.expected_category.name})"
                + (" **kritisch**" if o.critical else "")
            )
    if recommendations:
        lines += ["", "## 7. Verbesserungsvorschläge", ""]
        lines += [f"{i}. {r}" for i, r in enumerate(recommendations, 1)]
    lines += [
        "",
        "## Methodik",
        "",
        "* Der Router erhält nur, was das System zur Laufzeit weiß: Prompt, angehängte "
        "Kontextgröße, ob Bilder anhängen und ob Tools angeboten werden. Kategorie und "
        "Komplexität schätzt er selbst.",
        "* Eignung (kritisch) wird unabhängig vom Router geprüft: geforderte Fähigkeiten "
        "(Stufe > none), Tool-Calling, Kontextfenster ≥ Material + Prompt + 2048.",
        "* Fallback-Rate: Anteil der Aufgaben, deren Wahl sich gegenüber voller "
        "Verfügbarkeit ändert. Failover-Sicherheit: erste Wahl wird entfernt, ist die neue "
        "Wahl noch geeignet?",
        "* Labels: `evaluation/datasets/README.md`.",
    ]
    return "\n".join(lines) + "\n"


def write_json(reports: Sequence[ScenarioReport], path: Path) -> None:
    data = {
        "scenarios": [r.summary() for r in reports],
        "outcomes": {r.scenario.name: [o.to_dict() for o in r.outcomes] for r in reports},
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
