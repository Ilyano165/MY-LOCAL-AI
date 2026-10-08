"""Training und Vergleich: Rule Router vs. Learned Router auf dem Routing-Datensatz.

**Wichtig – Herkunft der Ergebnisse:** Für die 150 Datensatzaufgaben gibt es (noch) keine
echten Läufe mit lokalen Modellen. Die Ergebnisse (Erfolg, Qualität, Latenz) werden daher
aus den **Labels** und der Flottenkonfiguration abgeleitet (:func:`simulate_outcome`) und
sind überall als ``SIMULATED`` gekennzeichnet. Ein Vorsprung auf diesen Daten zeigt nur, dass
der Learned Router die Label-Regeln besser nachbildet als die Regel-Heuristik – nicht, dass
er mit echten Modellen bessere Antworten liefert. Echte Daten liefert das Routing-Log
(:func:`router.training_data.from_routing_log`).

Simulationsregeln (bewusst einfach, vollständig offengelegt):

* geforderte Fähigkeit fehlt / Kontext zu klein → harter Fehlschlag, Qualität 0
* benötigte Stufe = Komplexität (LOW 1, MEDIUM 2, HIGH 3); Stärke = schwächste geforderte
  Fähigkeit (ohne Anforderung: Mittel aus Reasoning und Coding)
* Stärke ≥ Stufe → Erfolg, Qualität 0.9; sonst Misserfolg, Qualität 0.9 − 0.35 · Defizit
* Latenz = Kontext / Prompt-Rate + Ausgabe / Dekodierrate + 0.3 s, Raten je Geschwindigkeits-
  klasse angenommen (fast 60 / medium 25 / slow 10 tok/s; Prompt 2000 / 800 / 300 tok/s),
  gemessene tok/s haben Vorrang; Ausgabelänge LOW 150, MEDIUM 500, HIGH 1200 Tokens
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evaluation.routing_evaluator import (
    DATASET_DIR,
    Fleet,
    RoutingTask,
    missing_capabilities,
    task_strength,
)
from models.capabilities import ModelMetadata
from models.model_registry import ModelRegistry
from router.availability import StaticAvailability
from router.base import Complexity, NoModelAvailableError
from router.classifier import RuleBasedClassifier
from router.feature_extractor import TaskView
from router.learned_router import LearnedRanker, LearnedRouter
from router.rule_router import RuleBasedRouter
from router.training_data import OutcomeRecord

SPLIT_FILE = DATASET_DIR / "routing_split.json"
SPLIT_SALT = "nova-routing-split-v1"
SOURCE = "simulated"

_REQUIRED = {Complexity.LOW: 1, Complexity.MEDIUM: 2, Complexity.HIGH: 3}
_OUTPUT_TOKENS = {Complexity.LOW: 150, Complexity.MEDIUM: 500, Complexity.HIGH: 1200}
_DECODE_TPS = {1: 10.0, 2: 25.0, 3: 60.0}
_PROMPT_TPS = {1: 300.0, 2: 800.0, 3: 2000.0}
ACCEPTABLE_GAP = 0.02


@dataclass(frozen=True)
class SimulatedOutcome:
    success: bool
    quality: float
    latency_s: float
    hard_failure: bool


def simulate_outcome(task: RoutingTask, model: ModelMetadata) -> SimulatedOutcome:
    tps = model.extra.get("measured_tokens_per_second")
    decode = float(tps) if isinstance(tps, int | float) and tps > 0 else _DECODE_TPS[model.speed]
    latency = (
        task.context_tokens / _PROMPT_TPS[model.speed]
        + _OUTPUT_TOKENS[task.expected_complexity] / decode
        + 0.3
    )
    if missing_capabilities(model, task):
        return SimulatedOutcome(False, 0.0, round(latency, 3), True)
    deficit = _REQUIRED[task.expected_complexity] - task_strength(model, task)
    if deficit <= 0:
        return SimulatedOutcome(True, 0.9, round(latency, 3), False)
    return SimulatedOutcome(
        False, round(max(0.1, 0.9 - 0.35 * deficit), 3), round(latency, 3), False
    )


def utility(outcome: SimulatedOutcome) -> float:
    record = OutcomeRecord(
        "g",
        "t",
        _dummy_view(),
        "m",
        outcome.success,
        outcome.quality,
        outcome.latency_s,
        hard_failure=outcome.hard_failure,
        source=SOURCE,
    )
    return record.utility


def _dummy_view() -> TaskView:
    from router.base import RoutingCategory

    return TaskView("", RoutingCategory.GENERAL, Complexity.LOW, 0.0, False, False, 0)


# ---------------------------------------------------------------------- Split


def make_split(tasks: Sequence[RoutingTask], test_fraction: float = 0.3) -> list[str]:
    """Fester, je Set geschichteter Split (gesalzener Hash der ID) – Test-IDs sortiert."""
    import hashlib

    test: list[str] = []
    sets = sorted({t.set for t in tasks})
    for name in sets:
        ids = sorted(
            (t.id for t in tasks if t.set == name),
            key=lambda i: hashlib.sha256(f"{SPLIT_SALT}:{i}".encode()).hexdigest(),
        )
        test += ids[: max(1, round(len(ids) * test_fraction))]
    return sorted(test)


def load_split(path: Path = SPLIT_FILE) -> list[str]:
    return list(json.loads(path.read_text(encoding="utf-8"))["test_ids"])


def split_tasks(
    tasks: Sequence[RoutingTask], test_ids: Sequence[str]
) -> tuple[list[RoutingTask], list[RoutingTask]]:
    test_set = set(test_ids)
    unknown = test_set - {t.id for t in tasks}
    if unknown:
        raise ValueError(f"Split enthält unbekannte IDs: {sorted(unknown)[:5]}")
    return [t for t in tasks if t.id not in test_set], [t for t in tasks if t.id in test_set]


# ---------------------------------------------------------------------- Trainingsdaten


async def simulated_records(tasks: Sequence[RoutingTask], fleet: Fleet) -> list[OutcomeRecord]:
    """Pro Aufgabe × Szenario: alle Kandidaten, die die harten Regeln zulassen, mit Ergebnis.

    Die Kandidaten kommen aus derselben Hard-Constraint-Stufe wie zur Laufzeit; die Merkmale
    aus der Klassifikation des Routers – nicht aus den Labels.
    """
    classifier = RuleBasedClassifier()
    records: list[OutcomeRecord] = []
    for scenario in fleet.scenarios:
        router = RuleBasedRouter(
            ModelRegistry(fleet.models),
            StaticAvailability(m.name for m in fleet.available(scenario)),
        )
        for task in tasks:
            request = task.to_request()
            classification = await classifier.classify(request)
            try:
                stage = await router.candidates(request, classification)
            except NoModelAvailableError:
                continue
            view = TaskView.from_routing(request, classification)
            for model in stage.candidates:
                outcome = simulate_outcome(task, model)
                records.append(
                    OutcomeRecord(
                        group=f"{task.id}@{scenario.name}",
                        task_id=task.id,
                        task=view,
                        model=model.name,
                        success=outcome.success,
                        quality=outcome.quality,
                        latency_s=outcome.latency_s,
                        hard_failure=outcome.hard_failure,
                        source=SOURCE,
                    )
                )
    return records


# ---------------------------------------------------------------------- Vergleich


@dataclass
class RouterResult:
    task_id: str
    selected: str | None
    ranker: str
    outcome: SimulatedOutcome | None
    utility: float | None
    best_utility: float | None
    oracle: list[str]
    critical: list[str]
    reference: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def exact(self) -> bool:
        return self.selected is not None and self.selected in self.oracle

    @property
    def acceptable(self) -> bool:
        return (
            self.utility is not None
            and self.best_utility is not None
            and self.utility >= self.best_utility - ACCEPTABLE_GAP
        )


@dataclass
class ComparisonReport:
    router: str
    scenario: str
    results: list[RouterResult]
    guard_interventions: int = 0

    def metrics(self) -> dict[str, Any]:
        routed = [r for r in self.results if r.selected is not None]
        expected = [r for r in self.results if r.oracle]
        n = len(expected) or 1
        outcomes = [r.outcome for r in routed if r.outcome is not None]
        return {
            "router": self.router,
            "scenario": self.scenario,
            "tasks": len(self.results),
            "routing_accuracy": sum(r.exact for r in expected) / n,
            "acceptable_rate": sum(r.acceptable for r in expected) / n,
            "verification_success": (sum(o.success for o in outcomes) / len(outcomes))
            if outcomes
            else 0.0,
            "avg_quality": statistics.fmean(o.quality for o in outcomes) if outcomes else 0.0,
            "avg_latency_s": statistics.fmean(o.latency_s for o in outcomes) if outcomes else 0.0,
            "avg_utility": statistics.fmean(r.utility for r in routed if r.utility is not None)
            if routed
            else 0.0,
            "critical_violations": sum(bool(r.critical) for r in self.results),
            "routing_failures": sum(1 for r in self.results if r.selected is None and r.oracle),
            "fallback_rate": (
                sum(1 for r in routed if r.reference not in (None, r.selected)) / len(routed)
            )
            if routed
            else 0.0,
            "guard_interventions": self.guard_interventions,
            "ranker_unavailable": sum(1 for r in routed if r.ranker.startswith("rules (")),
        }


RouterBuilder = Callable[[ModelRegistry, StaticAvailability], RuleBasedRouter]


async def compare(
    tasks: Sequence[RoutingTask],
    fleet: Fleet,
    builders: dict[str, RouterBuilder],
) -> list[ComparisonReport]:
    reports = []
    for name, build in builders.items():
        reference = build(
            ModelRegistry(fleet.models), StaticAvailability(m.name for m in fleet.models)
        )
        for scenario in fleet.scenarios:
            available = fleet.available(scenario)
            router = build(
                ModelRegistry(fleet.models), StaticAvailability(m.name for m in available)
            )
            results = [await _route_one(router, reference, task, available) for task in tasks]
            guard = getattr(router, "guard_interventions", 0)
            reports.append(ComparisonReport(name, scenario.name, results, guard))
    return reports


async def _route_one(
    router: RuleBasedRouter,
    reference: RuleBasedRouter,
    task: RoutingTask,
    available: Sequence[ModelMetadata],
) -> RouterResult:
    scored = {
        m.name: utility(simulate_outcome(task, m))
        for m in available
        if not missing_capabilities(m, task)
    }
    best = max(scored.values()) if scored else None
    oracle = sorted(n for n, u in scored.items() if best is not None and u >= best - 1e-9)
    try:
        decision = await router.route(task.to_request())
    except NoModelAvailableError:
        return RouterResult(task.id, None, "-", None, None, best, oracle, [])
    try:
        ref = (await reference.route(task.to_request())).model.name
    except NoModelAvailableError:
        ref = None
    model = decision.model
    outcome = simulate_outcome(task, model)
    return RouterResult(
        task_id=task.id,
        selected=model.name,
        ranker=decision.ranker,
        outcome=outcome,
        utility=utility(outcome),
        best_utility=best,
        oracle=oracle,
        critical=missing_capabilities(model, task),
        reference=ref,
        notes=[n for n in decision.notes if "verworfen" in n or "nicht nutzbar" in n],
    )


def rule_builder(registry: ModelRegistry, availability: StaticAvailability) -> RuleBasedRouter:
    return RuleBasedRouter(registry, availability)


def learned_builder(ranker: LearnedRanker) -> RouterBuilder:
    def build(registry: ModelRegistry, availability: StaticAvailability) -> RuleBasedRouter:
        return LearnedRouter(registry, availability, ranker, on_ranker_error="raise")

    return build
