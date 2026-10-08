"""Learned Router: lernt die Rangfolge – aber nur zwischen Modellen, die die harten Regeln erlauben.

Ablauf (gleiche Stufen wie :class:`~router.rule_router.RuleBasedRouter`)::

    Aufgabe → Klassifikation → HARTE REGELN (Verfügbarkeit, Speicher, Kontext, Vision,
    Coding/Reasoning, Tool-Calling) → gültige Kandidaten → LEARNED RANKING → Wächter
    (erneute Prüfung jedes Vorschlags) → Entscheidung → Verifikation → Ergebnis → Routing-Log

* Der Ranker sieht **ausschließlich** die Kandidaten aus der Hard-Constraint-Stufe.
* Jeder Vorschlag wird vor der Auswahl erneut gegen die harten Regeln geprüft. Ungültige
  Vorschläge werden verworfen und **im Routing-Log vermerkt** (kein stiller Austausch).
* Kein gültiger Kandidat → :class:`~router.base.NoModelAvailableError` mit allen Gründen.
* Fällt das Ranking selbst aus (kein Modell geladen, ungültige Scores), entscheidet
  ``on_ranker_error``: ``"raise"`` (Fehler) oder ``"rules"`` (Regelrangfolge, mit Vermerk
  ``ranker = "rules (learned unavailable: …)"`` in Entscheidung und Log).

:class:`LearnedRanker` bündelt ``train()``, ``evaluate()``, ``predict()``, ``save()``, ``load()``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from models.capabilities import ModelMetadata
from models.model_registry import ModelRegistry
from router.availability import ModelAvailability
from router.base import RoutingRequest, TaskClassifier
from router.feature_extractor import FEATURE_VERSION, FeatureExtractor, HistoryStats, TaskView
from router.ranking_model import Example, RankingModel, TrainingConfig
from router.resources import ResourceBudget
from router.routing_log import RoutingLog
from router.rule_router import CandidateSet, Ranking, RuleBasedRouter
from router.training_data import OutcomeRecord, build_examples, history_from

RANKER_FORMAT = 1


class LearnedRoutingError(Exception):
    """Das gelernte Ranking ist nicht nutzbar (und ``on_ranker_error="raise"``)."""


@dataclass
class RankingMetrics:
    """Offline-Gütemaße des Rankers auf Ergebnisdaten (z. B. dem Test-Split)."""

    groups: int
    pair_accuracy: float
    """Anteil der Paare (Nutzen verschieden), die richtig geordnet werden."""
    top1_match: float
    """Anteil der Gruppen, in denen der Top-Kandidat den höchsten beobachteten Nutzen hat."""
    mean_selected_utility: float
    mean_best_utility: float

    def to_dict(self) -> dict[str, float]:
        return {k: round(v, 4) if isinstance(v, float) else v for k, v in vars(self).items()}


@dataclass
class LearnedRanker:
    model: RankingModel
    history: HistoryStats = field(default_factory=HistoryStats)
    extractor: FeatureExtractor = field(default_factory=FeatureExtractor)
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ Training

    @classmethod
    def train(
        cls,
        records: Sequence[OutcomeRecord],
        models: Mapping[str, ModelMetadata],
        *,
        config: TrainingConfig | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> LearnedRanker:
        """Trainiert auf ``records`` (nur Trainings-Split übergeben!)."""
        extractor = FeatureExtractor()
        examples, skipped = build_examples(records, models, extractor=extractor)
        if not examples:
            raise ValueError("Keine verwertbaren Trainingsbeispiele (unbekannte Modelle?)")
        ranking = RankingModel(names=extractor.names(), config=config or TrainingConfig())
        ranking.fit(examples)
        meta = dict(metadata or {})
        meta.update(
            {
                "skipped_unknown_model": skipped,
                "train_task_ids": sorted({r.task_id for r in records}),
                "feature_version": FEATURE_VERSION,
            }
        )
        return cls(model=ranking, history=history_from(records), extractor=extractor, metadata=meta)

    # ------------------------------------------------------------------ Vorhersage

    def predict(self, task: TaskView, candidates: Sequence[ModelMetadata]) -> dict[str, float]:
        """Score je Kandidat (höher = besser). Bewertet nur die übergebenen Modelle."""
        scores = {}
        for model in candidates:
            value = self.model.score(self.extractor.pair_features(model, task, self.history))
            if not math.isfinite(value):
                raise ValueError(f"Score für {model.name} ist nicht endlich")
            scores[model.name] = value
        return scores

    def evaluate(
        self, records: Sequence[OutcomeRecord], models: Mapping[str, ModelMetadata]
    ) -> RankingMetrics:
        """Offline-Auswertung auf Ergebnisdaten, die **nicht** im Training waren."""
        overlap = set(self.metadata.get("train_task_ids", [])) & {r.task_id for r in records}
        if overlap:
            raise ValueError(f"Data Leakage: {len(overlap)} Testaufgaben waren im Training")
        examples, _ = build_examples(
            records,
            models,
            extractor=self.extractor,
            history=self.history,
            leave_one_task_out=False,
        )
        groups: dict[str, list[Example]] = {}
        for e in examples:
            groups.setdefault(e.group, []).append(e)
        pairs = correct = top1 = 0
        selected_utility = best_utility = 0.0
        multi = [g for g in groups.values() if len(g) > 1]
        for group in multi:
            scored = [(self.model.score(e.features), e) for e in group]
            for s_a, a in scored:
                for s_b, b in scored:
                    if a.utility - b.utility > 1e-9:
                        pairs += 1
                        correct += s_a > s_b
            chosen = max(scored, key=lambda p: (p[0], p[1].model))[1]
            best = max(e.utility for e in group)
            top1 += chosen.utility >= best - 1e-9
            selected_utility += chosen.utility
            best_utility += best
        n = len(multi) or 1
        return RankingMetrics(
            groups=len(multi),
            pair_accuracy=correct / pairs if pairs else 0.0,
            top1_match=top1 / n,
            mean_selected_utility=selected_utility / n,
            mean_best_utility=best_utility / n,
        )

    # ------------------------------------------------------------------ Persistenz

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "format": RANKER_FORMAT,
            "feature_version": self.extractor.version,
            "model": self.model.to_dict(),
            "history": self.history.to_dict(),
            "metadata": self.metadata,
        }
        path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> LearnedRanker:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("format") != RANKER_FORMAT:
            raise ValueError(f"Unbekanntes Ranker-Format: {data.get('format')!r}")
        extractor = FeatureExtractor()
        if data.get("feature_version") != extractor.version:
            raise ValueError(
                f"Merkmalsversion {data.get('feature_version')} ≠ {extractor.version} – "
                "Ranker neu trainieren"
            )
        model = RankingModel.from_dict(data["model"])
        if model.names != extractor.names():
            raise ValueError("Merkmalsnamen passen nicht zum Extraktor – Ranker neu trainieren")
        return cls(
            model=model,
            history=HistoryStats.from_dict(data.get("history", {})),
            extractor=extractor,
            metadata=dict(data.get("metadata", {})),
        )


class LearnedRouter(RuleBasedRouter):
    """Harte Regeln aus :class:`RuleBasedRouter`, Rangfolge vom :class:`LearnedRanker`."""

    def __init__(
        self,
        registry: ModelRegistry,
        availability: ModelAvailability,
        ranker: LearnedRanker | None,
        *,
        classifier: TaskClassifier | None = None,
        resources: ResourceBudget | None = None,
        log: RoutingLog | None = None,
        max_fallbacks: int = 3,
        on_ranker_error: Literal["raise", "rules"] = "rules",
    ) -> None:
        super().__init__(
            registry,
            availability,
            classifier=classifier,
            resources=resources,
            log=log,
            max_fallbacks=max_fallbacks,
        )
        self.ranker = ranker
        self.on_ranker_error = on_ranker_error
        self.guard_interventions = 0
        """Anzahl verworfener ungültiger Vorschläge (für Auswertung/Monitoring)."""

    def rank(self, stage: CandidateSet, request: RoutingRequest) -> Ranking:
        rule_order = super().rank(stage, request).ranked
        try:
            if self.ranker is None:
                raise LearnedRoutingError("kein trainierter Ranker geladen")
            view = TaskView.from_routing(request, stage.classification)
            scores = self.ranker.predict(view, stage.candidates)
        except (LearnedRoutingError, ValueError, RuntimeError) as exc:
            if self.on_ranker_error == "raise":
                raise LearnedRoutingError(f"Learned Ranking nicht nutzbar: {exc}") from exc
            return Ranking(
                ranked=rule_order,
                ranker=f"rules (learned unavailable: {exc})",
                notes=[f"Learned Ranking nicht nutzbar ({exc}) – Regelrangfolge verwendet"],
            )
        return self._guarded(stage, request, scores, rule_order)

    def _guarded(
        self,
        stage: CandidateSet,
        request: RoutingRequest,
        scores: Mapping[str, float],
        rule_order: Sequence[ModelMetadata],
    ) -> Ranking:
        """Wächter: jeder Vorschlag wird gegen die harten Regeln geprüft, bevor er zählt."""
        notes: list[str] = []
        by_name = {m.name: m for m in self.registry.list_effective()}
        rule_pos = {m.name: i for i, m in enumerate(rule_order)}
        proposals = sorted(scores, key=lambda n: (-scores[n], rule_pos.get(n, len(rule_pos)), n))
        valid: list[ModelMetadata] = []
        for name in proposals:
            model = by_name.get(name)
            problems = ["unbekanntes Modell"] if model is None else self.is_valid(model, stage)
            if problems:
                self.guard_interventions += 1
                notes.append(f"Learned-Vorschlag {name} verworfen: {', '.join(problems)}")
                continue
            assert model is not None
            valid.append(model)
        unscored = [m for m in rule_order if m.name not in scores]
        if unscored:
            notes.append(
                "ohne Learned-Score (Regelreihenfolge angehängt): "
                + ", ".join(m.name for m in unscored)
            )
            valid += unscored
        if not valid:  # pragma: no cover – Kandidatenliste ist nie leer
            raise LearnedRoutingError("kein gültiger Kandidat nach Prüfung")
        selected = valid[0]
        c = stage.classification
        reason = (
            f"{c.reason}. Gewählt (Learned Ranking): Score {scores.get(selected.name, 0):.2f}"
            f" unter {len(stage.candidates)} gültigen Kandidaten nach harten Regeln"
        )
        if len(valid) > 1 and valid[1].name in scores:
            reason += f"; nächster: {valid[1].name} ({scores[valid[1].name]:.2f})"
        if rule_order and rule_order[0].name != selected.name:
            reason += f"; Regel-Router hätte {rule_order[0].name} gewählt"
        return Ranking(
            ranked=valid, ranker="learned", scores=dict(scores), notes=notes, reason=reason
        )
