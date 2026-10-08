"""Trainingsdaten für den Learned Router.

Ein :class:`OutcomeRecord` ist ein beobachtetes Ergebnis: *Aufgabe → gewähltes Modell →
Verifikationsurteil, Quality Score, Latenz, Erfolg*. Quellen:

* ``routing_log`` – echte Läufe: :class:`~router.routing_log.JsonlRoutingLog` speichert die
  Entscheidung, der Agent trägt nach der Verifikation das Ergebnis nach (``joined()``).
* ``simulated`` – aus gelabelten Datensätzen abgeleitete Ergebnisse (siehe
  ``evaluation/learned_routing.py``); im Modell und in Berichten immer als solche gekennzeichnet.

**Data Leakage** wird an drei Stellen verhindert:

1. ``task_id`` ist die Leakage-Einheit (Hash des normalisierten Aufgabentexts bzw. die
   Datensatz-ID). :func:`split_by_task` teilt nach ``task_id`` – nie nach Einzelbeispiel.
2. Historische Statistiken (Erfolg/Qualität je Modell) werden nur aus Trainingsdaten gebildet
   und für Trainingsbeispiele *leave-one-task-out* berechnet (das eigene Ergebnis fließt nie
   in das eigene Merkmal ein).
3. Merkmale nutzen nur Laufzeitwissen (:mod:`router.feature_extractor`), keine Labels.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from models.capabilities import ModelMetadata
from router.feature_extractor import FeatureExtractor, HistoryStats, TaskView
from router.ranking_model import Example

LATENCY_CAP_S = 300.0
LATENCY_WEIGHT = 0.1
"""Nutzen = Qualität − 0.1 · min(Latenz, 300 s)/300: Latenz entscheidet bei gleicher Qualität,
überstimmt aber nie einen Qualitätsunterschied von mehr als 0.1."""


def task_key(text: str) -> str:
    """Leakage-Einheit für Log-Daten: gleicher Aufgabentext = gleiche Aufgabe."""
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    return "t-" + hashlib.sha256(normalized.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class OutcomeRecord:
    group: str
    """Vergleichsgruppe: eine Anfrage (Entscheidungs-ID bzw. Aufgabe × Szenario)."""
    task_id: str
    task: TaskView
    model: str
    success: bool
    quality: float | None = None
    latency_s: float | None = None
    verdict: str | None = None
    hard_failure: bool = False
    """Ungeeignetes Modell, Fehler, Abbruch – stärker als ein bloß schwaches Ergebnis."""
    source: str = "routing_log"

    def __post_init__(self) -> None:
        if self.quality is not None and not 0.0 <= self.quality <= 1.0:
            raise ValueError("quality muss in [0, 1] liegen")

    @property
    def utility(self) -> float:
        base = self.quality if self.quality is not None else (0.8 if self.success else 0.2)
        if not self.success:
            base = min(base, 0.5)
        if self.hard_failure:
            base = 0.0
        cost = 0.0
        if self.latency_s is not None:
            cost = LATENCY_WEIGHT * min(self.latency_s, LATENCY_CAP_S) / LATENCY_CAP_S
        return base - cost


# ---------------------------------------------------------------------- Quellen


def from_routing_log(entries: Iterable[Mapping[str, Any]]) -> tuple[list[OutcomeRecord], int]:
    """Echte Trainingsdaten aus ``JsonlRoutingLog.joined()``.

    Liefert (Datensätze, übersprungen). Entscheidungen ohne nachgetragenes Ergebnis zählen
    als übersprungen – ein fehlendes Ergebnis ist **kein** Misserfolg.
    """
    records: list[OutcomeRecord] = []
    skipped = 0
    for entry in entries:
        outcome = entry.get("outcome")
        if entry.get("type", "decision") != "decision" or not outcome:
            skipped += 1
            continue
        try:
            view = TaskView.from_log(entry)
        except (KeyError, ValueError):
            skipped += 1
            continue
        quality = outcome.get("quality")
        latency_ms = outcome.get("latency_ms")
        records.append(
            OutcomeRecord(
                group=str(entry["id"]),
                task_id=task_key(view.text),
                task=view,
                model=str(entry["selected_model"]),
                success=bool(outcome.get("success")),
                quality=float(quality) if isinstance(quality, int | float) else None,
                latency_s=latency_ms / 1000 if isinstance(latency_ms, int | float) else None,
                verdict=outcome.get("verdict"),
                hard_failure=bool(outcome.get("error")),
                source="routing_log",
            )
        )
    return records, skipped


# ---------------------------------------------------------------------- Split


def split_by_task(
    records: Sequence[OutcomeRecord],
    *,
    test_task_ids: Iterable[str] | None = None,
    test_fraction: float = 0.3,
    salt: str = "nova-routing-split-v1",
) -> tuple[list[OutcomeRecord], list[OutcomeRecord]]:
    """Fester Train/Test-Split nach Aufgabe (keine Aufgabe in beiden Teilen).

    Mit ``test_task_ids`` wird ein vorgegebener Split verwendet (z. B. aus einer Datei),
    sonst entscheidet ein gesalzener Hash der ``task_id`` (deterministisch, stabil).
    """
    if test_task_ids is not None:
        test_ids = set(test_task_ids)
    else:
        if not 0 < test_fraction < 1:
            raise ValueError("test_fraction muss in (0, 1) liegen")
        ids = sorted(
            {r.task_id for r in records},
            key=lambda i: hashlib.sha256(f"{salt}:{i}".encode()).hexdigest(),
        )
        test_ids = set(ids[: max(1, round(len(ids) * test_fraction))])
    train = [r for r in records if r.task_id not in test_ids]
    test = [r for r in records if r.task_id in test_ids]
    assert not ({r.task_id for r in train} & {r.task_id for r in test})
    return train, test


# ---------------------------------------------------------------------- Beispiele


def history_from(records: Iterable[OutcomeRecord]) -> HistoryStats:
    stats = HistoryStats()
    for r in records:
        stats.add(r.model, r.task.content, r.success, r.quality, r.hard_failure)
    return stats


def build_examples(
    records: Sequence[OutcomeRecord],
    models: Mapping[str, ModelMetadata],
    *,
    extractor: FeatureExtractor | None = None,
    history: HistoryStats | None = None,
    leave_one_task_out: bool = True,
) -> tuple[list[Example], int]:
    """Merkmalsvektoren für Datensätze; (Beispiele, übersprungen wegen unbekanntem Modell).

    Für Trainingsdaten ``history=None`` → Historie aus ``records`` selbst, mit
    leave-one-task-out. Für Testdaten die Trainingshistorie übergeben und
    ``leave_one_task_out=False``.
    """
    extractor = extractor or FeatureExtractor()
    own_history = history is None
    stats = history_from(records) if history is None else history
    by_task: dict[str, list[OutcomeRecord]] = {}
    for r in records:
        by_task.setdefault(r.task_id, []).append(r)
    examples: list[Example] = []
    skipped = 0
    for task_id, task_records in by_task.items():
        if own_history and leave_one_task_out:
            for r in task_records:
                stats.add(r.model, r.task.content, r.success, r.quality, r.hard_failure, sign=-1)
        for r in task_records:
            model = models.get(r.model)
            if model is None:
                skipped += 1
                continue
            examples.append(
                Example(
                    group=r.group,
                    task_id=task_id,
                    model=r.model,
                    features=extractor.pair_features(model, r.task, stats),
                    utility=r.utility,
                    success=r.success,
                    source=r.source,
                )
            )
        if own_history and leave_one_task_out:
            for r in task_records:
                stats.add(r.model, r.task.content, r.success, r.quality, r.hard_failure)
    return examples, skipped
