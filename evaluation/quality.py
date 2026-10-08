"""Quality Score: internes, heuristisches Qualitätsmaß für ein Verifikationsergebnis.

WICHTIG: Der Score ist **keine Wahrscheinlichkeit** und keine kalibrierte Aussage darüber,
wie sicher ein Ergebnis korrekt ist. Er fasst nachvollziehbar zusammen, welche Prüfungen
bestanden, fehlgeschlagen oder offen sind. Werte verschiedener Aufgabenarten sind nur
eingeschränkt vergleichbar. Eine Kalibrierung gegen menschliche Bewertungen steht aus
(siehe docs/evaluation.md).

Dimensionen (jeweils 0..1, ``None`` = nicht bewertbar):

* **correctness**  – gewichteter Anteil bestandener Korrektheits-Checks unter den ausgeführten
* **completeness** – gewichteter Anteil bestandener Vollständigkeits-Checks (z. B. Dateien da)
* **requirements** – Anteil erfüllter Anforderungen unter den prüfbaren
* **errors**       – 1.0 = keine Fehler; sinkt mit fehlgeschlagenen Checks, Check-Abstürzen und
                     vom Agenten gemeldeten Fehlern
* **confidence**   – wie stark die Bewertung auf *unabhängiger* Evidenz beruht (nicht: wie
                     sicher das Ergebnis stimmt)

``overall`` kombiniert die verfügbaren Dimensionen und wird mit geringer Konfidenz gedämpft.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from evaluation.verifier import (
    Aspect,
    CheckResult,
    CheckStatus,
    Verdict,
    VerificationContext,
    VerificationReport,
)

DISCLAIMER = (
    "Internes, heuristisches Qualitätsmaß – keine Wahrscheinlichkeit und keine kalibrierte "
    "Aussage über die Korrektheit."
)


@dataclass(frozen=True)
class QualityWeights:
    correctness: float = 0.4
    completeness: float = 0.2
    requirements: float = 0.2
    errors: float = 0.2


@dataclass(frozen=True)
class QualityScore:
    correctness: float | None
    completeness: float | None
    requirements: float | None
    errors: float
    confidence: float
    overall: float | None
    grade: str
    notes: list[str] = field(default_factory=list)

    def describe(self) -> str:
        def fmt(v: float | None) -> str:
            return "–" if v is None else f"{v:.2f}"

        return (
            "Qualität (intern, keine Wahrscheinlichkeit): "
            f"gesamt={fmt(self.overall)} [{self.grade}] "
            f"korrektheit={fmt(self.correctness)} vollständigkeit={fmt(self.completeness)} "
            f"anforderungen={fmt(self.requirements)} fehler={fmt(self.errors)} "
            f"konfidenz={fmt(self.confidence)}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "correctness": self.correctness,
            "completeness": self.completeness,
            "requirements": self.requirements,
            "errors": self.errors,
            "confidence": self.confidence,
            "overall": self.overall,
            "grade": self.grade,
            "notes": self.notes,
            "disclaimer": DISCLAIMER,
        }


def _ratio(results: list[CheckResult]) -> float | None:
    executed = [r for r in results if r.executed]
    total = sum(r.weight for r in executed)
    if total == 0:
        return None
    return sum(r.weight for r in executed if r.status == CheckStatus.PASSED) / total


class QualityScorer:
    def __init__(
        self, weights: QualityWeights | None = None, *, failed_overall_cap: float = 0.49
    ) -> None:
        self.weights = weights or QualityWeights()
        self.failed_overall_cap = failed_overall_cap

    def score(
        self, report: VerificationReport, ctx: VerificationContext | None = None
    ) -> QualityScore:
        results = report.results
        notes: list[str] = []
        correctness = _ratio([r for r in results if r.aspect == Aspect.CORRECTNESS])
        completeness = _ratio([r for r in results if r.aspect == Aspect.COMPLETENESS])
        requirements = self._requirements(results)

        failed = sum(r.weight for r in results if r.status == CheckStatus.FAILED)
        crashed = sum(1 for r in results if r.status == CheckStatus.ERROR)
        reported = ctx.reported_errors if ctx is not None else 0
        errors = 1.0 / (1.0 + failed + 0.5 * crashed + 0.25 * reported)
        if reported:
            notes.append(f"{reported} während der Ausführung gemeldete Fehler berücksichtigt")

        total_weight = sum(r.weight for r in results)
        independent = sum(r.weight for r in results if r.independent and r.executed)
        confidence = independent / total_weight if total_weight else 0.0
        open_ = [
            r.check
            for r in results
            if r.status in (CheckStatus.UNVERIFIABLE, CheckStatus.SKIPPED, CheckStatus.ERROR)
        ]
        if open_:
            notes.append(f"nicht geprüft: {', '.join(open_)}")

        parts = [
            (v, w)
            for v, w in (
                (correctness, self.weights.correctness),
                (completeness, self.weights.completeness),
                (requirements, self.weights.requirements),
                (errors, self.weights.errors),
            )
            if v is not None
        ]
        overall: float | None = None
        if correctness is not None or completeness is not None or requirements is not None:
            raw = sum(v * w for v, w in parts) / sum(w for _, w in parts)
            overall = raw * (0.5 + 0.5 * confidence)  # wenig Evidenz → gedämpft
            if report.verdict == Verdict.FAILED:
                overall = min(overall, self.failed_overall_cap)
                notes.append("Verifikation fehlgeschlagen – Gesamtwert begrenzt")
        grade = self._grade(overall, confidence, report.verdict)
        return QualityScore(
            correctness=_round(correctness),
            completeness=_round(completeness),
            requirements=_round(requirements),
            errors=round(errors, 3),
            confidence=round(confidence, 3),
            overall=_round(overall),
            grade=grade,
            notes=notes,
        )

    @staticmethod
    def _requirements(results: list[CheckResult]) -> float | None:
        items = [
            i for r in results for i in r.evidence.get("requirements", []) if isinstance(i, dict)
        ]
        checkable = [i for i in items if i.get("status") in ("passed", "failed")]
        if not checkable:
            return None
        return sum(1 for i in checkable if i["status"] == "passed") / len(checkable)

    @staticmethod
    def _grade(overall: float | None, confidence: float, verdict: Verdict) -> str:
        if verdict == Verdict.FAILED:
            return "failed"
        if overall is None or confidence < 0.3:
            return "insufficient_evidence"
        if overall >= 0.8:
            return "good"
        if overall >= 0.6:
            return "acceptable"
        return "poor"


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 3)
