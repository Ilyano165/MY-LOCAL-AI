"""Quality Score: nachvollziehbares internes Maß – ausdrücklich keine Wahrscheinlichkeit."""

from __future__ import annotations

import pytest

from evaluation import (
    DISCLAIMER,
    Aspect,
    CheckResult,
    CheckStatus,
    QualityScorer,
    Verdict,
    VerificationReport,
    decide,
)
from evaluation.quality import QualityWeights
from evaluation.verifier import VerificationContext

P, F, U, E = CheckStatus.PASSED, CheckStatus.FAILED, CheckStatus.UNVERIFIABLE, CheckStatus.ERROR
C, CO, R = Aspect.CORRECTNESS, Aspect.COMPLETENESS, Aspect.REQUIREMENTS


def r(
    status: CheckStatus,
    aspect: Aspect = C,
    *,
    independent: bool = True,
    weight: float = 1.0,
    evidence: dict[str, object] | None = None,
) -> CheckResult:
    return CheckResult("c", "x", aspect, status, "", independent, weight, dict(evidence or {}))


def score(*results: CheckResult, reported_errors: int = 0):  # type: ignore[no-untyped-def]
    verdict, reasons = decide(results)
    report = VerificationReport("t", verdict, list(results), None, reasons)
    ctx = VerificationContext(
        workspace=__import__("pathlib").Path("."), reported_errors=reported_errors
    )
    return QualityScorer().score(report, ctx)


def test_disclaimer_is_explicit() -> None:
    q = score(r(P))
    assert "keine Wahrscheinlichkeit" in DISCLAIMER
    assert q.to_dict()["disclaimer"] == DISCLAIMER
    assert "keine Wahrscheinlichkeit" in q.describe()
    assert "probability" not in q.to_dict()  # bewusst kein solches Feld


def test_all_independent_checks_passed() -> None:
    q = score(r(P), r(P, CO))
    assert (q.correctness, q.completeness, q.errors, q.confidence) == (1.0, 1.0, 1.0, 1.0)
    assert q.overall == 1.0 and q.grade == "good"


def test_weighted_correctness_and_failed_cap() -> None:
    q = score(r(P, weight=3), r(F, weight=1))
    assert q.correctness == 0.75
    assert q.errors == 0.5  # ein fehlgeschlagener Check (Gewicht 1)
    assert q.overall is not None and q.overall <= 0.49 and q.grade == "failed"


def test_unverifiable_checks_lower_confidence_not_correctness() -> None:
    q = score(r(P), r(U), r(U), r(U))
    assert q.correctness == 1.0 and q.confidence == 0.25
    assert q.grade == "insufficient_evidence"
    assert q.overall is not None and q.overall < 0.8  # gedämpft
    assert any("nicht geprüft" in n for n in q.notes)


def test_dependent_checks_do_not_raise_confidence() -> None:
    q = score(r(P, CO, independent=False))
    assert q.confidence == 0.0 and q.grade == "insufficient_evidence"


def test_requirements_dimension_ignores_unverifiable_items() -> None:
    items = [{"status": "passed"}, {"status": "failed"}, {"status": "unverifiable"}]
    q = score(r(F, R, evidence={"requirements": items}))
    assert q.requirements == 0.5


def test_errors_dimension_counts_crashes_and_reported_errors() -> None:
    assert score(r(P), r(E)).errors == pytest.approx(1 / 1.5, abs=1e-3)
    q = score(r(P), reported_errors=4)
    assert q.errors == 0.5 and any("gemeldete Fehler" in n for n in q.notes)


def test_nothing_measurable() -> None:
    q = score(r(U))
    assert q.overall is None and q.correctness is None and q.grade == "insufficient_evidence"


def test_custom_weights() -> None:
    scorer = QualityScorer(QualityWeights(correctness=1, completeness=0, requirements=0, errors=0))
    results = [r(P), r(F, CO)]
    verdict, reasons = decide(results)
    report = VerificationReport("t", verdict, results, None, reasons)
    q = scorer.score(report)
    assert q.correctness == 1.0 and q.overall is not None
    assert q.overall <= 0.49  # Gesamturteil FAILED begrenzt trotz perfekter Korrektheit
    assert report.verdict == Verdict.FAILED
