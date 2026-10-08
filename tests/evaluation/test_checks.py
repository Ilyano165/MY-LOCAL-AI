"""Datei-, Behauptungs-, Recherche-, Mathe- und Text-Checks."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from evaluation import Aspect, CheckResult, CheckStatus, ToolEvent, VerificationContext
from evaluation.checks import (
    ClaimAuditCheck,
    ContradictionCheck,
    FileContentCheck,
    FileExistsCheck,
    IndependentCalculationCheck,
    MathError,
    NonEmptyAnswerCheck,
    RequirementCheck,
    SourceValidationCheck,
    check_requirement,
    detect_language,
    extract_claimed_value,
    extract_expression,
    extract_requirements,
    number_candidates,
    safe_eval,
)

P, F, U = CheckStatus.PASSED, CheckStatus.FAILED, CheckStatus.UNVERIFIABLE

# =========================================================================== FILES


async def test_file_exists(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x")
    ctx = VerificationContext(tmp_path)
    assert (await FileExistsCheck("a.txt").run(ctx)).status == P
    assert (await FileExistsCheck("b.txt").run(ctx)).status == F
    assert (await FileExistsCheck("b.txt", should_exist=False).run(ctx)).status == P
    assert (await FileExistsCheck("../x").run(ctx)).status == F
    assert FileExistsCheck("a.txt").aspect == Aspect.COMPLETENESS


@pytest.mark.parametrize(
    ("name", "content", "kwargs", "status"),
    [
        ("c.json", '{"port": 8080}', {"contains": ["8080"]}, P),
        ("c.json", '{"port": 8080,}', {}, F),  # ungültiges JSON
        ("c.toml", "port = 8080\n", {}, P),
        ("c.toml", "port = \n", {}, F),
        ("m.py", "def f(:\n", {}, F),
        ("n.md", "Hallo Welt", {"contains": ["Welt"], "not_contains": ["TODO"]}, P),
        ("n.md", "TODO: fertig machen", {"not_contains": ["TODO"]}, F),
        ("n.md", "Version: 1.2.3", {"pattern": r"Version: \d+\.\d+\.\d+"}, P),
        ("n.md", "abc", {"min_bytes": 10}, F),
        ("n.md", "abc", {"max_bytes": 2}, F),
        ("n.md", "beliebig", {}, U),  # nichts Prüfbares
    ],
)
async def test_file_content(
    tmp_path: Path, name: str, content: str, kwargs: dict[str, object], status: CheckStatus
) -> None:
    (tmp_path / name).write_text(content)
    result = await FileContentCheck(name, **kwargs).run(VerificationContext(tmp_path))  # type: ignore[arg-type]
    assert result.status == status, result.detail


async def test_file_content_missing_file(tmp_path: Path) -> None:
    result = await FileContentCheck("fehlt.json").run(VerificationContext(tmp_path))
    assert result.status == F


# =========================================================================== CLAIMS


def prior(name: str, status: CheckStatus) -> CheckResult:
    return CheckResult(name, "code", Aspect.CORRECTNESS, status, f"{name}: 1 fehlgeschlagen", True)


async def test_claim_audit_refutes_missing_file(tmp_path: Path) -> None:
    (tmp_path / "calc.py").write_text("x = 1")
    ctx = VerificationContext(
        tmp_path,
        claim="Ich habe calc.py und utils.py erstellt.",
        tool_events=[ToolEvent("write_file", {"path": "calc.py"}, True)],
    )
    result = await ClaimAuditCheck().run(ctx)
    assert result.status == F and "utils.py" in result.detail
    assert any("Schreibaufruf belegt" in c for c in result.evidence["confirmed"])


async def test_claim_audit_compares_test_claim_with_real_run(tmp_path: Path) -> None:
    claim = "Implementiert. Alle Tests bestanden."
    ctx = VerificationContext(tmp_path, claim=claim, results=[prior("unit_tests", F)])
    assert (await ClaimAuditCheck().run(ctx)).status == F
    ctx = VerificationContext(tmp_path, claim=claim, results=[prior("unit_tests", P)])
    assert (await ClaimAuditCheck().run(ctx)).status == P
    ctx = VerificationContext(tmp_path, claim=claim)
    assert (await ClaimAuditCheck().run(ctx)).status == U  # behauptet, aber nicht belegt
    failing_event = ToolEvent("execute_command", {"command": ["python", "-m", "pytest"]}, False)
    ctx = VerificationContext(tmp_path, claim=claim, tool_events=[failing_event])
    assert (await ClaimAuditCheck().run(ctx)).status == F


async def test_claim_audit_ignores_negated_and_empty_claims(tmp_path: Path) -> None:
    negated = VerificationContext(
        tmp_path, claim="Die Tests sind nicht bestanden. Ich habe x.py nicht erstellt."
    )
    assert (await ClaimAuditCheck().run(negated)).status == U
    assert (await ClaimAuditCheck().run(VerificationContext(tmp_path))).status == U
    nothing = VerificationContext(tmp_path, claim="Hier ist eine Erklärung ohne Behauptungen.")
    assert (await ClaimAuditCheck().run(nothing)).status == U


# =========================================================================== RESEARCH

SOURCE = "Die Stadt hat 52000 Einwohner. Der Fluss ist 120 km lang. Die Brücke ist nicht gesperrt."
URL = "https://example.org/stadt"


@pytest.mark.parametrize(
    ("claim", "sources", "status"),
    [
        (f"Laut {URL}: „Die Stadt hat 52000 Einwohner.“", {URL: SOURCE}, P),
        (f"Laut {URL}: „Die Stadt hat 99000 Einwohner und mehr.“", {URL: SOURCE}, F),
        ("Siehe https://erfunden.example.com/x.", {URL: SOURCE}, F),  # nicht im Material
        ("Siehe https://example.com/unbekannt.", {}, U),  # kein Netzwerk → nicht prüfbar
        ("Siehe https://localhost-ohne-punkt/x.", {}, F),  # ungültige URL
        ("Der Fluss ist 120 km lang.", {}, F),  # keine Quelle
    ],
)
async def test_source_validation(
    tmp_path: Path, claim: str, sources: dict[str, str], status: CheckStatus
) -> None:
    result = await SourceValidationCheck().run(
        VerificationContext(tmp_path, claim=claim, sources=sources)
    )
    assert result.status == status, result.detail


async def test_local_sources_and_quotes(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "notes.md").write_text("Der Router wählt das Modell nach Aufgabe.")
    ok = VerificationContext(
        tmp_path, claim="In `docs/notes.md`: „Der Router wählt das Modell nach Aufgabe.“"
    )
    assert (await SourceValidationCheck().run(ok)).status == P
    missing = VerificationContext(tmp_path, claim="Siehe `docs/fehlt.md`.")
    assert (await SourceValidationCheck().run(missing)).status == F
    optional = SourceValidationCheck(require_sources=False)
    assert (await optional.run(VerificationContext(tmp_path, claim="ohne"))).status == U


@pytest.mark.parametrize(
    ("claim", "sources", "status"),
    [
        ("Die Datei hat 120 Zeilen. Später: Die Datei hat 80 Zeilen.", {}, F),
        ("Der Server ist erreichbar. Der Server ist nicht erreichbar.", {}, F),
        ("Gemäß Quelle ist der Fluss 95 km lang.", {URL: SOURCE}, F),
        ("Die Brücke ist gesperrt.", {URL: SOURCE}, F),
        ("Die Stadt hat 52000 Einwohner. Der Fluss ist 120 km lang.", {URL: SOURCE}, P),
        ("Ein einzelner Satz.", {}, U),
    ],
)
async def test_contradictions(
    tmp_path: Path, claim: str, sources: dict[str, str], status: CheckStatus
) -> None:
    result = await ContradictionCheck().run(
        VerificationContext(tmp_path, claim=claim, sources=sources)
    )
    assert result.status == status, result.evidence
    assert result.can_confirm is False  # „kein Widerspruch“ bestätigt nichts


# =========================================================================== MATH


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("17*23", 391),
        ("0.1 + 0.2", Fraction(3, 10)),
        ("2**10", 1024),
        ("(3+4)**2/7", 7),
        ("7 // 2", 3),
        ("7 % 4", 3),
        ("-(2+3)", -5),
        ("factorial(5)", 120),
        ("abs(-3)", 3),
        ("max(1, 9, 4)", 9),
    ],
)
def test_safe_eval_exact(expr: str, expected: object) -> None:
    assert safe_eval(expr) == expected


def test_safe_eval_floats_and_constants() -> None:
    assert float(safe_eval("sqrt(2)")) == pytest.approx(1.41421356)
    assert float(safe_eval("2*pi")) == pytest.approx(6.2831853)


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('ls')",
        "open('x')",
        "x + 1",
        "().__class__",
        "lambda: 1",
        "[1, 2]",
        "1/0",
        "2**100000",
        "10**10**10",
        "factorial(100000)",
        "(-8)**(1/3)",
        "True + 1",
        "'a' * 3",
    ],
)
def test_safe_eval_rejects_unsafe_or_huge(expr: str) -> None:
    with pytest.raises(MathError):
        safe_eval(expr)


@pytest.mark.parametrize(
    ("task", "expected"),
    [
        ("Was ist 17 * 23?", "17 * 23"),
        ("Berechne 15% von 240.", "(15/100)*240"),
        ("Wie viel ist 3,5 × 2?", "3.5 * 2"),
        ("Was ist 12 : 4?", "12/4"),
        ("Wurzel(16) + 1?", "sqrt(16) + 1"),
        ("Wie heißt du?", None),
        ("Das Jahr 2026", None),
    ],
)
def test_extract_expression(task: str, expected: str | None) -> None:
    assert extract_expression(task) == expected


@pytest.mark.parametrize(
    ("raw", "values"),
    [
        ("391", {391}),
        ("3,5", {Fraction(7, 2)}),
        ("1.234", {Fraction(1234, 1000), 1234}),
        ("1.234,5", {Fraction(24690, 20)}),
        ("1,234.5", {Fraction(24690, 20)}),
    ],
)
def test_number_candidates(raw: str, values: set[object]) -> None:
    assert values <= set(number_candidates(raw))


def test_extract_claimed_value_prefers_marked_result() -> None:
    assert extract_claimed_value("17 * 23 = 391")[0] == "391"  # type: ignore[index]
    assert extract_claimed_value("Schritt 1 ergibt 20, das Ergebnis ist 42.")[0] == "42."  # type: ignore[index]
    assert extract_claimed_value("keine Zahl") is None


@pytest.mark.parametrize(
    ("task", "claim", "status"),
    [
        ("Was ist 17 * 23?", "17 * 23 = 391", P),
        ("Was ist 17 * 23?", "Das Ergebnis ist 381.", F),
        ("Was ist 0,1 + 0,2?", "0,3", P),
        ("Wie viel ist sqrt(2) * 3?", "ungefähr 4,243", P),
        ("Wie viel ist sqrt(2) * 3?", "ungefähr 4,25", F),  # falsch gerundet
        ("Was ist 1000 * 1,5?", "1.500", P),  # deutsche Tausender
        ("Was ist 17 * 23?", "Weiß ich nicht.", F),
        ("Ist P = NP?", "Nein.", U),
    ],
)
async def test_independent_calculation(
    tmp_path: Path, task: str, claim: str, status: CheckStatus
) -> None:
    result = await IndependentCalculationCheck().run(
        VerificationContext(tmp_path, task=task, claim=claim)
    )
    assert result.status == status, result.detail


async def test_calculation_ignores_the_agents_own_arithmetic(tmp_path: Path) -> None:
    # Der Agent „rechnet vor“ und behauptet ein falsches Ergebnis – zählt nur das Nachrechnen.
    ctx = VerificationContext(
        tmp_path, task="Was ist 12 * 12?", claim="12 * 12 = 12 * 10 + 12 * 2 = 140"
    )
    result = await IndependentCalculationCheck().run(ctx)
    assert result.status == F and result.evidence["expected"] == "144"


# =========================================================================== TEXT


@pytest.mark.parametrize(
    ("requirement", "answer", "status"),
    [
        ("maximal 5 Wörter", "Eins zwei drei vier.", P),
        ("maximal 5 Wörter", "Eins zwei drei vier fünf sechs.", F),
        ("mindestens 3 Sätze", "A. B. C.", P),
        ("genau 2 Sätze", "A. B. C.", F),
        ("höchstens 10 Zeichen", "kurz", P),
        ("als Liste", "- a\n- b", P),
        ("als Liste", "a und b", F),
        ("genau 3 Stichpunkte", "- a\n- b\n- c", P),
        ("auf Deutsch", "Das ist ein Test und es ist nicht schwer, aber auch nicht leicht.", P),
        ("auf Deutsch", "This is a test and it is not hard, but it is also not easy.", F),
        ("in English", "This is a test and it is not hard, but it is also not easy.", P),
        ("als JSON", '```json\n{"a": 1}\n```', P),
        ("als JSON", "a: 1", F),
        ("mit Überschrift", "# Titel\nText", P),
        ("Erwähne „Datenschutz“", "Wir achten auf Datenschutz.", P),
        ("Erwähne „Datenschutz“", "Wir achten auf Sicherheit.", F),
        ("Sei freundlich", "Hallo!", U),
    ],
)
def test_check_requirement(requirement: str, answer: str, status: CheckStatus) -> None:
    results = check_requirement(requirement, answer)
    assert [r.status for r in results] == [status], results


def test_language_detection_needs_enough_signal() -> None:
    assert detect_language("ok") is None
    assert detect_language("Der Hund und die Katze sind nicht im Haus.") == "de"


def test_extract_requirements() -> None:
    task = (
        "Schreibe eine Zusammenfassung.\n- maximal 50 Wörter\n- auf Deutsch\n"
        "Gib das Ergebnis als JSON zurück."
    )
    reqs = extract_requirements(task)
    assert "maximal 50 Wörter" in reqs and "auf Deutsch" in reqs
    assert "Gib das Ergebnis als JSON zurück." in reqs


async def test_requirement_check_aggregates(tmp_path: Path) -> None:
    ctx = VerificationContext(
        tmp_path,
        task="Antworte in maximal 5 Wörtern und als JSON.",
        claim='{"ok": true}',
        requirements=["Sei höflich"],
    )
    result = await RequirementCheck().run(ctx)
    assert result.status == P and result.evidence["unverifiable"] == 1
    empty = await RequirementCheck().run(VerificationContext(tmp_path, task="Mach was.", claim="x"))
    assert empty.status == U
    no_answer = await RequirementCheck().run(VerificationContext(tmp_path, task="maximal 5 Wörter"))
    assert no_answer.status == F


async def test_non_empty_answer_is_not_independent(tmp_path: Path) -> None:
    result = await NonEmptyAnswerCheck().run(VerificationContext(tmp_path, claim="Antwort"))
    assert result.status == P and not result.independent
    assert (await NonEmptyAnswerCheck().run(VerificationContext(tmp_path))).status == F


async def test_url_followed_by_punctuation(tmp_path: Path) -> None:
    for claim in (f"Laut {URL}: „Die Stadt hat 52000 Einwohner.“", f"Quelle: {URL}.", f"({URL})"):
        result = await SourceValidationCheck().run(
            VerificationContext(tmp_path, claim=claim, sources={URL: SOURCE})
        )
        assert result.status == P, (claim, result.detail)
