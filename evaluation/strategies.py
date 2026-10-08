"""Standard-Verifikationsstrategien je Aufgabenart.

| Strategie | Checks |
|---|---|
| ``code``     | Syntax · Typprüfung · Unit-Tests · Integrationstests · Behauptungsabgleich |
| ``files``    | Existenz · Inhaltsvalidierung (Format, erwartete Inhalte) · Behauptungsabgleich |
| ``research`` | Quellenvalidierung · Widerspruchserkennung · Anforderungen |
| ``math``     | unabhängige Nachrechnung · Anforderungen |
| ``text``     | Anforderungsprüfung · nicht-leere Antwort |

Strategieparameter kommen aus ``VerificationContext.params`` (z. B. ``expected_files``,
``file_expectations``, ``expression``, ``require_sources``).
"""

from __future__ import annotations

from evaluation.checks import (
    ClaimAuditCheck,
    ContradictionCheck,
    FileContentCheck,
    FileExistsCheck,
    IndependentCalculationCheck,
    NonEmptyAnswerCheck,
    RequirementCheck,
    SourceValidationCheck,
)
from evaluation.test_runner import IntegrationTestCheck, PythonSyntaxCheck, TypeCheck, UnitTestCheck
from evaluation.verifier import Check, VerificationContext, VerificationStrategy


def _code(ctx: VerificationContext) -> list[Check]:
    checks: list[Check] = [PythonSyntaxCheck(), TypeCheck(), UnitTestCheck()]
    if ctx.integration_command or ctx.params.get("integration_tests"):
        checks.append(IntegrationTestCheck())
    checks.append(ClaimAuditCheck())
    return checks


def _files(ctx: VerificationContext) -> list[Check]:
    expected = list(dict.fromkeys([*ctx.params.get("expected_files", []), *ctx.artifacts]))
    expectations: dict[str, dict[str, object]] = dict(ctx.params.get("file_expectations", {}))
    checks: list[Check] = [FileExistsCheck(p) for p in expected]
    for path in dict.fromkeys([*expected, *expectations]):
        spec = expectations.get(path, {})
        checks.append(
            FileContentCheck(
                path,
                contains=list(spec.get("contains", [])),  # type: ignore[call-overload]
                not_contains=list(spec.get("not_contains", [])),  # type: ignore[call-overload]
                pattern=spec.get("pattern"),  # type: ignore[arg-type]
                min_bytes=spec.get("min_bytes"),  # type: ignore[arg-type]
                max_bytes=spec.get("max_bytes"),  # type: ignore[arg-type]
            )
        )
    checks.append(ClaimAuditCheck())
    return checks


def _research(ctx: VerificationContext) -> list[Check]:
    return [
        SourceValidationCheck(require_sources=bool(ctx.params.get("require_sources", True))),
        ContradictionCheck(),
        RequirementCheck(),
    ]


def _math(ctx: VerificationContext) -> list[Check]:
    return [IndependentCalculationCheck(), RequirementCheck()]


def _text(ctx: VerificationContext) -> list[Check]:
    return [NonEmptyAnswerCheck(), RequirementCheck(), ContradictionCheck()]


def default_strategies() -> list[VerificationStrategy]:
    return [
        VerificationStrategy("code", "Syntax, Typen, Tests, Behauptungsabgleich", _code),
        VerificationStrategy("files", "Existenz und Inhalt von Dateien", _files),
        VerificationStrategy("research", "Quellen und Widersprüche", _research),
        VerificationStrategy("math", "Unabhängige Nachrechnung", _math),
        VerificationStrategy("text", "Anforderungsprüfung", _text),
    ]
