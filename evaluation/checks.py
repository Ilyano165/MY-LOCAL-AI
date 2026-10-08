"""Checks für Dateien, Recherche, Mathematik, Text und Behauptungen des Agenten.

Alle Checks erzeugen eigene Evidenz (Dateisystem, Nachrechnen, Abgleich mit Quelltexten,
Zählen). Heuristische Checks (Widerspruchserkennung, Anforderungsextraktion) sind als solche
gekennzeichnet und haben geringeres Gewicht.
"""

from __future__ import annotations

import ast
import json
import math
import operator
import re
import tomllib
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from evaluation.verifier import Aspect, Check, CheckResult, CheckStatus, VerificationContext

# --------------------------------------------------------------------------- Text-Hilfen

_WORD_RE = re.compile(r"[\wäöüß]+", re.IGNORECASE)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
NEGATIONS = frozenset(
    {
        "nicht",
        "kein",
        "keine",
        "keinen",
        "keiner",
        "keinem",
        "nie",
        "niemals",
        "not",
        "no",
        "never",
        "none",
        "without",
        "ohne",
        "isn't",
        "doesn't",
        "nichts",
    }
)
_DE_STOP = frozenset(
    [
        "der",
        "die",
        "das",
        "und",
        "ist",
        "nicht",
        "ein",
        "eine",
        "zu",
        "mit",
        "von",
        "für",
        "auf",
        "sich",
        "dem",
        "den",
        "des",
        "im",
        "auch",
        "es",
        "wir",
        "ich",
        "sie",
        "aber",
        "als",
        "wenn",
        "wie",
        "oder",
        "noch",
        "nur",
    ]
)
_EN_STOP = frozenset(
    [
        "the",
        "and",
        "is",
        "not",
        "a",
        "an",
        "to",
        "with",
        "of",
        "for",
        "on",
        "it",
        "we",
        "i",
        "they",
        "but",
        "as",
        "if",
        "how",
        "or",
        "also",
        "only",
        "this",
        "that",
        "are",
        "was",
        "be",
    ]
)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[„“”\"'‚‘’«»]", "", text)
    return " ".join(text.split())


def words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text.strip()) if s.strip()]


def _resolve(ctx: VerificationContext, path: str) -> Path | None:
    root = ctx.workspace.resolve()
    candidate = (root / path).resolve()
    return candidate if candidate.is_relative_to(root) else None


# =========================================================================== FILES


class FileExistsCheck(Check):
    name = "file_exists"
    category = "files"
    aspect = Aspect.COMPLETENESS

    def __init__(self, path: str, *, should_exist: bool = True) -> None:
        self.path = path
        self.should_exist = should_exist
        self.name = f"file_exists({path})"

    async def run(self, ctx: VerificationContext) -> CheckResult:
        target = _resolve(ctx, self.path)
        if target is None:
            return self.failed(f"{self.path} liegt außerhalb des Workspace")
        exists = target.exists()
        if exists == self.should_exist:
            return self.passed(f"{self.path} {'existiert' if exists else 'existiert nicht'}")
        return self.failed(
            f"{self.path} {'fehlt' if self.should_exist else 'existiert unerwartet'}"
        )


class FileContentCheck(Check):
    """Inhaltsprüfung: Pflichttexte, verbotene Texte, Regex, Formatgültigkeit, Größe."""

    name = "file_content"
    category = "files"

    def __init__(
        self,
        path: str,
        *,
        contains: Sequence[str] = (),
        not_contains: Sequence[str] = (),
        pattern: str | None = None,
        validate_format: bool = True,
        min_bytes: int | None = None,
        max_bytes: int | None = None,
    ) -> None:
        self.path = path
        self.contains = list(contains)
        self.not_contains = list(not_contains)
        self.pattern = pattern
        self.validate_format = validate_format
        self.min_bytes = min_bytes
        self.max_bytes = max_bytes
        self.name = f"file_content({path})"

    async def run(self, ctx: VerificationContext) -> CheckResult:
        target = _resolve(ctx, self.path)
        if target is None or not target.is_file():
            return self.failed(f"{self.path} nicht vorhanden")
        data = target.read_bytes()
        problems: list[str] = []
        checked: list[str] = []
        if self.min_bytes is not None:
            checked.append("min_bytes")
            if len(data) < self.min_bytes:
                problems.append(f"zu klein ({len(data)} < {self.min_bytes} Bytes)")
        if self.max_bytes is not None:
            checked.append("max_bytes")
            if len(data) > self.max_bytes:
                problems.append(f"zu groß ({len(data)} > {self.max_bytes} Bytes)")
        text = data.decode("utf-8", errors="replace")
        for needle in self.contains:
            checked.append(f"contains:{needle}")
            if needle not in text:
                problems.append(f"enthält {needle!r} nicht")
        for needle in self.not_contains:
            checked.append(f"not_contains:{needle}")
            if needle in text:
                problems.append(f"enthält unerwünscht {needle!r}")
        if self.pattern is not None:
            checked.append("pattern")
            if not re.search(self.pattern, text, re.MULTILINE):
                problems.append(f"Muster {self.pattern!r} nicht gefunden")
        if self.validate_format:
            fmt_error = validate_format(target.suffix.lower(), text)
            if fmt_error is not None:
                checked.append("format")
                if fmt_error:
                    problems.append(fmt_error)
        if not checked:
            return self.unverifiable(f"keine prüfbaren Inhaltsanforderungen für {self.path}")
        if problems:
            return self.failed(f"{self.path}: " + "; ".join(problems), checked=checked)
        return self.passed(
            f"{self.path}: {len(checked)} Inhaltsprüfung(en) bestanden", checked=checked
        )


def validate_format(suffix: str, text: str) -> str | None:
    """``None`` = Format nicht prüfbar, ``""`` = gültig, sonst Fehlermeldung."""
    try:
        if suffix == ".json":
            json.loads(text)
        elif suffix == ".toml":
            tomllib.loads(text)
        elif suffix == ".py":
            ast.parse(text)
        else:
            return None
    except (ValueError, SyntaxError) as exc:
        return f"ungültiges {suffix[1:].upper()}: {str(exc).splitlines()[0][:200]}"
    return ""


# =========================================================================== CLAIMS


_FILE_CLAIM = re.compile(
    r"(erstellt|angelegt|geschrieben|gespeichert|geändert|aktualisiert|created|wrote|written|"
    r"saved|updated|modified|added)",
    re.IGNORECASE,
)
_FILE_TOKEN = re.compile(r"`?([\w./-]+\.[A-Za-z0-9]{1,6})`?")
_TEST_CLAIM = re.compile(r"\b(tests?|pytest|testsuite)\b", re.IGNORECASE)
_SUCCESS_WORD = re.compile(
    r"\b(bestanden|grün|erfolgreich|passed|pass|passing|green|succeed\w*)\b", re.IGNORECASE
)


class ClaimAuditCheck(Check):
    """Gleicht konkrete Behauptungen des Agenten mit der Realität ab.

    * „Datei X erstellt/geändert“ → existiert X? Gibt es einen erfolgreichen Schreibaufruf?
    * „Tests bestanden“ → stimmt das mit einem tatsächlichen Testlauf überein?
    Widerlegte Behauptungen lassen die Verifikation scheitern.
    """

    name = "claim_audit"
    category = "claims"

    async def run(self, ctx: VerificationContext) -> CheckResult:
        if not ctx.claim.strip():
            return self.unverifiable("keine Behauptung zu prüfen")
        refuted: list[str] = []
        confirmed: list[str] = []
        unconfirmed: list[str] = []
        written = {
            str(e.arguments.get("path"))
            for e in ctx.tool_events
            if e.success and e.tool in ("write_file", "edit_file")
        }
        for sentence in sentences(ctx.claim):
            negated = bool(set(words(sentence)) & NEGATIONS)
            if _FILE_CLAIM.search(sentence) and not negated:
                for token in _FILE_TOKEN.findall(sentence):
                    if "://" in token or token.replace(".", "").isdigit():
                        continue
                    target = _resolve(ctx, token)
                    if target is None or not target.exists():
                        refuted.append(f"Datei {token} behauptet, existiert aber nicht")
                    elif token in written:
                        confirmed.append(f"Datei {token}: existiert, Schreibaufruf belegt")
                    else:
                        confirmed.append(f"Datei {token}: existiert")
            if _TEST_CLAIM.search(sentence) and _SUCCESS_WORD.search(sentence) and not negated:
                test_result = ctx.result("unit_tests") or ctx.result("integration_tests")
                if test_result is not None and test_result.executed:
                    if test_result.status == CheckStatus.FAILED:
                        refuted.append(
                            f"„Tests bestanden“ behauptet, tatsächlich: {test_result.detail}"
                        )
                    else:
                        confirmed.append("Testbehauptung durch eigenen Testlauf bestätigt")
                elif self._test_event_failed(ctx):
                    refuted.append(
                        "„Tests bestanden“ behauptet, letzter Testlauf des Agenten schlug fehl"
                    )
                else:
                    unconfirmed.append(
                        "„Tests bestanden“ behauptet, aber kein unabhängiger Testlauf"
                    )
        evidence = {"refuted": refuted, "confirmed": confirmed, "unconfirmed": unconfirmed}
        if refuted:
            return self.failed("Behauptung widerlegt: " + "; ".join(refuted), **evidence)
        if confirmed and not unconfirmed:
            return self.passed(f"{len(confirmed)} Behauptung(en) bestätigt", **evidence)
        if confirmed or unconfirmed:
            return self.unverifiable(
                "nicht alle Behauptungen belegbar: " + "; ".join(unconfirmed), **evidence
            )
        return self.unverifiable("keine prüfbaren Behauptungen gefunden", **evidence)

    @staticmethod
    def _test_event_failed(ctx: VerificationContext) -> bool:
        runs = [
            e
            for e in ctx.tool_events
            if e.tool == "execute_command"
            and "pytest" in " ".join(map(str, e.arguments.get("command", [])))
        ]
        return bool(runs) and not runs[-1].success


# =========================================================================== RESEARCH


_URL = re.compile(r"https?://[^\s)\]>\"'„“”]+")
_QUOTE = re.compile(r"[„“\"]([^„“”\"]{12,400})[“”\"]")
_LOCAL_REF = re.compile(r"`([\w./-]+\.[A-Za-z0-9]{1,6})(?::\d+)?`")


class SourceValidationCheck(Check):
    """Prüft angegebene Quellen und wörtliche Zitate gegen tatsächlich verfügbare Quelltexte.

    * Keine Quelle angegeben → FAILED (wenn ``require_sources``).
    * Lokale Quelle (`pfad/datei.md`) → muss existieren.
    * URL → syntaktisch gültig; ihr Text muss über ``ctx.sources`` bereitgestellt sein, sonst
      nicht prüfbar (NOVA lädt im Kern nichts aus dem Netz).
    * Bei ``closed_world`` (Quellen bereitgestellt) → zitierte URLs außerhalb des Materials gelten
      als erfunden.
    * Wörtliche Zitate müssen in einem verfügbaren Quelltext vorkommen.
    """

    name = "source_validation"
    category = "research"

    def __init__(self, *, require_sources: bool = True, closed_world: bool | None = None) -> None:
        self.require_sources = require_sources
        self.closed_world = closed_world

    async def run(self, ctx: VerificationContext) -> CheckResult:
        claim = ctx.claim
        # Satzzeichen am Ende („…/stadt:“ vor einem Zitat) gehören nicht zur URL.
        urls = list(dict.fromkeys(u.rstrip(".,;:!?") for u in _URL.findall(claim)))
        local = list(dict.fromkeys(_LOCAL_REF.findall(claim)))
        quotes = [q.strip() for q in _QUOTE.findall(claim)]
        closed = self.closed_world if self.closed_world is not None else bool(ctx.sources)
        problems: list[str] = []
        verified: list[str] = []
        open_: list[str] = []
        texts: dict[str, str] = {}

        if not urls and not local and not any(k in claim for k in ctx.sources):
            if self.require_sources:
                return self.failed("keine Quellen angegeben")
            return self.unverifiable("keine Quellen angegeben")
        for url in urls:
            parts = urlsplit(url)
            if not parts.netloc or "." not in parts.netloc:
                problems.append(f"ungültige URL: {url}")
            elif url in ctx.sources:
                texts[url] = ctx.sources[url]
                verified.append(f"Quelle verfügbar: {url}")
            elif closed:
                problems.append(f"Quelle nicht im bereitgestellten Material (erfunden?): {url}")
            else:
                open_.append(f"Quelle nicht abrufbar (kein Netzwerkzugriff): {url}")
        for ref in local:
            target = _resolve(ctx, ref)
            if target is None or not target.is_file():
                problems.append(f"lokale Quelle existiert nicht: {ref}")
            else:
                texts[ref] = target.read_text(encoding="utf-8", errors="replace")
                verified.append(f"lokale Quelle vorhanden: {ref}")
        for key, text in ctx.sources.items():
            if key in claim and key not in texts:
                texts[key] = text
                verified.append(f"Quelle verfügbar: {key}")
        normalized_texts = [normalize(t) for t in texts.values()]
        for quote in quotes:
            if not normalized_texts:
                open_.append(f"Zitat nicht prüfbar (keine Quelltexte): „{quote[:60]}“")
            elif any(normalize(quote) in t for t in normalized_texts):
                verified.append(f"Zitat belegt: „{quote[:60]}“")
            else:
                problems.append(f"Zitat in keiner Quelle gefunden: „{quote[:80]}“")
        evidence = {
            "verified": verified,
            "problems": problems,
            "unverifiable": open_,
            "urls": urls,
            "local": local,
            "quotes": len(quotes),
        }
        if problems:
            return self.failed("; ".join(problems), **evidence)
        if verified and not open_:
            return self.passed(f"{len(verified)} Quellenprüfung(en) bestanden", **evidence)
        if verified:
            return self.passed(f"{len(verified)} bestanden, {len(open_)} nicht prüfbar", **evidence)
        return self.unverifiable("; ".join(open_), **evidence)


_FACT = re.compile(
    r"(?P<subject>[\wäöüß][\wäöüß\s-]{2,60}?)\s+(ist|sind|beträgt|betragen|hat|haben|kostet|kosten|"
    r"liegt bei|is|are|was|were|has|have|costs?|equals?)\s+"
    r"(ca\.\s*|etwa\s*|about\s*|approximately\s*)?"
    r"(?P<value>-?\d[\d.,]*)\s*(?P<unit>%|[a-zA-Zäöü€$]+)?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _Fact:
    subject: str
    value: float
    unit: str
    sentence: str


def _to_number(raw: str) -> float | None:
    text = raw.strip().rstrip(".,")
    if "," in text and "." in text:
        text = (
            text.replace(".", "").replace(",", ".")
            if text.rfind(",") > text.rfind(".")
            else text.replace(",", "")
        )
    elif "," in text:
        text = text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


# Deutsche Inversion nach vorangestelltem Satzteil: „Gemäß Quelle ist der Fluss 95 km lang“.
_FACT_INVERTED = re.compile(
    r"\b(ist|sind|beträgt|betragen|hat|haben|kostet|kosten)\s+(?P<subject>(?:[a-zäöüß-]+\s+){1,4}?)"
    r"(ca\.\s*|etwa\s*|rund\s*)?(?P<value>-?\d[\d.,]*)\s*(?P<unit>%|[a-zA-Zäöü€$]+)?",
    re.IGNORECASE,
)


def _facts(text: str) -> list[_Fact]:
    found = []
    for sentence in sentences(text):
        for m in [*_FACT.finditer(sentence), *_FACT_INVERTED.finditer(sentence)]:
            value = _to_number(m.group("value"))
            if value is None:
                continue
            subject_words = [w for w in words(m.group("subject")) if w not in _DE_STOP | _EN_STOP]
            if subject_words:
                found.append(
                    _Fact(
                        " ".join(subject_words[-3:]),
                        value,
                        (m.group("unit") or "").lower(),
                        sentence,
                    )
                )
    return found


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if sa | sb else 0.0


class ContradictionCheck(Check):
    """Heuristische Erkennung *möglicher* Widersprüche – innerhalb der Antwort und gegenüber
    bereitgestellten Quellen.

    * Zahlenwidersprüche: gleiches Subjekt, gleiche Einheit, unterschiedlicher Wert
      („Die Datei hat 120 Zeilen“ vs. „… hat 80 Zeilen“).
    * Negationswidersprüche: fast gleiche Aussage, einmal verneint, einmal nicht.

    Heuristik mit geringerem Gewicht; findet nicht jeden Widerspruch und kann Fehlalarme geben.
    """

    name = "contradiction_check"
    category = "research"
    weight = 0.5
    can_confirm = False  # findet Probleme, belegt aber keine Korrektheit

    def __init__(self, similarity: float = 0.6) -> None:
        self.similarity = similarity

    async def run(self, ctx: VerificationContext) -> CheckResult:
        if not ctx.claim.strip():
            return self.unverifiable("keine Antwort zu prüfen")
        claim_sentences = sentences(ctx.claim)
        source_sentences = [s for t in ctx.sources.values() for s in sentences(t)]
        conflicts: list[str] = []
        conflicts += self._numeric(_facts(ctx.claim), _facts(ctx.claim), internal=True)
        conflicts += self._numeric(
            _facts(ctx.claim), [f for t in ctx.sources.values() for f in _facts(t)], internal=False
        )
        conflicts += self._negation(claim_sentences, claim_sentences, internal=True)
        conflicts += self._negation(claim_sentences, source_sentences, internal=False)
        conflicts = list(dict.fromkeys(conflicts))
        evidence = {"conflicts": conflicts, "heuristic": True, "compared_sources": len(ctx.sources)}
        if conflicts:
            return self.failed(
                f"{len(conflicts)} möglicher Widerspruch/Widersprüche: "
                + " | ".join(conflicts[:3]),
                **evidence,
            )
        if len(claim_sentences) < 2 and not source_sentences:
            return self.unverifiable("zu wenig Text für eine Widerspruchsprüfung", **evidence)
        return self.passed("keine Widersprüche gefunden (heuristisch)", **evidence)

    @staticmethod
    def _numeric(left: Sequence[_Fact], right: Sequence[_Fact], *, internal: bool) -> list[str]:
        out = []
        for i, a in enumerate(left):
            for j, b in enumerate(right):
                if internal and j <= i:
                    continue
                if (
                    a.subject == b.subject
                    and a.unit == b.unit
                    and not math.isclose(a.value, b.value)
                ):
                    where = "in der Antwort" if internal else "gegenüber Quelle"
                    out.append(
                        f"{a.subject}: {a.value:g}{a.unit} vs. {b.value:g}{b.unit} ({where})"
                    )
        return out

    def _negation(self, left: Sequence[str], right: Sequence[str], *, internal: bool) -> list[str]:
        out = []
        for i, a in enumerate(left):
            wa = words(a)
            content_a = [w for w in wa if w not in NEGATIONS]
            for j, b in enumerate(right):
                if internal and j <= i:
                    continue
                wb = words(b)
                negation_differs = bool(set(wa) & NEGATIONS) != bool(set(wb) & NEGATIONS)
                if (
                    negation_differs
                    and _jaccard(content_a, [w for w in wb if w not in NEGATIONS])
                    >= self.similarity
                ):
                    where = "in der Antwort" if internal else "gegenüber Quelle"
                    out.append(f"„{a[:80]}“ vs. „{b[:80]}“ ({where})")
        return out


# =========================================================================== MATH


class MathError(ValueError):
    pass


_BINOPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_FUNCS: dict[str, Callable[..., Any]] = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "floor": math.floor,
    "ceil": math.ceil,
    "factorial": lambda n: math.factorial(_int_arg(n, 1000)),
}
_CONSTS = {"pi": math.pi, "e": math.e}


def _int_arg(value: Any, limit: int) -> int:
    if isinstance(value, Fraction) and value.denominator == 1:
        value = int(value)
    if not isinstance(value, int) or not 0 <= value <= limit:
        raise MathError(f"Argument muss eine ganze Zahl 0..{limit} sein")
    return value


def safe_eval(expression: str) -> Fraction | float:
    """Rechnet einen arithmetischen Ausdruck **ohne** ``eval`` aus.

    Exakt mit Brüchen, solange möglich (``0.1 + 0.2`` ergibt genau ``3/10``). Erlaubt:
    + - * / // % **, Klammern, Funktionen (sqrt, log, sin …), Konstanten pi und e.
    Schutz gegen Ressourcenmissbrauch: Exponent ≤ 10 000, Ergebnisgröße begrenzt.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise MathError(f"kein gültiger Ausdruck: {expression!r}") from exc

    def ev(node: ast.AST) -> Fraction | float:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, int | float)
            and not isinstance(node.value, bool)
        ):
            return (
                Fraction(str(node.value)) if isinstance(node.value, float) else Fraction(node.value)
            )
        if isinstance(node, ast.Name) and node.id in _CONSTS:
            return _CONSTS[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd | ast.USub):
            value = ev(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow):
                base, exponent = abs(float(left)), abs(float(right))
                if exponent > 10_000 or (base > 1 and exponent * math.log10(base + 1) > 10_000):
                    raise MathError("Potenz zu groß")
                if isinstance(right, Fraction) and right.denominator != 1:
                    # math.pow wirft bei negativer Basis statt ein komplexes Ergebnis zu liefern
                    return math.pow(float(left), float(right))
            if isinstance(node.op, ast.Div | ast.FloorDiv | ast.Mod) and right == 0:
                raise MathError("Division durch null")
            result = _BINOPS[type(node.op)](left, right)
            if isinstance(result, complex):
                raise MathError("komplexes Ergebnis wird nicht unterstützt")
            return result  # type: ignore[no-any-return]
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCS
            and not node.keywords
        ):
            args = [ev(a) for a in node.args]
            func = _FUNCS[node.func.id]
            if node.func.id in ("abs", "min", "max", "round", "factorial", "floor", "ceil"):
                result = func(*args)
            else:
                result = func(*[float(a) for a in args])
            return Fraction(result) if isinstance(result, int) else result
        raise MathError(f"nicht unterstützt: {ast.dump(node)[:80]}")

    try:
        return ev(tree)
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        if isinstance(exc, MathError):
            raise
        raise MathError(str(exc)) from exc


_NUMBER = re.compile(r"-?\d+(?:[.,]\d+)*")
_FUNC_NAMES = "|".join(sorted(_FUNCS, key=len, reverse=True))
_EXPR_TOKEN = rf"(?:{_FUNC_NAMES}|pi|[\d\s+\-*/().,^×÷·:%])"
_EXPR = re.compile(rf"(?:(?:{_FUNC_NAMES}|pi)|[\d(])(?:{_EXPR_TOKEN})*[\d)]", re.IGNORECASE)


def normalize_expression(text: str) -> str:
    expr = text.strip().rstrip("?=.").strip()
    expr = re.sub(r"(\d+(?:[.,]\d+)?)\s*%\s*(?:von|of)\s*", r"(\1/100)*", expr, flags=re.IGNORECASE)
    expr = expr.replace("×", "*").replace("·", "*").replace("÷", "/").replace("^", "**")
    expr = re.sub(r"(?<=[\d)])\s*:\s*(?=[\d(])", "/", expr)  # 12 : 4
    expr = re.sub(r"(?<=\d),(?=\d)", ".", expr)  # Dezimalkomma
    expr = re.sub(r"\bwurzel\s*\(", "sqrt(", expr, flags=re.IGNORECASE)
    return expr


def extract_expression(task: str) -> str | None:
    """Längster rechenbarer Ausdruck mit Operator oder Funktion im Aufgabentext."""
    text = re.sub(
        r"(\d+(?:[.,]\d+)?)\s*%\s*(?:von|of)\s*(\d)", r"(\1/100)*\2", task, flags=re.IGNORECASE
    )
    text = re.sub(r"\bwurzel\s*\(", "sqrt(", text, flags=re.IGNORECASE)
    best: str | None = None
    for m in _EXPR.finditer(text):
        candidate = normalize_expression(m.group(0))
        if not re.search(r"[+\-*/%]|\b(?:" + _FUNC_NAMES + r")\b", candidate):
            continue
        try:
            safe_eval(candidate)
        except MathError:
            continue
        if best is None or len(candidate) > len(best):
            best = candidate
    return best


_ANSWER_HINT = re.compile(
    r"(?:=|≈|\b(?:ergebnis(?:\s+ist)?|lautet|ist|beträgt|result(?:\s+is)?|answer(?:\s+is)?|is|equals)\b)"
    r"\s*:?\s*(-?\d[\d.,]*)",
    re.IGNORECASE,
)


def number_candidates(raw: str) -> list[Fraction]:
    """Mögliche Lesarten einer Zahl (deutsche/englische Tausender- und Dezimaltrenner)."""
    text = raw.strip().rstrip(".,")
    variants = {text.replace(",", ".")} if text.count(",") == 1 and "." not in text else set()
    if "," in text and "." in text:
        variants.add(
            text.replace(".", "").replace(",", ".")
            if text.rfind(",") > text.rfind(".")
            else text.replace(",", "")
        )
    if re.fullmatch(r"-?\d{1,3}(\.\d{3})+", text):
        variants.add(text.replace(".", ""))  # 1.234 (dt. Tausender)
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+", text):
        variants.add(text.replace(",", ""))  # 1,234 (engl. Tausender)
    variants.add(text)
    out = []
    for v in variants:
        try:
            out.append(Fraction(v))
        except ValueError:
            continue
    return out


def extract_claimed_value(claim: str) -> tuple[str, list[Fraction]] | None:
    hints = _ANSWER_HINT.findall(claim)
    raw = hints[-1] if hints else (_NUMBER.findall(claim) or [None])[-1]
    if raw is None:
        return None
    candidates = number_candidates(raw)
    return (raw, candidates) if candidates else None


def _decimals(raw: str) -> int | None:
    m = re.search(r"[.,](\d+)$", raw.strip())
    return len(m.group(1)) if m else 0


def values_match(expected: Fraction | float, claimed: Fraction, raw: str) -> bool:
    if isinstance(expected, Fraction) and expected == claimed:
        return True
    exp = float(expected)
    if math.isclose(exp, float(claimed), rel_tol=1e-9, abs_tol=1e-12):
        return True
    # Gerundete Angabe („≈ 1,414“): innerhalb der halben letzten Stelle
    decimals = _decimals(raw)
    return (
        decimals is not None
        and abs(exp - float(claimed)) <= 0.5 * 10 ** (-decimals) + 1e-12
        and decimals > 0
    )


class IndependentCalculationCheck(Check):
    """Rechnet die Aufgabe selbst nach (sicherer Auswerter, exakte Brüche) und vergleicht mit
    dem vom Agenten genannten Ergebnis."""

    name = "independent_calculation"
    category = "math"

    def __init__(self, expression: str | None = None) -> None:
        self.expression = expression

    async def run(self, ctx: VerificationContext) -> CheckResult:
        expression = self.expression or ctx.params.get("expression") or extract_expression(ctx.task)
        if not expression:
            return self.unverifiable("kein rechenbarer Ausdruck in der Aufgabe gefunden")
        try:
            expected = safe_eval(normalize_expression(str(expression)))
        except MathError as exc:
            return self.unverifiable(f"Ausdruck nicht auswertbar: {exc}", expression=expression)
        claimed = extract_claimed_value(ctx.claim)
        expected_text = _format(expected)
        if claimed is None:
            return self.failed(
                f"keine Zahl in der Antwort (erwartet {expected_text})",
                expression=expression,
                expected=expected_text,
            )
        raw, candidates = claimed
        evidence = {"expression": expression, "expected": expected_text, "claimed": raw}
        if any(values_match(expected, c, raw) for c in candidates):
            return self.passed(
                f"{expression} = {expected_text} – unabhängig nachgerechnet", **evidence
            )
        return self.failed(
            f"Antwort {raw} ≠ nachgerechnet {expected_text} ({expression})", **evidence
        )


def _format(value: Fraction | float) -> str:
    if isinstance(value, Fraction):
        if value.denominator == 1:
            return str(value.numerator)
        return f"{float(value):.10g} (= {value})"
    return f"{value:.10g}"


# =========================================================================== TEXT


@dataclass
class RequirementResult:
    requirement: str
    status: CheckStatus
    detail: str


_LIMIT_WORDS = re.compile(
    r"(maximal|höchstens|max\.?|nicht mehr als|unter|at most|no more than|under|mindestens|"
    r"wenigstens|at least|genau|exactly)\s*(\d+)\s*(wörter|worte|words|sätze[n]?|sentences?|"
    r"zeichen|characters|chars|(?:stich)?punkte[n]?|bullet points|items)",
    re.IGNORECASE,
)
_LIST = re.compile(
    r"\b(als (stichpunkt\w*|aufzählung|liste)|as a (bulleted )?list|bullet points?)\b",
    re.IGNORECASE,
)
_LANG = re.compile(r"\b(auf|in)\s+(deutsch|englisch|german|english)\b", re.IGNORECASE)
_JSON_REQ = re.compile(r"\b(als|in|as)\s+json\b", re.IGNORECASE)
_HEADING = re.compile(
    r"\b(mit überschrift\w*|with (a )?heading\w*|mit titel|with (a )?title)\b", re.IGNORECASE
)
_MENTION = re.compile(
    r"\b(erwähne|nenne|enthalte|include|mention|verwende das wort|use the word)\b"
    r"[^„\"“]*[„\"“]([^„\"“”]+)[“\"”]",
    re.IGNORECASE,
)
_REQ_LINE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+(.+)$", re.MULTILINE)
_LIST_ITEM = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+\S", re.MULTILINE)


def detect_language(text: str) -> str | None:
    tokens = words(text)
    de = sum(t in _DE_STOP for t in tokens)
    en = sum(t in _EN_STOP for t in tokens)
    if de + en < 3:
        return None
    if de >= 2 * en:
        return "de"
    if en >= 2 * de:
        return "en"
    return None


def _strip_fences(text: str) -> str:
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    return m.group(1) if m else text


def check_requirement(requirement: str, answer: str) -> list[RequirementResult]:
    """Prüft eine Anforderung deterministisch. Mehrere Teilregeln möglich; keine Regel →
    UNVERIFIABLE."""
    results: list[RequirementResult] = []

    def add(ok: bool, detail: str) -> None:
        results.append(
            RequirementResult(requirement, CheckStatus.PASSED if ok else CheckStatus.FAILED, detail)
        )

    for m in _LIMIT_WORDS.finditer(requirement):
        mode, limit, unit = m.group(1).lower(), int(m.group(2)), m.group(3).lower()
        if unit.startswith(("wört", "wort", "word")):
            count, label = len(words(answer)), "Wörter"
        elif unit.startswith(("satz", "sätz", "sentence")):
            count, label = len(sentences(answer)), "Sätze"
        elif unit.startswith(("zeichen", "char")):
            count, label = len(answer.strip()), "Zeichen"
        else:
            count, label = len(_LIST_ITEM.findall(answer)), "Listenpunkte"
        if mode in ("mindestens", "wenigstens", "at least"):
            add(count >= limit, f"{count} {label} (mindestens {limit})")
        elif mode in ("genau", "exactly"):
            add(count == limit, f"{count} {label} (genau {limit})")
        else:
            add(count <= limit, f"{count} {label} (höchstens {limit})")
    if _LIST.search(requirement):
        items = len(_LIST_ITEM.findall(answer))
        add(items >= 2, f"{items} Listenpunkte")
    if lang := _LANG.search(requirement):
        wanted = "de" if lang.group(2).lower() in ("deutsch", "german") else "en"
        detected = detect_language(answer)
        if detected is None:
            results.append(
                RequirementResult(
                    requirement, CheckStatus.UNVERIFIABLE, "Sprache nicht sicher erkennbar"
                )
            )
        else:
            add(detected == wanted, f"erkannte Sprache: {detected}")
    if _JSON_REQ.search(requirement):
        try:
            json.loads(_strip_fences(answer))
            add(True, "gültiges JSON")
        except ValueError as exc:
            add(False, f"kein gültiges JSON: {exc}")
    if _HEADING.search(requirement):
        add(bool(re.search(r"^\s*#{1,6}\s+\S", answer, re.MULTILINE)), "Markdown-Überschrift")
    for m in _MENTION.finditer(requirement):
        term = m.group(2).strip()
        add(term.lower() in answer.lower(), f"enthält „{term}“")
    if not results:
        results.append(
            RequirementResult(requirement, CheckStatus.UNVERIFIABLE, "nicht automatisch prüfbar")
        )
    return results


def extract_requirements(task: str) -> list[str]:
    """Anforderungen aus dem Aufgabentext: Aufzählungspunkte und Sätze mit prüfbaren Vorgaben."""
    reqs = [m.group(1).strip() for m in _REQ_LINE.finditer(task)]
    prose = [line for line in task.splitlines() if line.strip() and not _REQ_LINE.match(line)]
    for sentence in (s for line in prose for s in sentences(line)):
        if any(
            p.search(sentence) for p in (_LIMIT_WORDS, _LIST, _LANG, _JSON_REQ, _HEADING, _MENTION)
        ):
            reqs.append(sentence)
    return list(dict.fromkeys(reqs))


class RequirementCheck(Check):
    """Prüft die Antwort gegen ausdrückliche und aus der Aufgabe abgeleitete Anforderungen."""

    name = "requirements"
    category = "text"
    aspect = Aspect.REQUIREMENTS

    async def run(self, ctx: VerificationContext) -> CheckResult:
        requirements = list(dict.fromkeys([*ctx.requirements, *extract_requirements(ctx.task)]))
        if not requirements:
            return self.unverifiable("keine Anforderungen gefunden")
        if not ctx.claim.strip():
            return self.failed("keine Antwort vorhanden")
        items = [r for req in requirements for r in check_requirement(req, ctx.claim)]
        passed = [i for i in items if i.status == CheckStatus.PASSED]
        failed = [i for i in items if i.status == CheckStatus.FAILED]
        open_ = [i for i in items if i.status == CheckStatus.UNVERIFIABLE]
        evidence = {
            "requirements": [
                {"requirement": i.requirement, "status": i.status.value, "detail": i.detail}
                for i in items
            ],
            "passed": len(passed),
            "failed": len(failed),
            "unverifiable": len(open_),
        }
        if failed:
            return self.failed(
                "nicht erfüllt: " + "; ".join(f"{i.requirement} → {i.detail}" for i in failed[:3]),
                **evidence,
            )
        if passed:
            suffix = f", {len(open_)} nicht prüfbar" if open_ else ""
            return self.passed(f"{len(passed)} Anforderung(en) erfüllt{suffix}", **evidence)
        return self.unverifiable(
            f"{len(open_)} Anforderung(en) nicht automatisch prüfbar", **evidence
        )


class NonEmptyAnswerCheck(Check):
    """Minimalprüfung: Gibt es überhaupt eine inhaltliche Antwort? (Vollständigkeit)"""

    name = "non_empty_answer"
    category = "text"
    aspect = Aspect.COMPLETENESS
    independent = False  # sagt nichts über Korrektheit aus

    async def run(self, ctx: VerificationContext) -> CheckResult:
        n = len(words(ctx.claim))
        return self.passed(f"Antwort mit {n} Wörtern") if n else self.failed("leere Antwort")
