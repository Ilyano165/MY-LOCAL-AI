"""Aufgabenklassifikation: Kategorie, Komplexität, Bedarf.

* :class:`RuleBasedClassifier` – deterministisch, ohne Latenz, jede Entscheidung mit Signalen.
* :class:`LLMClassifier` – fragt ein (kleines) Modell nach Kategorie/Komplexität; messbare
  Fakten (Bilder, Kontextgröße, Tools) kommen aber immer aus den Regeln. Bei Fehlern oder
  ungültiger Antwort → Rückfall auf die Regeln.
* :class:`HybridClassifier` – LLM nur, wenn die Regeln unsicher sind (spart Latenz).

Ein gelernter Klassifikator (z. B. auf Basis der Routing-Logs) implementiert ebenfalls
:class:`~router.base.TaskClassifier` – Router und Rest bleiben unverändert.
"""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable

from router.base import (
    Complexity,
    RoutingCategory,
    RoutingRequest,
    TaskClassification,
    TaskClassifier,
)


def _rx(*patterns: str) -> re.Pattern[str]:
    return re.compile("|".join(patterns), re.IGNORECASE)


CODING = _rx(
    r"\bcode\b",
    r"\bdebug\w*",
    r"\bbug\w*",
    r"\bfix\w*\b.*\b(error|fehler|test|crash)",
    r"\b(funktion|function|klasse|class|methode|method|variable)\b",
    r"\btests?\b",
    r"\bpytest\b",
    r"\b(python|javascript|typescript|rust|golang|java|c\+\+|sql|bash)\b",
    r"\bimplementier\w*",
    r"\bimplement\w*",
    r"\brefactor\w*",
    r"\bcompil\w*",
    r"\bstack ?trace\b",
    r"\btraceback\b",
    r"\bexception\b",
    r"\bsyntax\w*",
    r"\bapi\b",
    r"\brepo(sitory)?\b",
    r"\bprogramm\w*",
    r"\.(py|js|ts|tsx|rs|go|java|cpp|c|h|rb|sh|sql)\b",
    r"```",
    r"\bdef |\bimport |=>|\{\s*$",
)
REASONING = _rx(
    r"\banalys\w*",
    r"\bbeweis\w*",
    r"\bprove\b",
    r"\bproof\b",
    r"\bherleit\w*",
    r"\bderive\b",
    r"\bwarum\b",
    r"\bwhy\b",
    r"\bvergleich\w*",
    r"\bcompare\b",
    r"\babwäg\w*",
    r"\btrade-?offs?\b",
    r"\bstrateg\w*",
    r"\bplan(e|en|ung)?\b",
    r"\barchitektur\w*",
    r"\barchitecture\b",
    r"\blogi(k|c)\w*",
    r"\bschritt für schritt\b",
    r"\bstep by step\b",
    r"\bberechn\w*",
    r"\bcalculat\w*",
    r"\bwahrscheinlichkeit\b",
    r"\bprobability\b",
    r"\boptimi\w*",
    r"\bentscheid\w*",
    r"\bdecide\b",
    r"\brätsel\b",
    r"\bpuzzle\b",
)
VISION = _rx(
    r"\.(png|jpe?g|gif|webp|bmp|svg)\b",
    r"\bbild\w*",
    r"\bimage\w*",
    r"\bscreenshot\w*",
    r"\bfoto\w*",
    r"\bphoto\w*",
    r"\bdiagramm\w* (an|auf)\b",
)
MULTI_STEP = _rx(
    r"\bund dann\b",
    r"\bthen\b",
    r"\bdanach\b",
    r"\banschließend\b",
    r"\bafterwards\b",
    r"\bfirst\b.*\bthen\b",
    r"\bzuerst\b",
    r"\b(nächsten |next )?(schritte|steps)\b",
    r"\broadmap\b",
)
BROAD_SCOPE = _rx(
    r"\b(gesamte[ns]?|ganze[ns]?|alle[ns]?|entire|whole|all)\b.{0,30}"
    r"\b(projekt|project|codebase|repo\w*|system|module|dateien|files)\b",
    r"\bend[- ]to[- ]end\b",
    r"\bmigration\b",
    r"\barchitektur\b",
    r"\barchitecture\b",
    r"\brefactor\w*",
)
PROJECT_SCOPE = _rx(
    r"\b(projekt\w*|project|codebase|code-?basis|repo\w*|anwendung|application|"
    r"service|monorepo)\b"
)
DEEP_WORK = _rx(
    r"\bdebug\w*",
    r"\brefactor\w*",
    r"\bfix\w*",
    r"\bbehebe?\w*",
    r"\bmigrier\w*",
    r"\bmigrat\w*",
    r"\boptimi\w*",
    r"\bumbau\w*",
    r"\brestructur\w*",
)
FORMAL = _rx(
    r"\bbeweis\w*", r"\bprove\b", r"\bproof\b", r"\bherleit\w*", r"\bderive\b", r"\bformal\w*"
)
TRIVIAL = _rx(r"^\s*(hallo|hi|hey|danke|thanks|ok|ja|nein|yes|no|guten (morgen|tag|abend))\W*$")
LIST_ITEM = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+", re.MULTILINE)


class RuleBasedClassifier(TaskClassifier):
    name = "rules"

    def __init__(self, *, long_context_threshold: int = 24_000, fast_max_words: int = 15) -> None:
        self.long_context_threshold = long_context_threshold
        self.fast_max_words = fast_max_words

    async def classify(self, request: RoutingRequest) -> TaskClassification:
        return self.classify_sync(request)

    def classify_sync(self, request: RoutingRequest) -> TaskClassification:
        text = request.task
        signals: list[str] = []

        def hits(pattern: re.Pattern[str], label: str) -> int:
            found = {m.group(0).lower().strip() for m in pattern.finditer(text)}
            if found:
                signals.append(f"{label}: {', '.join(sorted(found)[:4])}")
            return len(found)

        coding, reasoning = hits(CODING, "code"), hits(REASONING, "reasoning")
        vision = request.needs_vision or bool(hits(VISION, "vision"))
        words = len(text.split())
        needs_tools = bool(request.required_tools)
        context = request.total_input_tokens + request.expected_output_tokens
        context = max(context, request.min_context_tokens)

        complexity = request.complexity or self._complexity(
            text, coding, reasoning, needs_tools, request.total_input_tokens, signals
        )

        # Kategorie: Fakten vor Inhalt
        secondary: RoutingCategory | None = None
        if vision:
            category, confidence, reason = (
                RoutingCategory.VISION,
                0.95,
                "Bildinhalt muss verstanden werden",
            )
        else:
            content, confidence, reason = self._content_category(
                request, coding, reasoning, words, needs_tools, complexity, signals
            )
            if request.total_input_tokens > self.long_context_threshold:
                category, secondary = RoutingCategory.LONG_CONTEXT, content
                signals.append(f"kontext: ~{request.total_input_tokens} Tokens")
                reason = (
                    f"Sehr großer Kontext (~{request.total_input_tokens} Tokens) – "
                    f"inhaltlich {content.value}"
                )
                confidence = 0.9
            else:
                category = content
        if request.complexity is None and RoutingCategory.REASONING in (category, secondary):
            # Analyse/Vergleich/Planung ist nie trivial
            complexity = max(complexity, Complexity.MEDIUM)
        return TaskClassification(
            category=category,
            complexity=complexity,
            needs_tools=needs_tools,
            context_tokens=context,
            confidence=confidence,
            secondary=secondary,
            signals=signals,
            reason=reason,
            classifier=self.name,
        )

    def _content_category(
        self,
        request: RoutingRequest,
        coding: int,
        reasoning: int,
        words: int,
        needs_tools: bool,
        complexity: Complexity,
        signals: list[str],
    ) -> tuple[RoutingCategory, float, str]:
        hint = request.category_hint
        if hint is not None and hint not in (RoutingCategory.VISION, RoutingCategory.LONG_CONTEXT):
            signals.append(f"hinweis: {hint.value}")
            return hint, 0.9, f"Vorgabe des Aufrufers: {hint.value}"
        if coding and coding >= reasoning:
            reason = "Erfordert Codeverständnis"
            if complexity is Complexity.HIGH:
                reason += " und mehrstufiges Vorgehen (z. B. Debugging über mehrere Dateien)"
            return RoutingCategory.CODING, min(0.55 + 0.15 * coding, 0.95), reason
        if reasoning:
            return (
                RoutingCategory.REASONING,
                min(0.5 + 0.15 * reasoning, 0.9),
                "Erfordert Analyse/mehrstufiges Schlussfolgern",
            )
        if TRIVIAL.match(request.task):
            signals.append("floskel")
            return RoutingCategory.FAST, 0.9, "Kurze Floskel – schnelle Antwort genügt"
        if (
            words <= self.fast_max_words
            and not needs_tools
            and request.conversation_tokens < 2000
            and complexity is Complexity.LOW
        ):
            signals.append(f"kurz: {words} Wörter")
            # Kürze allein ist ein schwaches Signal → geringe Sicherheit (Hybrid fragt nach)
            return RoutingCategory.FAST, 0.55, "Kurze, einfache Anfrage – schnelle Antwort genügt"
        return RoutingCategory.GENERAL, 0.5, "Allgemeine Aufgabe ohne spezielle Anforderungen"

    @staticmethod
    def _complexity(
        text: str,
        coding: int,
        reasoning: int,
        needs_tools: bool,
        input_tokens: int,
        signals: list[str],
    ) -> Complexity:
        points = 0
        words = len(text.split())
        if words > 60:
            points += 1
        if words > 200:
            points += 1
        if MULTI_STEP.search(text) or len(LIST_ITEM.findall(text)) >= 3:
            points += 1
            signals.append("mehrstufig")
        if BROAD_SCOPE.search(text):
            points += 2
            signals.append("großer Umfang")
        if DEEP_WORK.search(text):
            points += 1  # Debuggen/Beheben/Umbauen: Hypothesen bilden, prüfen, verwerfen
            signals.append("tiefe Arbeit")
        if PROJECT_SCOPE.search(text):
            # Arbeit auf Projektebene: mehrere Dateien verstehen; mit Debug/Refactor/Fix → tief
            deep = bool(DEEP_WORK.search(text))
            points += 2 if deep else 1
            signals.append("projektebene" + (" + tiefe Arbeit" if deep else ""))
        if coding and reasoning:
            points += 1
        if reasoning >= 2:
            points += 1
        if reasoning >= 3:
            points += 1  # mehrere unterschiedliche Denkanforderungen (analysieren + planen …)
        if FORMAL.search(text):
            points += 1
            signals.append("formales Schließen")
        if needs_tools:
            points += 1
        if input_tokens > 8000:
            points += 1
        if points >= 3:
            return Complexity.HIGH
        if points >= 1:
            return Complexity.MEDIUM
        return Complexity.LOW


# --------------------------------------------------------------------------- LLM

Completion = Callable[[str], Awaitable[str]]

_LLM_PROMPT = """Classify the task for model routing. Answer with JSON only:
{{"category": "fast|general|coding|reasoning", "complexity": "low|medium|high",
 "reason": "<short>"}}
- fast: trivial, short answer. coding: involves source code. reasoning: analysis, planning, math.
- complexity high: multi-step, large scope, debugging, design decisions.

Task:
{task}"""


class LLMClassifier(TaskClassifier):
    """Modellbasierte Klassifikation über eine beliebige Completion-Funktion.

    ``complete`` ist bewusst eine schlichte ``async (prompt) -> text``-Funktion, damit kein
    Rückbezug auf den Router entsteht (das Klassifikationsmodell wird direkt angesprochen).
    """

    name = "llm"

    def __init__(
        self,
        complete: Completion,
        *,
        rules: RuleBasedClassifier | None = None,
        max_task_chars: int = 4000,
    ) -> None:
        self.complete = complete
        self.rules = rules or RuleBasedClassifier()
        self.max_task_chars = max_task_chars

    async def classify(self, request: RoutingRequest) -> TaskClassification:
        base = self.rules.classify_sync(request)
        if base.category in (RoutingCategory.VISION, RoutingCategory.LONG_CONTEXT):
            return base  # messbare Fakten – kein Modell nötig
        try:
            raw = await self.complete(_LLM_PROMPT.format(task=request.task[: self.max_task_chars]))
            data = json.loads(_json_block(raw))
            category = RoutingCategory(str(data["category"]).lower())
            complexity = Complexity.parse(str(data["complexity"]))
            if category in (RoutingCategory.VISION, RoutingCategory.LONG_CONTEXT):
                raise ValueError("Kategorie aus Fakten, nicht vom Modell")
        except Exception as exc:
            base.signals.append(f"llm-fallback: {type(exc).__name__}")
            base.classifier = "rules (llm-fallback)"
            return base
        if request.complexity is not None:
            complexity = request.complexity
        return TaskClassification(
            category=category,
            complexity=complexity,
            needs_tools=base.needs_tools,
            context_tokens=base.context_tokens,
            confidence=0.8,
            signals=[*base.signals, "llm"],
            reason=str(data.get("reason") or base.reason)[:200],
            classifier=self.name,
        )


def _json_block(text: str) -> str:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("keine JSON-Antwort")
    return match.group(0)


class HybridClassifier(TaskClassifier):
    """Regeln zuerst; das LLM nur bei geringer Regel-Sicherheit."""

    name = "hybrid"

    def __init__(
        self, rules: RuleBasedClassifier, llm: LLMClassifier, *, confidence_threshold: float = 0.6
    ) -> None:
        self.rules = rules
        self.llm = llm
        self.confidence_threshold = confidence_threshold

    async def classify(self, request: RoutingRequest) -> TaskClassification:
        result = self.rules.classify_sync(request)
        if result.confidence >= self.confidence_threshold:
            return result
        return await self.llm.classify(request)
