"""Klassifikation verschiedener Aufgabentypen."""

from __future__ import annotations

import pytest

from router import (
    Complexity,
    HybridClassifier,
    LLMClassifier,
    RoutingCategory,
    RoutingRequest,
    RuleBasedClassifier,
)

C = RoutingCategory


@pytest.mark.parametrize(
    ("task", "category", "complexity"),
    [
        # CODING
        ("Debug this Python project", C.CODING, Complexity.HIGH),
        ("Refactor the entire codebase to use async IO", C.CODING, Complexity.HIGH),
        ("Schreibe eine Funktion, die eine Liste sortiert", C.CODING, Complexity.LOW),
        (
            "Warum wirft mein Code einen KeyError?\n```python\nd = {}\nd['x']\n```",
            C.CODING,
            Complexity.MEDIUM,
        ),
        ("Fix the failing test in tests/test_api.py", C.CODING, Complexity.MEDIUM),
        # REASONING
        (
            "Vergleiche die Vor- und Nachteile von SQLite und PostgreSQL",
            C.REASONING,
            Complexity.MEDIUM,
        ),
        ("Beweise, dass die Wurzel aus 2 irrational ist", C.REASONING, Complexity.MEDIUM),
        (
            "Analysiere die Strategie und plane die nächsten Schritte für unser Produkt",
            C.REASONING,
            Complexity.HIGH,
        ),
        # VISION
        ("Was ist auf screenshot.png zu sehen?", C.VISION, Complexity.LOW),
        ("Describe this image", C.VISION, Complexity.LOW),
        # FAST
        ("Hallo!", C.FAST, Complexity.LOW),
        ("Danke", C.FAST, Complexity.LOW),
        ("Wie spät ist es in Tokio?", C.FAST, Complexity.LOW),
        # GENERAL
        (
            "Schreibe eine freundliche E-Mail an mein Team, dass das Meeting am Freitag ausfällt "
            "und auf nächste Woche verschoben wird.",
            C.GENERAL,
            Complexity.LOW,
        ),
    ],
)
def test_rule_classification(task: str, category: RoutingCategory, complexity: Complexity) -> None:
    result = RuleBasedClassifier().classify_sync(RoutingRequest(task))
    assert (result.category, result.complexity) == (category, complexity), result.signals
    assert result.reason and 0 < result.confidence <= 1


def test_long_context_keeps_content_category() -> None:
    result = RuleBasedClassifier().classify_sync(
        RoutingRequest("Finde den Bug in diesem Python-Code", conversation_tokens=50_000)
    )
    assert result.category is C.LONG_CONTEXT and result.secondary is C.CODING
    assert result.context_tokens >= 50_000 + 2048
    assert any("kontext" in s for s in result.signals)


def test_vision_flag_beats_text_and_long_context() -> None:
    result = RuleBasedClassifier().classify_sync(
        RoutingRequest("Debug this", needs_vision=True, conversation_tokens=99_999)
    )
    assert result.category is C.VISION


def test_tools_raise_complexity_and_are_reported() -> None:
    plain = RuleBasedClassifier().classify_sync(RoutingRequest("Lies die Datei notes.md"))
    with_tools = RuleBasedClassifier().classify_sync(
        RoutingRequest("Lies die Datei notes.md", required_tools=("read_file",))
    )
    assert with_tools.needs_tools and not plain.needs_tools
    assert with_tools.complexity > plain.complexity
    assert with_tools.category is not C.FAST  # Tool-Aufgaben sind nie „schnell beantwortet“


def test_external_hints_and_estimates_win() -> None:
    rules = RuleBasedClassifier()
    hinted = rules.classify_sync(RoutingRequest("Mach das", category_hint=C.REASONING))
    assert hinted.category is C.REASONING and "hinweis" in " ".join(hinted.signals)
    fixed = rules.classify_sync(
        RoutingRequest("Debug this Python project", complexity=Complexity.LOW)
    )
    assert fixed.complexity is Complexity.LOW


def test_configurable_thresholds() -> None:
    small = RuleBasedClassifier(long_context_threshold=100)
    assert (
        small.classify_sync(RoutingRequest("x", conversation_tokens=500)).category is C.LONG_CONTEXT
    )


# --------------------------------------------------------------------------- LLM / Hybrid


def completion(answer: str | Exception):  # type: ignore[no-untyped-def]
    calls: list[str] = []

    async def complete(prompt: str) -> str:
        calls.append(prompt)
        if isinstance(answer, Exception):
            raise answer
        return answer

    complete.calls = calls  # type: ignore[attr-defined]
    return complete


async def test_llm_classifier_uses_model_answer() -> None:
    llm = LLMClassifier(
        completion(
            'Antwort: {"category": "reasoning", "complexity": "high", '
            '"reason": "multi-step planning"}'
        )
    )
    result = await llm.classify(RoutingRequest("Mach mal was mit dem Projektplan"))
    assert (result.category, result.complexity) == (C.REASONING, Complexity.HIGH)
    assert result.classifier == "llm" and result.reason == "multi-step planning"


@pytest.mark.parametrize(
    "answer",
    [
        "kein json",
        '{"category": "magic", "complexity": "high"}',
        '{"category": "coding"}',
        '{"category": "vision", "complexity": "low"}',
        RuntimeError("Modell weg"),
    ],
)
async def test_llm_classifier_falls_back_to_rules(answer: str | Exception) -> None:
    result = await LLMClassifier(completion(answer)).classify(
        RoutingRequest("Debug this Python project")
    )
    assert result.category is C.CODING and result.classifier == "rules (llm-fallback)"


async def test_llm_cannot_override_measurable_facts() -> None:
    complete = completion('{"category": "fast", "complexity": "low"}')
    llm = LLMClassifier(complete)
    vision = await llm.classify(RoutingRequest("Was ist das?", needs_vision=True))
    long = await llm.classify(RoutingRequest("Fasse zusammen", conversation_tokens=80_000))
    assert vision.category is C.VISION and long.category is C.LONG_CONTEXT
    assert complete.calls == []  # Fakten brauchen kein Modell
    tools = await llm.classify(RoutingRequest("Lies notes.md", required_tools=("read_file",)))
    assert tools.needs_tools  # Tool-Bedarf kommt aus der Anfrage, nicht vom Modell


async def test_hybrid_asks_llm_only_when_rules_are_unsure() -> None:
    complete = completion('{"category": "coding", "complexity": "medium"}')
    hybrid = HybridClassifier(
        RuleBasedClassifier(), LLMClassifier(complete), confidence_threshold=0.6
    )
    sure = await hybrid.classify(RoutingRequest("Was ist auf screenshot.png?"))
    assert sure.classifier == "rules" and complete.calls == []
    unsure = await hybrid.classify(
        RoutingRequest("Bitte kümmere dich um das Ding von gestern, du weißt schon, mit den Zahlen")
    )
    assert unsure.classifier == "llm" and len(complete.calls) == 1
