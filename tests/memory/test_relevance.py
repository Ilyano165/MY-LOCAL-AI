"""Relevanzbewertung: was wird wo gespeichert – und was ausdrücklich nicht."""

from __future__ import annotations

import pytest

from memory import HeuristicRelevanceAssessor, MemoryContext, MemoryKind, MemoryLayer, MemorySource

FULL = MemoryContext(session_id="s", project_id="p", task_id="t")
NO_PROJECT = MemoryContext(session_id="s")
NOTHING = MemoryContext()

L, P, S = MemoryLayer.LONG_TERM, MemoryLayer.PROJECT, MemoryLayer.SESSION


@pytest.mark.parametrize(
    ("text", "ctx", "store", "layer", "kind"),
    [
        # nie dauerhaft: Smalltalk, Fragen, Secrets
        ("Hallo!", FULL, False, None, MemoryKind.MESSAGE),
        ("danke", FULL, False, None, MemoryKind.MESSAGE),
        ("Wie funktioniert der Router?", FULL, True, S, MemoryKind.MESSAGE),
        ("Kannst du mir helfen", FULL, True, S, MemoryKind.MESSAGE),
        ("Mein Passwort: hunter22", FULL, False, None, MemoryKind.NOTE),
        ("Merk dir mein Token: abcdef123456", FULL, False, None, MemoryKind.NOTE),
        # Präferenzen und Anweisungen → Langzeit
        ("Ich bevorzuge kurze Antworten.", FULL, True, L, MemoryKind.PREFERENCE),
        ("I prefer pytest over unittest", FULL, True, L, MemoryKind.PREFERENCE),
        ("Antworte immer auf Deutsch.", FULL, True, L, MemoryKind.INSTRUCTION),
        ("Never use emojis in commit messages", FULL, True, L, MemoryKind.INSTRUCTION),
        ("Merk dir: mein Geburtstag ist am 3. Mai", FULL, True, L, MemoryKind.FACT),
        # Projektbezug → Projekt
        ("Wir verwenden SQLite für die Persistenz.", FULL, True, P, MemoryKind.DECISION),
        (
            "Merk dir: in diesem Projekt liegen Tests unter tests/unit",
            FULL,
            True,
            P,
            MemoryKind.FACT,
        ),
        ("Nutze in diesem Repo immer ruff statt black", FULL, True, P, MemoryKind.INSTRUCTION),
        (
            "Grundsätzlich bevorzuge ich in allen Projekten Typannotationen",
            FULL,
            True,
            L,
            MemoryKind.PREFERENCE,
        ),
        # ohne Projektkontext
        ("Wir verwenden SQLite für die Persistenz.", NO_PROJECT, True, S, MemoryKind.DECISION),
        (
            "Merk dir: in diesem Projekt liegen Tests unter tests/unit",
            NO_PROJECT,
            True,
            L,
            MemoryKind.FACT,
        ),
        # gewöhnliche Aussagen → nur Session (nicht dauerhaft)
        ("Der Build ist gerade rot.", FULL, True, S, MemoryKind.FACT),
        ("Der Build ist gerade rot.", NOTHING, False, None, MemoryKind.FACT),
    ],
)
def test_routing(
    text: str, ctx: MemoryContext, store: bool, layer: MemoryLayer | None, kind: MemoryKind
) -> None:
    result = HeuristicRelevanceAssessor().assess(text, MemorySource.USER, ctx)
    assert (result.store, result.layer, result.kind) == (store, layer, kind), result.reasons
    assert result.reasons  # jede Entscheidung ist begründet


def test_explicit_prefix_is_stripped_and_importance_high() -> None:
    result = HeuristicRelevanceAssessor().assess("Merk dir: ich mag Tee", MemorySource.USER, FULL)
    assert result.content == "ich mag Tee" and result.importance >= 0.9


def test_agent_statements_weigh_less_than_user_statements() -> None:
    assessor = HeuristicRelevanceAssessor()
    user = assessor.assess("Wir verwenden SQLite.", MemorySource.USER, FULL)
    agent = assessor.assess("Wir verwenden SQLite.", MemorySource.AGENT, FULL)
    assert agent.importance < user.importance
    assert agent.layer is S  # unter der Schwelle → nicht dauerhaft


def test_tool_output_only_in_working_memory() -> None:
    assessor = HeuristicRelevanceAssessor()
    with_task = assessor.assess("3 passed, 1 failed", MemorySource.TOOL, FULL)
    assert (with_task.store, with_task.layer) == (True, MemoryLayer.WORKING)
    without = assessor.assess("3 passed", MemorySource.TOOL, NO_PROJECT)
    assert not without.store


def test_threshold_is_configurable() -> None:
    strict = HeuristicRelevanceAssessor(durable_threshold=0.95)
    assert strict.assess("Antworte immer auf Deutsch.", MemorySource.USER, FULL).layer is S


@pytest.mark.parametrize(
    "text", ["Für Tests bevorzuge ich pytest", "Lange Antworten mag ich nicht"]
)
def test_german_verb_second_word_order(text: str) -> None:
    result = HeuristicRelevanceAssessor().assess(text, MemorySource.USER, FULL)
    assert result.kind is MemoryKind.PREFERENCE


def test_mixed_message_extracts_the_durable_sentence() -> None:
    result = HeuristicRelevanceAssessor().assess(
        "Was ist 2+2? Und antworte ab jetzt immer auf Deutsch.", MemorySource.USER, FULL
    )
    assert (result.layer, result.kind) == (L, MemoryKind.INSTRUCTION)
    assert result.content == "antworte ab jetzt immer auf Deutsch."


def test_mixed_message_without_durable_part_stays_whole() -> None:
    text = "Der Build ist rot. Kannst du nachsehen?"
    result = HeuristicRelevanceAssessor().assess(text, MemorySource.USER, FULL)
    assert result.layer is S and result.content == text


def test_version_numbers_do_not_split_sentences() -> None:
    result = HeuristicRelevanceAssessor().assess(
        "Wir verwenden Python 3.12 im Projekt.", MemorySource.USER, FULL
    )
    assert result.content == "Wir verwenden Python 3.12 im Projekt."
