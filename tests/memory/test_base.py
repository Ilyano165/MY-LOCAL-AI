from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from memory.base import (
    MemoryItem,
    MemoryKind,
    MemoryLayer,
    RetrievalWeights,
    contains_secret,
    content_hash,
    lexical_relevance,
    recency_score,
    redact_secrets,
    score_memory,
    term_weight,
    tokenize,
)

NOW = datetime(2026, 10, 8, 12, 0, tzinfo=UTC)


def test_tokenize_removes_stopwords_and_splits_hyphens() -> None:
    assert tokenize("Die Datenbank-Migration in db/migrate.py ist kaputt.") == [
        "datenbank",
        "migration",
        "db/migrate.py",
        "kaputt",
    ]
    assert tokenize("What is the plan for the API?") == ["plan", "api"]
    assert tokenize("und oder 2026") == []


def test_lexical_relevance_exact_partial_and_empty() -> None:
    assert lexical_relevance(["sqlite", "persistenz"], ["wir", "sqlite", "persistenz"]) == 1.0
    assert lexical_relevance(["sqlite", "redis"], ["sqlite"]) == pytest.approx(0.6 / 1.1)
    assert lexical_relevance(["datenbank"], ["datenbankschema"]) == pytest.approx(0.7)
    assert lexical_relevance(["db"], ["dbase"]) == 0.0  # kurze Begriffe nur exakt
    assert lexical_relevance([], ["x"]) == 0.0


def test_lexical_relevance_is_robust_to_long_queries() -> None:
    task = tokenize("Füge eine neue Datenbank-Migration für die Volltextsuche hinzu")
    core = lexical_relevance(task, tokenize("Wir verwenden SQLite mit FTS5 für die Volltextsuche"))
    filler = lexical_relevance(task, tokenize("Die neue Kaffeemaschine ist da"))
    assert core >= 0.5  # ein spezifischer Kernbegriff genügt
    assert filler < 0.3  # ein Allerweltswort genügt nicht
    assert term_weight("neue") < term_weight("volltextsuche")


def test_recency_halves_per_half_life() -> None:
    assert recency_score(NOW, NOW, 24) == 1.0
    assert recency_score(NOW - timedelta(hours=24), NOW, 24) == pytest.approx(0.5)
    assert recency_score(NOW - timedelta(hours=48), NOW, 24) == pytest.approx(0.25)
    assert recency_score(NOW + timedelta(hours=1), NOW, 24) == 1.0  # Uhrzeit-Drift ≠ Bonus


@pytest.mark.parametrize(
    "text",
    [
        "API_KEY=sk-abcdefghijklmnopqrstuvwxyz",
        "passwort: hunter22",
        "Token: abcd1234efgh",
        "ghp_" + "x" * 30,
        "-----BEGIN RSA PRIVATE KEY-----",
        "Kennwort = geheim123",
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U",
    ],
)
def test_secret_detection(text: str) -> None:
    assert contains_secret(text)
    assert "[REDACTED]" in redact_secrets(f"vorher {text} nachher")


@pytest.mark.parametrize(
    "text",
    [
        "Das Token-Budget ist 4000",
        "Passwort-Reset-Flow bauen",
        "Wir nutzen API keys aus der Umgebung",
    ],
)
def test_no_false_secret_alarm(text: str) -> None:
    assert not contains_secret(text)


def test_memory_item_validation_and_roundtrip() -> None:
    item = MemoryItem(
        "  Inhalt  ",
        MemoryLayer.PROJECT,
        "p1",
        kind=MemoryKind.DECISION,
        importance=0.7,
        tags=["a"],
        created_at=NOW,
        updated_at=NOW,
        last_accessed_at=NOW,
        expires_at=NOW + timedelta(days=1),
    )
    assert item.content == "Inhalt"
    assert MemoryItem.from_dict(item.to_dict()) == item
    assert not item.is_expired(NOW) and item.is_expired(NOW + timedelta(days=2))
    for bad in ({"content": " "}, {"scope": ""}, {"importance": 1.5}, {"content": "x" * 9000}):
        kwargs = {"content": "x", "layer": MemoryLayer.SESSION, "scope": "s", **bad}
        with pytest.raises(ValueError):
            MemoryItem(**kwargs)  # type: ignore[arg-type]


def test_content_hash_normalizes_case_and_whitespace() -> None:
    assert content_hash("Wir  nutzen SQLite") == content_hash("wir nutzen sqlite")


def test_score_memory_combines_components() -> None:
    item = MemoryItem("x", MemoryLayer.LONG_TERM, "global", importance=1.0, last_accessed_at=NOW)
    weights = RetrievalWeights(relevance=1, importance=1, recency=0)
    scored = score_memory(item, 0.5, NOW, 24, weights)
    assert scored.score == pytest.approx(0.75) and scored.recency == 1.0
    assert "relevanz=0.50" in scored.explain()
    with pytest.raises(ValueError):
        RetrievalWeights(relevance=-1)
    with pytest.raises(ValueError):
        RetrievalWeights(0, 0, 0)
