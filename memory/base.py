"""Grundbausteine des Memory-Systems.

* :class:`MemoryItem` – eine Erinnerung (Inhalt, Schicht, Scope, Art, Wichtigkeit, Zugriffsdaten).
* :class:`MemoryStore` – Speicher-Backend (``InMemoryStore`` hier, ``SQLiteMemoryStore`` in
  :mod:`memory.sqlite_store`). Ein Backend kann mehrere Schichten/Scopes halten.
* :class:`LayerMemory` – Sicht auf *eine* Schicht mit eigener Halbwertszeit, Lebensdauer und
  Kapazität; Basis für Working/Session/Project/Long-Term Memory.
* Text- und Scoring-Hilfen: Tokenisierung, lexikalische Relevanz, Aktualität (exponentieller
  Zerfall seit letztem Zugriff, vgl. Park et al. 2023 „Generative Agents“), Gesamtscore.
"""

from __future__ import annotations

import abc
import builtins
import hashlib
import math
import re
import unicodedata
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any, ClassVar

Clock = Callable[[], datetime]
MAX_CONTENT_CHARS = 8000


def utc_now() -> datetime:
    return datetime.now(UTC)


class MemoryStoreError(Exception):
    """Fehler des Memory-Systems."""


class MemoryNotFoundError(MemoryStoreError):
    pass


class MemoryLayer(StrEnum):
    WORKING = "working"
    SESSION = "session"
    PROJECT = "project"
    LONG_TERM = "long_term"


class MemoryKind(StrEnum):
    MESSAGE = "message"  # Gesprächsnachricht (nur Session)
    FACT = "fact"
    PREFERENCE = "preference"  # Vorliebe des Nutzers
    INSTRUCTION = "instruction"  # dauerhafte Anweisung („antworte immer auf Deutsch“)
    DECISION = "decision"  # Projektentscheidung
    TASK_RESULT = "task_result"
    LESSON = "lesson"  # Erkenntnis aus Fehler/Korrektur
    OBSERVATION = "observation"
    NOTE = "note"


class MemorySource(StrEnum):
    USER = "user"
    AGENT = "agent"
    TOOL = "tool"
    SYSTEM = "system"


# --------------------------------------------------------------------------- Text


_STOPWORDS = frozenset(
    [
        "der",
        "die",
        "das",
        "den",
        "dem",
        "des",
        "ein",
        "eine",
        "einer",
        "eines",
        "einem",
        "einen",
        "und",
        "oder",
        "aber",
        "nicht",
        "kein",
        "keine",
        "ist",
        "sind",
        "war",
        "waren",
        "wird",
        "werden",
        "wurde",
        "hat",
        "haben",
        "hatte",
        "ich",
        "du",
        "er",
        "sie",
        "es",
        "wir",
        "ihr",
        "mich",
        "mir",
        "dich",
        "dir",
        "uns",
        "euch",
        "sich",
        "mein",
        "dein",
        "sein",
        "unser",
        "euer",
        "im",
        "in",
        "am",
        "an",
        "auf",
        "aus",
        "bei",
        "mit",
        "nach",
        "von",
        "vor",
        "zu",
        "zum",
        "zur",
        "für",
        "fuer",
        "über",
        "ueber",
        "unter",
        "um",
        "als",
        "wie",
        "wenn",
        "dass",
        "da",
        "so",
        "auch",
        "noch",
        "nur",
        "schon",
        "sehr",
        "mal",
        "bitte",
        "was",
        "wer",
        "wo",
        "wann",
        "warum",
        "welche",
        "welcher",
        "welches",
        "diese",
        "dieser",
        "dieses",
        "jetzt",
        "hier",
        "dort",
        "ja",
        "nein",
        "doch",
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "not",
        "no",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "me",
        "my",
        "your",
        "our",
        "their",
        "this",
        "that",
        "these",
        "those",
        "to",
        "of",
        "in",
        "on",
        "at",
        "for",
        "with",
        "from",
        "by",
        "as",
        "if",
        "then",
        "so",
        "do",
        "does",
        "did",
        "can",
        "could",
        "should",
        "would",
        "will",
        "just",
        "please",
        "what",
        "who",
        "where",
        "when",
        "why",
        "which",
        "how",
    ]
)
# Punkte und Schrägstriche bleiben Teil des Tokens (Dateinamen, Pfade); Bindestriche trennen
# (deutsche Bindestrich-Komposita: „Datenbank-Migration“ → „datenbank“, „migration“).
_TOKEN_RE = re.compile(r"[\w][\w./]*", re.UNICODE)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return " ".join(text.split())


def tokenize(text: str) -> list[str]:
    """Kleingeschriebene Inhaltswörter ohne Stoppwörter (Reihenfolge bleibt erhalten)."""
    tokens = []
    for raw in _TOKEN_RE.findall(normalize_text(text)):
        token = raw.strip("./")
        if len(token) >= 2 and token not in _STOPWORDS and not token.isdigit():
            tokens.append(token)
    return tokens


def content_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def term_weight(term: str) -> float:
    """Spezifität eines Begriffs: längere Wörter tragen mehr Information (0.3..1.0)."""
    return min(max(len(term), 3), 10) / 10


def lexical_relevance(
    query_tokens: Sequence[str], doc_tokens: Iterable[str], saturation: float = 2.0
) -> float:
    """Lexikalische Relevanz eines Dokuments für eine Anfrage (0..1).

    * Treffer: exakt = 1, Präfix/Kompositum (≥ 4 Zeichen, z. B. „datenbank“ in
      „datenbankschema“) = 0.7.
    * Gewichtung nach Spezifität (:func:`term_weight`): „volltextsuche“ zählt mehr als „neue“.
    * Sättigung: Treffer im Umfang von ``saturation`` (≈ zwei spezifische Begriffe) gelten als
      voll relevant. Damit bestrafen lange Aufgabenbeschreibungen eine Erinnerung nicht dafür,
      dass sie nur deren Kernbegriffe enthält; ein einzelnes Allerweltswort reicht nicht.

    Bewusst einfach und erklärbar; semantische Ähnlichkeit (Embeddings) kann später ergänzen.
    """
    query = list(dict.fromkeys(query_tokens))
    if not query:
        return 0.0
    doc = set(doc_tokens)
    matched = 0.0
    for term in query:
        if term in doc:
            matched += term_weight(term)
        elif len(term) >= 4 and any(term in d or (len(d) >= 4 and d in term) for d in doc):
            matched += 0.7 * term_weight(term)
    possible = sum(term_weight(t) for t in query)
    return min(matched / min(possible, saturation), 1.0)


def recency_score(last_access: datetime, now: datetime, half_life_hours: float) -> float:
    """Exponentieller Zerfall seit dem letzten Zugriff.

    1.0 = gerade eben, 0.5 = vor einer Halbwertszeit.
    """
    hours = max((now - last_access).total_seconds() / 3600.0, 0.0)
    return math.pow(0.5, hours / half_life_hours) if half_life_hours > 0 else 0.0


# --------------------------------------------------------------------------- Secrets

_SECRET_PATTERNS = re.compile(
    r"(sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|xox[abposr]-[A-Za-z0-9-]{10,}"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----"
    r"|eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"
    r"|(?i:(?:passwor[dt]|passphrase|api[_ -]?key|secret|token|kennwort|pin)\s*[:=]\s*\S{4,}))"
)


def contains_secret(text: str) -> bool:
    return bool(_SECRET_PATTERNS.search(text))


def redact_secrets(text: str) -> str:
    return _SECRET_PATTERNS.sub("[REDACTED]", text)


# --------------------------------------------------------------------------- Datentypen


@dataclass
class MemoryItem:
    content: str
    layer: MemoryLayer
    scope: str
    kind: MemoryKind = MemoryKind.NOTE
    importance: float = 0.5
    source: MemorySource = MemorySource.AGENT
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    last_accessed_at: datetime = field(default_factory=utc_now)
    access_count: int = 0
    expires_at: datetime | None = None

    def __post_init__(self) -> None:
        self.layer = MemoryLayer(self.layer)
        self.kind = MemoryKind(self.kind)
        self.source = MemorySource(self.source)
        self.content = self.content.strip()
        if not self.content:
            raise ValueError("Memory-Inhalt darf nicht leer sein")
        if len(self.content) > MAX_CONTENT_CHARS:
            raise ValueError(f"Memory-Inhalt zu lang ({len(self.content)} > {MAX_CONTENT_CHARS})")
        if not self.scope:
            raise ValueError("scope darf nicht leer sein")
        if not 0.0 <= self.importance <= 1.0:
            raise ValueError("importance muss in [0, 1] liegen")

    @property
    def content_hash(self) -> str:
        return content_hash(self.content)

    def is_expired(self, now: datetime) -> bool:
        return self.expires_at is not None and self.expires_at <= now

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "layer": self.layer.value,
            "scope": self.scope,
            "kind": self.kind.value,
            "importance": self.importance,
            "source": self.source.value,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_accessed_at": self.last_accessed_at.isoformat(),
            "access_count": self.access_count,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MemoryItem:
        def dt(value: Any) -> datetime:
            return datetime.fromisoformat(value) if isinstance(value, str) else value

        return cls(
            content=data["content"],
            layer=MemoryLayer(data["layer"]),
            scope=data["scope"],
            kind=MemoryKind(data.get("kind", MemoryKind.NOTE)),
            importance=float(data.get("importance", 0.5)),
            source=MemorySource(data.get("source", MemorySource.AGENT)),
            tags=list(data.get("tags", [])),
            metadata=dict(data.get("metadata", {})),
            id=data["id"],
            created_at=dt(data["created_at"]),
            updated_at=dt(data["updated_at"]),
            last_accessed_at=dt(data["last_accessed_at"]),
            access_count=int(data.get("access_count", 0)),
            expires_at=dt(data["expires_at"]) if data.get("expires_at") else None,
        )


@dataclass(frozen=True)
class RetrievalWeights:
    relevance: float = 0.6
    importance: float = 0.25
    recency: float = 0.15

    def __post_init__(self) -> None:
        if min(self.relevance, self.importance, self.recency) < 0:
            raise ValueError("Gewichte dürfen nicht negativ sein")
        if self.relevance + self.importance + self.recency <= 0:
            raise ValueError("Mindestens ein Gewicht muss > 0 sein")


@dataclass(frozen=True)
class ScoredMemory:
    item: MemoryItem
    score: float
    relevance: float
    importance: float
    recency: float

    def explain(self) -> str:
        return (
            f"score={self.score:.2f} (relevanz={self.relevance:.2f}, "
            f"wichtigkeit={self.importance:.2f}, aktualität={self.recency:.2f})"
        )


def score_memory(
    item: MemoryItem,
    relevance: float,
    now: datetime,
    half_life_hours: float,
    weights: RetrievalWeights,
) -> ScoredMemory:
    recency = recency_score(item.last_accessed_at, now, half_life_hours)
    total = weights.relevance + weights.importance + weights.recency
    score = (
        weights.relevance * relevance
        + weights.importance * item.importance
        + weights.recency * recency
    ) / total
    return ScoredMemory(item, score, relevance, item.importance, recency)


# --------------------------------------------------------------------------- Store


class MemoryStore(abc.ABC):
    """Persistenz-Backend. Alle Methoden sind scope- und schichtbewusst."""

    @abc.abstractmethod
    async def add(self, item: MemoryItem) -> MemoryItem: ...

    @abc.abstractmethod
    async def get(self, item_id: str) -> MemoryItem | None: ...

    @abc.abstractmethod
    async def save(self, item: MemoryItem) -> MemoryItem:
        """Überschreibt ein vorhandenes Item vollständig."""

    @abc.abstractmethod
    async def delete(self, item_id: str) -> bool: ...

    @abc.abstractmethod
    async def list(
        self,
        layer: MemoryLayer,
        scope: str | None = None,
        *,
        kinds: Iterable[MemoryKind] | None = None,
        limit: int | None = None,
    ) -> list[MemoryItem]:
        """Neueste zuerst."""

    @abc.abstractmethod
    async def candidates(
        self,
        query: str,
        layer: MemoryLayer,
        scope: str | None = None,
        *,
        kinds: Iterable[MemoryKind] | None = None,
        limit: int = 200,
    ) -> builtins.list[MemoryItem]:
        """Vorauswahl möglicher Treffer (das Ranking erfolgt einheitlich im Aufrufer)."""

    @abc.abstractmethod
    async def find_by_hash(
        self, layer: MemoryLayer, scope: str, digest: str
    ) -> MemoryItem | None: ...

    @abc.abstractmethod
    async def delete_expired(self, now: datetime) -> int: ...

    @abc.abstractmethod
    async def count(self, layer: MemoryLayer, scope: str | None = None) -> int: ...

    async def touch(self, items: Iterable[MemoryItem], now: datetime) -> None:
        """Zugriff vermerken (Grundlage für Aktualität und Häufigkeit)."""
        for item in items:
            item.last_accessed_at = now
            item.access_count += 1
            await self.save(item)

    async def close(self) -> None:  # noqa: B027 – optional
        """Ressourcen freigeben."""


class InMemoryStore(MemoryStore):
    """Speicher im Prozess (Working Memory, Tests). Gleiche Semantik wie SQLite-Store."""

    def __init__(self) -> None:
        self._items: dict[str, MemoryItem] = {}

    @staticmethod
    def _copy(item: MemoryItem) -> MemoryItem:
        return replace(item, tags=list(item.tags), metadata=dict(item.metadata))

    def _select(
        self, layer: MemoryLayer, scope: str | None, kinds: Iterable[MemoryKind] | None
    ) -> list[MemoryItem]:
        kind_set = set(kinds) if kinds is not None else None
        return [
            i
            for i in self._items.values()
            if i.layer == layer
            and (scope is None or i.scope == scope)
            and (kind_set is None or i.kind in kind_set)
        ]

    async def add(self, item: MemoryItem) -> MemoryItem:
        if item.id in self._items:
            raise MemoryStoreError(f"Memory {item.id} existiert bereits")
        self._items[item.id] = self._copy(item)
        return item

    async def get(self, item_id: str) -> MemoryItem | None:
        item = self._items.get(item_id)
        return self._copy(item) if item else None

    async def save(self, item: MemoryItem) -> MemoryItem:
        if item.id not in self._items:
            raise MemoryNotFoundError(item.id)
        self._items[item.id] = self._copy(item)
        return item

    async def delete(self, item_id: str) -> bool:
        return self._items.pop(item_id, None) is not None

    async def list(
        self,
        layer: MemoryLayer,
        scope: str | None = None,
        *,
        kinds: Iterable[MemoryKind] | None = None,
        limit: int | None = None,
    ) -> list[MemoryItem]:
        items = sorted(self._select(layer, scope, kinds), key=lambda i: i.created_at, reverse=True)
        return [self._copy(i) for i in items[:limit]]

    async def candidates(
        self,
        query: str,
        layer: MemoryLayer,
        scope: str | None = None,
        *,
        kinds: Iterable[MemoryKind] | None = None,
        limit: int = 200,
    ) -> builtins.list[MemoryItem]:
        q = tokenize(query)
        hits = [
            i
            for i in self._select(layer, scope, kinds)
            if lexical_relevance(q, tokenize(i.content + " " + " ".join(i.tags))) > 0
        ]
        return [self._copy(i) for i in hits[:limit]]

    async def find_by_hash(self, layer: MemoryLayer, scope: str, digest: str) -> MemoryItem | None:
        for item in self._select(layer, scope, None):
            if item.content_hash == digest:
                return self._copy(item)
        return None

    async def delete_expired(self, now: datetime) -> int:
        expired = [i.id for i in self._items.values() if i.is_expired(now)]
        for item_id in expired:
            del self._items[item_id]
        return len(expired)

    async def count(self, layer: MemoryLayer, scope: str | None = None) -> int:
        return len(self._select(layer, scope, None))


# --------------------------------------------------------------------------- Schicht


class LayerMemory:
    """Eine Gedächtnisschicht über einem :class:`MemoryStore`.

    Unterklassen legen ``layer`` und Defaults fest. Alle Operationen sind auf diese Schicht
    beschränkt; der Scope (Task, Session, Projekt, ``global``) wird je Aufruf übergeben.
    """

    layer: ClassVar[MemoryLayer]
    default_half_life_hours: ClassVar[float]
    default_ttl: ClassVar[timedelta | None] = None
    default_capacity: ClassVar[int | None] = None

    def __init__(
        self,
        store: MemoryStore,
        *,
        clock: Clock = utc_now,
        half_life_hours: float | None = None,
        ttl: timedelta | bool | None = True,
        capacity: int | bool | None = True,
        weights: RetrievalWeights | None = None,
    ) -> None:
        self.store = store
        self.clock = clock
        self.half_life_hours = half_life_hours or self.default_half_life_hours
        self.ttl = self.default_ttl if ttl is True else (ttl or None)
        self.capacity = self.default_capacity if capacity is True else (capacity or None)
        self.weights = weights or RetrievalWeights()

    # ------------------------------------------------------------------ CRUD

    async def add(
        self,
        content: str,
        scope: str,
        *,
        kind: MemoryKind = MemoryKind.NOTE,
        importance: float = 0.5,
        source: MemorySource = MemorySource.AGENT,
        tags: Sequence[str] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> tuple[MemoryItem, bool]:
        """Speichert eine Erinnerung. Rückgabe: (Item, neu angelegt?).

        Identischer Inhalt im selben Scope wird nicht doppelt gespeichert, sondern
        aufgefrischt (Wichtigkeit = Maximum, Zugriff vermerkt).
        """
        if contains_secret(content):
            raise MemoryStoreError("Inhalt enthält ein Secret und wird nicht gespeichert")
        now = self.clock()
        existing = await self.store.find_by_hash(self.layer, scope, content_hash(content))
        if existing is not None:
            existing.importance = max(existing.importance, importance)
            existing.tags = sorted(set(existing.tags) | set(tags))
            existing.updated_at = existing.last_accessed_at = now
            existing.access_count += 1
            if self.ttl is not None:
                existing.expires_at = now + self.ttl
            await self.store.save(existing)
            return existing, False
        item = MemoryItem(
            content=content,
            layer=self.layer,
            scope=scope,
            kind=kind,
            importance=importance,
            source=source,
            tags=sorted(set(tags)),
            metadata=dict(metadata or {}),
            created_at=now,
            updated_at=now,
            last_accessed_at=now,
            expires_at=now + self.ttl if self.ttl is not None else None,
        )
        await self.store.add(item)
        if self.capacity is not None:
            await self.prune(scope)
        return item, True

    async def get(self, item_id: str) -> MemoryItem | None:
        item = await self.store.get(item_id)
        if item is None or item.layer != self.layer or item.is_expired(self.clock()):
            return None
        return item

    async def update(
        self,
        item_id: str,
        *,
        content: str | None = None,
        importance: float | None = None,
        kind: MemoryKind | None = None,
        tags: Sequence[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> MemoryItem:
        item = await self.get(item_id)
        if item is None:
            raise MemoryNotFoundError(f"Memory {item_id} nicht gefunden in {self.layer.value}")
        if content is not None:
            if contains_secret(content):
                raise MemoryStoreError("Inhalt enthält ein Secret und wird nicht gespeichert")
            item.content = content
        if importance is not None:
            item.importance = importance
        if kind is not None:
            item.kind = MemoryKind(kind)
        if tags is not None:
            item.tags = sorted(set(tags))
        if metadata is not None:
            item.metadata = {**item.metadata, **metadata}
        item.updated_at = self.clock()
        item.__post_init__()  # Validierung erneut ausführen
        await self.store.save(item)
        return item

    async def delete(self, item_id: str) -> bool:
        item = await self.store.get(item_id)
        if item is None or item.layer != self.layer:
            return False
        return await self.store.delete(item_id)

    async def list(
        self,
        scope: str | None = None,
        *,
        kinds: Iterable[MemoryKind] | None = None,
        limit: int | None = None,
    ) -> list[MemoryItem]:
        now = self.clock()
        items = await self.store.list(self.layer, scope, kinds=kinds, limit=None)
        live = [i for i in items if not i.is_expired(now)]
        return live[:limit]

    async def clear(self, scope: str) -> int:
        items = await self.store.list(self.layer, scope)
        for item in items:
            await self.store.delete(item.id)
        return len(items)

    # ------------------------------------------------------------------ Suche

    async def search(
        self,
        query: str,
        scope: str | None = None,
        *,
        kinds: Iterable[MemoryKind] | None = None,
        limit: int = 10,
        min_relevance: float = 0.01,
    ) -> builtins.list[ScoredMemory]:
        """Rangliste nach Relevanz × Wichtigkeit × Aktualität. Verändert keine Zugriffsdaten."""
        now = self.clock()
        q = tokenize(query)
        scored = []
        for item in await self.store.candidates(query, self.layer, scope, kinds=kinds):
            if item.is_expired(now):
                continue
            relevance = lexical_relevance(q, tokenize(item.content + " " + " ".join(item.tags)))
            if relevance >= min_relevance:
                scored.append(
                    score_memory(item, relevance, now, self.half_life_hours, self.weights)
                )
        scored.sort(key=lambda s: (s.score, s.item.created_at), reverse=True)
        return scored[:limit]

    # ------------------------------------------------------------------ Pflege

    def value(self, item: MemoryItem, now: datetime) -> float:
        """Behaltenswert für Pruning: Wichtigkeit × Aktualität, plus Nutzungshäufigkeit."""
        frequency = min(math.log1p(item.access_count) / math.log(10), 1.0)
        recency = recency_score(item.last_accessed_at, now, self.half_life_hours)
        return item.importance * (0.5 + 0.5 * recency) + 0.2 * frequency

    async def prune(self, scope: str) -> builtins.list[str]:
        """Entfernt die Einträge mit dem geringsten Behaltenswert über der Kapazität."""
        if self.capacity is None:
            return []
        items = await self.store.list(self.layer, scope)
        excess = len(items) - self.capacity
        if excess <= 0:
            return []
        now = self.clock()
        victims = sorted(items, key=lambda i: (self.value(i, now), i.created_at))[:excess]
        for item in victims:
            await self.store.delete(item.id)
        return [v.id for v in victims]
