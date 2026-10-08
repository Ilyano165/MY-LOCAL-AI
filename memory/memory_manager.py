"""MemoryManager: zentrale Schnittstelle des Memory-Systems.

* :meth:`remember` – bewertet Relevanz und entscheidet automatisch über die Schicht
  (oder speichert explizit in eine vorgegebene Schicht).
* :meth:`observe_message` – protokolliert eine Chat-Nachricht in der Session und übernimmt
  *nur* dauerhaft Relevantes in Projekt- oder Langzeit-Memory.
* :meth:`recall` – findet relevante Erinnerungen für eine neue Aufgabe.
* :meth:`get` / :meth:`update` / :meth:`delete` / :meth:`search` / :meth:`list` – Verwaltung.
* :meth:`record_outcome`, :meth:`end_task`, :meth:`maintenance` – Lebenszyklus.
"""

from __future__ import annotations

import builtins
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from memory.base import (
    Clock,
    LayerMemory,
    MemoryItem,
    MemoryKind,
    MemoryLayer,
    MemoryNotFoundError,
    MemorySource,
    MemoryStore,
    MemoryStoreError,
    RetrievalWeights,
    ScoredMemory,
    contains_secret,
    utc_now,
)
from memory.long_term_memory import GLOBAL_SCOPE, LongTermMemory
from memory.project_memory import ProjectMemory
from memory.relevance import (
    Assessment,
    HeuristicRelevanceAssessor,
    MemoryContext,
    RelevanceAssessor,
)
from memory.retrieval import MemoryRetriever, RecallResult, default_scopes
from memory.session_memory import SessionMemory
from memory.sqlite_store import SQLiteMemoryStore
from memory.working_memory import WorkingMemory


@dataclass
class StoreResult:
    stored: bool
    layer: MemoryLayer | None
    item: MemoryItem | None
    created: bool
    reasons: list[str] = field(default_factory=list)


class MemoryManager:
    def __init__(
        self,
        *,
        working: WorkingMemory,
        session: SessionMemory,
        project: ProjectMemory,
        long_term: LongTermMemory,
        assessor: RelevanceAssessor | None = None,
        clock: Clock = utc_now,
        min_relevance: float = 0.3,
    ) -> None:
        self.working = working
        self.session = session
        self.project = project
        self.long_term = long_term
        self.assessor = assessor or HeuristicRelevanceAssessor()
        self.clock = clock
        self.layers: dict[MemoryLayer, LayerMemory] = {
            MemoryLayer.WORKING: working,
            MemoryLayer.SESSION: session,
            MemoryLayer.PROJECT: project,
            MemoryLayer.LONG_TERM: long_term,
        }
        self.retriever = MemoryRetriever(self.layers, clock=clock, min_relevance=min_relevance)

    # ------------------------------------------------------------------ Fabriken

    @classmethod
    def with_store(
        cls,
        store: MemoryStore,
        *,
        clock: Clock = utc_now,
        weights: RetrievalWeights | None = None,
        **kwargs: Any,
    ) -> MemoryManager:
        """Session/Projekt/Langzeit auf ``store``; Working Memory im Prozess."""
        common: dict[str, Any] = {"clock": clock, "weights": weights}
        return cls(
            working=WorkingMemory(**common),
            session=SessionMemory(store, **common),
            project=ProjectMemory(store, **common),
            long_term=LongTermMemory(store, **common),
            clock=clock,
            **kwargs,
        )

    @classmethod
    def open(cls, path: str | Path = "~/.nova/memory.db", **kwargs: Any) -> MemoryManager:
        return cls.with_store(SQLiteMemoryStore(path), **kwargs)

    @classmethod
    def in_memory(cls, **kwargs: Any) -> MemoryManager:
        return cls.with_store(SQLiteMemoryStore(":memory:"), **kwargs)

    async def close(self) -> None:
        await self.long_term.store.close()
        await self.working.store.close()

    # ------------------------------------------------------------------ Scopes

    @staticmethod
    def scope_for(layer: MemoryLayer, ctx: MemoryContext) -> str | None:
        return {
            MemoryLayer.WORKING: ctx.task_id,
            MemoryLayer.SESSION: ctx.session_id,
            MemoryLayer.PROJECT: ctx.project_id,
            MemoryLayer.LONG_TERM: GLOBAL_SCOPE,
        }[layer]

    def _require_scope(self, layer: MemoryLayer, ctx: MemoryContext) -> str:
        scope = self.scope_for(layer, ctx)
        if not scope:
            raise MemoryStoreError(f"Für {layer.value} fehlt der Kontext (Task/Session/Projekt-ID)")
        return scope

    # ------------------------------------------------------------------ Speichern

    async def remember(
        self,
        content: str,
        ctx: MemoryContext,
        *,
        source: MemorySource = MemorySource.USER,
        layer: MemoryLayer | None = None,
        kind: MemoryKind | None = None,
        importance: float | None = None,
        tags: Sequence[str] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> StoreResult:
        """Speichert ``content`` nach Relevanzbewertung.

        Mit ``layer`` wird die automatische Schichtwahl übersteuert (z. B. Nutzer sagt
        „speichere das im Projekt“); die Secret-Prüfung gilt trotzdem.
        """
        assessment = self.assessor.assess(content, source, ctx)
        if contains_secret(content):
            return StoreResult(False, None, None, False, assessment.reasons)
        if layer is not None:
            assessment = Assessment(
                True,
                layer,
                kind or assessment.kind,
                importance if importance is not None else max(assessment.importance, 0.5),
                assessment.content,
                [*assessment.reasons, f"Schicht vorgegeben: {layer.value}"],
            )
        else:
            if kind is not None:
                assessment.kind = kind
            if importance is not None:
                assessment.importance = importance
        if not assessment.store or assessment.layer is None:
            return StoreResult(False, None, None, False, assessment.reasons)

        target = assessment.layer
        scope = self.scope_for(target, ctx)
        if not scope:
            # Fallback ohne passenden Kontext: Projektwissen ohne Projekt → Session.
            assessment.reasons.append(f"kein Kontext für {target.value}")
            if target is MemoryLayer.PROJECT and ctx.session_id:
                target, scope = MemoryLayer.SESSION, ctx.session_id
            else:
                return StoreResult(False, None, None, False, assessment.reasons)
        item, created = await self.layers[target].add(
            assessment.content,
            scope,
            kind=assessment.kind,
            importance=assessment.importance,
            source=source,
            tags=tags,
            metadata=metadata,
        )
        if not created:
            assessment.reasons.append("bereits bekannt – aufgefrischt statt dupliziert")
        return StoreResult(True, target, item, created, assessment.reasons)

    async def observe_message(self, role: str, content: str, ctx: MemoryContext) -> StoreResult:
        """Chat-Nachricht: immer ins Session-Protokoll (geschwärzt), dauerhaft nur wenn relevant."""
        if ctx.session_id:
            await self.session.add_message(ctx.session_id, role, content)
        source = {
            "user": MemorySource.USER,
            "assistant": MemorySource.AGENT,
            "tool": MemorySource.TOOL,
        }.get(role, MemorySource.SYSTEM)
        assessment = self.assessor.assess(content, source, ctx)
        if assessment.layer in (MemoryLayer.PROJECT, MemoryLayer.LONG_TERM):
            return await self.remember(content, ctx, source=source)
        return StoreResult(
            False, None, None, False, [*assessment.reasons, "nur im Session-Protokoll"]
        )

    async def note(
        self,
        content: str,
        ctx: MemoryContext,
        *,
        importance: float = 0.5,
        kind: MemoryKind = MemoryKind.NOTE,
        tags: Sequence[str] = (),
    ) -> MemoryItem:
        """Notiz im Working Memory der laufenden Aufgabe."""
        scope = self._require_scope(MemoryLayer.WORKING, ctx)
        item, _ = await self.working.add(
            content, scope, kind=kind, importance=importance, source=MemorySource.AGENT, tags=tags
        )
        return item

    # ------------------------------------------------------------------ Abrufen

    async def recall(
        self,
        query: str,
        ctx: MemoryContext,
        *,
        limit: int = 8,
        layers: Iterable[MemoryLayer] | None = None,
        kinds: Iterable[MemoryKind] | None = None,
        include_guidance: bool = True,
    ) -> RecallResult:
        """Relevante Erinnerungen für eine neue Aufgabe (vermerkt den Zugriff)."""
        scopes = default_scopes(ctx.session_id, ctx.project_id, ctx.task_id)
        if layers is not None:
            wanted = set(layers)
            scopes = {k: v for k, v in scopes.items() if k in wanted}
        return await self.retriever.retrieve(
            query, scopes, limit=limit, kinds=kinds, include_guidance=include_guidance
        )

    async def search(
        self, query: str, ctx: MemoryContext, *, layer: MemoryLayer | None = None, limit: int = 20
    ) -> list[ScoredMemory]:
        """Suche ohne Seiteneffekte (Zugriffsdaten bleiben unverändert)."""
        scopes = default_scopes(ctx.session_id, ctx.project_id, ctx.task_id)
        if layer is not None:
            scopes = {layer: scopes[layer]} if layer in scopes else {}
        result = await self.retriever.retrieve(
            query, scopes, limit=limit, include_guidance=False, touch=False
        )
        return result.memories

    # ------------------------------------------------------------------ Verwalten

    async def get(self, item_id: str) -> MemoryItem | None:
        for memory in self.layers.values():
            item = await memory.get(item_id)
            if item is not None:
                return item
        return None

    async def _layer_of(self, item_id: str) -> LayerMemory:
        item = await self.get(item_id)
        if item is None:
            raise MemoryNotFoundError(f"Memory {item_id} nicht gefunden")
        return self.layers[item.layer]

    async def update(self, item_id: str, **changes: Any) -> MemoryItem:
        return await (await self._layer_of(item_id)).update(item_id, **changes)

    async def delete(self, item_id: str) -> bool:
        try:
            return await (await self._layer_of(item_id)).delete(item_id)
        except MemoryNotFoundError:
            return False

    async def list(
        self,
        layer: MemoryLayer,
        ctx: MemoryContext,
        *,
        kinds: Iterable[MemoryKind] | None = None,
        limit: int | None = None,
    ) -> list[MemoryItem]:
        scope = self._require_scope(layer, ctx)
        return await self.layers[layer].list(scope, kinds=kinds, limit=limit)

    async def move(self, item_id: str, target: MemoryLayer, ctx: MemoryContext) -> MemoryItem:
        """Verschiebt eine Erinnerung in eine andere Schicht (z. B. Session → Projekt)."""
        item = await self.get(item_id)
        if item is None:
            raise MemoryNotFoundError(f"Memory {item_id} nicht gefunden")
        scope = self._require_scope(target, ctx)
        moved, _ = await self.layers[target].add(
            item.content,
            scope,
            kind=item.kind,
            importance=item.importance,
            source=item.source,
            tags=item.tags,
            metadata=item.metadata,
        )
        await self.layers[item.layer].delete(item.id)
        return moved

    # ------------------------------------------------------------------ Lebenszyklus

    async def record_outcome(
        self,
        objective: str,
        status: str,
        summary: str,
        ctx: MemoryContext,
        *,
        lessons: Sequence[str] = (),
        verified: bool = False,
    ) -> builtins.list[MemoryItem]:
        """Hält das Ergebnis einer Aufgabe fest (Projekt, sonst Session) – nie im Langzeit-Memory.

        Lehren (z. B. die Ursachenanalyse einer erfolgreichen Korrektur) werden nur aus
        verifizierten Ergebnissen übernommen; unbestätigte Vermutungen bleiben draußen.
        """
        layer = MemoryLayer.PROJECT if ctx.project_id else MemoryLayer.SESSION
        scope = self.scope_for(layer, ctx)
        if not scope:
            return []
        stored: list[MemoryItem] = []
        content = f"Aufgabe: {objective.strip()[:300]} → Status {status}. {summary.strip()[:500]}"
        if not contains_secret(content):
            item, _ = await self.layers[layer].add(
                content,
                scope,
                kind=MemoryKind.TASK_RESULT,
                importance=0.55 if verified else 0.4,
                source=MemorySource.AGENT,
                metadata={"status": status, "verified": verified},
            )
            stored.append(item)
        if verified:
            for lesson in lessons:
                if lesson.strip() and not contains_secret(lesson):
                    item, _ = await self.layers[layer].add(
                        lesson.strip()[:1000],
                        scope,
                        kind=MemoryKind.LESSON,
                        importance=0.6,
                        source=MemorySource.AGENT,
                        metadata={"from_task": objective[:120]},
                    )
                    stored.append(item)
        return stored

    async def end_task(self, ctx: MemoryContext) -> int:
        """Leert das Working Memory der Aufgabe."""
        return await self.working.clear(ctx.task_id) if ctx.task_id else 0

    async def maintenance(self) -> dict[str, int]:
        """Abgelaufene Einträge löschen (Session, Working)."""
        now = self.clock()
        removed = await self.long_term.store.delete_expired(now)
        removed_working = await self.working.store.delete_expired(now)
        return {"expired": removed + removed_working}
