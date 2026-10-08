"""Memory Retrieval: relevante Erinnerungen über alle Schichten finden.

Score je Erinnerung (vgl. Park et al. 2023): gewichtete Summe aus
* **Relevanz** – lexikalische Übereinstimmung mit der Anfrage (0..1),
* **Wichtigkeit** – beim Speichern bewertet (0..1),
* **Aktualität** – exponentieller Zerfall seit dem letzten Zugriff, Halbwertszeit je Schicht.

Ohne Mindestrelevanz wird nichts zurückgegeben: eine wichtige, aber themenfremde Erinnerung
soll den Kontext nicht verstopfen. Ausnahme sind aktive Präferenzen/Anweisungen aus dem
Langzeit-Memory, die immer gelten (``include_guidance``).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from memory.base import (
    Clock,
    LayerMemory,
    MemoryItem,
    MemoryKind,
    MemoryLayer,
    ScoredMemory,
    utc_now,
)
from memory.long_term_memory import GLOBAL_SCOPE, LongTermMemory

# Gleichstandsauflösung: bei (nahezu) gleichem Score gewinnt die spezifischere Schicht.
LAYER_PRIORITY = {
    MemoryLayer.WORKING: 0.03,
    MemoryLayer.SESSION: 0.02,
    MemoryLayer.PROJECT: 0.01,
    MemoryLayer.LONG_TERM: 0.0,
}


@dataclass
class RecallResult:
    query: str
    memories: list[ScoredMemory] = field(default_factory=list)
    guidance: list[MemoryItem] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not self.memories and not self.guidance

    def as_prompt(self, max_chars: int = 3000) -> str:
        """Kompakter Kontextblock für Modell-Prompts."""
        lines: list[str] = []
        if self.guidance:
            lines.append("User preferences and standing instructions:")
            lines += [f"- {g.content}" for g in self.guidance]
        if self.memories:
            lines.append("Possibly relevant memories (may be outdated – verify before relying):")
            lines += [
                f"- [{m.item.layer.value}/{m.item.kind.value}] {m.item.content}"
                for m in self.memories
            ]
        text = "\n".join(lines)
        return text if len(text) <= max_chars else text[:max_chars] + "…"


class MemoryRetriever:
    def __init__(
        self,
        layers: Mapping[MemoryLayer, LayerMemory],
        *,
        clock: Clock = utc_now,
        min_relevance: float = 0.3,
    ) -> None:
        self.layers = dict(layers)
        self.clock = clock
        self.min_relevance = min_relevance

    async def retrieve(
        self,
        query: str,
        scopes: Mapping[MemoryLayer, str],
        *,
        limit: int = 8,
        kinds: Iterable[MemoryKind] | None = None,
        include_guidance: bool = True,
        guidance_limit: int = 10,
        touch: bool = True,
    ) -> RecallResult:
        """Durchsucht alle Schichten, für die ein Scope bekannt ist."""
        kind_list = list(kinds) if kinds is not None else None
        result = RecallResult(query)
        if include_guidance and MemoryLayer.LONG_TERM in self.layers:
            long_term = self.layers[MemoryLayer.LONG_TERM]
            if isinstance(long_term, LongTermMemory):
                result.guidance = await long_term.active_guidance(guidance_limit)
        guidance_ids = {g.id for g in result.guidance}

        scored: list[ScoredMemory] = []
        for layer, scope in scopes.items():
            memory = self.layers.get(layer)
            if memory is None:
                continue
            hits = await memory.search(
                query, scope, kinds=kind_list, limit=limit * 3, min_relevance=self.min_relevance
            )
            scored += [h for h in hits if h.item.id not in guidance_ids]

        # Doppelte Inhalte über Schichten hinweg: die spezifischere Schicht gewinnt.
        scored.sort(key=lambda s: s.score + LAYER_PRIORITY[s.item.layer], reverse=True)
        seen: set[str] = set()
        unique: list[ScoredMemory] = []
        for hit in scored:
            if hit.item.content_hash in seen or hit.item.id in guidance_ids:
                continue
            seen.add(hit.item.content_hash)
            unique.append(hit)
        result.memories = unique[:limit]

        if touch:
            now = self.clock()
            for hit in result.memories:
                await self.layers[hit.item.layer].store.touch([hit.item], now)
        return result


def default_scopes(
    session_id: str | None, project_id: str | None, task_id: str | None
) -> dict[MemoryLayer, str]:
    scopes = {MemoryLayer.LONG_TERM: GLOBAL_SCOPE}
    if project_id:
        scopes[MemoryLayer.PROJECT] = project_id
    if session_id:
        scopes[MemoryLayer.SESSION] = session_id
    if task_id:
        scopes[MemoryLayer.WORKING] = task_id
    return scopes
