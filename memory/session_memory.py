"""Session Memory: Informationen aus der aktuellen Unterhaltung.

Scope = Session-ID. Enthält das Gesprächsprotokoll (Nachrichten, Secrets geschwärzt) und
sessionbezogene Notizen. Läuft nach ``ttl`` (Default 7 Tage) ab; dauerhaft Relevantes wird vom
MemoryManager beim Schreiben in Projekt- oder Langzeit-Memory übernommen.
"""

from __future__ import annotations

from datetime import timedelta
from typing import ClassVar

from memory.base import (
    LayerMemory,
    MemoryItem,
    MemoryKind,
    MemoryLayer,
    MemorySource,
    redact_secrets,
)

_ROLE_SOURCE = {
    "user": MemorySource.USER,
    "assistant": MemorySource.AGENT,
    "tool": MemorySource.TOOL,
    "system": MemorySource.SYSTEM,
}


class SessionMemory(LayerMemory):
    layer: ClassVar[MemoryLayer] = MemoryLayer.SESSION
    default_half_life_hours: ClassVar[float] = 24.0
    default_ttl: ClassVar[timedelta | None] = timedelta(days=7)
    default_capacity: ClassVar[int | None] = 1000

    async def add_message(
        self, session_id: str, role: str, content: str, importance: float = 0.1
    ) -> MemoryItem | None:
        """Protokolliert eine Nachricht (geschwärzt). Leere Nachrichten werden ignoriert."""
        text = redact_secrets(content).strip()
        if not text:
            return None
        source = _ROLE_SOURCE.get(role, MemorySource.USER)
        # Gleiche Nachricht zweimal ist ein eigenes Ereignis → Zeitstempel macht den Inhalt
        # nicht eindeutig; daher Rolle + Zeit in die Metadaten, Deduplizierung nur bei Notizen.
        item = MemoryItem(
            content=text,
            layer=self.layer,
            scope=session_id,
            kind=MemoryKind.MESSAGE,
            importance=importance,
            source=source,
            metadata={"role": role},
            created_at=self.clock(),
            updated_at=self.clock(),
            last_accessed_at=self.clock(),
            expires_at=self.clock() + self.ttl if self.ttl else None,
        )
        await self.store.add(item)
        if self.capacity is not None:
            await self.prune(session_id)
        return item

    async def history(self, session_id: str, limit: int = 20) -> list[MemoryItem]:
        """Letzte Nachrichten in chronologischer Reihenfolge."""
        items = await self.list(session_id, kinds=[MemoryKind.MESSAGE], limit=limit)
        return list(reversed(items))
