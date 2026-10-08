"""Long-Term Memory: dauerhaft relevante Informationen über den Nutzer und seine Arbeitsweise.

Scope = ``global``. Enthält Präferenzen, dauerhafte Anweisungen und allgemeine Fakten. Lange
Halbwertszeit, kein Ablaufen, Kapazitätsgrenze mit Pruning. Präferenzen und Anweisungen
werden beim Abruf unabhängig von der Anfrage mitgeliefert (siehe :meth:`active_guidance`).
"""

from __future__ import annotations

from datetime import timedelta
from typing import ClassVar

from memory.base import LayerMemory, MemoryItem, MemoryKind, MemoryLayer

GLOBAL_SCOPE = "global"
GUIDANCE_KINDS = (MemoryKind.PREFERENCE, MemoryKind.INSTRUCTION)


class LongTermMemory(LayerMemory):
    layer: ClassVar[MemoryLayer] = MemoryLayer.LONG_TERM
    default_half_life_hours: ClassVar[float] = 24.0 * 180
    default_ttl: ClassVar[timedelta | None] = None
    default_capacity: ClassVar[int | None] = 10_000

    async def active_guidance(self, limit: int = 10) -> list[MemoryItem]:
        """Präferenzen und Anweisungen, wichtigste und neueste zuerst."""
        items = await self.list(GLOBAL_SCOPE, kinds=GUIDANCE_KINDS)
        items.sort(key=lambda i: (i.importance, i.updated_at), reverse=True)
        return items[:limit]
