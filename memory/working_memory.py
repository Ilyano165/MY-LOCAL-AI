"""Working Memory: Informationen für die *aktuelle* Aufgabe.

Lebt nur im Prozess (kein Persistieren), Scope = Task-ID, kleine Kapazität, kurze
Halbwertszeit. Wird am Ende einer Aufgabe geleert; was davon dauerhaft relevant ist, muss
vorher explizit über den :class:`~memory.memory_manager.MemoryManager` gesichert werden.
"""

from __future__ import annotations

from datetime import timedelta
from typing import ClassVar

from memory.base import InMemoryStore, LayerMemory, MemoryLayer, MemoryStore


class WorkingMemory(LayerMemory):
    layer: ClassVar[MemoryLayer] = MemoryLayer.WORKING
    default_half_life_hours: ClassVar[float] = 1.0
    default_ttl: ClassVar[timedelta | None] = timedelta(hours=12)
    default_capacity: ClassVar[int | None] = 100

    def __init__(self, store: MemoryStore | None = None, **kwargs: object) -> None:
        super().__init__(store or InMemoryStore(), **kwargs)  # type: ignore[arg-type]
