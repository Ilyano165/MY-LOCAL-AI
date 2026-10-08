"""Project Memory: projektbezogene Informationen.

Scope = Projekt-ID (stabil aus dem Projektpfad abgeleitet). Enthält Entscheidungen,
Konventionen, Fakten über die Codebasis, Aufgabenergebnisse und Lehren aus Fehlern. Kein
automatisches Ablaufen; begrenzt durch Kapazität mit Pruning nach Behaltenswert.
"""

from __future__ import annotations

import hashlib
import re
from datetime import timedelta
from pathlib import Path
from typing import ClassVar

from memory.base import LayerMemory, MemoryLayer


def project_id_for(path: str | Path) -> str:
    """Stabile, lesbare Projekt-ID: ``<ordnername>-<hash des absoluten Pfads>``."""
    resolved = Path(path).expanduser().resolve()
    slug = re.sub(r"[^a-z0-9]+", "-", resolved.name.lower()).strip("-") or "projekt"
    digest = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:8]
    return f"{slug}-{digest}"


class ProjectMemory(LayerMemory):
    layer: ClassVar[MemoryLayer] = MemoryLayer.PROJECT
    default_half_life_hours: ClassVar[float] = 24.0 * 30
    default_ttl: ClassVar[timedelta | None] = None
    default_capacity: ClassVar[int | None] = 5000
