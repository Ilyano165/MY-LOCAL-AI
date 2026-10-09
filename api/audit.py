"""Audit-Log für sicherheitsrelevante Aktionen (JSON Lines).

Protokolliert werden Ereignisse, **keine Inhalte**: kein Prompt, keine Antwort, keine
Schlüssel – nur Schlüsselpräfixe, Integrations-IDs, Scopes, Statuscodes und Client-Adressen.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("nova.audit")
_FORBIDDEN_KEYS = {"key", "api_key", "authorization", "prompt", "messages", "content", "token"}


class AuditLog:
    def __init__(self, path: Path | None, *, max_bytes: int = 10_000_000) -> None:
        self.path = path
        self.max_bytes = max_bytes
        self._lock = threading.Lock()
        self.recent: list[dict[str, Any]] = []

    def record(self, event: str, **fields: Any) -> dict[str, Any]:
        clean = {k: v for k, v in fields.items() if k.lower() not in _FORBIDDEN_KEYS}
        entry = {"ts": datetime.now(UTC).isoformat(timespec="seconds"), "event": event, **clean}
        with self._lock:
            self.recent = [*self.recent[-199:], entry]
            if self.path is not None:
                try:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    if self.path.exists() and self.path.stat().st_size > self.max_bytes:
                        self.path.replace(self.path.with_suffix(".1.jsonl"))
                    with self.path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
                except OSError as exc:
                    logger.warning("Audit-Log nicht schreibbar: %s", exc)
        return entry
