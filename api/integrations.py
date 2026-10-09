"""Integrationen: Schlüssel, Berechtigungen (Scopes), Rate Limits.

* Ein Schlüssel ``nova_<präfix>_<geheimnis>`` wird **einmal** im Klartext ausgegeben und nur
  als SHA-256-Hash gespeichert (256 Bit Zufall – ein schneller Hash genügt, es gibt nichts zu
  erraten). Vergleich in konstanter Zeit.
* Jede Integration hat eine **explizite** Scope-Liste. Standard ist nichts.
* Agent-Tools sind privilegiert: Ein Agent-Task darf ein Tool nur nutzen, wenn die Integration
  den Scope ``agent:tool:<toolname>`` besitzt **und** der Task es anfordert.
* Widerrufen, Rotieren (alter Schlüssel sofort ungültig), Ablaufdatum optional.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
import sqlite3
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCOPES: dict[str, str] = {
    "chat:complete": "Chat-Antworten erzeugen (/api/v1/chat/completions, /v1/chat/completions)",
    "models:read": "Modelle und Modellstatus lesen",
    "agent:run": "Agent-Aufgaben starten",
    "agent:read": "Status und Ergebnis eigener Agent-Aufgaben lesen",
    "agent:cancel": "Eigene Agent-Aufgaben abbrechen",
    "system:read": "Systemstatus lesen (Hardware, Versionen, Auslastung)",
}
TOOL_SCOPE_PREFIX = "agent:tool:"
"""Privilegiert: ``agent:tool:<name>`` erlaubt einem Agent-Task genau dieses Tool."""

_KEY_RE = re.compile(r"^nova_([0-9a-f]{12})_([A-Za-z0-9_-]{40,})$")
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._-]{0,63}$")

SCHEMA = """
CREATE TABLE IF NOT EXISTS integrations (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    key_prefix TEXT NOT NULL UNIQUE,
    key_hash TEXT NOT NULL,
    scopes TEXT NOT NULL,
    rate_limit_per_minute INTEGER NOT NULL,
    max_request_bytes INTEGER NOT NULL,
    allowed_models TEXT NOT NULL DEFAULT '[]',
    agent_workspace TEXT,
    created_at TEXT NOT NULL,
    expires_at TEXT,
    revoked_at TEXT,
    last_used_at TEXT
);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


class IntegrationError(ValueError):
    pass


def validate_scopes(scopes: list[str], known_tools: set[str] | None = None) -> list[str]:
    cleaned = sorted(set(scopes))
    for scope in cleaned:
        if scope in SCOPES:
            continue
        if scope.startswith(TOOL_SCOPE_PREFIX):
            tool = scope[len(TOOL_SCOPE_PREFIX) :]
            if known_tools is not None and tool not in known_tools:
                raise IntegrationError(f"Unknown tool in scope {scope!r}")
            if not tool:
                raise IntegrationError("Empty tool scope")
            continue
        raise IntegrationError(f"Unknown scope {scope!r} (known: {', '.join(sorted(SCOPES))})")
    return cleaned


@dataclass
class Integration:
    id: str
    name: str
    key_prefix: str
    scopes: list[str]
    rate_limit_per_minute: int
    max_request_bytes: int
    allowed_models: list[str] = field(default_factory=list)
    """Leer = alle Modelle (einschließlich ``auto``)."""
    agent_workspace: str | None = None
    created_at: str = ""
    expires_at: str | None = None
    revoked_at: str | None = None
    last_used_at: str | None = None

    @property
    def active(self) -> bool:
        if self.revoked_at:
            return False
        return not (self.expires_at and self.expires_at <= _now())

    def has(self, scope: str) -> bool:
        return scope in self.scopes

    @property
    def tool_scopes(self) -> set[str]:
        return {s[len(TOOL_SCOPE_PREFIX) :] for s in self.scopes if s.startswith(TOOL_SCOPE_PREFIX)}

    def model_allowed(self, model: str) -> bool:
        return not self.allowed_models or model in self.allowed_models

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "key_prefix": f"nova_{self.key_prefix}_…",
            "scopes": self.scopes,
            "rate_limit_per_minute": self.rate_limit_per_minute,
            "max_request_bytes": self.max_request_bytes,
            "allowed_models": self.allowed_models,
            "agent_workspace": self.agent_workspace,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "revoked_at": self.revoked_at,
            "last_used_at": self.last_used_at,
            "active": self.active,
        }


class IntegrationStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(SCHEMA)
        self._lock = threading.Lock()

    def close(self) -> None:
        self._db.close()

    @staticmethod
    def _row(row: sqlite3.Row) -> Integration:
        return Integration(
            id=row["id"],
            name=row["name"],
            key_prefix=row["key_prefix"],
            scopes=json.loads(row["scopes"]),
            rate_limit_per_minute=row["rate_limit_per_minute"],
            max_request_bytes=row["max_request_bytes"],
            allowed_models=json.loads(row["allowed_models"]),
            agent_workspace=row["agent_workspace"],
            created_at=row["created_at"],
            expires_at=row["expires_at"],
            revoked_at=row["revoked_at"],
            last_used_at=row["last_used_at"],
        )

    @staticmethod
    def _new_key() -> tuple[str, str]:
        prefix = secrets.token_hex(6)
        return prefix, f"nova_{prefix}_{secrets.token_urlsafe(32)}"

    def create(
        self,
        name: str,
        scopes: list[str],
        *,
        rate_limit_per_minute: int = 60,
        max_request_bytes: int = 1_000_000,
        allowed_models: list[str] | None = None,
        agent_workspace: str | None = None,
        expires_at: str | None = None,
        known_tools: set[str] | None = None,
    ) -> tuple[Integration, str]:
        """Legt eine Integration an; gibt (Integration, Klartextschlüssel – nur jetzt!) zurück."""
        if not _NAME_RE.match(name):
            raise IntegrationError("Name: 1–64 characters, letters, digits, space . _ -")
        scopes = validate_scopes(scopes, known_tools)
        if not scopes:
            raise IntegrationError("At least one scope is required")
        if not 1 <= rate_limit_per_minute <= 10_000:
            raise IntegrationError("rate_limit_per_minute must be between 1 and 10000")
        if not 1_024 <= max_request_bytes <= 50_000_000:
            raise IntegrationError("max_request_bytes must be between 1 KB and 50 MB")
        if agent_workspace is not None and not Path(agent_workspace).is_dir():
            raise IntegrationError(f"Agent workspace does not exist: {agent_workspace}")
        prefix, key = self._new_key()
        integration = Integration(
            id=uuid.uuid4().hex[:12],
            name=name,
            key_prefix=prefix,
            scopes=scopes,
            rate_limit_per_minute=rate_limit_per_minute,
            max_request_bytes=max_request_bytes,
            allowed_models=sorted(set(allowed_models or [])),
            agent_workspace=agent_workspace,
            created_at=_now(),
            expires_at=expires_at,
        )
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO integrations (id, name, key_prefix, key_hash, scopes, "
                "rate_limit_per_minute, max_request_bytes, allowed_models, agent_workspace, "
                "created_at, expires_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    integration.id,
                    name,
                    prefix,
                    _hash(key),
                    json.dumps(scopes),
                    rate_limit_per_minute,
                    max_request_bytes,
                    json.dumps(integration.allowed_models),
                    agent_workspace,
                    integration.created_at,
                    expires_at,
                ),
            )
        return integration, key

    def list_all(self) -> list[Integration]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM integrations ORDER BY created_at").fetchall()
        return [self._row(r) for r in rows]

    def get(self, integration_id: str) -> Integration:
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM integrations WHERE id = ? OR name = ?",
                (integration_id, integration_id),
            ).fetchone()
        if row is None:
            raise IntegrationError(f"Integration not found: {integration_id}")
        return self._row(row)

    def authenticate(self, key: str | None) -> Integration | None:
        """Prüft einen Schlüssel; ``None`` bei unbekannt, falsch, widerrufen oder abgelaufen."""
        if not key:
            return None
        match = _KEY_RE.match(key.strip())
        if not match:
            return None
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM integrations WHERE key_prefix = ?", (match.group(1),)
            ).fetchone()
        # Auch bei unbekanntem Präfix hashen: gleiche Laufzeit, kein Orakel
        expected = row["key_hash"] if row is not None else _hash("nova_000000000000_x")
        if not hmac.compare_digest(expected, _hash(key.strip())) or row is None:
            return None
        integration = self._row(row)
        if not integration.active:
            return None
        with self._lock, self._db:
            self._db.execute(
                "UPDATE integrations SET last_used_at = ? WHERE id = ?", (_now(), integration.id)
            )
        return integration

    def revoke(self, integration_id: str) -> Integration:
        integration = self.get(integration_id)
        with self._lock, self._db:
            self._db.execute(
                "UPDATE integrations SET revoked_at = ? WHERE id = ?", (_now(), integration.id)
            )
        return self.get(integration.id)

    def rotate(self, integration_id: str) -> tuple[Integration, str]:
        """Neuer Schlüssel, alter sofort ungültig; Scopes und Limits bleiben."""
        integration = self.get(integration_id)
        if integration.revoked_at:
            raise IntegrationError("Revoked integrations cannot be rotated")
        prefix, key = self._new_key()
        with self._lock, self._db:
            self._db.execute(
                "UPDATE integrations SET key_prefix = ?, key_hash = ? WHERE id = ?",
                (prefix, _hash(key), integration.id),
            )
        return self.get(integration.id), key

    def update_scopes(
        self, integration_id: str, scopes: list[str], known_tools: set[str] | None = None
    ) -> Integration:
        integration = self.get(integration_id)
        cleaned = validate_scopes(scopes, known_tools)
        with self._lock, self._db:
            self._db.execute(
                "UPDATE integrations SET scopes = ? WHERE id = ?",
                (json.dumps(cleaned), integration.id),
            )
        return self.get(integration.id)


class RateLimiter:
    """Gleitendes Fenster (60 s) je Integration – im Speicher, pro Prozess."""

    def __init__(self, clock: Any = time.monotonic) -> None:
        self.clock = clock
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, per_minute: int) -> tuple[bool, float]:
        """(erlaubt, Sekunden bis zum nächsten freien Platz)."""
        now = self.clock()
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and now - hits[0] >= 60:
                hits.popleft()
            if len(hits) >= per_minute:
                return False, 60 - (now - hits[0])
            hits.append(now)
            return True, 0.0
