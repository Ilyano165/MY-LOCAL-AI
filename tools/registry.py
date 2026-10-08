"""ToolRegistry: der einzige Weg, ein Tool auszuführen.

Ablauf jedes Aufrufs (:meth:`ToolRegistry.execute`):

1. Tool nachschlagen
2. Begründung (``reason``) prüfen – ohne Begründung keine Ausführung
3. Argumente gegen das JSON-Schema validieren
4. benötigte Berechtigungen bestimmen (je nach Argumenten)
5. :class:`PermissionPolicy` entscheidet: ALLOW / ASK / DENY; bei ASK fragt der
   :data:`ApprovalHandler` (ohne Handler → verweigert)
6. Ausführung mit hartem Timeout, jede Ausnahme → strukturierter Fehler
7. Ausgabe auf ``max_output_chars`` begrenzen
8. Audit-Eintrag schreiben – auch für verweigerte und fehlerhafte Aufrufe
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from models.base import ToolSpec
from tools.base import (
    REASON_PARAM,
    Permission,
    Tool,
    ToolContext,
    ToolError,
    ToolInvocation,
    ToolResult,
    sha256_text,
    validate_arguments,
)

# --------------------------------------------------------------------------- Policy


class Decision(StrEnum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


_STRICTNESS = {Decision.ALLOW: 0, Decision.ASK: 1, Decision.DENY: 2}

DEFAULT_MODES: Mapping[Permission, Decision] = {
    Permission.READ: Decision.ALLOW,
    Permission.WRITE: Decision.ALLOW,
    Permission.EXECUTE_ALLOWLISTED: Decision.ALLOW,
    Permission.EXECUTE: Decision.ASK,
    Permission.NETWORK: Decision.DENY,
    Permission.DANGEROUS: Decision.ASK,
}


@dataclass(frozen=True)
class PermissionPolicy:
    """Bildet Berechtigungen auf Entscheidungen ab; die strengste gewinnt."""

    modes: Mapping[Permission, Decision] = field(default_factory=lambda: dict(DEFAULT_MODES))
    denied_tools: frozenset[str] = frozenset()

    def decide(self, tool: str, required: Iterable[Permission]) -> Decision:
        if tool in self.denied_tools:
            return Decision.DENY
        decision = Decision.ALLOW
        for permission in required:
            mode = self.modes.get(permission, Decision.DENY)
            if _STRICTNESS[mode] > _STRICTNESS[decision]:
                decision = mode
        return decision

    @classmethod
    def read_only(cls) -> PermissionPolicy:
        return cls(
            {
                **DEFAULT_MODES,
                Permission.WRITE: Decision.DENY,
                Permission.EXECUTE_ALLOWLISTED: Decision.DENY,
                Permission.EXECUTE: Decision.DENY,
                Permission.DANGEROUS: Decision.DENY,
            }
        )


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    invocation: ToolInvocation
    permissions: frozenset[Permission]
    description: str


ApprovalHandler = Callable[[ApprovalRequest], Awaitable[bool]]

# --------------------------------------------------------------------------- Audit

_SECRET_KEY_RE = re.compile(r"(key|token|secret|password|passphrase|credential)", re.IGNORECASE)
_SECRET_VALUE_RE = re.compile(
    r"(sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}"
    r"|xox[abposr]-[A-Za-z0-9-]{10,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)"
)
_MAX_AUDIT_STRING = 500


def redact(value: Any, key: str = "") -> Any:
    """Entfernt Secrets und kürzt lange Werte (z. B. Dateiinhalte) für das Audit-Log."""
    if key and _SECRET_KEY_RE.search(key):
        return "***"
    if isinstance(value, str):
        value = _SECRET_VALUE_RE.sub("***", value)
        if len(value) > _MAX_AUDIT_STRING:
            return f"{value[:200]}…[{len(value)} Zeichen, sha256={sha256_text(value)[:16]}]"
        return value
    if isinstance(value, Mapping):
        return {str(k): redact(v, str(k)) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [redact(v) for v in value]
    return value


class AuditLog(Protocol):
    def record(self, entry: Mapping[str, Any]) -> None: ...


class MemoryAuditLog:
    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []

    def record(self, entry: Mapping[str, Any]) -> None:
        self.entries.append(dict(entry))


class JsonlAuditLog:
    """Append-only JSON-Lines-Datei (eine Zeile pro Tool-Aufruf)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self._lock = threading.Lock()

    def record(self, entry: Mapping[str, Any]) -> None:
        line = json.dumps(entry, ensure_ascii=False, default=str)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]


# --------------------------------------------------------------------------- Registry


def _with_reason(spec: ToolSpec) -> ToolSpec:
    params = dict(spec.parameters)
    properties = dict(params.get("properties", {}))
    properties[REASON_PARAM] = {
        "type": "string",
        "description": "Why you are calling this tool now (one short sentence).",
    }
    required = [r for r in params.get("required", []) if r != REASON_PARAM]
    params.update(properties=properties, required=[*required, REASON_PARAM])
    return ToolSpec(spec.name, spec.description, params)


class ToolRegistry:
    def __init__(
        self,
        tools: Iterable[Tool] = (),
        *,
        policy: PermissionPolicy | None = None,
        approval: ApprovalHandler | None = None,
        audit: AuditLog | None = None,
        max_output_chars: int = 20_000,
        require_reason: bool = True,
    ) -> None:
        self._tools: dict[str, Tool] = {}
        self.policy = policy or PermissionPolicy()
        self.approval = approval
        self.audit = audit or MemoryAuditLog()
        self.max_output_chars = max_output_chars
        self.require_reason = require_reason
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool {tool.name!r} ist bereits registriert")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            raise KeyError(f"Unbekanntes Tool {name!r}") from None

    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self, names: Iterable[str] | None = None) -> list[ToolSpec]:
        """Tool-Spezifikationen für das Modell – inklusive Pflichtfeld ``reason``."""
        selected = self.names() if names is None else [n for n in names if n in self._tools]
        specs = [self._tools[n].spec for n in selected]
        return [_with_reason(s) for s in specs] if self.require_reason else specs

    async def execute(self, invocation: ToolInvocation, ctx: ToolContext) -> ToolResult:
        started = time.perf_counter()
        decision: Decision | None = None
        permissions: frozenset[Permission] = frozenset()
        result: ToolResult
        try:
            tool = self._tools.get(invocation.tool)
            if tool is None:
                raise ToolError(
                    f"Unbekanntes Tool {invocation.tool!r}. Verfügbar: {', '.join(self.names())}"
                )
            if self.require_reason and not invocation.reason.strip():
                raise ToolError("Begründung fehlt: jeder Tool-Aufruf braucht 'reason'")
            problems = validate_arguments(tool.parameters, invocation.arguments)
            if problems:
                raise ToolError("Ungültige Argumente: " + "; ".join(problems))
            permissions = tool.required_permissions(invocation.arguments, ctx)
            decision = self.policy.decide(tool.name, permissions)
            if decision is Decision.ASK:
                decision = await self._ask(tool, invocation, permissions)
            if decision is Decision.DENY:
                raise ToolError(
                    "Nicht erlaubt durch Berechtigungsrichtlinie "
                    f"(benötigt: {', '.join(sorted(p.value for p in permissions))})"
                )
            timeout = tool.timeout_for(invocation.arguments, ctx)
            try:
                result = await asyncio.wait_for(tool.run(invocation.arguments, ctx), timeout)
            except TimeoutError:
                raise ToolError(f"Zeitlimit von {timeout:.0f}s überschritten") from None
        except ToolError as exc:
            result = ToolResult.fail(str(exc))
        except Exception as exc:
            result = ToolResult.fail(f"Interner Tool-Fehler ({type(exc).__name__}): {exc}")

        result = self._limit(result)
        duration_ms = (time.perf_counter() - started) * 1000
        result = result.with_metadata(
            tool=invocation.tool,
            invocation_id=invocation.id,
            duration_ms=round(duration_ms, 1),
            permissions=sorted(p.value for p in permissions),
            decision=decision.value if decision else None,
        )
        self._audit(invocation, result, duration_ms)
        return result

    async def _ask(
        self, tool: Tool, invocation: ToolInvocation, permissions: frozenset[Permission]
    ) -> Decision:
        if self.approval is None:
            raise ToolError(
                "Bestätigung erforderlich, aber kein Bestätigungskanal konfiguriert "
                f"(benötigt: {', '.join(sorted(p.value for p in permissions))})"
            )
        request = ApprovalRequest(invocation, permissions, tool.describe(invocation.arguments))
        approved = await self.approval(request)
        if not approved:
            raise ToolError("Vom Nutzer abgelehnt")
        return Decision.ALLOW

    def _limit(self, result: ToolResult) -> ToolResult:
        if result.error is not None and len(result.error) > 4000:
            return ToolResult.fail(result.error[:4000] + "…[gekürzt]", **result.metadata)
        if result.output is None or len(result.output) <= self.max_output_chars:
            return result
        clipped = result.output[: self.max_output_chars]
        note = f"\n…[gekürzt: {len(result.output) - self.max_output_chars} weitere Zeichen]"
        return ToolResult(
            True,
            clipped + note,
            None,
            {**result.metadata, "truncated": True, "output_chars_total": len(result.output)},
        )

    def _audit(self, invocation: ToolInvocation, result: ToolResult, duration_ms: float) -> None:
        meta = result.metadata
        self.audit.record(
            {
                "timestamp": datetime.now(UTC).isoformat(timespec="milliseconds"),
                "invocation_id": invocation.id,
                "task_id": invocation.task_id,
                "subtask_id": invocation.subtask_id,
                "tool": invocation.tool,
                "arguments": redact(dict(invocation.arguments)),
                "reason": redact(invocation.reason),
                "permissions": meta.get("permissions", []),
                "decision": meta.get("decision"),
                "success": result.success,
                "error": redact(result.error) if result.error else None,
                "output_chars": len(result.output or ""),
                "duration_ms": round(duration_ms, 1),
            }
        )
