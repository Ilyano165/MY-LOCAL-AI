"""Grundbausteine des Tool-Systems.

* :class:`ToolInvocation` – *was* aufgerufen wird: Tool, Argumente und **Begründung**.
* :class:`ToolResult` – strukturiertes Ergebnis ``{success, output, error, metadata}``.
* :class:`Permission` – welche Art von Wirkung ein Aufruf hat; entschieden wird darüber in
  :mod:`tools.registry` (Policy + Bestätigung), nie im Tool selbst.
* :class:`ToolContext` – Laufzeitumgebung: Workspace, erlaubte Verzeichnisse, Limits. Alle
  Pfadzugriffe laufen über :meth:`ToolContext.resolve`.
"""

from __future__ import annotations

import abc
import hashlib
import os
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar

from models.base import ToolSpec

REASON_PARAM = "reason"
_SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "PASSPHRASE")

# Dateien/Verzeichnisse, auf die Tools grundsätzlich nicht zugreifen (docs/security.md §3).
_SENSITIVE_DIRS = frozenset({".ssh", ".gnupg", ".aws", ".azure", ".kube", ".docker"})
_SENSITIVE_NAMES = frozenset({".netrc", ".pgpass", ".git-credentials", ".npmrc", ".pypirc"})
_SENSITIVE_PREFIXES = ("id_rsa", "id_dsa", "id_ecdsa", "id_ed25519")
_SENSITIVE_SUFFIXES = (".pem", ".key", ".p12", ".pfx", ".keystore")


class ToolError(Exception):
    """Erwarteter, fachlicher Fehler eines Tools (z. B. Datei nicht gefunden)."""


class Permission(StrEnum):
    READ = "read"
    """Lesen innerhalb erlaubter Verzeichnisse."""
    WRITE = "write"
    """Dateien anlegen/ändern innerhalb erlaubter Verzeichnisse."""
    EXECUTE_ALLOWLISTED = "execute_allowlisted"
    """Prozess starten, dessen Befehl auf der Allowlist steht."""
    EXECUTE = "execute"
    """Beliebigen Prozess starten."""
    NETWORK = "network"
    """Netzwerkzugriff."""
    DANGEROUS = "dangerous"
    """Potenziell zerstörerisch oder nicht analysierbar (Shell, Löschen, Interpreter-Code)."""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _is_git_config_env(name: str) -> bool:
    # GIT_CONFIG_COUNT/KEY_n/VALUE_n gehören zusammen (Teilmengen lassen git abbrechen) und die
    # Werte können Zugangsdaten enthalten (z. B. http.extraHeader) → immer komplett entfernen.
    return name in ("GIT_CONFIG_COUNT", "GIT_CONFIG_PARAMETERS") or name.startswith(
        ("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_")
    )


def safe_environment(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """Umgebung für Kindprozesse: ohne Secrets, ohne Bytecode-Cache, ohne Interaktivität."""
    env = {
        k: v
        for k, v in os.environ.items()
        if not any(m in k.upper() for m in _SECRET_MARKERS) and not _is_git_config_env(k)
    }
    env["PYTHONDONTWRITEBYTECODE"] = "1"  # siehe agents/verifier.py: veralteter Bytecode
    env["GIT_TERMINAL_PROMPT"] = "0"
    env.setdefault("PAGER", "cat")
    if extra:
        env.update(extra)
    return env


def is_sensitive_path(path: Path) -> bool:
    parts = set(path.parts)
    if parts & _SENSITIVE_DIRS:
        return True
    name = path.name.lower()
    if name in _SENSITIVE_NAMES or name.startswith(_SENSITIVE_PREFIXES):
        return True
    if name == ".env" or (name.startswith(".env.") and name != ".env.example"):
        return True
    return name.endswith(_SENSITIVE_SUFFIXES)


@dataclass(frozen=True, slots=True)
class ToolContext:
    workspace: Path
    extra_roots: tuple[Path, ...] = ()
    """Weitere erlaubte Verzeichnisse neben dem Workspace."""
    timeout_s: float = 60.0
    backup_dir: Path | None = None
    """Ist das gesetzt, wird der alte Inhalt vor dem Überschreiben/Ändern hier gesichert."""
    task_id: str | None = None

    @property
    def allowed_roots(self) -> tuple[Path, ...]:
        return tuple(p.resolve() for p in (self.workspace, *self.extra_roots))

    def resolve(self, path: str, *, write: bool = False) -> Path:
        """Pfad auflösen und gegen erlaubte Verzeichnisse und Sperrliste prüfen.

        Relative Pfade gelten relativ zum Workspace. Die Prüfung erfolgt *nach*
        Symlink-Auflösung, damit Links nicht aus dem erlaubten Bereich herausführen.
        """
        if not isinstance(path, str) or not path.strip():
            raise ToolError("Leerer Pfad")
        if "\x00" in path:
            raise ToolError("Ungültiger Pfad")
        raw = Path(path).expanduser()
        candidate = (raw if raw.is_absolute() else self.workspace / raw).resolve()
        root = next((r for r in self.allowed_roots if candidate.is_relative_to(r)), None)
        if root is None:
            raise ToolError(f"Pfad {path!r} liegt außerhalb der erlaubten Verzeichnisse")
        if is_sensitive_path(candidate.relative_to(root)) or is_sensitive_path(candidate):
            raise ToolError(f"Zugriff auf {path!r} ist per Sicherheitsrichtlinie gesperrt")
        if write and ".git" in candidate.relative_to(root).parts:
            raise ToolError("Schreiben in Git-Interna (.git/) ist nicht erlaubt")
        return candidate

    def display(self, path: Path) -> str:
        """Pfad relativ zum Workspace (falls möglich) für Ausgaben."""
        try:
            return str(path.relative_to(self.workspace.resolve())) or "."
        except ValueError:
            return str(path)


@dataclass(frozen=True, slots=True)
class ToolInvocation:
    tool: str
    arguments: Mapping[str, Any] = field(default_factory=dict, hash=False)
    reason: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    task_id: str | None = None
    subtask_id: str | None = None

    @classmethod
    def from_model(
        cls,
        tool: str,
        arguments: Mapping[str, Any],
        *,
        call_id: str | None = None,
        task_id: str | None = None,
        subtask_id: str | None = None,
    ) -> ToolInvocation:
        """Trennt die vom Modell mitgelieferte Begründung von den eigentlichen Argumenten."""
        args = dict(arguments)
        reason = args.pop(REASON_PARAM, "")
        return cls(
            tool=tool,
            arguments=args,
            reason=reason if isinstance(reason, str) else str(reason),
            id=call_id or uuid.uuid4().hex[:12],
            task_id=task_id,
            subtask_id=subtask_id,
        )


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Strukturiertes Tool-Ergebnis.

    Invariante: bei Erfolg ist ``error`` ``None``; bei Fehlschlag ist ``output`` ``None``
    und ``error`` beschreibt den Fehler. Zusatzinformationen stehen in ``metadata``.
    """

    success: bool
    output: str | None
    error: str | None
    metadata: Mapping[str, Any] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        if self.success and self.error is not None:
            raise ValueError("Erfolgreiches ToolResult darf keinen Fehler haben")
        if not self.success and (self.output is not None or not self.error):
            raise ValueError("Fehlgeschlagenes ToolResult braucht error und output=None")

    @classmethod
    def ok(cls, output: str, **metadata: Any) -> ToolResult:
        return cls(True, output, None, metadata)

    @classmethod
    def fail(cls, error: str, **metadata: Any) -> ToolResult:
        return cls(False, None, error, metadata)

    def with_metadata(self, **extra: Any) -> ToolResult:
        return ToolResult(self.success, self.output, self.error, {**self.metadata, **extra})

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "metadata": dict(self.metadata),
        }


class Tool(abc.ABC):
    name: ClassVar[str]
    description: ClassVar[str]
    parameters: ClassVar[Mapping[str, Any]]
    permissions: ClassVar[frozenset[Permission]] = frozenset({Permission.READ})

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, dict(self.parameters))

    def required_permissions(
        self, args: Mapping[str, Any], ctx: ToolContext
    ) -> frozenset[Permission]:
        """Berechtigungen für *diesen* Aufruf (Tools wie ``execute_command`` entscheiden je
        nach Argumenten). Darf :class:`ToolError` werfen, wenn ein Aufruf grundsätzlich
        verboten ist."""
        return self.permissions

    def describe(self, args: Mapping[str, Any]) -> str:
        """Kurze, menschenlesbare Beschreibung der Aktion (für Bestätigungsdialoge)."""
        return f"{self.name}({', '.join(f'{k}={v!r}' for k, v in args.items())})"[:300]

    def timeout_for(self, args: Mapping[str, Any], ctx: ToolContext) -> float:
        return ctx.timeout_s

    @abc.abstractmethod
    async def run(self, args: Mapping[str, Any], ctx: ToolContext) -> ToolResult:
        """Führt das Tool aus. Fachliche Fehler als :class:`ToolError` werfen oder als
        ``ToolResult.fail`` zurückgeben."""


_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list, tuple),
    "object": (dict,),
}


def validate_arguments(schema: Mapping[str, Any], args: Mapping[str, Any]) -> list[str]:
    """Prüft die für Tool-Aufrufe relevante Teilmenge von JSON Schema.

    Unterstützt: ``required``, ``properties.*.type`` (auch Typlisten), ``enum``,
    ``minimum``/``maximum``, ``items.type``, ``additionalProperties: false``.
    """
    if not isinstance(args, Mapping):
        return ["Argumente müssen ein JSON-Objekt sein"]
    problems: list[str] = []
    properties: Mapping[str, Any] = schema.get("properties", {})
    for name in schema.get("required", []):
        if name not in args:
            problems.append(f"Pflichtargument fehlt: {name}")
    if schema.get("additionalProperties") is False:
        problems += [f"Unbekanntes Argument: {k}" for k in args if k not in properties]
    for name, value in args.items():
        prop = properties.get(name)
        if not isinstance(prop, Mapping):
            continue
        problem = _check_value(name, value, prop)
        if problem:
            problems.append(problem)
    return problems


def _type_ok(value: Any, expected: str) -> bool:
    allowed = _JSON_TYPES.get(expected)
    if allowed is None:
        return True
    if isinstance(value, bool) and expected != "boolean":
        return False
    return isinstance(value, allowed)


def _check_value(name: str, value: Any, prop: Mapping[str, Any]) -> str | None:
    expected = prop.get("type")
    types = [expected] if isinstance(expected, str) else list(expected or [])
    if types and not any(_type_ok(value, t) for t in types):
        return f"{name}: erwartet {'/'.join(types)}, erhalten {type(value).__name__}"
    if "enum" in prop and value not in prop["enum"]:
        return f"{name}: {value!r} nicht in {prop['enum']}"
    if isinstance(value, int | float) and not isinstance(value, bool):
        if "minimum" in prop and value < prop["minimum"]:
            return f"{name}: {value} < Minimum {prop['minimum']}"
        if "maximum" in prop and value > prop["maximum"]:
            return f"{name}: {value} > Maximum {prop['maximum']}"
    item_type = (
        prop.get("items", {}).get("type") if isinstance(prop.get("items"), Mapping) else None
    )
    if isinstance(value, list | tuple) and isinstance(item_type, str):
        bad = [v for v in value if not _type_ok(v, item_type)]
        if bad:
            return f"{name}: Elemente müssen {item_type} sein"
    return None
