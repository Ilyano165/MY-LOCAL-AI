"""Benutzereinstellungen der UI (JSON in ``<data_dir>/ui_settings.json``)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


@dataclass
class Settings:
    model: str = "auto"
    """``auto`` = Router entscheidet; sonst Registry-Name eines Modells."""
    temperature: float = 0.7
    max_tokens: int = 2048
    system_prompt: str = ""
    history_messages: int = 20
    """Wie viele frühere Nachrichten als Kontext mitgeschickt werden."""
    request_timeout_s: float = 300.0
    agent_workspace: str = ""
    """Arbeitsverzeichnis für den Agent-Modus (leer = Agent-Modus deaktiviert)."""
    show_routing: bool = True

    def validate(self) -> None:
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature muss zwischen 0 und 2 liegen")
        if not 1 <= self.max_tokens <= 131_072:
            raise ValueError("max_tokens muss zwischen 1 und 131072 liegen")
        if not 0 <= self.history_messages <= 200:
            raise ValueError("history_messages muss zwischen 0 und 200 liegen")
        if not 5 <= self.request_timeout_s <= 3600:
            raise ValueError("request_timeout_s muss zwischen 5 und 3600 liegen")
        if len(self.system_prompt) > 20_000:
            raise ValueError("system_prompt ist zu lang (max. 20000 Zeichen)")
        if self.agent_workspace and not Path(self.agent_workspace).expanduser().is_dir():
            raise ValueError(f"Agent-Arbeitsverzeichnis existiert nicht: {self.agent_workspace}")

    def merged(self, changes: dict[str, Any]) -> Settings:
        known = {f.name: f.type for f in fields(self)}
        unknown = set(changes) - set(known)
        if unknown:
            raise ValueError(f"Unbekannte Einstellungen: {sorted(unknown)}")
        data = {**asdict(self), **changes}
        try:
            updated = Settings(
                model=str(data["model"]),
                temperature=float(data["temperature"]),
                max_tokens=int(data["max_tokens"]),
                system_prompt=str(data["system_prompt"]),
                history_messages=int(data["history_messages"]),
                request_timeout_s=float(data["request_timeout_s"]),
                agent_workspace=str(data["agent_workspace"]),
                show_routing=bool(data["show_routing"]),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Ungültiger Wert: {exc}") from exc
        updated.validate()
        return updated

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SettingsStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> Settings:
        if not self.path.is_file():
            return Settings()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return Settings().merged(
                {k: v for k, v in data.items() if k in {f.name for f in fields(Settings)}}
            )
        except (ValueError, OSError):
            return Settings()

    def save(self, settings: Settings) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(settings.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        tmp.replace(self.path)
