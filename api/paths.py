"""Plattformgerechte Orte für Benutzerdaten – getrennt von den Programmdateien.

* Windows: ``%LOCALAPPDATA%\\NOVA`` (Programmdateien liegen getrennt unter
  ``%LOCALAPPDATA%\\Programs\\NOVA`` und werden vom Installer verwaltet)
* andere Systeme: ``~/.nova``
* immer überschreibbar mit ``NOVA_DATA_DIR``

Der Installer schreibt nie in das Datenverzeichnis – Updates und Deinstallation lassen
Konfiguration, Verlauf, Memory, Logs und Modelle unangetastet.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


def default_data_dir(env: dict[str, str] | None = None, platform: str | None = None) -> Path:
    env = dict(os.environ) if env is None else env
    platform = platform or sys.platform
    if env.get("NOVA_DATA_DIR"):
        return Path(env["NOVA_DATA_DIR"]).expanduser()
    if platform.startswith("win"):
        base = env.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "NOVA"
    return Path.home() / ".nova"


@dataclass(frozen=True)
class DataLayout:
    """Unterverzeichnisse des Datenverzeichnisses (eine Stelle für alle Pfade)."""

    root: Path

    @property
    def models_config(self) -> Path:
        return self.root / "models.toml"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def downloads(self) -> Path:
        return self.root / "models" / ".downloads"

    @property
    def catalog(self) -> Path:
        return self.root / "model-catalog.json"

    @property
    def integrations_db(self) -> Path:
        return self.root / "integrations.db"

    @property
    def audit_log(self) -> Path:
        return self.root / "logs" / "audit.jsonl"

    @property
    def service_pid(self) -> Path:
        return self.root / "nova-service.json"

    def ensure(self) -> DataLayout:
        for directory in (self.root, self.logs, self.models):
            directory.mkdir(parents=True, exist_ok=True)
        return self
