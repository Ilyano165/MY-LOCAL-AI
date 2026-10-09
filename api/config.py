"""Konfiguration der NOVA API (Kommandozeile/Umgebung, keine Secrets im Code)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from api.paths import default_data_dir as _platform_data_dir


def default_data_dir() -> Path:
    return _platform_data_dir()


@dataclass
class ApiConfig:
    models_config: Path | None = None
    """Modellkonfiguration (TOML). Ohne Datei nur im Development Mode zulässig."""
    dev_mode: bool = False
    """UI startet auch ohne Konfiguration/Modell – zeigt dann klar „No local model available.“"""
    data_dir: Path = field(default_factory=default_data_dir)
    router: str = "rules"
    """``rules`` (Standard) oder ``learned`` (nur mit trainiertem Ranker)."""
    learned_ranker: Path | None = None
    api_token_env: str | None = "NOVA_API_TOKEN"
    """Name der Umgebungsvariable mit optionalem API-Token (nie der Wert selbst)."""
    allow_remote: bool = False
    """Zugriff von anderen Rechnern. Nur bewusst aktivieren (siehe ``api.__main__``: erfordert
    TLS und API-Token). Standard: nur Loopback-Clients werden bedient."""
    trusted_clients: tuple[str, ...] = ("127.0.0.1", "::1", "localhost")
    cors_origins: tuple[str, ...] = ()
    """Browser-Origins, die die Integrations-API nutzen dürfen (Standard: keine)."""
    mode: str = ""
    """``normal`` | ``setup`` (installiert, noch kein Modell) | ``development``."""
    max_attachment_bytes: int = 10 * 1024 * 1024
    max_text_attachment_chars: int = 200_000

    def __post_init__(self) -> None:
        if self.router not in ("rules", "learned"):
            raise ValueError("router must be 'rules' or 'learned'")
        if self.models_config is None and not self.dev_mode:
            raise ValueError(
                "No model configuration given (--config). To start without a model, "
                "use development mode (--dev)."
            )
        if self.models_config is not None:
            self.models_config = Path(self.models_config).expanduser()
            if not self.models_config.is_file() and not self.dev_mode:
                raise ValueError(f"Model configuration not found: {self.models_config}")
        self.data_dir = Path(self.data_dir).expanduser()
        if not self.mode:
            self.mode = "development" if self.dev_mode else "normal"
        if self.mode not in ("normal", "setup", "development"):
            raise ValueError("mode must be normal, setup or development")

    @property
    def api_token(self) -> str | None:
        return os.environ.get(self.api_token_env) if self.api_token_env else None
