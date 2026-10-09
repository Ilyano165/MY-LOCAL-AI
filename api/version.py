"""Versionsnummer – einzige Quelle ist ``pyproject.toml``.

Reihenfolge: im PyInstaller-Paket die vom Build-Skript geschriebene Datei ``VERSION``
(``packaging/build.py``), im Quellbaum ``pyproject.toml``, installiert ``importlib.metadata``.
"""

from __future__ import annotations

import sys
import tomllib
from importlib import metadata
from pathlib import Path


def _from_pyproject() -> str | None:
    path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    try:
        with path.open("rb") as fh:
            return str(tomllib.load(fh)["project"]["version"])
    except (OSError, KeyError, tomllib.TOMLDecodeError):
        return None


def _from_bundle() -> str | None:
    bundle = getattr(sys, "_MEIPASS", None)
    if not getattr(sys, "frozen", False) or bundle is None:
        return None
    try:
        return (Path(bundle) / "VERSION").read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def get_version() -> str:
    source = _from_bundle() or _from_pyproject()
    if source:
        return source
    try:
        return metadata.version("nova")
    except metadata.PackageNotFoundError:
        return "0.0.0+unknown"


__version__ = get_version()
