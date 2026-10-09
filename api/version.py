"""Versionsnummer – einzige Quelle ist ``pyproject.toml``.

Installiert (auch im PyInstaller-Paket, dort per ``copy_metadata``) liefert
``importlib.metadata`` die Version; im Quellbaum wird ``pyproject.toml`` gelesen.
"""

from __future__ import annotations

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


def get_version() -> str:
    source = _from_pyproject()
    if source:
        return source
    try:
        return metadata.version("nova")
    except metadata.PackageNotFoundError:
        return "0.0.0+unknown"


__version__ = get_version()
