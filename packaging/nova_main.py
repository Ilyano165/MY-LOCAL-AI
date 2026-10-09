"""Einstieg für ``nova.exe`` (Konsole): identisch zu ``python -m api``."""

from __future__ import annotations

import sys


def _utf8_console() -> None:
    # Windows-Konsolen nutzen sonst cp1252/cp850 – Umlaute/Sonderzeichen würden abbrechen
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    _utf8_console()
    from api.cli import main

    main()
