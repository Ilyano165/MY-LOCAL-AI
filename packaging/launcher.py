"""Einstieg für ``nova-launcher.exe`` (ohne Konsolenfenster) – Startmenü, Autostart, Installer.

    nova-launcher.exe            Dienst starten (falls nötig) und Oberfläche im Browser öffnen
    nova-launcher.exe start      nur Dienst starten (Autostart)
    nova-launcher.exe stop       Dienst stoppen (Startmenü „Stop NOVA service“, Installer)
    --quiet                      keine Meldungsfenster (Installer)
    --port N                     anderer Port (Standard 8765)

Fehler landen verständlich in einem Meldungsfenster und in ``<Daten>/logs/launcher.log``.
"""

from __future__ import annotations

import sys
import traceback
import webbrowser
from datetime import UTC, datetime


def _message(text: str, *, error: bool) -> None:
    if sys.platform == "win32":
        import ctypes

        flags = 0x10 if error else 0x40  # MB_ICONERROR / MB_ICONINFORMATION
        getattr(ctypes, "windll").user32.MessageBoxW(None, text, "NOVA", flags)  # noqa: B009
    else:
        print(text, file=sys.stderr)


def run(argv: list[str]) -> int:
    from api.cli import DEFAULT_PORT, service_start, service_stop
    from api.config import default_data_dir
    from api.paths import DataLayout

    quiet = "--quiet" in argv
    args = [a for a in argv if a != "--quiet"]
    port = DEFAULT_PORT
    if "--port" in args:
        i = args.index("--port")
        port = int(args[i + 1])
        del args[i : i + 2]
    action = args[0] if args else "open"
    layout = DataLayout(default_data_dir())
    try:
        if action == "stop":
            result = service_stop(layout, port)
            if not quiet and result["status"] != "stopped":
                _message("NOVA service could not be stopped.", error=True)
            return 0 if result["status"] == "stopped" else 1
        if action not in ("open", "start"):
            raise SystemExit(f"Unknown action: {action}")
        service_start(layout, port)
        if action == "open":
            webbrowser.open(f"http://127.0.0.1:{port}/")
        return 0
    except (SystemExit, Exception) as exc:  # alles verständlich melden
        detail = str(exc) if isinstance(exc, SystemExit) else f"{type(exc).__name__}: {exc}"
        try:
            layout.logs.mkdir(parents=True, exist_ok=True)
            with (layout.logs / "launcher.log").open("a", encoding="utf-8") as fh:
                fh.write(f"{datetime.now(UTC).isoformat()} {action}: {detail}\n")
                if not isinstance(exc, SystemExit):
                    fh.write(traceback.format_exc())
        except OSError:
            pass
        if not quiet:
            _message(
                f"NOVA could not {action if action != 'open' else 'start'}.\n\n{detail}\n\n"
                f"Logs: {layout.logs}",
                error=True,
            )
        return 1


if __name__ == "__main__":
    sys.exit(run(sys.argv[1:]))
