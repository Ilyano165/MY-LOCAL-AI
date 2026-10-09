"""``nova-desktop`` – NOVA als Windows-Desktop-Anwendung.

Ablauf:
1. Prüfen, ob Edge WebView2 verfügbar ist (Windows). Ohne WebView2 kein stiller Rückfall auf
   den alten IE-Renderer, sondern eine klare Meldung mit Angebot, den Browser zu nutzen.
2. Core-Dienst (``nova serve``, eigener Prozess) starten oder einen laufenden verwenden.
3. Natives Fenster öffnen, das die Core-UI von ``http://127.0.0.1:<port>/`` lädt.
4. Beim Schließen: Core stoppen, wenn dieses Fenster ihn gestartet hat und
   „Core im Hintergrund weiterlaufen lassen“ aus ist (Integrationen brauchen ihn sonst weiter).

Die Fenster-Bridge (``window.pywebview.api``) bietet nur Fenster- und Dienststeuerung –
keine Geschäftslogik. Alles andere läuft über die normale Core-API.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import subprocess
import sys
import threading
import time
import webbrowser
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from api.cli import DEFAULT_PORT, service_start, service_status, service_stop
from api.config import default_data_dir
from api.paths import DataLayout
from api.version import __version__

log = logging.getLogger("nova.desktop")

WINDOW_TITLE = "NOVA"
MIN_SIZE = (720, 520)
WEBVIEW2_DOWNLOAD = "https://developer.microsoft.com/microsoft-edge/webview2/"
_WEBVIEW2_KEYS = (
    "{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}",  # Evergreen Runtime
    "{2CD8A007-E189-409D-A2C8-9AF4EF3C72AA}",  # Beta
    "{0D50BFEC-CD6A-4F9A-964C-C7416E3ACB10}",  # Dev
    "{65C35B14-6C1D-4122-AC46-7148CC9D6497}",  # Canary
)
_NET_462_RELEASE = 394802


# ---------------------------------------------------------------------- Einstellungen


@dataclass
class DesktopPrefs:
    width: int = 1200
    height: int = 820
    keep_core_running: bool = False

    @classmethod
    def load(cls, path: Path) -> DesktopPrefs:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls()
        prefs = cls()
        if isinstance(data.get("width"), int) and data["width"] >= MIN_SIZE[0]:
            prefs.width = data["width"]
        if isinstance(data.get("height"), int) and data["height"] >= MIN_SIZE[1]:
            prefs.height = data["height"]
        if isinstance(data.get("keep_core_running"), bool):
            prefs.keep_core_running = data["keep_core_running"]
        return prefs

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        os.replace(tmp, path)


def prefs_path(layout: DataLayout) -> Path:
    return layout.root / "desktop.json"


# ---------------------------------------------------------------------- WebView2


def webview2_status(winreg_module: Any = None) -> tuple[bool, str]:
    """(verfügbar, Begründung). Nur unter Windows relevant; sonst (True, "not windows")."""
    if winreg_module is None:
        if sys.platform != "win32":
            return True, "not windows"
        import winreg as winreg_module
    reg = winreg_module
    try:
        with reg.OpenKey(
            reg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\NET Framework Setup\NDP\v4\Full"
        ) as key:
            release, _ = reg.QueryValueEx(key, "Release")
    except OSError:
        return False, ".NET Framework 4.6.2 or newer is not installed"
    if int(release) < _NET_462_RELEASE:
        return False, ".NET Framework 4.6.2 or newer is not installed"
    for client in _WEBVIEW2_KEYS:
        for hive, prefix in (
            ("HKEY_CURRENT_USER", r"SOFTWARE\Microsoft\EdgeUpdate\Clients"),
            ("HKEY_LOCAL_MACHINE", r"SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients"),
            ("HKEY_LOCAL_MACHINE", r"SOFTWARE\Microsoft\EdgeUpdate\Clients"),
        ):
            try:
                with reg.OpenKey(getattr(reg, hive), rf"{prefix}\{client}") as key:
                    version, _ = reg.QueryValueEx(key, "pv")
            except OSError:
                continue
            if version and str(version) != "0.0.0.0":
                return True, f"WebView2 {version}"
    return False, "Microsoft Edge WebView2 Runtime is not installed"


# ---------------------------------------------------------------------- Core-Steuerung


class CoreController:
    """Startet/stoppt den Core-Dienst (eigener Prozess) über die CLI-Funktionen."""

    def __init__(
        self,
        layout: DataLayout,
        port: int = DEFAULT_PORT,
        *,
        start: Callable[..., dict[str, Any]] = service_start,
        stop: Callable[..., dict[str, Any]] = service_stop,
        status: Callable[..., dict[str, Any]] = service_status,
    ) -> None:
        self.layout = layout
        self.port = port
        self._start = start
        self._stop = stop
        self._status = status
        self.started_by_us = False
        self._lock = threading.Lock()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/"

    def ensure_running(self) -> dict[str, Any]:
        with self._lock:
            result = self._start(self.layout, self.port)
            if result.get("started"):
                self.started_by_us = True
            return result

    def stop(self) -> dict[str, Any]:
        with self._lock:
            result = self._stop(self.layout, self.port)
            if result.get("status") == "stopped":
                self.started_by_us = False
            return result

    def restart(self) -> dict[str, Any]:
        self.stop()
        return self.ensure_running()

    def status(self) -> dict[str, Any]:
        data = dict(self._status(self.layout, self.port))
        data["started_by_desktop"] = self.started_by_us
        return data

    def stop_on_exit(self, prefs: DesktopPrefs) -> bool:
        """True, wenn der Core beim Schließen des Fensters gestoppt werden soll."""
        return self.started_by_us and not prefs.keep_core_running


# ---------------------------------------------------------------------- Bridge (JS-API)


def _open_path(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        os.startfile(path)  # lokaler Ordner des Benutzers
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


class DesktopBridge:
    """Als ``window.pywebview.api`` im Fenster verfügbar. Nur öffentliche Methoden werden
    freigegeben; Zustand liegt in ``_``-Attributen, damit pywebview ihn nicht exportiert."""

    def __init__(
        self,
        core: CoreController,
        prefs: DesktopPrefs,
        prefs_file: Path,
        *,
        quit_window: Callable[[], None] | None = None,
        open_url: Callable[[str], Any] = webbrowser.open,
        open_path: Callable[[Path], None] = _open_path,
    ) -> None:
        self._core = core
        self._prefs = prefs
        self._prefs_file = prefs_file
        self._quit = quit_window
        self._open_url = open_url
        self._open_path = open_path

    def info(self) -> dict[str, Any]:
        return {
            "desktop": True,
            "version": __version__,
            "keep_core_running": self._prefs.keep_core_running,
            "core": self._core.status(),
        }

    def core_status(self) -> dict[str, Any]:
        return self._core.status()

    def restart_core(self) -> dict[str, Any]:
        return self._core.restart()

    def stop_core_and_quit(self) -> dict[str, Any]:
        result = self._core.stop()
        if self._quit is not None:
            threading.Thread(target=self._quit, daemon=True).start()
        return result

    def set_keep_core_running(self, value: bool) -> dict[str, Any]:
        self._prefs.keep_core_running = bool(value)
        self._prefs.save(self._prefs_file)
        return {"keep_core_running": self._prefs.keep_core_running}

    def open_in_browser(self) -> bool:
        self._open_url(self._core.url)
        return True

    def open_data_folder(self) -> bool:
        self._open_path(self._core.layout.root)
        return True

    def open_logs(self) -> bool:
        self._open_path(self._core.layout.logs)
        return True


# ---------------------------------------------------------------------- Meldungen


def message_box(text: str, *, error: bool = True, ask: bool = False) -> bool:
    """Windows-Meldungsfenster; ``ask`` → Ja/Nein (True = Ja). Sonst stderr."""
    if sys.platform == "win32":
        import ctypes

        flags = (0x10 if error else 0x40) | (0x04 if ask else 0)  # ICON + MB_YESNO
        result = getattr(ctypes, "windll").user32.MessageBoxW(  # noqa: B009
            None, text, WINDOW_TITLE, flags
        )
        return bool(result == 6) if ask else True  # IDYES = 6
    print(text, file=sys.stderr)
    return False


# ---------------------------------------------------------------------- Selbsttest

SELF_TEST_JS = """
(() => {
  const ids = ["app", "messages", "composer", "input", "send", "status-label"];
  const missing = ids.filter((id) => !document.getElementById(id));
  return JSON.stringify({
    title: document.title,
    missing,
    bridge: !!(window.pywebview && window.pywebview.api),
    desktopMarker: document.documentElement.dataset.desktop || null,
    status: (document.getElementById("status-label") || {}).textContent || null,
  });
})()
"""


def _self_test(window: Any, out: Path, renderer: Callable[[], str | None], wait_s: float) -> None:
    result: dict[str, Any] = {"ok": False}
    try:
        if not window.events.loaded.wait(wait_s):
            result["error"] = "page did not load"
        else:
            deadline = time.time() + wait_s
            page: dict[str, Any] = {}
            while time.time() < deadline:  # UI baut sich asynchron auf (Status, Bridge)
                page = json.loads(window.evaluate_js(SELF_TEST_JS))
                if page.get("bridge") and page.get("desktopMarker") and page.get("status"):
                    break
                time.sleep(0.5)
            result.update(page)
            result["renderer"] = renderer()
            result["ok"] = (
                not page.get("missing")
                and bool(page.get("bridge"))
                and page.get("desktopMarker") == "1"
                and (sys.platform != "win32" or result["renderer"] == "edgechromium")
            )
    except Exception as exc:  # Ergebnis immer schreiben
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        window.destroy()


def _verify_renderer(window: Any, renderer: Callable[[], str | None], core: CoreController) -> None:
    """Zweite Absicherung: Fällt pywebview trotz Vorprüfung auf MSHTML (IE) zurück, wird das
    Fenster geschlossen und NOVA im Standardbrowser geöffnet – die UI braucht Chromium."""
    window.events.shown.wait(30)
    used = renderer()
    log.info("renderer: %s", used)
    if sys.platform == "win32" and used not in (None, "edgechromium"):
        window.destroy()
        message_box(
            f"NOVA could not use Microsoft Edge WebView2 (renderer: {used}).\n\n"
            f"Install the WebView2 Runtime from {WEBVIEW2_DOWNLOAD}.\n"
            "NOVA opens in your web browser instead."
        )
        webbrowser.open(core.url)


# ---------------------------------------------------------------------- Start


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="nova-desktop", description="NOVA desktop app")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--self-test", type=Path, help="open, verify the UI, write JSON, exit")
    p.add_argument("--self-test-timeout", type=float, default=60.0)
    p.add_argument("--debug", action="store_true")
    return p.parse_args(argv)


def run(argv: list[str], *, webview_module: Any = None) -> int:
    args = parse_args(argv)
    layout = DataLayout(args.data_dir or default_data_dir())
    layout.ensure()
    logging.basicConfig(
        filename=str(layout.logs / "desktop.log"),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    available, reason = webview2_status()
    if not available:
        log.error("WebView2 unavailable: %s", reason)
        if args.self_test:
            args.self_test.write_text(json.dumps({"ok": False, "error": reason}), "utf-8")
            return 2
        if message_box(
            f"NOVA needs the Microsoft Edge WebView2 Runtime to show its window.\n\n"
            f"Problem: {reason}\n\nInstall it from {WEBVIEW2_DOWNLOAD}\n\n"
            "Open NOVA in your web browser instead?",
            ask=True,
        ):
            core = CoreController(layout, args.port)
            core.ensure_running()
            webbrowser.open(core.url)
        return 2

    prefs_file = prefs_path(layout)
    prefs = DesktopPrefs.load(prefs_file)
    core = CoreController(layout, args.port)
    try:
        core.ensure_running()
    except SystemExit as exc:  # service_start meldet Fehler per SystemExit(Text)
        log.error("core start failed: %s", exc)
        if args.self_test:
            args.self_test.write_text(json.dumps({"ok": False, "error": str(exc)}), "utf-8")
        else:
            message_box(f"NOVA could not start its core service.\n\n{exc}\n\nLogs: {layout.logs}")
        return 1

    if webview_module is None:
        import webview as webview_module
    webview = webview_module

    window: Any = None

    def quit_window() -> None:
        if window is not None:
            window.destroy()

    bridge = DesktopBridge(core, prefs, prefs_file, quit_window=quit_window)
    window = webview.create_window(
        WINDOW_TITLE,
        core.url,
        js_api=bridge,
        width=prefs.width,
        height=prefs.height,
        min_size=MIN_SIZE,
        text_select=True,  # Antworten markieren/kopieren
        background_color="#0f1217",
    )

    def on_resized(width: int, height: int) -> None:
        prefs.width, prefs.height = max(width, MIN_SIZE[0]), max(height, MIN_SIZE[1])

    def on_closed() -> None:
        with contextlib.suppress(OSError):
            prefs.save(prefs_file)
        if core.stop_on_exit(prefs):
            log.info("window closed – stopping core started by the desktop app")
            core.stop()

    window.events.resized += on_resized
    window.events.closed += on_closed

    start_kwargs: dict[str, Any] = {
        "private_mode": False,  # localStorage (z. B. UI-Einstellungen) bleibt erhalten
        "storage_path": str(layout.root / "webview"),
        "debug": args.debug,
    }
    if sys.platform == "win32":
        start_kwargs["gui"] = "edgechromium"
    if args.self_test:
        start_kwargs["func"] = _self_test
        start_kwargs["args"] = (
            window,
            args.self_test,
            lambda: getattr(webview, "renderer", None),
            args.self_test_timeout,
        )
    else:
        start_kwargs["func"] = _verify_renderer
        start_kwargs["args"] = (window, lambda: getattr(webview, "renderer", None), core)
    webview.start(**start_kwargs)
    if args.self_test:
        try:
            return 0 if json.loads(args.self_test.read_text("utf-8")).get("ok") else 1
        except (OSError, ValueError):
            return 1
    return 0


def main() -> None:
    sys.exit(run(sys.argv[1:]))
