"""Desktop-App: Einstellungen, WebView2-Prüfung, Core-Steuerung, Bridge, Fensteraufbau.

Das echte Fenster (WebView2) wird im Windows-Release-Workflow per ``--self-test`` geprüft;
hier läuft ein echter Core-Prozess mit einem Fake-``webview``-Modul.
"""

from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any

import pytest

from api.cli import service_status
from api.paths import DataLayout
from desktop import app
from desktop.app import CoreController, DesktopBridge, DesktopPrefs, webview2_status


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


# ---------------------------------------------------------------------- Einstellungen


def test_prefs_roundtrip_and_validation(tmp_path: Path) -> None:
    path = tmp_path / "desktop.json"
    assert DesktopPrefs.load(path) == DesktopPrefs()  # fehlt → Standard
    DesktopPrefs(width=1000, height=700, keep_core_running=True).save(path)
    assert DesktopPrefs.load(path) == DesktopPrefs(1000, 700, True)
    path.write_text(json.dumps({"width": 10, "height": "x", "keep_core_running": "yes"}))
    assert DesktopPrefs.load(path) == DesktopPrefs()  # ungültige Werte → Standard
    path.write_text("{kaputt")
    assert DesktopPrefs.load(path) == DesktopPrefs()


# ---------------------------------------------------------------------- WebView2


class FakeWinreg:
    HKEY_LOCAL_MACHINE = "HKLM"
    HKEY_CURRENT_USER = "HKCU"

    def __init__(self, values: dict[tuple[str, str], dict[str, Any]]) -> None:
        self.values = values

    def OpenKey(self, hive: str, path: str) -> Any:
        if (hive, path) not in self.values:
            raise OSError("missing")
        data = self.values[(hive, path)]

        class Key:
            def __enter__(self) -> dict[str, Any]:
                return data

            def __exit__(self, *exc: object) -> None:
                return None

        return Key()

    def QueryValueEx(self, key: dict[str, Any], name: str) -> tuple[Any, int]:
        if name not in key:
            raise OSError("missing value")
        return key[name], 1


NET = ("HKLM", r"SOFTWARE\Microsoft\NET Framework Setup\NDP\v4\Full")
RUNTIME = (
    "HKLM",
    r"SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}",
)


def test_webview2_detection() -> None:
    ok, reason = webview2_status(FakeWinreg({NET: {"Release": 533320}, RUNTIME: {"pv": "141.0.1"}}))
    assert ok and "141.0.1" in reason
    ok, reason = webview2_status(FakeWinreg({NET: {"Release": 533320}}))
    assert not ok and "WebView2" in reason
    ok, reason = webview2_status(FakeWinreg({NET: {"Release": 1000}, RUNTIME: {"pv": "1"}}))
    assert not ok and ".NET" in reason
    ok, _ = webview2_status(FakeWinreg({NET: {"Release": 533320}, RUNTIME: {"pv": "0.0.0.0"}}))
    assert not ok  # Platzhalter-Version nach Deinstallation


# ---------------------------------------------------------------------- Core-Steuerung


def test_controller_tracks_ownership(tmp_path: Path) -> None:
    calls: list[str] = []

    def start(layout: DataLayout, port: int) -> dict[str, Any]:
        calls.append("start")
        return {"status": "running", "started": len(calls) == 1}

    def stop(layout: DataLayout, port: int) -> dict[str, Any]:
        calls.append("stop")
        return {"status": "stopped"}

    core = CoreController(
        DataLayout(tmp_path), 1, start=start, stop=stop, status=lambda *_: {"status": "x"}
    )
    core.ensure_running()
    assert core.started_by_us and core.stop_on_exit(DesktopPrefs())
    assert not core.stop_on_exit(DesktopPrefs(keep_core_running=True))
    assert core.status()["started_by_desktop"] is True
    core.stop()
    assert not core.started_by_us


def test_controller_does_not_claim_existing_core(tmp_path: Path) -> None:
    core = CoreController(
        DataLayout(tmp_path), 1, start=lambda *_: {"status": "running", "started": False}
    )
    core.ensure_running()
    assert not core.stop_on_exit(DesktopPrefs())  # fremden/laufenden Core nicht beenden


def test_bridge_methods(tmp_path: Path) -> None:
    opened: list[Any] = []
    quit_called: list[bool] = []
    core = CoreController(
        DataLayout(tmp_path),
        4321,
        start=lambda *_: {"status": "running", "started": True},
        stop=lambda *_: {"status": "stopped"},
        status=lambda *_: {"status": "running"},
    )
    prefs_file = tmp_path / "desktop.json"
    bridge = DesktopBridge(
        core,
        DesktopPrefs(),
        prefs_file,
        quit_window=lambda: quit_called.append(True),
        open_url=opened.append,
        open_path=opened.append,
    )
    assert bridge.info()["desktop"] is True
    assert bridge.set_keep_core_running(True) == {"keep_core_running": True}
    assert DesktopPrefs.load(prefs_file).keep_core_running is True
    bridge.open_in_browser()
    bridge.open_logs()
    assert opened == ["http://127.0.0.1:4321/", tmp_path / "logs"]
    assert bridge.stop_core_and_quit()["status"] == "stopped"
    # pywebview exportiert öffentliche Attribute – Zustand muss privat sein
    public = [n for n in vars(bridge) if not n.startswith("_")]
    assert public == []


# ---------------------------------------------------------------------- Fenster + echter Core


class FakeEvent:
    def __init__(self) -> None:
        self.handlers: list[Any] = []

    def __iadd__(self, fn: Any) -> FakeEvent:
        self.handlers.append(fn)
        return self

    def fire(self, *args: Any) -> None:
        for fn in self.handlers:
            fn(*args)


class FakeWindow:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.events = type("E", (), {})()
        for name in ("resized", "closed", "shown", "loaded"):
            setattr(self.events, name, FakeEvent())
        self.destroyed = False

    def destroy(self) -> None:
        self.destroyed = True


class FakeWebview:
    renderer = "edgechromium"

    def __init__(self) -> None:
        self.window: FakeWindow | None = None
        self.start_kwargs: dict[str, Any] = {}

    def create_window(self, title: str, url: str, **kwargs: Any) -> FakeWindow:
        self.window = FakeWindow(title=title, url=url, **kwargs)
        return self.window

    def start(self, **kwargs: Any) -> None:
        # Benutzer ändert die Größe und schließt das Fenster
        self.start_kwargs = kwargs
        assert self.window is not None
        self.window.events.resized.fire(1111, 777)
        self.window.events.closed.fire()


@pytest.mark.parametrize("keep", [False, True])
def test_run_starts_core_opens_window_and_handles_close(tmp_path: Path, keep: bool) -> None:
    port = free_port()
    layout = DataLayout(tmp_path / "data")
    layout.ensure()
    DesktopPrefs(keep_core_running=keep).save(app.prefs_path(layout))
    fake = FakeWebview()
    try:
        code = app.run(["--data-dir", str(layout.root), "--port", str(port)], webview_module=fake)
        assert code == 0
        assert fake.window is not None
        kw = fake.window.kwargs
        assert kw["url"] == f"http://127.0.0.1:{port}/" and kw["title"] == "NOVA"
        assert isinstance(kw["js_api"], DesktopBridge) and kw["text_select"] is True
        assert fake.start_kwargs["private_mode"] is False
        assert fake.start_kwargs["storage_path"] == str(layout.root / "webview")
        assert DesktopPrefs.load(app.prefs_path(layout)).width == 1111  # Größe gespeichert
        status = service_status(layout, port)["status"]
        assert status == ("running" if keep else "stopped")
    finally:
        from api.cli import service_stop

        service_stop(layout, port)


def test_run_without_webview2_reports_and_does_not_open_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app, "webview2_status", lambda: (False, "WebView2 missing"))
    out = tmp_path / "self-test.json"
    fake = FakeWebview()
    code = app.run(
        ["--data-dir", str(tmp_path), "--self-test", str(out), "--port", str(free_port())],
        webview_module=fake,
    )
    assert code == 2 and fake.window is None
    assert json.loads(out.read_text())["error"] == "WebView2 missing"


def test_self_test_result_is_written(tmp_path: Path) -> None:
    out = tmp_path / "r.json"
    window = FakeWindow()
    window.events.loaded = type("L", (), {"wait": staticmethod(lambda t: True)})()
    page = {"title": "NOVA", "missing": [], "bridge": True, "desktopMarker": "1", "status": "Ready"}
    window.evaluate_js = lambda js: json.dumps(page)  # type: ignore[attr-defined]
    app._self_test(window, out, lambda: "edgechromium", 5)
    result = json.loads(out.read_text())
    assert result["ok"] is True and window.destroyed
    page["status"] = "Checking status…"  # UI hat den Core noch nicht erreicht
    app._self_test(window, out, lambda: "edgechromium", 1)
    assert json.loads(out.read_text())["ok"] is False
    page["status"] = "Ready"
    page["missing"] = ["composer"]
    app._self_test(window, out, lambda: "edgechromium", 5)
    assert json.loads(out.read_text())["ok"] is False
