"""Packaging-Bausteine ohne echten Build (der Build läuft im Release-Workflow)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_frozen_server_command_uses_console_sibling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from api import cli

    for suffix in (".exe", ""):
        folder = tmp_path / (suffix or "plain")
        folder.mkdir()
        launcher = folder / f"nova-launcher{suffix}"
        launcher.touch()
        monkeypatch.setattr(sys, "frozen", True, raising=False)
        monkeypatch.setattr(sys, "executable", str(launcher))
        with pytest.raises(SystemExit, match="not found"):
            cli._server_command()
        (folder / f"nova{suffix}").touch()
        assert cli._server_command() == [str(folder / f"nova{suffix}")]


def test_bundled_version_file_wins_when_frozen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from api import version

    (tmp_path / "VERSION").write_text("9.8.7\n", encoding="utf-8")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    assert version.get_version() == "9.8.7"
    monkeypatch.setattr(sys, "frozen", False)
    assert version.get_version() == version._from_pyproject()


def _launcher() -> object:
    sys.path.insert(0, str(ROOT / "packaging"))
    try:
        import launcher

        return launcher
    finally:
        sys.path.pop(0)


def test_launcher_reports_errors_to_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    launcher = _launcher()
    monkeypatch.setenv("NOVA_DATA_DIR", str(tmp_path))
    assert launcher.run(["frobnicate", "--quiet"]) == 1  # type: ignore[attr-defined]
    log = (tmp_path / "logs" / "launcher.log").read_text(encoding="utf-8")
    assert "Unknown action: frobnicate" in log


def test_launcher_stop_when_not_running(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    launcher = _launcher()
    monkeypatch.setenv("NOVA_DATA_DIR", str(tmp_path))
    assert launcher.run(["stop", "--quiet", "--port", "1"]) == 0  # type: ignore[attr-defined]


def test_wix_source_invariants() -> None:
    wxs = (ROOT / "packaging" / "windows" / "nova.wxs").read_text(encoding="utf-8")
    assert 'Scope="perUser"' in wxs
    assert "UpgradeCode=" in wxs and "<MajorUpgrade" in wxs and "DowngradeErrorMessage" in wxs
    assert "VersionNT64" in wxs  # Launch-Bedingung 64-bit
    # Benutzerdaten (%LOCALAPPDATA%\NOVA) gehören nicht zum Paket
    assert 'Name="Programs"' in wxs and "RemoveFolderEx" not in wxs
