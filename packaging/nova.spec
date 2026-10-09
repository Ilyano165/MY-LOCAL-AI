# PyInstaller-Spezifikation: ein Ordner (onedir) mit zwei Programmen.
#   nova.exe           Konsole – CLI und der eigentliche Server (`nova serve`)
#   nova-launcher.exe  ohne Konsolenfenster – Startmenü/Autostart/Installer
# Aufruf über `python packaging/build.py` (schreibt VERSION, prüft das Ergebnis).
# ruff: noqa
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

ROOT = Path(SPECPATH).parent
BUILD = ROOT / "build" / "packaging"

hidden = collect_submodules("uvicorn")
for package in ("api", "router", "models", "agents", "tools", "memory", "evaluation"):
    hidden += collect_submodules(package)

datas = [
    (str(ROOT / "api" / "static"), "api/static"),
    (str(ROOT / "evaluation" / "datasets"), "evaluation/datasets"),
    (str(ROOT / "config" / "models.example.toml"), "config"),
    (str(ROOT / "config" / "model-catalog.example.json"), "config"),
    (str(BUILD / "VERSION"), "."),
]
excludes = ["tkinter", "pytest", "playwright", "mypy", "ruff", "openai", "tests"]


def analysis(script):
    return Analysis(
        [str(ROOT / "packaging" / script)],
        pathex=[str(ROOT)],
        datas=datas,
        hiddenimports=hidden,
        excludes=excludes,
        noarchive=False,
    )


cli = analysis("nova_main.py")
launcher = analysis("launcher.py")

cli_exe = EXE(
    PYZ(cli.pure),
    cli.scripts,
    [],
    exclude_binaries=True,
    name="nova",
    console=True,
    upx=False,
)
launcher_exe = EXE(
    PYZ(launcher.pure),
    launcher.scripts,
    [],
    exclude_binaries=True,
    name="nova-launcher",
    console=False,
    upx=False,
)
COLLECT(
    cli_exe,
    cli.binaries,
    cli.datas,
    launcher_exe,
    launcher.binaries,
    launcher.datas,
    name="NOVA",
    upx=False,
)
