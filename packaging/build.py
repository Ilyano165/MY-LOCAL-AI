"""Release-Build des Programmordners (PyInstaller, onedir) – plattformunabhängig.

    python packaging/build.py [--version X.Y.Z] [--skip-smoke]

1. Version aus ``pyproject.toml`` (einzige Quelle) oder ``--version`` → ``build/packaging/VERSION``
2. PyInstaller mit ``packaging/nova.spec`` → ``dist/NOVA/``
3. Rauchtest: ``nova --version`` muss die Version melden; ``nova serve`` im Setup-Modus muss
   ``/api/v1/health`` beantworten (eigenes temporäres Datenverzeichnis).

Das MSI baut ``packaging/windows/build-msi.ps1`` aus ``dist/NOVA``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import tomllib
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist" / "NOVA"
EXE_SUFFIX = ".exe" if sys.platform == "win32" else ""


def project_version() -> str:
    with (ROOT / "pyproject.toml").open("rb") as fh:
        return str(tomllib.load(fh)["project"]["version"])


def build(version: str) -> None:
    stage = ROOT / "build" / "packaging"
    stage.mkdir(parents=True, exist_ok=True)
    (stage / "VERSION").write_text(version + "\n", encoding="utf-8")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            str(ROOT / "packaging" / "nova.spec"),
            "--noconfirm",
            "--clean",
            "--distpath",
            str(ROOT / "dist"),
            "--workpath",
            str(ROOT / "build" / "pyinstaller"),
        ],
        check=True,
        cwd=ROOT,
    )


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def smoke(version: str) -> None:
    nova = DIST / f"nova{EXE_SUFFIX}"
    out = subprocess.run([nova, "--version"], capture_output=True, text=True, check=True)
    reported = out.stdout.strip()
    if reported != f"NOVA {version}":
        raise SystemExit(f"Smoke test: expected 'NOVA {version}', got {reported!r}")
    if not (DIST / f"nova-launcher{EXE_SUFFIX}").is_file():
        raise SystemExit("Smoke test: nova-launcher missing")
    port = _free_port()
    with tempfile.TemporaryDirectory() as data:
        env = {**os.environ, "NOVA_DATA_DIR": data}
        server = subprocess.Popen(
            [nova, "--data-dir", data, "serve", "--port", str(port)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.time() + 60
            health = None
            while time.time() < deadline and server.poll() is None:
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/api/v1/health", timeout=2
                    ) as r:
                        health = json.loads(r.read())
                        break
                except OSError:
                    time.sleep(0.5)
            if health is None or health.get("version") != version:
                log = server.stdout.read().decode(errors="replace") if server.poll() else ""
                raise SystemExit(f"Smoke test: server not healthy ({health!r})\n{log}")
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as r:
                if b"NOVA" not in r.read():
                    raise SystemExit("Smoke test: UI not served")
        finally:
            server.terminate()
            server.wait(timeout=30)
    print(f"Smoke test passed: {reported}, health OK, UI served")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default=None)
    parser.add_argument("--skip-smoke", action="store_true")
    args = parser.parse_args()
    version = args.version or project_version()
    if DIST.exists():
        shutil.rmtree(DIST)
    build(version)
    if not args.skip_smoke:
        smoke(version)


if __name__ == "__main__":
    main()
