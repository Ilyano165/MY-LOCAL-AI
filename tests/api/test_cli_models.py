"""CLI, Hintergrunddienst, Modellverwaltung, Datenerhalt."""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from api import cli
from api.model_manager import CatalogEntry, CatalogError, ModelManager
from api.paths import DataLayout, default_data_dir
from api.version import __version__
from tests.api.conftest import Factory

# ---------------------------------------------------------------------- Pfade & Version


def test_data_dir_is_platform_specific() -> None:
    win = default_data_dir({"LOCALAPPDATA": r"C:\Users\a\AppData\Local"}, "win32")
    assert str(win).replace("\\", "/").endswith("AppData/Local/NOVA")
    assert default_data_dir({}, "linux").name == ".nova"
    assert default_data_dir({"NOVA_DATA_DIR": "/srv/nova"}, "win32") == Path("/srv/nova")


def test_version_matches_pyproject() -> None:
    import tomllib

    with (Path(__file__).resolve().parents[2] / "pyproject.toml").open("rb") as fh:
        assert __version__ == tomllib.load(fh)["project"]["version"]


def test_cli_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        cli.main(["--version"])
    assert __version__ in capsys.readouterr().out


# ---------------------------------------------------------------------- serve-Konfiguration


def ns(tmp_path: Path, **kw: Any) -> argparse.Namespace:
    data = {
        "data_dir": tmp_path,
        "config": None,
        "dev": False,
        "host": "127.0.0.1",
        "router": "rules",
        "learned_ranker": None,
        "allow_remote": False,
        "tls_cert": None,
        "tls_key": None,
        "cors_origin": None,
    }
    data.update(kw)
    return argparse.Namespace(**data)


def test_installed_without_model_starts_in_setup_mode(tmp_path: Path) -> None:
    config, mode = cli.build_config(ns(tmp_path))
    assert mode == "setup" and config.dev_mode and config.mode == "setup"


def test_uses_models_toml_from_data_dir(tmp_path: Path) -> None:
    (tmp_path / "models.toml").write_text("")
    config, mode = cli.build_config(ns(tmp_path))
    assert mode == "normal" and config.models_config == tmp_path / "models.toml"


def test_network_access_needs_tls_token_and_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(SystemExit, match=r"--allow-remote.*TLS.*NOVA_API_TOKEN"):
        cli.build_config(ns(tmp_path, host="0.0.0.0"))
    monkeypatch.setenv("NOVA_API_TOKEN", "t")
    config, _ = cli.build_config(
        ns(tmp_path, host="0.0.0.0", allow_remote=True, tls_cert="c.pem", tls_key="k.pem")
    )
    assert config.allow_remote is True
    local, _ = cli.build_config(ns(tmp_path))
    assert local.allow_remote is False


# ---------------------------------------------------------------------- Integrationen per CLI


def run_cli(args: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, Any]:
    with pytest.raises(SystemExit) as info:
        cli.main(args)
    out = capsys.readouterr().out
    try:
        return int(info.value.code or 0), json.loads(out)
    except ValueError:
        return int(info.value.code or 0), out


def test_integration_lifecycle_via_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    base = ["--data-dir", str(tmp_path), "integrations"]
    code, created = run_cli(
        [
            *base,
            "create",
            "--name",
            "ic-ware-hq",
            "--scope",
            "chat:complete",
            "--scope",
            "models:read",
            "--rate-limit",
            "30",
        ],
        capsys,
    )
    assert code == 0 and created["api_key"].startswith("nova_")
    assert created["scopes"] == ["chat:complete", "models:read"]
    code, listed = run_cli([*base, "list"], capsys)
    assert "api_key" not in listed[0] and listed[0]["key_prefix"].endswith("_…")
    code, rotated = run_cli([*base, "rotate", "ic-ware-hq"], capsys)
    assert rotated["api_key"] != created["api_key"]
    code, revoked = run_cli([*base, "revoke", "ic-ware-hq"], capsys)
    assert revoked["active"] is False
    code, _ = run_cli([*base, "create", "--name", "bad", "--scope", "root:all"], capsys)
    assert code == 2
    code, _ = run_cli([*base, "create", "--name", "bad", "--scope", "agent:tool:format_c"], capsys)
    assert code == 2


def test_data_purge_requires_confirmation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = tmp_path / "nova"
    data.mkdir()
    (data / "conversations.db").write_text("x")
    code, _ = run_cli(["--data-dir", str(data), "data", "purge", "--port", "1"], capsys)
    assert code == 1 and data.exists()
    code, _ = run_cli(
        ["--data-dir", str(data), "data", "purge", "--port", "1", "--confirm", str(data)], capsys
    )
    assert code == 0 and not data.exists()


# ---------------------------------------------------------------------- Hintergrunddienst (echt)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_background_service_start_status_stop_and_data_survives(tmp_path: Path) -> None:
    """Startet NOVA wirklich als Hintergrundprozess, nutzt die API, stoppt, startet erneut."""
    layout = DataLayout(tmp_path / "data")
    port = free_port()
    started = cli.service_start(layout, port, wait_s=60)
    try:
        assert started["status"] == "running" and started["version"] == __version__
        assert cli.service_status(layout, port)["status"] == "running"
        assert cli.service_start(layout, port)["started"] is False  # läuft schon
        system = httpx.get(f"http://127.0.0.1:{port}/system/status").json()
        assert system["mode"] == "setup" and system["model_message"] == "No local model available."
        r = httpx.post(f"http://127.0.0.1:{port}/conversations", headers={"X-NOVA-Client": "t"})
        cid = r.json()["id"]
        httpx.put(
            f"http://127.0.0.1:{port}/settings",
            headers={"X-NOVA-Client": "t"},
            json={"temperature": 0.25},
        )
    finally:
        stopped = cli.service_stop(layout, port)
    assert stopped["stopped"] is True and cli.service_status(layout, port)["status"] == "stopped"
    assert not layout.service_pid.exists()
    # „Update“: neuer Prozess, gleiches Datenverzeichnis → Daten sind noch da
    cli.service_start(layout, port, wait_s=60)
    try:
        conversations = httpx.get(f"http://127.0.0.1:{port}/conversations").json()
        assert [c["id"] for c in conversations["conversations"]] == [cid]
        assert httpx.get(f"http://127.0.0.1:{port}/settings").json()["temperature"] == 0.25
    finally:
        cli.service_stop(layout, port)
    assert (layout.logs / "nova.log").is_file() and (layout.logs / "service.log").is_file()


def test_shutdown_endpoint_requires_token(make_client: Factory) -> None:
    client, _ = make_client()
    assert client.post("/internal/shutdown").status_code == 403
    assert (
        client.post("/internal/shutdown", headers={"X-NOVA-Shutdown-Token": "guess"}).status_code
        == 403
    )


# ---------------------------------------------------------------------- Modellverwaltung


GGUF = b"GGUF" + b"\x03\x00\x00\x00" + b"\x00" * 2040


def catalog_entry(data: bytes = GGUF, **kw: Any) -> dict[str, Any]:
    entry = {
        "id": "test-model-q4",
        "name": "Test Model",
        "file": "test-model.gguf",
        "url": "https://models.example/test-model.gguf",
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
        "license": "Apache-2.0",
        "license_url": "https://www.apache.org/licenses/LICENSE-2.0",
        "requires_license_acceptance": True,
        "min_ram_gb": 1,
    }
    entry.update(kw)
    return entry


def write_catalog(layout: DataLayout, *entries: dict[str, Any]) -> None:
    layout.ensure()
    layout.catalog.write_text(json.dumps({"models": list(entries)}))


class Server:
    def __init__(self, data: bytes, *, honor_range: bool = True, fail_after: int | None = None):
        self.data = data
        self.honor_range = honor_range
        self.fail_after = fail_after
        self.requests: list[httpx.Request] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        start = 0
        header = request.headers.get("range")
        if header and self.honor_range:
            start = int(header.split("=")[1].split("-")[0])
        body = self.data[start:]
        if self.fail_after is not None:
            failing = self.fail_after

            class Broken(httpx.AsyncByteStream):
                async def __aiter__(self):  # type: ignore[no-untyped-def]
                    yield body[:failing]
                    raise httpx.ReadError("connection reset")

            return httpx.Response(200, stream=Broken())  # type: ignore[arg-type]
        return httpx.Response(206 if start else 200, content=body)

    def manager(self, layout: DataLayout, ram: float | None = 64.0) -> ModelManager:
        transport = httpx.MockTransport(self.handle)
        return ModelManager(
            layout, client=httpx.AsyncClient(transport=transport), ram_gb=lambda: ram
        )


def test_catalog_rejects_placeholders_and_unsupported_formats(tmp_path: Path) -> None:
    example = json.loads(
        (Path(__file__).resolve().parents[2] / "config" / "model-catalog.example.json").read_text()
    )["models"][0]
    with pytest.raises(CatalogError, match="https"):
        CatalogEntry.from_dict(example)
    with pytest.raises(CatalogError, match="unsupported format"):
        CatalogEntry.from_dict(catalog_entry(file="model.bin"))
    with pytest.raises(CatalogError, match="plain file name"):
        CatalogEntry.from_dict(catalog_entry(file="../evil.gguf"))
    with pytest.raises(CatalogError, match="sha256"):
        CatalogEntry.from_dict(catalog_entry(sha256="abc"))


def test_plan_detects_ram_and_disk_problems(tmp_path: Path) -> None:
    layout = DataLayout(tmp_path)
    write_catalog(
        layout,
        catalog_entry(min_ram_gb=128),
        catalog_entry(id="huge", file="huge.gguf", size_bytes=10**15),
    )
    manager = Server(GGUF).manager(layout, ram=16.0)
    plan = manager.plan(manager.entry("test-model-q4"))
    assert not plan.ok and "Not enough memory" in plan.problems[0]
    huge = manager.plan(manager.entry("huge"))
    assert any("disk space" in p for p in huge.problems)
    assert plan.to_dict()["size_bytes"] == len(GGUF) and plan.license == "Apache-2.0"


async def test_download_success_with_integrity_check(tmp_path: Path) -> None:
    layout = DataLayout(tmp_path)
    write_catalog(layout, catalog_entry())
    manager = Server(GGUF).manager(layout)
    entry = manager.entry("test-model-q4")
    with pytest.raises(CatalogError, match="License"):
        await manager.download(entry, accept_license=False)
    seen: list[int] = []
    meta = await manager.download(
        entry, accept_license=True, progress=lambda done, total: seen.append(done)
    )
    target = layout.models / "test-model.gguf"
    assert target.read_bytes() == GGUF and seen[-1] == len(GGUF)
    assert meta["license_accepted_at"] and meta["sha256"] == entry.sha256
    assert manager.installed()[0]["id"] == "test-model-q4"
    assert not list(layout.downloads.glob("*.part"))
    again = await manager.download(entry, accept_license=True)
    assert again["skipped"] is True


async def test_checksum_mismatch_discards_file(tmp_path: Path) -> None:
    layout = DataLayout(tmp_path)
    write_catalog(layout, catalog_entry(sha256="0" * 64))
    manager = Server(GGUF).manager(layout)
    with pytest.raises(CatalogError, match="SHA-256 mismatch"):
        await manager.download(manager.entry("test-model-q4"), accept_license=True)
    assert not (layout.models / "test-model.gguf").exists()
    assert not list(layout.downloads.glob("*.part"))
    assert manager.progress["test-model-q4"]["status"] == "failed"


async def test_non_gguf_is_rejected(tmp_path: Path) -> None:
    data = b"PK\x03\x04" + b"\x00" * 100
    layout = DataLayout(tmp_path)
    write_catalog(layout, catalog_entry(data))
    manager = Server(data).manager(layout)
    with pytest.raises(CatalogError, match="not a GGUF"):
        await manager.download(manager.entry("test-model-q4"), accept_license=True)


async def test_interrupted_download_resumes(tmp_path: Path) -> None:
    layout = DataLayout(tmp_path)
    write_catalog(layout, catalog_entry())
    broken = Server(GGUF, fail_after=1000)
    manager = broken.manager(layout)
    with pytest.raises(CatalogError, match="interrupted"):
        await manager.download(manager.entry("test-model-q4"), accept_license=True)
    part = layout.downloads / "test-model.gguf.part"
    assert part.stat().st_size == 1000
    assert "Resuming" in " ".join(manager.plan(manager.entry("test-model-q4")).warnings)
    server = Server(GGUF)
    resumed = server.manager(layout)
    await resumed.download(resumed.entry("test-model-q4"), accept_license=True)
    assert server.requests[0].headers["range"] == "bytes=1000-"
    assert (layout.models / "test-model.gguf").read_bytes() == GGUF


async def test_server_without_range_support_restarts_cleanly(tmp_path: Path) -> None:
    layout = DataLayout(tmp_path)
    write_catalog(layout, catalog_entry())
    layout.downloads.mkdir(parents=True)
    (layout.downloads / "test-model.gguf.part").write_bytes(GGUF[:500])
    manager = Server(GGUF, honor_range=False).manager(layout)
    await manager.download(manager.entry("test-model-q4"), accept_license=True)
    assert (layout.models / "test-model.gguf").read_bytes() == GGUF


def test_setup_endpoints(make_client: Factory) -> None:
    client, service = make_client()
    empty = client.get("/models/catalog").json()
    assert empty["models"] == [] and empty["catalog_path"].endswith("model-catalog.json")
    write_catalog(service.layout, catalog_entry())
    data = client.get("/models/catalog").json()
    assert data["models"][0]["plan"]["requires_license_acceptance"] is True
    r = client.post("/models/downloads", json={"id": "test-model-q4"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "license_not_accepted"
    assert (
        client.post("/models/downloads", json={"id": "nope", "accept_license": True}).status_code
        == 404
    )


def test_cli_models_catalog_and_plan(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    layout = DataLayout(tmp_path)
    write_catalog(layout, catalog_entry())
    code, catalog = run_cli(["--data-dir", str(tmp_path), "models", "catalog"], capsys)
    assert code == 0 and catalog[0]["id"] == "test-model-q4"
    code, plan = run_cli(["--data-dir", str(tmp_path), "models", "plan", "test-model-q4"], capsys)
    assert code == 0 and plan["size_bytes"] == len(GGUF)
    start = time.time()
    code, _ = run_cli(["--data-dir", str(tmp_path), "models", "download", "test-model-q4"], capsys)
    assert code == 2 and time.time() - start < 5  # ohne --accept-license kein Download
