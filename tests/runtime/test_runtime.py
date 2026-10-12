"""Ein-Klick-Runtime: Auflösung (GitHub/HF-API), geprüfter Download, Entpacken, Start, Chat.

GitHub, Hugging Face und die Downloads werden per MockTransport simuliert (gleiche HTTP-Pfade);
``llama-server`` ist ein echter Prozess (OpenAI-kompatible Test-Runtime), damit Start,
Health-Check, models.toml und Engine-Neuladen wirklich laufen.
"""

from __future__ import annotations

import hashlib
import io
import json
import stat
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import httpx
import pytest

from api.paths import DataLayout
from api.runtime import (
    MANAGED_MARKER,
    Artifact,
    GpuInfo,
    RuntimeError_,
    RuntimeManager,
    choose_model,
    detect_gpu,
    fetch_verified,
    pick_llama_assets,
    pick_model_file,
    safe_extract,
)
from tests.api.conftest import Factory

ROOT = Path(__file__).resolve().parents[2]
RECOMMENDED = ROOT / "config" / "recommended-models.json"
FAKE_SERVER = f"""#!{sys.executable}
import sys
sys.path.insert(0, {str(ROOT)!r})
import argparse, uvicorn
from tests.api.fake_runtime import create_runtime
p = argparse.ArgumentParser()
p.add_argument("--port", type=int)
p.add_argument("--alias")
args, _ = p.parse_known_args()
app = create_runtime(args.alias, delay=0.0)
uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
"""


def zip_bytes(files: dict[str, tuple[str, int]]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, (content, mode) in files.items():
            info = zipfile.ZipInfo(name)
            info.external_attr = (stat.S_IFREG | mode) << 16
            zf.writestr(info, content)
    return buf.getvalue()


class FakeInternet:
    def __init__(self) -> None:
        self.server_zip = zip_bytes({"build/bin/llama-server": (FAKE_SERVER, 0o755)})
        self.cudart_zip = zip_bytes({"cudart64_12.dll": ("dll", 0o644)})
        self.model = b"GGUF" + b"\0" * 2000
        self.requests: list[str] = []
        self.fail_model_once = False

    @staticmethod
    def sha(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def release(self) -> dict[str, Any]:
        def asset(name: str, data: bytes) -> dict[str, Any]:
            return {
                "name": name,
                "size": len(data),
                "digest": f"sha256:{self.sha(data)}",
                "browser_download_url": f"https://github.com/dl/{name}",
            }

        return {
            "tag_name": "b9999",
            "assets": [
                asset("llama-b9999-bin-win-cpu-x64.zip", b"cpu"),
                asset("llama-b9999-bin-win-cuda-12.4-x64.zip", self.server_zip),
                asset("llama-b9999-bin-win-cuda-13.1-x64.zip", b"13"),
                asset("cudart-llama-bin-win-cuda-12.4-x64.zip", self.cudart_zip),
            ],
        }

    def hf(self) -> dict[str, Any]:
        return {
            "sha": "a" * 40,
            "cardData": {"license": "apache-2.0"},
            "siblings": [
                {"rfilename": "mmproj-F16.gguf", "lfs": {"sha256": "b" * 64, "size": 10}},
                {
                    "rfilename": "Qwen3.5-9B-Q4_K_M.gguf",
                    "lfs": {"sha256": self.sha(self.model), "size": len(self.model)},
                },
                {"rfilename": "Qwen3.5-9B-Q8_0.gguf", "lfs": {"sha256": "c" * 64, "size": 99}},
            ],
        }

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.requests.append(url)
        if url.startswith("https://api.github.com/"):
            return httpx.Response(200, json=self.release())
        if url.startswith("https://huggingface.co/api/models/"):
            return httpx.Response(200, json=self.hf())
        if url.endswith("cuda-12.4-x64.zip") and "cudart" not in url:
            return httpx.Response(200, content=self.server_zip)
        if "cudart" in url:
            return httpx.Response(200, content=self.cudart_zip)
        if url.endswith("Qwen3.5-9B-Q4_K_M.gguf"):
            assert "/resolve/" + "a" * 40 + "/" in url  # fester Commit, nicht „main“
            if self.fail_model_once:
                self.fail_model_once = False
                raise httpx.ReadError("connection dropped", request=request)
            return httpx.Response(200, content=self.model)
        return httpx.Response(404)

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handler), follow_redirects=True)


GPU = GpuInfo("NVIDIA GeForce GTX 1080 Ti", 11.0, "6.1", "580.00")


def manager(tmp_path: Path, web: FakeInternet, gpu: GpuInfo | None = GPU) -> RuntimeManager:
    layout = DataLayout(tmp_path / "data")
    layout.ensure()
    return RuntimeManager(
        layout, RECOMMENDED, client_factory=web.client, gpu_detector=lambda: gpu, platform="win32"
    )


# ---------------------------------------------------------------------- Auswahl & Auflösung


def test_model_choice_by_hardware() -> None:
    models = json.loads(RECOMMENDED.read_text())["models"]
    assert choose_model(models, GPU, 32.0)["id"] == "general-9b"
    assert choose_model(models, GpuInfo("small", 4.0, "7.5", None), 16.0)["id"] == "general-4b"
    assert choose_model(models, None, 16.0)["id"] == "general-4b"


def test_pick_assets_cuda12_and_checksums() -> None:
    web = FakeInternet()
    tag, assets = pick_llama_assets(web.release(), "cuda12")
    assert tag == "b9999"
    assert [a.name for a in assets] == [
        "llama-b9999-bin-win-cuda-12.4-x64.zip",
        "cudart-llama-bin-win-cuda-12.4-x64.zip",
    ]
    release = web.release()
    for a in release["assets"]:
        a.pop("digest")
    with pytest.raises(RuntimeError_, match="no published SHA-256"):
        pick_llama_assets(release, "cuda12")
    with pytest.raises(RuntimeError_, match="Windows only"):
        pick_llama_assets(web.release(), "unsupported")


def test_pick_model_file_prefers_q4_and_pins_revision() -> None:
    web = FakeInternet()
    spec = json.loads(RECOMMENDED.read_text())["models"][0]
    model = pick_model_file(web.hf(), spec)
    assert model.file == "Qwen3.5-9B-Q4_K_M.gguf" and model.revision == "a" * 40
    assert model.url.endswith("/resolve/" + "a" * 40 + "/Qwen3.5-9B-Q4_K_M.gguf")
    assert model.license == "apache-2.0"
    broken = web.hf()
    broken["siblings"][1]["lfs"] = {}
    with pytest.raises(RuntimeError_, match="SHA-256"):
        pick_model_file(broken, spec)


def test_detect_gpu_parses_nvidia_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    import api.runtime as rt

    monkeypatch.setattr(rt.shutil, "which", lambda name: "/usr/bin/nvidia-smi")

    class Out:
        returncode = 0
        stdout = "NVIDIA GeForce GTX 1080 Ti, 11264, 580.88, 6.1\n"

    gpu = detect_gpu(run=lambda *a, **k: Out())
    assert gpu == GpuInfo("NVIDIA GeForce GTX 1080 Ti", 11.0, "6.1", "580.88")
    monkeypatch.setattr(rt.shutil, "which", lambda name: None)
    assert detect_gpu() is None


# ---------------------------------------------------------------------- Download & Entpacken


async def test_fetch_verified_resume_and_mismatch(tmp_path: Path) -> None:
    web = FakeInternet()
    target = tmp_path / "m.gguf"
    art = Artifact(
        "m.gguf",
        "https://huggingface.co/x/resolve/" + "a" * 40 + "/Qwen3.5-9B-Q4_K_M.gguf",
        web.sha(web.model),
        len(web.model),
    )
    async with web.client() as client:
        web.fail_model_once = True
        with pytest.raises(RuntimeError_, match="resume"):
            await fetch_verified(client, art, target)
        await fetch_verified(client, art, target)
        assert target.read_bytes() == web.model
        bad = Artifact("m2.gguf", art.url, "0" * 64, len(web.model))
        with pytest.raises(RuntimeError_, match="mismatch"):
            await fetch_verified(client, bad, tmp_path / "m2.gguf")
        assert not (tmp_path / "m2.gguf").exists() and not (tmp_path / "m2.gguf.part").exists()


def test_safe_extract_rejects_traversal(tmp_path: Path) -> None:
    evil = tmp_path / "evil.zip"
    evil.write_bytes(zip_bytes({"../outside.txt": ("x", 0o644)}))
    with pytest.raises(RuntimeError_, match="unsafe"):
        safe_extract(evil, tmp_path / "out")
    assert not (tmp_path / "outside.txt").exists()


# ---------------------------------------------------------------------- Ende-zu-Ende


def wait_status(client: Any, want: str, timeout: float = 60) -> dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        status: dict[str, Any] = client.get("/runtime/status").json()
        if status["setup"].get("status") in (want, "failed"):
            return status
        time.sleep(0.2)
    raise AssertionError("setup did not finish")


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="the fake llama-server is a Python script; Windows can only start real .exe files",
)
def test_one_click_setup_end_to_end(
    make_client: Factory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import api.runtime as rt

    monkeypatch.setattr(rt, "total_ram_gb", lambda: 32.0)  # Ziel-PC des Nutzers
    web = FakeInternet()
    client, service = make_client()
    service.runtime.client_factory = web.client
    service.runtime.gpu_detector = lambda: GPU
    service.runtime.platform = "win32"
    service.runtime._gpu_checked = False
    with client:
        status = client.get("/runtime/status").json()
        assert status["gpu"]["name"] == "NVIDIA GeForce GTX 1080 Ti"
        assert status["recommended"]["id"] == "general-9b" and status["variant"] == "cuda12"
        r = client.post("/runtime/setup", json={"accept_license": False})
        assert r.status_code == 400 and r.json()["error"]["code"] == "license_not_accepted"
        r = client.post("/runtime/setup", json={"accept_license": True})
        assert r.status_code == 202, r.text
        status = wait_status(client, "completed")
        assert status["setup"]["status"] == "completed", status["setup"]
        assert status["running"] is True and status["installed"]["file"] == "Qwen3.5-9B-Q4_K_M.gguf"
        toml = service.layout.models_config.read_text()
        assert toml.startswith(MANAGED_MARKER) and 'name = "qwen3.5-9b"' in toml
        assert "-ngl" in service.runtime.process.args and "--jinja" in service.runtime.process.args
        # Core hat die Modelle neu geladen → Chat läuft gegen den echten llama-server-Prozess
        sys_status = client.get("/system/status").json()
        assert sys_status["mode"] == "normal"
        events = client.post(
            "/chat/stream", json={"message": "Hello there"}, headers={"X-NOVA-Client": "t"}
        )
        assert events.status_code == 200 and '"event": "done"' in events.text
        # Neustart des Core: verwaltete Runtime startet wieder, ohne erneuten Download
        downloads = len(web.requests)
        service.runtime.stop()
        assert service.runtime.start_runtime() is True
        assert len(web.requests) == downloads
    service.runtime.stop()


def test_custom_config_is_never_overwritten(tmp_path: Path) -> None:
    web = FakeInternet()
    m = manager(tmp_path, web)
    m.layout.models_config.write_text("# my own config\n")
    status = m.status()
    assert status["custom_config"] is True

    async def run() -> None:
        with pytest.raises(RuntimeError_, match="not overwritten"):
            await m.setup(None, True, lambda: None)

    import asyncio

    asyncio.run(run())


def test_non_windows_platform_is_reported(tmp_path: Path) -> None:
    m = manager(tmp_path, FakeInternet())
    m.platform = "linux"
    assert m.status()["platform_supported"] is False
