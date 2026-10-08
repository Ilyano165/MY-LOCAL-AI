"""Provider-Erweiterungen fürs Benchmarking: Bild-Eingaben, Runtime-Timings, Runtime-Info."""

from __future__ import annotations

import base64

import pytest

from models.base import ChatRequest, ImageInput, Message
from tests.evaluation.bench_fakes import FakeBenchServer
from tests.models.fakes import make_meta


async def test_images_are_sent_as_openai_content_parts() -> None:
    server = FakeBenchServer(vision_enabled=True)
    provider = server.provider()
    png = b"\x89PNG fake"
    await provider.chat(
        make_meta(), ChatRequest((Message.user("two colored halves?", images=(ImageInput(png),)),))
    )
    content = server.requests[-1]["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "two colored halves?"}
    url = content[1]["image_url"]["url"]
    assert url == "data:image/png;base64," + base64.b64encode(png).decode()


async def test_plain_messages_unchanged() -> None:
    server = FakeBenchServer()
    await server.provider().chat(make_meta(), ChatRequest((Message.user("hi"),)))
    assert server.requests[-1]["messages"][0]["content"] == "hi"


def test_image_input_validation() -> None:
    with pytest.raises(ValueError):
        ImageInput(b"")
    with pytest.raises(ValueError):
        ImageInput(b"x", "text/plain")


async def test_llama_timings_become_runtime_stats() -> None:
    server = FakeBenchServer(per_token_s=0.01)
    response = await server.provider().chat(make_meta(), ChatRequest((Message.user("hi"),)))
    assert response.runtime_stats["predicted_per_second"] == pytest.approx(100.0)
    assert "prompt_per_second" in response.runtime_stats


async def test_ollama_has_no_runtime_stats() -> None:
    server = FakeBenchServer(runtime="ollama")
    response = await server.provider().chat(make_meta(), ChatRequest((Message.user("hi"),)))
    assert response.runtime_stats == {}


async def test_runtime_info_llama_cpp() -> None:
    server = FakeBenchServer(n_ctx=16384, vision_enabled=True, model_path="/m/x.gguf")
    info = await server.provider().runtime_info(make_meta())
    assert info["runtime"] == "llama.cpp"
    assert info["n_ctx"] == 16384
    assert info["modalities"]["vision"] is True
    assert info["model_path"] == "/m/x.gguf"
    assert info["n_params"] == 8_030_261_248 and info["n_ctx_train"] == 131072
    assert "/props" in info["sources"]


async def test_runtime_info_ollama() -> None:
    server = FakeBenchServer(runtime="ollama", ollama_loaded=True)
    info = await server.provider().runtime_info(make_meta())
    assert info["runtime"] == "ollama" and info["runtime_version"] == "0.12.3"
    assert info["loaded"] is True
    assert info["size_vram_bytes"] == 5_500_000_000 and info["n_ctx"] == 8192
    cold = await FakeBenchServer(runtime="ollama").provider().runtime_info(make_meta())
    assert cold["loaded"] is False and "n_ctx" not in cold


async def test_runtime_info_unknown_runtime_reports_nothing_invented() -> None:
    import httpx

    from models.local_provider import OpenAICompatibleProvider

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "x"}]})
        return httpx.Response(404)

    provider = OpenAICompatibleProvider(
        "p", "http://127.0.0.1:1", transport=httpx.MockTransport(handler)
    )
    info = await provider.runtime_info(make_meta())
    assert info == {"runtime": "openai-compatible", "sources": []}


async def test_load_and_unload_ollama() -> None:
    server = FakeBenchServer(runtime="ollama", ollama_loaded=True)
    provider = server.provider()
    await provider.unload(make_meta())
    assert server.ollama_loaded is False
    stats = await provider.load(make_meta())
    assert stats["load_duration_s"] == pytest.approx(0.048)
    assert server.ollama_loaded is True


async def test_load_not_available_on_llama_cpp() -> None:
    provider = FakeBenchServer().provider()
    with pytest.raises(NotImplementedError):
        await provider.load(make_meta())
    with pytest.raises(NotImplementedError):
        await provider.unload(make_meta())
