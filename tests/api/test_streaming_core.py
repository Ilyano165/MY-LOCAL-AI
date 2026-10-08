"""Streaming im Provider (SSE) und in der InferenceEngine (Fallback vor dem ersten Token)."""

from __future__ import annotations

import httpx
import pytest

from models.base import (
    ChatRequest,
    ContextLengthExceededError,
    Message,
    ProviderUnavailableError,
    ToolSpec,
)
from tests.api.conftest import SSERuntime, engine_with
from tests.models.fakes import StubProvider, make_meta


async def collect(provider, request):  # type: ignore[no-untyped-def]
    return [c async for c in provider.stream(make_meta(), request)]


async def test_sse_stream_yields_deltas_and_final_usage() -> None:
    runtime = SSERuntime("one two three")
    chunks = await collect(runtime.provider(), ChatRequest((Message.user("hi"),)))
    assert "".join(c.delta for c in chunks) == "one two three"
    assert all(c.streamed for c in chunks)
    final = chunks[-1]
    assert final.finish_reason is not None and final.finish_reason.value == "stop"
    assert final.usage is not None and final.usage.completion_tokens == 3
    assert runtime.payloads[0]["stream"] is True
    assert runtime.payloads[0]["stream_options"] == {"include_usage": True}


async def test_stream_without_usage_reports_none() -> None:
    chunks = await collect(
        SSERuntime("a b", usage=False).provider(), ChatRequest((Message.user("hi"),))
    )
    assert chunks[-1].usage is None  # nichts erfinden


async def test_stream_maps_http_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "context length exceeded"}})

    from models.local_provider import OpenAICompatibleProvider

    provider = OpenAICompatibleProvider(
        "p", "http://127.0.0.1:9", transport=httpx.MockTransport(handler)
    )
    with pytest.raises(ContextLengthExceededError):
        await collect(provider, ChatRequest((Message.user("hi"),)))


async def test_stream_with_tools_falls_back_to_single_chunk() -> None:
    runtime = SSERuntime("tool answer")
    request = ChatRequest((Message.user("hi"),), tools=(ToolSpec("t", "d"),))
    chunks = await collect(runtime.provider(), request)
    assert len(chunks) == 1 and chunks[0].streamed is False
    assert chunks[0].delta == "tool answer"


async def test_default_provider_stream_is_marked_not_streamed() -> None:
    chunks = await collect(StubProvider(), ChatRequest((Message.user("hi"),)))
    assert len(chunks) == 1 and chunks[0].streamed is False


async def test_engine_stream_falls_back_before_first_token() -> None:
    broken, good = SSERuntime(fail=True), SSERuntime("ok then")
    engine = engine_with(("a", broken), ("b", good))
    models = engine.models.list()
    out = [
        (m.name, c.delta) async for m, c in engine.stream(models, ChatRequest((Message.user("x"),)))
    ]
    assert {name for name, _ in out} == {"b"}
    assert "".join(d for _, d in out) == "ok then"


async def test_engine_stream_all_fail() -> None:
    engine = engine_with(("a", SSERuntime(fail=True)))
    with pytest.raises(ProviderUnavailableError, match="Alle Modelle fehlgeschlagen"):
        async for _ in engine.stream(engine.models.list(), ChatRequest((Message.user("x"),))):
            pass
