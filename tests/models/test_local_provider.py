from __future__ import annotations

import json

import httpx
import pytest

from models.base import (
    ChatRequest,
    ContextLengthExceededError,
    FinishReason,
    GenerationParams,
    InvalidResponseError,
    Message,
    ModelError,
    ModelNotFoundError,
    ModelTimeoutError,
    ProviderUnavailableError,
    ToolCall,
    ToolSpec,
)
from models.local_provider import OpenAICompatibleProvider
from tests.models.fakes import FakeLlamaServer, make_meta

BASE = "http://127.0.0.1:8080"


def provider_for(server: FakeLlamaServer, **kwargs: object) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider("local", BASE, transport=server.transport(), **kwargs)  # type: ignore[arg-type]


def simple_request(**kwargs: object) -> ChatRequest:
    return ChatRequest(messages=(Message.user("Reply with exactly: NOVA-ABC123"),), **kwargs)  # type: ignore[arg-type]


async def test_chat_roundtrip_and_payload() -> None:
    server = FakeLlamaServer()
    provider = provider_for(server)
    meta = make_meta(served_name="alias-in-runtime")
    request = simple_request(
        params=GenerationParams(temperature=0.0, max_tokens=8, stop=("\n",), seed=1)
    )
    response = await provider.chat(meta, request)

    assert response.message.content == "NOVA-ABC123"
    assert response.finish_reason is FinishReason.STOP
    assert response.usage.completion_tokens == 5
    assert response.latency_ms >= 0
    payload = json.loads(server.requests[-1].content)
    assert payload["model"] == "alias-in-runtime"
    assert payload["max_tokens"] == 8 and payload["stop"] == ["\n"] and payload["seed"] == 1
    assert payload["stream"] is False
    assert "tools" not in payload
    await provider.aclose()


async def test_tool_calls_are_serialized_and_parsed() -> None:
    server = FakeLlamaServer()
    provider = provider_for(server)
    tool = ToolSpec(
        "record_number", "rec", {"type": "object", "properties": {"value": {"type": "integer"}}}
    )
    history = (
        Message.system("sys"),
        Message.user("Call the record_number tool with value 7."),
        Message.assistant("", [ToolCall("c0", "record_number", {"value": 1})]),
        Message.tool("c0", "ok"),
        Message.user("Again with value 7."),
    )
    response = await provider.chat(make_meta(), ChatRequest(messages=history, tools=(tool,)))

    assert response.finish_reason is FinishReason.TOOL_CALLS
    assert response.message.tool_calls[0].name == "record_number"
    assert response.message.tool_calls[0].arguments == {"value": 7}
    payload = json.loads(server.requests[-1].content)
    assert payload["tools"][0]["function"]["name"] == "record_number"
    assert payload["messages"][2]["tool_calls"][0]["function"]["arguments"] == '{"value": 1}'
    assert payload["messages"][3] == {"role": "tool", "content": "ok", "tool_call_id": "c0"}


async def test_invalid_tool_arguments_raise() -> None:
    body = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "x", "function": {"name": "f", "arguments": "{not json"}}
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }
    provider = provider_for(FakeLlamaServer(raw_chat_body=body))
    with pytest.raises(InvalidResponseError, match="kein gültiges JSON"):
        await provider.chat(make_meta(), simple_request())


@pytest.mark.parametrize("body", [{}, {"choices": []}, {"choices": [{"finish_reason": "stop"}]}])
async def test_malformed_responses_raise(body: dict[str, object]) -> None:
    provider = provider_for(FakeLlamaServer(raw_chat_body=body))
    with pytest.raises(InvalidResponseError):
        await provider.chat(make_meta(), simple_request())


async def test_unknown_finish_reason_maps_to_other() -> None:
    body = {"choices": [{"message": {"content": "hi"}, "finish_reason": "eos_token"}]}
    response = await provider_for(FakeLlamaServer(raw_chat_body=body)).chat(
        make_meta(), simple_request()
    )
    assert response.finish_reason is FinishReason.OTHER
    assert response.usage.total_tokens == 0


@pytest.mark.parametrize(
    ("status", "message", "error"),
    [
        (404, "model not found", ModelNotFoundError),
        (400, "the request exceeds the available context size", ContextLengthExceededError),
        (503, "Loading model", ProviderUnavailableError),
        (500, "boom", ModelError),
    ],
)
async def test_http_errors_are_mapped(status: int, message: str, error: type[Exception]) -> None:
    provider = provider_for(FakeLlamaServer(chat_status=status, chat_error=message))
    with pytest.raises(error, match=message.split()[0]):
        await provider.chat(make_meta(), simple_request())


async def test_transport_errors_are_mapped() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    def slow(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(ProviderUnavailableError):
        await OpenAICompatibleProvider("p", BASE, transport=httpx.MockTransport(refuse)).chat(
            make_meta(), simple_request()
        )
    with pytest.raises(ModelTimeoutError):
        await OpenAICompatibleProvider("p", BASE, transport=httpx.MockTransport(slow)).chat(
            make_meta(), simple_request()
        )


async def test_list_models() -> None:
    provider = provider_for(FakeLlamaServer(models=["a", "b"]))
    assert await provider.list_models() == ["a", "b"]


async def test_health_states() -> None:
    assert (await provider_for(FakeLlamaServer()).health()).ready
    loading = await provider_for(FakeLlamaServer(health_status=503)).health()
    assert loading.reachable and not loading.ready and "Loading" in loading.detail

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    down = await OpenAICompatibleProvider("p", BASE, transport=httpx.MockTransport(refuse)).health()
    assert not down.reachable


async def test_health_without_health_endpoint_uses_models() -> None:
    server = FakeLlamaServer()
    health = await provider_for(server, health_path=None).health()
    assert health.ready
    assert server.requests[-1].url.path == "/v1/models"


async def test_ensure_loaded_requests_single_token() -> None:
    server = FakeLlamaServer()
    await provider_for(server).ensure_loaded(make_meta())
    assert json.loads(server.requests[-1].content)["max_tokens"] == 1


@pytest.mark.parametrize("url", ["http://192.168.1.10:8080", "http://example.com"])
def test_remote_hosts_rejected_by_default(url: str) -> None:
    with pytest.raises(ValueError, match="Loopback"):
        OpenAICompatibleProvider("p", url)
    OpenAICompatibleProvider("p", url, allow_remote=True)


@pytest.mark.parametrize(
    "url", ["http://localhost:11434", "http://[::1]:8080", "http://127.0.0.2:1"]
)
def test_loopback_variants_allowed(url: str) -> None:
    OpenAICompatibleProvider("p", url)


def test_invalid_scheme_rejected() -> None:
    with pytest.raises(ValueError, match="http"):
        OpenAICompatibleProvider("p", "ftp://127.0.0.1")


async def test_api_key_comes_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="NOVA_TEST_KEY"):
        OpenAICompatibleProvider("p", BASE, api_key_env="NOVA_TEST_KEY")
    monkeypatch.setenv("NOVA_TEST_KEY", "secret-value")
    server = FakeLlamaServer()
    await provider_for(server, api_key_env="NOVA_TEST_KEY").list_models()
    assert server.requests[-1].headers["Authorization"] == "Bearer secret-value"


def test_tool_message_without_id_rejected() -> None:
    from models.base import Role

    with pytest.raises(ValueError):
        OpenAICompatibleProvider._serialize_message(Message(Role.TOOL, "x"))


def test_request_validation() -> None:
    with pytest.raises(ValueError):
        ChatRequest(messages=())
    with pytest.raises(ValueError):
        GenerationParams(temperature=3)
    with pytest.raises(ValueError):
        GenerationParams(max_tokens=0)
    with pytest.raises(ValueError):
        simple_request(timeout_s=0)
