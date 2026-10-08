"""Test-Doubles für die Modellschicht (nur für Tests, siehe docs/architecture.md §13)."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

from models.base import (
    ChatRequest,
    ChatResponse,
    FinishReason,
    Message,
    ModelError,
    ModelProvider,
    ProviderHealth,
    TokenUsage,
)
from models.capabilities import ModelMetadata

_CODE_RE = re.compile(r"NOVA-[0-9A-F]{6}")
_INT_RE = re.compile(r"value (\d+)")


def make_meta(**overrides: Any) -> ModelMetadata:
    data: dict[str, Any] = {
        "name": "test-model",
        "provider": "local",
        "parameter_count": "8B",
        "context_length": 32768,
        "reasoning_capability": "good",
        "coding_capability": "good",
        "vision_capability": "none",
        "tool_calling": True,
        "speed": "medium",
        "memory_requirement": 6.0,
        "quantization": "Q4_K_M",
    }
    data.update(overrides)
    return ModelMetadata.from_dict(data)


@dataclass
class FakeLlamaServer:
    """Simuliert die HTTP-API eines llama.cpp ``llama-server``.

    ``mode``:
      * ``"smart"``  – beantwortet Health-Check-Prompts korrekt
      * ``"dumb"``   – antwortet immer mit festem Text, keine Tool-Calls
      * ``"short_memory"`` – sieht nur die letzten ``memory_chars`` Zeichen des Prompts
    """

    models: list[str] = field(default_factory=lambda: ["test-model"])
    health_status: int = 200
    mode: str = "smart"
    delay_s: float = 0.0
    memory_chars: int = 2000
    wrong_tool_value: bool = False
    chat_status: int = 200
    chat_error: str = ""
    raw_chat_body: dict[str, Any] | None = None
    requests: list[httpx.Request] = field(default_factory=list)

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    async def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path == "/health":
            if self.health_status == 200:
                return httpx.Response(200, json={"status": "ok"})
            return httpx.Response(
                self.health_status,
                json={"error": {"code": self.health_status, "message": "Loading model"}},
            )
        if path == "/v1/models":
            return httpx.Response(
                200, json={"object": "list", "data": [{"id": m} for m in self.models]}
            )
        if path == "/v1/chat/completions":
            if self.delay_s:
                await asyncio.sleep(self.delay_s)
            if self.chat_status != 200:
                return httpx.Response(
                    self.chat_status,
                    json={"error": {"code": self.chat_status, "message": self.chat_error}},
                )
            if self.raw_chat_body is not None:
                return httpx.Response(200, json=self.raw_chat_body)
            return httpx.Response(200, json=self._complete(json.loads(request.content)))
        return httpx.Response(404, json={"error": {"message": "File Not Found"}})

    def _complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        prompt = "\n".join(str(m.get("content") or "") for m in payload["messages"])
        message: dict[str, Any] = {"role": "assistant", "content": ""}
        finish = "stop"
        if self.mode == "dumb":
            message["content"] = "I am not sure."
        elif payload.get("tools"):
            match = _INT_RE.search(prompt)
            value = int(match.group(1)) if match else 0
            if self.wrong_tool_value:
                value += 1
            message["tool_calls"] = [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": payload["tools"][0]["function"]["name"],
                        "arguments": json.dumps({"value": value}),
                    },
                }
            ]
            finish = "tool_calls"
        else:
            visible = prompt[-self.memory_chars :] if self.mode == "short_memory" else prompt
            match = _CODE_RE.search(visible)
            message["content"] = match.group(0) if match else "No code found."
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "model": payload["model"],
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": len(prompt) // 4, "completion_tokens": 5, "total_tokens": 0},
        }


class StubProvider(ModelProvider):
    """In-Memory-Provider mit pro Modell programmierbarem Verhalten."""

    def __init__(
        self,
        name: str = "local",
        *,
        behaviors: dict[str, Callable[[ChatRequest], str] | Exception | float] | None = None,
        served: list[str] | None = None,
    ) -> None:
        super().__init__(name)
        self.behaviors = behaviors or {}
        self.served = served
        self.calls: list[str] = []
        self.closed = False

    async def chat(self, model: ModelMetadata, request: ChatRequest) -> ChatResponse:
        self.calls.append(model.name)
        behavior = self.behaviors.get(model.name)
        if isinstance(behavior, ModelError):
            raise behavior
        if isinstance(behavior, float):
            await asyncio.sleep(behavior)
            text = "slow"
        elif callable(behavior):
            text = behavior(request)
        else:
            text = f"answer from {model.name}"
        return ChatResponse(
            Message.assistant(text), FinishReason.STOP, TokenUsage(1, 1), model.name, 1.0
        )

    async def list_models(self) -> list[str]:
        if self.served is None:
            raise NotImplementedError
        return list(self.served)

    async def health(self) -> ProviderHealth:
        return ProviderHealth(True, True, "ok")

    async def aclose(self) -> None:
        self.closed = True
