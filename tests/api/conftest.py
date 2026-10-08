from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from api.app import create_app
from api.config import ApiConfig
from api.service import NovaService
from models.inference import InferenceEngine, ProviderRegistry
from models.local_provider import OpenAICompatibleProvider
from models.model_registry import ModelRegistry
from tests.models.fakes import make_meta


class SSERuntime:
    """MockTransport-Runtime: OpenAI-kompatibel, streamt Antworten als SSE."""

    def __init__(
        self,
        reply: str = "Hello **world**",
        *,
        usage: bool = True,
        fail: bool = False,
        models: list[str] | None = None,
    ) -> None:
        self.reply = reply
        self.usage = usage
        self.fail = fail
        self.models = models or ["test-model"]
        self.payloads: list[dict[str, Any]] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": m} for m in self.models]})
        payload = json.loads(request.content)
        self.payloads.append(payload)
        if self.fail:
            return httpx.Response(503, json={"error": {"message": "Loading model"}})
        tokens = self.reply.split(" ")
        if not payload.get("stream"):
            return httpx.Response(
                200,
                json={
                    "model": payload["model"],
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": self.reply},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 5, "completion_tokens": len(tokens)},
                },
            )
        lines = []
        for i, tok in enumerate(tokens):
            chunk = {
                "model": payload["model"],
                "choices": [{"delta": {"content": tok if i == 0 else " " + tok}}],
            }
            lines.append(f"data: {json.dumps(chunk)}\n\n")
        final: dict[str, Any] = {
            "model": payload["model"],
            "choices": [{"delta": {}, "finish_reason": "stop"}],
        }
        if self.usage:
            final["usage"] = {"prompt_tokens": 5, "completion_tokens": len(tokens)}
        lines.append(f"data: {json.dumps(final)}\n\ndata: [DONE]\n\n")
        return httpx.Response(
            200, content="".join(lines).encode(), headers={"content-type": "text/event-stream"}
        )

    def provider(self, name: str = "local") -> OpenAICompatibleProvider:
        return OpenAICompatibleProvider(
            name, "http://127.0.0.1:9", transport=httpx.MockTransport(self.handle)
        )


def engine_with(*runtimes: tuple[str, SSERuntime]) -> InferenceEngine:
    models = []
    providers = []
    for name, runtime in runtimes:
        models.append(make_meta(name=name, provider=f"p-{name}"))
        runtime.models = [name]
        providers.append(runtime.provider(f"p-{name}"))
    return InferenceEngine(ModelRegistry(models), ProviderRegistry(providers))


Factory = Callable[..., tuple[Any, NovaService]]


@pytest.fixture
def make_client(tmp_path: Path) -> Factory:
    from fastapi.testclient import TestClient

    def factory(engine: InferenceEngine | None = None, **config: Any) -> tuple[Any, NovaService]:
        cfg = ApiConfig(models_config=None, dev_mode=True, data_dir=tmp_path / "data", **config)
        service = NovaService(cfg, engine=engine)
        if engine is not None:
            service.config_error = None
        client = TestClient(create_app(cfg, service=service))
        client.headers.update({"X-NOVA-Client": "test"})
        return client, service

    return factory


def sse_events(text: str) -> list[dict[str, Any]]:
    events = []
    for block in text.strip().split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data:"):
                events.append(json.loads(line[5:]))
    return events
