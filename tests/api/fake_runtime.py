"""Test-Double einer llama.cpp-Runtime (``llama-server``) mit echtem SSE-Streaming.

NUR für Tests und den lokalen UI-Smoke-Test: Die Antworten sind fest geskriptet und als
solche erkennbar. Start als eigener Prozess:
    python -m tests.api.fake_runtime --port 18091 [--delay 0.02] [--model test-model]
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

REPLY = (
    "This is a **scripted test reply** from the fake runtime.\n\n"
    "```python\ndef add(a, b):\n    return a + b\n```\n\n"
    "- item one\n- item two"
)


def reply_for(prompt: str) -> str:
    if "long" in prompt.lower():
        return " ".join(f"word{i}" for i in range(400))
    return REPLY


def create_runtime(model: str = "test-model", delay: float = 0.01) -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": model}]}

    @app.get("/props")
    async def props() -> dict[str, Any]:
        return {"default_generation_settings": {"n_ctx": 8192}, "modalities": {"vision": False}}

    @app.post("/v1/chat/completions")
    async def chat(request: Request) -> Response:
        body = await request.json()
        prompt = str(body["messages"][-1].get("content"))
        text = reply_for(prompt)
        tokens = text.split(" ")
        usage = {"prompt_tokens": len(prompt) // 4 + 1, "completion_tokens": len(tokens)}
        if not body.get("stream"):
            return JSONResponse(
                {
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": text},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": usage,
                }
            )

        async def events() -> AsyncIterator[bytes]:
            for i, token in enumerate(tokens):
                delta = token if i == 0 else " " + token
                chunk = {
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n".encode()
                await asyncio.sleep(delay)
            final = {
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": usage,
                "timings": {"predicted_per_second": 1 / delay if delay else 0.0},
            }
            yield f"data: {json.dumps(final)}\n\n".encode()
            yield b"data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18091)
    parser.add_argument("--delay", type=float, default=0.01)
    parser.add_argument("--model", default="test-model")
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(
        create_runtime(args.model, args.delay),
        host="127.0.0.1",
        port=args.port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
