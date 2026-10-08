"""Simulierte Runtimes für Benchmark-Tests (llama.cpp ``llama-server`` und Ollama).

Die Antworten sind skriptgesteuert (``skill="good"`` löst alle Aufgaben, ``"bad"`` liefert
plausible, aber falsche Antworten). Die Dekodierzeit wird mit ``per_token_s`` simuliert,
damit Durchsatzmessungen echte Wanduhr-Messungen gegen einen bekannten Sollwert sind.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any

import httpx

from models.local_provider import OpenAICompatibleProvider

GOOD_PALINDROME = """```python
def is_palindrome(s: str) -> bool:
    cleaned = [c.lower() for c in s if c.isalnum()]
    return cleaned == cleaned[::-1]
```"""
BAD_PALINDROME = """```python
def is_palindrome(s):
    return s == s[::-1]
```"""
GOOD_MEDIAN = """Here is the fix:
```python
def median(values):
    if not values:
        raise ValueError("empty")
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2
```"""


@dataclass
class FakeBenchServer:
    skill: str = "good"
    runtime: str = "llama.cpp"
    """``"llama.cpp"`` oder ``"ollama"``."""
    per_token_s: float = 0.001
    overhead_s: float = 0.005
    n_ctx: int = 32768
    tools_enabled: bool = True
    vision_enabled: bool = False
    model_name: str = "test-model"
    model_path: str | None = None
    ollama_loaded: bool = False
    fail_chat: bool = False
    ready_after: int = 0
    """``/health`` meldet erst nach so vielen Aufrufen „bereit“ (Ladevorgang simulieren)."""
    requests: list[dict[str, Any]] = field(default_factory=list)
    health_calls: int = 0
    unloads: int = 0
    loads: int = 0

    def provider(self, name: str = "local") -> OpenAICompatibleProvider:
        port = "11434" if self.runtime == "ollama" else "8080"
        return OpenAICompatibleProvider(
            name,
            f"http://127.0.0.1:{port}",
            health_path=None if self.runtime == "ollama" else "/health",
            transport=httpx.MockTransport(self.handle),
        )

    # ------------------------------------------------------------------ HTTP

    async def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/health":
            self.health_calls += 1
            if self.health_calls <= self.ready_after:
                return httpx.Response(503, json={"error": {"message": "Loading model"}})
            return httpx.Response(200, json={"status": "ok"})
        if path == "/v1/models":
            entry: dict[str, Any] = {"id": self.model_name}
            if self.runtime == "llama.cpp":
                entry["meta"] = {
                    "n_params": 8_030_261_248,
                    "size": 4_920_000_000,
                    "n_ctx_train": 131072,
                }
            return httpx.Response(200, json={"object": "list", "data": [entry]})
        if self.runtime == "llama.cpp" and path == "/props":
            props: dict[str, Any] = {
                "default_generation_settings": {"n_ctx": self.n_ctx},
                "total_slots": 1,
                "build_info": "b6500-test",
                "modalities": {"vision": self.vision_enabled, "audio": False},
            }
            if self.model_path:
                props["model_path"] = self.model_path
            return httpx.Response(200, json=props)
        if self.runtime == "ollama":
            if path == "/api/version":
                return httpx.Response(200, json={"version": "0.12.3"})
            if path == "/api/ps":
                models = []
                if self.ollama_loaded:
                    models.append(
                        {
                            "name": self.model_name,
                            "size": 6_000_000_000,
                            "size_vram": 5_500_000_000,
                            "context_length": 8192,
                        }
                    )
                return httpx.Response(200, json={"models": models})
            if path == "/api/generate":
                body = json.loads(request.content)
                if body.get("keep_alive") == 0:
                    self.unloads += 1
                    self.ollama_loaded = False
                    return httpx.Response(200, json={"done": True, "done_reason": "unload"})
                self.loads += 1
                await asyncio.sleep(0.05)
                self.ollama_loaded = True
                return httpx.Response(
                    200,
                    json={"done": True, "load_duration": 48_000_000, "total_duration": 50_000_000},
                )
        if path == "/v1/chat/completions":
            payload = json.loads(request.content)
            self.requests.append(payload)
            if self.fail_chat:
                return httpx.Response(500, json={"error": {"message": "backend crashed"}})
            if payload.get("tools") and not self.tools_enabled:
                return httpx.Response(
                    500, json={"error": {"message": "tools param requires --jinja flag"}}
                )
            if self._has_image(payload) and not self.vision_enabled:
                return httpx.Response(
                    500, json={"error": {"message": "image input is not supported by this server"}}
                )
            return await self._complete(payload)
        return httpx.Response(404, json={"error": {"message": "File Not Found"}})

    @staticmethod
    def _has_image(payload: dict[str, Any]) -> bool:
        return any(isinstance(m.get("content"), list) for m in payload["messages"])

    @staticmethod
    def _text(payload: dict[str, Any]) -> str:
        parts = []
        for m in payload["messages"]:
            content = m.get("content")
            if isinstance(content, list):
                parts += [c.get("text", "") for c in content if c.get("type") == "text"]
            else:
                parts.append(str(content or ""))
        return "\n".join(parts)

    # ------------------------------------------------------------------ Antworten

    async def _complete(self, payload: dict[str, Any]) -> httpx.Response:
        text = self._text(payload)
        max_tokens = int(payload.get("max_tokens") or 64)
        good = self.skill == "good"
        message: dict[str, Any] = {"role": "assistant", "content": ""}
        tokens = 8
        if "Count upward" in text:
            tokens = max_tokens
            message["content"] = ", ".join(["word"] * tokens)
        elif payload.get("tools"):
            message = self._tool_turn(payload, good)
        elif "capital of France" in text:
            message["content"] = (
                "The capital of France is Paris." if good else "Lyon. It is a big city. Maybe."
            )
        elif "Quarterly revenue" in text:
            message["content"] = (
                "Q4 is highest.\nRESULT: QUARTER=Q4; CHANGE=+100%"
                if good
                else "RESULT: QUARTER=Q2; CHANGE=25%"
            )
        elif "Four people" in text:
            message["content"] = (
                "<think>reasoning</think>Step by step...\n"
                "ANSWER: Anna=fish, Ben=dog, Cem=bird, Dana=cat"
                if good
                else "ANSWER: Anna=cat, Ben=dog, Cem=bird, Dana=fish"
            )
        elif "is_palindrome" in text:
            message["content"] = GOOD_PALINDROME if good else BAD_PALINDROME
        elif "def median" in text:
            message["content"] = (
                GOOD_MEDIAN
                if good
                else "```python\n" + text.split("```python\n")[1].split("```")[0] + "```"
            )
        elif "secret codes" in text:
            if good:
                codes = dict(re.findall(r"the (\w+) door is (\d+)", text))
                message["content"] = (
                    f"CODES: BLUE={codes['blue']}; RED={codes['red']}; GREEN={codes['green']}"
                )
            else:
                message["content"] = "CODES: BLUE=1; RED=2; GREEN=3"
        elif "two colored halves" in text:
            message["content"] = "LEFT=red; RIGHT=blue" if good else "LEFT=green; RIGHT=green"
        else:
            tokens = 1
            message["content"] = "OK"
        tokens = min(tokens, max_tokens)
        prompt_tokens = len(text) // 4
        await asyncio.sleep(self.overhead_s + self.per_token_s * tokens)
        body: dict[str, Any] = {
            "id": "chatcmpl-bench",
            "object": "chat.completion",
            "model": payload["model"],
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls" if message.get("tool_calls") else "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": tokens,
                "total_tokens": prompt_tokens + tokens,
            },
        }
        if self.runtime == "llama.cpp":
            body["timings"] = {
                "prompt_n": prompt_tokens,
                "prompt_ms": 1.0,
                "prompt_per_second": 2500.0,
                "predicted_n": tokens,
                "predicted_ms": tokens * self.per_token_s * 1000,
                "predicted_per_second": 1.0 / self.per_token_s,
            }
        return httpx.Response(200, json=body)

    def _tool_turn(self, payload: dict[str, Any], good: bool) -> dict[str, Any]:
        messages = payload["messages"]
        names = {t["function"]["name"] for t in payload["tools"]}
        tool_msgs = [m for m in messages if m["role"] == "tool"]

        def call(name: str, args: dict[str, Any]) -> dict[str, Any]:
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": f"call_{len(tool_msgs)}",
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args)},
                    }
                ],
            }

        if "get_weather" in names:
            if tool_msgs:
                return {
                    "role": "assistant",
                    "content": "It is 18°C and cloudy in Berlin." if good else "It is warm.",
                }
            return call(
                "get_weather",
                {"city": "Berlin", "unit": "celsius"}
                if good
                else {"city": "Munich", "unit": "kelvin"},
            )
        if not good:
            return {"role": "assistant", "content": "I cannot do that."}
        steps = [
            ("list_files", {}),
            ("read_file", {"path": "sales.csv"}),
            ("write_file", {"path": "result.txt", "content": "133.49"}),
        ]
        if len(tool_msgs) < len(steps):
            return call(*steps[len(tool_msgs)])
        return {"role": "assistant", "content": "Done: result.txt contains 133.49."}
