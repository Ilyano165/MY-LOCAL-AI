"""Provider für lokale Runtimes mit OpenAI-kompatibler HTTP-API.

Referenz-Runtime ist llama.cpp ``llama-server`` (``/health``, ``/v1/models``,
``/v1/chat/completions``; Tool-Calling erfordert den Start mit ``--jinja``).
Dieselbe API sprechen u. a. Ollama (``/v1``), vLLM und LM Studio.

Sicherheit: Standardmäßig sind nur Loopback-Adressen erlaubt (Kernsystem ist lokal).
API-Keys werden nur über den *Namen* einer Umgebungsvariable referenziert.
"""

from __future__ import annotations

import ipaddress
import json
import os
import time
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

import httpx

from models.base import (
    ChatRequest,
    ChatResponse,
    ContextLengthExceededError,
    FinishReason,
    InvalidResponseError,
    Message,
    ModelError,
    ModelNotFoundError,
    ModelProvider,
    ModelTimeoutError,
    ProviderHealth,
    ProviderUnavailableError,
    Role,
    TokenUsage,
    ToolCall,
    ToolSpec,
)
from models.capabilities import ModelMetadata

_CONTEXT_ERROR_MARKERS = ("context", "exceed", "too long", "n_ctx")


def _is_loopback(url: str) -> bool:
    host = urlsplit(url).hostname or ""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class OpenAICompatibleProvider(ModelProvider):
    """Spricht eine lokale Runtime über ``/v1/chat/completions`` an."""

    def __init__(
        self,
        name: str,
        base_url: str,
        *,
        api_prefix: str = "/v1",
        health_path: str | None = "/health",
        api_key_env: str | None = None,
        timeout_s: float = 120.0,
        allow_remote: bool = False,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        super().__init__(name)
        scheme = urlsplit(base_url).scheme
        if scheme not in ("http", "https"):
            raise ValueError(f"{name}: base_url muss http(s) sein, erhalten: {base_url!r}")
        if not allow_remote and not _is_loopback(base_url):
            raise ValueError(
                f"{name}: {base_url!r} ist keine Loopback-Adresse. NOVA ist lokal-first; "
                "entfernte Hosts nur mit allow_remote=True."
            )
        headers = {"Accept": "application/json"}
        if api_key_env:
            key = os.environ.get(api_key_env)
            if not key:
                raise ValueError(f"{name}: Umgebungsvariable {api_key_env} ist nicht gesetzt")
            headers["Authorization"] = f"Bearer {key}"
        self._api_prefix = "/" + api_prefix.strip("/") if api_prefix.strip("/") else ""
        self._health_path = health_path
        self._timeout_s = timeout_s
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=httpx.Timeout(timeout_s, connect=min(10.0, timeout_s)),
            transport=transport,
        )

    # ------------------------------------------------------------------ öffentliche API

    async def chat(self, model: ModelMetadata, request: ChatRequest) -> ChatResponse:
        payload = self._build_payload(model, request)
        timeout = request.timeout_s or self._timeout_s
        started = time.perf_counter()
        response = await self._request(
            "POST", f"{self._api_prefix}/chat/completions", json=payload, timeout_s=timeout
        )
        latency_ms = (time.perf_counter() - started) * 1000
        self._raise_for_status(response, model)
        return self._parse_chat(response, model, latency_ms)

    async def list_models(self) -> list[str]:
        response = await self._request("GET", f"{self._api_prefix}/models")
        self._raise_for_status(response, None)
        data = self._json(response)
        items = data.get("data")
        if not isinstance(items, list):
            raise InvalidResponseError(f"{self.name}: /models ohne 'data'-Liste")
        return [str(item["id"]) for item in items if isinstance(item, dict) and "id" in item]

    async def health(self) -> ProviderHealth:
        try:
            if self._health_path is None:
                await self.list_models()
                return ProviderHealth(reachable=True, ready=True, detail="models endpoint ok")
            response = await self._request("GET", self._health_path)
        except ProviderUnavailableError as exc:
            return ProviderHealth(reachable=False, ready=False, detail=str(exc))
        except ModelError as exc:
            return ProviderHealth(reachable=True, ready=False, detail=str(exc))
        if response.status_code == 200:
            return ProviderHealth(reachable=True, ready=True, detail="ok")
        if response.status_code == 503:
            return ProviderHealth(reachable=True, ready=False, detail=self._error_text(response))
        return ProviderHealth(
            reachable=True,
            ready=False,
            detail=f"HTTP {response.status_code}: {self._error_text(response)}",
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------ HTTP-Helfer

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: Mapping[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> httpx.Response:
        try:
            return await self._client.request(
                method,
                path,
                json=json,
                timeout=httpx.Timeout(timeout_s)
                if timeout_s is not None
                else httpx.USE_CLIENT_DEFAULT,
            )
        except httpx.TimeoutException as exc:
            raise ModelTimeoutError(f"{self.name}: Zeitlimit überschritten ({path})") from exc
        except httpx.TransportError as exc:
            raise ProviderUnavailableError(
                f"{self.name}: Runtime nicht erreichbar ({type(exc).__name__}: {exc})"
            ) from exc

    @staticmethod
    def _json(response: httpx.Response) -> dict[str, Any]:
        try:
            data = response.json()
        except ValueError as exc:
            raise InvalidResponseError(f"Antwort ist kein JSON: {response.text[:200]!r}") from exc
        if not isinstance(data, dict):
            raise InvalidResponseError("Antwort ist kein JSON-Objekt")
        return data

    @staticmethod
    def _error_text(response: httpx.Response) -> str:
        try:
            data = response.json()
        except ValueError:
            return response.text[:500]
        if isinstance(data, dict):
            err = data.get("error")
            if isinstance(err, dict):
                return str(err.get("message") or err)
            if err:
                return str(err)
        return str(data)[:500]

    def _raise_for_status(self, response: httpx.Response, model: ModelMetadata | None) -> None:
        status = response.status_code
        if status < 400:
            return
        text = self._error_text(response)
        label = f"{self.name}" + (f"/{model.name}" if model else "")
        lowered = text.lower()
        if status == 404 or (status == 400 and "model" in lowered and "not found" in lowered):
            raise ModelNotFoundError(f"{label}: {text}")
        if status in (400, 413) and any(m in lowered for m in _CONTEXT_ERROR_MARKERS):
            raise ContextLengthExceededError(f"{label}: {text}")
        if status in (502, 503, 504):
            raise ProviderUnavailableError(f"{label}: HTTP {status}: {text}")
        raise ModelError(f"{label}: HTTP {status}: {text}")

    # ------------------------------------------------------------------ Serialisierung

    @staticmethod
    def _serialize_message(message: Message) -> dict[str, Any]:
        out: dict[str, Any] = {"role": message.role.value, "content": message.content}
        if message.tool_calls:
            out["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": json.dumps(dict(call.arguments))},
                }
                for call in message.tool_calls
            ]
        if message.role is Role.TOOL:
            if not message.tool_call_id:
                raise ValueError("Tool-Nachricht ohne tool_call_id")
            out["tool_call_id"] = message.tool_call_id
        return out

    @staticmethod
    def _serialize_tool(tool: ToolSpec) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": dict(tool.parameters),
            },
        }

    def _build_payload(self, model: ModelMetadata, request: ChatRequest) -> dict[str, Any]:
        params = request.params
        payload: dict[str, Any] = {
            "model": model.runtime_name,
            "messages": [self._serialize_message(m) for m in request.messages],
            "temperature": params.temperature,
            "stream": False,
        }
        if params.max_tokens is not None:
            payload["max_tokens"] = params.max_tokens
        if params.top_p is not None:
            payload["top_p"] = params.top_p
        if params.stop:
            payload["stop"] = list(params.stop)
        if params.seed is not None:
            payload["seed"] = params.seed
        if request.tools:
            payload["tools"] = [self._serialize_tool(t) for t in request.tools]
        return payload

    def _parse_chat(
        self, response: httpx.Response, model: ModelMetadata, latency_ms: float
    ) -> ChatResponse:
        data = self._json(response)
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise InvalidResponseError(f"{self.name}/{model.name}: Antwort ohne 'choices'")
        choice = choices[0]
        raw = choice.get("message")
        if not isinstance(raw, dict):
            raise InvalidResponseError(f"{self.name}/{model.name}: choice ohne 'message'")

        tool_calls: list[ToolCall] = []
        for index, item in enumerate(raw.get("tool_calls") or []):
            function = item.get("function") if isinstance(item, dict) else None
            if not isinstance(function, dict) or not function.get("name"):
                raise InvalidResponseError(f"{model.name}: ungültiger tool_call: {item!r}")
            arguments = function.get("arguments") or {}
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments) if arguments.strip() else {}
                except json.JSONDecodeError as exc:
                    raise InvalidResponseError(
                        f"{model.name}: tool_call-Argumente sind kein gültiges JSON: {arguments!r}"
                    ) from exc
            if not isinstance(arguments, dict):
                raise InvalidResponseError(f"{model.name}: tool_call-Argumente sind kein Objekt")
            tool_calls.append(
                ToolCall(
                    id=str(item.get("id") or f"call_{index}"),
                    name=str(function["name"]),
                    arguments=arguments,
                )
            )

        reason_raw = choice.get("finish_reason")
        try:
            finish = FinishReason(str(reason_raw))
        except ValueError:
            finish = FinishReason.OTHER
        if tool_calls and finish is not FinishReason.TOOL_CALLS:
            finish = FinishReason.TOOL_CALLS

        usage_value = data.get("usage")
        usage_raw: dict[str, Any] = usage_value if isinstance(usage_value, dict) else {}
        usage = TokenUsage(
            prompt_tokens=int(usage_raw.get("prompt_tokens") or 0),
            completion_tokens=int(usage_raw.get("completion_tokens") or 0),
        )
        return ChatResponse(
            message=Message.assistant(str(raw.get("content") or ""), tool_calls),
            finish_reason=finish,
            usage=usage,
            model=str(data.get("model") or model.runtime_name),
            latency_ms=latency_ms,
        )
