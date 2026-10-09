"""Integrations-API: ``/api/v1/*`` (NOVA) und ``/v1/*`` (OpenAI-kompatibel).

Beide nutzen dieselbe Backend-Funktion :meth:`api.service.NovaService.generate` – dieselbe
InferenceEngine und denselben Router wie die Oberfläche. Unterschiede:

* ``/api/v1``: NOVA-Fehlerformat ``{"error": {"code", "message", "detail", "request_id"}}``,
  Antworten enthalten zusätzlich ``nova`` (Routing, Messwerte – nur wenn vorhanden).
* ``/v1``: OpenAI-Fehlerformat ``{"error": {"message", "type", "param", "code"}}``,
  keine NOVA-Zusatzfelder (für Programme, die nur das OpenAI-Schema erwarten).

Authentifizierung: ``Authorization: Bearer <Integrationsschlüssel>``. Jede Route prüft ihren
Scope. Rate Limit und Größenlimit gelten je Integration.
"""

from __future__ import annotations

import base64
import binascii
import inspect
import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any, Literal

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field, ValidationError

from api.agent_tasks import TaskPermissionError
from api.integrations import SCOPES, TOOL_SCOPE_PREFIX, Integration
from api.service import NO_MODEL, NovaService, ServiceError, _Stopped, _Timeout
from api.version import __version__
from models.base import ChatRequest, GenerationParams, ImageInput, Message, ModelError

AUTO_MODEL = "nova-auto"
Flavor = Literal["nova", "openai"]

_OPENAI_TYPES = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    409: "invalid_request_error",
    413: "invalid_request_error",
    422: "invalid_request_error",
    429: "rate_limit_error",
    500: "server_error",
    502: "server_error",
    503: "service_unavailable_error",
    504: "timeout_error",
}


class ApiFailure(Exception):
    def __init__(
        self,
        status: int,
        code: str,
        message: str,
        detail: Any = None,
        param: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.detail = detail
        self.param = param
        self.headers = headers or {}


def error_response(exc: ApiFailure, flavor: Flavor, request_id: str) -> JSONResponse:
    if flavor == "openai":
        body: dict[str, Any] = {
            "error": {
                "message": exc.message,
                "type": _OPENAI_TYPES.get(exc.status, "api_error"),
                "param": exc.param,
                "code": exc.code,
            }
        }
    else:
        body = {
            "error": {
                "code": exc.code,
                "message": exc.message,
                "detail": exc.detail,
                "request_id": request_id,
            }
        }
    return JSONResponse(
        body, status_code=exc.status, headers={"X-Request-ID": request_id, **exc.headers}
    )


# ---------------------------------------------------------------------- Schemas


class ContentPart(BaseModel):
    type: str
    text: str | None = None
    image_url: dict[str, Any] | None = None


class ChatMessageIn(BaseModel):
    role: str
    content: str | list[ContentPart] | None = None
    name: str | None = None


class StreamOptions(BaseModel):
    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    model: str = Field(min_length=1, max_length=200)
    messages: list[ChatMessageIn] = Field(min_length=1, max_length=500)
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, gt=0, le=1)
    max_tokens: int | None = Field(default=None, ge=1, le=131_072)
    max_completion_tokens: int | None = Field(default=None, ge=1, le=131_072)
    stop: str | list[str] | None = None
    seed: int | None = None
    n: int | None = Field(default=None, ge=1)
    stream: bool = False
    stream_options: StreamOptions | None = None
    user: str | None = None
    tools: list[Any] | None = None
    tool_choice: Any = None
    functions: list[Any] | None = None
    function_call: Any = None
    response_format: dict[str, Any] | None = None
    logprobs: bool | None = None
    nova_timeout_s: float | None = Field(default=None, gt=0, le=3600)
    """NOVA-Erweiterung: Gesamt-Zeitlimit (gedeckelt durch die Server-Einstellung)."""


class AgentTaskRequest(BaseModel):
    objective: str = Field(min_length=1, max_length=20_000)
    allowed_tools: list[str] = Field(default_factory=list, max_length=20)
    timeout_s: float = Field(default=600.0, gt=0, le=7200)


# ---------------------------------------------------------------------- Hilfen


def _unsupported(field_name: str, what: str) -> ApiFailure:
    return ApiFailure(
        400, "unsupported_parameter", f"{what} is not supported by NOVA", param=field_name
    )


def build_chat_request(
    body: ChatCompletionRequest, default_timeout: float
) -> tuple[ChatRequest, bool]:
    """OpenAI-Nachrichten → NOVA-ChatRequest; nicht unterstützte Optionen werden abgelehnt,
    nie still ignoriert (außer ``user``, das nur der Zuordnung dient)."""
    if body.n not in (None, 1):
        raise _unsupported("n", "n > 1")
    if (
        body.tools
        or body.functions
        or body.tool_choice not in (None, "none")
        or body.function_call not in (None, "none")
    ):
        raise _unsupported("tools", "Tool/function calling via the chat API")
    if body.logprobs:
        raise _unsupported("logprobs", "logprobs")
    if body.response_format and body.response_format.get("type") not in (None, "text"):
        raise _unsupported("response_format", f"response_format {body.response_format.get('type')}")
    messages: list[Message] = []
    has_image = False
    for index, m in enumerate(body.messages):
        param = f"messages[{index}]"
        if m.role not in ("system", "developer", "user", "assistant"):
            raise _unsupported(f"{param}.role", f"Role {m.role!r}")
        text_parts: list[str] = []
        images: list[ImageInput] = []
        if isinstance(m.content, str):
            text_parts.append(m.content)
        elif isinstance(m.content, list):
            for part in m.content:
                if part.type == "text":
                    text_parts.append(part.text or "")
                elif part.type == "image_url" and m.role == "user":
                    images.append(_image(part.image_url or {}, param))
                else:
                    raise _unsupported(f"{param}.content", f"Content part {part.type!r}")
        text = "\n".join(text_parts)
        if m.role in ("system", "developer"):
            messages.append(Message.system(text))
        elif m.role == "user":
            messages.append(Message.user(text, images=tuple(images)))
            has_image = has_image or bool(images)
        else:
            messages.append(Message.assistant(text))
    if not any(m.role.value == "user" for m in messages):
        raise ApiFailure(
            400, "invalid_request", "At least one user message is required", param="messages"
        )
    stop = (body.stop,) if isinstance(body.stop, str) else tuple(body.stop or ())
    params = GenerationParams(
        temperature=0.7 if body.temperature is None else body.temperature,
        max_tokens=body.max_completion_tokens or body.max_tokens,
        top_p=body.top_p,
        stop=stop,
        seed=body.seed,
    )
    timeout = min(body.nova_timeout_s or default_timeout, default_timeout)
    return ChatRequest(tuple(messages), params=params, timeout_s=timeout), has_image


def _image(image_url: dict[str, Any], param: str) -> ImageInput:
    url = str(image_url.get("url", ""))
    if not url.startswith("data:image/") or ";base64," not in url:
        raise ApiFailure(
            400,
            "unsupported_parameter",
            "Only base64 data URLs are supported for images (no remote fetching)",
            param=f"{param}.image_url",
        )
    header, data = url.split(",", 1)
    mime = header[5:].split(";")[0]
    try:
        return ImageInput(base64.b64decode(data, validate=True), mime)
    except (binascii.Error, ValueError) as exc:
        raise ApiFailure(400, "invalid_request", "Invalid image data", param=param) from exc


def _service_failure(exc: ServiceError) -> ApiFailure:
    status = exc.status if exc.status >= 400 else 400
    return ApiFailure(status, exc.code, exc.message, exc.detail)


# ---------------------------------------------------------------------- Router


def create_routers(svc: NovaService) -> tuple[APIRouter, APIRouter]:
    nova = APIRouter(prefix="/api/v1")
    openai = APIRouter(prefix="/v1")

    async def authorize(request: Request, scope: str | None, flavor: Flavor) -> Integration:
        header = request.headers.get("authorization", "")
        key = (
            header[7:].strip()
            if header.lower().startswith("bearer ")
            else request.headers.get("x-api-key")
        )
        client = request.client.host if request.client else "?"
        integration = svc.integrations.authenticate(key)
        if integration is None:
            svc.audit.record(
                "auth_failed",
                path=request.url.path,
                client=client,
                key_prefix=(key or "")[:17] or None,
            )
            raise ApiFailure(
                401,
                "invalid_api_key" if key else "missing_api_key",
                "Invalid or missing API key" if key else "Missing API key",
                headers={"WWW-Authenticate": "Bearer"},
            )
        allowed, retry = svc.rate_limiter.allow(integration.id, integration.rate_limit_per_minute)
        if not allowed:
            svc.audit.record("rate_limited", integration=integration.id, path=request.url.path)
            raise ApiFailure(
                429,
                "rate_limited",
                "Rate limit exceeded",
                headers={"Retry-After": str(max(1, int(retry) + 1))},
            )
        if scope is not None and not integration.has(scope):
            svc.audit.record(
                "permission_denied", integration=integration.id, scope=scope, path=request.url.path
            )
            raise ApiFailure(
                403,
                "insufficient_scope",
                f"Missing scope {scope!r}",
                detail={"required_scope": scope},
            )
        return integration

    async def body_of(request: Request, integration: Integration) -> bytes:
        raw = await request.body()
        if len(raw) > integration.max_request_bytes:
            raise ApiFailure(
                413,
                "request_too_large",
                f"Request body exceeds {integration.max_request_bytes} bytes",
            )
        return raw

    def guarded(flavor: Flavor, scope: str | None):  # type: ignore[no-untyped-def]
        """Dekorator: Authentifizierung + einheitliche Fehlerbehandlung je Format."""

        def wrap(handler):  # type: ignore[no-untyped-def]
            async def endpoint(request: Request, **kwargs: Any) -> Response:
                request_id = "req_" + uuid.uuid4().hex[:16]
                try:
                    integration = await authorize(request, scope, flavor)
                    response: Response = await handler(request, integration, request_id, **kwargs)
                except ApiFailure as exc:
                    return error_response(exc, flavor, request_id)
                except ServiceError as exc:
                    return error_response(_service_failure(exc), flavor, request_id)
                response.headers["X-Request-ID"] = request_id
                return response

            endpoint.__name__ = handler.__name__
            endpoint.__doc__ = handler.__doc__
            signature = inspect.signature(handler)
            endpoint.__signature__ = signature.replace(  # type: ignore[attr-defined]
                parameters=[
                    p
                    for name, p in signature.parameters.items()
                    if name not in ("integration", "request_id")
                ],
                return_annotation="Response",
            )
            return endpoint

        return wrap

    # ------------------------------------------------------------------ öffentlich

    @nova.get("/health")
    async def health() -> dict[str, Any]:
        """Ohne Authentifizierung: nur Lebenszeichen und Version (keine internen Details)."""
        return {"status": "ok", "version": __version__, "api_version": "v1"}

    # ------------------------------------------------------------------ Modelle

    def model_ids(integration: Integration) -> list[str]:
        ids = [AUTO_MODEL] + [m.name for m in svc.engine.models.list()]
        return [i for i in ids if integration.model_allowed(i)]

    async def models_nova(request: Request, integration: Integration, request_id: str) -> Response:
        allowed = set(model_ids(integration))
        models = [m for m in svc.models() if m["name"] in allowed]
        return JSONResponse(
            {"auto_model": AUTO_MODEL if AUTO_MODEL in allowed else None, "models": models}
        )

    nova.add_api_route("/models", guarded("nova", "models:read")(models_nova), methods=["GET"])

    async def models_status(
        request: Request, integration: Integration, request_id: str
    ) -> Response:
        status = await svc.models_status(refresh=request.query_params.get("refresh") == "true")
        allowed = set(model_ids(integration))
        status["models"] = [m for m in status["models"] if m["name"] in allowed]
        status.pop("config_error", None)
        return JSONResponse(status)

    nova.add_api_route(
        "/models/status", guarded("nova", "models:read")(models_status), methods=["GET"]
    )

    async def models_openai(
        request: Request, integration: Integration, request_id: str
    ) -> Response:
        created = int(svc.started)
        return JSONResponse(
            {
                "object": "list",
                "data": [
                    {"id": i, "object": "model", "created": created, "owned_by": "nova"}
                    for i in model_ids(integration)
                ],
            }
        )

    openai.add_api_route(
        "/models", guarded("openai", "models:read")(models_openai), methods=["GET"]
    )

    # ------------------------------------------------------------------ Chat

    def chat_handler(flavor: Flavor):  # type: ignore[no-untyped-def]
        async def chat_completions(
            request: Request, integration: Integration, request_id: str
        ) -> Response:
            raw = await body_of(request, integration)
            try:
                body = ChatCompletionRequest.model_validate_json(raw)
            except ValidationError as exc:
                problem = exc.errors()[0]
                raise ApiFailure(
                    400,
                    "invalid_request",
                    problem.get("msg", "Invalid request"),
                    param=".".join(str(p) for p in problem.get("loc", ())),
                ) from exc
            model = body.model
            if not integration.model_allowed(model):
                raise ApiFailure(
                    403,
                    "model_not_allowed",
                    f"Model {model!r} is not allowed for this integration",
                    param="model",
                )
            choice = "auto" if model in (AUTO_MODEL, "auto") else model
            if choice != "auto" and choice not in svc.engine.models:
                raise ApiFailure(
                    404, "model_not_found", f"Model {model!r} does not exist", param="model"
                )
            settings = svc.settings()
            chat_request, needs_vision = build_chat_request(body, settings.request_timeout_s)
            if len(svc.engine.models) == 0 or not (await svc.models_status())["any_available"]:
                raise ApiFailure(503, "no_model", NO_MODEL)
            deadline = chat_request.timeout_s or settings.request_timeout_s
            completion_id = "chatcmpl-" + uuid.uuid4().hex[:24]
            created = int(time.time())
            events = svc.generate(
                chat_request, model=choice, needs_vision=needs_vision, deadline_s=deadline
            )
            try:
                first = await events.__anext__()  # Routing: Fehler noch als HTTP-Status
            except ServiceError as exc:
                raise _service_failure(exc) from exc
            if body.stream:
                include_usage = bool(body.stream_options and body.stream_options.include_usage)
                return StreamingResponse(
                    _stream(events, first, completion_id, created, flavor, include_usage),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                )
            return await _complete(events, first, completion_id, created, flavor)

        return chat_completions

    nova.add_api_route(
        "/chat/completions",
        guarded("nova", "chat:complete")(chat_handler("nova")),
        methods=["POST"],
    )
    openai.add_api_route(
        "/chat/completions",
        guarded("openai", "chat:complete")(chat_handler("openai")),
        methods=["POST"],
    )

    # ------------------------------------------------------------------ Agent

    async def create_task(request: Request, integration: Integration, request_id: str) -> Response:
        raw = await body_of(request, integration)
        try:
            body = AgentTaskRequest.model_validate_json(raw)
        except ValidationError as exc:
            raise ApiFailure(400, "invalid_request", exc.errors()[0].get("msg", "invalid")) from exc
        if len(svc.engine.models) == 0 or not (await svc.models_status())["any_available"]:
            raise ApiFailure(503, "no_model", NO_MODEL)
        try:
            record = svc.agent_tasks.create(
                integration.id,
                body.objective,
                granted_tools=integration.tool_scopes,
                requested_tools=body.allowed_tools,
                workspace=integration.agent_workspace,
                timeout_s=body.timeout_s,
            )
        except TaskPermissionError as exc:
            svc.audit.record(
                "permission_denied", integration=integration.id, scope="agent:tool", detail=str(exc)
            )
            raise ApiFailure(403, "tool_not_allowed", str(exc)) from exc
        svc.audit.record(
            "agent_task_created",
            integration=integration.id,
            task=record.id,
            tools=record.allowed_tools,
        )
        return JSONResponse(
            record.to_dict(),
            status_code=202,
            headers={"Location": f"/api/v1/agent/tasks/{record.id}"},
        )

    nova.add_api_route("/agent/tasks", guarded("nova", "agent:run")(create_task), methods=["POST"])

    async def get_task(
        request: Request, integration: Integration, request_id: str, task_id: str
    ) -> Response:
        record = svc.agent_tasks.get(task_id, integration.id)
        if record is None:
            raise ApiFailure(404, "task_not_found", f"Task {task_id} not found")
        return JSONResponse(record.to_dict())

    nova.add_api_route(
        "/agent/tasks/{task_id}", guarded("nova", "agent:read")(get_task), methods=["GET"]
    )

    async def cancel_task(
        request: Request, integration: Integration, request_id: str, task_id: str
    ) -> Response:
        record = svc.agent_tasks.cancel(task_id, integration.id)
        if record is None:
            raise ApiFailure(404, "task_not_found", f"Task {task_id} not found")
        svc.audit.record(
            "agent_task_cancel_requested",
            integration=integration.id,
            task=task_id,
            status=record.status.value,
        )
        await svc.agent_tasks.wait(task_id)
        return JSONResponse(record.to_dict(), status_code=200)

    nova.add_api_route(
        "/agent/tasks/{task_id}/cancel",
        guarded("nova", "agent:cancel")(cancel_task),
        methods=["POST"],
    )

    # ------------------------------------------------------------------ Fähigkeiten & System

    async def capabilities(request: Request, integration: Integration, request_id: str) -> Response:
        status = await svc.models_status()
        return JSONResponse(
            {
                "api_version": "v1",
                "version": __version__,
                "integration": {
                    "id": integration.id,
                    "name": integration.name,
                    "scopes": integration.scopes,
                    "rate_limit_per_minute": integration.rate_limit_per_minute,
                    "max_request_bytes": integration.max_request_bytes,
                },
                "features": {
                    "chat_completions": integration.has("chat:complete"),
                    "streaming": True,
                    "openai_compatible": True,
                    "agent_tasks": integration.has("agent:run"),
                    "agent_tools": sorted(integration.tool_scopes),
                    "tool_calling_via_chat": False,
                    "embeddings": False,
                    "images_input": "base64 data URLs only",
                },
                "models_available": status["available_count"],
                "auto_model": AUTO_MODEL,
                "known_scopes": [*sorted(SCOPES), f"{TOOL_SCOPE_PREFIX}<tool>"],
            }
        )

    nova.add_api_route("/capabilities", guarded("nova", None)(capabilities), methods=["GET"])

    async def system_status(
        request: Request, integration: Integration, request_id: str
    ) -> Response:
        status = await svc.system_status()
        for private in ("data_dir", "config_path", "config_error"):
            status.pop(private, None)  # keine lokalen Pfade an fremde Programme
        return JSONResponse(status)

    nova.add_api_route(
        "/system/status", guarded("nova", "system:read")(system_status), methods=["GET"]
    )

    return nova, openai


# ---------------------------------------------------------------------- Antwortformen


def _nova_meta(routing: dict[str, Any], stats: dict[str, Any]) -> dict[str, Any]:
    routing = {k: v for k, v in routing.items() if k not in ("decision_id",)}
    return {"routing": routing, "stats": stats}


async def _complete(
    events: AsyncIterator[dict[str, Any]],
    first: dict[str, Any],
    completion_id: str,
    created: int,
    flavor: Flavor,
) -> Response:
    try:
        async for event in events:
            if event["event"] == "complete":
                stats = event["stats"]
                body: dict[str, Any] = {
                    "id": completion_id,
                    "object": "chat.completion",
                    "created": created,
                    "model": stats.get("model") or first["model"],
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": event["content"]},
                            "finish_reason": stats.get("finish_reason", "stop"),
                        }
                    ],
                }
                if "completion_tokens" in stats:  # nur, wenn die Runtime sie gemeldet hat
                    body["usage"] = {
                        "prompt_tokens": stats["prompt_tokens"],
                        "completion_tokens": stats["completion_tokens"],
                        "total_tokens": stats["prompt_tokens"] + stats["completion_tokens"],
                    }
                if flavor == "nova":
                    body["nova"] = _nova_meta(first["routing"], stats)
                return JSONResponse(body)
    except _Timeout as exc:
        raise ApiFailure(504, "timeout", "Generation exceeded the time limit") from exc
    except ModelError as exc:
        raise ApiFailure(502, "model_error", str(exc)) from exc
    except _Stopped as exc:  # pragma: no cover – ohne Stop-Event nicht erreichbar
        raise ApiFailure(499, "cancelled", "Generation cancelled") from exc
    raise ApiFailure(500, "internal_error", "Generation ended without result")


def _chunk(
    completion_id: str, created: int, model: str, delta: dict[str, Any], finish: str | None
) -> bytes:
    payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()


async def _stream(
    events: AsyncIterator[dict[str, Any]],
    first: dict[str, Any],
    completion_id: str,
    created: int,
    flavor: Flavor,
    include_usage: bool,
) -> AsyncIterator[bytes]:
    """OpenAI-Streamingformat; Verbindungsabbruch des Clients bricht die Generierung ab."""
    model = first["model"]
    if flavor == "nova":
        meta = {"object": "nova.routing", "routing": first["routing"]}
        yield f"data: {json.dumps(meta, ensure_ascii=False)}\n\n".encode()
    yield _chunk(completion_id, created, model, {"role": "assistant", "content": ""}, None)
    try:
        async for event in events:
            kind = event["event"]
            if kind == "model":
                model = event["model"]
            elif kind == "token":
                yield _chunk(completion_id, created, model, {"content": event["delta"]}, None)
            elif kind == "complete":
                stats = event["stats"]
                yield _chunk(
                    completion_id,
                    created,
                    stats.get("model", model),
                    {},
                    stats.get("finish_reason", "stop"),
                )
                if include_usage and "completion_tokens" in stats:
                    usage = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": stats.get("model", model),
                        "choices": [],
                        "usage": {
                            "prompt_tokens": stats["prompt_tokens"],
                            "completion_tokens": stats["completion_tokens"],
                            "total_tokens": stats["prompt_tokens"] + stats["completion_tokens"],
                        },
                    }
                    yield f"data: {json.dumps(usage)}\n\n".encode()
                if flavor == "nova":
                    meta = {"object": "nova.stats", "stats": stats}
                    yield f"data: {json.dumps(meta, ensure_ascii=False)}\n\n".encode()
    except (_Timeout, ModelError, ServiceError) as exc:
        code = (
            "timeout"
            if isinstance(exc, _Timeout)
            else (exc.code if isinstance(exc, ServiceError) else "model_error")
        )
        message = "Generation exceeded the time limit" if isinstance(exc, _Timeout) else str(exc)
        error = {"error": {"message": message, "type": "server_error", "param": None, "code": code}}
        yield f"data: {json.dumps(error)}\n\n".encode()
    yield b"data: [DONE]\n\n"
