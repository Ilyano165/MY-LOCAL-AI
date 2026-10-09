"""FastAPI-Anwendung: REST + Server-Sent Events für die NOVA-Oberfläche.

Sicherheit (lokal): Bindung an 127.0.0.1 (siehe ``api.__main__``); schreibende Anfragen
brauchen den Header ``X-NOVA-Client`` (erzwingt bei fremden Webseiten einen CORS-Preflight,
den die API nie erlaubt) und eine passende ``Origin``; optional ein API-Token aus der
Umgebung (``NOVA_API_TOKEN``).
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from api.config import ApiConfig
from api.conversations import NotFoundError
from api.service import AttachmentIn, NovaService, ServiceError
from api.v1 import create_routers
from api.version import __version__

STATIC = Path(__file__).resolve().parent / "static"
_UPLOAD_NAME = re.compile(r"^[0-9a-f]{16}\.(png|jpg|webp|gif)$")


class AttachmentModel(BaseModel):
    name: str = Field(max_length=300)
    mime: str = Field(default="", max_length=200)
    data_b64: str


class ChatBody(BaseModel):
    message: str = Field(default="", max_length=100_000)
    conversation_id: str | None = None
    attachments: list[AttachmentModel] = Field(default_factory=list, max_length=10)
    run_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{6,64}$")


class AgentBody(BaseModel):
    task: str = Field(min_length=1, max_length=20_000)
    conversation_id: str | None = None
    run_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{6,64}$")


class StopBody(BaseModel):
    run_id: str


class TitleBody(BaseModel):
    title: str = Field(min_length=1, max_length=120)


def _attachments(body: ChatBody) -> list[AttachmentIn]:
    return [AttachmentIn(a.name, a.mime, a.data_b64) for a in body.attachments]


def _sse(event: dict[str, Any]) -> bytes:
    name = event.get("event", "message")
    data = json.dumps(event, ensure_ascii=False)
    return f"event: {name}\ndata: {data}\n\n".encode()


async def _stream(agen: AsyncIterator[dict[str, Any]]) -> Response:
    """Erstes Ereignis vorab holen: Validierungsfehler werden zu normalen HTTP-Fehlern."""
    first = await agen.__anext__()

    async def body() -> AsyncIterator[bytes]:
        yield _sse(first)
        async for event in agen:
            yield _sse(event)

    return StreamingResponse(
        body(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def create_app(config: ApiConfig, *, service: NovaService | None = None) -> FastAPI:
    svc = service or NovaService(config)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        await svc.startup()
        yield
        await svc.shutdown()

    app = FastAPI(
        title="NOVA API",
        version=__version__,
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url=None,
        openapi_url="/api/openapi.json",
    )
    app.state.service = svc

    @app.middleware("http")
    async def guard(request: Request, call_next: Any) -> Response:
        token = config.api_token
        path = request.url.path
        client = request.client.host if request.client else ""
        integration_api = path.startswith(("/api/v1/", "/v1/"))
        if not config.allow_remote and client not in config.trusted_clients:
            svc.audit.record("network_denied", client=client, path=path)
            return _error(403, "forbidden", "Remote access is disabled (NOVA listens locally)")
        origin = request.headers.get("origin")
        if integration_api:
            if origin and origin not in config.cors_origins:
                svc.audit.record("cors_denied", origin=origin, path=path)
                return _error(403, "forbidden", "Origin not allowed")
            if request.method == "OPTIONS" and origin:
                return Response(status_code=204, headers=_cors_headers(origin))
            api_response: Response = await call_next(request)
            if origin:
                api_response.headers.update(_cors_headers(origin))
            api_response.headers.setdefault("X-Content-Type-Options", "nosniff")
            return api_response
        is_api = not (path == "/" or path.startswith("/static/") or path == "/favicon.svg")
        if token and is_api and path not in ("/system/status",):
            given = (
                request.headers.get("x-nova-token")
                or request.headers.get("authorization", "").removeprefix("Bearer ").strip()
            )
            if given != token:
                return _error(401, "unauthorized", "API token missing or invalid")
        if request.method in ("POST", "PUT", "PATCH", "DELETE"):
            if request.headers.get("x-nova-client") is None:
                return _error(403, "forbidden", "Missing X-NOVA-Client header")
            origin = request.headers.get("origin")
            if origin and urlsplit(origin).netloc != request.headers.get("host"):
                return _error(403, "forbidden", "Cross-origin request rejected")
        response: Response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        if path == "/" or path.startswith("/static/"):
            response.headers.setdefault(
                "Content-Security-Policy",
                "default-src 'self'; img-src 'self' data: blob:; style-src 'self'; "
                "script-src 'self'; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'",
            )
        return response

    @app.exception_handler(ServiceError)
    async def service_error(_request: Request, exc: ServiceError) -> JSONResponse:
        return _error(exc.status, exc.code, exc.message, exc.detail)

    @app.exception_handler(NotFoundError)
    async def not_found(_request: Request, exc: NotFoundError) -> JSONResponse:
        return _error(404, "not_found", f"Not found: {exc.args[0]}")

    # ------------------------------------------------------------------ UI

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/favicon.svg", include_in_schema=False)
    async def favicon() -> FileResponse:
        return FileResponse(STATIC / "favicon.svg")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    nova_v1, openai_v1 = create_routers(svc)
    app.include_router(nova_v1)
    app.include_router(openai_v1)

    # ------------------------------------------------------------------ Chat & Agent

    @app.post("/chat")
    async def chat(body: ChatBody) -> dict[str, Any]:
        return await svc.chat(body.conversation_id, body.message, _attachments(body))

    @app.post("/chat/stream")
    async def chat_stream(body: ChatBody) -> Response:
        return await _stream(
            svc.chat_stream(body.conversation_id, body.message, _attachments(body), body.run_id)
        )

    @app.post("/agent/run")
    async def agent_run(body: AgentBody) -> Response:
        return await _stream(svc.agent_stream(body.conversation_id, body.task, body.run_id))

    @app.post("/agent/stop")
    async def agent_stop(body: StopBody) -> dict[str, Any]:
        """Stoppt eine laufende Generierung (Chat-Stream oder Agent-Lauf)."""
        return {"run_id": body.run_id, "stopped": svc.stop(body.run_id)}

    # ------------------------------------------------------------------ Status

    @app.get("/models")
    async def models() -> dict[str, Any]:
        return {"models": svc.models()}

    @app.get("/models/status")
    async def models_status(refresh: bool = False) -> dict[str, Any]:
        return await svc.models_status(refresh=refresh)

    @app.get("/router/status")
    async def router_status() -> dict[str, Any]:
        return svc.router_status()

    @app.get("/system/status")
    async def system_status() -> dict[str, Any]:
        return await svc.system_status()

    # ------------------------------------------------------------------ Einstellungen

    @app.get("/settings")
    async def get_settings() -> dict[str, Any]:
        return svc.settings().to_dict()

    @app.put("/settings")
    async def put_settings(request: Request) -> dict[str, Any]:
        changes = await request.json()
        if not isinstance(changes, dict):
            raise ServiceError("invalid_settings", "Expected a JSON object")
        return svc.update_settings(changes).to_dict()

    # ------------------------------------------------------------------ Verlauf

    @app.get("/conversations")
    async def conversations() -> dict[str, Any]:
        return {"conversations": svc.conversations()}

    @app.post("/conversations")
    async def create_conversation() -> dict[str, Any]:
        return svc.store.create()

    @app.get("/conversations/{cid}")
    async def conversation(cid: str) -> dict[str, Any]:
        return svc.conversation(cid)

    @app.patch("/conversations/{cid}")
    async def rename(cid: str, body: TitleBody) -> dict[str, Any]:
        try:
            return svc.store.rename(cid, body.title)
        except ValueError as exc:
            raise ServiceError("invalid_request", str(exc)) from exc

    @app.delete("/conversations/{cid}")
    async def delete(cid: str) -> dict[str, Any]:
        svc.store.delete(cid)
        return {"deleted": cid}

    @app.get("/uploads/{name}")
    async def upload(name: str) -> FileResponse:
        if not _UPLOAD_NAME.match(name) or not (svc.uploads / name).is_file():
            raise NotFoundError(name)
        return FileResponse(svc.uploads / name)

    return app


def _cors_headers(origin: str) -> dict[str, str]:
    return {
        "Access-Control-Allow-Origin": origin,
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Authorization, Content-Type",
        "Access-Control-Max-Age": "600",
        "Vary": "Origin",
    }


def _error(status: int, code: str, message: str, detail: Any = None) -> JSONResponse:
    return JSONResponse(
        {"error": {"code": code, "message": message, "detail": detail}}, status_code=status
    )
