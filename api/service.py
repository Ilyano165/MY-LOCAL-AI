"""NovaService: Fassade zwischen API und Core (Engine, Router, Agent, Verlauf).

Grundsätze:

* Die UI erreicht Modelle nur über diese Schicht; Runtime-Adressen verlassen den Server nie.
* **Keine Fake-Daten:** Kennzahlen (Tokens, tok/s, Routing, Verifikation) erscheinen nur, wenn
  sie tatsächlich vorliegen. Ohne verfügbares Modell gibt es eine klare Fehlermeldung –
  niemals eine simulierte Antwort.
* Abbrechen ist jederzeit möglich (``stop(run_id)`` oder Verbindungsabbruch); bis dahin
  erzeugter Text wird als *abgebrochen* gekennzeichnet gespeichert.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import logging
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator, Awaitable
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from typing import Any

from agents.agent import Agent
from agents.state import JsonFileTaskStore
from agents.task import Phase, Task
from api.config import ApiConfig
from api.conversations import ConversationStore, StoredMessage
from api.settings import Settings, SettingsStore
from evaluation.benchmark_results import ProfileStore
from evaluation.benchmark_tasks import suite_version
from evaluation.hardware import HardwareProfile, detect_hardware
from models.base import (
    ChatRequest,
    GenerationParams,
    ImageInput,
    Message,
    ModelError,
    StreamChunk,
)
from models.capabilities import ModelMetadata
from models.inference import InferenceEngine, ProviderRegistry
from models.model_registry import ModelRegistry
from router.availability import ProviderAvailability
from router.base import NoModelAvailableError, RoutingDecision, RoutingRequest
from router.learned_router import LearnedRanker, LearnedRouter
from router.resources import ResourceBudget, detect_resources
from router.routing_log import JsonlRoutingLog, RoutingLog
from router.rule_router import RuleBasedRouter
from tools import default_tools
from tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

VERSION = "0.1.0"
NO_MODEL = "No local model available."
TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
    ".py",
    ".js",
    ".mjs",
    ".ts",
    ".tsx",
    ".jsx",
    ".json",
    ".toml",
    ".yaml",
    ".yml",
    ".csv",
    ".tsv",
    ".log",
    ".html",
    ".css",
    ".sh",
    ".sql",
    ".rs",
    ".go",
    ".java",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".rb",
    ".php",
    ".xml",
    ".ini",
    ".cfg",
    ".env.example",
}
IMAGE_TYPES = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
}


class ServiceError(Exception):
    """Fehler mit Code für die API (``no_model``, ``invalid_request``, ``not_found`` …)."""

    def __init__(self, code: str, message: str, status: int = 400, detail: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.detail = detail


@dataclass
class AttachmentIn:
    name: str
    mime: str
    data_b64: str


@dataclass
class Run:
    id: str
    kind: str
    conversation_id: str
    started: float = field(default_factory=time.monotonic)
    stop: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task[Any] | None = None


class _RecentLog(RoutingLog):
    """Hält die letzten Entscheidungen für ``/router/status`` und schreibt zusätzlich JSONL."""

    def __init__(self, file_log: JsonlRoutingLog | None, size: int = 50) -> None:
        self.file_log = file_log
        self.entries: deque[dict[str, Any]] = deque(maxlen=size)

    def _write(self, entry: dict[str, Any]) -> None:
        self.entries.append(entry)
        if self.file_log is not None:
            try:
                self.file_log._write(entry)
            except OSError as exc:  # Log darf eine Antwort nie verhindern
                logger.warning("Routing-Log nicht schreibbar: %s", exc)

    def decision(self, decision_id: str) -> dict[str, Any] | None:
        for entry in reversed(self.entries):
            if entry.get("type") == "decision" and entry.get("id") == decision_id:
                return dict(entry)
        return None


async def _until_stopped[T](awaitable: Awaitable[T], stop: asyncio.Event) -> T:
    """Wartet auf ``awaitable`` oder bricht ab, sobald ``stop`` gesetzt wird."""
    task = asyncio.ensure_future(awaitable)
    stopper = asyncio.ensure_future(stop.wait())
    try:
        done, _ = await asyncio.wait({task, stopper}, return_when=asyncio.FIRST_COMPLETED)
    except asyncio.CancelledError:
        task.cancel()
        raise
    finally:
        stopper.cancel()
    if task in done:
        return task.result()
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await task
    raise _Stopped


class _Stopped(Exception):
    pass


class NovaService:
    def __init__(self, config: ApiConfig, *, engine: InferenceEngine | None = None) -> None:
        self.config = config
        self.config_error: str | None = None
        self.engine = engine or self._load_engine()
        self.availability = ProviderAvailability(self.engine.providers, ttl_s=10.0)
        data = config.data_dir
        self.routing_log = _RecentLog(JsonlRoutingLog(data / "routing.jsonl"))
        self.router = self._build_router()
        self.engine.router = self.router
        self.store = ConversationStore(data / "conversations.db")
        self.settings_store = SettingsStore(data / "ui_settings.json")
        self.uploads = data / "uploads"
        self.runs: dict[str, Run] = {}
        self.started = time.time()
        self.hardware: HardwareProfile | None = None
        self._resources: tuple[float, ResourceBudget] | None = None

    # ------------------------------------------------------------------ Aufbau

    def _load_engine(self) -> InferenceEngine:
        path = self.config.models_config
        if path is not None and path.is_file():
            try:
                return InferenceEngine.from_toml(path)
            except (ValueError, OSError) as exc:
                if not self.config.dev_mode:
                    raise
                self.config_error = f"Invalid model configuration: {exc}"
        elif path is not None:
            self.config_error = f"Model configuration not found: {path}"
        else:
            self.config_error = "No model configuration given (development mode)"
        return InferenceEngine(ModelRegistry(), ProviderRegistry())

    def _build_router(self) -> RuleBasedRouter:
        if self.config.router == "learned":
            ranker = None
            if self.config.learned_ranker is not None and self.config.learned_ranker.is_file():
                ranker = LearnedRanker.load(self.config.learned_ranker)
            return LearnedRouter(
                self.engine.models,
                self.availability,
                ranker,
                log=self.routing_log,
                on_ranker_error="rules",
            )
        return RuleBasedRouter(self.engine.models, self.availability, log=self.routing_log)

    async def startup(self) -> None:
        """Hardware erkennen und Benchmark-Profile anbinden (gemessene Werte im Router)."""
        try:
            self.hardware = await detect_hardware()
            self.engine.models.set_measurements(
                ProfileStore(
                    self.config.data_dir / "benchmarks",
                    hardware_fingerprint=self.hardware.fingerprint,
                    benchmark_version=suite_version(),
                )
            )
        except Exception as exc:  # Status-Infos dürfen den Start nie verhindern
            logger.warning("Hardware-Erkennung fehlgeschlagen: %s", exc)

    async def shutdown(self) -> None:
        for run in list(self.runs.values()):
            run.stop.set()
            if run.task is not None:
                run.task.cancel()
        await self.engine.aclose()
        self.store.close()

    # ------------------------------------------------------------------ Einstellungen

    def settings(self) -> Settings:
        return self.settings_store.load()

    def update_settings(self, changes: dict[str, Any]) -> Settings:
        current = self.settings()
        try:
            updated = current.merged(changes)
        except ValueError as exc:
            raise ServiceError("invalid_settings", str(exc)) from exc
        if updated.model != "auto" and updated.model not in self.engine.models:
            raise ServiceError("invalid_settings", f"Unknown model: {updated.model}")
        self.settings_store.save(updated)
        return updated

    # ------------------------------------------------------------------ Status

    def models(self) -> list[dict[str, Any]]:
        out = []
        for m in self.engine.models.list_effective():
            status = self.engine.models.data_status(m.name)
            out.append(
                {
                    "name": m.name,
                    "provider": m.provider,
                    "parameter_count": m.parameter_count,
                    "context_length": m.context_length,
                    "quantization": m.quantization,
                    "reasoning": m.reasoning_capability.name.lower(),
                    "coding": m.coding_capability.name.lower(),
                    "vision": m.vision_capability.name.lower(),
                    "tool_calling": m.tool_calling,
                    "speed": m.speed.name.lower(),
                    "memory_gb": m.memory_requirement,
                    "data_status": status.status.value,
                    "data_source": status.describe(),
                }
            )
        return out

    async def models_status(self, refresh: bool = False) -> dict[str, Any]:
        if refresh:
            self.availability.invalidate()
        models = self.engine.models.list()
        availability = await self.availability.check_all(models)
        providers = []
        for name in self.engine.providers.names():
            provider = self.engine.providers.get(name)
            try:
                health = await asyncio.wait_for(provider.health(), timeout=5)
                providers.append(
                    {
                        "name": name,
                        "reachable": health.reachable,
                        "ready": health.ready,
                        "detail": health.detail,
                    }
                )
            except (ModelError, TimeoutError) as exc:
                providers.append(
                    {
                        "name": name,
                        "reachable": False,
                        "ready": False,
                        "detail": str(exc) or "timeout",
                    }
                )
        items = [
            {
                "name": m.name,
                "provider": m.provider,
                "available": availability[m.name].available,
                "reason": availability[m.name].reason,
            }
            for m in models
        ]
        available = [i["name"] for i in items if i["available"]]
        return {
            "models": items,
            "providers": providers,
            "available_count": len(available),
            "any_available": bool(available),
            "message": None if available else NO_MODEL,
            "config_error": self.config_error,
        }

    def router_status(self) -> dict[str, Any]:
        ranker = getattr(self.router, "ranker", None)
        decisions = [e for e in self.routing_log.entries if e.get("type") == "decision"]
        failures = [e for e in self.routing_log.entries if e.get("type") == "failure"]
        return {
            "router": type(self.router).__name__,
            "mode": self.config.router,
            "classifier": type(self.router.classifier).__name__,
            "learned_ranker": None
            if not isinstance(self.router, LearnedRouter)
            else {
                "loaded": ranker is not None,
                "path": str(self.config.learned_ranker) if self.config.learned_ranker else None,
                "trained_on": ranker.model.trained_on if ranker is not None else None,
            },
            "log_path": str(self.config.data_dir / "routing.jsonl"),
            "recent_decisions": [
                {
                    k: e.get(k)
                    for k in (
                        "id",
                        "timestamp",
                        "category",
                        "complexity",
                        "selected_model",
                        "reason",
                        "confidence",
                        "ranker",
                        "data_status",
                    )
                }
                for e in reversed(decisions[-20:])
            ],
            "recent_failures": [
                {"timestamp": e.get("timestamp"), "error": e.get("error")}
                for e in reversed(failures[-10:])
            ],
        }

    async def system_status(self) -> dict[str, Any]:
        now = time.monotonic()
        if self._resources is None or now - self._resources[0] > 15:
            self._resources = (now, await detect_resources())
        resources = self._resources[1]
        models = await self.models_status()
        return {
            "version": VERSION,
            "dev_mode": self.config.dev_mode,
            "config_path": str(self.config.models_config) if self.config.models_config else None,
            "config_error": self.config_error,
            "data_dir": str(self.config.data_dir),
            "uptime_s": round(time.time() - self.started, 1),
            "hardware": self.hardware.to_dict() if self.hardware else None,
            "hardware_summary": self.hardware.summary() if self.hardware else None,
            "resources": {
                "vram_free_gb": resources.vram_gb,
                "ram_free_gb": resources.ram_gb,
                "source": resources.source,
            },
            "models_configured": len(self.engine.models),
            "models_available": models["available_count"],
            "model_message": models["message"],
            "router": type(self.router).__name__,
            "active_runs": [
                {"id": r.id, "kind": r.kind, "running_s": round(time.monotonic() - r.started, 1)}
                for r in self.runs.values()
            ],
            "auth_required": bool(self.config.api_token),
        }

    # ------------------------------------------------------------------ Gespräche

    def conversations(self) -> list[dict[str, Any]]:
        return self.store.list_all()

    def conversation(self, cid: str) -> dict[str, Any]:
        info = self.store.get(cid)
        info["messages"] = [m.to_dict() for m in self.store.messages(cid)]
        return info

    def stop(self, run_id: str) -> bool:
        run = self.runs.get(run_id)
        if run is None:
            return False
        run.stop.set()
        if run.task is not None:
            run.task.cancel()
        return True

    # ------------------------------------------------------------------ Anhänge

    def _attachments(
        self, items: list[AttachmentIn]
    ) -> tuple[list[dict[str, Any]], list[ImageInput]]:
        stored: list[dict[str, Any]] = []
        images: list[ImageInput] = []
        for item in items:
            name = PurePath(item.name).name[:200] or "datei"
            try:
                raw = base64.b64decode(item.data_b64, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ServiceError("invalid_attachment", f"{name}: invalid encoding") from exc
            if len(raw) > self.config.max_attachment_bytes:
                limit = self.config.max_attachment_bytes // (1024 * 1024)
                raise ServiceError("invalid_attachment", f"{name}: larger than {limit} MB")
            suffix = PurePath(name).suffix.lower()
            if item.mime in IMAGE_TYPES:
                image_id = uuid.uuid4().hex[:16]
                self.uploads.mkdir(parents=True, exist_ok=True)
                path = self.uploads / f"{image_id}{IMAGE_TYPES[item.mime]}"
                path.write_bytes(raw)
                images.append(ImageInput(raw, item.mime))
                stored.append(
                    {
                        "name": name,
                        "mime": item.mime,
                        "size": len(raw),
                        "kind": "image",
                        "file": path.name,
                    }
                )
            elif (
                item.mime.startswith("text/")
                or suffix in TEXT_EXTENSIONS
                or item.mime in ("application/json", "application/toml", "application/xml")
            ):
                try:
                    text = raw.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise ServiceError("invalid_attachment", f"{name}: not UTF-8 text") from exc
                if len(text) > self.config.max_text_attachment_chars:
                    raise ServiceError(
                        "invalid_attachment", f"{name}: text too long for the context"
                    )
                stored.append(
                    {
                        "name": name,
                        "mime": item.mime or "text/plain",
                        "size": len(raw),
                        "kind": "text",
                        "text": text,
                    }
                )
            else:
                raise ServiceError(
                    "unsupported_attachment",
                    f"{name}: file type {item.mime or suffix or 'unknown'} is not supported yet "
                    "(text and image files are)",
                )
        return stored, images

    @staticmethod
    def _with_attachments(text: str, attachments: list[dict[str, Any]]) -> str:
        parts = [text]
        for a in attachments:
            if a.get("kind") == "text":
                parts.append(f"\n\n[File: {a['name']}]\n```\n{a['text']}\n```")
            elif a.get("kind") == "image":
                parts.append(f"\n\n[Image attached: {a['name']}]")
        return "".join(parts)

    # ------------------------------------------------------------------ Chat

    def _history(self, cid: str, settings: Settings, exclude: str) -> list[Message]:
        messages = [
            m
            for m in self.store.messages(cid)
            if m.id != exclude
            and m.role in ("user", "assistant")
            and m.content
            and not m.meta.get("error")
        ]
        if settings.history_messages == 0:
            return []
        out = []
        for m in messages[-settings.history_messages :]:
            if m.role == "user":
                out.append(Message.user(self._with_attachments(m.content, m.attachments)))
            else:
                out.append(Message.assistant(m.content))
        return out

    async def _select(
        self, request: ChatRequest, settings: Settings, needs_vision: bool
    ) -> tuple[list[ModelMetadata], dict[str, Any], RoutingDecision | None]:
        if len(self.engine.models) == 0:
            raise ServiceError(
                "no_model", NO_MODEL, 503, self.config_error or "No models configured"
            )
        if settings.model != "auto":
            if settings.model not in self.engine.models:
                raise ServiceError("invalid_settings", f"Unknown model: {settings.model}")
            model = self.engine.models.effective(settings.model)
            availability = await self.availability.check(model)
            if not availability.available:
                raise ServiceError(
                    "model_unavailable",
                    f"{model.name} is not available: {availability.reason}",
                    503,
                )
            return [model], {"mode": "manual", "model": model.name}, None
        routing_request = RoutingRequest.from_chat(request, needs_vision=needs_vision)
        try:
            decision = await self.router.route(routing_request)
        except NoModelAvailableError as exc:
            models = await self.models_status()
            if not models["any_available"]:
                raise ServiceError("no_model", NO_MODEL, 503, str(exc)) from exc
            raise ServiceError("no_suitable_model", str(exc), 503, exc.rejected) from exc
        c = decision.classification
        routing = {
            "mode": "auto",
            "decision_id": decision.id,
            "model": decision.model.name,
            "category": c.category.value,
            "secondary": c.secondary.value if c.secondary else None,
            "complexity": c.complexity.name,
            "confidence": round(c.confidence, 2),
            "reason": decision.reason,
            "ranker": decision.ranker,
            "fallbacks": [m.name for m in decision.fallbacks],
            "data_status": decision.data_status.get(decision.model.name),
        }
        return [decision.model, *decision.fallbacks], routing, decision

    async def _require_model(self) -> None:
        """Ohne verfügbares Modell keine Anfrage – und kein Eintrag im Verlauf."""
        if len(self.engine.models) == 0:
            raise ServiceError(
                "no_model", NO_MODEL, 503, self.config_error or "No models configured"
            )
        status = await self.models_status()
        if not status["any_available"]:
            reasons = "; ".join(f"{m['name']}: {m['reason']}" for m in status["models"])
            raise ServiceError("no_model", NO_MODEL, 503, reasons or self.config_error)

    def _prepare(
        self, conversation_id: str | None, text: str, attachments: list[AttachmentIn]
    ) -> tuple[str, StoredMessage, list[ImageInput], bool]:
        text = text.strip()
        if not text and not attachments:
            raise ServiceError("invalid_request", "Empty message")
        if len(text) > 100_000:
            raise ServiceError("invalid_request", "Message too long (max. 100000 characters)")
        stored_attachments, images = self._attachments(attachments)
        created = False
        if conversation_id is None:
            title = (text or stored_attachments[0]["name"]).splitlines()[0][:60]
            conversation_id = self.store.create(title or "Neuer Chat")["id"]
            created = True
        else:
            self.store.get(conversation_id)
        user = self.store.add_message(conversation_id, "user", text, attachments=stored_attachments)
        return conversation_id, user, images, created

    async def chat_stream(
        self,
        conversation_id: str | None,
        text: str,
        attachments: list[AttachmentIn] | None = None,
        run_id: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Ereignisse: run → routing → status → token* → done | stopped | error."""
        settings = self.settings()
        run_id = run_id or uuid.uuid4().hex[:16]
        if run_id in self.runs:
            raise ServiceError("invalid_request", "run_id is already in use")
        await self._require_model()
        cid, user, images, created = self._prepare(conversation_id, text, attachments or [])
        run = Run(run_id, "chat", cid)
        self.runs[run_id] = run
        yield {
            "event": "run",
            "run_id": run_id,
            "conversation_id": cid,
            "conversation_created": created,
            "user_message": user.to_dict(),
        }
        content: list[str] = []
        meta: dict[str, Any] = {}
        finished = False
        started = time.perf_counter()
        try:
            messages: list[Message] = []
            if settings.system_prompt.strip():
                messages.append(Message.system(settings.system_prompt.strip()))
            messages += self._history(cid, settings, exclude=user.id)
            messages.append(
                Message.user(
                    self._with_attachments(user.content, user.attachments), images=tuple(images)
                )
            )
            request = ChatRequest(
                tuple(messages),
                params=GenerationParams(
                    temperature=settings.temperature, max_tokens=settings.max_tokens
                ),
                timeout_s=settings.request_timeout_s,
            )
            candidates, routing, _decision = await self._select(request, settings, bool(images))
            meta["routing"] = routing
            yield {
                "event": "routing",
                "routing": routing,
                "model": candidates[0].name,
                "provider": candidates[0].provider,
            }
            yield {"event": "status", "state": "generating"}
            first_at: float | None = None
            last: StreamChunk | None = None
            used: ModelMetadata | None = None
            stream = self.engine.stream(candidates, request).__aiter__()
            while True:
                try:
                    model, chunk = await _until_stopped(stream.__anext__(), run.stop)
                except StopAsyncIteration:
                    break
                if used is None or model.name != used.name:
                    if used is not None or model.name != candidates[0].name:
                        yield {"event": "model", "model": model.name, "fallback": True}
                    used = model
                if chunk.delta:
                    if first_at is None:
                        first_at = time.perf_counter()
                    content.append(chunk.delta)
                    yield {"event": "token", "delta": chunk.delta}
                last = chunk
            ended = time.perf_counter()
            meta.update(self._stats(used, last, started, first_at, ended))
            finished = True
            # Kein record_outcome: Ein Chat ohne Verifikation hat kein Erfolgsurteil – es
            # wird keines erfunden (Trainingsdaten entstehen aus verifizierten Agent-Läufen).
            message = self.store.add_message(cid, "assistant", "".join(content), meta=meta)
            yield {"event": "done", "message": message.to_dict()}
        except _Stopped:
            finished = True
            message = self._store_partial(cid, content, meta, started, "stopped")
            yield {"event": "stopped", "message": message.to_dict()}
        except ServiceError as exc:
            finished = True
            message = self.store.add_message(
                cid,
                "assistant",
                "",
                meta={
                    **meta,
                    "error": {
                        "code": exc.code,
                        "message": exc.message,
                        "detail": _text(exc.detail),
                    },
                },
            )
            yield {
                "event": "error",
                "code": exc.code,
                "message": exc.message,
                "detail": _text(exc.detail),
                "message_id": message.id,
            }
        except ModelError as exc:
            finished = True
            partial = "".join(content)
            message = self.store.add_message(
                cid,
                "assistant",
                partial,
                meta={**meta, "error": {"code": "model_error", "message": str(exc)}},
            )
            yield {
                "event": "error",
                "code": "model_error",
                "message": str(exc),
                "message_id": message.id,
            }
        finally:
            self.runs.pop(run_id, None)
            if not finished:  # Verbindung abgebrochen (Client hat gestoppt)
                self._store_partial(cid, content, meta, started, "disconnected")

    def _store_partial(
        self, cid: str, content: list[str], meta: dict[str, Any], started: float, reason: str
    ) -> StoredMessage:
        meta = {
            **meta,
            "stopped": True,
            "stop_reason": reason,
            "generation_time_s": round(time.perf_counter() - started, 3),
        }
        return self.store.add_message(cid, "assistant", "".join(content), meta=meta)

    @staticmethod
    def _stats(
        model: ModelMetadata | None,
        last: StreamChunk | None,
        started: float,
        first_at: float | None,
        ended: float,
    ) -> dict[str, Any]:
        """Nur Werte, die tatsächlich gemessen bzw. von der Runtime gemeldet wurden."""
        stats: dict[str, Any] = {"generation_time_s": round(ended - started, 3)}
        if model is not None:
            stats["model"] = model.name
            stats["provider"] = model.provider
        if last is not None:
            stats["streamed"] = last.streamed
            if last.model:
                stats["runtime_model"] = last.model
            if last.finish_reason is not None:
                stats["finish_reason"] = last.finish_reason.value
        if first_at is not None and last is not None and last.streamed:
            stats["time_to_first_token_s"] = round(first_at - started, 3)
        usage = last.usage if last is not None else None
        if usage is not None and (usage.prompt_tokens or usage.completion_tokens):
            stats["prompt_tokens"] = usage.prompt_tokens
            stats["completion_tokens"] = usage.completion_tokens
            decode = (
                ended - first_at
                if first_at is not None and last is not None and last.streamed
                else None
            )
            if decode and decode > 0 and usage.completion_tokens > 1:
                stats["tokens_per_second"] = round((usage.completion_tokens - 1) / decode, 2)
                stats["tokens_per_second_source"] = "measured"
        if last is not None and "predicted_per_second" in last.runtime_stats:
            stats["runtime_tokens_per_second"] = round(
                last.runtime_stats["predicted_per_second"], 2
            )
        return stats

    async def chat(
        self, conversation_id: str | None, text: str, attachments: list[AttachmentIn] | None = None
    ) -> dict[str, Any]:
        """Nicht gestreamte Variante (gleiche Logik, gesammeltes Ergebnis)."""
        result: dict[str, Any] = {}
        async for event in self.chat_stream(conversation_id, text, attachments):
            kind = event["event"]
            if kind == "run":
                result["conversation_id"] = event["conversation_id"]
                result["user_message"] = event["user_message"]
            elif kind in ("done", "stopped"):
                result["message"] = event["message"]
            elif kind == "error":
                raise ServiceError(
                    event["code"],
                    event["message"],
                    503
                    if event["code"] in ("no_model", "model_unavailable", "no_suitable_model")
                    else 502,
                    event.get("detail"),
                )
        return result

    # ------------------------------------------------------------------ Agent

    async def agent_stream(
        self, conversation_id: str | None, text: str, run_id: str | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        """Agent-Lauf mit Phasen-Ereignissen; Ergebnis inklusive Verifikationsurteil."""
        settings = self.settings()
        await self._require_model()  # fehlendes Modell ist der grundlegendere Fehler
        if not settings.agent_workspace:
            raise ServiceError(
                "agent_disabled",
                "Agent mode is disabled: set a workspace directory in Settings.",
            )
        run_id = run_id or uuid.uuid4().hex[:16]
        if run_id in self.runs:
            raise ServiceError("invalid_request", "run_id is already in use")
        cid, user, _images, created = self._prepare(conversation_id, text, [])
        run = Run(run_id, "agent", cid)
        self.runs[run_id] = run
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        def on_event(task: Task, phase: Phase, message: str) -> None:
            queue.put_nowait({"event": "phase", "phase": phase.value, "message": message[:500]})

        workspace = await asyncio.to_thread(Path(settings.agent_workspace).expanduser)
        agent = Agent(
            self.engine,
            ToolRegistry(default_tools()),
            JsonFileTaskStore(self.config.data_dir / "tasks"),
            workspace,
            on_event=on_event,
        )
        started = time.perf_counter()
        run.task = asyncio.create_task(agent.run(user.content))
        yield {
            "event": "run",
            "run_id": run_id,
            "conversation_id": cid,
            "conversation_created": created,
            "user_message": user.to_dict(),
        }
        yield {"event": "status", "state": "running"}
        finished = False
        try:
            while not run.task.done():
                getter = asyncio.ensure_future(queue.get())
                done, _ = await asyncio.wait(
                    {getter, run.task}, return_when=asyncio.FIRST_COMPLETED
                )
                if getter in done:
                    yield getter.result()
                else:
                    getter.cancel()
            while not queue.empty():
                yield queue.get_nowait()
            task = run.task.result()
            meta = self._agent_meta(task, started)
            finished = True
            answer = task.final_result.answer if task.final_result else ""
            message = self.store.add_message(cid, "assistant", answer, meta=meta)
            yield {"event": "done", "message": message.to_dict()}
        except asyncio.CancelledError:
            finished = True
            message = self._store_partial(cid, [], {"agent": True}, started, "stopped")
            yield {"event": "stopped", "message": message.to_dict()}
        except ModelError as exc:
            finished = True
            message = self.store.add_message(
                cid,
                "assistant",
                "",
                meta={"agent": True, "error": {"code": "model_error", "message": str(exc)}},
            )
            yield {
                "event": "error",
                "code": "model_error",
                "message": str(exc),
                "message_id": message.id,
            }
        finally:
            self.runs.pop(run_id, None)
            if not finished:
                run.task.cancel()
                self._store_partial(cid, [], {"agent": True}, started, "disconnected")

    def _agent_meta(self, task: Task, started: float) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "agent": True,
            "generation_time_s": round(time.perf_counter() - started, 3),
        }
        if task.final_result is not None:
            r = task.final_result
            meta["verification"] = {
                "status": r.status.value,
                "verified": r.verified,
                "summary": r.summary,
            }
            if r.quality:
                meta["verification"]["quality"] = r.quality.get("overall")
        decisions = [d for d in (self.routing_log.decision(i) for i in task.routing_ids) if d]
        if decisions:
            meta["model"] = decisions[-1].get("selected_model")
            meta["models_used"] = sorted({str(d.get("selected_model")) for d in decisions})
            meta["routing"] = {
                "mode": "auto",
                "category": decisions[-1].get("category"),
                "complexity": decisions[-1].get("complexity"),
                "decisions": len(decisions),
            }
        return meta


def _text(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool | list | dict):
        return value
    return str(value)


__all__ = ["NO_MODEL", "AttachmentIn", "NovaService", "ServiceError"]
