"""Model Health Check.

Prüft ein registriertes Modell in festgelegter Reihenfolge:

1. ``present``        – Modell vorhanden (Datei unter ``local_path`` und/oder
                         von der Runtime gelistet)
2. ``loadable``       – Runtime bereit und Modell ladbar
3. ``inference``      – einfache Generierung liefert die erwartete Antwort
4. ``response_time``  – Antwort kam innerhalb von ``max_response_s``
5. ``context``        – Needle-Test: Information vom Anfang eines langen Prompts wird wiedergegeben
6. ``tool_calling``   – Modell erzeugt einen korrekten, parsebaren Tool-Call

Jeder Check läuft mit hartem Timeout. Scheitert eine Voraussetzung, werden abhängige
Checks als ``skipped`` markiert (nicht als bestanden).
"""

from __future__ import annotations

import asyncio
import secrets
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from models.base import (
    ChatRequest,
    ChatResponse,
    GenerationParams,
    Message,
    ModelError,
    ToolSpec,
)
from models.capabilities import ModelMetadata
from models.inference import InferenceEngine


class CheckStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    status: CheckStatus
    detail: str = ""
    duration_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class HealthReport:
    model: str
    checks: tuple[CheckResult, ...]

    @property
    def healthy(self) -> bool:
        return all(c.status is not CheckStatus.FAILED for c in self.checks) and any(
            c.status is CheckStatus.PASSED for c in self.checks
        )

    def get(self, name: str) -> CheckResult:
        for check in self.checks:
            if check.name == name:
                return check
        raise KeyError(name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "healthy": self.healthy,
            "checks": [
                {
                    "name": c.name,
                    "status": c.status.value,
                    "detail": c.detail,
                    "duration_ms": round(c.duration_ms, 1),
                }
                for c in self.checks
            ],
        }


@dataclass(frozen=True, slots=True)
class HealthCheckConfig:
    timeout_s: float = 120.0
    """Hartes Zeitlimit pro Check (inkl. Laden)."""
    max_response_s: float = 30.0
    """Maximal akzeptierte Antwortzeit für die einfache Inference."""
    context_fill_ratio: float = 0.5
    """Anteil des Kontextfensters, den der Needle-Test ungefähr füllt."""
    context_max_tokens: int = 8192
    """Obergrenze (geschätzte Tokens) für den Needle-Test, um Health Checks kurz zu halten."""
    max_tokens: int = 64
    checks: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {"present", "loadable", "inference", "response_time", "context", "tool_calling"}
        )
    )

    def __post_init__(self) -> None:
        if self.timeout_s <= 0 or self.max_response_s <= 0:
            raise ValueError("Zeitlimits müssen > 0 sein")
        if not 0 < self.context_fill_ratio <= 1:
            raise ValueError("context_fill_ratio muss in (0, 1] liegen")
        if self.context_max_tokens < 64:
            raise ValueError("context_max_tokens muss >= 64 sein")


# Konservative Schätzung: ein Fülltext-Satz unten hat ~16 Tokens bei gängigen Tokenizern.
_FILLER = "Line {i}: The weather report for this region was unremarkable and calm today.\n"
_TOKENS_PER_FILLER_LINE = 16

_TOOL_NAME = "record_number"
_TOOL_SPEC = ToolSpec(
    name=_TOOL_NAME,
    description="Records a single integer value.",
    parameters={
        "type": "object",
        "properties": {"value": {"type": "integer", "description": "The number to record."}},
        "required": ["value"],
    },
)


def _nonce() -> str:
    return "NOVA-" + secrets.token_hex(3).upper()


class _CheckFailed(Exception):
    pass


class _CheckSkipped(Exception):
    pass


class ModelHealthChecker:
    def __init__(self, engine: InferenceEngine, config: HealthCheckConfig | None = None) -> None:
        self.engine = engine
        self.config = config or HealthCheckConfig()

    async def check_all(self) -> list[HealthReport]:
        return [await self.check(m.name) for m in self.engine.models.list()]

    async def check(self, model_name: str) -> HealthReport:
        model = self.engine.models.get(model_name)
        results: list[CheckResult] = []
        state: dict[str, Any] = {}

        steps: list[tuple[str, Callable[[], Awaitable[str]], tuple[str, ...]]] = [
            ("present", lambda: self._check_present(model), ()),
            ("loadable", lambda: self._check_loadable(model), ("present",)),
            ("inference", lambda: self._check_inference(model, state), ("loadable",)),
            ("response_time", lambda: self._check_response_time(state), ("inference",)),
            ("context", lambda: self._check_context(model), ("inference",)),
            ("tool_calling", lambda: self._check_tool_calling(model), ("inference",)),
        ]
        status_by_name: dict[str, CheckStatus] = {}
        for name, fn, depends_on in steps:
            if name not in self.config.checks:
                continue
            # Deaktivierte Voraussetzungen blockieren nicht; ausgeführte müssen bestanden sein.
            blocked = [
                d
                for d in depends_on
                if d in status_by_name and status_by_name[d] is not CheckStatus.PASSED
            ]
            if blocked:
                result = CheckResult(
                    name, CheckStatus.SKIPPED, f"Voraussetzung nicht erfüllt: {blocked[0]}"
                )
            else:
                result = await self._run(name, fn)
            status_by_name[name] = result.status
            results.append(result)
        return HealthReport(model=model.name, checks=tuple(results))

    async def _run(self, name: str, fn: Callable[[], Awaitable[str]]) -> CheckResult:
        started = time.perf_counter()

        def elapsed() -> float:
            return (time.perf_counter() - started) * 1000

        try:
            detail = await asyncio.wait_for(fn(), timeout=self.config.timeout_s)
            return CheckResult(name, CheckStatus.PASSED, detail, elapsed())
        except _CheckSkipped as exc:
            return CheckResult(name, CheckStatus.SKIPPED, str(exc), elapsed())
        except _CheckFailed as exc:
            return CheckResult(name, CheckStatus.FAILED, str(exc), elapsed())
        except TimeoutError:
            return CheckResult(
                name,
                CheckStatus.FAILED,
                f"Zeitlimit {self.config.timeout_s:.1f}s überschritten",
                elapsed(),
            )
        except ModelError as exc:
            return CheckResult(name, CheckStatus.FAILED, f"{type(exc).__name__}: {exc}", elapsed())

    async def _chat(self, model: ModelMetadata, request: ChatRequest) -> ChatResponse:
        return await self.engine.run(model, request)

    def _params(self, max_tokens: int | None = None) -> GenerationParams:
        return GenerationParams(
            temperature=0.0, max_tokens=max_tokens or self.config.max_tokens, seed=0
        )

    # ------------------------------------------------------------------ Checks

    async def _check_present(self, model: ModelMetadata) -> str:
        evidence: list[str] = []
        if model.local_path is not None:
            path = Path(model.local_path)
            if not await asyncio.to_thread(path.exists):
                raise _CheckFailed(f"Datei fehlt: {path}")
            evidence.append(f"Datei vorhanden ({path})")
        provider = self.engine.providers.get(model.provider)
        try:
            served = await provider.list_models()
        except NotImplementedError:
            served = None
        if served is not None:
            if not self._is_served(model, served):
                raise _CheckFailed(
                    f"Runtime {provider.name!r} listet {model.runtime_name!r} nicht "
                    f"(gelistet: {', '.join(served) or '—'})"
                )
            evidence.append(f"von Runtime {provider.name!r} gelistet")
        return "; ".join(evidence)

    @staticmethod
    def _is_served(model: ModelMetadata, served: list[str]) -> bool:
        wanted = {model.runtime_name}
        if model.local_path is not None:
            wanted |= {str(model.local_path), Path(model.local_path).name}
        # llama-server meldet je nach Start-Option den Alias oder den Dateipfad.
        return any(s in wanted or Path(s).name in wanted for s in served)

    async def _check_loadable(self, model: ModelMetadata) -> str:
        provider = self.engine.providers.get(model.provider)
        health = await provider.health()
        if not health.reachable:
            raise _CheckFailed(f"Runtime nicht erreichbar: {health.detail}")
        if not health.ready:
            raise _CheckFailed(f"Runtime nicht bereit: {health.detail}")
        await provider.ensure_loaded(model)
        return "Runtime bereit, Modell geladen"

    async def _check_inference(self, model: ModelMetadata, state: dict[str, Any]) -> str:
        code = _nonce()
        request = ChatRequest(
            messages=(Message.user(f"Reply with exactly this text and nothing else: {code}"),),
            params=self._params(),
        )
        response = await self._chat(model, request)
        state["inference_latency_ms"] = response.latency_ms
        if code not in response.message.content:
            raise _CheckFailed(f"Erwartet {code!r}, erhalten {response.message.content[:120]!r}")
        tps = ""
        if response.usage.completion_tokens and response.latency_ms > 0:
            tps = f", ~{response.usage.completion_tokens / (response.latency_ms / 1000):.1f} tok/s"
        return f"korrekte Antwort in {response.latency_ms:.0f} ms{tps}"

    async def _check_response_time(self, state: dict[str, Any]) -> str:
        latency_ms = state.get("inference_latency_ms")
        if latency_ms is None:
            raise _CheckSkipped("keine Latenzmessung vorhanden")
        limit_ms = self.config.max_response_s * 1000
        if latency_ms > limit_ms:
            raise _CheckFailed(f"{latency_ms:.0f} ms > Limit {limit_ms:.0f} ms")
        return f"{latency_ms:.0f} ms <= {limit_ms:.0f} ms"

    async def _check_context(self, model: ModelMetadata) -> str:
        target = min(
            int(model.context_length * self.config.context_fill_ratio),
            self.config.context_max_tokens,
        )
        lines = max(1, target // _TOKENS_PER_FILLER_LINE)
        code = _nonce()
        filler = "".join(_FILLER.format(i=i) for i in range(lines))
        prompt = (
            f"Remember this secret code: {code}\n\n{filler}\n"
            "What was the secret code stated at the very beginning? Reply with the code only."
        )
        response = await self._chat(
            model, ChatRequest(messages=(Message.user(prompt),), params=self._params())
        )
        if code not in response.message.content:
            raise _CheckFailed(
                f"Code {code!r} aus ~{target} Tokens Kontext nicht wiedergegeben "
                f"(Antwort: {response.message.content[:120]!r})"
            )
        measured = response.usage.prompt_tokens
        return f"Needle aus ~{target} Tokens gefunden" + (
            f" (Runtime meldet {measured} Prompt-Tokens)" if measured else ""
        )

    async def _check_tool_calling(self, model: ModelMetadata) -> str:
        if not model.tool_calling:
            raise _CheckSkipped("Modell deklariert kein Tool-Calling")
        value = secrets.randbelow(900) + 100
        request = ChatRequest(
            messages=(
                Message.system("You are a function-calling assistant. Use the provided tools."),
                Message.user(f"Call the {_TOOL_NAME} tool with value {value}."),
            ),
            tools=(_TOOL_SPEC,),
            params=self._params(max_tokens=max(self.config.max_tokens, 128)),
        )
        response = await self._chat(model, request)
        calls = [c for c in response.message.tool_calls if c.name == _TOOL_NAME]
        if not calls:
            raise _CheckFailed(
                "kein Tool-Call erzeugt"
                + (
                    f" (Text: {response.message.content[:120]!r})"
                    if response.message.content
                    else ""
                )
            )
        got = calls[0].arguments.get("value")
        try:
            ok = int(str(got)) == value
        except ValueError:
            ok = False
        if not ok:
            raise _CheckFailed(f"falsches Argument: value={got!r}, erwartet {value}")
        return f"Tool-Call {_TOOL_NAME}(value={value}) korrekt"
