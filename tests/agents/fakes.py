"""Test-Doubles für den Agent Core."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from models.base import (
    ChatRequest,
    ChatResponse,
    FinishReason,
    Message,
    ModelError,
    ModelProvider,
    ProviderHealth,
    TokenUsage,
    ToolCall,
)
from models.capabilities import ModelMetadata
from models.inference import InferenceEngine, ProviderRegistry
from models.model_registry import ModelRegistry
from tests.models.fakes import make_meta

# Ein Skript-Eintrag:
#   str                         → Textantwort
#   list[tuple[str, dict]]      → Tool-Calls
#   ModelError                  → wird geworfen
#   Callable[[ChatRequest], …]  → dynamisch (gibt einen der obigen Typen zurück)
ScriptItem = Any

_COMPONENTS = {
    "planning component": "plan",
    "execution component": "execute",
    "error analysis component": "analyze",
    "Write the final answer": "final",
}


def plan_json(*subtasks: dict[str, Any]) -> str:
    return "```json\n" + json.dumps({"subtasks": list(subtasks)}) + "\n```"


class ScriptedProvider(ModelProvider):
    """Antwortet je Komponente (erkannt am System-Prompt) aus einer eigenen Queue."""

    def __init__(self, **scripts: list[ScriptItem]) -> None:
        super().__init__("local")
        self.scripts: dict[str, list[ScriptItem]] = {k: list(v) for k, v in scripts.items()}
        self.requests: dict[str, list[ChatRequest]] = {}
        self._call_ids = 0

    @staticmethod
    def component(request: ChatRequest) -> str:
        system = request.messages[0].content if request.messages else ""
        for marker, name in _COMPONENTS.items():
            if marker in system:
                return name
        return "other"

    async def chat(self, model: ModelMetadata, request: ChatRequest) -> ChatResponse:
        name = self.component(request)
        self.requests.setdefault(name, []).append(request)
        queue = self.scripts.get(name)
        if not queue:
            raise AssertionError(f"Kein Skript-Eintrag mehr für Komponente {name!r}")
        item = queue.pop(0)
        if callable(item) and not isinstance(item, ModelError):
            item = item(request)
        if isinstance(item, ModelError):
            raise item
        if isinstance(item, list):
            calls = []
            for tool, args in item:
                self._call_ids += 1
                # Modelle müssen eine Begründung liefern; Skripte dürfen sie weglassen.
                args = {"reason": f"Test: {tool}", **args}
                if args["reason"] is None:
                    del args["reason"]
                calls.append(ToolCall(f"call_{self._call_ids}", tool, args))
            message = Message.assistant("", calls)
            reason = FinishReason.TOOL_CALLS
        else:
            message = Message.assistant(str(item))
            reason = FinishReason.STOP
        return ChatResponse(message, reason, TokenUsage(10, 5), model.name, 1.0)

    async def list_models(self) -> list[str]:
        return ["test-model"]

    async def health(self) -> ProviderHealth:
        return ProviderHealth(True, True, "ok")

    def remaining(self) -> dict[str, int]:
        return {k: len(v) for k, v in self.scripts.items() if v}


def engine_for(provider: ModelProvider) -> InferenceEngine:
    return InferenceEngine(
        ModelRegistry([make_meta(name="test-model", reasoning_capability="strong")]),
        ProviderRegistry([provider]),
    )


def tool_messages(request: ChatRequest) -> list[str]:
    return [m.content for m in request.messages if m.role.value == "tool"]


ResponseFn = Callable[[ChatRequest], ScriptItem]
