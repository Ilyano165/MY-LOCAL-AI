"""Tool-Abstraktion und Registry.

Ein Tool wird nie direkt vom Modell ausgeführt, sondern immer über
:meth:`ToolRegistry.execute`. Diese Methode validiert die Argumente gegen das
JSON-Schema, erzwingt ein Zeitlimit, kürzt Ausgaben und wandelt *jede* Ausnahme in ein
fehlgeschlagenes :class:`ToolOutput` um – ein Tool-Fehler darf den Agenten nicht
abstürzen lassen, er muss als Beobachtung im Task State landen.
"""

from __future__ import annotations

import abc
import asyncio
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from models.base import ToolSpec


class ToolError(Exception):
    """Erwarteter, fachlicher Fehler eines Tools (z. B. Datei nicht gefunden)."""


@dataclass(frozen=True, slots=True)
class ToolContext:
    workspace: Path
    timeout_s: float = 30.0


@dataclass(frozen=True, slots=True)
class ToolOutput:
    ok: bool
    output: str = ""
    data: Mapping[str, Any] = field(default_factory=dict, hash=False)
    error: str | None = None
    duration_ms: float = 0.0
    truncated: bool = False


class Tool(abc.ABC):
    name: ClassVar[str]
    description: ClassVar[str]
    parameters: ClassVar[Mapping[str, Any]]

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, dict(self.parameters))

    @abc.abstractmethod
    async def run(self, args: Mapping[str, Any], ctx: ToolContext) -> ToolOutput:
        """Führt das Tool aus. Fachliche Fehler als :class:`ToolError` werfen."""


_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list, tuple),
    "object": (dict,),
}


def validate_arguments(schema: Mapping[str, Any], args: Mapping[str, Any]) -> list[str]:
    """Prüft die für Tool-Aufrufe relevante Teilmenge von JSON Schema.

    Unterstützt: ``required``, ``properties.*.type``, ``enum``,
    ``additionalProperties: false``.
    """
    if not isinstance(args, Mapping):
        return ["Argumente müssen ein JSON-Objekt sein"]
    problems: list[str] = []
    properties: Mapping[str, Any] = schema.get("properties", {})
    for name in schema.get("required", []):
        if name not in args:
            problems.append(f"Pflichtargument fehlt: {name}")
    if schema.get("additionalProperties") is False:
        problems += [f"Unbekanntes Argument: {k}" for k in args if k not in properties]
    for name, value in args.items():
        prop = properties.get(name)
        if not isinstance(prop, Mapping):
            continue
        expected = prop.get("type")
        if isinstance(expected, str) and expected in _JSON_TYPES:
            allowed = _JSON_TYPES[expected]
            is_bool = isinstance(value, bool)
            if not isinstance(value, allowed) or (is_bool and expected != "boolean"):
                problems.append(f"{name}: erwartet {expected}, erhalten {type(value).__name__}")
                continue
        if "enum" in prop and value not in prop["enum"]:
            problems.append(f"{name}: {value!r} nicht in {prop['enum']}")
    return problems


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] = (), *, max_output_chars: int = 20_000) -> None:
        self._tools: dict[str, Tool] = {}
        self.max_output_chars = max_output_chars
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool {tool.name!r} ist bereits registriert")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            raise KeyError(f"Unbekanntes Tool {name!r}") from None

    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self, names: Iterable[str] | None = None) -> list[ToolSpec]:
        selected = self.names() if names is None else [n for n in names if n in self._tools]
        return [self._tools[n].spec for n in selected]

    async def execute(self, name: str, args: Mapping[str, Any], ctx: ToolContext) -> ToolOutput:
        started = time.perf_counter()

        def failed(error: str) -> ToolOutput:
            return ToolOutput(
                ok=False, error=error, duration_ms=(time.perf_counter() - started) * 1000
            )

        tool = self._tools.get(name)
        if tool is None:
            return failed(f"Unbekanntes Tool {name!r}. Verfügbar: {', '.join(self.names())}")
        problems = validate_arguments(tool.parameters, args)
        if problems:
            return failed("Ungültige Argumente: " + "; ".join(problems))
        try:
            result = await asyncio.wait_for(tool.run(args, ctx), timeout=ctx.timeout_s)
        except TimeoutError:
            return failed(f"Zeitlimit von {ctx.timeout_s:.0f}s überschritten")
        except ToolError as exc:
            return failed(str(exc))
        except Exception as exc:
            return failed(f"Interner Tool-Fehler ({type(exc).__name__}): {exc}")

        output, truncated = result.output, result.truncated
        if len(output) > self.max_output_chars:
            output = output[: self.max_output_chars] + "\n…[gekürzt]"
            truncated = True
        return ToolOutput(
            ok=result.ok,
            output=output,
            data=dict(result.data),
            error=result.error,
            duration_ms=(time.perf_counter() - started) * 1000,
            truncated=truncated,
        )
