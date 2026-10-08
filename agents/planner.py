"""ANALYZE und PLAN.

* :class:`HeuristicTaskAnalyzer` – deterministische Klassifikation der Aufgabe
  (Aufgabentyp, Komplexität, Tool-Bedarf). Kein Modellaufruf → schnell, testbar,
  nachvollziehbar. Ein modellbasierter Analyzer kann später dieselbe Schnittstelle erfüllen.
* :class:`LLMPlanner` – zerlegt die Aufgabe per Modell in Teilaufgaben mit prüfbaren
  Verifikationen. Ausgabe ist strikt validiertes JSON; ungültige Pläne werden mit
  Fehlerrückmeldung erneut angefordert.
"""

from __future__ import annotations

import abc
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from agents.task import Complexity, Subtask, SubtaskStatus, Task, TaskAnalysis, VerificationSpec
from models.base import ChatRequest, GenerationParams, Message, ModelError, ToolSpec
from models.capabilities import TaskRequirements, TaskType
from models.inference import InferenceEngine


class PlanningError(Exception):
    """Es konnte kein gültiger Plan erzeugt werden."""


# --------------------------------------------------------------------------- ANALYZE


class TaskAnalyzer(abc.ABC):
    @abc.abstractmethod
    async def analyze(self, task: Task) -> TaskAnalysis: ...


def _rx(*words: str) -> re.Pattern[str]:
    return re.compile(r"(" + "|".join(words) + r")", re.IGNORECASE)


_VISION = _rx(r"\.(png|jpe?g|gif|webp|bmp)\b", r"\bbild", r"\bimage", r"screenshot", r"\bfoto")
_CODING = _rx(
    r"\bcode\b",
    r"\bfunktion",
    r"\bfunction",
    r"\bklasse\b",
    r"\bclass\b",
    r"\bbug",
    r"\btests?\b",
    r"\bpython\b",
    r"implement",
    r"refactor",
    r"\bskript",
    r"\bscript",
    r"compil",
    r"syntax",
    r"```",
    r"\.(py|js|ts|tsx|rs|go|java|c|cpp|h|rb|sh|toml|json|ya?ml)\b",
    r"\bapi\b",
    r"programm",
)
_REASONING = _rx(
    r"analys",
    r"\bplan",
    r"beweis",
    r"\bprove",
    r"berechn",
    r"calculat",
    r"\bwarum\b",
    r"\bwhy\b",
    r"vergleich",
    r"compare",
    r"abwäg",
    r"trade-?off",
    r"strateg",
    r"evaluier",
)
_TOOLS = _rx(
    r"\bdatei",
    r"\bfile",
    r"\bordner",
    r"\bdirector",
    r"\bschreib",
    r"\bwrite",
    r"\blies\b",
    r"\bread\b",
    r"\berstell",
    r"\bcreate",
    r"ändere",
    r"\bmodify",
    r"\bspeicher",
    r"\bsave",
    r"ausführ",
    r"\brun\b",
    r"\.(py|md|txt|json|toml|ya?ml|csv)\b",
)
_MULTI = _rx(r"\bund dann\b", r"\bthen\b", r"\bdanach\b", r"\banschließend", r"\bafterwards")
_LIST_LINE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+", re.MULTILINE)


class HeuristicTaskAnalyzer(TaskAnalyzer):
    def __init__(self, *, long_task_chars: int = 400, short_task_chars: int = 80) -> None:
        self.long_task_chars = long_task_chars
        self.short_task_chars = short_task_chars

    async def analyze(self, task: Task) -> TaskAnalysis:
        text = task.objective
        signals: list[str] = []

        def hit(pattern: re.Pattern[str], label: str) -> bool:
            match = pattern.search(text)
            if match:
                signals.append(f"{label}: {match.group(0)!r}")
            return match is not None

        vision, coding = hit(_VISION, "vision"), hit(_CODING, "coding")
        reasoning, tools = hit(_REASONING, "reasoning"), hit(_TOOLS, "tools")

        if vision:
            task_type = TaskType.VISION
        elif coding:
            task_type = TaskType.CODING
        elif reasoning:
            task_type = TaskType.REASONING
        elif len(text) <= self.short_task_chars and not tools:
            task_type = TaskType.FAST
        else:
            task_type = TaskType.GENERAL

        list_items = len(_LIST_LINE.findall(text))
        multi = (
            (coding and tools)
            or list_items >= 2
            or hit(_MULTI, "sequence")
            or len(text) > self.long_task_chars
        )
        if list_items >= 2:
            signals.append(f"list_items: {list_items}")
        return TaskAnalysis(
            task_type=task_type,
            complexity=Complexity.MULTI_STEP if multi else Complexity.SIMPLE,
            requires_tools=tools or coding,
            success_criteria=list(task.constraints.notes),
            signals=signals,
        )


# --------------------------------------------------------------------------- PLAN


class Planner(abc.ABC):
    @abc.abstractmethod
    async def plan(self, task: Task) -> list[Subtask]: ...

    @abc.abstractmethod
    async def replan(self, task: Task, failed: Subtask) -> list[Subtask]:
        """Neue Teilaufgaben, die die gescheiterte und alle folgenden ersetzen."""


def _balanced_end(text: str, start: int) -> int | None:
    """Index der zu ``text[start] == "{"`` passenden schließenden Klammer (string-bewusst)."""
    depth, in_str, escape = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
    return None


def _try_object(candidate: str) -> dict[str, Any] | None:
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def extract_json_object(text: str) -> dict[str, Any]:
    """Findet das erste gültige JSON-Objekt in Modelltext (bevorzugt aus ```json-Blöcken)."""
    for fenced in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL):
        if (value := _try_object(fenced.group(1))) is not None:
            return value
    start = text.find("{")
    while start != -1:
        end = _balanced_end(text, start)
        if end is not None and (value := _try_object(text[start : end + 1])) is not None:
            return value
        start = text.find("{", start + 1)
    raise ValueError("Keine gültige JSON-Objekt-Ausgabe gefunden")


def validate_plan(
    data: Mapping[str, Any],
    *,
    known_checks: Iterable[str],
    max_subtasks: int,
    id_prefix: str = "",
    existing_ids: Iterable[str] = (),
) -> list[Subtask]:
    raw = data.get("subtasks")
    if not isinstance(raw, list) or not raw:
        raise ValueError("'subtasks' muss eine nicht-leere Liste sein")
    if len(raw) > max_subtasks:
        raise ValueError(f"Zu viele Teilaufgaben ({len(raw)} > {max_subtasks})")
    checks = set(known_checks)
    seen: set[str] = set(existing_ids)
    result: list[Subtask] = []
    for index, item in enumerate(raw, 1):
        if not isinstance(item, dict):
            raise ValueError(f"Teilaufgabe {index} ist kein Objekt")
        description = str(item.get("description", "")).strip()
        if not description:
            raise ValueError(f"Teilaufgabe {index} ohne 'description'")
        sid = id_prefix + str(item.get("id") or f"s{index}")
        if sid in seen:
            raise ValueError(f"Doppelte Teilaufgaben-ID {sid!r}")
        deps = [
            id_prefix + str(d) if id_prefix + str(d) in seen else str(d)
            for d in item.get("depends_on", []) or []
        ]
        unknown = [d for d in deps if d not in seen]
        if unknown:
            raise ValueError(f"{sid}: depends_on verweist auf unbekannte/spätere IDs {unknown}")
        specs: list[VerificationSpec] = []
        for spec_raw in item.get("verification", []) or []:
            if not isinstance(spec_raw, dict):
                raise ValueError(f"{sid}: Verifikation ist kein Objekt")
            spec = VerificationSpec.from_dict(spec_raw)
            if spec.type not in checks:
                raise ValueError(f"{sid}: unbekannter Verifikationstyp {spec.type!r}")
            spec.auto = False
            specs.append(spec)
        seen.add(sid)
        result.append(Subtask(id=sid, description=description, depends_on=deps, verification=specs))
    return result


_CHECK_DOCS = """\
- {"type": "file_exists", "path": "<rel path>"}
- {"type": "file_contains", "path": "<rel path>", "text": "<substring>"}
- {"type": "python_syntax", "path": "<file or directory>"}
- {"type": "command", "command": ["python", "-m", "pytest", "-q"]}   (only allow-listed commands)
- {"type": "output_contains", "text": "<substring of your final answer>"}
- {"type": "tool_succeeded", "tool": "<tool name>"}"""

_PLAN_SYSTEM = (
    "You are the planning component of NOVA, a local AI agent. Break the user's objective "
    "into a short sequence of concrete, executable subtasks. Each subtask must be "
    "completable by one model with the listed tools. Attach verification checks wherever "
    "the result can be checked objectively (files, syntax, tests). Prefer few subtasks. "
    "If a test command is configured it runs after every Python file change, so put an "
    "implementation and its tests into the SAME subtask (or the implementation first). "
    "Respond with JSON only."
)


class LLMPlanner(Planner):
    def __init__(
        self,
        engine: InferenceEngine,
        *,
        tools: Sequence[ToolSpec] = (),
        known_checks: Iterable[str] = (),
        max_parse_retries: int = 1,
    ) -> None:
        self.engine = engine
        self.tools = list(tools)
        self.known_checks = list(known_checks)
        self.max_parse_retries = max_parse_retries

    def _tool_block(self) -> str:
        if not self.tools:
            return "(no tools available)"
        return "\n".join(f"- {t.name}: {t.description}" for t in self.tools)

    def _format(self) -> str:
        return (
            'Output format:\n{"subtasks": [{"id": "s1", "description": "...", '
            '"depends_on": [], "verification": [...]}]}\n'
            f"Available verification checks:\n{_CHECK_DOCS}\n"
        )

    async def plan(self, task: Task) -> list[Subtask]:
        notes = "\n".join(f"- {n}" for n in task.constraints.notes) or "- none"
        memory = f"Memory:\n{task.memory_context}\n\n" if task.memory_context else ""
        prompt = (
            f"Objective:\n{task.objective}\n\n{memory}Constraints:\n{notes}\n"
            f"Maximum subtasks: {task.constraints.max_subtasks}\n\n"
            f"Available tools:\n{self._tool_block()}\n\n{self._format()}"
        )
        return await self._ask(task, prompt, id_prefix="", existing_ids=())

    async def replan(self, task: Task, failed: Subtask) -> list[Subtask]:
        done = [s for s in task.subtasks if s.status == SubtaskStatus.DONE]
        done_block = "\n".join(f"- {s.id}: {s.description} → {s.output[:300]}" for s in done)
        errors = [e for e in task.errors if e.subtask_id == failed.id]
        error_block = "\n".join(
            f"- [{e.kind}] {e.message}" + (f" | analysis: {e.analysis}" if e.analysis else "")
            for e in errors[-6:]
        )
        prompt = (
            f"Objective:\n{task.objective}\n\nAlready completed:\n{done_block or '- nothing'}\n\n"
            f"This subtask failed after {failed.attempts} attempts:\n- {failed.id}: "
            f"{failed.description}\nErrors:\n{error_block or '- none recorded'}\n\n"
            "Propose a DIFFERENT approach for the remaining work (replaces the failed subtask "
            "and everything after it). Completed subtasks may be referenced in depends_on.\n\n"
            f"Available tools:\n{self._tool_block()}\n\n{self._format()}"
        )
        prefix = f"r{task.replans + 1}-"
        return await self._ask(task, prompt, id_prefix=prefix, existing_ids=[s.id for s in done])

    async def _ask(
        self, task: Task, prompt: str, *, id_prefix: str, existing_ids: Iterable[str]
    ) -> list[Subtask]:
        messages = [Message.system(_PLAN_SYSTEM), Message.user(prompt)]
        last_error = ""
        for _ in range(self.max_parse_retries + 1):
            request = ChatRequest(
                messages=tuple(messages), params=GenerationParams(temperature=0.0, seed=0)
            )
            try:
                result = await self.engine.chat(
                    request, task=TaskRequirements(task_type=TaskType.REASONING), fallback=True
                )
            except ModelError as exc:
                raise PlanningError(f"Planungsmodell nicht verfügbar: {exc}") from exc
            text = result.response.message.content
            try:
                return validate_plan(
                    extract_json_object(text),
                    known_checks=self.known_checks,
                    max_subtasks=task.constraints.max_subtasks,
                    id_prefix=id_prefix,
                    existing_ids=existing_ids,
                )
            except ValueError as exc:
                last_error = str(exc)
                messages += [
                    Message.assistant(text),
                    Message.user(f"Invalid plan: {last_error}. Return corrected JSON only."),
                ]
        raise PlanningError(
            f"Kein gültiger Plan nach {self.max_parse_retries + 1} Versuchen: {last_error}"
        )
