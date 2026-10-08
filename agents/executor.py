"""EXECUTE / OBSERVE und CORRECT.

:class:`Executor` bearbeitet genau eine Teilaufgabe in einer Tool-Calling-Schleife:
Modell → Tool-Calls → Tool-Ergebnisse (als Beobachtungen in den Task State) → Modell …
bis das Modell ohne Tool-Call antwortet oder das Budget erschöpft ist.

Fehlgeschlagene Tool-Aufrufe werden in ``task.errors`` geschrieben und dem Modell mit der
Aufforderung zurückgegeben, die Ursache zu analysieren und einen anderen Weg zu wählen.

:class:`FailureAnalyzer` erzeugt in der CORRECT-Phase eine Ursachenanalyse für den
nächsten Versuch: zuerst deterministische Fakten (fehlgeschlagene Checks, Fehler), dann –
falls verfügbar – eine kurze Modellanalyse.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agents.task import Phase, Subtask, SubtaskStatus, Task, ToolResultRecord, Verdict
from models.base import ChatRequest, GenerationParams, Message, ModelError
from models.capabilities import TaskRequirements, TaskType
from models.inference import InferenceEngine
from tools.base import ToolContext, ToolInvocation, ToolResult
from tools.registry import ToolRegistry

_EXEC_SYSTEM = (
    "You are the execution component of NOVA, a local AI agent. Complete ONLY the current "
    "subtask. Use the tools to read or change files – never pretend an action happened. "
    "Every tool call needs a short 'reason'. Tool results are structured JSON "
    "{success, output, error, metadata} and authoritative: if a tool fails, analyse the cause "
    "and try a different approach. When the subtask is finished, reply WITHOUT tool calls "
    "with a concise, factual "
    "summary of what you did and what the result is. If you could not complete it, say so."
)

_ANALYZE_SYSTEM = (
    "You are the error analysis component of NOVA. Given a failed attempt, state in at most "
    "4 short sentences: (1) the most likely root cause, (2) a concrete different approach "
    "for the next attempt. Do not repeat the raw logs."
)


@dataclass
class ExecutionOutcome:
    output: str
    completed: bool
    """True, wenn das Modell die Teilaufgabe regulär (ohne offene Tool-Calls) beendet hat."""
    tool_calls: int = 0
    errors: list[str] = field(default_factory=list)


def _jsonable(value: Any) -> Any:
    """Metadaten für den persistenten State auf JSON-Typen reduzieren."""
    return json.loads(json.dumps(value, default=str))


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "…"


class Executor:
    def __init__(
        self,
        engine: InferenceEngine,
        tools: ToolRegistry,
        workspace: Path,
        *,
        temperature: float = 0.2,
        context_chars_per_result: int = 1500,
        extra_roots: tuple[Path, ...] = (),
        backup_dir: Path | None = None,
    ) -> None:
        self.extra_roots = extra_roots
        self.backup_dir = backup_dir
        self.engine = engine
        self.tools = tools
        self.workspace = workspace
        self.temperature = temperature
        self.context_chars_per_result = context_chars_per_result

    def _tool_names(self, task: Task) -> list[str]:
        allowed = task.constraints.allowed_tools
        return (
            self.tools.names()
            if allowed is None
            else [n for n in self.tools.names() if n in allowed]
        )

    def build_messages(self, task: Task, subtask: Subtask) -> list[Message]:
        plan = "\n".join(f"- [{s.status.value}] {s.id}: {s.description}" for s in task.subtasks)
        deps = [task.subtask(d) for d in subtask.depends_on]
        dep_block = "\n".join(
            f"- {d.id}: {_clip(d.output, self.context_chars_per_result)}" for d in deps
        )
        checks = "\n".join(
            f"- {v.type} {json.dumps(v.params, ensure_ascii=False)}" for v in subtask.verification
        )
        feedback = "\n".join(f"- {f}" for f in subtask.feedback[-3:])
        notes = "\n".join(f"- {n}" for n in task.constraints.notes)
        parts = [
            f"Overall objective:\n{task.objective}",
            f"Plan:\n{plan}",
        ]
        if notes:
            parts.append(f"Constraints:\n{notes}")
        if dep_block:
            parts.append(f"Results of prerequisite subtasks:\n{dep_block}")
        if feedback:
            parts.append(
                f"Previous attempts of this subtask FAILED. Feedback:\n{feedback}\n"
                "Use a corrected or different approach."
            )
        if checks:
            parts.append(f"Your result will be verified automatically by:\n{checks}")
        parts.append(
            f"Current subtask ({subtask.id}, attempt {subtask.attempts}):\n{subtask.description}"
        )
        return [
            Message.system(_EXEC_SYSTEM + f"\nWorkspace root: {self.workspace}"),
            Message.user("\n\n".join(parts)),
        ]

    async def execute(self, task: Task, subtask: Subtask) -> ExecutionOutcome:
        attempt = subtask.attempts
        # Reine Wissens-/Textaufgaben bekommen keine Tools: kleinerer Prompt und auch Modelle
        # ohne Tool-Calling sind dann wählbar. Mehrschritt-Pläne bekommen immer Tools.
        wants_tools = (
            task.analysis is None or task.analysis.requires_tools or len(task.subtasks) > 1
        )
        tool_names = self._tool_names(task) if wants_tools else []
        specs = tuple(self.tools.specs(tool_names))
        analysis_type = task.analysis.task_type if task.analysis else TaskType.GENERAL
        requirements = TaskRequirements(task_type=analysis_type, needs_tool_calling=bool(specs))
        ctx = ToolContext(
            workspace=self.workspace,
            extra_roots=self.extra_roots,
            timeout_s=task.constraints.tool_timeout_s,
            backup_dir=self.backup_dir,
            task_id=task.id,
        )
        messages = self.build_messages(task, subtask)
        outcome = ExecutionOutcome(output="", completed=False)

        while True:
            request = ChatRequest(
                messages=tuple(messages),
                tools=specs,
                params=GenerationParams(temperature=self.temperature),
            )
            try:
                result = await self.engine.chat(request, task=requirements, fallback=True)
            except ModelError as exc:
                message = f"Modellaufruf fehlgeschlagen: {type(exc).__name__}: {exc}"
                task.record_error(Phase.EXECUTE, "model", message, subtask.id, attempt)
                outcome.errors.append(message)
                return outcome

            reply = result.response.message
            if reply.content:
                task.observe("model", reply.content, subtask.id, attempt)
            if not reply.tool_calls:
                outcome.output = reply.content.strip()
                outcome.completed = bool(outcome.output)
                if not outcome.completed:
                    message = "Modell hat weder Tool-Calls noch eine Antwort geliefert"
                    task.record_error(Phase.EXECUTE, "model", message, subtask.id, attempt)
                    outcome.errors.append(message)
                return outcome

            messages.append(reply)
            for call in reply.tool_calls:
                if outcome.tool_calls >= task.constraints.max_tool_calls_per_attempt:
                    message = (
                        f"Tool-Budget erschöpft ({task.constraints.max_tool_calls_per_attempt} "
                        "Aufrufe pro Versuch)"
                    )
                    task.record_error(Phase.EXECUTE, "budget", message, subtask.id, attempt)
                    outcome.errors.append(message)
                    outcome.output = reply.content.strip()
                    return outcome
                outcome.tool_calls += 1
                invocation = ToolInvocation.from_model(
                    call.name,
                    call.arguments,
                    call_id=call.id,
                    task_id=task.id,
                    subtask_id=subtask.id,
                )
                if call.name not in tool_names:
                    res = ToolResult.fail(f"Tool {call.name!r} ist für diese Aufgabe nicht erlaubt")
                else:
                    res = await self.tools.execute(invocation, ctx)
                task.tool_results.append(
                    ToolResultRecord(
                        subtask_id=subtask.id,
                        attempt=attempt,
                        tool=call.name,
                        arguments=dict(invocation.arguments),
                        success=res.success,
                        output=_clip(res.output, 4000) if res.output is not None else None,
                        error=res.error,
                        metadata=_jsonable(res.metadata),
                        reason=invocation.reason,
                        invocation_id=invocation.id,
                    )
                )
                if res.success:
                    task.observe(
                        f"tool:{call.name}", _clip(res.output or "", 1000), subtask.id, attempt
                    )
                else:
                    message = f"{call.name} fehlgeschlagen: {res.error}"
                    task.record_error(Phase.OBSERVE, "tool", message, subtask.id, attempt)
                    task.observe(f"tool:{call.name}", f"ERROR: {res.error}", subtask.id, attempt)
                    outcome.errors.append(message)
                content = json.dumps(res.to_dict(), ensure_ascii=False, default=str)
                if not res.success:
                    content += (
                        "\nThe tool call FAILED. Analyse the cause. Do not claim success; try a "
                        "different approach or report that the subtask cannot be completed."
                    )
                messages.append(Message.tool(call.id, content))


class FailureAnalyzer:
    def __init__(self, engine: InferenceEngine | None) -> None:
        self.engine = engine

    @staticmethod
    def facts(task: Task, subtask: Subtask, outcome: ExecutionOutcome) -> str:
        attempt = subtask.attempts
        failed_checks = [
            f"{v.check}: {v.detail}"
            for v in task.verification_results
            if v.subtask_id == subtask.id and v.attempt == attempt and v.verdict == Verdict.FAILED
        ]
        lines = [f"Versuch {attempt} fehlgeschlagen."]
        if not outcome.completed:
            lines.append("Ausführung nicht regulär abgeschlossen.")
        lines += [f"Fehler: {e}" for e in outcome.errors[-5:]]
        lines += [f"Check fehlgeschlagen: {c}" for c in failed_checks[-5:]]
        return " ".join(lines)

    async def analyze(self, task: Task, subtask: Subtask, outcome: ExecutionOutcome) -> str:
        facts = self.facts(task, subtask, outcome)
        record = task.record_error(
            Phase.CORRECT,
            "verification" if outcome.completed else "execution",
            facts,
            subtask.id,
            subtask.attempts,
        )
        if self.engine is None:
            return facts
        prompt = (
            f"Subtask: {subtask.description}\nModel output of the attempt: "
            f"{_clip(outcome.output, 1500) or '(none)'}\nFacts: {facts}"
        )
        request = ChatRequest(
            messages=(Message.system(_ANALYZE_SYSTEM), Message.user(prompt)),
            params=GenerationParams(temperature=0.0, max_tokens=300),
        )
        try:
            result = await self.engine.chat(
                request, task=TaskRequirements(task_type=TaskType.REASONING), fallback=True
            )
        except ModelError as exc:
            task.record_error(
                Phase.CORRECT,
                "model",
                f"Fehleranalyse nicht möglich: {exc}",
                subtask.id,
                subtask.attempts,
            )
            return facts
        analysis = result.response.message.content.strip()
        record.analysis = analysis
        return f"{facts} Analyse: {analysis}" if analysis else facts


def reset_interrupted(task: Task) -> None:
    """Teilaufgaben, die beim Abbruch liefen, werden erneut ausgeführt."""
    for sub in task.subtasks:
        if sub.status == SubtaskStatus.IN_PROGRESS:
            sub.status = SubtaskStatus.PENDING
