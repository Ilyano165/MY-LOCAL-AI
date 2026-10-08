"""Standard-Benchmark-Aufgaben für Modellfähigkeit (reproduzierbar, unabhängig bewertet).

Jede Aufgabe

* hat feste Prompts und läuft mit ``temperature=0`` und festem ``seed``,
* wird **ohne Modellurteil** bewertet: Regex-/Wertprüfung, versteckte Unit-Tests in einem
  Subprozess, Dateizustand im Arbeitsverzeichnis,
* liefert einen Score = Anteil erfüllter Prüfkriterien in [0, 1].

Die Suite-Version enthält einen Hash über alle Prompts und Prüfregeln: Ändert sich eine
Aufgabe, ändert sich die Version, und alte Profile gelten als veraltet.

Aufgaben: ``chat``, ``short_analysis``, ``complex_analysis``, ``coding``, ``debugging``,
``agent_multistep``, ``tool_calling``, ``long_context`` (wenn Kontext reicht) und die
Zusatzprobe ``vision`` (nur wenn die Runtime Bilder annimmt).

Reproduzierbarkeit: Gleiches Modell + gleiche Runtime-Version + gleiche Server-Optionen →
gleiche Ausgabe bei Greedy-Decoding. Andere Builds/Batchgrößen können abweichen.
"""

from __future__ import annotations

import abc
import asyncio
import hashlib
import json
import random
import re
import struct
import time
import uuid
import zlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from evaluation.benchmark_results import TaskResult, TaskStatus
from models.base import (
    ChatRequest,
    ChatResponse,
    ContextLengthExceededError,
    GenerationParams,
    ImageInput,
    InvalidResponseError,
    Message,
    ModelError,
    ModelProvider,
    ToolCall,
    ToolSpec,
)
from models.capabilities import ModelMetadata
from tools.base import ToolError
from tools.process import run_process

SUITE_VERSION = "1.0"
SEED = 42


@dataclass
class TaskContext:
    provider: ModelProvider
    model: ModelMetadata
    workdir: Path
    """Leeres, nur für diesen Lauf angelegtes Verzeichnis."""
    timeout_s: float = 300.0
    context_length: int | None = None
    """Real verfügbares Kontextfenster (Runtime-Angabe bevorzugt)."""
    vision_supported: bool | None = None
    """Von der Runtime gemeldet (llama.cpp ``/props`` modalities) oder unbekannt."""
    execute_code: bool = True
    """Generierten Code in einem Subprozess testen (sonst werden Coding-Aufgaben übersprungen)."""
    long_context_max_tokens: int = 16_384


def params(max_tokens: int) -> GenerationParams:
    return GenerationParams(temperature=0.0, max_tokens=max_tokens, seed=SEED)


@dataclass
class _Run:
    """Sammelt Token- und Zeitdaten über alle Modellaufrufe einer Aufgabe."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    runtime_stats: dict[str, float] = field(default_factory=dict)
    last_text: str = ""

    async def chat(self, ctx: TaskContext, request: ChatRequest) -> ChatResponse:
        response = await ctx.provider.chat(ctx.model, request)
        self.prompt_tokens += response.usage.prompt_tokens
        self.completion_tokens += response.usage.completion_tokens
        for key, value in response.runtime_stats.items():
            if key.endswith(("_n", "_ms")):
                self.runtime_stats[key] = self.runtime_stats.get(key, 0.0) + value
        self.last_text = response.message.content
        return response


class BenchmarkTask(abc.ABC):
    id: str
    category: str
    """Bereich: general, reasoning, coding, tool_calling, agentic, long_context, vision."""
    title: str
    revision: int = 1

    @abc.abstractmethod
    def spec(self) -> str:
        """Alle Prompts und Prüfregeln als Text – Grundlage des Versions-Hashes."""

    @abc.abstractmethod
    async def _execute(self, ctx: TaskContext, run: _Run) -> tuple[dict[str, bool], str]:
        """Führt die Aufgabe aus und gibt (Prüfkriterien, Detail) zurück."""

    def applicable(self, ctx: TaskContext) -> tuple[bool, str]:
        return True, ""

    async def run(self, ctx: TaskContext) -> TaskResult:
        ok, why = self.applicable(ctx)
        if not ok:
            return TaskResult(self.id, self.category, TaskStatus.SKIPPED, None, detail=why)
        run = _Run()
        started = time.perf_counter()
        try:
            checks, detail = await self._execute(ctx, run)
        except _Unsupported as exc:
            return TaskResult(
                self.id,
                self.category,
                TaskStatus.UNSUPPORTED,
                None,
                detail=str(exc),
                duration_s=time.perf_counter() - started,
            )
        except ModelError as exc:
            return TaskResult(
                self.id,
                self.category,
                TaskStatus.ERROR,
                None,
                detail=f"{type(exc).__name__}: {exc}",
                duration_s=time.perf_counter() - started,
                prompt_tokens=run.prompt_tokens,
                completion_tokens=run.completion_tokens,
            )
        score = sum(checks.values()) / len(checks) if checks else 0.0
        return TaskResult(
            self.id,
            self.category,
            TaskStatus.COMPLETED,
            round(score, 4),
            checks=checks,
            detail=detail,
            duration_s=time.perf_counter() - started,
            prompt_tokens=run.prompt_tokens,
            completion_tokens=run.completion_tokens,
            runtime_stats=run.runtime_stats,
            output_excerpt=run.last_text[:400],
        )


class _Unsupported(Exception):
    pass


# ---------------------------------------------------------------------- Hilfsfunktionen

_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(text: str) -> str:
    """Längster Python-Codeblock; ohne Codeblock der gesamte Text."""
    blocks = _FENCE.findall(text)
    return max(blocks, key=len).strip() if blocks else text.strip()


def _strip_reasoning(text: str) -> str:
    """Entfernt ``<think>…</think>``-Blöcke von Reasoning-Modellen vor der Bewertung."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()


async def run_hidden_tests(
    workdir: Path, code: str, tests: str, timeout_s: float = 30.0
) -> dict[str, bool]:
    """Führt ``tests`` gegen ``code`` in einem isolierten Subprozess aus.

    ``tests`` definiert ``CASES = [(name, callable_returning_bool), …]``; das Ergebnis wird als
    JSON auf stdout geschrieben. Kein Netz-/Shell-Zugriff durch NOVA; Zeitlimit; leere
    Umgebung ohne Secrets (``run_process``).
    """
    marker = f"__NOVA_{uuid.uuid4().hex}__"  # zufällig: vom Modellcode nicht vorhersehbar
    harness = (
        "import json, sys\n"
        "sys.path.insert(0, '.')\n"
        "results = {}\n"
        "try:\n"
        "    import solution\n"
        "except BaseException as exc:\n"
        "    print(json.dumps({'__import__': False})); sys.exit(0)\n"
        f"{tests}\n"
        "for name, case in CASES:\n"
        "    try:\n"
        "        results[name] = bool(case())\n"
        "    except BaseException:\n"
        "        results[name] = False\n"
        f"print({marker!r} + json.dumps(results))\n"
    )

    def prepare() -> None:
        workdir.mkdir(parents=True, exist_ok=True)
        (workdir / "solution.py").write_text(code, encoding="utf-8")
        (workdir / "harness.py").write_text(harness, encoding="utf-8")

    await asyncio.to_thread(prepare)
    try:
        result = await run_process(
            ["python", "-I", "harness.py"], workdir, timeout_s=timeout_s, max_output_bytes=100_000
        )
    except ToolError as exc:
        return {"harness_started": False, "error:" + str(exc)[:60]: False}
    position = result.output.rfind(marker)
    if result.timed_out:
        return {"completes_within_timeout": False}
    if position < 0:
        return {"imports": False}
    try:
        data = json.loads(result.output[position + len(marker) :].splitlines()[0])
    except (ValueError, IndexError):
        return {"imports": False}
    return {"imports": True, **{str(k): bool(v) for k, v in data.items()}}


# ---------------------------------------------------------------------- 1. Chat


class SimpleChatTask(BenchmarkTask):
    id, category, title = "chat", "general", "Einfacher Chat"
    PROMPT = "What is the capital of France? Answer in exactly one short sentence."

    def spec(self) -> str:
        return f"{self.PROMPT}|checks: mentions Paris; one sentence; <= 200 chars"

    async def _execute(self, ctx: TaskContext, run: _Run) -> tuple[dict[str, bool], str]:
        response = await run.chat(
            ctx,
            ChatRequest((Message.user(self.PROMPT),), params=params(256), timeout_s=ctx.timeout_s),
        )
        text = _strip_reasoning(response.message.content)
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
        return {
            "mentions_paris": bool(re.search(r"\bParis\b", text)),
            "one_sentence": len(sentences) == 1,
            "concise": 0 < len(text) <= 200,
        }, text[:120]


# ---------------------------------------------------------------------- 2. Kurze Analyse


class ShortAnalysisTask(BenchmarkTask):
    id, category, title = "short_analysis", "reasoning", "Kurze Analyse"
    PROMPT = (
        "Quarterly revenue of a small shop (thousand EUR): Q1 = 120, Q2 = 150, Q3 = 90, "
        "Q4 = 180.\nWhich quarter had the highest revenue, and by what percentage did revenue "
        "change compared to the previous quarter?\nEnd your answer with exactly one line in "
        "this format:\nRESULT: QUARTER=<Q1|Q2|Q3|Q4>; CHANGE=<signed number>%"
    )

    def spec(self) -> str:
        return f"{self.PROMPT}|expect Q4, +100%"

    async def _execute(self, ctx: TaskContext, run: _Run) -> tuple[dict[str, bool], str]:
        response = await run.chat(
            ctx,
            ChatRequest((Message.user(self.PROMPT),), params=params(512), timeout_s=ctx.timeout_s),
        )
        text = _strip_reasoning(response.message.content)
        matches = re.findall(
            r"RESULT:\s*QUARTER\s*=\s*(Q[1-4])\s*;\s*CHANGE\s*=\s*([+-]?\d+(?:[.,]\d+)?)\s*%",
            text,
            re.IGNORECASE,
        )
        quarter, change = matches[-1] if matches else ("", "")
        value = float(change.replace(",", ".")) if change else None
        return {
            "format": bool(matches),
            "quarter_q4": quarter.upper() == "Q4",
            "change_100_percent": value is not None and abs(value - 100.0) < 0.5,
        }, f"quarter={quarter or '?'} change={change or '?'}"


# ---------------------------------------------------------------------- 3. Komplexe Analyse

PEOPLE = ("Anna", "Ben", "Cem", "Dana")
PETS = ("cat", "dog", "fish", "bird")
PUZZLE_SOLUTION = {"Anna": "fish", "Ben": "dog", "Cem": "bird", "Dana": "cat"}
PUZZLE_CLUES = (
    "Anna owns neither the cat nor the dog.",
    "The owner of the bird has a name with exactly three letters.",
    "Ben owns neither the bird nor the cat.",
    "Dana does not own the dog.",
    "Anna's pet cannot fly.",
)


def puzzle_constraints(assign: dict[str, str]) -> bool:
    """Formale Fassung der Hinweise – im Test per Brute Force auf Eindeutigkeit geprüft."""
    owner = {pet: person for person, pet in assign.items()}
    return (
        assign["Anna"] not in ("cat", "dog")
        and len(owner["bird"]) == 3
        and assign["Ben"] not in ("bird", "cat")
        and assign["Dana"] != "dog"
        and assign["Anna"] != "bird"
    )


class ComplexAnalysisTask(BenchmarkTask):
    id, category, title = "complex_analysis", "reasoning", "Komplexe Analyse (Logikrätsel)"
    PROMPT = (
        "Four people (Anna, Ben, Cem, Dana) each own exactly one different pet "
        "(cat, dog, fish, bird).\nClues:\n"
        + "\n".join(f"{i}. {c}" for i, c in enumerate(PUZZLE_CLUES, 1))
        + "\nWork it out step by step. Then end with exactly one line in this format:\n"
        "ANSWER: Anna=<pet>, Ben=<pet>, Cem=<pet>, Dana=<pet>"
    )

    def spec(self) -> str:
        return f"{self.PROMPT}|solution {sorted(PUZZLE_SOLUTION.items())}"

    async def _execute(self, ctx: TaskContext, run: _Run) -> tuple[dict[str, bool], str]:
        response = await run.chat(
            ctx,
            ChatRequest((Message.user(self.PROMPT),), params=params(1536), timeout_s=ctx.timeout_s),
        )
        text = _strip_reasoning(response.message.content)
        lines = [ln for ln in text.splitlines() if re.match(r"\W*ANSWER\s*:", ln, re.I)]
        answer = lines[-1] if lines else ""
        found = {
            person: (
                m.group(1).lower()
                if (
                    m := re.search(
                        rf"\b{person}\s*[=:]\s*(?:the\s+)?(cat|dog|fish|bird)\b", answer, re.I
                    )
                )
                else ""
            )
            for person in PEOPLE
        }
        checks = {"format": bool(answer)}
        checks |= {f"{p.lower()}_correct": found[p] == PUZZLE_SOLUTION[p] for p in PEOPLE}
        return checks, answer.strip()[:120]


# ---------------------------------------------------------------------- 4. Coding


class CodingTask(BenchmarkTask):
    id, category, title = "coding", "coding", "Coding (Funktion schreiben)"
    PROMPT = (
        "Write a Python function `is_palindrome(s: str) -> bool` that returns True if `s` "
        "reads the same forwards and backwards when case is ignored and every character that "
        "is not alphanumeric (per str.isalnum) is removed. An empty string is a palindrome.\n"
        "Reply with only the code in a single ```python code block, no explanation."
    )
    TESTS = """
f = solution.is_palindrome
CASES = [
    ("panama", lambda: f("A man, a plan, a canal: Panama") is True),
    ("race_a_car", lambda: f("race a car") is False),
    ("empty", lambda: f("") is True),
    ("single", lambda: f("x") is True),
    ("nixon", lambda: f("No 'x' in Nixon") is True),
    ("two_different", lambda: f("ab") is False),
    ("digits_letters", lambda: f("0P") is False),
    ("unicode", lambda: f("Ünü") is True),
    ("punctuation_only", lambda: f("!!??") is True),
]
"""

    def spec(self) -> str:
        return self.PROMPT + self.TESTS

    def applicable(self, ctx: TaskContext) -> tuple[bool, str]:
        if not ctx.execute_code:
            return False, "Code-Ausführung deaktiviert (--no-exec)"
        return True, ""

    async def _execute(self, ctx: TaskContext, run: _Run) -> tuple[dict[str, bool], str]:
        response = await run.chat(
            ctx,
            ChatRequest((Message.user(self.PROMPT),), params=params(1024), timeout_s=ctx.timeout_s),
        )
        code = extract_code(_strip_reasoning(response.message.content))
        checks = await run_hidden_tests(ctx.workdir / self.id, code, self.TESTS)
        passed = sum(checks.values())
        return checks, f"{passed}/{len(checks)} Prüfungen bestanden"


# ---------------------------------------------------------------------- 5. Debugging


class DebuggingTask(BenchmarkTask):
    id, category, title = "debugging", "coding", "Debugging (Fehler finden und beheben)"
    BUGGY = (
        "def median(values):\n"
        "    s = sorted(values)\n"
        "    n = len(s)\n"
        "    mid = n // 2\n"
        "    if n % 2 == 1:\n"
        "        return (s[mid - 1] + s[mid]) / 2\n"
        "    return s[mid]\n"
    )
    PROMPT = (
        "The following function should return the median of a list of numbers, but it returns "
        "wrong results.\n```python\n" + BUGGY + "```\n"
        "Fix the bug. Additionally, the function must raise ValueError for an empty list and "
        "must not modify its input. Reply with only the corrected function in a single "
        "```python code block."
    )
    TESTS = """
m = solution.median
def _empty():
    try:
        m([])
    except ValueError:
        return True
    return False
def _no_mutation():
    data = [3, 1, 2]
    m(data)
    return data == [3, 1, 2]
CASES = [
    ("odd", lambda: m([3, 1, 2]) == 2),
    ("even", lambda: m([4, 1, 3, 2]) == 2.5),
    ("single", lambda: m([7]) == 7),
    ("negatives", lambda: m([-5, -1, -3]) == -3),
    ("floats", lambda: abs(m([1.5, 2.5]) - 2.0) < 1e-9),
    ("empty_raises", _empty),
    ("no_mutation", _no_mutation),
]
"""

    def spec(self) -> str:
        return self.PROMPT + self.TESTS

    def applicable(self, ctx: TaskContext) -> tuple[bool, str]:
        if not ctx.execute_code:
            return False, "Code-Ausführung deaktiviert (--no-exec)"
        return True, ""

    async def _execute(self, ctx: TaskContext, run: _Run) -> tuple[dict[str, bool], str]:
        response = await run.chat(
            ctx,
            ChatRequest((Message.user(self.PROMPT),), params=params(1024), timeout_s=ctx.timeout_s),
        )
        code = extract_code(_strip_reasoning(response.message.content))
        checks = await run_hidden_tests(ctx.workdir / self.id, code, self.TESTS)
        return checks, f"{sum(checks.values())}/{len(checks)} Prüfungen bestanden"


# ---------------------------------------------------------------------- 6./7. Tool-Aufrufe


def _tool_unsupported(exc: ModelError) -> bool:
    text = str(exc).lower()
    return "tool" in text and any(
        marker in text for marker in ("not supported", "unsupported", "jinja", "does not support")
    )


WEATHER_TOOL = ToolSpec(
    name="get_weather",
    description="Get the current weather for a city.",
    parameters={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name, e.g. Paris"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["city", "unit"],
    },
)


class ToolCallingTask(BenchmarkTask):
    id, category, title = "tool_calling", "tool_calling", "Tool-Calling"
    PROMPT = "What is the current weather in Berlin? Use the available tool, in celsius."
    TOOL_RESULT = '{"city": "Berlin", "temperature": 18, "unit": "celsius", "condition": "cloudy"}'

    def spec(self) -> str:
        return (
            f"{self.PROMPT}|{json.dumps(WEATHER_TOOL.parameters, sort_keys=True)}|"
            f"{self.TOOL_RESULT}|checks: call, name, city, unit, uses result (18)"
        )

    async def _execute(self, ctx: TaskContext, run: _Run) -> tuple[dict[str, bool], str]:
        messages: list[Message] = [Message.user(self.PROMPT)]
        checks = dict.fromkeys(
            ("tool_called", "correct_tool", "city_berlin", "unit_celsius", "uses_tool_result"),
            False,
        )
        try:
            response = await run.chat(
                ctx,
                ChatRequest(
                    tuple(messages),
                    tools=(WEATHER_TOOL,),
                    params=params(512),
                    timeout_s=ctx.timeout_s,
                ),
            )
        except InvalidResponseError as exc:
            return checks, f"ungültiger Tool-Aufruf: {exc}"
        except ModelError as exc:
            if _tool_unsupported(exc):
                raise _Unsupported(f"Runtime lehnt Tools ab: {exc}") from exc
            raise
        calls = response.message.tool_calls
        if not calls:
            return checks, "kein Tool-Aufruf"
        call = calls[0]
        checks["tool_called"] = True
        checks["correct_tool"] = call.name == "get_weather"
        checks["city_berlin"] = str(call.arguments.get("city", "")).strip().lower() == "berlin"
        checks["unit_celsius"] = str(call.arguments.get("unit", "")).strip().lower() == "celsius"
        messages += [response.message, Message.tool(call.id, self.TOOL_RESULT)]
        try:
            final = await run.chat(
                ctx,
                ChatRequest(
                    tuple(messages),
                    tools=(WEATHER_TOOL,),
                    params=params(512),
                    timeout_s=ctx.timeout_s,
                ),
            )
        except InvalidResponseError as exc:
            return checks, f"ungültige Antwort nach Tool-Ergebnis: {exc}"
        checks["uses_tool_result"] = bool(re.search(r"\b18\b", final.message.content))
        return checks, f"{call.name}({dict(call.arguments)})"


AGENT_CSV = (
    "item,quantity,unit_price\n"
    "widget,3,2.50\n"
    "gadget,2,10.00\n"
    "doohickey,5,1.20\n"
    "thingamajig,1,99.99\n"
)
AGENT_EXPECTED = 133.49


def _agent_tools() -> tuple[ToolSpec, ...]:
    path = {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
    return (
        ToolSpec(
            "list_files", "List the files in the workspace.", {"type": "object", "properties": {}}
        ),
        ToolSpec("read_file", "Read a text file from the workspace.", path),
        ToolSpec(
            "write_file",
            "Write text to a file in the workspace (creates or overwrites).",
            {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
            },
        ),
    )


class AgentMultiStepTask(BenchmarkTask):
    id, category, title = "agent_multistep", "agentic", "Mehrschritt-Agentenaufgabe"
    PROMPT = (
        "You are working in a workspace with tools. Find the CSV file, compute the total "
        "revenue (sum of quantity * unit_price over all rows) and write only that number, "
        "rounded to two decimals, into a new file named result.txt. When you are done, reply "
        "with a short confirmation."
    )
    MAX_STEPS = 8

    def spec(self) -> str:
        return (
            f"{self.PROMPT}|{AGENT_CSV}|{AGENT_EXPECTED}|max_steps={self.MAX_STEPS}|"
            f"{[t.name for t in _agent_tools()]}"
        )

    def _tool(self, workspace: Path, call: ToolCall, trace: list[str]) -> str:
        trace.append(call.name)
        name = str(call.arguments.get("path", "")).strip()
        root = workspace.resolve()
        target = (root / name).resolve()
        if name and target != root and root not in target.parents:
            return "error: path outside workspace"
        if call.name == "list_files":
            return "\n".join(sorted(p.name for p in workspace.iterdir()))
        if call.name == "read_file":
            if not target.is_file():
                return f"error: file not found: {name}"
            trace.append(f"read:{target.name}")
            return target.read_text(encoding="utf-8")
        if call.name == "write_file":
            if not name:
                return "error: path required"
            target.write_text(str(call.arguments.get("content", "")), encoding="utf-8")
            trace.append(f"write:{target.name}")
            return "ok"
        return f"error: unknown tool {call.name}"

    async def _execute(self, ctx: TaskContext, run: _Run) -> tuple[dict[str, bool], str]:
        workspace = ctx.workdir / self.id
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / "sales.csv").write_text(AGENT_CSV, encoding="utf-8")
        tools = _agent_tools()
        messages: list[Message] = [Message.user(self.PROMPT)]
        trace: list[str] = []
        finished = False
        for _step in range(self.MAX_STEPS):
            try:
                response = await run.chat(
                    ctx,
                    ChatRequest(
                        tuple(messages), tools=tools, params=params(768), timeout_s=ctx.timeout_s
                    ),
                )
            except InvalidResponseError as exc:
                trace.append(f"invalid:{str(exc)[:40]}")
                break
            except ModelError as exc:
                if _tool_unsupported(exc):
                    raise _Unsupported(f"Runtime lehnt Tools ab: {exc}") from exc
                raise
            messages.append(response.message)
            if not response.message.tool_calls:
                finished = True
                break
            for call in response.message.tool_calls:
                messages.append(Message.tool(call.id, self._tool(workspace, call, trace)))
        result = workspace / "result.txt"
        value: float | None = None
        if result.is_file():
            match = re.search(r"-?\d+(?:\.\d+)?", result.read_text(encoding="utf-8"))
            value = float(match.group()) if match else None
        return {
            "used_tools": any(t in ("list_files", "read_file", "write_file") for t in trace),
            "read_csv": "read:sales.csv" in trace,
            "wrote_result": result.is_file(),
            "correct_value": value is not None and abs(value - AGENT_EXPECTED) < 0.005,
            "finished_within_budget": finished,
        }, " → ".join(trace)[:200]


# ---------------------------------------------------------------------- 8. Langer Kontext

_WORDS = (
    "river",
    "mountain",
    "lantern",
    "harbor",
    "meadow",
    "copper",
    "violet",
    "thunder",
    "garden",
    "pencil",
    "orbit",
    "maple",
    "canvas",
    "silver",
    "whisper",
    "market",
    "bridge",
    "feather",
    "castle",
    "ocean",
)
NEEDLES = (
    (0.1, "BLUE", 4817),
    (0.5, "RED", 2953),
    (0.9, "GREEN", 7364),
)


def haystack(approx_tokens: int, seed: int = SEED) -> str:
    """Deterministischer Fülltext mit drei Codes an festen relativen Positionen."""
    rng = random.Random(seed)
    sentences: list[str] = []
    # ≈ 4 Zeichen/Token bei englischem Text; Satzlänge ≈ 60 Zeichen
    count = max(30, approx_tokens * 4 // 60)
    for _ in range(count):
        words = [rng.choice(_WORDS) for _ in range(8)]
        sentences.append(" ".join(words).capitalize() + ".")
    for position, color, code in sorted(NEEDLES, reverse=True):
        sentences.insert(
            int(len(sentences) * position),
            f"Important: the secret code for the {color.lower()} door is {code}.",
        )
    return " ".join(sentences)


class LongContextTask(BenchmarkTask):
    id, category, title = "long_context", "long_context", "Langer Kontext (Needles)"
    QUESTION = (
        "\n\nWhat are the secret codes for the blue, red and green doors? End with exactly one "
        "line: CODES: BLUE=<n>; RED=<n>; GREEN=<n>"
    )
    MIN_CONTEXT = 4096

    def target_tokens(self, ctx: TaskContext) -> int:
        available = ctx.context_length or ctx.model.context_length
        return min(int(available * 0.6), ctx.long_context_max_tokens)

    def spec(self) -> str:
        return f"{NEEDLES}|{_WORDS}|{self.QUESTION}|fill=0.6 max|seed={SEED}"

    def applicable(self, ctx: TaskContext) -> tuple[bool, str]:
        available = ctx.context_length or ctx.model.context_length
        if available < self.MIN_CONTEXT:
            return False, f"Kontextfenster {available} < {self.MIN_CONTEXT} Tokens"
        return True, ""

    async def _execute(self, ctx: TaskContext, run: _Run) -> tuple[dict[str, bool], str]:
        target = self.target_tokens(ctx)
        prompt = haystack(target) + self.QUESTION
        try:
            response = await run.chat(
                ctx,
                ChatRequest((Message.user(prompt),), params=params(256), timeout_s=ctx.timeout_s),
            )
        except ContextLengthExceededError as exc:
            return {"fits_context": False}, f"Kontext überschritten bei ~{target} Tokens: {exc}"
        text = _strip_reasoning(response.message.content)
        checks = {"fits_context": True}
        for _pos, color, code in NEEDLES:
            checks[f"{color.lower()}_code"] = bool(
                re.search(rf"{color}\s*[=:]\s*{code}\b", text, re.I)
            )
        tokens = response.usage.prompt_tokens or target
        return checks, f"{tokens} Prompt-Tokens getestet"


# ---------------------------------------------------------------------- Vision-Probe


def make_png(width: int, height: int, pixel: Callable[[int, int], tuple[int, int, int]]) -> bytes:
    """Erzeugt ein PNG ohne externe Bibliothek (RGB, 8 Bit)."""
    raw = b"".join(
        b"\x00" + b"".join(bytes(pixel(x, y)) for x in range(width)) for y in range(height)
    )

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def vision_image() -> bytes:
    """Linke Hälfte rot, rechte Hälfte blau (128×64)."""
    return make_png(128, 64, lambda x, _y: (220, 20, 20) if x < 64 else (20, 40, 220))


_VISION_UNSUPPORTED = (
    "image input is not supported",
    "does not support image",
    "multimodal",
    "mmproj",
    "vision is not supported",
    "image_url",
)


class VisionProbeTask(BenchmarkTask):
    id, category, title = "vision", "vision", "Vision-Probe"
    PROMPT = (
        "The image consists of two colored halves. Which color is the left half and which "
        "color is the right half? End with exactly one line: LEFT=<color>; RIGHT=<color>"
    )

    def spec(self) -> str:
        digest = hashlib.sha256(vision_image()).hexdigest()[:16]
        return f"{self.PROMPT}|image={digest}|expect left red, right blue"

    async def _execute(self, ctx: TaskContext, run: _Run) -> tuple[dict[str, bool], str]:
        if ctx.vision_supported is False:
            raise _Unsupported("Runtime meldet keine Bild-Unterstützung (modalities.vision=false)")
        message = Message.user(self.PROMPT, images=(ImageInput(vision_image(), "image/png"),))
        try:
            response = await run.chat(
                ctx, ChatRequest((message,), params=params(256), timeout_s=ctx.timeout_s)
            )
        except ModelError as exc:
            if any(marker in str(exc).lower() for marker in _VISION_UNSUPPORTED):
                raise _Unsupported(f"Runtime lehnt Bilder ab: {exc}") from exc
            raise
        text = _strip_reasoning(response.message.content)
        match = re.search(r"LEFT\s*[=:]\s*(\w+)\s*;?\s*RIGHT\s*[=:]\s*(\w+)", text, re.I)
        left, right = (match.group(1).lower(), match.group(2).lower()) if match else ("", "")
        return {
            "format": bool(match),
            "left_red": left == "red",
            "right_blue": right == "blue",
        }, f"left={left or '?'} right={right or '?'}"


# ---------------------------------------------------------------------- Suite


def standard_tasks() -> list[BenchmarkTask]:
    return [
        SimpleChatTask(),
        ShortAnalysisTask(),
        ComplexAnalysisTask(),
        CodingTask(),
        DebuggingTask(),
        AgentMultiStepTask(),
        ToolCallingTask(),
        LongContextTask(),
        VisionProbeTask(),
    ]


def suite_version(tasks: Sequence[BenchmarkTask] | None = None) -> str:
    """``<SUITE_VERSION>+<hash>`` über alle Aufgaben-Spezifikationen (Prompts, Prüfregeln)."""
    digest = hashlib.sha256()
    for task in tasks if tasks is not None else standard_tasks():
        digest.update(f"{task.id}:{task.revision}:{task.spec()}\n".encode())
    return f"{SUITE_VERSION}+{digest.hexdigest()[:8]}"


def task_ids(tasks: Sequence[BenchmarkTask] | None = None) -> list[str]:
    return [t.id for t in (tasks if tasks is not None else standard_tasks())]


__all__ = [
    "SEED",
    "SUITE_VERSION",
    "BenchmarkTask",
    "TaskContext",
    "extract_code",
    "haystack",
    "make_png",
    "params",
    "puzzle_constraints",
    "run_hidden_tests",
    "standard_tasks",
    "suite_version",
    "task_ids",
]
