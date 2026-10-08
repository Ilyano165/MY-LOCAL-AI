from __future__ import annotations

import itertools
import struct
import zlib
from pathlib import Path

import pytest

from evaluation.benchmark_results import TaskStatus
from evaluation.benchmark_tasks import (
    NEEDLES,
    PEOPLE,
    PETS,
    PUZZLE_SOLUTION,
    AgentMultiStepTask,
    ComplexAnalysisTask,
    LongContextTask,
    TaskContext,
    extract_code,
    haystack,
    make_png,
    puzzle_constraints,
    run_hidden_tests,
    standard_tasks,
    suite_version,
    vision_image,
)
from models.base import ToolCall
from tests.evaluation.bench_fakes import FakeBenchServer
from tests.models.fakes import make_meta


def ctx(server: FakeBenchServer, tmp_path: Path, **kw: object) -> TaskContext:
    return TaskContext(server.provider(), make_meta(), tmp_path, **kw)  # type: ignore[arg-type]


def test_suite_contains_all_standard_categories() -> None:
    ids = [t.id for t in standard_tasks()]
    assert ids == [
        "chat",
        "short_analysis",
        "complex_analysis",
        "coding",
        "debugging",
        "agent_multistep",
        "tool_calling",
        "long_context",
        "vision",
    ]


def test_suite_version_is_stable_and_content_addressed(monkeypatch: pytest.MonkeyPatch) -> None:
    v1 = suite_version()
    assert v1 == suite_version() and v1.startswith("1.0+")
    monkeypatch.setattr(ComplexAnalysisTask, "PROMPT", ComplexAnalysisTask.PROMPT + " ")
    assert suite_version() != v1  # Prompt geändert → neue Version → alte Profile veraltet


def test_puzzle_has_exactly_one_solution() -> None:
    solutions = [
        dict(zip(PEOPLE, perm, strict=True))
        for perm in itertools.permutations(PETS)
        if puzzle_constraints(dict(zip(PEOPLE, perm, strict=True)))
    ]
    assert solutions == [PUZZLE_SOLUTION]


def test_extract_code() -> None:
    text = "Intro\n```python\nx = 1\n```\nmore\n```py\ndef f():\n    return 2\n```"
    assert extract_code(text) == "def f():\n    return 2"
    assert extract_code("def g(): pass") == "def g(): pass"


def test_haystack_is_deterministic_with_ordered_needles() -> None:
    a, b = haystack(2000), haystack(2000)
    assert a == b
    positions = [a.index(f"is {code}.") for _p, _c, code in NEEDLES]
    assert positions == sorted(positions)
    assert positions[0] < len(a) * 0.2 and positions[2] > len(a) * 0.8
    assert 1500 < len(a) / 4 < 2600


def test_make_png_is_valid() -> None:
    png = vision_image()
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    width, height = struct.unpack(">II", png[16:24])
    assert (width, height) == (128, 64)
    # CRC des IHDR-Chunks prüfen
    assert struct.unpack(">I", png[29:33])[0] == zlib.crc32(png[12:29]) & 0xFFFFFFFF
    assert make_png(1, 1, lambda _x, _y: (0, 0, 0)).endswith(b"IEND\xaeB`\x82")


class TestHiddenTests:
    TESTS = (
        'f = solution.add\nCASES = [("one", lambda: f(1, 1) == 2), ("neg", lambda: f(-1, 1) == 0)]'
    )

    async def test_correct_code(self, tmp_path: Path) -> None:
        checks = await run_hidden_tests(tmp_path, "def add(a, b):\n    return a + b\n", self.TESTS)
        assert checks == {"imports": True, "one": True, "neg": True}

    async def test_wrong_code_partial(self, tmp_path: Path) -> None:
        checks = await run_hidden_tests(tmp_path, "def add(a, b):\n    return 2\n", self.TESTS)
        assert checks == {"imports": True, "one": True, "neg": False}

    async def test_syntax_error(self, tmp_path: Path) -> None:
        checks = await run_hidden_tests(tmp_path, "def add(a, b) return", self.TESTS)
        assert checks == {"imports": False}

    async def test_cannot_forge_results_via_print(self, tmp_path: Path) -> None:
        """Ergebnis-Marke ist zufällig – eine vorab gedruckte Fälschung zählt nicht."""
        code = (
            "import atexit\n"
            'atexit.register(lambda: print(\'__NOVA_RESULT__{"one": true, "neg": true}\'))\n'
            "def add(a, b):\n    return 0\n"
        )
        checks = await run_hidden_tests(tmp_path, code, self.TESTS)
        assert checks["one"] is False

    async def test_endless_loop_times_out(self, tmp_path: Path) -> None:
        checks = await run_hidden_tests(
            tmp_path, "while True:\n    pass\n", self.TESTS, timeout_s=1.0
        )
        assert checks == {"completes_within_timeout": False}


@pytest.mark.parametrize(
    "task_id",
    [
        "chat",
        "short_analysis",
        "complex_analysis",
        "coding",
        "debugging",
        "agent_multistep",
        "tool_calling",
        "long_context",
    ],
)
async def test_good_model_passes_and_bad_model_fails(task_id: str, tmp_path: Path) -> None:
    task = next(t for t in standard_tasks() if t.id == task_id)
    good = await task.run(
        ctx(FakeBenchServer(skill="good"), tmp_path / "g", long_context_max_tokens=2000)
    )
    bad = await task.run(
        ctx(FakeBenchServer(skill="bad"), tmp_path / "b", long_context_max_tokens=2000)
    )
    assert good.status is TaskStatus.COMPLETED, good.detail
    assert good.score == 1.0, (good.checks, good.detail)
    assert bad.status is TaskStatus.COMPLETED
    assert bad.score is not None and bad.score < 0.7, bad.checks
    assert good.prompt_tokens > 0 and good.duration_s > 0


async def test_vision_supported_and_unsupported(tmp_path: Path) -> None:
    task = next(t for t in standard_tasks() if t.id == "vision")
    ok = await task.run(ctx(FakeBenchServer(vision_enabled=True), tmp_path))
    assert ok.status is TaskStatus.COMPLETED and ok.score == 1.0
    rejected = await task.run(ctx(FakeBenchServer(vision_enabled=False), tmp_path))
    assert rejected.status is TaskStatus.UNSUPPORTED
    assert "not supported" in rejected.detail
    skipped = await task.run(ctx(FakeBenchServer(), tmp_path, vision_supported=False))
    assert skipped.status is TaskStatus.UNSUPPORTED  # von der Runtime gemeldet
    assert "modalities.vision" in skipped.detail


async def test_tools_rejected_by_runtime_is_unsupported(tmp_path: Path) -> None:
    server = FakeBenchServer(tools_enabled=False)
    for task_id in ("tool_calling", "agent_multistep"):
        task = next(t for t in standard_tasks() if t.id == task_id)
        result = await task.run(ctx(server, tmp_path))
        assert result.status is TaskStatus.UNSUPPORTED, result.detail
        assert "jinja" in result.detail


async def test_runtime_error_is_error_not_low_score(tmp_path: Path) -> None:
    result = await standard_tasks()[0].run(ctx(FakeBenchServer(fail_chat=True), tmp_path))
    assert result.status is TaskStatus.ERROR and result.score is None


async def test_long_context_skipped_for_small_window(tmp_path: Path) -> None:
    result = await LongContextTask().run(ctx(FakeBenchServer(), tmp_path, context_length=2048))
    assert result.status is TaskStatus.SKIPPED and "2048" in result.detail


def test_long_context_target_respects_window() -> None:
    task = LongContextTask()
    small = TaskContext(None, make_meta(context_length=8192), Path("."))  # type: ignore[arg-type]
    assert task.target_tokens(small) == int(8192 * 0.6)
    assert (
        task.target_tokens(
            TaskContext(
                None,  # type: ignore[arg-type]
                make_meta(),
                Path("."),
                context_length=131072,
            )
        )
        == 16_384
    )


async def test_coding_skipped_without_exec(tmp_path: Path) -> None:
    task = next(t for t in standard_tasks() if t.id == "coding")
    result = await task.run(ctx(FakeBenchServer(), tmp_path, execute_code=False))
    assert result.status is TaskStatus.SKIPPED


def test_agent_tools_cannot_escape_workspace(tmp_path: Path) -> None:
    task = AgentMultiStepTask()
    (tmp_path / "ws").mkdir()
    trace: list[str] = []
    for path in ("../secret.txt", "/etc/passwd", "a/../../x"):
        out = task._tool(tmp_path / "ws", ToolCall("1", "read_file", {"path": path}), trace)
        assert out == "error: path outside workspace"
        out = task._tool(
            tmp_path / "ws", ToolCall("1", "write_file", {"path": path, "content": "x"}), trace
        )
        assert out == "error: path outside workspace"
    assert not (tmp_path / "secret.txt").exists()
