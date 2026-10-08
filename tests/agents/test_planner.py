from __future__ import annotations

import pytest

from agents.planner import (
    HeuristicTaskAnalyzer,
    LLMPlanner,
    PlanningError,
    extract_json_object,
    validate_plan,
)
from agents.task import Complexity, Constraints, Subtask, SubtaskStatus, Task
from models.base import ProviderUnavailableError
from models.capabilities import TaskType
from tests.agents.fakes import ScriptedProvider, engine_for, plan_json

CHECKS = ["file_exists", "python_syntax", "command"]


@pytest.mark.parametrize(
    ("objective", "task_type", "complexity", "tools"),
    [
        ("Was ist 2+2?", TaskType.FAST, Complexity.SIMPLE, False),
        (
            "Schreibe eine Python-Funktion in utils.py und teste sie",
            TaskType.CODING,
            Complexity.MULTI_STEP,
            True,
        ),
        (
            "Analysiere die Vor- und Nachteile von SQLite gegenüber PostgreSQL "
            "für ein lokales Tool.",
            TaskType.REASONING,
            Complexity.SIMPLE,
            False,
        ),
        ("Beschreibe den Inhalt von screenshot.png", TaskType.VISION, Complexity.SIMPLE, False),
        (
            "Erstelle notes.md.\n- Abschnitt A\n- Abschnitt B",
            TaskType.GENERAL,
            Complexity.MULTI_STEP,
            True,
        ),
        (
            "Fasse zusammen, was ein Kompressor in der Tontechnik macht, in drei Sätzen bitte.",
            TaskType.GENERAL,
            Complexity.SIMPLE,
            False,
        ),
    ],
)
async def test_heuristic_analyzer(
    objective: str, task_type: TaskType, complexity: Complexity, tools: bool
) -> None:
    analysis = await HeuristicTaskAnalyzer().analyze(Task(objective))
    assert (analysis.task_type, analysis.complexity, analysis.requires_tools) == (
        task_type,
        complexity,
        tools,
    )


async def test_analyzer_records_signals_and_notes() -> None:
    task = Task("Fix the bug in app.py", Constraints(notes=["keine neuen Abhängigkeiten"]))
    analysis = await HeuristicTaskAnalyzer().analyze(task)
    assert any(s.startswith("coding") for s in analysis.signals)
    assert analysis.success_criteria == ["keine neuen Abhängigkeiten"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"a": 1}', {"a": 1}),
        ('Hier der Plan:\n```json\n{"a": {"b": "}"}}\n```\nFertig.', {"a": {"b": "}"}}),
        ('Erst {kaputt} dann {"ok": true}', {"ok": True}),
        ('text {"s": "escaped \\" quote"} end', {"s": 'escaped " quote'}),
    ],
)
def test_extract_json_object(text: str, expected: dict[str, object]) -> None:
    assert extract_json_object(text) == expected


@pytest.mark.parametrize("text", ["kein json", "[1, 2]", "{offen"])
def test_extract_json_object_fails(text: str) -> None:
    with pytest.raises(ValueError):
        extract_json_object(text)


def test_validate_plan_success() -> None:
    subtasks = validate_plan(
        {
            "subtasks": [
                {
                    "id": "a",
                    "description": "schreiben",
                    "verification": [{"type": "python_syntax", "path": "x.py"}],
                },
                {
                    "description": "testen",
                    "depends_on": ["a"],
                    "verification": [{"type": "command", "command": ["python", "-m", "pytest"]}],
                },
            ]
        },
        known_checks=CHECKS,
        max_subtasks=5,
    )
    assert [s.id for s in subtasks] == ["a", "s2"]
    assert subtasks[1].depends_on == ["a"]
    assert subtasks[1].verification[0].params == {"command": ["python", "-m", "pytest"]}


@pytest.mark.parametrize(
    ("plan", "message"),
    [
        ({}, "nicht-leere Liste"),
        ({"subtasks": [{"description": "x"}] * 3}, "Zu viele"),
        ({"subtasks": [{"id": "a"}]}, "description"),
        (
            {"subtasks": [{"id": "a", "description": "x"}, {"id": "a", "description": "y"}]},
            "Doppelte",
        ),
        (
            {
                "subtasks": [
                    {"id": "a", "description": "x", "depends_on": ["b"]},
                    {"id": "b", "description": "y"},
                ]
            },
            "unbekannte/spätere",
        ),
        (
            {"subtasks": [{"description": "x", "verification": [{"type": "rm_rf"}]}]},
            "unbekannter Verifikationstyp",
        ),
        ({"subtasks": ["text"]}, "kein Objekt"),
    ],
)
def test_validate_plan_errors(plan: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_plan(plan, known_checks=CHECKS, max_subtasks=2)


async def test_llm_planner_retries_after_invalid_plan() -> None:
    provider = ScriptedProvider(
        plan=[
            "Ich denke, wir sollten ... (kein JSON)",
            plan_json(
                {"id": "s1", "description": "schreiben"},
                {"id": "s2", "description": "prüfen", "depends_on": ["s1"]},
            ),
        ]
    )
    planner = LLMPlanner(engine_for(provider), known_checks=CHECKS, max_parse_retries=1)
    subtasks = await planner.plan(Task("Baue X"))
    assert [s.id for s in subtasks] == ["s1", "s2"]
    retry_request = provider.requests["plan"][1]
    assert "Invalid plan" in retry_request.messages[-1].content


async def test_llm_planner_gives_up() -> None:
    provider = ScriptedProvider(plan=["nein", "immer noch nein"])
    with pytest.raises(PlanningError, match="Kein gültiger Plan"):
        await LLMPlanner(engine_for(provider), known_checks=CHECKS).plan(Task("Baue X"))


async def test_llm_planner_model_unavailable() -> None:
    provider = ScriptedProvider(plan=[ProviderUnavailableError("down")])
    with pytest.raises(PlanningError, match="nicht verfügbar"):
        await LLMPlanner(engine_for(provider), known_checks=CHECKS).plan(Task("Baue X"))


async def test_replan_prefixes_ids_and_references_done_subtasks() -> None:
    task = Task("Baue X")
    task.subtasks = [
        Subtask("s1", "erledigt", status=SubtaskStatus.DONE, output="ok"),
        Subtask("s2", "kaputt", status=SubtaskStatus.IN_PROGRESS, attempts=3),
    ]
    task.record_error(task.phase, "tool", "write_file fehlgeschlagen", "s2", 3)
    provider = ScriptedProvider(
        plan=[
            plan_json(
                {"id": "a", "description": "anders", "depends_on": ["s1"]},
                {"id": "b", "description": "danach", "depends_on": ["a"]},
            )
        ]
    )
    new = await LLMPlanner(engine_for(provider), known_checks=CHECKS).replan(task, task.subtasks[1])
    assert [s.id for s in new] == ["r1-a", "r1-b"]
    assert new[0].depends_on == ["s1"] and new[1].depends_on == ["r1-a"]
    prompt = provider.requests["plan"][0].messages[1].content
    assert "write_file fehlgeschlagen" in prompt and "DIFFERENT approach" in prompt
