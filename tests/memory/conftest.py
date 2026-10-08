from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from memory import MemoryContext, MemoryManager


class FakeClock:
    """Steuerbare Uhr: Aktualität, Ablauf und Zerfall werden deterministisch testbar."""

    def __init__(self) -> None:
        self.now = datetime(2026, 10, 8, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **delta: float) -> None:
        self.now += timedelta(**delta)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def ctx() -> MemoryContext:
    return MemoryContext(session_id="sess-1", project_id="nova-abc12345", task_id="task-1")


@pytest.fixture
async def manager(clock: FakeClock):  # type: ignore[no-untyped-def]
    m = MemoryManager.in_memory(clock=clock)
    yield m
    await m.close()
