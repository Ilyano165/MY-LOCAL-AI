from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tools import ToolContext, ToolInvocation, ToolRegistry, ToolResult, default_tools
from tools.registry import ApprovalRequest, MemoryAuditLog, PermissionPolicy


class Harness:
    """Registry + Kontext + Audit für Tool-Tests (echte Tools, echtes Dateisystem)."""

    def __init__(self, workspace: Path, **registry_kwargs: Any) -> None:
        self.workspace = workspace
        self.audit = MemoryAuditLog()
        self.approvals: list[ApprovalRequest] = []
        self.registry = ToolRegistry(default_tools(), audit=self.audit, **registry_kwargs)
        self.ctx = ToolContext(workspace=workspace, timeout_s=30)

    async def call(
        self, tool: str, reason: str = "Test", ctx: ToolContext | None = None, **arguments: Any
    ) -> ToolResult:
        return await self.registry.execute(
            ToolInvocation(tool, arguments, reason=reason), ctx or self.ctx
        )


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    return workspace


@pytest.fixture
def harness(ws: Path) -> Harness:
    return Harness(ws)


@pytest.fixture
def make_harness(ws: Path) -> Callable[..., Harness]:
    def factory(approve: bool | None = None, policy: PermissionPolicy | None = None) -> Harness:
        h: Harness

        async def approval(request: ApprovalRequest) -> bool:
            h.approvals.append(request)
            return bool(approve)

        h = Harness(ws, policy=policy, approval=approval if approve is not None else None)
        return h

    return factory
