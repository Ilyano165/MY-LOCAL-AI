"""NOVA Tool-System: kontrollierter Zugriff auf Dateisystem, Terminal und Git."""

from tools.base import (
    Permission,
    Tool,
    ToolContext,
    ToolError,
    ToolInvocation,
    ToolResult,
    safe_environment,
    sha256_text,
    validate_arguments,
)
from tools.filesystem import (
    EditFileTool,
    ListDirectoryTool,
    ReadFileTool,
    SearchFilesTool,
    WriteFileTool,
    filesystem_tools,
)
from tools.git import GitDiffTool, GitLogTool, GitStatusTool, git_tools
from tools.registry import (
    ApprovalHandler,
    ApprovalRequest,
    Decision,
    JsonlAuditLog,
    MemoryAuditLog,
    PermissionPolicy,
    ToolRegistry,
)
from tools.terminal import DEFAULT_ALLOWLIST, ExecuteCommandTool


def default_tools() -> list[Tool]:
    """Alle Standard-Tools: Dateisystem, Terminal, Git."""
    return [*filesystem_tools(), ExecuteCommandTool(), *git_tools()]


__all__ = [
    "DEFAULT_ALLOWLIST",
    "ApprovalHandler",
    "ApprovalRequest",
    "Decision",
    "EditFileTool",
    "ExecuteCommandTool",
    "GitDiffTool",
    "GitLogTool",
    "GitStatusTool",
    "JsonlAuditLog",
    "ListDirectoryTool",
    "MemoryAuditLog",
    "Permission",
    "PermissionPolicy",
    "ReadFileTool",
    "SearchFilesTool",
    "Tool",
    "ToolContext",
    "ToolError",
    "ToolInvocation",
    "ToolRegistry",
    "ToolResult",
    "WriteFileTool",
    "default_tools",
    "filesystem_tools",
    "git_tools",
    "safe_environment",
    "sha256_text",
    "validate_arguments",
]
