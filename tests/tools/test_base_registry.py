"""Basis, Registry, Policy, Bestätigung und Audit-Logging."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, ClassVar

import pytest

from tests.tools.conftest import Harness
from tools import Permission, Tool, ToolContext, ToolError, ToolInvocation, ToolRegistry, ToolResult
from tools.base import is_sensitive_path, safe_environment, validate_arguments
from tools.registry import Decision, JsonlAuditLog, MemoryAuditLog, PermissionPolicy, redact

# --------------------------------------------------------------------------- ToolResult


def test_tool_result_shape_and_invariants() -> None:
    ok = ToolResult.ok("hallo", path="a.txt")
    assert ok.to_dict() == {
        "success": True,
        "output": "hallo",
        "error": None,
        "metadata": {"path": "a.txt"},
    }
    fail = ToolResult.fail("kaputt", code=2)
    assert fail.to_dict() == {
        "success": False,
        "output": None,
        "error": "kaputt",
        "metadata": {"code": 2},
    }
    with pytest.raises(ValueError):
        ToolResult(True, "x", "fehler")
    with pytest.raises(ValueError):
        ToolResult(False, "x", "fehler")
    with pytest.raises(ValueError):
        ToolResult(False, None, "")


def test_invocation_from_model_separates_reason() -> None:
    inv = ToolInvocation.from_model(
        "read_file",
        {"path": "a", "reason": "Datei prüfen"},
        call_id="c1",
        task_id="t",
        subtask_id="s1",
    )
    assert inv.arguments == {"path": "a"} and inv.reason == "Datei prüfen"
    assert (inv.id, inv.task_id, inv.subtask_id) == ("c1", "t", "s1")


# --------------------------------------------------------------------------- Kontext & Pfade


def test_resolve_allows_workspace_and_extra_roots(tmp_path: Path) -> None:
    ws, extra = tmp_path / "ws", tmp_path / "extra"
    ws.mkdir()
    extra.mkdir()
    ctx = ToolContext(ws, extra_roots=(extra,))
    assert ctx.resolve("a/b.txt") == (ws / "a/b.txt").resolve()
    assert ctx.resolve(str(extra / "x")) == (extra / "x").resolve()
    for bad in ("../outside", "/etc/passwd", str(tmp_path)):
        with pytest.raises(ToolError, match="außerhalb"):
            ctx.resolve(bad)
    for bad in ("", "   ", "a\x00b"):
        with pytest.raises(ToolError):
            ctx.resolve(bad)


def test_resolve_blocks_symlink_escape(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    (tmp_path / "secret").mkdir()
    (ws / "link").symlink_to(tmp_path / "secret")
    with pytest.raises(ToolError, match="außerhalb"):
        ToolContext(ws).resolve("link/x")


@pytest.mark.parametrize(
    "path",
    [
        ".env",
        "config/.env.local",
        ".ssh/id_rsa",
        "keys/server.pem",
        "id_ed25519.pub",
        ".git-credentials",
        "deploy.key",
        ".aws/credentials",
    ],
)
def test_sensitive_paths_blocked(tmp_path: Path, path: str) -> None:
    assert is_sensitive_path(Path(path))
    with pytest.raises(ToolError, match="gesperrt"):
        ToolContext(tmp_path).resolve(path)


def test_env_example_and_normal_files_allowed(tmp_path: Path) -> None:
    for path in (".env.example", "src/environment.py", "keyboard.txt"):
        assert not is_sensitive_path(Path(path))
    ToolContext(tmp_path).resolve(".env.example")


def test_write_into_git_internals_blocked(tmp_path: Path) -> None:
    ctx = ToolContext(tmp_path)
    ctx.resolve(".git/config")  # Lesen erlaubt
    with pytest.raises(ToolError, match=r"\.git"):
        ctx.resolve(".git/config", write=True)


def test_safe_environment_strips_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NOVA_API_KEY", "geheim")
    monkeypatch.setenv("GITHUB_TOKEN", "geheim")
    monkeypatch.setenv("HARMLESS", "ok")
    env = safe_environment()
    assert "NOVA_API_KEY" not in env and "GITHUB_TOKEN" not in env
    assert env["HARMLESS"] == "ok" and env["PYTHONDONTWRITEBYTECODE"] == "1"


# --------------------------------------------------------------------------- Schema


def test_validate_arguments() -> None:
    schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "n": {"type": "integer", "minimum": 1, "maximum": 5},
            "mode": {"enum": ["a", "b"]},
            "cmd": {"type": ["array", "string"], "items": {"type": "string"}},
        },
        "required": ["path"],
        "additionalProperties": False,
    }
    assert validate_arguments(schema, {"path": "x", "n": 3, "mode": "a", "cmd": ["ls"]}) == []
    assert validate_arguments(schema, {"path": "x", "cmd": "ls -la"}) == []
    problems = validate_arguments(schema, {"n": True, "mode": "z", "cmd": [1], "extra": 1})
    assert "Pflichtargument fehlt: path" in problems
    assert any("n:" in p for p in problems)  # bool ist kein integer
    assert any("mode" in p for p in problems)
    assert any("cmd" in p for p in problems)
    assert "Unbekanntes Argument: extra" in problems
    assert any("Maximum" in p for p in validate_arguments(schema, {"path": "x", "n": 9}))
    assert validate_arguments(schema, "nope") == ["Argumente müssen ein JSON-Objekt sein"]  # type: ignore[arg-type]


# --------------------------------------------------------------------------- Registry-Pipeline


class Probe(Tool):
    name = "probe"
    description = "test"
    parameters: ClassVar[Mapping[str, Any]] = {
        "type": "object",
        "properties": {"how": {"type": "string"}},
        "required": ["how"],
    }
    permissions = frozenset({Permission.READ})

    def required_permissions(
        self, args: Mapping[str, Any], ctx: ToolContext
    ) -> frozenset[Permission]:
        if args["how"] == "forbidden":
            raise ToolError("grundsätzlich verboten")
        return (
            frozenset({Permission(args["how"])})
            if args["how"] in Permission._value2member_map_
            else self.permissions
        )

    async def run(self, args: Mapping[str, Any], ctx: ToolContext) -> ToolResult:
        match args["how"]:
            case "slow":
                await asyncio.sleep(2)
            case "tool_error":
                raise ToolError("fachlicher Fehler")
            case "crash":
                raise RuntimeError("bug")
            case "big":
                return ToolResult.ok("x" * 500)
            case "fail":
                return ToolResult.fail("selbst gemeldet", detail=1)
        return ToolResult.ok("fine", k=1)


def registry(**kwargs: Any) -> tuple[ToolRegistry, MemoryAuditLog]:
    audit = MemoryAuditLog()
    return ToolRegistry([Probe()], audit=audit, max_output_chars=100, **kwargs), audit


async def run_probe(
    reg: ToolRegistry, how: str, reason: str = "weil", tmp: Path = Path(".")
) -> ToolResult:
    return await reg.execute(
        ToolInvocation("probe", {"how": how}, reason=reason), ToolContext(tmp, timeout_s=0.2)
    )


async def test_success_has_structured_metadata(tmp_path: Path) -> None:
    reg, audit = registry()
    res = await run_probe(reg, "ok", tmp=tmp_path)
    assert res.success and res.output == "fine" and res.error is None
    meta = res.metadata
    assert meta["k"] == 1 and meta["tool"] == "probe" and meta["decision"] == "allow"
    assert meta["permissions"] == ["read"] and meta["duration_ms"] >= 0
    entry = audit.entries[-1]
    assert entry["tool"] == "probe" and entry["reason"] == "weil" and entry["success"] is True


@pytest.mark.parametrize(
    ("how", "fragment"),
    [
        ("slow", "Zeitlimit"),
        ("tool_error", "fachlicher Fehler"),
        ("crash", "RuntimeError"),
        ("fail", "selbst gemeldet"),
        ("forbidden", "grundsätzlich verboten"),
    ],
)
async def test_failures_are_structured_never_raised(
    tmp_path: Path, how: str, fragment: str
) -> None:
    reg, audit = registry()
    res = await run_probe(reg, how, tmp=tmp_path)
    assert not res.success and res.output is None and fragment in (res.error or "")
    assert audit.entries[-1]["success"] is False


async def test_missing_reason_unknown_tool_invalid_args(tmp_path: Path) -> None:
    reg, audit = registry()
    assert "Begründung fehlt" in (
        (await run_probe(reg, "ok", reason="  ", tmp=tmp_path)).error or ""
    )
    unknown = await reg.execute(ToolInvocation("nope", {}, reason="x"), ToolContext(tmp_path))
    assert "Unbekanntes Tool" in (unknown.error or "")
    invalid = await reg.execute(ToolInvocation("probe", {}, reason="x"), ToolContext(tmp_path))
    assert "Pflichtargument" in (invalid.error or "")
    assert len(audit.entries) == 3  # auch abgelehnte Aufrufe werden protokolliert


async def test_output_is_truncated(tmp_path: Path) -> None:
    reg, _ = registry()
    res = await run_probe(reg, "big", tmp=tmp_path)
    assert res.success and res.metadata["truncated"] and res.metadata["output_chars_total"] == 500
    assert len(res.output or "") < 200


@pytest.mark.parametrize(
    ("permission", "approve", "expected"),
    [
        ("network", None, "Berechtigungsrichtlinie"),
        ("execute", None, "kein Bestätigungskanal"),
        ("execute", False, "abgelehnt"),
        ("execute", True, None),
    ],
)
async def test_permission_decisions(
    tmp_path: Path, permission: str, approve: bool | None, expected: str | None
) -> None:
    asked: list[Any] = []

    async def approval(request: Any) -> bool:
        asked.append(request)
        return bool(approve)

    reg, audit = registry(approval=approval if approve is not None else None)
    res = await run_probe(reg, permission, tmp=tmp_path)
    if expected is None:
        assert res.success and res.metadata["decision"] == "allow"
        assert asked[0].invocation.reason == "weil" and asked[0].permissions == {Permission.EXECUTE}
    else:
        assert not res.success and expected in (res.error or "")
    assert audit.entries[-1]["permissions"] == [permission]


def test_policy_strictest_wins_and_denied_tools() -> None:
    policy = PermissionPolicy()
    assert policy.decide("x", [Permission.READ]) is Decision.ALLOW
    assert policy.decide("x", [Permission.READ, Permission.EXECUTE]) is Decision.ASK
    assert policy.decide("x", [Permission.EXECUTE, Permission.NETWORK]) is Decision.DENY
    assert PermissionPolicy(denied_tools=frozenset({"x"})).decide("x", []) is Decision.DENY
    read_only = PermissionPolicy.read_only()
    assert read_only.decide("x", [Permission.WRITE]) is Decision.DENY
    assert read_only.decide("x", [Permission.READ]) is Decision.ALLOW


def test_specs_inject_required_reason() -> None:
    reg, _ = registry()
    spec = reg.specs()[0]
    assert spec.parameters["required"] == ["how", "reason"]
    assert spec.parameters["properties"]["reason"]["type"] == "string"
    assert Probe().spec.parameters["required"] == ["how"]  # Original unverändert
    assert ToolRegistry([Probe()], require_reason=False).specs()[0].parameters["required"] == [
        "how"
    ]


def test_registry_rejects_duplicates() -> None:
    with pytest.raises(ValueError):
        ToolRegistry([Probe(), Probe()])
    with pytest.raises(KeyError):
        ToolRegistry().get("nope")


# --------------------------------------------------------------------------- Audit


def test_redaction() -> None:
    assert redact({"api_key": "abc", "path": "a"}) == {"api_key": "***", "path": "a"}
    assert redact("token sk-abcdefghijklmnopqrstuv end") == "token *** end"
    long = redact("x" * 2000)
    assert len(long) < 300 and "2000 Zeichen" in long and "sha256=" in long
    assert redact(["ghp_" + "a" * 30]) == ["***"]


async def test_jsonl_audit_log_persists_every_call(tmp_path: Path) -> None:
    log = JsonlAuditLog(tmp_path / "audit" / "tools.jsonl")
    reg = ToolRegistry([Probe()], audit=log)
    await reg.execute(ToolInvocation("probe", {"how": "ok"}, reason="erst"), ToolContext(tmp_path))
    await reg.execute(
        ToolInvocation("probe", {"how": "network"}, reason="dann"), ToolContext(tmp_path)
    )
    entries = log.read()
    assert [e["reason"] for e in entries] == ["erst", "dann"]
    assert [e["success"] for e in entries] == [True, False]
    assert entries[1]["decision"] == "deny"
    for line in (tmp_path / "audit" / "tools.jsonl").read_text().splitlines():
        json.loads(line)


async def test_audit_never_contains_file_content(harness: Harness) -> None:
    secret_body = "Zeile\n" * 500
    await harness.call("write_file", path="big.txt", content=secret_body)
    entry = harness.audit.entries[-1]
    assert len(json.dumps(entry)) < 2000
    assert "Zeichen, sha256=" in entry["arguments"]["content"]


MakeHarness = Callable[..., Harness]


def test_safe_environment_drops_git_config_group_consistently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "http.extraHeader")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "Authorization: Bearer geheim")
    env = safe_environment()
    assert not [k for k in env if k.startswith("GIT_CONFIG_")]
