"""git_status, git_diff, git_log – inkl. Schutz gegen ausführbare Repo-Konfiguration."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests.tools.conftest import Harness
from tools import ToolContext
from tools.git import dangerous_config, parse_config_z


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        env={
            "GIT_AUTHOR_NAME": "Nova Test",
            "GIT_AUTHOR_EMAIL": "t@example.com",
            "GIT_COMMITTER_NAME": "Nova Test",
            "GIT_COMMITTER_EMAIL": "t@example.com",
            "HOME": str(repo.parent),
            "PATH": "/usr/bin:/bin",
        },
    ).stdout


@pytest.fixture
def repo(ws: Path) -> Path:
    git(ws, "init", "-q", "-b", "main")
    (ws / "app.py").write_text("print('v1')\n")
    (ws / "README.md").write_text("# Projekt\n")
    git(ws, "add", ".")
    git(ws, "commit", "-q", "-m", "Erster Commit")
    (ws / "app.py").write_text("print('v2')\n")
    git(ws, "commit", "-q", "-am", "Zweiter Commit")
    return ws


async def test_git_status(harness: Harness, repo: Path) -> None:
    clean = await harness.call("git_status")
    assert clean.success and clean.metadata["clean"] is True and clean.metadata["branch"] == "main"

    (repo / "app.py").write_text("print('v3')\n")
    (repo / "neu.txt").write_text("x")
    (repo / "README.md").write_text("# Geändert\n")
    git(repo, "add", "README.md")
    status = await harness.call("git_status")
    meta = status.metadata
    assert meta["staged"] == ["M README.md"]
    assert meta["unstaged"] == ["M app.py"]
    assert meta["untracked"] == ["neu.txt"]
    assert "Staged (1)" in (status.output or "") and "Untracked (1)" in (status.output or "")


async def test_git_status_handles_renames_and_spaces(harness: Harness, repo: Path) -> None:
    git(repo, "mv", "README.md", "LIES MICH.md")
    meta = (await harness.call("git_status")).metadata
    assert meta["staged"] == ["R LIES MICH.md"]


async def test_git_diff(harness: Harness, repo: Path) -> None:
    empty = await harness.call("git_diff")
    assert (
        empty.success
        and empty.output == "Keine Änderungen"
        and empty.metadata["files_changed"] == 0
    )

    (repo / "app.py").write_text("print('v3')\nprint('mehr')\n")
    diff = await harness.call("git_diff")
    assert "-print('v2')" in (diff.output or "") and "+print('mehr')" in (diff.output or "")
    assert diff.metadata["files"] == [{"file": "app.py", "added": "2", "removed": "1"}]

    stat = await harness.call("git_diff", stat=True)
    assert "1 file changed" in (stat.output or "")
    assert (await harness.call("git_diff", staged=True)).output == "Keine Änderungen"
    git(repo, "add", "app.py")
    assert "+print('mehr')" in ((await harness.call("git_diff", staged=True)).output or "")

    (repo / "README.md").write_text("anders\n")
    only = await harness.call("git_diff", files=["README.md"])
    assert "README.md" in (only.output or "") and "app.py" not in (only.output or "")


async def test_git_log(harness: Harness, repo: Path) -> None:
    log = await harness.call("git_log")
    commits = log.metadata["commits"]
    assert [c["subject"] for c in commits] == ["Zweiter Commit", "Erster Commit"]
    assert commits[0]["author"] == "Nova Test" and len(commits[0]["hash"]) == 40
    one = await harness.call("git_log", max_count=1)
    assert len(one.metadata["commits"]) == 1
    readme = await harness.call("git_log", file="README.md")
    assert [c["subject"] for c in readme.metadata["commits"]] == ["Erster Commit"]


async def test_git_log_empty_repository(harness: Harness, ws: Path) -> None:
    git(ws, "init", "-q")
    res = await harness.call("git_log")
    assert res.success and res.output == "Noch keine Commits"


async def test_not_a_repository(harness: Harness) -> None:
    for tool in ("git_status", "git_diff", "git_log"):
        res = await harness.call(tool)
        assert not res.success and "Kein Git-Repository" in (res.error or "")


async def test_repository_root_outside_allowed_dirs(tmp_path: Path) -> None:
    git(tmp_path, "init", "-q")
    inner = tmp_path / "inner"
    inner.mkdir()
    harness = Harness(inner)
    res = await harness.call("git_status")
    assert not res.success and "außerhalb der erlaubten" in (res.error or "")


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("core.fsmonitor", "touch {marker}"),
        ("filter.evil.clean", "touch {marker}"),
        ("diff.external", "touch {marker}"),
        ("core.pager", "touch {marker}"),
    ],
)
async def test_malicious_repo_config_is_refused_and_never_executed(
    harness: Harness, repo: Path, key: str, value: str
) -> None:
    marker = repo.parent / "PWNED"
    git(repo, "config", key, value.format(marker=marker))
    if key.startswith("filter."):
        (repo / ".gitattributes").write_text("*.py filter=evil\n")
    (repo / "app.py").write_text("print('geändert')\n")
    for tool in ("git_status", "git_diff", "git_log"):
        res = await harness.call(tool)
        assert not res.success and "Befehle ausführen" in (res.error or ""), tool
        assert key in (res.error or "")
    assert not marker.exists()


async def test_fsmonitor_boolean_is_harmless(harness: Harness, repo: Path) -> None:
    git(repo, "config", "core.fsmonitor", "false")
    assert (await harness.call("git_status")).success


def test_config_parsing_and_detection() -> None:
    raw = (
        "local\0filter.lfs.clean\ngit-lfs clean -- %f\0"
        "global\0core.pager\nless\0local\0user.name\nx\0"
    )
    entries = parse_config_z(raw)
    assert entries == [
        ("local", "filter.lfs.clean", "git-lfs clean -- %f"),
        ("global", "core.pager", "less"),
        ("local", "user.name", "x"),
    ]
    # Nur repo-lokale Einträge zählen; globale Nutzerkonfiguration ist vertrauenswürdig.
    assert dangerous_config(entries) == ["filter.lfs.clean"]
    assert dangerous_config([("local", "core.fsmonitor", "true")]) == []
    assert dangerous_config([("worktree", "diff.pdf.textconv", "pdftotext")]) == [
        "diff.pdf.textconv"
    ]


async def test_git_tools_are_read_only_permission(harness: Harness, repo: Path) -> None:
    res = await harness.call("git_status")
    assert res.metadata["permissions"] == ["read"]


async def test_pathspec_outside_repo_rejected(harness: Harness, repo: Path, tmp_path: Path) -> None:
    other = tmp_path / "other"
    other.mkdir()
    ctx = ToolContext(repo, extra_roots=(other,))
    res = await harness.call("git_diff", ctx=ctx, files=[str(other / "x")])
    assert not res.success and "nicht im Repository" in (res.error or "")
