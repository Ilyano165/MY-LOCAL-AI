"""``nova`` – Kommandozeile für Betrieb, Dienst, Integrationen und Daten.

    nova serve [--config F] [--dev] [--port 8765]   Server im Vordergrund
    nova service start|stop|status [--port 8765]    Hintergrunddienst (Benutzerprozess)
    nova open                                       Dienst bei Bedarf starten, UI öffnen
    nova integrations create|list|revoke|rotate|scopes …
    nova models catalog|plan|download|list …
    nova data path | nova data purge
    nova --version

Ohne ``--config`` nutzt NOVA ``<Datenverzeichnis>/models.toml``; fehlt sie, startet NOVA im
**Setup-Modus** (Oberfläche erreichbar, „No local model available.“, Modell-Einrichtung).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import logging.handlers
import os
import secrets
import shutil
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from typing import Any

import httpx

from api.config import ApiConfig, default_data_dir
from api.integrations import SCOPES, TOOL_SCOPE_PREFIX, IntegrationError, IntegrationStore
from api.paths import DataLayout
from api.version import __version__

LOOPBACK = ("127.0.0.1", "localhost", "::1")
DEFAULT_PORT = 8765


def _print_json(data: Any) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------- serve


def _setup_logging(log_file: Path | None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(
            logging.handlers.RotatingFileHandler(
                log_file, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
            )
        )
    logging.basicConfig(
        level=logging.INFO,
        handlers=handlers,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def build_config(args: argparse.Namespace) -> tuple[ApiConfig, str]:
    """Konfiguration für ``serve``; liefert (Config, Modus: normal|setup|development)."""
    layout = DataLayout(args.data_dir)
    config_path: Path | None = args.config
    mode = "development" if args.dev else "normal"
    if config_path is None and layout.models_config.is_file():
        config_path = layout.models_config
    if config_path is None and not args.dev:
        mode = "setup"  # installiert, aber noch kein Modell eingerichtet
    remote = args.host not in LOOPBACK
    if remote:
        problems = []
        if not args.allow_remote:
            problems.append("--allow-remote")
        if not (args.tls_cert and args.tls_key):
            problems.append("--tls-cert/--tls-key (TLS is mandatory for network access)")
        if not os.environ.get("NOVA_API_TOKEN"):
            problems.append("NOVA_API_TOKEN (protects the web UI on the network)")
        if problems:
            raise SystemExit("Network access requires: " + ", ".join(problems))
    config = ApiConfig(
        models_config=config_path,
        dev_mode=mode != "normal",
        data_dir=args.data_dir,
        router=args.router,
        learned_ranker=args.learned_ranker,
        allow_remote=remote,
        cors_origins=tuple(args.cors_origin or ()),
        mode=mode,
    )
    return config, mode


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from api.app import create_app

    layout = DataLayout(args.data_dir).ensure()
    _setup_logging(args.log_file or layout.logs / "nova.log")
    try:
        config, mode = build_config(args)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    app = create_app(config)
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host=args.host,
            port=args.port,
            log_level="info",
            log_config=None,
            ssl_certfile=args.tls_cert,
            ssl_keyfile=args.tls_key,
        )
    )
    app.state.server = server
    scheme = "https" if args.tls_cert else "http"
    logging.getLogger("nova").info(
        "NOVA %s on %s://%s:%s (mode: %s)", __version__, scheme, args.host, args.port, mode
    )
    server.run()
    return 0


# ---------------------------------------------------------------------- Dienst


def _health(port: int, timeout: float = 2.0) -> dict[str, Any] | None:
    try:
        r = httpx.get(f"http://127.0.0.1:{port}/api/v1/health", timeout=timeout)
        return r.json() if r.status_code == 200 else None
    except (httpx.HTTPError, ValueError):
        return None


def _server_command() -> list[str]:
    if getattr(sys, "frozen", False):  # PyInstaller: immer das Konsolenprogramm nova(.exe)
        exe = Path(sys.executable)
        console = exe.with_name("nova" + exe.suffix)
        if not console.is_file():
            raise SystemExit(f"{console.name} not found next to {exe.name} – reinstall NOVA")
        return [str(console)]
    return [sys.executable, "-m", "api"]


def _read_pid(layout: DataLayout) -> dict[str, Any] | None:
    try:
        data: dict[str, Any] = json.loads(layout.service_pid.read_text(encoding="utf-8"))
        return data
    except (OSError, ValueError):
        return None


def service_start(layout: DataLayout, port: int, wait_s: float = 45.0) -> dict[str, Any]:
    if (health := _health(port)) is not None:
        return {
            "status": "running",
            "port": port,
            "version": health.get("version"),
            "started": False,
        }
    layout.ensure()
    token = secrets.token_urlsafe(24)
    env = {**os.environ, "NOVA_SERVICE_TOKEN": token, "NOVA_DATA_DIR": str(layout.root)}
    command = [*_server_command(), "--data-dir", str(layout.root), "serve", "--port", str(port)]
    log = (layout.logs / "service.log").open("ab")
    kwargs: dict[str, Any] = {
        "stdout": log,
        "stderr": subprocess.STDOUT,
        "stdin": subprocess.DEVNULL,
        "env": env,
        "close_fds": True,
    }
    if sys.platform == "win32":
        kwargs["creationflags"] = (
            subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.CREATE_NO_WINDOW
        )
    else:
        kwargs["start_new_session"] = True
    process = subprocess.Popen(command, **kwargs)
    layout.service_pid.write_text(
        json.dumps({"pid": process.pid, "port": port, "token": token, "started_at": time.time()}),
        encoding="utf-8",
    )
    deadline = time.time() + wait_s
    while time.time() < deadline:
        if process.poll() is not None:
            raise SystemExit(
                f"NOVA service exited with code {process.returncode} – see "
                f"{layout.logs / 'service.log'}"
            )
        if (health := _health(port, timeout=1.0)) is not None:
            return {
                "status": "running",
                "port": port,
                "pid": process.pid,
                "version": health.get("version"),
                "started": True,
            }
        time.sleep(0.3)
    raise SystemExit(
        f"NOVA service did not become ready within {wait_s:.0f} s – see "
        f"{layout.logs / 'service.log'}"
    )


def service_stop(layout: DataLayout, port: int, wait_s: float = 15.0) -> dict[str, Any]:
    info = _read_pid(layout)
    if info is None and _health(port) is None:
        return {"status": "stopped", "stopped": False}
    if info is not None:
        with contextlib.suppress(httpx.HTTPError):
            httpx.post(
                f"http://127.0.0.1:{info.get('port', port)}/internal/shutdown",
                headers={"X-NOVA-Shutdown-Token": str(info.get("token", ""))},
                timeout=5,
            )
    deadline = time.time() + wait_s
    while time.time() < deadline and _health(port, timeout=0.5) is not None:
        time.sleep(0.2)
    if _health(port, timeout=0.5) is not None and info is not None:
        with contextlib.suppress(OSError):
            os.kill(int(info["pid"]), 15)  # Windows: TerminateProcess; POSIX: SIGTERM
        time.sleep(1)
    running = _health(port, timeout=0.5) is not None
    if not running:
        with contextlib.suppress(OSError):
            layout.service_pid.unlink()
    return {"status": "running" if running else "stopped", "stopped": not running}


def service_status(layout: DataLayout, port: int) -> dict[str, Any]:
    health = _health(port)
    info = _read_pid(layout) or {}
    return {
        "status": "running" if health else "stopped",
        "port": port,
        "version": health.get("version") if health else None,
        "pid": info.get("pid"),
        "url": f"http://127.0.0.1:{port}/",
        "data_dir": str(layout.root),
        "log": str(layout.logs / "service.log"),
    }


def cmd_service(args: argparse.Namespace) -> int:
    layout = DataLayout(args.data_dir)
    if args.action == "start":
        _print_json(service_start(layout, args.port))
    elif args.action == "stop":
        _print_json(service_stop(layout, args.port))
    else:
        status = service_status(layout, args.port)
        _print_json(status)
        return 0 if status["status"] == "running" else 3
    return 0


def cmd_open(args: argparse.Namespace) -> int:
    layout = DataLayout(args.data_dir)
    service_start(layout, args.port)
    url = f"http://127.0.0.1:{args.port}/"
    if not args.no_browser:
        webbrowser.open(url)
    print(url)
    return 0


# ---------------------------------------------------------------------- Integrationen


def _store(args: argparse.Namespace) -> IntegrationStore:
    return IntegrationStore(DataLayout(args.data_dir).integrations_db)


def cmd_integrations(args: argparse.Namespace) -> int:
    from tools import default_tools

    store = _store(args)
    tools = {t.name for t in default_tools()}
    try:
        if args.action == "create":
            integration, key = store.create(
                args.name,
                args.scope or [],
                rate_limit_per_minute=args.rate_limit,
                max_request_bytes=args.max_request_bytes,
                allowed_models=args.model,
                agent_workspace=args.agent_workspace,
                expires_at=args.expires_at,
                known_tools=tools,
            )
            _print_json({**integration.to_dict(), "api_key": key})
            print(
                "\nStore this key now – NOVA keeps only a hash and cannot show it again.",
                file=sys.stderr,
            )
        elif args.action == "list":
            _print_json([i.to_dict() for i in store.list_all()])
        elif args.action == "revoke":
            _print_json(store.revoke(args.id).to_dict())
        elif args.action == "rotate":
            integration, key = store.rotate(args.id)
            _print_json({**integration.to_dict(), "api_key": key})
        elif args.action == "scopes":
            _print_json(store.update_scopes(args.id, args.scope or [], tools).to_dict())
        elif args.action == "list-scopes":
            _print_json(
                {
                    **SCOPES,
                    f"{TOOL_SCOPE_PREFIX}<tool>": "privileged: allow one agent "
                    "tool, tools: " + ", ".join(sorted(tools)),
                }
            )
    except IntegrationError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    finally:
        store.close()
    return 0


# ---------------------------------------------------------------------- Modelle


# ---------------------------------------------------------------------- Research


def _research_service(args: argparse.Namespace) -> Any:
    """NovaService mit derselben Modellkonfiguration wie ``serve`` (Setup-Modus ohne Modell)."""
    from api.service import NovaService

    layout = DataLayout(args.data_dir)
    config_path = layout.models_config if layout.models_config.is_file() else None
    config = ApiConfig(
        models_config=config_path,
        dev_mode=config_path is None,
        data_dir=args.data_dir,
        mode="normal" if config_path else "setup",
    )
    return NovaService(config)


async def _run_research_foreground(svc: Any, run_id: str) -> dict[str, Any]:
    """Wartet auf den Lauf; Strg+C pausiert ihn (fortsetzbar), statt Daten zu verlieren."""
    from research.manager import summary

    manager = svc.research
    last = ""
    try:
        while run_id in manager._active:
            info = summary(manager._active[run_id].state)
            c = info["counts"]
            line = (
                f"{info['status']}: {c['searches']} searches, {c['sources']} sources, "
                f"{c['claims']} statements, {c['errors']} errors, {info['active_seconds']:.0f} s"
            )
            if line != last:
                print(line, flush=True)
                last = line
            await asyncio.sleep(1)
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("Pausing – resume later with: nova research resume " + run_id, flush=True)
    finally:
        await svc.shutdown()
    data: dict[str, Any] = manager.get(run_id)
    return data


def cmd_research(args: argparse.Namespace) -> int:
    from research.engine import Budget
    from research.manager import ResearchError

    if args.action == "config":
        svc = _research_service(args)
        try:
            if args.provider:
                search = {"provider": args.provider}
                if args.url:
                    search["url"] = args.url
                if args.api_key_env:
                    search["api_key_env"] = args.api_key_env
                svc.research.set_config(search)
            _print_json({**svc.research.status(), "config": svc.research.config()})
        except ResearchError as exc:
            raise SystemExit(exc.message) from exc
        finally:
            asyncio.run(svc.shutdown())
        return 0
    if args.action == "list":
        svc = _research_service(args)
        _print_json(svc.research.list_runs())
        asyncio.run(svc.shutdown())
        return 0
    if args.action in ("show", "report"):
        if not args.target:
            raise SystemExit("research run id required")
        svc = _research_service(args)
        try:
            if args.action == "show":
                _print_json(svc.research.get(args.target))
            else:
                print(svc.research.report(args.target)["markdown"])
        except ResearchError as exc:
            raise SystemExit(exc.message) from exc
        finally:
            asyncio.run(svc.shutdown())
        return 0
    if args.action in ("start", "resume"):
        if not args.target:
            raise SystemExit("objective (start) or run id (resume) required")

        async def go() -> dict[str, Any]:
            svc = _research_service(args)
            try:
                if args.action == "start":
                    budget = Budget(
                        max_duration_s=args.minutes * 60,
                        max_sources=args.max_sources,
                        max_searches=args.max_searches,
                        max_fetches=max(args.max_sources * 2, 10),
                    )
                    started = await svc.research.start(args.target, budget, args.url or [])
                else:
                    started = await svc.research.resume(args.target)
            except ResearchError as exc:
                await svc.shutdown()
                raise SystemExit(exc.message) from exc
            print(f"Research run {started['id']} started", flush=True)
            return await _run_research_foreground(svc, started["id"])

        result = asyncio.run(go())
        print(f"Finished with status {result['status']}: {result['stop_reason']}")
        print(f"Report: {DataLayout(args.data_dir).research / result['id'] / 'report.md'}")
        return 0 if result["status"] in ("completed", "budget_exhausted") else 1
    if args.action == "knowledge":
        svc = _research_service(args)
        _print_json(svc.research.knowledge.search(args.target or ""))
        asyncio.run(svc.shutdown())
        return 0
    return 2


def cmd_models(args: argparse.Namespace) -> int:
    import asyncio

    from api.model_manager import CatalogError, ModelManager

    manager = ModelManager(DataLayout(args.data_dir))
    try:
        if args.action == "catalog":
            _print_json([e.to_dict() for e in manager.catalog()])
        elif args.action == "list":
            _print_json(manager.installed())
        elif args.action == "plan":
            _print_json(manager.plan(manager.entry(args.id)).to_dict())
        elif args.action == "download":
            entry = manager.entry(args.id)
            plan = manager.plan(entry)
            _print_json(plan.to_dict())
            if not plan.ok:
                return 2
            if entry.requires_license_acceptance and not args.accept_license:
                print(
                    f"License: {entry.license} – {entry.license_url}\n"
                    "Re-run with --accept-license after reading the terms.",
                    file=sys.stderr,
                )
                return 2

            def progress(done: int, total: int) -> None:
                pct = f"{done / total:6.1%}" if total else ""
                print(f"\r{done / 1e9:7.2f} / {total / 1e9:.2f} GB {pct}", end="", file=sys.stderr)

            result = asyncio.run(
                manager.download(entry, accept_license=args.accept_license, progress=progress)
            )
            print(file=sys.stderr)
            _print_json(result)
    except CatalogError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


# ---------------------------------------------------------------------- Daten


def cmd_data(args: argparse.Namespace) -> int:
    layout = DataLayout(args.data_dir)
    if args.action == "path":
        print(layout.root)
        return 0
    if not layout.root.exists():
        print(f"Nothing to delete: {layout.root} does not exist")
        return 0
    if service_status(layout, args.port)["status"] == "running":
        print("Stop the NOVA service first: nova service stop", file=sys.stderr)
        return 2
    print(
        f"This permanently deletes ALL NOVA user data in:\n  {layout.root}\n"
        "(chat history, memory, settings, logs, integration keys, downloaded models)"
    )
    if args.confirm != str(layout.root):
        answer = input("Type DELETE to confirm: ") if sys.stdin.isatty() else ""
        if answer.strip() != "DELETE":
            print("Aborted – nothing was deleted.")
            return 1
    shutil.rmtree(layout.root)
    print("Deleted.")
    return 0


# ---------------------------------------------------------------------- Parser


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="nova", description="NOVA – local AI platform")
    p.add_argument("--version", action="version", version=f"NOVA {__version__}")
    p.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir(),
        help="user data directory (default: platform-specific, NOVA_DATA_DIR)",
    )
    sub = p.add_subparsers(dest="command")

    serve = sub.add_parser("serve", help="run the NOVA server in the foreground")
    serve.add_argument("--config", type=Path, help="model configuration (TOML)")
    serve.add_argument("--dev", action="store_true", help="development mode (no model needed)")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=DEFAULT_PORT)
    serve.add_argument("--router", choices=["rules", "learned"], default="rules")
    serve.add_argument("--learned-ranker", type=Path)
    serve.add_argument("--allow-remote", action="store_true")
    serve.add_argument("--tls-cert")
    serve.add_argument("--tls-key")
    serve.add_argument(
        "--cors-origin",
        action="append",
        help="browser origin allowed to call /api/v1 and /v1 (repeatable)",
    )
    serve.add_argument("--log-file", type=Path)
    serve.set_defaults(func=cmd_serve)

    service = sub.add_parser("service", help="background service (start/stop/status)")
    service.add_argument("action", choices=["start", "stop", "status"])
    service.add_argument("--port", type=int, default=DEFAULT_PORT)
    service.set_defaults(func=cmd_service)

    open_ = sub.add_parser("open", help="start the service if needed and open the UI")
    open_.add_argument("--port", type=int, default=DEFAULT_PORT)
    open_.add_argument("--no-browser", action="store_true")
    open_.set_defaults(func=cmd_open)

    integ = sub.add_parser("integrations", help="manage integration API keys")
    integ.add_argument(
        "action", choices=["create", "list", "revoke", "rotate", "scopes", "list-scopes"]
    )
    integ.add_argument("id", nargs="?", help="integration id or name (revoke/rotate/scopes)")
    integ.add_argument("--name")
    integ.add_argument("--scope", action="append")
    integ.add_argument("--model", action="append", help="allowed model id (repeatable)")
    integ.add_argument("--rate-limit", type=int, default=60)
    integ.add_argument("--max-request-bytes", type=int, default=1_000_000)
    integ.add_argument("--agent-workspace")
    integ.add_argument("--expires-at", help="ISO timestamp")
    integ.set_defaults(func=cmd_integrations)

    models = sub.add_parser("models", help="model catalog and downloads")
    models.add_argument("action", choices=["catalog", "list", "plan", "download"])
    models.add_argument("id", nargs="?")
    models.add_argument("--accept-license", action="store_true")
    models.set_defaults(func=cmd_models)

    research = sub.add_parser("research", help="web research runs and knowledge")
    research.add_argument(
        "action", choices=["start", "resume", "list", "show", "report", "config", "knowledge"]
    )
    research.add_argument("target", nargs="?", help="objective (start), run id, or query")
    research.add_argument("--minutes", type=float, default=60)
    research.add_argument("--max-sources", type=int, default=30)
    research.add_argument("--max-searches", type=int, default=20)
    research.add_argument("--url", action="append", help="seed URL (repeatable)")
    research.add_argument("--provider", choices=["none", "searxng", "brave"])
    research.add_argument("--api-key-env", help="environment variable holding the API key")
    research.set_defaults(func=cmd_research)

    data = sub.add_parser("data", help="user data")
    data.add_argument("action", choices=["path", "purge"])
    data.add_argument("--confirm", help="non-interactive: repeat the exact data path")
    data.add_argument("--port", type=int, default=DEFAULT_PORT)
    data.set_defaults(func=cmd_data)
    return p


LEGACY_FLAGS = {
    "--config",
    "--dev",
    "--host",
    "--port",
    "--router",
    "--learned-ranker",
    "--allow-remote",
}


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Abwärtskompatibel: ``python -m api --dev`` == ``nova serve --dev``
    if not argv or argv[0] in LEGACY_FLAGS:
        argv = ["serve", *argv]
    args = parser().parse_args(argv)
    if (
        args.command in ("integrations",)
        and args.action in ("revoke", "rotate", "scopes")
        and not args.id
    ):
        raise SystemExit("integration id or name required")
    if args.command == "integrations" and args.action == "create" and not args.name:
        raise SystemExit("--name is required")
    if args.command == "models" and args.action in ("plan", "download") and not args.id:
        raise SystemExit("model id required")
    sys.exit(args.func(args))
