"""Startet NOVA lokal: ``python -m api --config ~/.nova/models.toml [--dev] [--port 8765]``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from api.config import ApiConfig, default_data_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="NOVA – lokale Web-Oberfläche und API")
    parser.add_argument("--config", type=Path, help="Modellkonfiguration (TOML)")
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Development Mode: startet auch ohne Konfiguration/Modell",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--router", choices=["rules", "learned"], default="rules")
    parser.add_argument("--learned-ranker", type=Path)
    parser.add_argument(
        "--allow-remote",
        action="store_true",
        help="an andere Adressen als 127.0.0.1 binden (nicht empfohlen)",
    )
    args = parser.parse_args()
    if args.host not in ("127.0.0.1", "localhost", "::1") and not args.allow_remote:
        sys.exit("NOVA bindet nur an localhost. Andere Adressen nur mit --allow-remote.")
    try:
        config = ApiConfig(
            models_config=args.config,
            dev_mode=args.dev,
            data_dir=args.data_dir,
            router=args.router,
            learned_ranker=args.learned_ranker,
        )
    except ValueError as exc:
        sys.exit(f"Fehler: {exc}")
    import uvicorn

    from api.app import create_app

    print(
        f"NOVA läuft auf http://{args.host}:{args.port}"
        + ("  (Development Mode)" if args.dev else "")
    )
    uvicorn.run(create_app(config), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
