"""Health Check aller (oder eines) konfigurierten Modells.

Verwendung:
    python -m scripts.model_health --config ~/.nova/models.toml [--model NAME] [--json]

Exit-Code 0 = alle geprüften Modelle gesund, 1 = mindestens ein Fehler.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from models.health import CheckStatus, HealthCheckConfig, ModelHealthChecker
from models.inference import InferenceEngine

_SYMBOL = {CheckStatus.PASSED: "OK  ", CheckStatus.FAILED: "FAIL", CheckStatus.SKIPPED: "SKIP"}


async def _main(args: argparse.Namespace) -> int:
    engine = InferenceEngine.from_toml(args.config)
    config = HealthCheckConfig(timeout_s=args.timeout, max_response_s=args.max_response)
    try:
        checker = ModelHealthChecker(engine, config)
        reports = [await checker.check(args.model)] if args.model else await checker.check_all()
    finally:
        await engine.aclose()

    if args.json:
        print(json.dumps([r.to_dict() for r in reports], indent=2, ensure_ascii=False))
    else:
        for report in reports:
            print(f"\n{report.model}: {'GESUND' if report.healthy else 'FEHLERHAFT'}")
            for c in report.checks:
                print(f"  [{_SYMBOL[c.status]}] {c.name:<14} {c.duration_ms:8.0f} ms  {c.detail}")
    return 0 if all(r.healthy for r in reports) else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="NOVA Model Health Check")
    parser.add_argument("--config", required=True, help="Pfad zur Modellkonfiguration (TOML)")
    parser.add_argument("--model", help="nur dieses Registry-Modell prüfen")
    parser.add_argument("--timeout", type=float, default=120.0, help="Zeitlimit pro Check (s)")
    parser.add_argument("--max-response", type=float, default=30.0, help="max. Antwortzeit (s)")
    parser.add_argument("--json", action="store_true", help="Ausgabe als JSON")
    args = parser.parse_args()
    args.config = Path(args.config).expanduser()
    sys.exit(asyncio.run(_main(args)))


if __name__ == "__main__":
    main()
