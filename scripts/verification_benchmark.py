"""Benchmark der Verification Engine.

Verwendung:
    python -m scripts.verification_benchmark [--cases eigene.jsonl] [--out bericht.md]

Exit-Code 1, wenn mindestens ein Fehler durchgelassen wurde (False Accept).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from evaluation.benchmarks import BenchmarkRunner, builtin_cases, load_cases


def main() -> None:
    parser = argparse.ArgumentParser(description="NOVA Verification-Benchmark")
    parser.add_argument("--cases", help="JSONL-Datei mit eigenen Fällen (Default: eingebaute)")
    parser.add_argument("--out", help="Markdown-Bericht in diese Datei schreiben")
    args = parser.parse_args()
    cases = load_cases(args.cases) if args.cases else builtin_cases()
    result = asyncio.run(BenchmarkRunner().run(cases))
    report = result.to_markdown()
    if args.out:
        Path(args.out).write_text(report + "\n", encoding="utf-8")
    print(report)
    sys.exit(1 if result.false_accepts else 0)


if __name__ == "__main__":
    main()
