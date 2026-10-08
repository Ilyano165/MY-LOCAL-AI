"""Architekturregel: Anwendungscode darf keine konkreten Modellnamen enthalten.

Modellnamen gehören ausschließlich in die Konfiguration (config/*.toml).
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = [
    "app",
    "core",
    "models",
    "agents",
    "tools",
    "memory",
    "rag",
    "router",
    "evaluation",
    "scripts",
]

# Bekannte Open-Weight-Modellfamilien. "llama.cpp" (Runtime) ist ausdrücklich erlaubt.
MODEL_NAME_RE = re.compile(
    r"\b(qwen\d*|llama[-_ ]?\d|gemma|mistral|mixtral|phi-?\d|deepseek|gpt-oss|devstral|codestral"
    r"|kimi|glm-?\d|minimax|granite|olmo|smollm|starcoder|codellama|yi-\d|internlm|nemotron)",
    re.IGNORECASE,
)


def test_no_model_names_in_source() -> None:
    offenders: list[str] = []
    for directory in SOURCE_DIRS:
        for path in (ROOT / directory).rglob("*.py"):
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if MODEL_NAME_RE.search(line):
                    offenders.append(f"{path.relative_to(ROOT)}:{lineno}: {line.strip()}")
    assert not offenders, "Hardcodierte Modellnamen gefunden:\n" + "\n".join(offenders)


def test_detector_catches_names() -> None:
    assert MODEL_NAME_RE.search('name = "Qwen3-8B"')
    assert MODEL_NAME_RE.search("llama-3.1")
    assert not MODEL_NAME_RE.search("llama.cpp llama-server")
