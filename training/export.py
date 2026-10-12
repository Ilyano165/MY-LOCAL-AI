"""Export: trainierten LoRA-Adapter nach GGUF umwandeln (läuft in der Trainingsumgebung).

    python export.py job.json <llama.cpp-quellordner>

Nutzt ``convert_lora_to_gguf.py`` und ``gguf-py`` aus **demselben** llama.cpp-Release wie die
installierte Runtime (Skripte und gguf-Paket müssen zusammenpassen). Ergebnis:
``adapter.gguf`` im Job-Ordner, das ``llama-server --lora`` lädt.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def base_config_dir(job: dict[str, Any]) -> str:
    """Ordner mit config.json/Tokenizer des Basismodells (aus dem Hugging-Face-Cache)."""
    if job.get("local_model_dir"):
        return str(job["local_model_dir"])
    from huggingface_hub import snapshot_download

    return str(
        snapshot_download(
            job["base_model"],
            revision=job["base_revision"],
            allow_patterns=["*.json", "*.model", "*.txt", "*.jinja"],
        )
    )


def main(job_path: str, llama_src: str) -> int:
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    out = Path(job["output_dir"])
    src = Path(llama_src)
    script = src / "convert_lora_to_gguf.py"
    if not script.is_file():
        print(f"converter not found: {script}", file=sys.stderr)
        return 2
    env = {
        **os.environ,
        "PYTHONPATH": str(src / "gguf-py") + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }
    target = out / "adapter.gguf"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--outfile",
            str(target),
            "--outtype",
            "f16",
            "--base",
            base_config_dir(job),
            str(out / "adapter"),
        ],
        cwd=str(src),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    (out / "export.log").write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
    if result.returncode != 0 or not target.is_file():
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
