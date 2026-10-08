from __future__ import annotations

import struct
from pathlib import Path

import pytest

from evaluation.gguf import GGUFError, read_metadata


def _s(text: str) -> bytes:
    raw = text.encode()
    return struct.pack("<Q", len(raw)) + raw


def write_gguf(path: Path, file_type: int = 15) -> Path:
    kvs = [
        _s("general.architecture") + struct.pack("<I", 8) + _s("llama"),
        _s("general.name") + struct.pack("<I", 8) + _s("Test Model"),
        _s("llama.context_length") + struct.pack("<I", 4) + struct.pack("<I", 131072),
        # großes String-Array (Vokabular) – muss übersprungen werden
        _s("tokenizer.ggml.tokens")
        + struct.pack("<I", 9)
        + struct.pack("<I", 8)
        + struct.pack("<Q", 3)
        + _s("a")
        + _s("b")
        + _s("c"),
        _s("tokenizer.ggml.scores")
        + struct.pack("<I", 9)
        + struct.pack("<I", 6)
        + struct.pack("<Q", 3)
        + struct.pack("<fff", 0.1, 0.2, 0.3),
        _s("general.file_type") + struct.pack("<I", 4) + struct.pack("<I", file_type),
    ]
    header = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", len(kvs))
    path.write_bytes(header + b"".join(kvs))
    return path


def test_reads_quantization_and_context(tmp_path: Path) -> None:
    meta = read_metadata(write_gguf(tmp_path / "m.gguf"))
    assert meta["quantization"] == "Q4_K_M"
    assert meta["general.architecture"] == "llama"
    assert meta["context_length_train"] == 131072
    assert "tokenizer.ggml.tokens" not in meta


def test_unknown_file_type(tmp_path: Path) -> None:
    assert read_metadata(write_gguf(tmp_path / "m.gguf", 999))["quantization"] == "unknown(999)"


def test_rejects_non_gguf(tmp_path: Path) -> None:
    (tmp_path / "x.bin").write_bytes(b"NOPE" + b"\0" * 20)
    with pytest.raises(GGUFError):
        read_metadata(tmp_path / "x.bin")


def test_truncated_file(tmp_path: Path) -> None:
    data = write_gguf(tmp_path / "m.gguf").read_bytes()
    (tmp_path / "t.gguf").write_bytes(data[:40])
    with pytest.raises(GGUFError):
        read_metadata(tmp_path / "t.gguf")
