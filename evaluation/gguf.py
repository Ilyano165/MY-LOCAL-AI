"""Minimaler GGUF-Metadatenleser (nur Header-Schlüssel, keine Tensoren).

Liefert *erkannte* Fakten aus der Modelldatei: Architektur, Quantisierungstyp
(``general.file_type``), Trainingskontext, Größenlabel. Format: GGUF v2/v3, little-endian
(https://github.com/ggml-org/ggml/blob/master/docs/gguf.md).
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, BinaryIO

# llama_ftype (llama.h) – Werte, die in general.file_type stehen
FILE_TYPES = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    7: "Q8_0",
    8: "Q5_0",
    9: "Q5_1",
    10: "Q2_K",
    11: "Q3_K_S",
    12: "Q3_K_M",
    13: "Q3_K_L",
    14: "Q4_K_S",
    15: "Q4_K_M",
    16: "Q5_K_S",
    17: "Q5_K_M",
    18: "Q6_K",
    19: "IQ2_XXS",
    20: "IQ2_XS",
    21: "Q2_K_S",
    22: "IQ3_XS",
    23: "IQ3_XXS",
    24: "IQ1_S",
    25: "IQ4_NL",
    26: "IQ3_S",
    27: "IQ3_M",
    28: "IQ2_S",
    29: "IQ2_M",
    30: "IQ4_XS",
    31: "IQ1_M",
    32: "BF16",
    36: "TQ1_0",
    37: "TQ2_0",
}

_SCALAR = {
    0: "<B",
    1: "<b",
    2: "<H",
    3: "<h",
    4: "<I",
    5: "<i",
    6: "<f",
    7: "<?",
    10: "<Q",
    11: "<q",
    12: "<d",
}
_STRING, _ARRAY = 8, 9
_WANTED_SUFFIXES = (".context_length", ".block_count", ".embedding_length")


class GGUFError(ValueError):
    pass


def _read(fh: BinaryIO, fmt: str) -> Any:
    size = struct.calcsize(fmt)
    data = fh.read(size)
    if len(data) != size:
        raise GGUFError("Datei endet unerwartet")
    return struct.unpack(fmt, data)[0]


def _string(fh: BinaryIO, keep: bool = True) -> str | None:
    length = _read(fh, "<Q")
    if length > 1 << 30:
        raise GGUFError("unplausible Stringlänge")
    if not keep:
        fh.seek(length, 1)
        return None
    return fh.read(length).decode("utf-8", errors="replace")


def _value(fh: BinaryIO, vtype: int, keep: bool) -> Any:
    if vtype in _SCALAR:
        return _read(fh, _SCALAR[vtype])
    if vtype == _STRING:
        return _string(fh, keep)
    if vtype == _ARRAY:
        item_type = _read(fh, "<I")
        count = _read(fh, "<Q")
        if item_type in _SCALAR:  # große Arrays (Vokabular-Scores) überspringen
            fh.seek(struct.calcsize(_SCALAR[item_type]) * count, 1)
        else:
            for _ in range(count):
                _value(fh, item_type, keep=False)
        return None
    raise GGUFError(f"unbekannter GGUF-Werttyp {vtype}")


def read_metadata(path: Path) -> dict[str, Any]:
    """Relevante Header-Schlüssel; wirft :class:`GGUFError` bei ungültiger Datei."""
    out: dict[str, Any] = {}
    with Path(path).open("rb") as fh:
        if fh.read(4) != b"GGUF":
            raise GGUFError(f"{path}: keine GGUF-Datei")
        version = _read(fh, "<I")
        if version not in (2, 3):
            raise GGUFError(f"{path}: GGUF-Version {version} nicht unterstützt")
        _read(fh, "<Q")  # tensor_count
        kv_count = _read(fh, "<Q")
        out["gguf_version"] = version
        for _ in range(kv_count):
            key = _string(fh) or ""
            vtype = _read(fh, "<I")
            keep = key.startswith("general.") or key.endswith(_WANTED_SUFFIXES)
            value = _value(fh, vtype, keep=keep and vtype != _ARRAY)
            if keep and value is not None:
                out[key] = value
    if isinstance(out.get("general.file_type"), int):
        out["quantization"] = FILE_TYPES.get(
            out["general.file_type"], f"unknown({out['general.file_type']})"
        )
    arch = out.get("general.architecture")
    if isinstance(arch, str) and f"{arch}.context_length" in out:
        out["context_length_train"] = out[f"{arch}.context_length"]
    return out
