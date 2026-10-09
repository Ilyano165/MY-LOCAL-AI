"""Modellverwaltung: Katalog, Prüfung vor dem Download, sicherer Download.

Modelle sind **nicht** Teil des Installers. Der Katalog (``<Daten>/model-catalog.json``) wird
vom Benutzer oder der IT gepflegt – NOVA erfindet keine Download-Adressen oder Prüfsummen.
Vorlage: ``config/model-catalog.example.json``.

Vor dem Download (:meth:`ModelManager.plan`): Format unterstützt (GGUF), Größe, freier
Speicherplatz (+5 % Reserve), Arbeitsspeicher gegen ``min_ram_gb``, Lizenz/Bestätigung.
Download (:meth:`ModelManager.download`): nur HTTPS, Fortsetzen unterbrochener Downloads
(HTTP Range, ``.part``-Datei), SHA-256 über die gesamte Datei, GGUF-Signatur, atomares
Umbenennen, Metadaten mit Lizenzbestätigung. Bei Prüfsummenfehler wird die Datei verworfen.

Hinweis: Eine heruntergeladene GGUF-Datei braucht zusätzlich eine lokale Runtime
(llama.cpp ``llama-server`` oder Ollama), die NOVA in ``models.toml`` anspricht.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePath
from typing import Any

import httpx

from api.paths import DataLayout
from router.resources import _sysconf_total_gb, windows_memory_gb

_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,79}$")
SUPPORTED_FORMATS = (".gguf",)
GGUF_MAGIC = b"GGUF"
Progress = Callable[[int, int], None]


class CatalogError(ValueError):
    pass


@dataclass(frozen=True)
class CatalogEntry:
    id: str
    name: str
    file: str
    url: str
    sha256: str
    size_bytes: int
    license: str
    license_url: str
    requires_license_acceptance: bool = True
    min_ram_gb: float | None = None
    context_length: int | None = None
    quantization: str | None = None
    notes: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CatalogEntry:
        try:
            entry = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        except TypeError as exc:
            raise CatalogError(f"Catalog entry incomplete: {exc}") from exc
        entry.validate()
        return entry

    def validate(self) -> None:
        problems = []
        if not _ID_RE.match(self.id):
            problems.append("id: lowercase letters, digits, . _ -")
        if PurePath(self.file).name != self.file or not self.file:
            problems.append("file must be a plain file name")
        if not self.file.lower().endswith(SUPPORTED_FORMATS):
            problems.append(f"unsupported format (supported: {', '.join(SUPPORTED_FORMATS)})")
        if not self.url.startswith("https://") or "<" in self.url:
            problems.append("url must be a real https:// address")
        if not _SHA_RE.match(self.sha256):
            problems.append("sha256 must be 64 lowercase hex characters")
        if self.size_bytes <= 0:
            problems.append("size_bytes must be > 0")
        if not self.license or not self.license_url.startswith("https://"):
            problems.append("license and https license_url are required")
        if problems:
            raise CatalogError(f"Catalog entry {self.id!r}: " + "; ".join(problems))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DownloadPlan:
    entry: str
    size_bytes: int
    free_bytes: int | None
    ram_total_gb: float | None
    installed: bool
    license: str
    license_url: str
    requires_license_acceptance: bool
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "ok": self.ok}


def total_ram_gb() -> float | None:
    win = windows_memory_gb()
    if win is not None:
        return win[1]
    return _sysconf_total_gb()


class ModelManager:
    def __init__(
        self,
        layout: DataLayout,
        *,
        client: httpx.AsyncClient | None = None,
        ram_gb: Callable[[], float | None] = total_ram_gb,
    ) -> None:
        self.layout = layout
        self._client = client
        self._ram_gb = ram_gb
        self._locks: dict[str, asyncio.Lock] = {}
        self.progress: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------ Katalog

    def catalog(self) -> list[CatalogEntry]:
        path = self.layout.catalog
        if not path.is_file():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise CatalogError(f"{path}: invalid JSON ({exc})") from exc
        entries = data.get("models", []) if isinstance(data, dict) else []
        return [CatalogEntry.from_dict(e) for e in entries]

    def entry(self, entry_id: str) -> CatalogEntry:
        for entry in self.catalog():
            if entry.id == entry_id:
                return entry
        raise CatalogError(f"Model {entry_id!r} is not in the catalog ({self.layout.catalog})")

    def target(self, entry: CatalogEntry) -> Path:
        return self.layout.models / entry.file

    def installed(self) -> list[dict[str, Any]]:
        items = []
        for meta in sorted(self.layout.models.glob("*.gguf.json")):
            try:
                items.append(json.loads(meta.read_text(encoding="utf-8")))
            except ValueError:
                continue
        return items

    # ------------------------------------------------------------------ Prüfung

    def plan(self, entry: CatalogEntry) -> DownloadPlan:
        self.layout.ensure()
        installed = self.target(entry).is_file()
        part = self.layout.downloads / f"{entry.file}.part"
        already = part.stat().st_size if part.is_file() else 0
        free = shutil.disk_usage(self.layout.models).free
        ram = self._ram_gb()
        plan = DownloadPlan(
            entry.id,
            entry.size_bytes,
            free,
            ram,
            installed,
            entry.license,
            entry.license_url,
            entry.requires_license_acceptance,
        )
        needed = int((entry.size_bytes - already) * 1.05)
        if free < needed:
            plan.problems.append(
                f"Not enough disk space: {needed / 1e9:.1f} GB needed, {free / 1e9:.1f} GB free"
            )
        if entry.min_ram_gb is not None:
            if ram is None:
                plan.warnings.append("RAM could not be determined")
            elif ram < entry.min_ram_gb:
                plan.problems.append(
                    f"Not enough memory: model needs {entry.min_ram_gb:g} GB "
                    f"RAM, this machine has {ram:.1f} GB"
                )
        if installed:
            plan.warnings.append("Already installed – download would be skipped")
        if already:
            plan.warnings.append(f"Resuming a previous download ({already / 1e9:.2f} GB done)")
        return plan

    # ------------------------------------------------------------------ Download

    async def download(
        self, entry: CatalogEntry, *, accept_license: bool, progress: Progress | None = None
    ) -> dict[str, Any]:
        if entry.requires_license_acceptance and not accept_license:
            raise CatalogError(
                f"License must be accepted first: {entry.license} ({entry.license_url})"
            )
        plan = self.plan(entry)
        if not plan.ok:
            raise CatalogError("; ".join(plan.problems))
        target = self.target(entry)
        if target.is_file():
            return self._metadata(entry, target, skipped=True)
        lock = self._locks.setdefault(entry.id, asyncio.Lock())
        if lock.locked():
            raise CatalogError(f"Download of {entry.id} is already running")
        async with lock:
            return await self._download(entry, target, progress, accept_license)

    async def _download(
        self, entry: CatalogEntry, target: Path, progress: Progress | None, accepted: bool
    ) -> dict[str, Any]:
        self.layout.downloads.mkdir(parents=True, exist_ok=True)
        part = self.layout.downloads / f"{entry.file}.part"
        digest = hashlib.sha256()
        offset = part.stat().st_size if part.is_file() else 0
        if offset > entry.size_bytes:
            part.unlink()
            offset = 0
        if offset:
            await asyncio.to_thread(_hash_into, part, digest)
        client = self._client or httpx.AsyncClient(follow_redirects=True, timeout=60)
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        state = self.progress.setdefault(entry.id, {})
        state.update(
            {"status": "downloading", "done": offset, "total": entry.size_bytes, "error": None}
        )
        try:
            async with client.stream("GET", entry.url, headers=headers) as response:
                if offset and response.status_code == 200:  # Server ignoriert Range: neu
                    offset = 0
                    digest = hashlib.sha256()
                    part.unlink(missing_ok=True)
                elif response.status_code not in (200, 206):
                    raise CatalogError(f"Download failed: HTTP {response.status_code}")
                if not str(response.url).startswith("https://"):
                    raise CatalogError("Redirected to a non-HTTPS address – aborted")
                with part.open("ab") as fh:
                    done = offset
                    async for chunk in response.aiter_bytes():  # sofort schreiben (Fortsetzen)
                        fh.write(chunk)
                        digest.update(chunk)
                        done += len(chunk)
                        if done > entry.size_bytes:
                            raise CatalogError("Server sent more data than the catalog size")
                        state["done"] = done
                        if progress is not None:
                            progress(done, entry.size_bytes)
        except (httpx.HTTPError, OSError) as exc:
            state.update({"status": "interrupted", "error": str(exc)})
            raise CatalogError(
                f"Download interrupted ({type(exc).__name__}) – run it again to resume"
            ) from exc
        finally:
            if self._client is None:
                await client.aclose()
        size = part.stat().st_size
        if size != entry.size_bytes:
            state.update({"status": "interrupted", "error": "incomplete"})
            raise CatalogError(
                f"Incomplete download: {size} of {entry.size_bytes} bytes – run it again to resume"
            )
        if digest.hexdigest() != entry.sha256:
            part.unlink()
            state.update({"status": "failed", "error": "checksum mismatch"})
            raise CatalogError("SHA-256 mismatch – file discarded (corrupt or wrong file)")
        with part.open("rb") as fh:
            magic = fh.read(4)
        if magic != GGUF_MAGIC:  # erst schließen, dann löschen (Windows sperrt offene Dateien)
            part.unlink()
            state.update({"status": "failed", "error": "not a GGUF file"})
            raise CatalogError("File is not a GGUF model – discarded")
        part.replace(target)
        meta = self._metadata(entry, target, skipped=False, accepted=accepted)
        target.with_suffix(target.suffix + ".json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        state.update({"status": "completed", "done": entry.size_bytes})
        return meta

    @staticmethod
    def _metadata(
        entry: CatalogEntry, target: Path, *, skipped: bool, accepted: bool = False
    ) -> dict[str, Any]:
        return {
            "id": entry.id,
            "name": entry.name,
            "file": str(target),
            "sha256": entry.sha256,
            "size_bytes": entry.size_bytes,
            "source_url": entry.url,
            "license": entry.license,
            "license_url": entry.license_url,
            "license_accepted_at": datetime.now(UTC).isoformat(timespec="seconds")
            if accepted
            else None,
            "skipped": skipped,
        }


def _hash_into(path: Path, digest: Any) -> None:
    with path.open("rb") as fh:
        while chunk := fh.read(1024 * 1024):
            digest.update(chunk)
