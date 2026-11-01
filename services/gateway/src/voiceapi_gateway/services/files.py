"""Uploaded and generated audio files in the shared data volume."""

from __future__ import annotations

import asyncio
import hashlib
import mimetypes
import shutil
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from voiceapi_gateway import audio
from voiceapi_gateway.db import Database
from voiceapi_gateway.errors import ApiError, ErrorCode, not_found
from voiceapi_gateway.ids import is_id, new_id

GENERATED_TTL_S = 7 * 24 * 3600
_SAFE_NAME = str.maketrans({c: "_" for c in '/\\:*?"<>|\x00'})


class FilePurpose:
    UPLOAD = "upload"
    VOICE_SOURCE = "voice_source"
    VOICE_REFERENCE = "voice_reference"
    SPEECH = "speech"
    ENHANCED = "enhanced"


@dataclass(frozen=True, slots=True)
class StoredFile:
    id: str
    filename: str
    content_type: str
    bytes: int
    sha256: str
    duration_s: float | None
    sample_rate: int | None
    channels: int | None
    purpose: str
    path: Path
    created_at: float
    expires_at: float | None

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> StoredFile:
        return cls(**{**row, "path": Path(row["path"])})

    def public(self, base_url: str) -> dict[str, Any]:
        return {
            "id": self.id,
            "object": "file",
            "filename": self.filename,
            "content_type": self.content_type,
            "bytes": self.bytes,
            "sha256": self.sha256,
            "duration_s": self.duration_s,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "purpose": self.purpose,
            "created_at": int(self.created_at),
            "expires_at": int(self.expires_at) if self.expires_at else None,
            "url": f"{base_url}/v1/files/{self.id}/content",
        }


class FileStore:
    def __init__(self, db: Database, root: Path, *, max_bytes: int) -> None:
        self.db = db
        self.root = root
        self.tmp = root.parent / "tmp"
        self.work = root.parent / "work"
        for directory in (self.root, self.tmp, self.work):
            directory.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes

    def temp_path(self, suffix: str = "") -> Path:
        return self.tmp / f"{new_id('req')}{suffix}"

    def work_dir(self, name: str) -> Path:
        path = self.work / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    async def save_stream(
        self,
        chunks: AsyncIterator[bytes],
        *,
        filename: str,
        content_type: str | None,
        purpose: str,
        probe_audio: bool = True,
        ttl_s: float | None = None,
        max_bytes: int | None = None,
    ) -> StoredFile:
        limit = max_bytes or self.max_bytes
        file_id = new_id("file")
        safe_name = (filename or "audio").translate(_SAFE_NAME)[-200:] or "audio"
        suffix = Path(safe_name).suffix[:10]
        path = self.root / f"{file_id}{suffix}"
        digest = hashlib.sha256()
        size = 0
        try:
            handle = await asyncio.to_thread(path.open, "wb")
            try:
                async for chunk in chunks:
                    size += len(chunk)
                    if size > limit:
                        raise ApiError(
                            413, ErrorCode.PAYLOAD_TOO_LARGE, f"file exceeds {limit} bytes",
                            "Upload a shorter recording or a compressed format.",
                        )
                    digest.update(chunk)
                    await asyncio.to_thread(handle.write, chunk)
            finally:
                await asyncio.to_thread(handle.close)
            if size == 0:
                raise ApiError(422, ErrorCode.AUDIO_UNREADABLE, "uploaded file is empty")
            info = await audio.probe(path) if probe_audio else None
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        guessed = content_type or mimetypes.guess_type(safe_name)[0] or "application/octet-stream"
        now = time.time()
        stored = StoredFile(
            id=file_id,
            filename=safe_name,
            content_type=guessed,
            bytes=size,
            sha256=digest.hexdigest(),
            duration_s=info.duration_s if info else None,
            sample_rate=info.sample_rate if info else None,
            channels=info.channels if info else None,
            purpose=purpose,
            path=path,
            created_at=now,
            expires_at=now + ttl_s if ttl_s else None,
        )
        await self._insert(stored)
        return stored

    async def save_bytes(self, data: bytes, **kwargs: Any) -> StoredFile:
        async def one() -> AsyncIterator[bytes]:
            yield data

        return await self.save_stream(one(), **kwargs)

    async def adopt(self, source: Path, *, filename: str, content_type: str, purpose: str,
                    ttl_s: float | None = None) -> StoredFile:
        """Register a file a worker wrote into the data volume (moves it under files/)."""
        resolved = await asyncio.to_thread(source.resolve)
        data_root = await asyncio.to_thread(self.root.parent.resolve)
        if not resolved.is_relative_to(data_root):
            raise ApiError(500, ErrorCode.INTERNAL, "worker output escaped the data directory")
        file_id = new_id("file")
        target = self.root / f"{file_id}{Path(filename).suffix}"
        digest, size = await asyncio.to_thread(_move_and_hash, resolved, target)
        info = await audio.probe(target)
        now = time.time()
        stored = StoredFile(
            id=file_id, filename=filename, content_type=content_type, bytes=size, sha256=digest,
            duration_s=info.duration_s, sample_rate=info.sample_rate, channels=info.channels, purpose=purpose,
            path=target, created_at=now, expires_at=now + ttl_s if ttl_s else None,
        )
        await self._insert(stored)
        return stored

    async def _insert(self, stored: StoredFile) -> None:
        await self.db.insert(
            "files",
            {
                "id": stored.id, "filename": stored.filename, "content_type": stored.content_type,
                "bytes": stored.bytes, "sha256": stored.sha256, "duration_s": stored.duration_s,
                "sample_rate": stored.sample_rate, "channels": stored.channels, "purpose": stored.purpose,
                "path": str(stored.path), "created_at": stored.created_at, "expires_at": stored.expires_at,
            },
        )

    async def get(self, file_id: str) -> StoredFile:
        if not is_id(file_id, "file"):
            raise not_found(ErrorCode.FILE_NOT_FOUND, f"file {file_id!r}", "File ids look like file_<26 chars>.")
        row = await self.db.fetch_one("SELECT * FROM files WHERE id = ?", (file_id,))
        if row is None:
            raise not_found(ErrorCode.FILE_NOT_FOUND, f"file {file_id!r}", "List files with GET /v1/files.")
        stored = StoredFile.from_row(row)
        if not stored.path.is_file():
            raise not_found(ErrorCode.FILE_NOT_FOUND, f"file content for {file_id!r}")
        return stored

    async def delete(self, file_id: str) -> None:
        stored = await self.get(file_id)
        in_use = await self.db.fetch_one(
            "SELECT id FROM voices WHERE reference_file_id = ?", (file_id,)
        )
        if in_use:
            raise ApiError(
                409, ErrorCode.VOICE_IN_USE, f"file is the reference of voice {in_use['id']}",
                "Delete the voice first.",
            )
        await self.db.execute("DELETE FROM files WHERE id = ?", (file_id,))
        stored.path.unlink(missing_ok=True)

    async def list(self, *, limit: int, after: str | None, purpose: str | None):
        if purpose:
            return await self.db.page("files", limit=limit, after=after, where="purpose = ?", params=(purpose,))
        return await self.db.page("files", limit=limit, after=after)

    async def sweep_expired(self) -> int:
        rows = await self.db.fetch_all(
            "SELECT id, path FROM files WHERE expires_at IS NOT NULL AND expires_at < ?", (time.time(),)
        )
        for row in rows:
            await self.db.execute("DELETE FROM files WHERE id = ?", (row["id"],))
        await asyncio.to_thread(_remove_paths, [Path(row["path"]) for row in rows], self.tmp)
        return len(rows)


def _move_and_hash(source: Path, target: Path) -> tuple[str, int]:
    shutil.move(str(source), target)
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest(), target.stat().st_size


def _remove_paths(paths: list[Path], tmp_dir: Path) -> None:
    """Delete expired files plus temp uploads older than a day."""
    for path in paths:
        path.unlink(missing_ok=True)
    cutoff = time.time() - 24 * 3600
    for stale in tmp_dir.iterdir():
        if stale.is_file() and stale.stat().st_mtime < cutoff:
            stale.unlink(missing_ok=True)


__all__ = ["GENERATED_TTL_S", "FilePurpose", "FileStore", "StoredFile"]
