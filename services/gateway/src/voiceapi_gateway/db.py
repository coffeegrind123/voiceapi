"""SQLite persistence for files, voices, jobs and idempotency records.

One connection in WAL mode guarded by a lock; async callers hop through
``asyncio.to_thread``. The write rate here (uploads, voice builds, jobs) is
far below what serialising writes costs, and it keeps the store a single file
in the data volume.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS files (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    content_type TEXT NOT NULL,
    bytes INTEGER NOT NULL,
    sha256 TEXT NOT NULL,
    duration_s REAL,
    sample_rate INTEGER,
    channels INTEGER,
    purpose TEXT NOT NULL,
    path TEXT NOT NULL,
    created_at REAL NOT NULL,
    expires_at REAL
);
CREATE INDEX IF NOT EXISTS files_created ON files(created_at);

CREATE TABLE IF NOT EXISTS voices (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE COLLATE NOCASE,
    description TEXT NOT NULL DEFAULT '',
    tags TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL,
    default_mode TEXT NOT NULL DEFAULT 'auto',
    language TEXT NOT NULL DEFAULT 'auto',
    supports_icl INTEGER NOT NULL DEFAULT 0,
    ref_transcript TEXT,
    ref_duration_s REAL,
    reference_file_id TEXT,
    prompt BLOB,
    quality TEXT NOT NULL DEFAULT '{}',
    error TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS voice_sources (
    voice_id TEXT NOT NULL REFERENCES voices(id) ON DELETE CASCADE,
    file_id TEXT NOT NULL,
    position INTEGER NOT NULL,
    PRIMARY KEY (voice_id, position)
);

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    status TEXT NOT NULL,
    input TEXT NOT NULL,
    result TEXT,
    error TEXT,
    progress REAL NOT NULL DEFAULT 0,
    resource_id TEXT,
    created_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL
);
CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status, created_at);

CREATE TABLE IF NOT EXISTS idempotency (
    key TEXT NOT NULL,
    scope TEXT NOT NULL,
    body_hash TEXT NOT NULL,
    status INTEGER NOT NULL,
    response TEXT NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (key, scope)
);
"""


@dataclass(frozen=True, slots=True)
class Page:
    items: list[dict[str, Any]]
    has_more: bool


class Database:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.executescript(_SCHEMA)
            self._conn.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),)
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ---- primitives ---------------------------------------------------------------

    def _run(self, fn: Callable[[sqlite3.Connection], T]) -> T:
        with self._lock:
            return fn(self._conn)

    async def run(self, fn: Callable[[sqlite3.Connection], T]) -> T:
        return await asyncio.to_thread(self._run, fn)

    async def execute(self, sql: str, params: Iterable[Any] = ()) -> int:
        return await self.run(lambda conn: conn.execute(sql, tuple(params)).rowcount)

    async def fetch_one(self, sql: str, params: Iterable[Any] = ()) -> dict[str, Any] | None:
        def go(conn: sqlite3.Connection) -> dict[str, Any] | None:
            row = conn.execute(sql, tuple(params)).fetchone()
            return dict(row) if row is not None else None

        return await self.run(go)

    async def fetch_all(self, sql: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
        return await self.run(lambda conn: [dict(row) for row in conn.execute(sql, tuple(params)).fetchall()])

    async def insert(self, table: str, values: dict[str, Any]) -> None:
        columns = ", ".join(values)
        marks = ", ".join("?" for _ in values)
        await self.execute(f"INSERT INTO {table} ({columns}) VALUES ({marks})", values.values())  # noqa: S608

    async def update(self, table: str, row_id: str, values: dict[str, Any]) -> int:
        assignments = ", ".join(f"{column} = ?" for column in values)
        return await self.execute(
            f"UPDATE {table} SET {assignments} WHERE id = ?",  # noqa: S608 - column names are code constants
            [*values.values(), row_id],
        )

    async def page(
        self,
        table: str,
        *,
        limit: int,
        after: str | None,
        where: str = "1=1",
        params: Iterable[Any] = (),
    ) -> Page:
        """Keyset pagination by id descending (ids are time-sortable ULIDs)."""
        clauses = [where]
        values = list(params)
        if after:
            clauses.append("id < ?")
            values.append(after)
        rows = await self.fetch_all(
            f"SELECT * FROM {table} WHERE {' AND '.join(clauses)} ORDER BY id DESC LIMIT ?",  # noqa: S608
            [*values, limit + 1],
        )
        return Page(items=rows[:limit], has_more=len(rows) > limit)

    # ---- idempotency ----------------------------------------------------------------

    async def idempotent_lookup(self, key: str, scope: str, ttl_s: int) -> dict[str, Any] | None:
        row = await self.fetch_one("SELECT * FROM idempotency WHERE key = ? AND scope = ?", (key, scope))
        if row is None:
            return None
        if row["created_at"] < time.time() - ttl_s:
            await self.execute("DELETE FROM idempotency WHERE key = ? AND scope = ?", (key, scope))
            return None
        row["response"] = json.loads(row["response"])
        return row

    async def idempotent_store(self, key: str, scope: str, body_hash: str, status: int, response: Any) -> None:
        await self.execute(
            "INSERT OR REPLACE INTO idempotency(key, scope, body_hash, status, response, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (key, scope, body_hash, status, json.dumps(response), time.time()),
        )


def dumps(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def loads(value: str | None, default: Any = None) -> Any:
    return default if value is None else json.loads(value)


__all__ = ["Database", "Page", "dumps", "loads"]
