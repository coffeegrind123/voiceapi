"""Durable background jobs with progress, cancellation and blocking waits.

Jobs live in SQLite so an agent can poll them across gateway restarts. A
gateway restart re-queues jobs that were running; every handler is written to
be safely re-runnable from the start.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from voiceapi_gateway.db import Database, dumps, loads
from voiceapi_gateway.errors import ApiError, ErrorCode, not_found
from voiceapi_gateway.ids import is_id, new_id

log = logging.getLogger("voiceapi.jobs")


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED)


class JobType(StrEnum):
    VOICE_BUILD = "voice.build"
    AUDIO_ENHANCE = "audio.enhance"
    AUDIO_TRANSCRIBE = "audio.transcribe"


@dataclass(slots=True)
class JobContext:
    job_id: str
    input: dict[str, Any]
    _jobs: JobQueue

    async def progress(self, fraction: float, stage: str | None = None) -> None:
        await self._jobs._set_progress(self.job_id, fraction, stage)


Handler = Callable[[JobContext], Awaitable[dict[str, Any]]]


class JobQueue:
    def __init__(self, db: Database, *, workers: int) -> None:
        self.db = db
        self.workers = workers
        self._handlers: dict[JobType, Handler] = {}
        self._wake = asyncio.Event()
        self._done: dict[str, asyncio.Event] = {}
        self._running: dict[str, asyncio.Task] = {}
        self._tasks: list[asyncio.Task] = []
        self._stopping = False

    def register(self, job_type: JobType, handler: Handler) -> None:
        self._handlers[job_type] = handler

    async def start(self) -> None:
        await self.db.execute(
            "UPDATE jobs SET status = ?, started_at = NULL, progress = 0 WHERE status = ?",
            (JobStatus.QUEUED.value, JobStatus.RUNNING.value),
        )
        self._tasks = [asyncio.create_task(self._worker(index)) for index in range(self.workers)]
        self._wake.set()

    async def stop(self) -> None:
        self._stopping = True
        self._wake.set()
        for task in [*self._tasks, *self._running.values()]:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def submit(self, job_type: JobType, payload: dict[str, Any], *, resource_id: str | None = None) -> dict:
        if job_type not in self._handlers:
            raise ApiError(500, ErrorCode.INTERNAL, f"no handler for job type {job_type.value}")
        job_id = new_id("job")
        await self.db.insert(
            "jobs",
            {
                "id": job_id, "type": job_type.value, "status": JobStatus.QUEUED.value, "input": dumps(payload),
                "progress": 0.0, "resource_id": resource_id, "created_at": time.time(),
            },
        )
        self._done[job_id] = asyncio.Event()
        self._wake.set()
        return await self.get(job_id)

    async def get(self, job_id: str) -> dict[str, Any]:
        if not is_id(job_id, "job"):
            raise not_found(ErrorCode.JOB_NOT_FOUND, f"job {job_id!r}", "Job ids look like job_<26 chars>.")
        row = await self.db.fetch_one("SELECT * FROM jobs WHERE id = ?", (job_id,))
        if row is None:
            raise not_found(ErrorCode.JOB_NOT_FOUND, f"job {job_id!r}")
        return self.public(row)

    @staticmethod
    def public(row: dict[str, Any]) -> dict[str, Any]:
        result = loads(row["result"])
        error = loads(row["error"])
        return {
            "id": row["id"],
            "object": "job",
            "type": row["type"],
            "status": row["status"],
            "progress": round(float(row["progress"]), 4),
            "stage": (loads(row["input"]) or {}).get("_stage"),
            "resource_id": row["resource_id"],
            "result": result,
            "error": error,
            "created_at": int(row["created_at"]),
            "started_at": int(row["started_at"]) if row["started_at"] else None,
            "finished_at": int(row["finished_at"]) if row["finished_at"] else None,
        }

    async def wait(self, job_id: str, timeout_s: float) -> dict[str, Any]:
        job = await self.get(job_id)
        if JobStatus(job["status"]).terminal or timeout_s <= 0:
            return job
        event = self._done.setdefault(job_id, asyncio.Event())
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(event.wait(), timeout=timeout_s)
        return await self.get(job_id)

    async def cancel(self, job_id: str) -> dict[str, Any]:
        job = await self.get(job_id)
        status = JobStatus(job["status"])
        if status.terminal:
            raise ApiError(409, ErrorCode.JOB_NOT_CANCELLABLE, f"job is already {status.value}")
        if status is JobStatus.QUEUED:
            await self._finish(job_id, JobStatus.CANCELLED, error={"code": "cancelled", "message": "cancelled"})
        else:
            task = self._running.get(job_id)
            if task is not None:
                task.cancel()
        return await self.wait(job_id, 5.0)

    async def list(self, *, limit: int, after: str | None, status: str | None):
        if status:
            return await self.db.page("jobs", limit=limit, after=after, where="status = ?", params=(status,))
        return await self.db.page("jobs", limit=limit, after=after)

    # ---- internals ------------------------------------------------------------------

    async def _claim(self) -> dict[str, Any] | None:
        def go(conn):
            row = conn.execute(
                "SELECT * FROM jobs WHERE status = ? ORDER BY created_at LIMIT 1", (JobStatus.QUEUED.value,)
            ).fetchone()
            if row is None:
                return None
            claimed = conn.execute(
                "UPDATE jobs SET status = ?, started_at = ? WHERE id = ? AND status = ?",
                (JobStatus.RUNNING.value, time.time(), row["id"], JobStatus.QUEUED.value),
            ).rowcount
            return dict(row) if claimed else None

        return await self.db.run(go)

    async def _worker(self, index: int) -> None:
        while not self._stopping:
            row = await self._claim()
            if row is None:
                self._wake.clear()
                await self._wake.wait()
                continue
            job_id = row["id"]
            handler = self._handlers.get(JobType(row["type"]))
            context = JobContext(job_id=job_id, input=loads(row["input"]) or {}, _jobs=self)
            task = asyncio.create_task(handler(context))
            self._running[job_id] = task
            try:
                result = await task
            except asyncio.CancelledError:
                if self._stopping:
                    raise
                await self._finish(job_id, JobStatus.CANCELLED, error={"code": "cancelled", "message": "cancelled"})
            except ApiError as error:
                await self._finish(
                    job_id, JobStatus.FAILED,
                    error={"code": error.code.value, "message": error.detail, "hint": error.hint},
                )
            except Exception as error:  # noqa: BLE001 - a job failure must never kill the worker
                log.exception("job %s failed", job_id)
                await self._finish(job_id, JobStatus.FAILED, error={"code": "internal_error", "message": str(error)})
            else:
                await self._finish(job_id, JobStatus.SUCCEEDED, result=result)
            finally:
                self._running.pop(job_id, None)

    async def _set_progress(self, job_id: str, fraction: float, stage: str | None) -> None:
        def go(conn):
            row = conn.execute("SELECT input FROM jobs WHERE id = ?", (job_id,)).fetchone()
            payload = loads(row["input"]) or {}
            if stage is not None:
                payload["_stage"] = stage
            conn.execute(
                "UPDATE jobs SET progress = ?, input = ? WHERE id = ?",
                (max(0.0, min(1.0, fraction)), dumps(payload), job_id),
            )

        await self.db.run(go)

    async def _finish(self, job_id: str, status: JobStatus, *, result: Any = None, error: Any = None) -> None:
        await self.db.execute(
            "UPDATE jobs SET status = ?, result = ?, error = ?, finished_at = ?, progress = COALESCE(?, progress) "
            "WHERE id = ?",
            (
                status.value,
                dumps(result) if result is not None else None,
                dumps(error) if error is not None else None,
                time.time(),
                1.0 if status is JobStatus.SUCCEEDED else None,
                job_id,
            ),
        )
        event = self._done.setdefault(job_id, asyncio.Event())
        event.set()


__all__ = ["JobContext", "JobQueue", "JobStatus", "JobType"]
