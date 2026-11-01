"""Composition root: every service the routes and MCP tools share."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass

from voiceapi_gateway.clients.engine import EngineClient
from voiceapi_gateway.clients.worker import WorkerClient
from voiceapi_gateway.config import Settings
from voiceapi_gateway.db import Database
from voiceapi_gateway.services.enhance import EnhanceService
from voiceapi_gateway.services.files import FileStore
from voiceapi_gateway.services.jobs import JobQueue
from voiceapi_gateway.services.speech import SpeechService
from voiceapi_gateway.services.transcription import TranscriptionService
from voiceapi_gateway.services.voices import VoiceService

log = logging.getLogger("voiceapi")
MAINTENANCE_INTERVAL_S = 30.0


@dataclass(slots=True)
class AppState:
    settings: Settings
    db: Database
    files: FileStore
    jobs: JobQueue
    engine: EngineClient
    worker: WorkerClient
    voices: VoiceService
    speech: SpeechService
    transcription: TranscriptionService
    enhance: EnhanceService
    engine_was_ready: bool = False
    _maintenance: asyncio.Task | None = None

    @classmethod
    def build(
        cls,
        settings: Settings,
        *,
        engine: EngineClient | None = None,
        worker: WorkerClient | None = None,
    ) -> AppState:
        db = Database(settings.db_path)
        files = FileStore(db, settings.files_dir, max_bytes=settings.limits.max_upload_bytes)
        jobs = JobQueue(db, workers=settings.limits.job_workers)
        engine = engine or EngineClient(settings.engine_url, timeout_s=settings.engine_timeout_s)
        worker = worker or WorkerClient(settings.worker_url, timeout_s=settings.worker_timeout_s)
        voices = VoiceService(
            db=db, files=files, jobs=jobs, engine=engine, worker=worker,
            max_clips=settings.limits.max_voice_clips, max_total_seconds=settings.limits.max_voice_total_seconds,
        )
        return cls(
            settings=settings,
            db=db,
            files=files,
            jobs=jobs,
            engine=engine,
            worker=worker,
            voices=voices,
            speech=SpeechService(
                engine=engine, voices=voices, files=files, default_voice=settings.default_voice,
                max_input_chars=settings.limits.max_speech_input_chars,
            ),
            transcription=TranscriptionService(worker=worker, jobs=jobs),
            enhance=EnhanceService(worker=worker, files=files, jobs=jobs),
        )

    async def start(self) -> None:
        await self.jobs.start()
        self._maintenance = asyncio.create_task(self._maintain())

    async def stop(self) -> None:
        if self._maintenance:
            self._maintenance.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._maintenance
        await self.jobs.stop()
        await self.engine.aclose()
        await self.worker.aclose()
        self.db.close()

    async def _maintain(self) -> None:
        """Sweep expired files and re-register voices whenever the engine (re)becomes ready."""
        while True:
            try:
                ready, _reason = await self.engine.ready()
                if ready and not self.engine_was_ready:
                    count = await self.voices.sync_engine()
                    log.info("engine ready; re-registered %d voice(s)", count)
                self.engine_was_ready = ready
                await self.files.sweep_expired()
            except Exception:  # noqa: BLE001 - maintenance must keep running
                log.exception("maintenance pass failed")
            await asyncio.sleep(MAINTENANCE_INTERVAL_S)


__all__ = ["AppState"]
