"""Speech enhancement (NVIDIA RE-USE) for arbitrary audio."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from voiceapi_gateway.clients.worker import WorkerClient
from voiceapi_gateway.services.files import GENERATED_TTL_S, FilePurpose, FileStore, StoredFile
from voiceapi_gateway.services.jobs import JobContext, JobQueue, JobType

BANDWIDTH_EXTENSION_RATES = (16_000, 22_050, 24_000, 32_000, 44_100, 48_000)


class EnhanceService:
    def __init__(self, *, worker: WorkerClient, files: FileStore, jobs: JobQueue) -> None:
        self.worker = worker
        self.files = files
        self.jobs = jobs
        jobs.register(JobType.AUDIO_ENHANCE, self._job)

    async def enhance_file(self, source: StoredFile, *, bandwidth_extension_hz: int | None) -> tuple[StoredFile, dict]:
        output = self.files.temp_path(".wav")
        metrics = await self.worker.enhance(source.path, output, bandwidth_extension_hz=bandwidth_extension_hz)
        stem = Path(source.filename).stem or "audio"
        stored = await self.files.adopt(
            output, filename=f"{stem}-enhanced.wav", content_type="audio/wav", purpose=FilePurpose.ENHANCED,
            ttl_s=GENERATED_TTL_S,
        )
        return stored, metrics

    async def _job(self, context: JobContext) -> dict[str, Any]:
        source = await self.files.get(context.input["file_id"])
        await context.progress(0.1, "enhancing")
        stored, metrics = await self.enhance_file(
            source, bandwidth_extension_hz=context.input.get("bandwidth_extension_hz")
        )
        return {"file_id": stored.id, "metrics": metrics}


__all__ = ["BANDWIDTH_EXTENSION_RATES", "EnhanceService"]
