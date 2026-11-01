"""Speech-to-text via the worker (Qwen3-ASR + optional ForcedAligner timestamps)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from enum import StrEnum
from pathlib import Path
from typing import Any

from voiceapi_gateway.clients.worker import WorkerClient
from voiceapi_gateway.errors import ApiError, ErrorCode
from voiceapi_gateway.services.jobs import JobContext, JobQueue, JobType


class TranscriptFormat(StrEnum):
    JSON = "json"
    TEXT = "text"
    VERBOSE_JSON = "verbose_json"
    SRT = "srt"
    VTT = "vtt"


def pick_language(language: str | None, languages: list[str] | None) -> str | None:
    """One hint forces the language; several (Ovis ``languages[]``) mean auto-detect among them."""
    candidates = [item.strip() for item in ([language] if language else []) + list(languages or []) if item]
    unique = list(dict.fromkeys(candidates))
    if len(unique) == 1 and unique[0].lower() not in {"auto", ""}:
        return unique[0]
    return None


def _timestamp(seconds: float, *, comma: bool) -> str:
    millis = int(round(seconds * 1000))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    separator = "," if comma else "."
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def _segments(result: dict[str, Any]) -> list[dict[str, Any]]:
    segments = result.get("segments")
    if segments:
        return segments
    duration = float(result.get("duration") or 0.0)
    return [{"id": 0, "start": 0.0, "end": duration, "text": result.get("text", "")}]


def render(result: dict[str, Any], fmt: TranscriptFormat) -> tuple[str, str]:
    """Return (body, media type) for a worker transcription result."""
    text = result.get("text", "")
    if fmt is TranscriptFormat.TEXT:
        return text, "text/plain; charset=utf-8"
    if fmt is TranscriptFormat.JSON:
        return json.dumps({"text": text}), "application/json"
    if fmt is TranscriptFormat.VERBOSE_JSON:
        body = {
            "task": "transcribe",
            "language": result.get("language"),
            "duration": result.get("duration"),
            "text": text,
            "segments": _segments(result),
        }
        if result.get("words") is not None:
            body["words"] = result["words"]
        return json.dumps(body), "application/json"
    lines: list[str] = ["WEBVTT", ""] if fmt is TranscriptFormat.VTT else []
    for index, segment in enumerate(_segments(result), start=1):
        start = _timestamp(segment["start"], comma=fmt is TranscriptFormat.SRT)
        end = _timestamp(segment["end"], comma=fmt is TranscriptFormat.SRT)
        if fmt is TranscriptFormat.SRT:
            lines.append(str(index))
        lines += [f"{start} --> {end}", segment["text"].strip(), ""]
    return "\n".join(lines), "application/x-subrip" if fmt is TranscriptFormat.SRT else "text/vtt"


class TranscriptionService:
    def __init__(self, *, worker: WorkerClient, jobs: JobQueue) -> None:
        self.worker = worker
        self.jobs = jobs
        jobs.register(JobType.AUDIO_TRANSCRIBE, self._job)

    async def transcribe(
        self,
        path: Path,
        *,
        language: str | None,
        prompt: str | None,
        word_timestamps: bool,
    ) -> dict[str, Any]:
        result = await self.worker.transcribe(path, language=language, prompt=prompt, word_timestamps=word_timestamps)
        if "text" not in result:
            raise ApiError(502, ErrorCode.UPSTREAM_ERROR, "worker returned no transcript")
        return result

    async def stream(self, path: Path, *, language: str | None, prompt: str | None) -> AsyncIterator[str]:
        """Ovis/OpenAI SSE: ``transcript.text.delta`` events then ``transcript.text.done``."""
        async for event in self.worker.transcribe_stream(path, language=language, prompt=prompt):
            if event.get("done"):
                done = {"type": "transcript.text.done", "text": event.get("text", "")}
                if event.get("language"):
                    done["language"] = event["language"]
                yield f"data: {json.dumps(done)}\n\n"
                return
            if event.get("delta"):
                yield f"data: {json.dumps({'type': 'transcript.text.delta', 'delta': event['delta']})}\n\n"

    async def _job(self, context: JobContext) -> dict[str, Any]:
        payload = context.input
        await context.progress(0.1, "transcribing")
        return await self.transcribe(
            Path(payload["path"]),
            language=payload.get("language"),
            prompt=payload.get("prompt"),
            word_timestamps=bool(payload.get("word_timestamps")),
        )


__all__ = ["TranscriptFormat", "TranscriptionService", "pick_language", "render"]
