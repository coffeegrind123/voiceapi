"""Voice catalogue: build clone voices from clips and keep the engine in sync.

Build pipeline (job ``voice.build``):

    source clips ──► worker /v1/reference ──► cleaned clips, chosen reference
                     (RE-USE, trim, ASR, align,     span + exact transcript
                      reference selection)
                 ──► engine POST /v1/voices ──► prompt (safetensors) stored here

The gateway owns durable state (SQLite + files). The engine only holds prompts
resident in GPU rows; ``ensure_resident`` re-registers a stored prompt when the
engine restarted or evicted it.
"""

from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path
from typing import Any

from voiceapi_gateway.clients.engine import EngineClient
from voiceapi_gateway.clients.worker import WorkerClient
from voiceapi_gateway.db import Database, dumps, loads
from voiceapi_gateway.errors import ApiError, ErrorCode, not_found
from voiceapi_gateway.ids import engine_voice_handle, is_id, new_id
from voiceapi_gateway.services.files import FilePurpose, FileStore
from voiceapi_gateway.services.jobs import JobContext, JobQueue, JobType


class VoiceStatus:
    BUILDING = "building"
    READY = "ready"
    FAILED = "failed"


CLONE_MODES = ("auto", "icl", "xvector")
_NAME = re.compile(r"^[\w][\w .'\-]{0,62}$", re.UNICODE)
MIN_TOTAL_SECONDS = 3.0


def _validate_name(name: str) -> str:
    cleaned = (name or "").strip()
    if not _NAME.fullmatch(cleaned):
        raise ApiError(
            422, ErrorCode.VALIDATION_FAILED, f"invalid voice name {name!r}",
            "Use 1-63 letters, digits, spaces, dots, apostrophes or hyphens.",
        )
    if is_id(cleaned, "voice"):
        raise ApiError(422, ErrorCode.VALIDATION_FAILED, "a voice name cannot look like a voice id")
    return cleaned


class VoiceService:
    def __init__(
        self,
        *,
        db: Database,
        files: FileStore,
        jobs: JobQueue,
        engine: EngineClient,
        worker: WorkerClient,
        max_clips: int,
        max_total_seconds: float,
    ) -> None:
        self.db = db
        self.files = files
        self.jobs = jobs
        self.engine = engine
        self.worker = worker
        self.max_clips = max_clips
        self.max_total_seconds = max_total_seconds
        self._resident_lock = asyncio.Lock()
        jobs.register(JobType.VOICE_BUILD, self._build_job)

    # ---- public operations ----------------------------------------------------------

    async def create(
        self,
        *,
        name: str,
        file_ids: list[str],
        description: str = "",
        tags: list[str] | None = None,
        transcript: str | None = None,
        language: str = "auto",
        clean: bool = True,
        mode: str = "auto",
        consent: bool = False,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if not consent:
            raise ApiError(
                422, ErrorCode.CONSENT_REQUIRED, "consent must be true to create a voice",
                "Set consent=true to confirm you have the speaker's permission to clone this voice.",
            )
        name = _validate_name(name)
        if mode not in CLONE_MODES:
            raise ApiError(422, ErrorCode.VALIDATION_FAILED, f"mode must be one of {', '.join(CLONE_MODES)}")
        if not 1 <= len(file_ids) <= self.max_clips:
            raise ApiError(422, ErrorCode.VALIDATION_FAILED, f"provide 1 to {self.max_clips} clips")
        if transcript and len(file_ids) != 1:
            raise ApiError(
                422, ErrorCode.VALIDATION_FAILED, "transcript applies only to a single clip",
                "Omit transcript to auto-transcribe, or send one clip.",
            )
        sources = [await self.files.get(file_id) for file_id in file_ids]
        total = sum(source.duration_s or 0.0 for source in sources)
        if total > self.max_total_seconds:
            raise ApiError(
                422, ErrorCode.AUDIO_TOO_LONG, f"clips total {total:.0f} s; max {self.max_total_seconds:.0f} s"
            )
        if total < MIN_TOTAL_SECONDS:
            raise ApiError(
                422, ErrorCode.AUDIO_TOO_SHORT, f"clips total {total:.1f} s; need at least {MIN_TOTAL_SECONDS:.0f} s",
                "5-15 s of clean speech from one speaker works best.",
            )
        if await self.db.fetch_one("SELECT id FROM voices WHERE name = ?", (name,)):
            raise ApiError(409, ErrorCode.VOICE_NAME_TAKEN, f"a voice named {name!r} exists", "Pick another name.")

        voice_id = new_id("voice")
        now = time.time()
        await self.db.insert(
            "voices",
            {
                "id": voice_id, "name": name, "description": description, "tags": dumps(tags or []),
                "status": VoiceStatus.BUILDING, "default_mode": mode, "language": language,
                "created_at": now, "updated_at": now,
            },
        )
        for position, source in enumerate(sources):
            await self.db.insert("voice_sources", {"voice_id": voice_id, "file_id": source.id, "position": position})
        job = await self.jobs.submit(
            JobType.VOICE_BUILD,
            {"voice_id": voice_id, "transcript": transcript, "language": language, "clean": clean, "mode": mode},
            resource_id=voice_id,
        )
        return await self.get(voice_id), job

    async def get(self, voice_ref: str) -> dict[str, Any]:
        return self.public(await self._row(voice_ref))

    async def list(self, *, limit: int, after: str | None, status: str | None = None):
        where, params = ("status = ?", (status,)) if status else ("1=1", ())
        page = await self.db.page("voices", limit=limit, after=after, where=where, params=params)
        return [self.public(row) for row in page.items], page.has_more

    async def update(self, voice_ref: str, changes: dict[str, Any]) -> dict[str, Any]:
        row = await self._row(voice_ref)
        values: dict[str, Any] = {}
        if "name" in changes and changes["name"] is not None:
            name = _validate_name(changes["name"])
            clash = await self.db.fetch_one("SELECT id FROM voices WHERE name = ? AND id != ?", (name, row["id"]))
            if clash:
                raise ApiError(409, ErrorCode.VOICE_NAME_TAKEN, f"a voice named {name!r} exists")
            values["name"] = name
        if changes.get("description") is not None:
            values["description"] = str(changes["description"])[:2000]
        if changes.get("tags") is not None:
            values["tags"] = dumps([str(tag)[:64] for tag in changes["tags"]][:32])
        if changes.get("default_mode") is not None:
            if changes["default_mode"] not in CLONE_MODES:
                raise ApiError(422, ErrorCode.VALIDATION_FAILED, f"default_mode must be one of {CLONE_MODES}")
            if changes["default_mode"] == "icl" and not row["supports_icl"]:
                raise ApiError(422, ErrorCode.VALIDATION_FAILED, "this voice has no reference transcript for ICL")
            values["default_mode"] = changes["default_mode"]
        if values:
            values["updated_at"] = time.time()
            await self.db.update("voices", row["id"], values)
        return await self.get(row["id"])

    async def delete(self, voice_ref: str) -> None:
        row = await self._row(voice_ref)
        if row["status"] == VoiceStatus.READY:
            await self.engine.delete_voice(engine_voice_handle(row["id"]))
        await self.db.execute("DELETE FROM voices WHERE id = ?", (row["id"],))
        if row["reference_file_id"]:
            try:
                await self.files.delete(row["reference_file_id"])
            except ApiError:
                pass

    async def resolve_for_speech(self, voice_ref: str) -> dict[str, Any]:
        """Return a ready voice row (by id or name) and make sure the engine holds it."""
        row = await self._row(voice_ref)
        if row["status"] != VoiceStatus.READY:
            raise ApiError(
                409, ErrorCode.VOICE_NOT_READY, f"voice {row['id']} is {row['status']}",
                "Wait for its voice.build job to finish (GET /v1/jobs?status=running).",
            )
        return row

    async def ensure_resident(self, row: dict[str, Any], *, force: bool = False) -> None:
        handle = engine_voice_handle(row["id"])
        async with self._resident_lock:
            if not force and handle in await self.engine.resident_voices():
                return
            prompt = row.get("prompt")
            if not prompt:
                prompt_row = await self.db.fetch_one("SELECT prompt FROM voices WHERE id = ?", (row["id"],))
                prompt = prompt_row["prompt"] if prompt_row else None
            if not prompt:
                raise ApiError(500, ErrorCode.INTERNAL, f"voice {row['id']} has no stored prompt")
            await self.engine.put_voice(handle, bytes(prompt))

    async def sync_engine(self) -> int:
        """Re-register every ready voice (after the engine restarts). Returns the count."""
        rows = await self.db.fetch_all("SELECT id, prompt FROM voices WHERE status = ? ORDER BY updated_at DESC",
                                       (VoiceStatus.READY,))
        resident = await self.engine.resident_voices()
        registered = 0
        for row in rows:
            if engine_voice_handle(row["id"]) in resident:
                continue
            try:
                await self.engine.put_voice(engine_voice_handle(row["id"]), bytes(row["prompt"]))
                registered += 1
            except ApiError as error:
                if error.code is ErrorCode.CAPACITY_EXCEEDED:
                    break  # the pool is full; the rest load on demand
                raise
        return registered

    # ---- rendering ------------------------------------------------------------------

    @staticmethod
    def public(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": row["id"],
            "object": "voice",
            "name": row["name"],
            "description": row["description"],
            "tags": loads(row["tags"], []),
            "status": row["status"],
            "default_mode": row["default_mode"],
            "language": row["language"],
            "supports_icl": bool(row["supports_icl"]),
            "reference": (
                {
                    "file_id": row["reference_file_id"],
                    "transcript": row["ref_transcript"],
                    "duration_s": row["ref_duration_s"],
                }
                if row["reference_file_id"]
                else None
            ),
            "quality": loads(row["quality"], {}),
            "error": loads(row["error"]),
            "created_at": int(row["created_at"]),
            "updated_at": int(row["updated_at"]),
        }

    # ---- internals ------------------------------------------------------------------

    async def _row(self, voice_ref: str) -> dict[str, Any]:
        if is_id(voice_ref, "voice"):
            row = await self.db.fetch_one("SELECT * FROM voices WHERE id = ?", (voice_ref,))
        else:
            row = await self.db.fetch_one("SELECT * FROM voices WHERE name = ?", (voice_ref.strip(),))
        if row is None:
            names = await self.db.fetch_all(
                "SELECT name FROM voices WHERE status = ? ORDER BY updated_at DESC LIMIT 10", (VoiceStatus.READY,)
            )
            available = ", ".join(item["name"] for item in names) or "none yet"
            raise not_found(
                ErrorCode.VOICE_NOT_FOUND, f"voice {voice_ref!r}",
                f"Use a voice id or name from GET /v1/voices (ready voices: {available}).",
            )
        return row

    async def _build_job(self, context: JobContext) -> dict[str, Any]:
        payload = context.input
        voice_id = payload["voice_id"]
        try:
            return await self._build(context, voice_id, payload)
        except BaseException as error:
            detail = error.detail if isinstance(error, ApiError) else str(error) or error.__class__.__name__
            code = error.code.value if isinstance(error, ApiError) else "internal_error"
            await self.db.update(
                "voices", voice_id,
                {"status": VoiceStatus.FAILED, "error": dumps({"code": code, "message": detail}),
                 "updated_at": time.time()},
            )
            raise

    async def _build(self, context: JobContext, voice_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        sources = await self.db.fetch_all(
            "SELECT f.path FROM voice_sources s JOIN files f ON f.id = s.file_id WHERE s.voice_id = ? "
            "ORDER BY s.position",
            (voice_id,),
        )
        if not sources:
            raise ApiError(422, ErrorCode.FILE_NOT_FOUND, "voice source files were deleted")
        work = self.files.work_dir(context.job_id)
        await context.progress(0.05, "preparing reference (clean, transcribe, align)")
        prepared = await self.worker.prepare_reference(
            [Path(source["path"]) for source in sources],
            work,
            transcript=payload.get("transcript"),
            language=None if payload.get("language") in (None, "auto") else payload["language"],
            clean=bool(payload.get("clean", True)),
        )
        await context.progress(0.7, "building clone prompt on the engine")
        reference_path = Path(prepared["reference_path"])
        transcript = (prepared.get("reference_text") or "").strip() or None
        mode = payload.get("mode", "auto")
        x_vector_only = mode == "xvector" or transcript is None
        handle = engine_voice_handle(voice_id)
        reference_wav = await asyncio.to_thread(reference_path.read_bytes)
        xvector_path = prepared.get("xvector_path")
        xvector_wav = await asyncio.to_thread(Path(xvector_path).read_bytes) if xvector_path else None
        engine_voice, prompt = await self.engine.build_voice(
            handle, reference_wav, ref_text=transcript, x_vector_only=x_vector_only, xvector_wav=xvector_wav
        )
        await context.progress(0.9, "storing reference")
        reference = await self.files.adopt(
            reference_path, filename=f"{voice_id}-reference.wav", content_type="audio/wav",
            purpose=FilePurpose.VOICE_REFERENCE,
        )
        quality = dict(prepared.get("quality") or {})
        quality["clips_used"] = prepared.get("clips_used")
        await self.db.update(
            "voices", voice_id,
            {
                "status": VoiceStatus.READY,
                "supports_icl": int(engine_voice.supports_icl),
                "ref_transcript": transcript if engine_voice.supports_icl else prepared.get("reference_text"),
                "ref_duration_s": prepared.get("reference_duration_s"),
                "reference_file_id": reference.id,
                "language": prepared.get("language") or payload.get("language") or "auto",
                "default_mode": mode if (mode != "icl" or engine_voice.supports_icl) else "xvector",
                "prompt": prompt,
                "quality": dumps(quality),
                "error": None,
                "updated_at": time.time(),
            },
        )
        voice = await self.get(voice_id)
        return {"voice": voice}


__all__ = ["CLONE_MODES", "VoiceService", "VoiceStatus"]
