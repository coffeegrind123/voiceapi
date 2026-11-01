"""Voices, files and jobs."""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field
from starlette.datastructures import UploadFile

from voiceapi_gateway.api.common import (
    Authenticated,
    base_url,
    form_bool,
    get_state,
    idempotent,
    job_response,
    limit_param,
    read_form,
    upload_chunks,
    wait_param,
)
from voiceapi_gateway.audio import MEDIA_TYPES, OutputFormat
from voiceapi_gateway.errors import ApiError, ErrorCode
from voiceapi_gateway.services.files import FilePurpose, StoredFile
from voiceapi_gateway.services.speech import SpeechRequest

router = APIRouter(prefix="/v1", dependencies=[Authenticated])

PREVIEW_TEXT = "Hello! This is a short preview of my cloned voice."


# ---- files -----------------------------------------------------------------------------


@router.post("/files", tags=["files"], status_code=201, summary="Upload an audio file")
async def upload_file(request: Request) -> JSONResponse:
    state = get_state(request)
    form = await read_form(request, max_fields=8)
    upload = form.get("file")
    if not isinstance(upload, UploadFile):
        raise ApiError(422, ErrorCode.VALIDATION_FAILED, "multipart field 'file' is required")
    stored = await state.files.save_stream(
        upload_chunks(upload), filename=upload.filename or "audio", content_type=upload.content_type,
        purpose=FilePurpose.UPLOAD,
    )
    return JSONResponse(stored.public(base_url(request)), status_code=201)


@router.get("/files", tags=["files"], summary="List files")
async def list_files(
    request: Request,
    limit: int = Depends(limit_param),
    after: str | None = None,
    purpose: str | None = None,
) -> dict[str, Any]:
    state = get_state(request)
    page = await state.files.list(limit=limit, after=after, purpose=purpose)
    data = [StoredFile.from_row(row).public(base_url(request)) for row in page.items]
    return {"object": "list", "data": data, "has_more": page.has_more,
            "next_after": data[-1]["id"] if page.has_more and data else None}


@router.get("/files/{file_id}", tags=["files"], summary="Get file metadata")
async def get_file(file_id: str, request: Request) -> dict[str, Any]:
    return (await get_state(request).files.get(file_id)).public(base_url(request))


@router.get("/files/{file_id}/content", tags=["files"], summary="Download file bytes")
async def file_content(file_id: str, request: Request) -> FileResponse:
    stored = await get_state(request).files.get(file_id)
    return FileResponse(stored.path, media_type=stored.content_type, filename=stored.filename)


@router.delete("/files/{file_id}", tags=["files"], status_code=204, summary="Delete a file")
async def delete_file(file_id: str, request: Request) -> Response:
    await get_state(request).files.delete(file_id)
    return Response(status_code=204)


# ---- voices ----------------------------------------------------------------------------


class VoiceCreateBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Annotated[str, Field(min_length=1, max_length=63)]
    file_ids: Annotated[list[str], Field(min_length=1, max_length=20)]
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    transcript: str | None = Field(
        None, description="Exact transcript; only with a single clip. Omit to auto-transcribe."
    )
    language: str = "auto"
    clean: bool = Field(True, description="Run NVIDIA RE-USE speech enhancement on the clips first.")
    mode: Literal["auto", "icl", "xvector"] = "auto"


class VoicePatchBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    description: str | None = None
    tags: list[str] | None = None
    default_mode: Literal["auto", "icl", "xvector"] | None = None


async def _voice_body_from_form(request: Request) -> tuple[VoiceCreateBody, dict[str, Any]]:
    """Multipart create: upload clips inline as ``files``; other fields as form values."""
    state = get_state(request)
    form = await read_form(request, max_fields=state.settings.limits.max_form_fields)
    uploads = [item for item in form.getlist("files") + form.getlist("files[]") + form.getlist("file")
               if isinstance(item, UploadFile)]
    file_ids = [str(item) for item in form.getlist("file_ids") + form.getlist("file_ids[]")]
    for upload in uploads:
        stored = await state.files.save_stream(
            upload_chunks(upload), filename=upload.filename or "clip", content_type=upload.content_type,
            purpose=FilePurpose.VOICE_SOURCE,
        )
        file_ids.append(stored.id)
    tags_raw = form.get("tags")
    tags = json.loads(tags_raw) if isinstance(tags_raw, str) and tags_raw.startswith("[") else (
        [tag.strip() for tag in str(tags_raw).split(",") if tag.strip()] if tags_raw else []
    )
    payload = {
        "name": str(form.get("name") or ""),
        "file_ids": file_ids,
        "description": str(form.get("description") or ""),
        "tags": tags,
        "transcript": str(form.get("transcript") or "") or None,
        "language": str(form.get("language") or "auto"),
        "clean": form_bool(form.get("clean"), default=True),
        "mode": str(form.get("mode") or "auto"),
    }
    try:
        return VoiceCreateBody(**payload), payload
    except Exception as error:  # noqa: BLE001 - surfaced as a validation problem
        raise ApiError(422, ErrorCode.VALIDATION_FAILED, str(error)) from error


@router.post("/voices", tags=["voices"], summary="Create a clone voice from audio clips (async job)")
async def create_voice(request: Request, wait: float = Depends(wait_param)) -> JSONResponse:
    state = get_state(request)
    if request.headers.get("content-type", "").startswith("multipart/"):
        body, payload = await _voice_body_from_form(request)
    else:
        try:
            raw = await request.json()
            body = VoiceCreateBody(**raw)
        except ApiError:
            raise
        except Exception as error:  # noqa: BLE001
            raise ApiError(422, ErrorCode.VALIDATION_FAILED, f"invalid voice body: {error}") from error
        payload = body.model_dump()

    async def create() -> tuple[int, Any]:
        voice, job = await state.voices.create(**body.model_dump())
        job = await state.jobs.wait(job["id"], wait)
        status, _ = job_response(job)
        return status, {"voice": await state.voices.get(voice["id"]), "job": job}

    return await idempotent(request, "voices.create", payload, create)


@router.get("/voices", tags=["voices"], summary="List voices")
async def list_voices(
    request: Request,
    limit: int = Depends(limit_param),
    after: str | None = None,
    status: Literal["building", "ready", "failed"] | None = None,
) -> dict[str, Any]:
    voices, has_more = await get_state(request).voices.list(limit=limit, after=after, status=status)
    return {"object": "list", "data": voices, "has_more": has_more,
            "next_after": voices[-1]["id"] if has_more and voices else None}


@router.get("/voices/{voice_ref}", tags=["voices"], summary="Get a voice by id or name")
async def get_voice(voice_ref: str, request: Request) -> dict[str, Any]:
    return await get_state(request).voices.get(voice_ref)


@router.patch("/voices/{voice_ref}", tags=["voices"], summary="Rename or retag a voice")
async def patch_voice(voice_ref: str, body: VoicePatchBody, request: Request) -> dict[str, Any]:
    return await get_state(request).voices.update(voice_ref, body.model_dump(exclude_unset=True))


@router.delete("/voices/{voice_ref}", tags=["voices"], status_code=204, summary="Delete a voice")
async def delete_voice(voice_ref: str, request: Request) -> Response:
    await get_state(request).voices.delete(voice_ref)
    return Response(status_code=204)


@router.get("/voices/{voice_ref}/reference", tags=["voices"], summary="Download the cleaned reference audio")
async def voice_reference(voice_ref: str, request: Request) -> FileResponse:
    state = get_state(request)
    voice = await state.voices.get(voice_ref)
    if not voice["reference"]:
        raise ApiError(409, ErrorCode.VOICE_NOT_READY, "voice has no reference yet")
    stored = await state.files.get(voice["reference"]["file_id"])
    return FileResponse(stored.path, media_type="audio/wav", filename=f"{voice['name']}-reference.wav")


class PreviewBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = PREVIEW_TEXT
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = "mp3"
    voice_mode: Literal["auto", "icl", "xvector"] | None = None


@router.post("/voices/{voice_ref}/preview", tags=["voices"], summary="Speak a short sample with this voice")
async def preview_voice(voice_ref: str, request: Request, body: PreviewBody | None = None) -> Response:
    state = get_state(request)
    body = body or PreviewBody()
    fmt = OutputFormat(body.response_format)
    data = await state.speech.to_bytes(
        SpeechRequest(text=body.text[:500], voice=voice_ref, response_format=fmt, voice_mode=body.voice_mode)
    )
    return Response(data, media_type=MEDIA_TYPES[fmt])


# ---- jobs ------------------------------------------------------------------------------


@router.get("/jobs", tags=["jobs"], summary="List jobs")
async def list_jobs(
    request: Request,
    limit: int = Depends(limit_param),
    after: str | None = None,
    status: Literal["queued", "running", "succeeded", "failed", "cancelled"] | None = None,
) -> dict[str, Any]:
    state = get_state(request)
    page = await state.jobs.list(limit=limit, after=after, status=status)
    data = [state.jobs.public(row) for row in page.items]
    return {"object": "list", "data": data, "has_more": page.has_more,
            "next_after": data[-1]["id"] if page.has_more and data else None}


@router.get("/jobs/{job_id}", tags=["jobs"], summary="Get a job; ?wait=N blocks up to N seconds")
async def get_job(job_id: str, request: Request, wait: float = Depends(wait_param)) -> JSONResponse:
    status, job = job_response(await get_state(request).jobs.wait(job_id, wait))
    return JSONResponse(job, status_code=status)


@router.post("/jobs/{job_id}/cancel", tags=["jobs"], summary="Cancel a queued or running job")
async def cancel_job(job_id: str, request: Request) -> dict[str, Any]:
    return await get_state(request).jobs.cancel(job_id)


__all__ = ["router"]

