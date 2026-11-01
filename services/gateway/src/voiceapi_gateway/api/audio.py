"""OpenAI-compatible audio endpoints plus enhancement."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.datastructures import UploadFile

from voiceapi_gateway import audio
from voiceapi_gateway.api.common import (
    Authenticated,
    base_url,
    form_bool,
    get_state,
    idempotent,
    job_response,
    read_form,
    upload_chunks,
    wait_param,
)
from voiceapi_gateway.errors import ApiError, ErrorCode
from voiceapi_gateway.services.enhance import BANDWIDTH_EXTENSION_RATES
from voiceapi_gateway.services.files import FilePurpose
from voiceapi_gateway.services.jobs import JobType
from voiceapi_gateway.services.speech import SpeechRequest
from voiceapi_gateway.services.transcription import TranscriptFormat, pick_language, render

router = APIRouter(prefix="/v1/audio", tags=["audio"], dependencies=[Authenticated])

TTS_MODEL_IDS = frozenset({"qwen3-tts-1.7b-base", "tts-1", "tts-1-hd", "gpt-4o-mini-tts", "gpt-4o-tts"})
ASR_MODEL_IDS = frozenset({"qwen3-asr-1.7b", "whisper-1", "gpt-4o-transcribe", "gpt-4o-mini-transcribe",
                           "gpt-transcribe"})


class SpeechBody(BaseModel):
    """OpenAI ``audio.speech.create`` plus clone-voice controls."""

    model_config = ConfigDict(extra="ignore")

    model: str = "qwen3-tts-1.7b-base"
    input: Annotated[str, Field(min_length=1, description="Text to speak. Long text is chunked by sentence.")]
    voice: Annotated[str, Field(description="Voice id (voice_…) or name. OpenAI preset names map to the default.")]
    response_format: audio.OutputFormat = audio.OutputFormat.MP3
    speed: Annotated[float, Field(ge=0.25, le=4.0)] = 1.0
    stream_format: Literal["audio", "sse"] = "audio"
    instructions: str | None = None
    language: str = "auto"
    voice_mode: Literal["auto", "icl", "xvector"] | None = None
    seed: Annotated[int, Field(ge=0)] = 0
    temperature: Annotated[float, Field(ge=0.0, le=2.0)] = 0.9
    top_p: Annotated[float, Field(gt=0.0, le=1.0)] = 1.0
    save: bool = Field(False, description="Store the result and return file metadata as JSON instead of audio.")


def _speech_request(body: SpeechBody) -> SpeechRequest:
    if body.model not in TTS_MODEL_IDS:
        raise ApiError(422, ErrorCode.UNSUPPORTED_MODEL, f"unknown model {body.model!r}",
                       f"Use one of: {', '.join(sorted(TTS_MODEL_IDS))}.")
    if body.instructions and body.instructions.strip():
        raise ApiError(
            422, ErrorCode.INSTRUCTIONS_UNSUPPORTED,
            "the voice-clone model has no instruction (style) control",
            "Remove instructions; style follows the voice's reference recording.",
        )
    return SpeechRequest(
        text=body.input, voice=body.voice, response_format=body.response_format, speed=body.speed,
        language=body.language, voice_mode=body.voice_mode, seed=body.seed, temperature=body.temperature,
        top_p=body.top_p,
    )


async def _prime(stream: AsyncIterator[Any]) -> AsyncIterator[Any]:
    """Pull the first item before the response starts so early errors become proper HTTP errors."""
    first = await anext(stream, None)

    async def chained() -> AsyncIterator[Any]:
        if first is not None:
            yield first
        async for item in stream:
            yield item

    return chained()


@router.post("/speech", summary="Synthesize speech (OpenAI-compatible)")
async def speech(body: SpeechBody, request: Request) -> Response:
    state = get_state(request)
    speech_request = _speech_request(body)
    if body.save:
        stored = await state.speech.to_file(speech_request)
        return JSONResponse(stored.public(base_url(request)), status_code=201)
    plan = await state.speech.plan(speech_request)
    if body.stream_format == "sse":
        events = await _prime(state.speech.sse(plan))
        return StreamingResponse(events, media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
    chunks = await _prime(state.speech.encoded(plan))
    return StreamingResponse(
        chunks,
        media_type=audio.MEDIA_TYPES[body.response_format],
        headers={"Cache-Control": "no-cache"},
    )


@router.post("/transcriptions", summary="Transcribe audio (OpenAI/Ovis-compatible)")
async def transcriptions(request: Request) -> Response:
    state = get_state(request)
    form = await read_form(request, max_fields=state.settings.limits.max_form_fields)
    upload = form.get("file")
    if not isinstance(upload, UploadFile):
        raise ApiError(422, ErrorCode.VALIDATION_FAILED, "multipart field 'file' is required")
    model = str(form.get("model") or "qwen3-asr-1.7b")
    if model not in ASR_MODEL_IDS:
        raise ApiError(422, ErrorCode.UNSUPPORTED_MODEL, f"unknown model {model!r}",
                       f"Use one of: {', '.join(sorted(ASR_MODEL_IDS))}.")
    try:
        fmt = TranscriptFormat(str(form.get("response_format") or "json"))
    except ValueError as error:
        raise ApiError(
            422, ErrorCode.UNSUPPORTED_FORMAT, "response_format must be json, text, verbose_json, srt or vtt"
        ) from error
    languages = [str(item) for item in form.getlist("languages[]") + form.getlist("languages")]
    language = pick_language(str(form.get("language") or "") or None, languages)
    prompt = str(form.get("prompt") or "") or None
    granularities = {str(item) for item in form.getlist("timestamp_granularities[]")}
    stream = form_bool(form.get("stream"))

    stored = await state.files.save_stream(
        upload_chunks(upload),
        filename=upload.filename or "audio",
        content_type=upload.content_type,
        purpose=FilePurpose.UPLOAD,
        probe_audio=False,
        ttl_s=3600,
        max_bytes=state.settings.limits.max_transcription_upload_bytes,
    )
    if stream:
        events = await _prime(state.transcription.stream(stored.path, language=language, prompt=prompt))
        return StreamingResponse(events, media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
    result = await state.transcription.transcribe(
        stored.path, language=language, prompt=prompt, word_timestamps="word" in granularities,
    )
    body, media_type = render(result, fmt)
    return Response(body, media_type=media_type)


@router.post("/translations", summary="Not supported")
async def translations() -> Response:
    raise ApiError(
        501, ErrorCode.NOT_CONFIGURED, "speech translation is not supported by Qwen3-ASR",
        "Transcribe with /v1/audio/transcriptions and translate the text with an LLM.",
    )


class EnhanceBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_id: str
    bandwidth_extension_hz: int | None = Field(
        None, description=f"Upsample before enhancement to restore bandwidth: one of {BANDWIDTH_EXTENSION_RATES}"
    )
    response_format: Literal["json", "audio"] = "json"


@router.post("/enhance", summary="Clean speech audio with NVIDIA RE-USE")
async def enhance(body: EnhanceBody, request: Request, wait: float = Depends(wait_param)) -> Response:
    state = get_state(request)
    if body.bandwidth_extension_hz is not None and body.bandwidth_extension_hz not in BANDWIDTH_EXTENSION_RATES:
        raise ApiError(
            422, ErrorCode.VALIDATION_FAILED, f"bandwidth_extension_hz must be one of {BANDWIDTH_EXTENSION_RATES}"
        )
    source = await state.files.get(body.file_id)
    short = (source.duration_s or 0.0) <= state.settings.limits.max_sync_enhance_seconds
    if short:
        stored, metrics = await state.enhance.enhance_file(source, bandwidth_extension_hz=body.bandwidth_extension_hz)
        if body.response_format == "audio":
            return Response(stored.path.read_bytes(), media_type="audio/wav", headers={"X-File-ID": stored.id})
        return JSONResponse({"file": stored.public(base_url(request)), "metrics": metrics})

    async def create() -> tuple[int, Any]:
        job = await state.jobs.submit(JobType.AUDIO_ENHANCE, body.model_dump(), resource_id=source.id)
        return job_response(await state.jobs.wait(job["id"], wait))

    return await idempotent(request, "audio.enhance", body.model_dump(), create)


__all__ = ["router"]
