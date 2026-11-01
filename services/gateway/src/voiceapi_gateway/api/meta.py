"""Health, discovery and the optional chat passthrough."""

from __future__ import annotations

import time
from importlib import resources
from typing import Any

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse

from voiceapi_gateway import audio
from voiceapi_gateway.api.audio import ASR_MODEL_IDS, TTS_MODEL_IDS
from voiceapi_gateway.api.common import Authenticated, base_url, get_state
from voiceapi_gateway.errors import ApiError, ErrorCode
from voiceapi_gateway.services.enhance import BANDWIDTH_EXTENSION_RATES
from voiceapi_gateway.services.speech import LANGUAGES
from voiceapi_gateway.services.voices import CLONE_MODES

public = APIRouter(tags=["meta"])
private = APIRouter(prefix="/v1", tags=["meta"], dependencies=[Authenticated])

_CREATED = int(time.time())


@public.get("/health", summary="Liveness")
async def health() -> dict[str, Any]:
    return {"alive": True}


@public.get("/ready", summary="Readiness: engine and worker reachable and ready")
async def ready(request: Request) -> JSONResponse:
    state = get_state(request)
    engine_ready, engine_reason = await state.engine.ready()
    worker_ready, worker_reason = await state.worker.ready()
    body = {
        "ready": engine_ready and worker_ready,
        "engine": {"ready": engine_ready, "detail": engine_reason},
        "worker": {"ready": worker_ready, "detail": worker_reason},
    }
    return JSONResponse(body, status_code=200 if body["ready"] else 503)


@public.get("/llms.txt", summary="Plain-text guide for AI agents", response_class=PlainTextResponse)
async def llms_txt(request: Request) -> PlainTextResponse:
    text = resources.files("voiceapi_gateway").joinpath("llms.txt").read_text(encoding="utf-8")
    return PlainTextResponse(text.replace("{BASE_URL}", base_url(request)))


@private.get("/models", summary="OpenAI-compatible model list")
async def models() -> dict[str, Any]:
    ids = sorted(TTS_MODEL_IDS) + sorted(ASR_MODEL_IDS) + ["nvidia-re-use"]
    return {
        "object": "list",
        "data": [{"id": model_id, "object": "model", "created": _CREATED, "owned_by": "voiceapi"} for model_id in ids],
    }


@private.get("/capabilities", summary="Machine-readable limits, formats, languages and modes")
async def capabilities(request: Request) -> dict[str, Any]:
    limits = get_state(request).settings.limits
    return {
        "speech": {
            "models": sorted(TTS_MODEL_IDS),
            "response_formats": [fmt.value for fmt in audio.OutputFormat],
            "stream_formats": ["audio", "sse"],
            "languages": sorted(set(LANGUAGES.values())),
            "voice_modes": list(CLONE_MODES),
            "speed": {"min": 0.25, "max": 4.0},
            "max_input_chars": limits.max_speech_input_chars,
            "sample_rate": audio.ENGINE_SAMPLE_RATE,
            "instructions": False,
            "default_voice": (await get_state(request).voices.default_voice() or {}).get("id")
            or get_state(request).settings.default_voice,
        },
        "transcription": {
            "models": sorted(ASR_MODEL_IDS),
            "response_formats": ["json", "text", "verbose_json", "srt", "vtt"],
            "timestamp_granularities": ["word", "segment"],
            "streaming": True,
            "max_upload_bytes": limits.max_transcription_upload_bytes,
        },
        "voices": {
            "max_clips": limits.max_voice_clips,
            "max_total_seconds": limits.max_voice_total_seconds,
            "recommended_reference_seconds": [5, 15],
            "cleaning": "nvidia/NVIDIA-NemotronLabs-RE-USE",
        },
        "enhance": {
            "model": "nvidia/NVIDIA-NemotronLabs-RE-USE",
            "max_sync_seconds": limits.max_sync_enhance_seconds,
            "bandwidth_extension_hz": list(BANDWIDTH_EXTENSION_RATES),
        },
        "files": {"max_upload_bytes": limits.max_upload_bytes},
        "jobs": {"max_wait_seconds": limits.max_wait_seconds},
        "mcp": {"url": f"{base_url(request)}/mcp", "transport": "streamable-http"},
    }


@private.post("/chat/completions", summary="Optional passthrough to an upstream chat model (Ovis transcript polish)")
async def chat_completions(request: Request) -> Response:
    settings = get_state(request).settings
    if not settings.chat_upstream:
        raise ApiError(
            501, ErrorCode.NOT_CONFIGURED, "no chat upstream configured",
            "Set VOICEAPI_CHAT_UPSTREAM to an OpenAI-compatible base URL to enable this passthrough.",
        )
    body = await request.body()
    headers = {"Content-Type": "application/json"}
    if settings.chat_upstream_key:
        headers["Authorization"] = f"Bearer {settings.chat_upstream_key}"
    client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=5.0))
    upstream = client.build_request("POST", f"{settings.chat_upstream.rstrip('/')}/chat/completions",
                                    content=body, headers=headers)
    try:
        response = await client.send(upstream, stream=True)
    except httpx.HTTPError as error:
        await client.aclose()
        raise ApiError(502, ErrorCode.UPSTREAM_ERROR, f"chat upstream failed: {error}") from error

    async def relay():
        try:
            async for chunk in response.aiter_raw():
                yield chunk
        finally:
            await response.aclose()
            await client.aclose()

    return StreamingResponse(
        relay(), status_code=response.status_code,
        media_type=response.headers.get("content-type", "application/json"),
    )


__all__ = ["private", "public"]
