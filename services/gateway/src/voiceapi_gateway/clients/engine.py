"""HTTP client for the tts-engine (Nari fork) internal API."""

from __future__ import annotations

import base64
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx

from voiceapi_gateway.errors import ApiError, ErrorCode

ENGINE_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"


class EngineVoiceMissing(Exception):  # noqa: N818 - control-flow signal, not an error to surface
    """The engine does not hold this voice (restart or LRU eviction); re-register and retry."""


@dataclass(frozen=True, slots=True)
class EngineVoice:
    voice_id: str
    supports_icl: bool
    frames: int


@dataclass(frozen=True, slots=True)
class SynthesisParams:
    text: str
    voice: str
    language: str = "auto"
    voice_mode: str = "auto"
    seed: int = 0
    temperature: float = 0.9
    top_p: float = 1.0
    top_k: int = 50
    repetition_penalty: float = 1.05
    max_new_tokens: int = 2048
    non_streaming_mode: bool = False

    def body(self) -> dict[str, Any]:
        return {
            "model": ENGINE_MODEL_ID,
            "input": self.text,
            "voice": self.voice,
            "language": self.language,
            "voice_mode": self.voice_mode,
            "seed": self.seed,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "max_new_tokens": self.max_new_tokens,
            "non_streaming_mode": self.non_streaming_mode,
            "response_format": "pcm",
            "stream": True,
        }


def _detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text[:500]
    if isinstance(payload, dict):
        return str(payload.get("detail", payload))
    return str(payload)


class EngineClient:
    def __init__(self, base_url: str, *, timeout_s: float, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url,
            transport=transport,
            timeout=httpx.Timeout(timeout_s, connect=5.0),
            limits=httpx.Limits(max_connections=64, max_keepalive_connections=16),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def ready(self) -> tuple[bool, str]:
        try:
            response = await self._client.get("/ready", timeout=3.0)
        except httpx.HTTPError as error:
            return False, f"unreachable: {error.__class__.__name__}"
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        return response.status_code == 200, str(payload.get("reason") or payload.get("phase") or response.status_code)

    def _raise(self, response: httpx.Response, *, voice: str | None = None) -> None:
        detail = _detail(response)
        status = response.status_code
        if status == 422 and voice and "unknown voice" in detail:
            raise EngineVoiceMissing(voice)
        if status == 422:
            raise ApiError(422, ErrorCode.VALIDATION_FAILED, f"engine rejected the request: {detail}")
        if status == 409:
            raise ApiError(409, ErrorCode.VOICE_IN_USE, detail, "Retry once in-flight speech for this voice finishes.")
        if status == 507:
            raise ApiError(507, ErrorCode.CAPACITY_EXCEEDED, detail, "Delete unused voices or shorten the reference.")
        if status == 429:
            raise ApiError(429, ErrorCode.CAPACITY_EXCEEDED, "engine is at capacity", "Retry with backoff.")
        if status == 503:
            raise ApiError(503, ErrorCode.ENGINE_UNAVAILABLE, f"engine unavailable: {detail}", "Retry shortly.")
        raise ApiError(502, ErrorCode.UPSTREAM_ERROR, f"engine returned {status}: {detail}")

    async def synthesize(self, params: SynthesisParams) -> AsyncIterator[bytes]:
        """Stream 24 kHz mono s16le PCM for one engine-sized text chunk."""
        try:
            async with self._client.stream("POST", "/v1/audio/speech", json=params.body()) as response:
                if response.status_code != 200:
                    await response.aread()
                    self._raise(response, voice=params.voice)
                carry = b""
                async for chunk in response.aiter_bytes():
                    data = carry + chunk
                    # Keep sample alignment so downstream consumers never see half a sample.
                    cut = len(data) - (len(data) % 2)
                    carry = data[cut:]
                    if cut:
                        yield data[:cut]
        except httpx.HTTPError as error:
            raise ApiError(503, ErrorCode.ENGINE_UNAVAILABLE, f"engine connection failed: {error}") from error

    async def build_voice(
        self, handle: str, wav: bytes, *, ref_text: str | None, x_vector_only: bool, xvector_wav: bytes | None = None
    ) -> tuple[EngineVoice, bytes]:
        body = {
            "voice_id": handle,
            "audio_base64": base64.b64encode(wav).decode("ascii"),
            "ref_text": ref_text,
            "x_vector_only": x_vector_only,
        }
        if xvector_wav is not None:
            body["xvector_audio_base64"] = base64.b64encode(xvector_wav).decode("ascii")
        try:
            response = await self._client.post("/v1/voices", json=body)
        except httpx.HTTPError as error:
            raise ApiError(503, ErrorCode.ENGINE_UNAVAILABLE, f"engine connection failed: {error}") from error
        if response.status_code != 201:
            self._raise(response)
        payload = response.json()
        voice = payload["voice"]
        return (
            EngineVoice(voice["voice_id"], bool(voice["supports_icl"]), int(voice["frames"])),
            base64.b64decode(payload["prompt_base64"]),
        )

    async def put_voice(self, handle: str, prompt: bytes) -> EngineVoice:
        try:
            response = await self._client.put(
                f"/v1/voices/{handle}", content=prompt, headers={"Content-Type": "application/octet-stream"}
            )
        except httpx.HTTPError as error:
            raise ApiError(503, ErrorCode.ENGINE_UNAVAILABLE, f"engine connection failed: {error}") from error
        if response.status_code != 200:
            self._raise(response)
        voice = response.json()
        return EngineVoice(voice["voice_id"], bool(voice["supports_icl"]), int(voice["frames"]))

    async def delete_voice(self, handle: str) -> None:
        try:
            response = await self._client.delete(f"/v1/voices/{handle}")
        except httpx.HTTPError as error:
            raise ApiError(503, ErrorCode.ENGINE_UNAVAILABLE, f"engine connection failed: {error}") from error
        if response.status_code not in (204, 404):
            self._raise(response)

    async def resident_voices(self) -> set[str]:
        try:
            response = await self._client.get("/v1/voices")
        except httpx.HTTPError as error:
            raise ApiError(503, ErrorCode.ENGINE_UNAVAILABLE, f"engine connection failed: {error}") from error
        if response.status_code != 200:
            self._raise(response)
        return {item["voice_id"] for item in response.json()["data"]}


__all__ = ["ENGINE_MODEL_ID", "EngineClient", "EngineVoice", "EngineVoiceMissing", "SynthesisParams"]
