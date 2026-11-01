"""HTTP client for the audio-worker (RE-USE, Qwen3-ASR, ForcedAligner).

Gateway and worker share the data volume, so requests carry paths under the
data directory instead of audio bytes.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx

from voiceapi_gateway.errors import ApiError, ErrorCode


class WorkerClient:
    def __init__(self, base_url: str, *, timeout_s: float, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url,
            transport=transport,
            timeout=httpx.Timeout(timeout_s, connect=5.0),
            limits=httpx.Limits(max_connections=16, max_keepalive_connections=8),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def ready(self) -> tuple[bool, str]:
        try:
            response = await self._client.get("/ready", timeout=3.0)
        except httpx.HTTPError as error:
            return False, f"unreachable: {error.__class__.__name__}"
        return response.status_code == 200, str(response.status_code)

    async def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        try:
            response = await self._client.post(path, json=body)
        except httpx.HTTPError as error:
            raise ApiError(503, ErrorCode.WORKER_UNAVAILABLE, f"audio worker connection failed: {error}") from error
        if response.status_code == 200:
            return response.json()
        self._raise(response)
        raise AssertionError("unreachable")

    @staticmethod
    def _raise(response: httpx.Response) -> None:
        try:
            payload = response.json()
        except ValueError:
            payload = {"detail": response.text[:500]}
        detail = payload.get("detail", payload) if isinstance(payload, dict) else payload
        code = payload.get("code") if isinstance(payload, dict) else None
        if response.status_code == 422:
            mapped = ErrorCode(code) if code in ErrorCode._value2member_map_ else ErrorCode.VALIDATION_FAILED
            raise ApiError(422, mapped, str(detail))
        if response.status_code == 503:
            raise ApiError(503, ErrorCode.WORKER_UNAVAILABLE, f"audio worker unavailable: {detail}", "Retry shortly.")
        raise ApiError(502, ErrorCode.UPSTREAM_ERROR, f"audio worker returned {response.status_code}: {detail}")

    async def enhance(self, input_path: Path, output_path: Path, *, bandwidth_extension_hz: int | None) -> dict:
        return await self._post(
            "/v1/enhance",
            {
                "input_path": str(input_path),
                "output_path": str(output_path),
                "bandwidth_extension_hz": bandwidth_extension_hz,
            },
        )

    async def transcribe(
        self,
        input_path: Path,
        *,
        language: str | None,
        prompt: str | None,
        word_timestamps: bool,
    ) -> dict:
        return await self._post(
            "/v1/transcribe",
            {
                "input_path": str(input_path),
                "language": language,
                "prompt": prompt,
                "word_timestamps": word_timestamps,
            },
        )

    async def transcribe_stream(
        self, input_path: Path, *, language: str | None, prompt: str | None
    ) -> AsyncIterator[dict]:
        """Yield ``{"delta": str}`` events then one ``{"done": true, "text": ..., "language": ...}``."""
        body = {"input_path": str(input_path), "language": language, "prompt": prompt, "stream": True}
        try:
            async with self._client.stream("POST", "/v1/transcribe", json=body) as response:
                if response.status_code != 200:
                    await response.aread()
                    self._raise(response)
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        yield json.loads(line[6:])
        except httpx.HTTPError as error:
            raise ApiError(503, ErrorCode.WORKER_UNAVAILABLE, f"audio worker connection failed: {error}") from error

    async def prepare_reference(
        self,
        input_paths: list[Path],
        output_dir: Path,
        *,
        transcript: str | None,
        language: str | None,
        clean: bool,
    ) -> dict:
        return await self._post(
            "/v1/reference",
            {
                "input_paths": [str(path) for path in input_paths],
                "output_dir": str(output_dir),
                "transcript": transcript,
                "language": language,
                "clean": clean,
            },
        )


__all__ = ["WorkerClient"]
