"""MCP (streamable HTTP) surface: the same services the REST routes use.

Audio never travels inline: tools return file ids and download URLs so an
agent's context holds a link, not megabytes of base64.
"""

from __future__ import annotations

import base64
import binascii
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from mcp_types import ToolAnnotations
from starlette.types import ASGIApp, Receive, Scope, Send

from voiceapi_gateway import audio
from voiceapi_gateway.api.common import check_bearer
from voiceapi_gateway.errors import ApiError
from voiceapi_gateway.services.files import FilePurpose
from voiceapi_gateway.services.speech import SpeechRequest
from voiceapi_gateway.state import AppState

INSTRUCTIONS = """\
Self-hosted voice service: text-to-speech with cloned voices (Qwen3-TTS), speech-to-text (Qwen3-ASR)
and speech cleanup (NVIDIA RE-USE).

Typical flows:
- Speak: list_voices -> speak(text, voice) -> download the returned url.
- Clone: upload_audio (or use an existing file_id) -> create_voice(name, file_ids)
  -> get_job(job_id, wait_seconds=60) until status is succeeded -> speak with the new voice.
- Transcribe: upload_audio -> transcribe(file_id).
Audio results are returned as file ids plus URLs (GET with the same bearer token); never inline.
"""

READ_ONLY = ToolAnnotations(readOnlyHint=True, openWorldHint=False)
CREATES = ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=False)
DESTROYS = ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=True, openWorldHint=False)
MAX_INLINE_UPLOAD_BYTES = 25 * 1024 * 1024


def _tool_error(error: ApiError) -> ToolError:
    hint = f" Hint: {error.hint}" if error.hint else ""
    return ToolError(f"[{error.code.value}] {error.detail}.{hint}")


def build_mcp(state: AppState, base_url: str) -> MCPServer:
    mcp = MCPServer(name="voiceapi", title="Voice API", instructions=INSTRUCTIONS, version="0.1.0")
    _register_audio_tools(mcp, state, base_url)
    _register_voice_tools(mcp, state)
    return mcp


def _register_audio_tools(mcp: MCPServer, state: AppState, base_url: str) -> None:
    def file_out(stored) -> dict[str, Any]:
        return stored.public(base_url)

    @mcp.tool(annotations=CREATES)
    async def speak(
        *,
        text: str,
        voice: str | None = None,
        format: str = "mp3",
        language: str = "auto",
        speed: float = 1.0,
        voice_mode: str | None = None,
    ) -> dict[str, Any]:
        """Synthesize speech with a cloned voice and store it as a file.

        voice: voice id (voice_...) or name from list_voices; omit to use the default voice.
        format: mp3|opus|aac|flac|wav|pcm.
        language: auto or a language name/ISO code. speed: 0.25-4.0. voice_mode: auto|icl|xvector.
        Returns file metadata with a download url and duration_s.
        """
        try:
            fmt = audio.OutputFormat(format)
        except ValueError as error:
            raise ToolError(f"format must be one of {[f.value for f in audio.OutputFormat]}") from error
        try:
            stored = await state.speech.to_file(
                SpeechRequest(text=text, voice=voice, response_format=fmt, speed=speed, language=language,
                              voice_mode=voice_mode)
            )
        except ApiError as error:
            raise _tool_error(error) from error
        return file_out(stored)

    @mcp.tool(annotations=READ_ONLY)
    async def transcribe(*, file_id: str, language: str | None = None, prompt: str | None = None,
                         word_timestamps: bool = False) -> dict[str, Any]:
        """Transcribe an uploaded audio file. language forces a language; prompt biases vocabulary.

        Returns {text, language, duration, words?}.
        """
        try:
            stored = await state.files.get(file_id)
            return await state.transcription.transcribe(
                stored.path, language=language, prompt=prompt, word_timestamps=word_timestamps
            )
        except ApiError as error:
            raise _tool_error(error) from error

    @mcp.tool(annotations=CREATES)
    async def enhance_audio(*, file_id: str, bandwidth_extension_hz: int | None = None) -> dict[str, Any]:
        """Remove noise, reverb and codec damage from speech (NVIDIA RE-USE). Returns the cleaned file."""
        try:
            source = await state.files.get(file_id)
            stored, metrics = await state.enhance.enhance_file(source, bandwidth_extension_hz=bandwidth_extension_hz)
        except ApiError as error:
            raise _tool_error(error) from error
        return {"file": file_out(stored), "metrics": metrics}

    @mcp.tool(annotations=CREATES)
    async def upload_audio(*, data_base64: str, filename: str) -> dict[str, Any]:
        """Upload audio for agents that cannot POST multipart. Prefer HTTP POST /v1/files for large files."""
        try:
            data = base64.b64decode(data_base64, validate=True)
        except (binascii.Error, ValueError) as error:
            raise ToolError("data_base64 is not valid base64") from error
        if len(data) > MAX_INLINE_UPLOAD_BYTES:
            raise ToolError(f"inline uploads are limited to {MAX_INLINE_UPLOAD_BYTES} bytes; use POST /v1/files")
        try:
            stored = await state.files.save_bytes(data, filename=filename, content_type=None,
                                                  purpose=FilePurpose.UPLOAD)
        except ApiError as error:
            raise _tool_error(error) from error
        return file_out(stored)


def _register_voice_tools(mcp: MCPServer, state: AppState) -> None:
    @mcp.tool(annotations=CREATES)
    async def create_voice(
        *,
        name: str,
        file_ids: list[str],
        transcript: str | None = None,
        language: str = "auto",
        clean: bool = True,
        mode: str = "auto",
        description: str = "",
        make_default: bool = True,
    ) -> dict[str, Any]:
        """Clone a voice from 1-20 clips (5-15 s of clean single-speaker speech is ideal).

        transcript only with one clip (omit to auto-transcribe). clean runs RE-USE first.
        mode: auto|icl|xvector. make_default (default true) makes it the voice every later session
        uses when none is named, once built. Returns {voice, job}; poll get_job(job.id) until succeeded.
        """
        try:
            voice, job = await state.voices.create(
                name=name, file_ids=file_ids, transcript=transcript, language=language,
                clean=clean, mode=mode, description=description, make_default=make_default,
            )
        except ApiError as error:
            raise _tool_error(error) from error
        return {"voice": voice, "job": job}

    @mcp.tool(annotations=READ_ONLY)
    async def get_default_voice() -> dict[str, Any]:
        """The voice used whenever a request names none. It persists across sessions and clients."""
        return {"default_voice": await state.voices.default_voice()}

    @mcp.tool(annotations=CREATES)
    async def set_default_voice(*, voice: str | None) -> dict[str, Any]:
        """Make a voice (id or name) the default for every session and client; pass null to turn it off."""
        try:
            return {"default_voice": await state.voices.set_default(voice)}
        except ApiError as error:
            raise _tool_error(error) from error

    @mcp.tool(annotations=READ_ONLY)
    async def list_voices(*, status: str | None = "ready", limit: int = 50) -> dict[str, Any]:
        """List voices (default: ready ones). Use a voice's id or name with speak."""
        voices, has_more = await state.voices.list(limit=max(1, min(limit, 100)), after=None, status=status)
        return {"voices": voices, "has_more": has_more}

    @mcp.tool(annotations=READ_ONLY)
    async def get_voice(*, voice: str) -> dict[str, Any]:
        """Get one voice by id or name, including its reference transcript and quality metrics."""
        try:
            return await state.voices.get(voice)
        except ApiError as error:
            raise _tool_error(error) from error

    @mcp.tool(annotations=DESTROYS)
    async def delete_voice(*, voice: str) -> dict[str, Any]:
        """Permanently delete a voice (by id or name) and its stored reference."""
        try:
            await state.voices.delete(voice)
        except ApiError as error:
            raise _tool_error(error) from error
        return {"deleted": voice}

    @mcp.tool(annotations=READ_ONLY)
    async def get_job(*, job_id: str, wait_seconds: float = 0.0) -> dict[str, Any]:
        """Get a background job. wait_seconds (max 60) blocks until it finishes or the time passes."""
        try:
            return await state.jobs.wait(job_id, max(0.0, min(wait_seconds, 60.0)))
        except ApiError as error:
            raise _tool_error(error) from error

    @mcp.tool(annotations=READ_ONLY)
    async def get_capabilities() -> dict[str, Any]:
        """Supported languages, formats, limits and clone modes."""
        return {
            "speech_formats": [fmt.value for fmt in audio.OutputFormat],
            "voice_modes": ["auto", "icl", "xvector"],
            "instructions_supported": False,
            "limits": {
                "max_speech_chars": state.settings.limits.max_speech_input_chars,
                "max_voice_clips": state.settings.limits.max_voice_clips,
                "max_upload_bytes": state.settings.limits.max_upload_bytes,
            },
        }


def mcp_transport_security(state: AppState) -> TransportSecuritySettings:
    """Host-header checks guard local MCP servers against DNS rebinding; ours sits behind bearer auth."""
    hosts = list(state.settings.mcp_allowed_hosts)
    if not hosts:
        return TransportSecuritySettings(enable_dns_rebinding_protection=False)
    return TransportSecuritySettings(enable_dns_rebinding_protection=True, allowed_hosts=hosts, allowed_origins=[])


class BearerGuard:
    """ASGI wrapper applying the gateway's API keys to the mounted MCP app."""

    def __init__(self, app: ASGIApp, state: AppState) -> None:
        self.app = app
        self.state = state

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            headers = {key.decode().lower(): value.decode() for key, value in scope.get("headers", [])}
            try:
                check_bearer(self.state, headers.get("authorization"))
            except ApiError as error:
                body = (b'{"error":"unauthorized","detail":"' + error.detail.encode() + b'"}')
                await send({"type": "http.response.start", "status": 401,
                            "headers": [(b"content-type", b"application/json"),
                                        (b"www-authenticate", b"Bearer")]})
                await send({"type": "http.response.body", "body": body})
                return
        await self.app(scope, receive, send)


__all__ = ["BearerGuard", "build_mcp", "mcp_transport_security"]
