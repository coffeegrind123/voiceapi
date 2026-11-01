"""Text-to-speech orchestration: voice resolution, chunking, engine streaming, encoding."""

from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass

from voiceapi_gateway import audio
from voiceapi_gateway.clients.engine import EngineClient, EngineVoiceMissing, SynthesisParams
from voiceapi_gateway.errors import ApiError, ErrorCode
from voiceapi_gateway.ids import engine_voice_handle
from voiceapi_gateway.services.files import GENERATED_TTL_S, FilePurpose, FileStore, StoredFile
from voiceapi_gateway.services.voices import VoiceService
from voiceapi_gateway.text_split import split_for_speech

# OpenAI voice names that SDKs send by default. They map to the configured
# default voice so stock clients work unchanged.
OPENAI_PRESET_VOICES = frozenset(
    {"alloy", "ash", "ballad", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer", "verse", "marin", "cedar"}
)

# The engine accepts these language names; ISO codes and capitalised names map onto them.
LANGUAGES = {
    "auto": "auto", "zh": "chinese", "chinese": "chinese", "en": "english", "english": "english",
    "fr": "french", "french": "french", "de": "german", "german": "german", "it": "italian",
    "italian": "italian", "ja": "japanese", "japanese": "japanese", "ko": "korean", "korean": "korean",
    "pt": "portuguese", "portuguese": "portuguese", "ru": "russian", "russian": "russian",
    "es": "spanish", "spanish": "spanish",
}


def normalize_language(value: str | None) -> str:
    key = (value or "auto").strip().lower()
    key = key.split("-")[0] if key not in LANGUAGES else key
    if key not in LANGUAGES:
        raise ApiError(
            422, ErrorCode.UNSUPPORTED_LANGUAGE, f"unsupported language {value!r}",
            f"Use one of: {', '.join(sorted(set(LANGUAGES.values())))} (or ISO codes).",
        )
    return LANGUAGES[key]


@dataclass(frozen=True, slots=True)
class SpeechRequest:
    text: str
    voice: str
    response_format: audio.OutputFormat = audio.OutputFormat.MP3
    speed: float = 1.0
    language: str = "auto"
    voice_mode: str | None = None
    seed: int = 0
    temperature: float = 0.9
    top_p: float = 1.0


@dataclass(frozen=True, slots=True)
class SpeechPlan:
    request: SpeechRequest
    voice_row: dict
    params: list[SynthesisParams]


class SpeechService:
    def __init__(
        self,
        *,
        engine: EngineClient,
        voices: VoiceService,
        files: FileStore,
        default_voice: str | None,
        max_input_chars: int,
    ) -> None:
        self.engine = engine
        self.voices = voices
        self.files = files
        self.default_voice = default_voice
        self.max_input_chars = max_input_chars

    async def _voice_row(self, voice: str) -> dict:
        ref = (voice or "").strip()
        if not ref and self.default_voice:
            ref = self.default_voice
        if not ref:
            raise ApiError(422, ErrorCode.VALIDATION_FAILED, "voice is required", "Pass a voice id or name.")
        try:
            return await self.voices.resolve_for_speech(ref)
        except ApiError as error:
            if error.code is ErrorCode.VOICE_NOT_FOUND and ref.lower() in OPENAI_PRESET_VOICES and self.default_voice:
                return await self.voices.resolve_for_speech(self.default_voice)
            raise

    def _validate(self, request: SpeechRequest) -> list[str]:
        if not request.text or not request.text.strip():
            raise ApiError(422, ErrorCode.VALIDATION_FAILED, "input text is empty")
        if len(request.text) > self.max_input_chars:
            raise ApiError(
                413, ErrorCode.PAYLOAD_TOO_LARGE, f"input exceeds {self.max_input_chars} characters",
                "Split the text across several requests.",
            )
        audio.atempo_filter(request.speed)
        return split_for_speech(request.text)

    async def plan(self, request: SpeechRequest) -> SpeechPlan:
        """Validate and resolve everything that can fail before any audio byte is sent."""
        chunks = self._validate(request)
        row = await self._voice_row(request.voice)
        mode = request.voice_mode or row["default_mode"]
        if mode == "icl" and not row["supports_icl"]:
            raise ApiError(422, ErrorCode.VALIDATION_FAILED, "this voice has no reference transcript for icl mode")
        language = normalize_language(request.language if request.language != "auto" else row["language"])
        handle = engine_voice_handle(row["id"])
        params = [
            SynthesisParams(
                text=chunk_text, voice=handle, language=language, voice_mode=mode, seed=request.seed,
                temperature=float(request.temperature), top_p=float(request.top_p),
            )
            for chunk_text in chunks
        ]
        return SpeechPlan(request=request, voice_row=row, params=params)

    async def pcm(self, plan: SpeechPlan) -> AsyncIterator[bytes]:
        """Engine PCM for the whole input, chunk after chunk, re-registering an evicted voice once."""
        row = plan.voice_row
        for params in plan.params:
            for attempt in range(2):
                started = False
                try:
                    async for pcm in self.engine.synthesize(params):
                        started = True
                        yield pcm
                    break
                except EngineVoiceMissing:
                    if started or attempt:
                        raise ApiError(503, ErrorCode.ENGINE_UNAVAILABLE, "engine lost the voice mid-request") from None
                    await self.voices.ensure_resident(row, force=True)

    async def encoded(self, plan: SpeechPlan) -> AsyncIterator[bytes]:
        request = plan.request
        async for data in audio.transcode_stream(self.pcm(plan), request.response_format, speed=request.speed):
            yield data

    async def sse(self, plan: SpeechPlan) -> AsyncIterator[str]:
        """OpenAI ``stream_format=sse``: base64 audio deltas then a done event."""
        total = 0
        async for data in self.encoded(plan):
            total += len(data)
            yield f"data: {json.dumps({'type': 'speech.audio.delta', 'audio': base64.b64encode(data).decode()})}\n\n"
        yield f"data: {json.dumps({'type': 'speech.audio.done', 'usage': {'output_bytes': total}})}\n\n"

    async def to_bytes(self, request: SpeechRequest) -> bytes:
        plan = await self.plan(request)
        pcm = b"".join([chunk async for chunk in self.pcm(plan)])
        return await audio.encode_pcm(pcm, request.response_format, speed=request.speed)

    async def to_file(self, request: SpeechRequest) -> StoredFile:
        data = await self.to_bytes(request)
        extension = {audio.OutputFormat.OPUS: "ogg", audio.OutputFormat.PCM: "pcm"}.get(
            request.response_format, request.response_format.value
        )
        return await self.files.save_bytes(
            data,
            filename=f"speech.{extension}",
            content_type=audio.MEDIA_TYPES[request.response_format],
            purpose=FilePurpose.SPEECH,
            probe_audio=request.response_format is not audio.OutputFormat.PCM,
            ttl_s=GENERATED_TTL_S,
        )


__all__ = [
    "LANGUAGES",
    "OPENAI_PRESET_VOICES",
    "SpeechPlan",
    "SpeechRequest",
    "SpeechService",
    "normalize_language",
]
