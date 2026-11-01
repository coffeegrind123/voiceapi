"""ffmpeg/ffprobe wrappers: probing, decoding, encoding and streaming transcode.

All engine audio is 24 kHz mono signed 16-bit little-endian PCM.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import struct
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from voiceapi_gateway.errors import ApiError, ErrorCode

ENGINE_SAMPLE_RATE = 24_000
PCM_SAMPLE_WIDTH = 2
_PIPE_CHUNK = 64 * 1024


class OutputFormat(StrEnum):
    MP3 = "mp3"
    OPUS = "opus"
    AAC = "aac"
    FLAC = "flac"
    WAV = "wav"
    PCM = "pcm"


MEDIA_TYPES = {
    OutputFormat.MP3: "audio/mpeg",
    OutputFormat.OPUS: "audio/ogg",
    OutputFormat.AAC: "audio/aac",
    OutputFormat.FLAC: "audio/flac",
    OutputFormat.WAV: "audio/wav",
    OutputFormat.PCM: "audio/pcm",
}

_ENCODER_ARGS = {
    OutputFormat.MP3: ["-c:a", "libmp3lame", "-b:a", "128k", "-f", "mp3"],
    OutputFormat.OPUS: ["-c:a", "libopus", "-b:a", "64k", "-f", "ogg"],
    OutputFormat.AAC: ["-c:a", "aac", "-b:a", "128k", "-f", "adts"],
    OutputFormat.FLAC: ["-c:a", "flac", "-f", "flac"],
}


@dataclass(frozen=True, slots=True)
class AudioInfo:
    duration_s: float
    sample_rate: int
    channels: int
    codec: str
    container: str


def wav_header(pcm_bytes: int | None, *, sample_rate: int = ENGINE_SAMPLE_RATE, channels: int = 1) -> bytes:
    """RIFF header; ``None`` length writes the streaming sentinel 0xFFFFFFFF."""
    data_size = 0xFFFFFFFF - 36 if pcm_bytes is None else pcm_bytes
    byte_rate = sample_rate * channels * PCM_SAMPLE_WIDTH
    return (
        b"RIFF"
        + struct.pack("<I", (36 + data_size) & 0xFFFFFFFF)
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, channels, sample_rate, byte_rate, channels * PCM_SAMPLE_WIDTH, 16)
        + b"data"
        + struct.pack("<I", data_size & 0xFFFFFFFF)
    )


async def _run(args: list[str], stdin: bytes | None = None, *, timeout_s: float = 600.0) -> bytes:
    process = await asyncio.create_subprocess_exec(
        *args,
        stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(stdin), timeout=timeout_s)
    except TimeoutError:
        process.kill()
        await process.wait()
        raise ApiError(500, ErrorCode.INTERNAL, f"{args[0]} timed out after {timeout_s:.0f} s") from None
    if process.returncode != 0:
        message = stderr.decode("utf-8", "replace").strip().splitlines()[-3:]
        raise ApiError(
            422,
            ErrorCode.AUDIO_UNREADABLE,
            f"{args[0]} failed: {' | '.join(message) or 'no diagnostics'}",
            "Send a common audio format: wav, flac, mp3, ogg/opus, webm, m4a/mp4.",
        )
    return stdout


async def probe(path: Path) -> AudioInfo:
    raw = await _run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", "-select_streams", "a:0",
         str(path)],
        timeout_s=60,
    )
    data = json.loads(raw or b"{}")
    streams = data.get("streams") or []
    if not streams:
        raise ApiError(422, ErrorCode.AUDIO_UNREADABLE, "no audio stream found", "Upload a file that contains audio.")
    stream = streams[0]
    fmt = data.get("format") or {}
    duration = float(stream.get("duration") or fmt.get("duration") or 0.0)
    if duration <= 0:
        # Some streamed containers (webm from MediaRecorder) carry no duration; decode to measure.
        pcm = await decode_pcm(path, sample_rate=16_000)
        duration = len(pcm) / (16_000 * PCM_SAMPLE_WIDTH)
    return AudioInfo(
        duration_s=duration,
        sample_rate=int(stream.get("sample_rate") or 0),
        channels=int(stream.get("channels") or 0),
        codec=str(stream.get("codec_name") or "unknown"),
        container=str(fmt.get("format_name") or "unknown"),
    )


async def decode_pcm(source: Path | bytes, *, sample_rate: int = ENGINE_SAMPLE_RATE, mono: bool = True) -> bytes:
    """Decode anything ffmpeg reads to raw s16le PCM."""
    stdin = source if isinstance(source, bytes) else None
    input_arg = "pipe:0" if stdin is not None else str(source)
    args = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", input_arg, "-vn"]
    if mono:
        args += ["-ac", "1"]
    args += ["-ar", str(sample_rate), "-f", "s16le", "-acodec", "pcm_s16le", "pipe:1"]
    return await _run(args, stdin)


async def decode_wav(source: Path | bytes, *, sample_rate: int | None = None) -> bytes:
    """Decode to a mono 16-bit WAV, keeping the native rate unless one is given."""
    stdin = source if isinstance(source, bytes) else None
    input_arg = "pipe:0" if stdin is not None else str(source)
    args = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", input_arg, "-vn", "-ac", "1"]
    if sample_rate is not None:
        args += ["-ar", str(sample_rate)]
    args += ["-c:a", "pcm_s16le", "-f", "wav", "pipe:1"]
    return await _run(args, stdin)


def atempo_filter(speed: float) -> str | None:
    """ffmpeg atempo chain; each stage stays inside the universally supported [0.5, 2.0]."""
    if not 0.25 <= speed <= 4.0:
        raise ApiError(422, ErrorCode.VALIDATION_FAILED, "speed must be between 0.25 and 4.0")
    if abs(speed - 1.0) < 1e-6:
        return None
    stages = []
    remaining = speed
    while remaining > 2.0:
        stages.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        stages.append(0.5)
        remaining /= 0.5
    stages.append(remaining)
    return ",".join(f"atempo={stage:.6f}" for stage in stages)


def _encode_args(fmt: OutputFormat, *, speed: float) -> list[str]:
    args = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-f", "s16le", "-ar", str(ENGINE_SAMPLE_RATE), "-ac", "1", "-i", "pipe:0",
    ]
    tempo = atempo_filter(speed)
    if tempo:
        args += ["-filter:a", tempo]
    if fmt in _ENCODER_ARGS:
        args += _ENCODER_ARGS[fmt]
    elif fmt is OutputFormat.WAV:
        args += ["-c:a", "pcm_s16le", "-f", "wav"]
    else:
        args += ["-c:a", "pcm_s16le", "-f", "s16le"]
    args.append("pipe:1")
    return args


def needs_transcode(fmt: OutputFormat, speed: float) -> bool:
    return fmt not in (OutputFormat.PCM, OutputFormat.WAV) or atempo_filter(speed) is not None


async def encode_pcm(pcm: bytes, fmt: OutputFormat, *, speed: float = 1.0) -> bytes:
    if not needs_transcode(fmt, speed):
        return pcm if fmt is OutputFormat.PCM else wav_header(len(pcm)) + pcm
    return await _run(_encode_args(fmt, speed=speed), pcm)


async def transcode_stream(
    pcm_chunks: AsyncIterator[bytes],
    fmt: OutputFormat,
    *,
    speed: float = 1.0,
) -> AsyncIterator[bytes]:
    """Pipe engine PCM through ffmpeg while it is still being generated."""
    if not needs_transcode(fmt, speed):
        if fmt is OutputFormat.WAV:
            yield wav_header(None)
        async for chunk in pcm_chunks:
            yield chunk
        return

    process = await asyncio.create_subprocess_exec(
        *_encode_args(fmt, speed=speed),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert process.stdin is not None and process.stdout is not None

    async def feed() -> None:
        try:
            async for chunk in pcm_chunks:
                process.stdin.write(chunk)
                await process.stdin.drain()
        finally:
            with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                process.stdin.close()

    feeder = asyncio.create_task(feed())
    try:
        while True:
            data = await process.stdout.read(_PIPE_CHUNK)
            if not data:
                break
            yield data
        await feeder
        code = await process.wait()
        if code != 0:
            raise ApiError(500, ErrorCode.INTERNAL, f"audio encoder exited with status {code}")
    finally:
        if not feeder.done():
            feeder.cancel()
        if process.returncode is None:
            process.kill()
            await process.wait()


__all__ = [
    "ENGINE_SAMPLE_RATE",
    "MEDIA_TYPES",
    "AudioInfo",
    "OutputFormat",
    "atempo_filter",
    "decode_pcm",
    "decode_wav",
    "encode_pcm",
    "needs_transcode",
    "probe",
    "transcode_stream",
    "wav_header",
]
