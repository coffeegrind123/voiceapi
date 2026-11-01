"""Decode anything ffmpeg reads; write PCM WAV; resample; loudness normalisation."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import soundfile

TARGET_LUFS = -20.0
PEAK_CEILING = 0.97


class AudioDecodeError(ValueError):
    pass


def probe_rate(path: Path) -> int:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=sample_rate",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, check=False, timeout=60,
    )
    try:
        return int(result.stdout.strip().splitlines()[0])
    except (ValueError, IndexError) as error:
        raise AudioDecodeError(f"no audio stream in {path.name}") from error


def decode(path: Path, *, sample_rate: int | None = None) -> tuple[np.ndarray, int]:
    """Mono float32 at ``sample_rate`` (default: the file's own rate)."""
    rate = sample_rate or probe_rate(path)
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(path), "-vn", "-ac", "1",
         "-ar", str(rate), "-f", "f32le", "pipe:1"],
        capture_output=True, check=False, timeout=900,
    )
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", "replace").strip().splitlines()[-1:] or ["unknown error"]
        raise AudioDecodeError(f"cannot decode {path.name}: {message[0]}")
    audio = np.frombuffer(result.stdout, dtype=np.float32).copy()
    if audio.size == 0:
        raise AudioDecodeError(f"{path.name} decoded to no samples")
    return audio, rate


def resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio
    import librosa

    return librosa.resample(audio, orig_sr=source_rate, target_sr=target_rate, res_type="soxr_hq").astype(np.float32)


def nearest_supported_rate(rate: int, supported: tuple[int, ...]) -> int:
    """The smallest supported rate ≥ the input (never discard bandwidth), else the largest."""
    for candidate in sorted(supported):
        if candidate >= rate:
            return candidate
    return max(supported)


def normalize_loudness(audio: np.ndarray, sample_rate: int, target_lufs: float = TARGET_LUFS) -> np.ndarray:
    import pyloudnorm

    if len(audio) < int(0.4 * sample_rate):
        return audio
    meter = pyloudnorm.Meter(sample_rate)
    loudness = meter.integrated_loudness(audio.astype(np.float64))
    if not np.isfinite(loudness):
        return audio
    gained = audio * (10 ** ((target_lufs - loudness) / 20))
    peak = float(np.max(np.abs(gained))) if gained.size else 0.0
    if peak > PEAK_CEILING:
        gained = gained * (PEAK_CEILING / peak)
    return gained.astype(np.float32)


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(str(path), np.clip(audio, -1.0, 1.0), sample_rate, subtype="PCM_16")


__all__ = [
    "AudioDecodeError",
    "decode",
    "nearest_supported_rate",
    "normalize_loudness",
    "resample",
    "write_wav",
]
