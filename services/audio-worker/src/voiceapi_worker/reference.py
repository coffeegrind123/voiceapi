"""Turn raw voice clips into a clone reference: clean, trim, transcribe, align, pick a span.

Output (all under ``output_dir``):

* ``clean_<n>.wav``  every clip enhanced (RE-USE), silence-trimmed, loudness-normalised
* ``reference.wav``  the chosen 5-15 s span at 24 kHz, cut on word boundaries
* ``xvector.wav``    up to 60 s of all cleaned speech at 24 kHz for the speaker embedding

The span text is sliced from the transcript itself (not re-joined aligner tokens) so it keeps
punctuation; ICL conditions on that text, so it must match the reference audio exactly.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from voiceapi_worker.asr import ALIGNER_LANGUAGES, ASR_SAMPLE_RATE, Transcriber, resolve_language
from voiceapi_worker.audio_io import decode, nearest_supported_rate, normalize_loudness, resample, write_wav
from voiceapi_worker.reuse.enhancer import SUPPORTED_RATES, ReuseEnhancer, snr_estimate_db

log = logging.getLogger("voiceapi.worker.reference")

ENGINE_RATE = 24_000
MIN_REFERENCE_S = 3.0
TARGET_REFERENCE_S = 10.0
MAX_REFERENCE_S = 15.0
MAX_XVECTOR_S = 60.0
LEAD_PAD_S = 0.10
TAIL_PAD_S = 0.15
GOOD_PAUSE_S = 0.15
_SENTENCE_END = re.compile(r"[.!?。！？…]")
_PUNCT_AFTER = re.compile(r"^[\s]*[,.!?;:。！？；：、…\"'”’)\]]*")


class ReferenceError(ValueError):
    """Input cannot yield a usable reference (too short, silent, no speech)."""

    def __init__(self, message: str, code: str = "audio_too_short") -> None:
        super().__init__(message)
        self.code = code


@dataclass(slots=True)
class CleanClip:
    audio: np.ndarray  # mono float32 at ``rate``
    rate: int
    snr_before: float
    snr_after: float

    @property
    def duration(self) -> float:
        return len(self.audio) / self.rate


@dataclass(slots=True)
class Span:
    clip: int
    start: float
    end: float
    text: str
    score: float


def _trim(audio: np.ndarray) -> np.ndarray:
    import librosa

    trimmed, _ = librosa.effects.trim(audio, top_db=40, frame_length=1024, hop_length=256)
    return trimmed if trimmed.size else audio


def clean_clip(path: Path, enhancer: ReuseEnhancer | None) -> CleanClip:
    audio, rate = decode(path)
    snr_before = snr_estimate_db(audio, rate)
    if enhancer is not None:
        work_rate = nearest_supported_rate(rate, SUPPORTED_RATES)
        audio = enhancer.enhance(resample(audio, rate, work_rate), work_rate)
        rate = work_rate
    audio = normalize_loudness(_trim(audio), rate)
    return CleanClip(audio=audio, rate=rate, snr_before=snr_before, snr_after=snr_estimate_db(audio, rate))


def _locate_words(transcript: str, words: list[dict]) -> list[tuple[int, int]] | None:
    """Character span of every aligned word inside the transcript, or None if they do not line up."""
    lowered = transcript.lower()
    cursor = 0
    positions: list[tuple[int, int]] = []
    for word in words:
        token = word["word"].strip().lower()
        if not token:
            return None
        found = lowered.find(token, cursor)
        if found < 0:
            return None
        positions.append((found, found + len(token)))
        cursor = found + len(token)
    return positions


def _span_text(transcript: str, positions: list[tuple[int, int]] | None, words: list[dict], *, first: int,
               last: int, language: str | None) -> str:
    i, j = first, last
    if positions is not None:
        start = positions[i][0]
        end = positions[j][1]
        tail = _PUNCT_AFTER.match(transcript[end:])
        end += len(tail.group(0).rstrip()) if tail else 0
        return transcript[start:end].strip()
    joiner = "" if language in {"Chinese", "Cantonese", "Japanese", "Thai"} else " "
    return joiner.join(word["word"] for word in words[i : j + 1]).strip()


def _ends_sentence(transcript: str, positions: list[tuple[int, int]] | None, index: int) -> bool:
    if positions is None:
        return False
    following = transcript[positions[index][1] : positions[index][1] + 3]
    return bool(_SENTENCE_END.search(following))


def best_span(
    clip_index: int, clip: CleanClip, words: list[dict], transcript: str, language: str | None
) -> Span | None:
    """Score every word-bounded window of MIN..MAX seconds and return the best one."""
    if not words:
        return None
    positions = _locate_words(transcript, words)
    best: Span | None = None
    count = len(words)
    for i in range(count):
        gap_before = words[i]["start"] - (words[i - 1]["end"] if i else 0.0)
        starts_sentence = i == 0 or _ends_sentence(transcript, positions, i - 1)
        for j in range(i, count):
            duration = words[j]["end"] - words[i]["start"]
            if duration > MAX_REFERENCE_S:
                break
            if duration < MIN_REFERENCE_S:
                continue
            gap_after = (words[j + 1]["start"] if j + 1 < count else clip.duration) - words[j]["end"]
            ends_sentence = j == count - 1 or _ends_sentence(transcript, positions, j)
            segment = clip.audio[int(words[i]["start"] * clip.rate) : int(words[j]["end"] * clip.rate)]
            snr = snr_estimate_db(segment, clip.rate)
            score = (
                0.40 * max(0.0, 1.0 - abs(duration - TARGET_REFERENCE_S) / TARGET_REFERENCE_S)
                + 0.125 * min(1.0, gap_before / GOOD_PAUSE_S)
                + 0.125 * min(1.0, gap_after / GOOD_PAUSE_S)
                + 0.25 * (0.5 if np.isnan(snr) else min(1.0, max(0.0, snr / 30.0)))
                + 0.05 * starts_sentence
                + 0.05 * ends_sentence
            )
            if best is None or score > best.score:
                best = Span(clip_index, words[i]["start"], words[j]["end"],
                            _span_text(transcript, positions, words, first=i, last=j, language=language), score)
    return best


def _choose_span(
    clips: list[CleanClip],
    usable: list[int],
    transcript: str | None,
    forced: str | None,
    transcriber: Transcriber,
) -> tuple[Span | None, str | None]:
    """Transcribe (unless given) and align each usable clip; return the best-scoring span."""
    if transcript and len(clips) == 1 and clips[0].duration <= MAX_REFERENCE_S:
        return Span(0, 0.0, clips[0].duration, transcript.strip(), 1.0), forced
    span: Span | None = None
    detected_language = forced
    for index in usable:
        clip = clips[index]
        audio_16k = resample(clip.audio, clip.rate, ASR_SAMPLE_RATE)
        if transcript:
            text, clip_language = transcript.strip(), forced
        else:
            result = transcriber.transcribe(audio_16k, language=forced, prompt=None, word_timestamps=False)
            text, clip_language = result["text"], result["language"] or forced
        detected_language = detected_language or clip_language
        if not text:
            continue
        alignable = clip_language is None or clip_language in ALIGNER_LANGUAGES
        words = transcriber.align(audio_16k, text, clip_language) if alignable else []
        candidate = best_span(index, clip, words, text, clip_language)
        if candidate is None and clip.duration <= MAX_REFERENCE_S:
            candidate = Span(index, 0.0, clip.duration, text, 0.1)
        if candidate and (span is None or candidate.score > span.score):
            span = candidate
    return span, detected_language


def prepare_reference(
    input_paths: list[Path],
    output_dir: Path,
    *,
    transcript: str | None,
    language: str | None,
    clean: bool,
    enhancer: ReuseEnhancer | None,
    transcriber: Transcriber,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    clips = [clean_clip(path, enhancer if clean else None) for path in input_paths]
    for index, clip in enumerate(clips):
        write_wav(output_dir / f"clean_{index}.wav", clip.audio, clip.rate)
    usable = [index for index, clip in enumerate(clips) if clip.duration >= MIN_REFERENCE_S]
    total = sum(clip.duration for clip in clips)
    if not usable:
        raise ReferenceError(
            f"no clip has {MIN_REFERENCE_S:.0f} s of speech after trimming (total {total:.1f} s)", "audio_too_short"
        )
    span, detected_language = _choose_span(clips, usable, transcript, resolve_language(language), transcriber)
    if span is None:
        raise ReferenceError("no intelligible speech found in the clips", "audio_unreadable")

    source = clips[span.clip]
    start = max(0, int((span.start - LEAD_PAD_S) * source.rate))
    end = min(len(source.audio), int((span.end + TAIL_PAD_S) * source.rate))
    reference = resample(source.audio[start:end], source.rate, ENGINE_RATE)
    write_wav(output_dir / "reference.wav", reference, ENGINE_RATE)

    gap = np.zeros(int(0.2 * ENGINE_RATE), dtype=np.float32)
    pieces: list[np.ndarray] = []
    budget = int(MAX_XVECTOR_S * ENGINE_RATE)
    for clip in clips:
        piece = resample(clip.audio, clip.rate, ENGINE_RATE)[:budget]
        pieces += [piece, gap]
        budget -= len(piece)
        if budget <= 0:
            break
    write_wav(output_dir / "xvector.wav", np.concatenate(pieces[:-1]), ENGINE_RATE)

    finite = [clip.snr_after for clip in clips if np.isfinite(clip.snr_after)]
    finite_before = [clip.snr_before for clip in clips if np.isfinite(clip.snr_before)]
    return {
        "reference_path": str(output_dir / "reference.wav"),
        "xvector_path": str(output_dir / "xvector.wav"),
        "reference_text": span.text,
        "reference_duration_s": round(len(reference) / ENGINE_RATE, 3),
        "language": detected_language,
        "clips_used": len(usable),
        "cleaned_paths": [str(output_dir / f"clean_{index}.wav") for index in range(len(clips))],
        "quality": {
            "snr_before_db": round(float(np.median(finite_before)), 2) if finite_before else None,
            "snr_after_db": round(float(np.median(finite)), 2) if finite else None,
            "reference_score": round(span.score, 3),
            "total_speech_s": round(total, 2),
            "cleaned": clean,
        },
    }


__all__ = ["ReferenceError", "best_span", "clean_clip", "prepare_reference"]
