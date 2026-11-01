"""Qwen3-ASR transcription and Qwen3-ForcedAligner word timestamps (transformers >= 5.13).

Long audio is cut at pauses into segments of at most ``MAX_SEGMENT_SECONDS``
so each generate call stays inside the model's comfortable context and the
result carries real segment timestamps for SRT/VTT.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
import torch

ASR_REPO = "Qwen/Qwen3-ASR-1.7B-hf"
ASR_REVISION = "bcd2b5b7f32b480ab5790554cfa8347f246a14f3"
ALIGNER_REPO = "Qwen/Qwen3-ForcedAligner-0.6B-hf"
ALIGNER_REVISION = "c07281df297b9905d24a508279258cccf987a064"
ASR_SAMPLE_RATE = 16_000
MAX_SEGMENT_SECONDS = 28.0
MAX_NEW_TOKENS = 640
_ASR_TEXT_MARKER = "<asr_text>"

# Names the ASR prompt was trained on; codes and English names both map here.
LANGUAGE_NAMES = {
    "ar": "Arabic", "yue": "Cantonese", "zh": "Chinese", "cs": "Czech", "da": "Danish", "nl": "Dutch",
    "en": "English", "fil": "Filipino", "fi": "Finnish", "fr": "French", "de": "German", "el": "Greek",
    "hi": "Hindi", "hu": "Hungarian", "id": "Indonesian", "it": "Italian", "ja": "Japanese", "ko": "Korean",
    "mk": "Macedonian", "ms": "Malay", "fa": "Persian", "pl": "Polish", "pt": "Portuguese", "ro": "Romanian",
    "ru": "Russian", "es": "Spanish", "sv": "Swedish", "th": "Thai", "tr": "Turkish", "vi": "Vietnamese",
}
ALIGNER_LANGUAGES = frozenset(
    {"Chinese", "English", "Cantonese", "French", "German", "Italian", "Japanese", "Korean", "Portuguese",
     "Russian", "Spanish"}
)
_NO_SPACE_LANGUAGES = frozenset({"Chinese", "Cantonese", "Japanese", "Thai"})


class UnsupportedLanguageError(ValueError):
    pass


def resolve_language(value: str | None) -> str | None:
    if value is None or not value.strip() or value.strip().lower() == "auto":
        return None
    key = value.strip()
    lowered = key.lower().split("-")[0]
    if lowered in LANGUAGE_NAMES:
        return LANGUAGE_NAMES[lowered]
    for name in LANGUAGE_NAMES.values():
        if name.lower() == key.lower():
            return name
    raise UnsupportedLanguageError(f"unsupported ASR language {value!r}")


@dataclass(slots=True)
class Segment:
    start: float
    end: float
    text: str
    language: str | None
    words: list[dict] = field(default_factory=list)


def split_at_pauses(
    audio: np.ndarray, sample_rate: int, max_seconds: float = MAX_SEGMENT_SECONDS
) -> list[tuple[int, int]]:
    """Greedy segments ≤ max_seconds, cut inside silences found by an energy detector."""
    import librosa

    total = len(audio)
    limit = int(max_seconds * sample_rate)
    if total <= limit:
        return [(0, total)]
    voiced = librosa.effects.split(audio, top_db=35, frame_length=1024, hop_length=256)
    if len(voiced) == 0:
        return [(start, min(start + limit, total)) for start in range(0, total, limit)]
    segments: list[tuple[int, int]] = []
    seg_start = int(voiced[0][0])
    seg_end = int(voiced[0][1])
    for raw_start, raw_end in voiced[1:]:
        run_start, run_end = int(raw_start), int(raw_end)
        if run_end - seg_start <= limit:
            seg_end = run_end
            continue
        segments.append((seg_start, seg_end))
        seg_start, seg_end = run_start, run_end
    segments.append((seg_start, seg_end))
    # A single voiced run longer than the limit is cut hard.
    bounded: list[tuple[int, int]] = []
    for seg_begin, seg_finish in segments:
        cursor = seg_begin
        while seg_finish - cursor > limit:
            bounded.append((cursor, cursor + limit))
            cursor += limit
        bounded.append((cursor, seg_finish))
    return bounded


class Transcriber:
    def __init__(self, device: torch.device, *, cache_dir: str | None = None) -> None:
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        self.device = device
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(ASR_REPO, revision=ASR_REVISION, cache_dir=cache_dir)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            ASR_REPO, revision=ASR_REVISION, cache_dir=cache_dir, dtype=dtype
        ).to(device).eval()
        self._aligner = None
        self._aligner_processor = None
        self._cache_dir = cache_dir
        self._lock = threading.Lock()

    # ---- transcription ---------------------------------------------------------------

    def _inputs(self, audio: np.ndarray, language: str | None, prompt: str | None):
        inputs = self.processor.apply_transcription_request(
            audio=audio, language=language, prompt=prompt, sampling_rate=ASR_SAMPLE_RATE
        )
        return inputs.to(self.device, self.model.dtype)

    @staticmethod
    def _parse(raw: str, forced_language: str | None) -> tuple[str | None, str]:
        if forced_language is not None:
            return forced_language, raw.replace(_ASR_TEXT_MARKER, "").strip()
        if _ASR_TEXT_MARKER not in raw:
            return None, raw.strip()
        prefix, text = raw.split(_ASR_TEXT_MARKER, 1)
        match = re.search(r"language\s+(\w+)", prefix)
        language = match.group(1) if match and match.group(1).lower() != "none" else None
        return language, text.strip()

    @torch.inference_mode()
    def _generate(self, audio: np.ndarray, language: str | None, prompt: str | None) -> tuple[str | None, str]:
        inputs = self._inputs(audio, language, prompt)
        output = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        generated = output[:, inputs["input_ids"].shape[1] :]
        raw = self.processor.batch_decode(generated, skip_special_tokens=False)[0]
        raw = raw.replace(self.processor.tokenizer.eos_token or "", "")
        for token in ("<|im_end|>", "<|endoftext|>"):
            raw = raw.replace(token, "")
        return self._parse(raw, language)

    def transcribe(
        self,
        audio: np.ndarray,
        *,
        language: str | None,
        prompt: str | None,
        word_timestamps: bool,
    ) -> dict:
        forced = resolve_language(language)
        segments: list[Segment] = []
        with self._lock:
            for start, end in split_at_pauses(audio, ASR_SAMPLE_RATE):
                piece = audio[start:end]
                detected, text = self._generate(piece, forced, prompt)
                segment = Segment(start / ASR_SAMPLE_RATE, end / ASR_SAMPLE_RATE, text, detected or forced)
                if word_timestamps and text:
                    segment.words = self._align(piece, text, segment.language, offset=segment.start)
                segments.append(segment)
        languages = [segment.language for segment in segments if segment.language]
        language_out = max(set(languages), key=languages.count) if languages else None
        joiner = "" if language_out in _NO_SPACE_LANGUAGES else " "
        result = {
            "text": joiner.join(segment.text for segment in segments if segment.text).strip(),
            "language": language_out,
            "duration": len(audio) / ASR_SAMPLE_RATE,
            "segments": [
                {"id": index, "start": round(s.start, 3), "end": round(s.end, 3), "text": s.text}
                for index, s in enumerate(segments)
            ],
        }
        if word_timestamps:
            result["words"] = [word for segment in segments for word in segment.words]
        return result

    def stream(self, audio: np.ndarray, *, language: str | None, prompt: str | None) -> Iterator[dict]:
        """Yield ``{"delta"}`` events as tokens decode, then ``{"done", "text", "language"}``."""
        from transformers import TextIteratorStreamer

        forced = resolve_language(language)
        texts: list[str] = []
        detected_language = forced
        with self._lock:
            for start, end in split_at_pauses(audio, ASR_SAMPLE_RATE):
                inputs = self._inputs(audio[start:end], forced, prompt)
                streamer = TextIteratorStreamer(self.processor.tokenizer, skip_prompt=True, skip_special_tokens=False)
                error: list[BaseException] = []

                def run(inputs=inputs, streamer=streamer, error=error) -> None:
                    try:
                        with torch.inference_mode():
                            self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                                streamer=streamer)
                    except BaseException as exc:  # noqa: BLE001 - re-raised on the consumer thread
                        error.append(exc)
                        streamer.end()

                thread = threading.Thread(target=run, daemon=True)
                thread.start()
                raw = ""
                emitted = 0
                for piece in streamer:
                    raw += piece
                    if forced is None and _ASR_TEXT_MARKER not in raw:
                        continue
                    language_found, text = self._parse(_strip_end_tokens(raw), forced)
                    detected_language = detected_language or language_found
                    if len(text) > emitted:
                        prefix = " " if texts and emitted == 0 and detected_language not in _NO_SPACE_LANGUAGES else ""
                        yield {"delta": prefix + text[emitted:]}
                        emitted = len(text)
                thread.join()
                if error:
                    raise error[0]
                texts.append(self._parse(_strip_end_tokens(raw), forced)[1])
        joiner = "" if detected_language in _NO_SPACE_LANGUAGES else " "
        yield {"done": True, "text": joiner.join(t for t in texts if t).strip(), "language": detected_language}

    # ---- alignment -------------------------------------------------------------------

    def _load_aligner(self) -> None:
        if self._aligner is not None:
            return
        from transformers import AutoModelForTokenClassification, AutoProcessor

        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self._aligner_processor = AutoProcessor.from_pretrained(
            ALIGNER_REPO, revision=ALIGNER_REVISION, cache_dir=self._cache_dir
        )
        self._aligner = AutoModelForTokenClassification.from_pretrained(
            ALIGNER_REPO, revision=ALIGNER_REVISION, cache_dir=self._cache_dir, dtype=dtype
        ).to(self.device).eval()

    @torch.inference_mode()
    def _align(self, audio: np.ndarray, transcript: str, language: str | None, *, offset: float = 0.0) -> list[dict]:
        if language is not None and language not in ALIGNER_LANGUAGES:
            return []
        self._load_aligner()
        inputs, word_lists = self._aligner_processor.prepare_forced_aligner_inputs(
            audio=audio, transcript=transcript, language=language or "English", sampling_rate=ASR_SAMPLE_RATE
        )
        inputs = inputs.to(self.device, self._aligner.dtype)
        logits = self._aligner(**inputs).logits
        stamps = self._aligner_processor.decode_forced_alignment(
            logits=logits, input_ids=inputs["input_ids"], word_lists=word_lists,
            timestamp_token_id=self._aligner.config.timestamp_token_id,
        )[0]
        return [
            {"word": item["text"], "start": round(offset + float(item["start_time"]), 3),
             "end": round(offset + float(item["end_time"]), 3)}
            for item in stamps
        ]

    def align(self, audio: np.ndarray, transcript: str, language: str | None) -> list[dict]:
        resolved = resolve_language(language) if language else None
        with self._lock:
            return self._align(audio, transcript, resolved)


def _strip_end_tokens(raw: str) -> str:
    for token in ("<|im_end|>", "<|endoftext|>"):
        raw = raw.replace(token, "")
    return raw


__all__ = ["ALIGNER_LANGUAGES", "ASR_SAMPLE_RATE", "Transcriber", "UnsupportedLanguageError", "resolve_language",
           "split_at_pauses"]
