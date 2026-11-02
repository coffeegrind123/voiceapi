from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from voiceapi_worker import reference as reference_module
from voiceapi_worker.app import Models, WorkerSettings, create_app
from voiceapi_worker.asr import resolve_language, split_at_pauses
from voiceapi_worker.audio_io import nearest_supported_rate, normalize_loudness, write_wav
from voiceapi_worker.reference import CleanClip, _locate_words, best_span, prepare_reference
from voiceapi_worker.reuse import scan
from voiceapi_worker.reuse.enhancer import SUPPORTED_RATES, StftSizes, snr_estimate_db

_ffmpeg_dir = os.environ.get("VOICEAPI_TEST_FFMPEG_DIR")
if _ffmpeg_dir:
    os.environ["PATH"] = f"{_ffmpeg_dir}{os.pathsep}{os.environ['PATH']}"
HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
needs_ffmpeg = pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg/ffprobe not on PATH")


def _voiced(seconds: float, rate: int, *, pause_every: float = 1.0) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    envelope = ((t % pause_every) < pause_every * 0.8).astype(np.float32)
    tone = sum(np.sin(2 * np.pi * f * t) / (k + 1) for k, f in enumerate((150, 300, 450, 600)))
    return (0.2 * tone * envelope).astype(np.float32)


# ---- selective scan --------------------------------------------------------------------


def test_reference_scan_slicing_matches_unsliced(monkeypatch) -> None:
    generator = torch.Generator().manual_seed(0)
    b, d, n, length = 5, 6, 4, 9
    u, delta, z = (torch.randn(b, d, length, generator=generator) for _ in range(3))
    A = -torch.rand(d, n, generator=generator)
    B, C = (torch.randn(b, n, length, generator=generator) for _ in range(2))
    D, bias = torch.randn(d, generator=generator), torch.randn(d, generator=generator)
    whole = scan.selective_scan(u, delta, A, B, C, D, z, bias)
    monkeypatch.setattr(scan, "_REFERENCE_SLICE_ELEMENTS", d * length * n * 2)
    sliced = scan.selective_scan(u, delta, A, B, C, D, z, bias)
    torch.testing.assert_close(whole, sliced)


def test_reference_scan_matches_explicit_recurrence() -> None:
    generator = torch.Generator().manual_seed(1)
    u, delta, z = (torch.randn(1, 1, 3, generator=generator) for _ in range(3))
    A = torch.tensor([[-0.5]])
    B, C = (torch.randn(1, 1, 3, generator=generator) for _ in range(2))
    D, bias = torch.tensor([0.3]), torch.tensor([0.1])
    out = scan.selective_scan(u, delta, A, B, C, D, z, bias)
    dt = torch.nn.functional.softplus(delta + 0.1)[0, 0]
    x, expected = 0.0, []
    for t in range(3):
        x = torch.exp(dt[t] * -0.5) * x + dt[t] * B[0, 0, t] * u[0, 0, t]
        expected.append((x * C[0, 0, t] + 0.3 * u[0, 0, t]) * torch.nn.functional.silu(z[0, 0, t]))
    torch.testing.assert_close(out[0, 0], torch.stack(expected))


# ---- RE-USE model against the published checkpoint ------------------------------------


@pytest.mark.skipif(os.environ.get("VOICEAPI_SKIP_MODEL_DOWNLOAD") == "1", reason="model download disabled")
def test_checkpoint_loads_strictly_and_denoises() -> None:
    from voiceapi_worker.reuse.enhancer import ReuseEnhancer

    enhancer = ReuseEnhancer.load(torch.device("cpu"))
    assert sum(p.numel() for p in enhancer.model.parameters()) == 9_607_747
    rate = 8_000
    clean = _voiced(1.0, rate, pause_every=0.5)
    noisy = clean + 0.05 * np.random.default_rng(0).standard_normal(len(clean)).astype(np.float32)
    out = enhancer.enhance(noisy, rate)
    assert out.shape == noisy.shape and np.isfinite(out).all()
    assert snr_estimate_db(out, rate) > snr_estimate_db(noisy, rate) + 10


def test_stft_sizes_scale_from_the_8k_base() -> None:
    cfg = {"stft_cfg": {"sampling_rate": 8000, "n_fft": 320, "hop_size": 40, "win_size": 320}}
    assert StftSizes.for_rate(cfg, 48_000) == StftSizes(1920, 240, 1920)
    assert StftSizes.for_rate(cfg, 22_050) == StftSizes(882, 110, 882)


def test_enhancement_rate_never_drops_bandwidth() -> None:
    assert nearest_supported_rate(44_100, SUPPORTED_RATES) == 44_100
    assert nearest_supported_rate(11_025, SUPPORTED_RATES) == 16_000
    assert nearest_supported_rate(96_000, SUPPORTED_RATES) == 48_000


# ---- ASR helpers -------------------------------------------------------------------------


def test_language_resolution() -> None:
    assert resolve_language("en") == "English"
    assert resolve_language("en-GB") == "English"
    assert resolve_language("german") == "German"
    assert resolve_language("auto") is None
    with pytest.raises(ValueError):
        resolve_language("klingon")


def test_long_audio_is_split_at_pauses_within_the_limit() -> None:
    rate = 16_000
    audio = _voiced(70.0, rate, pause_every=2.0)
    segments = split_at_pauses(audio, rate, max_seconds=28.0)
    assert len(segments) >= 3
    assert all((end - start) / rate <= 28.0 for start, end in segments)
    assert segments[0][0] < rate and segments[-1][1] > len(audio) - 2 * rate


# ---- reference selection ------------------------------------------------------------------


def _words(text: str, start: float = 0.5, step: float = 0.45, gap_after: dict | None = None) -> list[dict]:
    words, cursor = [], start
    for index, token in enumerate(text.replace(",", "").replace(".", "").split()):
        words.append({"word": token, "start": round(cursor, 3), "end": round(cursor + 0.35, 3)})
        cursor += step + (gap_after or {}).get(index, 0.0)
    return words


def test_span_text_keeps_punctuation_from_the_transcript() -> None:
    transcript = "Hello, world. This is fine."
    words = _words(transcript, step=0.8)
    positions = _locate_words(transcript, words)
    assert positions is not None
    clip = CleanClip(audio=_voiced(5, 16_000), rate=16_000, snr_before=10, snr_after=30)
    span = best_span(0, clip, words, transcript, "English")
    assert span is not None and span.text in {"Hello, world. This is fine.", "world. This is fine."}


def test_best_span_prefers_target_duration_and_pauses() -> None:
    sentence = "one two three four five six seven eight nine ten eleven twelve."
    transcript = " ".join([sentence] * 6)
    gaps = {index: 0.6 for index in range(11, 72, 12)}  # a pause after every sentence
    words = _words(transcript, gap_after=gaps)
    clip = CleanClip(audio=_voiced(words[-1]["end"] + 1, 16_000), rate=16_000, snr_before=10, snr_after=30)
    span = best_span(0, clip, words, transcript, "English")
    assert span is not None
    assert 5.0 <= span.end - span.start <= 15.0
    assert span.text.endswith(".")


class _FakeTranscriber:
    def __init__(self, text: str) -> None:
        self.text = text
        self.align_calls = 0

    def transcribe(self, audio, *, language, prompt, word_timestamps):
        return {"text": self.text, "language": "English", "duration": len(audio) / 16_000}

    def align(self, audio, text, language):
        self.align_calls += 1
        return _words(text, step=0.5)


@needs_ffmpeg
def test_prepare_reference_end_to_end_without_gpu(tmp_path: Path) -> None:
    rate = 22_050
    clip_path = tmp_path / "in" / "clip.wav"
    write_wav(clip_path, _voiced(20, rate, pause_every=2.0), rate)
    transcript = " ".join(f"word{i}." if i % 6 == 5 else f"word{i}" for i in range(36))
    fake = _FakeTranscriber(transcript)
    result = prepare_reference([clip_path], tmp_path / "out", transcript=None, language=None, clean=False,
                               enhancer=None, transcriber=fake)
    assert Path(result["reference_path"]).is_file() and Path(result["xvector_path"]).is_file()
    assert 3.0 <= result["reference_duration_s"] <= 15.5
    assert result["reference_text"].startswith("word") and fake.align_calls == 1
    assert result["quality"]["cleaned"] is False


@needs_ffmpeg
def test_short_clip_with_transcript_skips_asr(tmp_path: Path) -> None:
    clip_path = tmp_path / "clip.wav"
    write_wav(clip_path, _voiced(8, 24_000), 24_000)
    fake = _FakeTranscriber("unused")
    result = prepare_reference([clip_path], tmp_path / "out", transcript="Exact words.", language="en",
                               clean=False, enhancer=None, transcriber=fake)
    assert result["reference_text"] == "Exact words." and fake.align_calls == 0


@needs_ffmpeg
def test_too_short_input_is_rejected_with_a_code(tmp_path: Path) -> None:
    clip_path = tmp_path / "clip.wav"
    write_wav(clip_path, _voiced(1.5, 16_000), 16_000)
    with pytest.raises(reference_module.ReferenceError) as caught:
        prepare_reference([clip_path], tmp_path / "out", transcript=None, language=None, clean=False,
                          enhancer=None, transcriber=_FakeTranscriber("x"))
    assert caught.value.code == "audio_too_short"


def test_loudness_normalisation_hits_target_without_clipping() -> None:
    import pyloudnorm

    audio = _voiced(5, 16_000) * 0.05
    normalised = normalize_loudness(audio, 16_000)
    loudness = pyloudnorm.Meter(16_000).integrated_loudness(normalised.astype(np.float64))
    assert abs(loudness - (-20.0)) < 1.0 and np.abs(normalised).max() <= 0.97


# ---- HTTP surface --------------------------------------------------------------------------


class _StubEnhancer:
    scan_backend = scan.ScanBackend.REFERENCE

    def enhance(self, audio, rate):
        return audio * 0.5


class _StubModels(Models):
    def load(self) -> None:
        self.enhancer = _StubEnhancer()

    def transcriber(self):
        return _FakeTranscriber("hello there")


@pytest.fixture
def client(tmp_path: Path):
    settings = WorkerSettings(data_dir=tmp_path.resolve(), device=torch.device("cpu"), cache_dir=None,
                              preload_asr=False, asr_idle_unload_s=0)
    app = create_app(settings, _StubModels(settings))
    with TestClient(app) as test_client:
        for _ in range(100):
            if test_client.get("/ready").status_code == 200:
                break
        yield test_client, tmp_path


@needs_ffmpeg
def test_enhance_endpoint_writes_output(client) -> None:
    test_client, root = client
    source = root / "in.wav"
    write_wav(source, _voiced(2, 16_000), 16_000)
    response = test_client.post("/v1/enhance", json={"input_path": str(source), "output_path": str(root / "o.wav")})
    assert response.status_code == 200, response.text
    assert response.json()["sample_rate"] == 16_000 and (root / "o.wav").is_file()


def test_paths_outside_the_data_dir_are_refused(client) -> None:
    test_client, _root = client
    response = test_client.post("/v1/enhance", json={"input_path": "/etc/passwd", "output_path": "/tmp/x.wav"})
    assert response.status_code == 422 and response.json()["code"] == "validation_failed"


def test_missing_input_is_reported(client) -> None:
    test_client, root = client
    response = test_client.post("/v1/transcribe", json={"input_path": str(root / "nope.wav")})
    assert response.status_code == 422 and response.json()["code"] == "file_not_found"


@needs_ffmpeg
def test_snr_after_is_comparable_to_before(tmp_path: Path) -> None:
    # Trimming leading/trailing silence removes the quiet frames the blind estimate uses as its
    # noise floor; measuring "after" on trimmed audio reported a drop for unchanged audio.
    rate = 16_000
    rng = np.random.default_rng(3)
    speech = _voiced(4.0, rate, pause_every=1.0)
    padded = np.concatenate([np.zeros(rate), speech, np.zeros(rate)])
    noisy = (padded + 0.0003 * rng.standard_normal(len(padded))).astype(np.float32)  # edges below -40 dB: trimmed
    path = tmp_path / "clip.wav"
    write_wav(path, noisy, rate)
    clip = reference_module.clean_clip(path, enhancer=None)
    assert abs(clip.snr_after - clip.snr_before) < 1.5, (clip.snr_before, clip.snr_after)
