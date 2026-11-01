from __future__ import annotations

import pytest

from voiceapi_gateway.audio import atempo_filter, wav_header
from voiceapi_gateway.errors import ApiError
from voiceapi_gateway.ids import engine_voice_handle, is_id, new_id
from voiceapi_gateway.services.speech import normalize_language
from voiceapi_gateway.services.transcription import pick_language
from voiceapi_gateway.text_split import CJK_CHUNK_CHARS, LATIN_CHUNK_CHARS, split_for_speech


def test_short_text_is_one_chunk() -> None:
    assert split_for_speech("  Hello   world.  ") == ["Hello world."]


def test_latin_text_splits_on_sentences_within_budget() -> None:
    text = " ".join(f"Sentence number {i} has a handful of words in it." for i in range(60))
    chunks = split_for_speech(text)
    assert len(chunks) > 1
    assert all(len(chunk) <= LATIN_CHUNK_CHARS for chunk in chunks)
    assert all(chunk.endswith(".") for chunk in chunks)
    assert " ".join(chunks) == text


def test_cjk_text_uses_the_smaller_budget() -> None:
    text = "今天天气很好，我们去公园散步。" * 40
    chunks = split_for_speech(text)
    assert all(len(chunk) <= CJK_CHUNK_CHARS for chunk in chunks)
    assert "".join(chunks) == text


def test_run_on_sentence_is_hard_split_at_clauses_then_spaces() -> None:
    text = ", ".join(["word " * 30] * 10)
    chunks = split_for_speech(text)
    assert all(len(chunk) <= LATIN_CHUNK_CHARS for chunk in chunks)
    assert " ".join(chunks).split() == text.replace(",", ",").split()


@pytest.mark.parametrize(
    ("speed", "expected"),
    [
        (1.0, None),
        (1.5, "atempo=1.500000"),
        (3.0, "atempo=2.000000,atempo=1.500000"),
        (0.3, "atempo=0.500000,atempo=0.600000"),
    ],
)
def test_atempo_chain_stays_within_stage_limits(speed: float, expected: str | None) -> None:
    assert atempo_filter(speed) == expected


def test_atempo_rejects_out_of_range() -> None:
    with pytest.raises(ApiError):
        atempo_filter(5.0)


def test_wav_header_layout() -> None:
    header = wav_header(48_000)
    assert header[:4] == b"RIFF" and header[8:16] == b"WAVEfmt " and len(header) == 44
    assert int.from_bytes(header[40:44], "little") == 48_000


def test_ids_and_engine_handles() -> None:
    voice_id = new_id("voice")
    assert is_id(voice_id, "voice") and not is_id(voice_id, "file")
    handle = engine_voice_handle(voice_id)
    assert handle.startswith("v_") and len(handle) <= 64
    with pytest.raises(ValueError):
        engine_voice_handle("file_01h")


@pytest.mark.parametrize(("value", "expected"), [("en", "english"), ("en-US", "english"), ("German", "german"),
                                                 (None, "auto"), ("zh", "chinese")])
def test_language_normalization(value, expected) -> None:
    assert normalize_language(value) == expected


def test_unknown_language_is_actionable() -> None:
    with pytest.raises(ApiError) as caught:
        normalize_language("klingon")
    assert caught.value.code.value == "unsupported_language" and "english" in caught.value.hint


def test_ovis_language_list_semantics() -> None:
    assert pick_language(None, ["en"]) == "en"
    assert pick_language(None, ["en", "de"]) is None
    assert pick_language("auto", []) is None
    assert pick_language("fr", ["fr"]) == "fr"


def test_settings_from_environment_uses_real_defaults(monkeypatch) -> None:
    from voiceapi_gateway.config import Settings

    for name in ("VOICEAPI_ENGINE_URL", "VOICEAPI_WORKER_URL", "VOICEAPI_DATA_DIR", "VOICEAPI_DB_DIR",
                 "VOICEAPI_API_KEYS", "VOICEAPI_API_KEYS_FILE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("VOICEAPI_ALLOW_ANONYMOUS", "1")
    settings = Settings.from_env()
    assert settings.engine_url == "http://tts-engine:8000"
    assert settings.worker_url == "http://audio-worker:8001"
    assert str(settings.data_dir) == "/data" and settings.db_dir is None


def test_settings_from_environment_reads_overrides(monkeypatch) -> None:
    from voiceapi_gateway.config import Settings

    monkeypatch.setenv("VOICEAPI_API_KEYS", "k" * 30)
    monkeypatch.setenv("VOICEAPI_ENGINE_URL", "http://engine.test:9000")
    monkeypatch.setenv("VOICEAPI_DB_DIR", "/db")
    settings = Settings.from_env()
    assert settings.engine_url == "http://engine.test:9000"
    assert str(settings.db_path) == "/db/voiceapi.sqlite3"


def test_limits_from_environment_are_integers(monkeypatch) -> None:
    from voiceapi_gateway.config import Settings

    monkeypatch.setenv("VOICEAPI_ALLOW_ANONYMOUS", "1")
    for name in ("VOICEAPI_MAX_UPLOAD_BYTES", "VOICEAPI_MAX_SPEECH_CHARS", "VOICEAPI_JOB_WORKERS"):
        monkeypatch.delenv(name, raising=False)
    limits = Settings.from_env().limits
    got = (limits.max_upload_bytes, limits.max_speech_input_chars, limits.job_workers)
    assert got == (200 * 1024 * 1024, 100_000, 2)
