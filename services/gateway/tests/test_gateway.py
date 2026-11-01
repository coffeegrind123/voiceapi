from __future__ import annotations

import json

import httpx
import pytest

from voiceapi_gateway.ids import engine_voice_handle

from .conftest import API_KEY, Harness
from .fakes import wav_bytes


async def _upload(h: Harness, seconds: float = 10.0, name: str = "clip.wav") -> str:
    response = await h.client.post("/v1/files", files={"file": (name, wav_bytes(seconds), "audio/wav")})
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["duration_s"] == pytest.approx(seconds, abs=0.05)
    return body["id"]


async def _ready_voice(h: Harness, name: str = "Narrator", **extra) -> dict:
    file_id = await _upload(h)
    response = await h.client.post(
        "/v1/voices?wait=30", json={"name": name, "file_ids": [file_id], **extra}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["job"]["status"] == "succeeded", body
    return body["voice"]


# ---- auth and discovery ----------------------------------------------------------------


async def test_auth_is_required_and_problem_documents_are_returned(harness: Harness) -> None:
    response = await harness.client.get("/v1/voices", headers={"Authorization": "Bearer wrong-key-xxxxxxxxxxxxxxxx"})
    assert response.status_code == 401
    assert response.headers["content-type"].startswith("application/problem+json")
    problem = response.json()
    assert problem["code"] == "unauthorized" and problem["request_id"].startswith("req_")
    assert problem["error"]["message"]


async def test_public_endpoints_need_no_key(harness: Harness) -> None:
    async with httpx.AsyncClient(transport=harness.client._transport, base_url="http://gateway") as anonymous:
        assert (await anonymous.get("/health")).status_code == 200
        ready = await anonymous.get("/ready")
        assert ready.status_code == 200 and ready.json()["ready"] is True
        guide = await anonymous.get("/llms.txt")
        assert "POST /v1/audio/speech" in guide.text and "{BASE_URL}" not in guide.text
        assert (await anonymous.get("/openapi.json")).status_code == 200


async def test_capabilities_and_models(harness: Harness) -> None:
    caps = (await harness.client.get("/v1/capabilities")).json()
    assert caps["speech"]["instructions"] is False
    assert "english" in caps["speech"]["languages"]
    models = {item["id"] for item in (await harness.client.get("/v1/models")).json()["data"]}
    assert {"tts-1", "whisper-1", "qwen3-tts-1.7b-base"} <= models


# ---- voices ----------------------------------------------------------------------------


async def test_voice_build_cleans_transcribes_and_registers(harness: Harness) -> None:
    voice = await _ready_voice(harness)
    assert voice["status"] == "ready" and voice["supports_icl"] is True
    assert voice["reference"]["transcript"] == "this is the reference transcript"
    assert voice["quality"]["snr_after_db"] == 30.0
    assert engine_voice_handle(voice["id"]) in harness.engine.voices
    assert harness.worker.reference_calls[0]["clean"] is True
    assert harness.engine.last_build["xvector_audio_base64"]
    reference = await harness.client.get(f"/v1/voices/{voice['id']}/reference")
    assert reference.status_code == 200 and reference.content.startswith(b"RIFF")


async def test_voice_names_are_unique(harness: Harness) -> None:
    await _ready_voice(harness, name="Taken")
    file_id = await _upload(harness)
    clash = await harness.client.post("/v1/voices", json={"name": "taken", "file_ids": [file_id]})
    assert clash.status_code == 409 and clash.json()["code"] == "voice_name_taken"


async def test_voice_rejects_too_little_audio(harness: Harness) -> None:
    file_id = await _upload(harness, seconds=1.0)
    response = await harness.client.post("/v1/voices", json={"name": "Short", "file_ids": [file_id]})
    assert response.status_code == 422 and response.json()["code"] == "audio_too_short"


async def test_multipart_voice_creation(harness: Harness) -> None:
    response = await harness.client.post(
        "/v1/voices?wait=30",
        data={"name": "Inline", "transcript": "exact words here"},
        files=[("files", ("a.wav", wav_bytes(6), "audio/wav"))],
    )
    assert response.status_code == 200, response.text
    assert response.json()["voice"]["reference"]["transcript"] == "exact words here"


async def test_idempotent_voice_creation_replays(harness: Harness) -> None:
    file_id = await _upload(harness)
    payload = {"name": "Once", "file_ids": [file_id]}
    headers = {"Idempotency-Key": "abc-123"}
    first = await harness.client.post("/v1/voices", json=payload, headers=headers)
    second = await harness.client.post("/v1/voices", json=payload, headers=headers)
    assert first.json()["voice"]["id"] == second.json()["voice"]["id"]
    assert second.headers.get("idempotent-replayed") == "true"
    conflict = await harness.client.post("/v1/voices", json={**payload, "name": "Twice"}, headers=headers)
    assert conflict.status_code == 409 and conflict.json()["code"] == "idempotency_conflict"


async def test_voice_lookup_by_name_patch_and_delete(harness: Harness) -> None:
    voice = await _ready_voice(harness, name="Guide")
    by_name = await harness.client.get("/v1/voices/Guide")
    assert by_name.json()["id"] == voice["id"]
    patched = await harness.client.patch(f"/v1/voices/{voice['id']}", json={"name": "Tour Guide", "tags": ["en"]})
    assert patched.json()["name"] == "Tour Guide" and patched.json()["tags"] == ["en"]
    assert (await harness.client.delete("/v1/voices/Tour Guide")).status_code == 204
    assert engine_voice_handle(voice["id"]) not in harness.engine.voices
    missing = await harness.client.get(f"/v1/voices/{voice['id']}")
    assert missing.status_code == 404 and missing.json()["code"] == "voice_not_found"


# ---- speech ----------------------------------------------------------------------------


@pytest.mark.parametrize(("fmt", "magic"), [("wav", b"RIFF"), ("mp3", None), ("flac", b"fLaC"), ("opus", b"OggS")])
async def test_speech_formats(harness: Harness, fmt: str, magic: bytes | None) -> None:
    await _ready_voice(harness)
    response = await harness.client.post(
        "/v1/audio/speech", json={"input": "Hello there, agent.", "voice": "Narrator", "response_format": fmt}
    )
    assert response.status_code == 200, response.text
    assert len(response.content) > 100
    if magic:
        assert response.content.startswith(magic)
    else:
        assert response.content[:3] == b"ID3" or response.content[0] == 0xFF


async def test_speech_pcm_is_sample_aligned_and_passes_clone_controls(harness: Harness) -> None:
    voice = await _ready_voice(harness)
    response = await harness.client.post(
        "/v1/audio/speech",
        json={"input": "Short line.", "voice": voice["id"], "response_format": "pcm", "language": "en",
              "voice_mode": "xvector", "seed": 7},
    )
    assert response.status_code == 200 and len(response.content) % 2 == 0
    sent = harness.engine.speech_requests[-1]
    assert sent["voice"] == engine_voice_handle(voice["id"])
    assert sent["language"] == "english" and sent["voice_mode"] == "xvector" and sent["seed"] == 7


async def test_long_input_is_chunked_in_order(harness: Harness) -> None:
    await _ready_voice(harness)
    sentence = "This sentence is here to make the input long enough to need several engine requests. "
    response = await harness.client.post(
        "/v1/audio/speech", json={"input": sentence * 30, "voice": "Narrator", "response_format": "pcm"}
    )
    assert response.status_code == 200
    chunks = [request["input"] for request in harness.engine.speech_requests]
    assert len(chunks) > 1 and all(len(chunk) <= 600 for chunk in chunks)
    assert " ".join(chunks).split() == (sentence * 30).split()


async def test_evicted_voice_is_reregistered_transparently(harness: Harness) -> None:
    voice = await _ready_voice(harness)
    harness.engine.voices.clear()  # engine restart / LRU eviction
    response = await harness.client.post(
        "/v1/audio/speech", json={"input": "Still works.", "voice": voice["id"], "response_format": "wav"}
    )
    assert response.status_code == 200
    assert engine_voice_handle(voice["id"]) in harness.engine.voices


async def test_speech_errors_are_actionable(harness: Harness) -> None:
    missing = await harness.client.post("/v1/audio/speech", json={"input": "x", "voice": "nobody"})
    assert missing.status_code == 404 and missing.json()["code"] == "voice_not_found"
    assert "GET /v1/voices" in missing.json()["hint"]
    await _ready_voice(harness)
    styled = await harness.client.post(
        "/v1/audio/speech", json={"input": "x", "voice": "Narrator", "instructions": "sound happy"}
    )
    assert styled.status_code == 422 and styled.json()["code"] == "instructions_unsupported"
    bad_model = await harness.client.post("/v1/audio/speech", json={"input": "x", "voice": "Narrator", "model": "x"})
    assert bad_model.json()["code"] == "unsupported_model"
    invalid = await harness.client.post("/v1/audio/speech", json={"input": "x", "voice": "Narrator", "speed": 9})
    assert invalid.status_code == 422 and invalid.json()["code"] == "validation_failed"


async def test_openai_preset_maps_to_default_voice(harness: Harness) -> None:
    voice = await _ready_voice(harness)
    harness.state.speech.default_voice = voice["id"]
    response = await harness.client.post("/v1/audio/speech", json={"input": "Hi.", "voice": "alloy",
                                                                   "response_format": "wav"})
    assert response.status_code == 200


async def test_speech_save_and_speed(harness: Harness) -> None:
    await _ready_voice(harness)
    normal = await harness.client.post(
        "/v1/audio/speech", json={"input": "A sentence to time.", "voice": "Narrator", "save": True,
                                  "response_format": "wav"},
    )
    fast = await harness.client.post(
        "/v1/audio/speech", json={"input": "A sentence to time.", "voice": "Narrator", "save": True,
                                  "response_format": "wav", "speed": 2.0},
    )
    assert normal.status_code == fast.status_code == 201
    assert fast.json()["duration_s"] == pytest.approx(normal.json()["duration_s"] / 2, rel=0.1)
    content = await harness.client.get(normal.json()["url"].replace("http://gateway", ""))
    assert content.status_code == 200 and content.content.startswith(b"RIFF")


async def test_speech_sse(harness: Harness) -> None:
    await _ready_voice(harness)
    response = await harness.client.post(
        "/v1/audio/speech", json={"input": "Stream me.", "voice": "Narrator", "stream_format": "sse",
                                  "response_format": "pcm"},
    )
    events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
    assert events[0]["type"] == "speech.audio.delta" and events[-1]["type"] == "speech.audio.done"


# ---- transcription (Ovis shapes) ---------------------------------------------------------


async def test_transcription_json_text_and_verbose(harness: Harness) -> None:
    files = {"file": ("audio.webm", wav_bytes(2), "audio/webm")}
    plain = await harness.client.post("/v1/audio/transcriptions", files=files, data={"model": "gpt-transcribe"})
    assert plain.status_code == 200 and plain.headers["content-type"].startswith("application/json")
    assert plain.json() == {"text": "hello from the fake worker"}
    text = await harness.client.post("/v1/audio/transcriptions", files=files, data={"response_format": "text"})
    assert text.text == "hello from the fake worker"
    verbose = await harness.client.post(
        "/v1/audio/transcriptions", files=files,
        data={"response_format": "verbose_json", "timestamp_granularities[]": "word"},
    )
    assert len(verbose.json()["words"]) == 5
    srt = await harness.client.post("/v1/audio/transcriptions", files=files, data={"response_format": "srt"})
    assert "00:00:00,000 -->" in srt.text


async def test_transcription_stream_matches_ovis_events(harness: Harness) -> None:
    response = await harness.client.post(
        "/v1/audio/transcriptions",
        files={"file": ("audio.webm", wav_bytes(2), "audio/webm")},
        data={"model": "gpt-transcribe", "stream": "true", "languages[]": ["en", "de"], "prompt": "names: Ada"},
    )
    assert response.headers["content-type"].startswith("text/event-stream")
    events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
    assert {event["type"] for event in events[:-1]} == {"transcript.text.delta"}
    assert events[-1] == {"type": "transcript.text.done", "text": "hello from the fake worker", "language": "English"}


async def test_transcription_field_limit(harness: Harness) -> None:
    data = {"languages[]": [f"l{i}" for i in range(40)]}
    response = await harness.client.post(
        "/v1/audio/transcriptions", files={"file": ("a.wav", wav_bytes(1), "audio/wav")}, data=data
    )
    assert response.status_code == 413 and response.json()["code"] == "too_many_fields"


# ---- enhance, files, jobs ---------------------------------------------------------------


async def test_enhance_short_file_synchronously(harness: Harness) -> None:
    file_id = await _upload(harness, seconds=3)
    response = await harness.client.post("/v1/audio/enhance", json={"file_id": file_id})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["metrics"]["snr_after_db"] == 25.0 and body["file"]["purpose"] == "enhanced"


async def test_enhance_long_file_runs_as_job(harness: Harness) -> None:
    file_id = await _upload(harness, seconds=40)
    response = await harness.client.post("/v1/audio/enhance?wait=30", json={"file_id": file_id})
    assert response.status_code == 200 and response.json()["status"] == "succeeded"
    assert response.json()["result"]["file_id"].startswith("file_")


async def test_unreadable_upload_is_rejected(harness: Harness) -> None:
    response = await harness.client.post("/v1/files", files={"file": ("x.wav", b"not audio at all", "audio/wav")})
    assert response.status_code == 422 and response.json()["code"] == "audio_unreadable"


async def test_job_listing_and_unknown_job(harness: Harness) -> None:
    await _ready_voice(harness)
    jobs = (await harness.client.get("/v1/jobs")).json()["data"]
    assert jobs[0]["type"] == "voice.build" and jobs[0]["status"] == "succeeded"
    missing = await harness.client.get("/v1/jobs/job_00000000000000000000000000")
    assert missing.status_code == 404 and missing.json()["code"] == "job_not_found"


async def test_reference_file_cannot_be_deleted_under_a_voice(harness: Harness) -> None:
    voice = await _ready_voice(harness)
    response = await harness.client.delete(f"/v1/files/{voice['reference']['file_id']}")
    assert response.status_code == 409 and response.json()["code"] == "voice_in_use"


# ---- MCP ---------------------------------------------------------------------------------


async def _mcp(harness: Harness, method: str, params: dict | None = None, *, key: str = API_KEY) -> httpx.Response:
    return await harness.client.post(
        "/mcp/",
        json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}},
        headers={"Authorization": f"Bearer {key}", "Accept": "application/json, text/event-stream",
                 "MCP-Protocol-Version": "2025-06-18"},
    )


async def test_mcp_requires_auth(harness: Harness) -> None:
    response = await _mcp(harness, "tools/list", key="nope-nope-nope-nope-nope-nope")
    assert response.status_code == 401


async def test_mcp_lists_and_calls_tools(harness: Harness) -> None:
    listing = await _mcp(harness, "tools/list")
    assert listing.status_code == 200, listing.text
    names = {tool["name"] for tool in listing.json()["result"]["tools"]}
    assert {"speak", "transcribe", "create_voice", "list_voices", "get_job", "enhance_audio"} <= names
    await _ready_voice(harness)
    call = await _mcp(harness, "tools/call", {"name": "speak", "arguments": {"text": "Hi agent.",
                                                                              "voice": "Narrator"}})
    result = call.json()["result"]
    assert result.get("isError") is not True, result
    payload = result["structuredContent"]
    assert payload["id"].startswith("file_") and payload["url"].endswith("/content")
