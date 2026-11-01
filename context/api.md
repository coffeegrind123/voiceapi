# Gateway API contract

Base URL `http://<host>:8080`. JSON unless noted. Auth: `Authorization: Bearer <key>`
on everything except `/health`, `/ready`, `/llms.txt`, `/openapi.json`.
The machine-readable contract is `/openapi.json` (generated); this file is the
design record.

## Conventions

* IDs: `voice_<26 lowercase ULID>`, `file_…`, `job_…`. The engine voice handle
  is the voice id with `voice_` → `v_` (the engine's name regex is
  `^[a-z0-9][a-z0-9_]{0,63}$`).
* Errors: RFC 9457 `application/problem+json`
  `{type, title, status, detail, code, hint?, request_id}`. `code` is stable and
  machine-matchable; `hint` tells an agent what to do next.
* Idempotency: `Idempotency-Key` header on every POST that creates something;
  same key + same body → same response for 24 h; same key + different body →
  409 `idempotency_conflict`.
* Long work returns `202` + a job (`Location: /v1/jobs/{id}`); `?wait=<s>`
  (≤ 60) blocks until done or the timeout, then answers 200 or 202.
* Every response carries `X-Request-ID`.

## Endpoints

| method | path | notes |
|---|---|---|
| GET | /health, /ready | ready = engine ready ∧ worker reachable |
| GET | /llms.txt | plain-text guide for agents |
| GET | /v1/capabilities | languages, formats, limits, models, clone modes |
| GET | /v1/models | OpenAI list: `qwen3-tts-1.7b-base` (+ aliases `tts-1`, `tts-1-hd`, `gpt-4o-mini-tts`), `qwen3-asr-1.7b` (+ `whisper-1`, `gpt-4o-transcribe`, `gpt-transcribe`) |
| POST | /v1/audio/speech | OpenAI-compatible; see below |
| POST | /v1/audio/transcriptions | OpenAI-compatible multipart; see below |
| POST | /v1/audio/enhance | RE-USE; sync ≤ 30 s audio, else job |
| POST | /v1/files | multipart upload, returns `file_…` (audio only, ≤ 200 MB) |
| GET | /v1/files/{id}, /v1/files/{id}/content | metadata / bytes |
| DELETE | /v1/files/{id} | |
| POST | /v1/voices | create voice from clips → job `voice.build` |
| GET | /v1/voices | list (`?limit&after`) |
| GET/PATCH/DELETE | /v1/voices/{id} | PATCH: name, description, tags, default_mode |
| GET | /v1/voices/{id}/reference | the chosen cleaned reference audio |
| POST | /v1/voices/{id}/preview | short speech sample |
| GET | /v1/jobs/{id}, POST /v1/jobs/{id}/cancel, GET /v1/jobs | |
| GET/PATCH | /v1/settings | `{"default_voice": id/name/null}`: persisted default voice for every client |
| * | /mcp | MCP streamable HTTP (same auth) |
| POST | /v1/chat/completions | optional passthrough to `VOICEAPI_CHAT_UPSTREAM` (Ovis "polish") |

### POST /v1/audio/speech

```json
{"model": "qwen3-tts-1.7b-base", "input": "text", "voice": "voice_… | name",
 "response_format": "mp3|opus|aac|flac|wav|pcm", "speed": 1.0,
 "stream_format": "audio|sse", "language": "auto|english|…",
 "voice_mode": "auto|icl|xvector", "seed": 0, "temperature": 0.9}
```

* `voice` resolves by id, then by unique name; OpenAI preset names
  (`alloy`, `echo`, …) map to `VOICEAPI_DEFAULT_VOICE` when configured, else
  404 `voice_not_found` with a hint listing voices.
* `speed` ≠ 1.0 → time-stretch after synthesis (ffmpeg `atempo`, 0.25–4.0).
* `instructions` → 422 `instructions_unsupported` (Base model has none).
* Long input is split on sentence boundaries into engine-sized chunks and the
  audio is concatenated in order; streaming starts with the first chunk.
* `stream_format: sse` → `speech.audio.delta` (base64) / `speech.audio.done`.

### POST /v1/audio/transcriptions (Ovis-compatible)

multipart: `file` (webm/opus, mp4/m4a, wav, mp3, ogg, flac), `model`,
`language` or repeated `languages[]`, `prompt`, `response_format`
(`json|text|verbose_json|srt|vtt`), `timestamp_granularities[]` (`word`,
`segment`), `stream` (bool).

* `stream=true` → SSE `transcript.text.delta {delta}` …
  `transcript.text.done {text}` (Ovis parses these).
* JSON reply has `Content-Type: application/json` and `{text}` minimum.
* Accept ≤ 32 form fields, ≤ 100 MB.

### POST /v1/voices

multipart or JSON:
`name`, `description?`, `files[]` or `file_ids[]` (1–20 clips, total ≤ 30 min),
`transcript?` (applies when exactly one clip), `language?`, `clean` (default
true), `mode` (`auto|icl|xvector`, default auto).

Job result: voice with `reference: {duration_s, transcript, file_id}`,
`quality: {snr_before_db, snr_after_db, alignment_confidence, clips_used}`,
`supports_icl`.

## MCP tools

`speak(text, voice?, format?, language?) → {file_id, url, duration_s}`,
`transcribe(file_id|url, language?) → {text, language, words?}`,
`enhance_audio(file_id) → {file_id, url}`,
`create_voice(name, file_ids, transcript?) → {job_id}`,
`list_voices()`, `get_voice(id)`, `delete_voice(id)`, `get_job(id)`,
`upload_audio(base64, filename) → {file_id}` (for agents without HTTP upload).
Audio is returned as URLs into `/v1/files/{id}/content`, never inline base64.
