# voiceapi

Self-hosted, agent-first voice service for one NVIDIA GPU:

- **Text-to-speech with cloned voices**: Qwen3-TTS 1.7B Base on a CUDA Graph engine
  (fork of Nari Qwen3-TTS, `engine/` submodule) with x-vector and in-context cloning.
- **Clip cleanup**: NVIDIA NemotronLabs RE-USE speech enhancement on every clip used to
  build a voice, also exposed as `/v1/audio/enhance`.
- **Speech-to-text**: Qwen3-ASR 1.7B, with Qwen3-ForcedAligner word timestamps.
- **Agent surface**: OpenAI-compatible `/v1/audio/speech` and `/v1/audio/transcriptions`,
  an MCP endpoint at `/mcp`, `/llms.txt`, RFC 9457 errors with stable codes and hints,
  durable jobs with `?wait=`, and idempotency keys.

```
agents / web app ──HTTP──► gateway :8080 ──► tts-engine (GPU)   Qwen3-TTS Base
                                         └─► audio-worker (GPU) RE-USE, Qwen3-ASR, aligner
```

## Run

```bash
cp .env.example .env              # set VOICEAPI_API_KEYS (>= 24 chars each)
docker compose pull
docker compose up -d
curl -s http://localhost:8080/ready
```

The first start downloads about 9 GB of model weights and captures CUDA graphs. Later
starts reuse the cache volumes; set `VOICEAPI_OFFLINE=1` to stop the engine downloading
at startup.

## Use

```bash
KEY=...; BASE=http://localhost:8080
# 1. upload a clip and clone a voice (consent is mandatory)
FILE=$(curl -s -H "Authorization: Bearer $KEY" -F file=@me.wav $BASE/v1/files | jq -r .id)
curl -s -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d "{\"name\":\"Me\",\"file_ids\":[\"$FILE\"],\"consent\":true}" "$BASE/v1/voices?wait=60"
# 2. speak
curl -s -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"input":"Hello from my cloned voice.","voice":"Me","response_format":"mp3"}' \
  $BASE/v1/audio/speech -o hello.mp3
# 3. transcribe
curl -s -H "Authorization: Bearer $KEY" -F file=@hello.mp3 $BASE/v1/audio/transcriptions
```

Agents: point MCP clients at `$BASE/mcp` (streamable HTTP, same bearer token), or read
`$BASE/llms.txt`. OpenAI SDKs work with `base_url=$BASE/v1`.

## Layout

| path | what |
|---|---|
| `engine/` | Nari Qwen3-TTS fork (Base-model cloning, consumer/Ada profiles) |
| `services/gateway/` | public API, MCP, jobs, storage, audio encoding |
| `services/audio-worker/` | RE-USE (vendored model code), Qwen3-ASR, ForcedAligner, reference selection |
| `context/` | design records: architecture, API contract, client integrations |

## Development

No local compilation: GitHub Actions runs lint and CPU tests (`ci.yml`) and builds the
three images into GHCR (`images.yml`). The GPU paths are validated on the target card
with `docker compose`.
