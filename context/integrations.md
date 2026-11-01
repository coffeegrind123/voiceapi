# Client integrations (researched 2026-11-01)

The user runs Ovis (devswha/ovis, bundle remote-v40) on herdr (herdrdev/herdr),
hosting Codex CLI 0.166.0 (patched, provider = local router :8877) and pi
(@earendil-works/pi-coding-agent 1.4.0). Wiring lives in
coffeegrind123/instantcoffee and coffeegrind123/codex-workspace. Neither repo
had any voice code as of this date.

## Ovis — speech-to-text: already supported via base URL

`server/voice.ts` forwards browser recordings to `${OVIS_OPENAI_BASE_URL}/audio/transcriptions`
(key `OVIS_OPENAI_API_KEY`, default model `gpt-transcribe`). Sends multipart
`file=audio.webm` (Safari: mp4), `model`, `stream=true`, repeated
`languages[]`, `prompt`; ≤ 32 fields; clips ≤ 10 MB / 120 s. Accepts SSE
`transcript.text.delta` / `transcript.text.done` or JSON `{text}`.
Optional "polish" hits `${base}/chat/completions` (failure only logs a warning).

→ Gateway must implement exactly that request/response shape. Integration =
set `OVIS_OPENAI_BASE_URL=http://<gateway>:8080/v1` and the key in instantcoffee
(verify `scripts/ovis.sh` passes them through). No TTS output in Ovis: reading
replies aloud in-browser needs an Ovis bundle patch (`scripts/ovis_patch.py`).
Maintainer declined a self-hosted STT provider (issue #911); base URL override
makes that moot.

## herdr

Plugin system: actions, `[[events]]` (e.g. `pane.agent_status_changed`),
socket API `agent.read`, `pane.input.set`. Agent-agnostic read-aloud = herdr
plugin: on agent settled → `agent.read` → POST /v1/audio/speech → play.
Prior art: chiptime/herdr-tts (Edge/OpenAI/ElevenLabs), herdr-voice (Whisper).

## Codex CLI 0.166.0

Native voice = OpenAI Realtime only (WS/WebRTC, local ALSA mic) — not usable
from the container/browser. Never calls /audio/speech or /audio/transcriptions.
Integration: MCP (`[mcp_servers]` → our `/mcp`) or a workspace plugin modelled
on `plugins/deepseek-balance`; read-aloud via `Stop` hook
(`last_assistant_message`) → /v1/audio/speech.
Optional later: `/v1/realtime` transcription sessions for
`experimental_realtime_ws_base_url`.

## pi 1.4.0

No core voice. Extensions API (`registerTool`, events `agent_settled`, …) and
built-in MCP (`mcp.json` / `pi.registerMcpServer`). khanra17/pi-extensions
dictation uses provider base URL `/audio/transcriptions` (after trying
`/realtime?intent=transcription`). Integration: MCP + optional `agent_settled`
read-aloud extension.

## Surface priority

1. OpenAI-compatible `/v1/audio/transcriptions` + `/v1/audio/speech` (Ovis, pi
   dictation, any Whisper/OpenAI client).
2. `/mcp` streamable HTTP (Codex, pi, any agent).
3. Thin read-aloud hooks (herdr plugin / Codex Stop hook / pi extension).
4. Optional `/v1/realtime` transcription WS.
