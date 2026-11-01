# voiceapi architecture

Agent-first, self-hosted voice service on one RTX 4090 (Ada, SM 8.9, 24 GB).
Decided 2026-11-01. Source of truth for every design decision below.

## Processes (docker-compose, one GPU)

```
            agents / web app
                  │  HTTPS-less LAN HTTP, bearer API keys
          ┌───────▼────────┐
          │    gateway     │  FastAPI, no GPU. Public API, MCP, jobs, storage,
          │   :8080        │  audio encoding, voice catalogue (SQLite + files)
          └──┬──────────┬──┘
   loopback  │          │  loopback
   ┌─────────▼───┐  ┌───▼──────────┐
   │ tts-engine  │  │ audio-worker │
   │ :8000 (GPU) │  │ :8001 (GPU)  │
   │ Nari fork,  │  │ RE-USE clean │
   │ Qwen3-TTS   │  │ Qwen3-ASR    │
   │ 1.7B-Base   │  │ ForcedAligner│
   └─────────────┘  └──────────────┘
```

Why three processes: Nari pins torch 2.13 / CUDA 13 / FlashInfer and captures
CUDA graphs at startup; RE-USE needs `mamba_ssm` and the ASR stack needs its
own transformers pin. Isolation keeps each dependency set exact and lets the
worker unload models without disturbing captured graphs.

Only the gateway publishes a port. Engine and worker bind the compose network.

## Engine (submodule `engine/`, fork coffeegrind123/nari-qwen3-tts)

Base: bglusman's consumer-GPU commit 2ec502b (`--allow-non-h100`, BF16 Talker
projection when SM90 FP8 is unavailable, `consumer` profile).

### Voice cloning without touching captured graphs

Nari's prefill graph materializes embeddings as

    text  = F.embedding(text_ids,  projected_text_cache)
    codec = F.embedding(codec_ids, codec_table)
    embed = where(codec_mask, text + codec, text)

Every voice-clone input the official `qwen_tts` builds is a float vector placed
where a codec embedding would go:

| official input                              | where                         |
|---------------------------------------------|-------------------------------|
| x-vector (speaker encoder output, D=2048)   | speaker slot of codec prefix  |
| ICL ref frame t: Σ_g emb_g(ref_code[t, g])  | one codec position per frame  |

So the engine owns an **extended codec table** `[V + R, D]`: rows `[0, V)` are
the checkpoint's codec embedding, rows `[V, V+R)` are a voice-row pool. A voice
registers its x-vector (1 row) and, for ICL, its summed reference-frame
embeddings (T rows). The prompt is then pure integer IDs and runs through the
existing captured prefill graphs unchanged. The table is allocated once, so
its address is stable under CUDA graph capture; registration writes rows in
place between steps on the engine thread.

Generated tokens are < 2048 and feedback embeddings come from the checkpoint
table, so voice rows can never be sampled or fed back.

Row pool: first-fit allocator over contiguous ranges, refcount pinned by every
admitted request using the voice, LRU eviction of unpinned voices when full.
`voice_table_rows` is an engine config value (default 16384 rows = 64 MiB BF16).

### Prompt layouts (must equal qwen_tts 0.1.1 `generate`)

Common prefix (role = first 3 wrapped text tokens):

    text : role(3) | pad × (len(codec_prefix)+1) | bos
    codec: pad×3   | codec_prefix | SPEAKER | pad          (mask True from prefix)

`SPEAKER` = preset `spk_id` (CustomVoice) or the voice's x-vector row (Base).

x-vector mode then continues exactly like CustomVoice.

ICL mode (`generate_icl_prompt`):

    ref_ids = tokenize("<|im_start|>assistant\n{ref}<|im_end|>\n")[3:-2]
    tgt_ids = wrapped target [3:-5]
    text_part  = ref_ids + tgt_ids + [eos]          (projected text embeddings)
    codec_part = [codec_bos] + ref_frame_rows        (T + 1 positions)

    non-streaming: text_part (+codec_pad, mask) | tts_pad + codec_part ; trailing = [pad]
    streaming    : zip-sum of text_part and codec_part over min length;
                   text longer  → trailing = text_part[codec_len:]
                   codec longer → text padded with tts_pad; trailing = [pad]

The ICL path drops the prefix's final `codec_bos` slot (the official code adds
`codec[:-1]`), because `codec_part` starts with its own `codec_bos`.

Parity is enforced by CPU unit tests that build embeddings both ways from
random tiny weights and require exact equality (`tests/voice_clone/`).

### Codec priming (ICL)

The official decode prepends `ref_code` and trims the reference audio
proportionally, so the first generated frames are decoded with reference
context. Nari decodes a cold stream. Tracked item: snapshot an
`IncrementalCodecState` primed on the reference codes at registration and clone
it per request, entering the warm codec phase directly. Gate: measure onset
spectral distance vs the official decode on the 4090 first.

### Base checkpoint support

* config identity accepts `tts_model_type ∈ {custom_voice, base}`; Base has no
  `spk_id`, no dialect languages, and adds `speaker_encoder_config`.
* speaker encoder (`speaker_encoder.*` weights, ECAPA-style) loaded only for
  Base; mel = n_fft 1024, 128 mels, 24 kHz, hop 256, win 1024, fmax 12 kHz.
* reference codes via the already-loaded 12 Hz speech tokenizer `encode`.
* `instruct` is rejected for Base (the model has no instruction control).

### 4090 profile `ada24g`

Starts from `consumer`; raises KV pages, capture batches and prefill token
buckets (ICL prompts ≈ 3 + 6 + ref_text + target + 120–180 frames). Exact
numbers come from measurement on the card, not from this file.

## Audio worker

Pipeline for a voice build (one job):

1. decode any container with ffmpeg → float32 mono at native rate
2. RE-USE enhance (native rate 8–48 kHz; chunked with overlap-add > 10 s,
   mirroring `inference_chunk.py`)
3. silence trim + loudness normalise (EBU R128 target −20 LUFS)
4. Qwen3-ASR-1.7B transcript (skipped when caller supplies one)
5. Qwen3-ForcedAligner-0.6B word timings
6. reference selection: 5–15 s spans cut on word boundaries, scored on
   duration fit, alignment confidence and estimated SNR
7. output: cleaned clips, chosen reference span(s) + exact transcript, metrics

RE-USE: the reference code requires `mamba_ssm` CUDA kernels. The worker uses
the prebuilt kernels when importable and otherwise a pure-PyTorch selective
scan (bidirectional Mamba, 30 layers, 9.6 M params) — verified numerically
against `mamba_ssm` on the 4090.

ASR choice (Open ASR Leaderboard, 2026-11): Qwen3-ASR-1.7B mean WER 5.40 vs
whisper-large-v3 7.16; only a 30B MoE beats it and that cannot share a 24 GB
card with the TTS engine. Apache-2.0.

## Gateway API

Contract: OpenAPI 3.1 generated from code, plus `/llms.txt`. See
`context/api.md` for the endpoint list and error codes.

## Build / test policy

No compiling on the dev box. GitHub Actions builds images (FlashInfer AOT for
sm_89 baked in) and pushes to GHCR; GPU tests run on the 4090 via
`docker compose` from the dev container.
