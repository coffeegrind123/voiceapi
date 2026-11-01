"""In-process stand-ins for the tts-engine and audio-worker HTTP APIs.

They implement the same request/response shapes as the real services so the
gateway is exercised end to end (routing, jobs, storage, ffmpeg) without a GPU.
"""

from __future__ import annotations

import base64
import json
import math
import struct
import subprocess
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

SAMPLE_RATE = 24_000


def sine_pcm(seconds: float, frequency: float = 220.0) -> bytes:
    count = int(SAMPLE_RATE * seconds)
    return b"".join(
        struct.pack("<h", int(12000 * math.sin(2 * math.pi * frequency * index / SAMPLE_RATE)))
        for index in range(count)
    )


def wav_bytes(seconds: float, frequency: float = 220.0) -> bytes:
    pcm = sine_pcm(seconds, frequency)
    header = (
        b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, SAMPLE_RATE, SAMPLE_RATE * 2, 2, 16)
        + b"data" + struct.pack("<I", len(pcm))
    )
    return header + pcm


class FakeEngine:
    def __init__(self) -> None:
        self.voices: dict[str, dict] = {}
        self.speech_requests: list[dict] = []
        self.ready = True
        self.app = self._build()

    def _build(self) -> FastAPI:
        app = FastAPI()

        @app.get("/ready")
        def ready() -> JSONResponse:
            return JSONResponse({"ready": self.ready, "phase": "ready" if self.ready else "starting"},
                                status_code=200 if self.ready else 503)

        @app.post("/v1/audio/speech")
        async def speech(request: Request) -> Response:
            body = await request.json()
            self.speech_requests.append(body)
            if body["voice"] not in self.voices:
                raise HTTPException(422, f"unknown voice {body['voice']!r}; register it first")
            seconds = min(2.0, 0.05 * len(body["input"]))
            pcm = sine_pcm(seconds)

            async def chunks():
                for start in range(0, len(pcm), 4801):  # odd size: exercises sample re-alignment
                    yield pcm[start : start + 4801]

            return StreamingResponse(chunks(), media_type="audio/pcm")

        @app.post("/v1/voices", status_code=201)
        async def build(request: Request) -> dict:
            body = await request.json()
            audio = base64.b64decode(body["audio_base64"])
            self.last_build = body
            if not audio.startswith(b"RIFF"):
                raise HTTPException(422, "unreadable reference audio")
            icl = bool(body.get("ref_text")) and not body.get("x_vector_only")
            info = {"voice_id": body["voice_id"], "supports_icl": icl, "frames": 120 if icl else 0, "rows": 1,
                    "pins": 0}
            prompt = json.dumps({"voice_id": body["voice_id"], "icl": icl}).encode()
            self.voices[body["voice_id"]] = {"info": info, "prompt": prompt}
            return {"voice": info, "prompt_base64": base64.b64encode(prompt).decode()}

        @app.put("/v1/voices/{voice_id}")
        async def put(voice_id: str, request: Request) -> dict:
            prompt = await request.body()
            stored = json.loads(prompt)
            info = {"voice_id": voice_id, "supports_icl": stored["icl"], "frames": 120 if stored["icl"] else 0,
                    "rows": 1, "pins": 0}
            self.voices[voice_id] = {"info": info, "prompt": prompt}
            return info

        @app.get("/v1/voices")
        def list_voices() -> dict:
            return {"object": "list", "data": [v["info"] for v in self.voices.values()]}

        @app.delete("/v1/voices/{voice_id}", status_code=204)
        def delete(voice_id: str) -> Response:
            if self.voices.pop(voice_id, None) is None:
                raise HTTPException(404, "not registered")
            return Response(status_code=204)

        return app


class FakeWorker:
    def __init__(self, ffmpeg: str) -> None:
        self.ffmpeg = ffmpeg
        self.reference_calls: list[dict] = []
        self.app = self._build()

    def _to_wav(self, source: str, target: Path, seconds: float | None = None) -> None:
        args = [self.ffmpeg, "-nostdin", "-loglevel", "error", "-y", "-i", source]
        if seconds:
            args += ["-t", str(seconds)]
        args += ["-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", str(target)]
        subprocess.run(args, check=True)

    def _build(self) -> FastAPI:
        app = FastAPI()

        @app.get("/ready")
        def ready() -> dict:
            return {"ready": True}

        @app.post("/v1/enhance")
        async def enhance(request: Request) -> dict:
            body = await request.json()
            self._to_wav(body["input_path"], Path(body["output_path"]))
            return {"duration_s": 1.0, "sample_rate": 24000, "snr_before_db": 5.0, "snr_after_db": 25.0}

        @app.post("/v1/transcribe")
        async def transcribe(request: Request):
            body = await request.json()
            if not Path(body["input_path"]).is_file():
                raise HTTPException(422, "missing input")
            text = "hello from the fake worker"
            if body.get("stream"):
                async def events():
                    for word in text.split():
                        yield f"data: {json.dumps({'delta': word + ' '})}\n\n"
                    yield f"data: {json.dumps({'done': True, 'text': text, 'language': 'English'})}\n\n"

                return StreamingResponse(events(), media_type="text/event-stream")
            result = {"text": text, "language": "English", "duration": 1.0}
            if body.get("word_timestamps"):
                result["words"] = [{"word": w, "start": i * 0.2, "end": i * 0.2 + 0.2}
                                   for i, w in enumerate(text.split())]
            return result

        @app.post("/v1/reference")
        async def reference(request: Request) -> dict:
            body = await request.json()
            self.reference_calls.append(body)
            output_dir = Path(body["output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            target = output_dir / "reference.wav"
            self._to_wav(body["input_paths"][0], target, seconds=8)
            transcript = body.get("transcript") or "this is the reference transcript"
            xvector = output_dir / "xvector.wav"
            self._to_wav(body["input_paths"][0], xvector)
            return {
                "reference_path": str(target),
                "xvector_path": str(xvector),
                "reference_text": transcript,
                "reference_duration_s": 8.0,
                "language": "English",
                "clips_used": len(body["input_paths"]),
                "quality": {"snr_before_db": 8.0, "snr_after_db": 30.0, "alignment_confidence": 0.97},
            }

        return app
