"""Audio worker HTTP API (internal; reached only by the gateway over the compose network).

Requests carry paths inside the shared data volume; every path is resolved
and confined to ``VOICEAPI_DATA_DIR`` before use.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from voiceapi_worker.asr import ASR_SAMPLE_RATE, Transcriber, UnsupportedLanguageError
from voiceapi_worker.audio_io import AudioDecodeError, decode, nearest_supported_rate, resample, write_wav
from voiceapi_worker.reference import ReferenceError, prepare_reference
from voiceapi_worker.reuse.enhancer import SUPPORTED_RATES, ReuseEnhancer, snr_estimate_db

log = logging.getLogger("voiceapi.worker")


@dataclass(frozen=True, slots=True)
class WorkerSettings:
    data_dir: Path
    device: torch.device
    cache_dir: str | None
    preload_asr: bool
    asr_idle_unload_s: float

    @classmethod
    def from_env(cls) -> WorkerSettings:
        device = os.environ.get("VOICEAPI_WORKER_DEVICE") or ("cuda:0" if torch.cuda.is_available() else "cpu")
        return cls(
            data_dir=Path(os.environ.get("VOICEAPI_DATA_DIR", "/data")).resolve(),
            device=torch.device(device),
            cache_dir=os.environ.get("HF_HOME"),
            preload_asr=os.environ.get("VOICEAPI_WORKER_PRELOAD_ASR", "1") not in {"0", "false", "no"},
            asr_idle_unload_s=float(os.environ.get("VOICEAPI_WORKER_ASR_IDLE_UNLOAD_S", "0")),
        )


class WorkerError(Exception):
    def __init__(self, status: int, code: str, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.code = code
        self.detail = detail


class Models:
    """Owns the GPU models; the ASR stack can be unloaded when idle to give VRAM back."""

    def __init__(self, settings: WorkerSettings) -> None:
        self.settings = settings
        self.enhancer: ReuseEnhancer | None = None
        self._transcriber: Transcriber | None = None
        self._lock = threading.Lock()
        self._last_asr_use = time.monotonic()

    def load(self) -> None:
        self.enhancer = ReuseEnhancer.load(self.settings.device, cache_dir=self.settings.cache_dir)
        log.info("RE-USE loaded on %s (scan backend: %s)", self.settings.device, self.enhancer.scan_backend.value)
        if self.settings.preload_asr:
            self.transcriber()

    def transcriber(self) -> Transcriber:
        with self._lock:
            self._last_asr_use = time.monotonic()
            if self._transcriber is None:
                started = time.monotonic()
                self._transcriber = Transcriber(self.settings.device, cache_dir=self.settings.cache_dir)
                log.info("Qwen3-ASR loaded in %.1f s", time.monotonic() - started)
            return self._transcriber

    def unload_idle(self) -> None:
        limit = self.settings.asr_idle_unload_s
        if limit <= 0:
            return
        with self._lock:
            if self._transcriber is not None and time.monotonic() - self._last_asr_use > limit:
                self._transcriber = None
                if self.settings.device.type == "cuda":
                    torch.cuda.empty_cache()
                log.info("Qwen3-ASR unloaded after %.0f s idle", limit)


class EnhanceBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input_path: str
    output_path: str
    bandwidth_extension_hz: int | None = None


class TranscribeBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input_path: str
    language: str | None = None
    prompt: Annotated[str | None, Field(max_length=4000)] = None
    word_timestamps: bool = False
    stream: bool = False


class ReferenceBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input_paths: Annotated[list[str], Field(min_length=1, max_length=20)]
    output_dir: str
    transcript: str | None = None
    language: str | None = None
    clean: bool = True


class WorkerService:
    """Endpoint logic; ``create_app`` only wires routes to these methods."""

    def __init__(self, settings: WorkerSettings, models: Models) -> None:
        self.settings = settings
        self.models = models
        self.ready = False
        self.error: str | None = None

    def start(self) -> None:
        def load() -> None:
            try:
                self.models.load()
                self.ready = True
            except Exception as error:  # noqa: BLE001 - reported through /ready
                log.exception("model loading failed")
                self.error = f"{error.__class__.__name__}: {error}"

        def janitor() -> None:
            while True:
                time.sleep(30)
                self.models.unload_idle()

        threading.Thread(target=load, name="worker-model-load", daemon=True).start()
        threading.Thread(target=janitor, name="worker-janitor", daemon=True).start()

    def confine(self, raw: str, *, must_exist: bool) -> Path:
        path = Path(raw).resolve()
        if not path.is_relative_to(self.settings.data_dir):
            raise WorkerError(422, "validation_failed", f"path outside the data directory: {raw}")
        if must_exist and not path.is_file():
            raise WorkerError(422, "file_not_found", f"no such file: {raw}")
        return path

    def require_ready(self) -> None:
        if not self.ready:
            raise WorkerError(503, "worker_unavailable", self.error or "models are still loading")

    def readiness(self) -> JSONResponse:
        enhancer = self.models.enhancer
        body = {"ready": self.ready, "error": self.error, "device": str(self.settings.device),
                "scan_backend": enhancer.scan_backend.value if enhancer else None}
        return JSONResponse(body, status_code=200 if self.ready else 503)

    def enhance(self, body: EnhanceBody) -> dict:
        self.require_ready()
        source = self.confine(body.input_path, must_exist=True)
        target = self.confine(body.output_path, must_exist=False)
        try:
            audio, rate = decode(source)
        except AudioDecodeError as error:
            raise WorkerError(422, "audio_unreadable", str(error)) from error
        work_rate = body.bandwidth_extension_hz or nearest_supported_rate(rate, SUPPORTED_RATES)
        if work_rate not in SUPPORTED_RATES:
            raise WorkerError(422, "validation_failed", f"bandwidth_extension_hz must be one of {SUPPORTED_RATES}")
        snr_before = snr_estimate_db(audio, rate)
        started = time.monotonic()
        enhanced = self.models.enhancer.enhance(resample(audio, rate, work_rate), work_rate)
        write_wav(target, enhanced, work_rate)
        return {
            "duration_s": round(len(enhanced) / work_rate, 3),
            "sample_rate": work_rate,
            "snr_before_db": round(snr_before, 2),
            "snr_after_db": round(snr_estimate_db(enhanced, work_rate), 2),
            "elapsed_s": round(time.monotonic() - started, 3),
            "scan_backend": self.models.enhancer.scan_backend.value,
        }

    def transcribe(self, body: TranscribeBody):
        self.require_ready()
        source = self.confine(body.input_path, must_exist=True)
        try:
            audio, _ = decode(source, sample_rate=ASR_SAMPLE_RATE)
        except AudioDecodeError as error:
            raise WorkerError(422, "audio_unreadable", str(error)) from error
        transcriber = self.models.transcriber()
        try:
            if not body.stream:
                return transcriber.transcribe(audio, language=body.language, prompt=body.prompt,
                                              word_timestamps=body.word_timestamps)
            events = transcriber.stream(audio, language=body.language, prompt=body.prompt)
            first = next(events)  # surface language errors before the response starts
        except UnsupportedLanguageError as error:
            raise WorkerError(422, "unsupported_language", str(error)) from error

        def sse() -> Iterator[str]:
            yield f"data: {json.dumps(first)}\n\n"
            for event in events:
                yield f"data: {json.dumps(event)}\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

    def reference(self, body: ReferenceBody) -> dict:
        self.require_ready()
        inputs = [self.confine(path, must_exist=True) for path in body.input_paths]
        output_dir = self.confine(body.output_dir, must_exist=False)
        try:
            return prepare_reference(
                inputs, output_dir, transcript=body.transcript, language=body.language, clean=body.clean,
                enhancer=self.models.enhancer, transcriber=self.models.transcriber(),
            )
        except ReferenceError as error:
            raise WorkerError(422, error.code, str(error)) from error
        except AudioDecodeError as error:
            raise WorkerError(422, "audio_unreadable", str(error)) from error
        except UnsupportedLanguageError as error:
            raise WorkerError(422, "unsupported_language", str(error)) from error


def create_app(settings: WorkerSettings | None = None, models: Models | None = None) -> FastAPI:
    settings = settings or WorkerSettings.from_env()
    service = WorkerService(settings, models or Models(settings))

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI):
        service.start()
        yield

    app = FastAPI(title="voiceapi audio worker", lifespan=lifespan)

    @app.exception_handler(WorkerError)
    async def worker_error(_request, error: WorkerError) -> JSONResponse:
        return JSONResponse({"detail": error.detail, "code": error.code}, status_code=error.status)

    app.get("/health")(lambda: {"alive": service.error is None})
    app.get("/ready")(service.readiness)
    app.post("/v1/enhance")(service.enhance)
    app.post("/v1/transcribe")(service.transcribe)
    app.post("/v1/reference")(service.reference)
    return app


__all__ = ["Models", "WorkerService", "WorkerSettings", "create_app"]
