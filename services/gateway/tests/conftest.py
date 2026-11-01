from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest

from voiceapi_gateway.app import create_app
from voiceapi_gateway.clients.engine import EngineClient
from voiceapi_gateway.clients.worker import WorkerClient
from voiceapi_gateway.config import Limits, Settings
from voiceapi_gateway.state import AppState

from .fakes import FakeEngine, FakeWorker

API_KEY = "test-key-0123456789abcdefghij"

# ffmpeg/ffprobe must be on PATH; VOICEAPI_TEST_FFMPEG_DIR points at a static build when the host has none.
_ffmpeg_dir = os.environ.get("VOICEAPI_TEST_FFMPEG_DIR")
if _ffmpeg_dir:
    os.environ["PATH"] = f"{_ffmpeg_dir}{os.pathsep}{os.environ['PATH']}"
FFMPEG = shutil.which("ffmpeg")
if FFMPEG is None or shutil.which("ffprobe") is None:
    pytest.exit("ffmpeg and ffprobe are required for the gateway tests", returncode=2)


class Harness:
    def __init__(self, client: httpx.AsyncClient, engine: FakeEngine, worker: FakeWorker, state: AppState) -> None:
        self.client = client
        self.engine = engine
        self.worker = worker
        self.state = state


@pytest.fixture
async def harness(tmp_path: Path) -> AsyncIterator[Harness]:
    engine = FakeEngine()
    worker = FakeWorker(FFMPEG)
    settings = Settings(
        data_dir=tmp_path / "data",
        api_keys=frozenset({API_KEY}),
        default_voice=None,
        limits=Limits(job_workers=1, max_sync_enhance_seconds=30.0),
    )
    state = AppState.build(
        settings,
        engine=EngineClient("http://engine", timeout_s=30, transport=httpx.ASGITransport(app=engine.app)),
        worker=WorkerClient("http://worker", timeout_s=30, transport=httpx.ASGITransport(app=worker.app)),
    )
    app = create_app(settings, state=state)
    started = asyncio.Event()
    stop = asyncio.Event()

    async def run_lifespan() -> None:
        # Enter and exit the lifespan (and the MCP task group inside it) in one task.
        async with app.router.lifespan_context(app):
            started.set()
            await stop.wait()

    lifespan = asyncio.create_task(run_lifespan())
    await asyncio.wait({lifespan, asyncio.create_task(started.wait())}, return_when=asyncio.FIRST_COMPLETED)
    if lifespan.done():
        lifespan.result()
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://gateway", headers={"Authorization": f"Bearer {API_KEY}"},
            timeout=60,
        ) as client:
            yield Harness(client, engine, worker, state)
    finally:
        stop.set()
        with contextlib.suppress(asyncio.CancelledError):
            await lifespan
