"""FastAPI application assembly."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from voiceapi_gateway.api import audio as audio_routes
from voiceapi_gateway.api import meta, resources
from voiceapi_gateway.config import Settings
from voiceapi_gateway.errors import PROBLEM_MEDIA_TYPE, ApiError, ErrorCode
from voiceapi_gateway.ids import new_id
from voiceapi_gateway.mcp_server import BearerGuard, build_mcp, mcp_transport_security
from voiceapi_gateway.state import AppState

log = logging.getLogger("voiceapi")

API_DESCRIPTION = """\
Agent-first voice API. OpenAI-compatible `/v1/audio/speech` and `/v1/audio/transcriptions`,
clone voices built from cleaned reference audio, speech enhancement, durable jobs, and an MCP
endpoint at `/mcp`. Errors are RFC 9457 problem documents with a stable `code` and an actionable
`hint`. Start with `GET /llms.txt` or `GET /v1/capabilities`.
"""

_STATUS_CODES = {
    400: ErrorCode.BAD_REQUEST,
    401: ErrorCode.UNAUTHORIZED,
    404: ErrorCode.NOT_FOUND,
    405: ErrorCode.BAD_REQUEST,
    413: ErrorCode.PAYLOAD_TOO_LARGE,
    415: ErrorCode.UNSUPPORTED_FORMAT,
    422: ErrorCode.VALIDATION_FAILED,
}


def _problem(error: ApiError, request: Request) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    headers = {"WWW-Authenticate": "Bearer"} if error.status == 401 else None
    return JSONResponse(error.problem(request_id), status_code=error.status, media_type=PROBLEM_MEDIA_TYPE,
                        headers=headers)


def create_app(settings: Settings | None = None, *, state: AppState | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    app_state = state or AppState.build(settings)
    mcp_base_url = (settings.public_base_url or "http://localhost:8080").rstrip("/")
    mcp = build_mcp(app_state, mcp_base_url)
    mcp_app = mcp.streamable_http_app(
        streamable_http_path="/",
        stateless_http=True,
        json_response=True,
        transport_security=mcp_transport_security(app_state),
    )

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await app_state.start()
        async with mcp.session_manager.run():
            try:
                yield
            finally:
                await app_state.stop()

    app = FastAPI(
        title="voiceapi",
        version="0.1.0",
        description=API_DESCRIPTION,
        lifespan=lifespan,
        openapi_tags=[
            {"name": "audio", "description": "Speech synthesis, transcription and enhancement"},
            {"name": "voices", "description": "Clone voices built from reference clips"},
            {"name": "files", "description": "Audio files in the data volume"},
            {"name": "jobs", "description": "Long-running work; poll or block with ?wait="},
            {"name": "meta", "description": "Health, discovery and capability description"},
        ],
    )
    app.state.voiceapi = app_state

    @app.middleware("http")
    async def request_id(request: Request, call_next):
        rid = request.headers.get("x-request-id") or new_id("req")
        request.state.request_id = rid[:100]
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    @app.exception_handler(ApiError)
    async def api_error(request: Request, error: ApiError) -> JSONResponse:
        return _problem(error, request)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, error: RequestValidationError) -> JSONResponse:
        issues = [
            {"location": ".".join(str(part) for part in issue.get("loc", ())), "message": issue.get("msg", "")}
            for issue in error.errors()
        ]
        summary = "; ".join(f"{item['location']}: {item['message']}" for item in issues[:5])
        problem = ApiError(422, ErrorCode.VALIDATION_FAILED, summary or "request validation failed",
                           "Check the field types against GET /openapi.json.", extra={"issues": issues})
        return _problem(problem, request)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, error: StarletteHTTPException) -> JSONResponse:
        code = _STATUS_CODES.get(error.status_code, ErrorCode.INTERNAL)
        return _problem(ApiError(error.status_code, code, str(error.detail)), request)

    @app.exception_handler(Exception)
    async def unexpected(request: Request, error: Exception) -> JSONResponse:
        log.exception("unhandled error on %s %s", request.method, request.url.path)
        problem = ApiError(500, ErrorCode.INTERNAL, "internal error", "Retry; report the request_id if it persists.")
        return _problem(problem, request)

    app.include_router(meta.public)
    app.include_router(meta.private)
    app.include_router(audio_routes.router)
    app.include_router(resources.router)
    app.mount("/mcp", BearerGuard(mcp_app, app_state))
    return app


__all__ = ["create_app"]
