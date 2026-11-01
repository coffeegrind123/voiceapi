"""Shared route plumbing: state access, auth, idempotency, waits, pagination."""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from fastapi import Depends, Header, Query, Request
from fastapi.responses import JSONResponse
from starlette.datastructures import UploadFile

from voiceapi_gateway.errors import ApiError, ErrorCode
from voiceapi_gateway.state import AppState

UPLOAD_CHUNK = 1024 * 1024
PUBLIC_PATHS = frozenset({"/health", "/ready", "/llms.txt", "/openapi.json", "/docs", "/redoc"})


def get_state(request: Request) -> AppState:
    return request.app.state.voiceapi


def base_url(request: Request) -> str:
    state = get_state(request)
    if state.settings.public_base_url:
        return state.settings.public_base_url.rstrip("/")
    return str(request.base_url).rstrip("/")


def check_bearer(state: AppState, authorization: str | None) -> None:
    if not state.settings.api_keys:
        return
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise ApiError(401, ErrorCode.UNAUTHORIZED, "missing bearer token", "Send Authorization: Bearer <api key>.")
    token = token.strip()
    if not any(hmac.compare_digest(token, key) for key in state.settings.api_keys):
        raise ApiError(401, ErrorCode.UNAUTHORIZED, "invalid API key")


async def require_auth(request: Request, authorization: str | None = Header(default=None)) -> None:
    check_bearer(get_state(request), authorization)


Authenticated = Depends(require_auth)


def wait_param(wait: float = Query(0.0, ge=0.0, le=60.0, description="Seconds to block for a job to finish")) -> float:
    return wait


def limit_param(limit: int = Query(20, ge=1, le=100)) -> int:
    return limit


def body_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


async def idempotent(
    request: Request,
    scope: str,
    payload: Any,
    create: Callable[[], Awaitable[tuple[int, Any]]],
) -> JSONResponse:
    """Replay the stored response for a repeated ``Idempotency-Key``; reject reuse with a different body."""
    state = get_state(request)
    key = request.headers.get("idempotency-key")
    if not key:
        status, body = await create()
        return JSONResponse(body, status_code=status)
    if len(key) > 255:
        raise ApiError(422, ErrorCode.VALIDATION_FAILED, "Idempotency-Key is longer than 255 characters")
    digest = body_hash(payload)
    previous = await state.db.idempotent_lookup(key, scope, state.settings.limits.idempotency_ttl_seconds)
    if previous is not None:
        if previous["body_hash"] != digest:
            raise ApiError(
                409, ErrorCode.IDEMPOTENCY_CONFLICT, "Idempotency-Key was used with a different request body",
                "Use a new key for a different request.",
            )
        return JSONResponse(
            previous["response"], status_code=previous["status"], headers={"Idempotent-Replayed": "true"}
        )
    status, body = await create()
    await state.db.idempotent_store(key, scope, digest, status, body)
    return JSONResponse(body, status_code=status)


async def read_form(request: Request, *, max_fields: int):
    try:
        return await request.form(max_files=4, max_fields=max_fields, max_part_size=1024 * 1024)
    except Exception as error:  # noqa: BLE001 - starlette raises several multipart error types
        message = str(error)
        if "fields" in message.lower():
            raise ApiError(413, ErrorCode.TOO_MANY_FIELDS, message) from error
        raise ApiError(400, ErrorCode.BAD_REQUEST, f"invalid multipart body: {message}") from error


async def upload_chunks(upload: UploadFile) -> AsyncIterator[bytes]:
    while chunk := await upload.read(UPLOAD_CHUNK):
        yield chunk


def form_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def job_response(job: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """200 when the job already finished, 202 while it is still running."""
    terminal = job["status"] in {"succeeded", "failed", "cancelled"}
    return (200 if terminal else 202), job


__all__ = [
    "PUBLIC_PATHS",
    "Authenticated",
    "base_url",
    "check_bearer",
    "form_bool",
    "get_state",
    "idempotent",
    "job_response",
    "limit_param",
    "read_form",
    "upload_chunks",
    "wait_param",
]
