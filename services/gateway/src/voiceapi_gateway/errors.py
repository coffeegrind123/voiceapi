"""RFC 9457 problem details with stable, agent-matchable codes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

PROBLEM_MEDIA_TYPE = "application/problem+json"
PROBLEM_TYPE_BASE = "https://voiceapi.local/problems/"


class ErrorCode(StrEnum):
    BAD_REQUEST = "bad_request"
    VALIDATION_FAILED = "validation_failed"
    UNAUTHORIZED = "unauthorized"
    NOT_FOUND = "not_found"
    VOICE_NOT_FOUND = "voice_not_found"
    FILE_NOT_FOUND = "file_not_found"
    JOB_NOT_FOUND = "job_not_found"
    VOICE_NOT_READY = "voice_not_ready"
    VOICE_NAME_TAKEN = "voice_name_taken"
    CONSENT_REQUIRED = "consent_required"
    INSTRUCTIONS_UNSUPPORTED = "instructions_unsupported"
    UNSUPPORTED_FORMAT = "unsupported_format"
    UNSUPPORTED_MODEL = "unsupported_model"
    UNSUPPORTED_LANGUAGE = "unsupported_language"
    AUDIO_UNREADABLE = "audio_unreadable"
    AUDIO_TOO_LONG = "audio_too_long"
    AUDIO_TOO_SHORT = "audio_too_short"
    PAYLOAD_TOO_LARGE = "payload_too_large"
    TOO_MANY_FIELDS = "too_many_fields"
    IDEMPOTENCY_CONFLICT = "idempotency_conflict"
    JOB_NOT_CANCELLABLE = "job_not_cancellable"
    VOICE_IN_USE = "voice_in_use"
    CAPACITY_EXCEEDED = "capacity_exceeded"
    ENGINE_UNAVAILABLE = "engine_unavailable"
    WORKER_UNAVAILABLE = "worker_unavailable"
    UPSTREAM_ERROR = "upstream_error"
    NOT_CONFIGURED = "not_configured"
    INTERNAL = "internal_error"


_STATUS_TITLES = {
    400: "Bad Request",
    401: "Unauthorized",
    404: "Not Found",
    409: "Conflict",
    413: "Payload Too Large",
    415: "Unsupported Media Type",
    422: "Unprocessable Content",
    429: "Too Many Requests",
    500: "Internal Server Error",
    501: "Not Implemented",
    502: "Bad Gateway",
    503: "Service Unavailable",
    507: "Insufficient Storage",
}


@dataclass(slots=True)
class ApiError(Exception):
    """Raised anywhere in the service layer; rendered once by the app."""

    status: int
    code: ErrorCode
    detail: str
    hint: str | None = None
    extra: dict[str, Any] | None = None

    def __str__(self) -> str:
        return f"{self.status} {self.code.value}: {self.detail}"

    def problem(self, request_id: str | None) -> dict[str, Any]:
        body: dict[str, Any] = {
            "type": PROBLEM_TYPE_BASE + self.code.value,
            "title": _STATUS_TITLES.get(self.status, "Error"),
            "status": self.status,
            "detail": self.detail,
            "code": self.code.value,
        }
        if self.hint:
            body["hint"] = self.hint
        if request_id:
            body["request_id"] = request_id
        if self.extra:
            body.update(self.extra)
        # OpenAI SDKs read error.message; keep both shapes in one body.
        body["error"] = {"message": self.detail, "type": self.code.value, "code": self.code.value}
        return body


def not_found(code: ErrorCode, what: str, hint: str | None = None) -> ApiError:
    return ApiError(404, code, f"{what} not found", hint)


__all__ = ["PROBLEM_MEDIA_TYPE", "ApiError", "ErrorCode", "not_found"]
