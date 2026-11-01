"""Gateway settings, read once from the environment at startup."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = _env(name)
    if raw is None:
        return default
    value = int(raw)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name)
    if raw is None:
        return default
    lowered = raw.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {raw!r}")


def _env_list(name: str) -> tuple[str, ...]:
    raw = _env(name, "")
    assert raw is not None
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _load_api_keys() -> frozenset[str]:
    keys = set(_env_list("VOICEAPI_API_KEYS"))
    key_file = _env("VOICEAPI_API_KEYS_FILE")
    if key_file:
        for raw_line in Path(key_file).read_text(encoding="utf-8").splitlines():
            entry = raw_line.strip()
            if entry and not entry.startswith("#"):
                keys.add(entry)
    weak = [key for key in keys if len(key) < 24]
    if weak:
        raise ValueError("every API key must be at least 24 characters")
    return frozenset(keys)


# Defaults live in module constants: on a slots dataclass `Cls.field` is a slot descriptor, not
# the default, so from_env must never read defaults off the class.
DEFAULT_ENGINE_URL = "http://tts-engine:8000"
DEFAULT_WORKER_URL = "http://audio-worker:8001"
DEFAULT_DATA_DIR = "/data"
DEFAULT_MAX_UPLOAD_BYTES = 200 * 1024 * 1024
DEFAULT_MAX_SPEECH_CHARS = 100_000
DEFAULT_JOB_WORKERS = 2


@dataclass(frozen=True, slots=True)
class Limits:
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES
    max_transcription_upload_bytes: int = 100 * 1024 * 1024
    max_speech_input_chars: int = DEFAULT_MAX_SPEECH_CHARS
    max_voice_clips: int = 20
    max_voice_total_seconds: float = 30 * 60
    max_sync_enhance_seconds: float = 30.0
    max_form_fields: int = 32
    max_wait_seconds: float = 60.0
    idempotency_ttl_seconds: int = 24 * 3600
    job_workers: int = DEFAULT_JOB_WORKERS


@dataclass(frozen=True, slots=True)
class Settings:
    engine_url: str = DEFAULT_ENGINE_URL
    worker_url: str = DEFAULT_WORKER_URL
    data_dir: Path = Path(DEFAULT_DATA_DIR)
    # SQLite needs real file locking; keep it off Windows bind mounts (default: inside data_dir).
    db_dir: Path | None = None
    public_base_url: str | None = None
    api_keys: frozenset[str] = field(default_factory=frozenset)
    allow_anonymous: bool = False
    default_voice: str | None = None
    chat_upstream: str | None = None
    chat_upstream_key: str | None = None
    mcp_allowed_hosts: tuple[str, ...] = ()
    engine_timeout_s: float = 600.0
    worker_timeout_s: float = 1800.0
    limits: Limits = field(default_factory=Limits)

    @classmethod
    def from_env(cls) -> Settings:
        settings = cls(
            engine_url=_env("VOICEAPI_ENGINE_URL", DEFAULT_ENGINE_URL) or DEFAULT_ENGINE_URL,
            worker_url=_env("VOICEAPI_WORKER_URL", DEFAULT_WORKER_URL) or DEFAULT_WORKER_URL,
            data_dir=Path(_env("VOICEAPI_DATA_DIR", DEFAULT_DATA_DIR) or DEFAULT_DATA_DIR),
            db_dir=Path(db_dir) if (db_dir := _env("VOICEAPI_DB_DIR")) else None,
            public_base_url=_env("VOICEAPI_PUBLIC_BASE_URL"),
            api_keys=_load_api_keys(),
            allow_anonymous=_env_bool("VOICEAPI_ALLOW_ANONYMOUS", False),
            default_voice=_env("VOICEAPI_DEFAULT_VOICE"),
            chat_upstream=_env("VOICEAPI_CHAT_UPSTREAM"),
            chat_upstream_key=_env("VOICEAPI_CHAT_UPSTREAM_KEY"),
            mcp_allowed_hosts=_env_list("VOICEAPI_MCP_ALLOWED_HOSTS"),
            limits=Limits(
                max_upload_bytes=_env_int("VOICEAPI_MAX_UPLOAD_BYTES", DEFAULT_MAX_UPLOAD_BYTES),
                max_speech_input_chars=_env_int("VOICEAPI_MAX_SPEECH_CHARS", DEFAULT_MAX_SPEECH_CHARS),
                job_workers=_env_int("VOICEAPI_JOB_WORKERS", DEFAULT_JOB_WORKERS),
            ),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if not self.api_keys and not self.allow_anonymous:
            raise ValueError(
                "no API keys configured: set VOICEAPI_API_KEYS or VOICEAPI_API_KEYS_FILE, "
                "or VOICEAPI_ALLOW_ANONYMOUS=1 for a trusted network"
            )
        for name in ("engine_url", "worker_url"):
            value = getattr(self, name)
            if not value.startswith(("http://", "https://")):
                raise ValueError(f"{name} must be an http(s) URL")

    @property
    def db_path(self) -> Path:
        return (self.db_dir or self.data_dir) / "voiceapi.sqlite3"

    @property
    def files_dir(self) -> Path:
        return self.data_dir / "files"


__all__ = ["Limits", "Settings"]
