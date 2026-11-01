"""Prefixed, time-sortable identifiers (lowercase Crockford ULID)."""

from __future__ import annotations

import os
import re
import time

_ALPHABET = "0123456789abcdefghjkmnpqrstvwxyz"
_ULID_LENGTH = 26
PREFIXES = frozenset({"voice", "file", "job", "req"})
_ID_PATTERN = re.compile(r"^(?P<prefix>[a-z]+)_(?P<ulid>[0-9a-hjkmnp-tv-z]{26})$")


def _encode(value: int, length: int) -> str:
    chars = []
    for _ in range(length):
        chars.append(_ALPHABET[value & 31])
        value >>= 5
    return "".join(reversed(chars))


def new_ulid() -> str:
    timestamp_ms = int(time.time() * 1000) & ((1 << 48) - 1)
    randomness = int.from_bytes(os.urandom(10), "big")
    return _encode((timestamp_ms << 80) | randomness, _ULID_LENGTH)


def new_id(prefix: str) -> str:
    if prefix not in PREFIXES:
        raise ValueError(f"unknown id prefix {prefix!r}")
    return f"{prefix}_{new_ulid()}"


def is_id(value: str, prefix: str) -> bool:
    match = _ID_PATTERN.fullmatch(value)
    return bool(match) and match.group("prefix") == prefix


def engine_voice_handle(voice_id: str) -> str:
    """The engine's voice name: ``voice_<ulid>`` → ``v_<ulid>`` (fits its 64-char regex)."""
    if not is_id(voice_id, "voice"):
        raise ValueError(f"not a voice id: {voice_id!r}")
    return "v_" + voice_id.removeprefix("voice_")


__all__ = ["engine_voice_handle", "is_id", "new_id", "new_ulid"]
