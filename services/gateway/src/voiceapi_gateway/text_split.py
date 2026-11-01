"""Split long speech input into engine-sized chunks on natural boundaries.

The engine's largest prefill bucket bounds one request (clone prompts add the
reference transcript and ~12.5 rows per reference second), so input is cut at
sentence ends, then clause marks, then whitespace. Budgets are in characters
with a per-script weight: CJK characters are roughly one token each, Latin
text roughly four characters per token.
"""

from __future__ import annotations

import re

LATIN_CHUNK_CHARS = 600
CJK_CHUNK_CHARS = 220

_SENTENCE_END = re.compile(r"(?<=[.!?。！？…])[\"'”’)\]]*\s+|(?<=[。！？])")
_CLAUSE = re.compile(r"(?<=[,;:，；：、])\s*")
# CJK scripts plus their punctuation blocks (CJK symbols, full-width forms): each is ~one token.
_CJK = re.compile(r"[\u3000-\u303f\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff\uff00-\uffef]")


def _budget(text: str) -> int:
    if not text:
        return LATIN_CHUNK_CHARS
    cjk_share = len(_CJK.findall(text)) / len(text)
    return int(LATIN_CHUNK_CHARS - (LATIN_CHUNK_CHARS - CJK_CHUNK_CHARS) * cjk_share)


def _hard_split(piece: str, budget: int) -> list[str]:
    parts: list[str] = []
    for clause in (c for c in _CLAUSE.split(piece) if c.strip()):
        rest = clause
        while len(rest) > budget:
            cut = rest.rfind(" ", 0, budget)
            if cut <= budget // 3:
                cut = budget
            parts.append(rest[:cut].strip())
            rest = rest[cut:].strip()
        if rest.strip():
            parts.append(rest.strip())
    return parts


def _join(left: str, right: str) -> str:
    """Join two pieces; CJK text has no inter-sentence spaces."""
    if not left:
        return right
    if _CJK.match(left[-1]) or _CJK.match(right[0]):
        return left + right
    return f"{left} {right}"


def split_for_speech(text: str) -> list[str]:
    normalized = re.sub(r"[ \t]+", " ", text.replace("\r\n", "\n")).strip()
    if not normalized:
        return []
    budget = _budget(normalized)
    if len(normalized) <= budget:
        return [normalized]

    sentences: list[str] = []
    for paragraph in (p.strip() for p in re.split(r"\n\s*\n", normalized)):
        if not paragraph:
            continue
        for sentence in (s.strip() for s in _SENTENCE_END.split(paragraph)):
            if not sentence:
                continue
            sentences.extend([sentence] if len(sentence) <= budget else _hard_split(sentence, budget))

    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = _join(current, sentence)
        if len(candidate) <= budget:
            current = candidate
            continue
        if current:
            chunks.append(current)
        current = sentence
    if current:
        chunks.append(current)
    return chunks


__all__ = ["CJK_CHUNK_CHARS", "LATIN_CHUNK_CHARS", "split_for_speech"]
