"""Paragraph-aware chunking with word overlap.

Paragraphs are kept whole when they fit, so a chunk rarely cuts a rule in half.
A paragraph longer than the limit is split with a sliding window.
"""
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    idx: int
    text: str


def split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _windows(words: list[str], size: int, overlap: int) -> list[list[str]]:
    step = size - overlap
    out = []
    for start in range(0, len(words), step):
        out.append(words[start:start + size])
        if start + size >= len(words):
            break
    return out


def chunk_text(doc_id: str, text: str, max_words: int = 60, overlap: int = 10) -> list[Chunk]:
    if max_words <= 0 or overlap < 0 or overlap >= max_words:
        raise ValueError("need max_words > overlap >= 0")

    units: list[list[str]] = []
    for para in split_paragraphs(text):
        words = para.split()
        units.extend([words] if len(words) <= max_words else _windows(words, max_words, overlap))

    chunks: list[list[str]] = []
    current: list[str] = []
    for unit in units:
        if current and len(current) + len(unit) > max_words:
            chunks.append(current)
            tail = current[-overlap:] if overlap else []
            current = tail if len(tail) + len(unit) <= max_words else []
        current = current + unit
    if current:
        chunks.append(current)

    return [Chunk(doc_id, i, " ".join(words)) for i, words in enumerate(chunks)]
