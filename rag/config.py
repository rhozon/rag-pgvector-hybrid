import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass(frozen=True)
class Settings:
    database_url: str = field(default_factory=lambda: _env("DATABASE_URL", "postgresql://rag:rag@localhost:5432/rag"))
    # "hash" needs no download and is deterministic (tests, CI); "fastembed" uses BAAI/bge-small-en-v1.5.
    embedder: str = field(default_factory=lambda: _env("EMBEDDER", "hash"))
    embed_dim: int = field(default_factory=lambda: int(_env("EMBED_DIM", "384")))
    # "extractive" needs no API key; "anthropic" calls Claude.
    llm: str = field(default_factory=lambda: _env("LLM", "extractive"))
    anthropic_model: str = field(default_factory=lambda: _env("ANTHROPIC_MODEL", "claude-sonnet-5-5"))
    chunk_words: int = field(default_factory=lambda: int(_env("CHUNK_WORDS", "60")))
    chunk_overlap: int = field(default_factory=lambda: int(_env("CHUNK_OVERLAP", "10")))
