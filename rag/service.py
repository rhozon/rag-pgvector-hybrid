"""Wiring from Settings to a ready RAG object (used by the API and the CLI)."""
from .config import Settings
from .embeddings import get_embedder
from .generate import RAG, get_llm
from .retrieve import Retriever
from .store import PgStore


def build(settings: Settings | None = None, reset: bool = False) -> tuple[RAG, PgStore]:
    s = settings or Settings()
    store = PgStore(s.database_url, s.embed_dim)
    if reset:
        store.reset()
    else:
        store.init_schema()
    retriever = Retriever(store, get_embedder(s.embedder, s.embed_dim))
    return RAG(retriever, get_llm(s.llm, s.anthropic_model)), store
