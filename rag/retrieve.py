"""Hybrid retrieval with Reciprocal Rank Fusion (Cormack et al., 2009).

Cosine similarity and ts_rank_cd live on different scales, so instead of normalising and
weighting scores, RRF only uses ranks: score(d) = sum over retrievers of 1 / (k + rank).
"""
from dataclasses import replace
from typing import Sequence

from .chunking import chunk_text
from .embeddings import Embedder
from .store import Hit, Store

MODES = ("vector", "keyword", "hybrid")


def rrf(rankings: Sequence[Sequence[Hit]], k: int = 60, top_n: int = 5) -> list[Hit]:
    scores: dict[int, float] = {}
    first_seen: dict[int, Hit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank)
            first_seen.setdefault(hit.chunk_id, hit)
    ordered = sorted(scores, key=lambda cid: (-scores[cid], cid))
    return [replace(first_seen[cid], score=scores[cid]) for cid in ordered[:top_n]]


class Retriever:
    def __init__(self, store: Store, embedder: Embedder, candidates: int = 20):
        self.store = store
        self.embedder = embedder
        self.candidates = candidates

    def ingest(self, doc_id: str, text: str, max_words: int = 60, overlap: int = 10) -> int:
        chunks = chunk_text(doc_id, text, max_words, overlap)
        vectors = self.embedder.embed([c.text for c in chunks])
        self.store.upsert_document(doc_id, chunks, vectors)
        return len(chunks)

    def search(self, query: str, k: int = 5, mode: str = "hybrid") -> list[Hit]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if mode == "keyword":
            return self.store.keyword_search(query, k)
        qvec = self.embedder.embed([query])[0]
        if mode == "vector":
            return self.store.vector_search(qvec, k)
        return rrf(
            [self.store.vector_search(qvec, self.candidates), self.store.keyword_search(query, self.candidates)],
            top_n=k,
        )
