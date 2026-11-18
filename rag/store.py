"""Chunk storage and search.

PgStore is the real backend: one table holds the text, a pgvector embedding (HNSW index,
cosine distance) and a generated tsvector (GIN index) for full-text search, so both
retrievers read the same rows and never drift apart.

MemoryStore mirrors the interface with numpy, only for fast unit tests.
"""
from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from .chunking import Chunk
from .embeddings import tokenize


@dataclass(frozen=True)
class Hit:
    chunk_id: int
    doc_id: str
    idx: int
    content: str
    score: float


class Store(Protocol):
    def upsert_document(self, doc_id: str, chunks: Sequence[Chunk], vectors: Sequence[np.ndarray]) -> None: ...
    def vector_search(self, qvec: np.ndarray, k: int) -> list[Hit]: ...
    def keyword_search(self, query: str, k: int) -> list[Hit]: ...


SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id        bigserial PRIMARY KEY,
    doc_id    text NOT NULL,
    idx       int  NOT NULL,
    content   text NOT NULL,
    embedding vector({dim}) NOT NULL,
    tsv       tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    UNIQUE (doc_id, idx)
);
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING gin (tsv);
"""


def or_tsquery(query: str) -> str:
    """Natural questions rarely contain every term of the answer, so terms are OR-ed and
    ts_rank_cd does the ordering. Tokens are [a-z0-9]+ only, so the string is safe for to_tsquery."""
    return " | ".join(dict.fromkeys(tokenize(query)))


class PgStore:
    def __init__(self, dsn: str, dim: int):
        import psycopg
        from pgvector.psycopg import register_vector

        self.dim = dim
        self.conn = psycopg.connect(dsn, autocommit=True)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)

    def init_schema(self) -> None:
        self.conn.execute(SCHEMA.format(dim=int(self.dim)))

    def reset(self) -> None:
        self.conn.execute("DROP TABLE IF EXISTS chunks")
        self.init_schema()

    def ping(self) -> bool:
        return self.conn.execute("SELECT 1").fetchone()[0] == 1

    def upsert_document(self, doc_id, chunks, vectors) -> None:
        with self.conn.transaction():
            self.conn.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))
            with self.conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO chunks (doc_id, idx, content, embedding) VALUES (%s, %s, %s, %s)",
                    [(c.doc_id, c.idx, c.text, v) for c, v in zip(chunks, vectors)],
                )

    def vector_search(self, qvec, k) -> list[Hit]:
        rows = self.conn.execute(
            """SELECT id, doc_id, idx, content, 1 - (embedding <=> %(q)s) AS score
               FROM chunks ORDER BY embedding <=> %(q)s LIMIT %(k)s""",
            {"q": qvec, "k": k},
        ).fetchall()
        return [Hit(*r) for r in rows]

    def keyword_search(self, query, k) -> list[Hit]:
        tsq = or_tsquery(query)
        if not tsq:
            return []
        rows = self.conn.execute(
            """SELECT id, doc_id, idx, content, ts_rank_cd(tsv, q) AS score
               FROM chunks, to_tsquery('english', %(tsq)s) AS q
               WHERE tsv @@ q ORDER BY score DESC, id LIMIT %(k)s""",
            {"tsq": tsq, "k": k},
        ).fetchall()
        return [Hit(*r) for r in rows]


class MemoryStore:
    def __init__(self):
        self._rows: list[tuple[int, Chunk, np.ndarray]] = []
        self._next = 1

    def upsert_document(self, doc_id, chunks, vectors) -> None:
        self._rows = [r for r in self._rows if r[1].doc_id != doc_id]
        for c, v in zip(chunks, vectors):
            self._rows.append((self._next, c, np.asarray(v, dtype=np.float32)))
            self._next += 1

    def vector_search(self, qvec, k) -> list[Hit]:
        scored = [(float(np.dot(v, qvec)), cid, c) for cid, c, v in self._rows]
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [Hit(cid, c.doc_id, c.idx, c.text, s) for s, cid, c in scored[:k]]

    def keyword_search(self, query, k) -> list[Hit]:
        terms = set(tokenize(query))
        scored = []
        for cid, c, _ in self._rows:
            overlap = len(terms & set(tokenize(c.text)))
            if overlap:
                scored.append((float(overlap), cid, c))
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [Hit(cid, c.doc_id, c.idx, c.text, s) for s, cid, c in scored[:k]]
